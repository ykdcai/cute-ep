# Copyright (c) 2026, QuACK team.
"""TilePipe device-side synchronization: the per-expert counting semaphore as
an object with member wait/arrive methods, so consumer kernels (GEMM) never
hardcode the flag format. The wait is a *predicate* — "expert e is ready" —
and swapping in a class with a different predicate (e.g. token counter AND
weight-ready flag for EPLB) specializes the consumer at compile time with no
consumer code changes.

Scope contract (works for warp-specialized and non-specialized kernels):

- `poll(e, target)`: single-thread acquire-poll until the predicate holds.
  No election, no sync — the escape hatch for any specialization scheme with
  its own leader election and broadcast (the observation is only as visible
  as the synchronization used to hand it off; mbarrier pipelines carry it).
- `wait_warp(e, target)`: warp-collective convenience — elect one lane,
  poll, sync_warp (propagates the acquire to all lanes at warp scope), then
  fence.proxy.async so subsequently issued TMA loads (async proxy) are
  ordered after the generic-proxy acquire. Call from a converged warp.
- `arrive(dst, e, count)`: producer publish — one release/sys atomic add on
  the destination rank's flag, after the caller's own completion sync
  (sync_warp / local counting; see the dispatch kernels' segment protocol).
"""

from typing import Optional
from dataclasses import dataclass

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from quack.cute_dsl_utils import nanosleep


@dataclass
class ExpertArrivalSemaphore:
    """flag[e] counts arrivals; expert e is ready when flag[e] >= target.

    Consumers construct it with `flags` (local view of the symmetric flag
    array); producers with `peer_ptrs` (per-rank int64 symmetric base
    addresses of that array).
    """

    flags: Optional[cute.Tensor] = None      # [num_experts] Int32, local
    peer_ptrs: Optional[cute.Tensor] = None  # [world_size] Int64

    @cute.jit
    def poll(self, e: Int32, target: Int32):
        """Single-thread: acquire-poll flag[e] until it reaches target."""
        arrived = cute.arch.atomic_add(
            self.flags.iterator + e, Int32(0), sem="acquire", scope="sys"
        )
        while arrived < target:
            nanosleep(256)
            arrived = cute.arch.atomic_add(
                self.flags.iterator + e, Int32(0), sem="acquire", scope="sys"
            )

    @cute.jit
    def wait_warp(self, e: Int32, target: Int32):
        """Warp-collective wait: one elected lane polls (redundant sys-scope
        atomics from all 32 lanes would serialize at the L2 atomic unit and
        contend with the producer's increments); sync_warp holds the other
        lanes and propagates the observation; the proxy fence orders it
        before subsequently issued TMA (async proxy) loads."""
        with cute.arch.elect_one():
            self.poll(e, target)
        cute.arch.sync_warp()
        cute.arch.fence_proxy("async")

    @cute.jit
    def arrive(self, dst: Int32, e: Int32, count: Int32):
        """Producer publish from one lane: release/sys add of `count` onto
        the destination rank's flag[e]. The caller must have ordered all
        contributing data stores before this (sync_warp for its own lanes;
        an acq_rel chain through a local completion counter to cover other
        warps' stores — see _flush_segment in the dispatch kernels)."""
        flag_ptr = cute.make_ptr(
            Int32, self.peer_ptrs[dst], cute.AddressSpace.gmem, assumed_align=4
        )
        cute.arch.atomic_add(flag_ptr + e, count, sem="release", scope="sys")

    @cute.jit
    def arrive_all(self, e: Int32, count: Int32):
        """Producer publish to EVERY rank's flag[e] (broadcast counters, e.g.
        the GEMM epilogue's tile-completion flags). Same ordering contract as
        arrive()."""
        num_ranks = cute.size(self.peer_ptrs.shape)
        for r in cutlass.range(num_ranks):
            self.arrive(Int32(r), e, count)


# ---------------------------------------------------------------------------
# Test util: data-then-flag trickle producer.
# ---------------------------------------------------------------------------
# Emulates a TilePipe producer for gating tests WITHOUT host CUDA calls while
# consumers spin (the host-side torch path — fill_/scalar H2D — device-syncs
# and deadlocks against a spinning gated kernel). Tile i's row range is
# caller-supplied ([tile_lo[i], tile_hi[i])), so the flag index space is
# exactly the consumer's — never re-derived here. Protocol per tile: copy the
# rows (data first), block barrier, then one thread publishes
# atomic_add(flag_base + i, pub_val, release, sys) on EVERY rank.
# Launch it on its own stream; warm-up-EXECUTE it once standalone before any
# overlapped use (first launch does lazy module load — host work).


@cute.kernel
def _flag_trickle_kernel(
    src: cute.Tensor,             # [rows, hidden] data the tiles should contain
    dst: cute.Tensor,             # [rows, hidden] buffer consumers read (starts stale)
    tile_lo: cute.Tensor,         # [num_tiles] int32 first row of tile i
    tile_hi: cute.Tensor,         # [num_tiles] int32 one-past-last row of tile i
    flag_peer_ptrs: cute.Tensor,  # [world] int64 peer flag-array base addrs
    flag_base: Int32,             # this producer's offset in the flag arrays
    pub_val: Int32,               # value published per tile (e.g. n_tiles target)
    num_tiles: Int32,
    delay_iters: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    hidden = cute.size(src, mode=[1])
    for t in range(num_tiles):
        lo = tile_lo[t]
        hi = tile_hi[t]
        for row in range(lo, hi):
            for h in range(tidx, hidden, bdim):
                dst[row, h] = src[row, h]
        cute.arch.barrier()
        if tidx == 0:
            for _ in cutlass.range(delay_iters):
                nanosleep(1024)
            sem = ExpertArrivalSemaphore(peer_ptrs=flag_peer_ptrs)
            for r in cutlass.range(world_size):
                sem.arrive(Int32(r), flag_base + t, pub_val)
        cute.arch.barrier()


@cute.jit
def flag_trickle(
    src: cute.Tensor,
    dst: cute.Tensor,
    tile_lo: cute.Tensor,
    tile_hi: cute.Tensor,
    flag_peer_ptrs: cute.Tensor,
    flag_base: Int32,
    pub_val: Int32,
    num_tiles: Int32,
    delay_iters: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _flag_trickle_kernel(
        src, dst, tile_lo, tile_hi, flag_peer_ptrs,
        flag_base, pub_val, num_tiles, delay_iters, world_size,
    ).launch(grid=[1, 1, 1], block=[256, 1, 1], stream=stream)


@cute.kernel
def _wait_flag_kernel(flags: cute.Tensor, idx: Int32, target: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        sem = ExpertArrivalSemaphore(flags=flags)
        sem.poll(idx, target)


@cute.jit
def wait_flag(flags: cute.Tensor, idx: Int32, target: Int32, stream: cuda.CUstream):
    """One-block gate: spins until flags[idx] >= target (acquire/sys), then
    exits. Enqueue it on a stream to make every LATER kernel on that stream
    (e.g. a plain torch reduction) wait on a TilePipe counter without any
    host synchronization."""
    _wait_flag_kernel(flags, idx, target).launch(
        grid=[1, 1, 1], block=[32, 1, 1], stream=stream)
