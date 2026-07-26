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
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from cutlass.cutlass_dsl import dsl_user_op

from quack.cute_dsl_utils import nanosleep


@dsl_user_op
def ld_acquire_sys(ptr, *, loc=None, ip=None) -> Int32:
    """`ld.acquire.sys.global.b32` — an acquire LOAD of a 32-bit flag.

    Polling with `atomic_add(ptr, 0, acquire, sys)` gives the same ordering but
    is a read-modify-write: it takes the line exclusively at L2 every
    iteration, which both costs the waiter latency on its critical path and
    invalidates the *publisher's* copy of the line it is waiting on. A load
    leaves the line shared.

    `has_side_effects=True` is load-bearing: without it the compiler treats the
    load as pure, hoists it out of the spin loop, and the wait never
    terminates. The "gate blocks until producer publishes" functional tests are
    the regression gate for exactly that.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_release_sys(ptr, val: Int32, *, loc=None, ip=None) -> None:
    """`red.release.sys.global.add.u32` — fire-and-forget release add.

    `cute.arch.atomic_add` returns the old value, so it lowers to `atom`, which
    allocates a destination register and (on the producer's critical path) is a
    round-trip the warp can be made to wait on even when the result is dead.
    `red` has no destination: the store unit takes it and the warp moves on.
    This is the GEMM epilogue's publish, issued once per work tile per rank, so
    it sits directly on the producer's critical path.
    """
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.release.sys.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def publish_tile_flag(peer_ptrs: cute.Tensor, e: Int32, count: Int32) -> None:
    """Broadcast `count` onto flag[e] on every rank — the GEMM epilogue's
    per-work-tile publish, on the producer's critical path.

    Two things make this cheaper than `ExpertArrivalSemaphore.arrive_all`:

    1. **Static width.** `peer_ptrs` must have a compile-time extent (see
       `tile_flag_world` in gemm.py), so the per-rank loop is a Python loop
       that unrolls fully. With a dynamic extent the DSL emits an
       unroll-by-16/8/4 branch ladder around every publish.
    2. **`red`, not `atom`.** No destination register, nothing for the warp
       to wait on.

    The `peer_ptrs[r]` loads are loop-invariant with r constant, so LICM
    hoists them out of the caller's tile loop; keeping this a plain function
    (rather than an object holding the bases) avoids making them loop-carried
    values the DSL would have to flatten across the `while`.
    """
    world = cute.size(peer_ptrs.shape)
    assert isinstance(world, int), (
        "publish_tile_flag needs a static peer-pointer extent; pass the world "
        "size through the compile key (see gemm.py tile_flag_world)"
    )
    for r in range(world):
        ptr = cute.make_ptr(Int32, peer_ptrs[r], cute.AddressSpace.gmem, assumed_align=4)
        red_release_sys(ptr + e, count)


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
        """Single-thread: acquire-poll flag[e] until it reaches target.

        Uses an acquire LOAD, not an atomic RMW — see ld_acquire_sys."""
        ptr = self.flags.iterator + e
        arrived = ld_acquire_sys(ptr)
        while arrived < target:
            nanosleep(256)
            arrived = ld_acquire_sys(ptr)

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


# ---------------------------------------------------------------------------
# LayoutSemaphore: readiness for GEMM-produced tiles.
# ---------------------------------------------------------------------------
# Scope: consumers of a GEMM's output tiles (GEMM->combine, GEMM->allreduce).
# Dispatch keeps ExpertArrivalSemaphore — its producer unit is a token, not a
# tile, and its target is ragged (split_sizes).
#
# The producer publishes at its NATURAL coordinate — the epilogue's
# (batch, m_tile) — and never changes. A consumer declares the unit it waits
# on; the semaphore derives both the flag index (via an indexer) and the
# target (the number of producer tiles covering one consumer unit, i.e. the
# fiber size). `n_tiles`/`W` stop being kernel arguments: they are properties
# of the two tilings, so a consumer cannot disagree with the producer about
# either the index space or the count. That divergence is what produced the
# trickle hang, the flag_idx mismatch and the tail-publish bug.
#
#   case        producer unit         consumer unit            fiber (target)
#   combine     (m_tile, n_tile)      (m_tile, all N)          ceil(N/tile_N)
#   allreduce   (m,n) tile on 1 rank  same tile across W ranks W
#
# Fan-out vs fan-in: with one flag per consumer unit the wait is a single
# poll and the publish fans out to every consumer unit a producer tile
# touches; with one flag per producer unit the publish is single and the wait
# fans in. Both cases above have fan-out 1 on the local flag array (combine:
# a tile maps to exactly one m-tile counter; allreduce: one counter per
# output tile, incremented once per rank), so the counter form is optimal for
# both — assert that when adding a third case.


@dataclass
class AffineIndexer:
    """flag = m_tile * stride + base. For regular (non-varlen) tilings."""

    stride: int = 1
    base: int = 0

    @cute.jit
    def index(self, batch: Int32, m_tile: Int32) -> Int32:
        return m_tile * Int32(self.stride) + Int32(self.base)


@dataclass
class RaggedTileIndexer:
    """flag = tile_offsets[batch] + m_tile — per-batch m-tile bases, i.e. the
    varlen grouped GEMM's tiling (each expert tiled separately)."""

    tile_offsets: cute.Tensor  # [num_batches] int32, exclusive cumsum

    @cute.jit
    def index(self, batch: Int32, m_tile: Int32) -> Int32:
        return self.tile_offsets[batch] + m_tile


@dataclass
class LayoutSemaphore:
    """Counting semaphore over GEMM output tiles.

    flags:     local view of the counter array (consumers)
    peer_ptrs: [world] int64 symmetric bases (producers); a 1-entry table
               means "publish locally only" (push-combine).
    indexer:   (batch, m_tile) -> flag index, shared by BOTH sides.
    target:    fiber size — how many producer tiles cover one consumer unit.
               Compile-time for both supported cases.
    """

    indexer: object
    target: cutlass.Constexpr
    flags: Optional[cute.Tensor] = None
    peer_ptrs: Optional[cute.Tensor] = None

    @cute.jit
    def arrive(self, batch: Int32, m_tile: Int32):
        """Producer: +1 on this tile's counter, on every rank in peer_ptrs.
        Caller must have ordered its data writes first (async-proxy stores
        need cp.async.bulk drain + fence.proxy.async — see the GEMM
        epilogue)."""
        idx = self.indexer.index(batch, m_tile)
        num_ranks = cute.size(self.peer_ptrs.shape)
        for r in cutlass.range(num_ranks):
            flag_ptr = cute.make_ptr(
                Int32, self.peer_ptrs[r], cute.AddressSpace.gmem, assumed_align=4
            )
            cute.arch.atomic_add(flag_ptr + idx, Int32(1), sem="release", scope="sys")

    @cute.jit
    def poll(self, batch: Int32, m_tile: Int32):
        """Single-thread: wait until this consumer unit's fiber is complete."""
        ptr = self.flags.iterator + self.indexer.index(batch, m_tile)
        arrived = ld_acquire_sys(ptr)
        while arrived < Int32(self.target):
            nanosleep(256)
            arrived = ld_acquire_sys(ptr)

    @cute.jit
    def wait_warp(self, batch: Int32, m_tile: Int32):
        """Warp-collective wait (see ExpertArrivalSemaphore.wait_warp)."""
        with cute.arch.elect_one():
            self.poll(batch, m_tile)
        cute.arch.sync_warp()
        cute.arch.fence_proxy("async")
