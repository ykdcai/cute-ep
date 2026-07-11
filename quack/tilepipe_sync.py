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
