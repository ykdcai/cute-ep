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
import cutlass.utils as cute_utils
from cutlass import Boolean, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def nanosleep(ns: int | Int32, *, loc=None, ip=None) -> None:
    """Suspend the calling warp for ~ns nanoseconds (sm_70+ scheduler hint).

    Used in the spin-wait loops below to avoid hammering the memory system
    with back-to-back polls.
    """
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
    )


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
# TileSemaphore: the GEMM output-tile readiness protocol, shared by BOTH sides.
# ---------------------------------------------------------------------------
# Scope: consumers of a GEMM's output tiles (GEMM->combine, GEMM->allreduce).
# Dispatch keeps ExpertArrivalSemaphore — its producer unit is a token, not a
# tile, and its target is ragged (split_sizes).
#
# The GEMM epilogue publishes at its NATURAL coordinate, (batch, m_tile,
# n_tile), and never changes. A subclass declares the CONSUMER's atom, which
# fixes three things at once:
#
#   * `index()`  — where in the flag array that atom's counter lives,
#   * `target()` — how many producer publishes complete it (the fiber size),
#   * `arrive()` — which instruction the publish uses.
#
# `n_tiles`/`W` stop being kernel arguments: they are properties of the two
# tilings, so a consumer cannot disagree with the producer about the index
# space or the count. That divergence is what produced the trickle hang, the
# flag_idx mismatch and the tail-publish bug. One object is handed to the GEMM
# (via TilePipeArgs) and to the comm kernel, so they cannot drift.
#
#   subclass              consumer atom              fiber (target)
#   RowBlockSemaphore     (batch, m_tile), all N     ceil(N / tile_N)
#   OutputTileSemaphore   one (m,n) tile, all ranks  world_size
#
# Both have fan-out 1 on a given flag array (combine: a work tile maps to
# exactly one row-block counter; allreduce: one counter per output tile, +1
# per rank), so the counter form is optimal for both — assert that when
# adding a third case.
#
# LAYERING. TileSemaphore is the PROTOCOL and is architecture-neutral: index
# arithmetic plus `ld.acquire.sys` / `red.release.sys`, all sm70+. It is the
# object the producer and the consumer share, and it must stay free of any
# epilogue or SM-generation detail so a Hopper producer, a Blackwell producer
# and a plain SIMT comm kernel can all speak it. The one hardware assumption
# is optional and behind a flag: OutputTileSemaphore(multicast=True) needs an
# NVLink Switch fabric for `multimem.red`, and multicast=False is the portable
# fallback.
#
# The SM-specific half lives in TileFlagPipeline below, which knows how a
# particular epilogue proves its stores are visible. Keep new arch knowledge
# there, not here.


@dataclass(frozen=True)
class TileSemaphore:
    """Base: the parts that do not vary with the consumer's atom.

    Frozen and field-free apart from `stride` so it is hashable — the object
    goes straight into the GEMM compile-cache key (see TilePipeArgs).

    stride: spacing in int32 elements between consecutive flags. 1 packs 32
    per 128B line. Measured to be noise on B200 (atomics execute at the L2
    slice, so there is no line ping-pong), kept as a knob for other targets.
    """

    stride: int = 1

    # -- subclass contract ---------------------------------------------------

    @cute.jit
    def index(self, offsets, batch: Int32, m_tile: Int32, n_tile: Int32, n_tiles) -> Int32:
        """Flag index for the consumer atom containing producer tile
        (batch, m_tile, n_tile). `offsets` is the per-batch exclusive cumsum
        of flag counts (None => single batch based at 0)."""
        raise NotImplementedError

    def target(self, n_tiles: int, world: int) -> int:
        """Producer publishes needed to complete one consumer atom."""
        raise NotImplementedError

    @cute.jit
    def arrive(self, peer_ptrs: cute.Tensor, idx: Int32) -> None:
        """Producer publish: +1 on flag[idx]. The caller must have ordered its
        data writes first — async-proxy (TMA) stores need a cp.async.bulk
        drain plus fence.proxy.async; see the GEMM epilogue."""
        raise NotImplementedError

    # -- consumer side (identical for every atom) ---------------------------

    @cute.jit
    def poll(self, flags: cute.Tensor, idx: Int32, target: Int32):
        """Single-thread: acquire-poll flag[idx] until it reaches target."""
        ptr = flags.iterator + idx
        arrived = ld_acquire_sys(ptr)
        while arrived < target:
            nanosleep(256)
            arrived = ld_acquire_sys(ptr)

    @cute.jit
    def wait_warp(self, flags: cute.Tensor, idx: Int32, target: Int32):
        """Warp-collective wait (see ExpertArrivalSemaphore.wait_warp)."""
        with cute.arch.elect_one():
            self.poll(flags, idx, target)
        cute.arch.sync_warp()
        cute.arch.fence_proxy("async")


@dataclass(frozen=True)
class RowBlockSemaphore(TileSemaphore):
    """GEMM -> combine. One counter per (batch, m_tile) row block; every
    n-tile of that row bumps it, so the consumer waits for ceil(N / tile_N).

    The combine consumer needs a whole row (all N) of a token before it can
    move it, so a finer counter would buy nothing — see status.md
    "Not worth doing / Per-(m,n)-tile pushing in combine".
    """

    @cute.jit
    def index(self, offsets, batch: Int32, m_tile: Int32, n_tile: Int32, n_tiles) -> Int32:
        return (offsets[batch] + m_tile) * Int32(self.stride)

    def target(self, n_tiles: int, world: int) -> int:
        return n_tiles

    @cute.jit
    def arrive(self, peer_ptrs: cute.Tensor, idx: Int32) -> None:
        publish_tile_flag(peer_ptrs, idx, Int32(1))


@dataclass(frozen=True)
class OutputTileSemaphore(TileSemaphore):
    """GEMM -> allreduce. One counter per (batch, m_tile, n_tile) output tile,
    bumped once by each rank, so the consumer waits for `world`.

    Source and destination layouts match here, so a per-(m,n) atom is both
    finer than the row block AND contiguous — the reason allreduce gets the
    1:1 producer-atom/consumer-atom mapping that combine cannot.

    multicast=True publishes with ONE `multimem.red` on a multicast flag
    address (peer_ptrs is then a 1-entry table holding that address), so the
    cost is O(1) in world size instead of the O(world) loop of `red`s. This is
    the case where multimem earns its place: push-combine's publish is
    rank-local (fan-out 1) and has nothing to collapse, but every rank must
    see an allreduce producer's tile.
    """

    multicast: bool = True

    @cute.jit
    def index(self, offsets, batch: Int32, m_tile: Int32, n_tile: Int32, n_tiles) -> Int32:
        return (offsets[batch] + m_tile * n_tiles + n_tile) * Int32(self.stride)

    def target(self, n_tiles: int, world: int) -> int:
        return world

    @cute.jit
    def arrive(self, peer_ptrs: cute.Tensor, idx: Int32) -> None:
        if const_expr(self.multicast):
            mc = cute.make_ptr(
                Int32, peer_ptrs[0], cute.AddressSpace.gmem, assumed_align=4
            )
            cute_utils.distributed.multimem_red_add1(mc + idx, scope="sys", order="release")
        else:
            publish_tile_flag(peer_ptrs, idx, Int32(1))


# ---------------------------------------------------------------------------
# TileFlagPipeline: the GEMM epilogue's half of the protocol.
# ---------------------------------------------------------------------------
# This is the arch-aware layer — it assumes a TMA-store epilogue whose D
# stores retire in cp.async.bulk commit-groups (SM90 and SM100 both). Porting
# to an epilogue with a different store-completion rule means changing THIS
# class only; TileSemaphore above is untouched, and consumers cannot tell.


@dataclass(frozen=True)
class TileFlagPipeline:
    """A one-deep publish pipeline over the epilogue's work tiles: `commit`
    once per tile, `tail` once after the loop.

    The names deliberately mirror the CUTLASS store pipeline on the adjacent
    lines of the epilogue (`epi_store_pipeline.producer_commit/producer_tail`),
    because this IS that pattern — a producer whose completions are proven
    asynchronously and therefore observed a fixed distance behind.

    WHY ONE-DEEP (the whole reason this class exists). D stores leave through
    the async proxy as TMA writes retired in bulk-commit-groups. Proving THIS
    tile's stores landed needs `cp.async.bulk.wait_group 0`, i.e. a full drain
    of the store pipeline every tile — that is what the fused CUTLASS
    allreduce example does, and it costs ~10% of the GEMM by destroying
    store/mainloop overlap. Waiting on `in_flight` groups instead is free (a
    no-op while epi_stage <= in_flight) but proves only that everything OLDER
    than the current tile has landed. So `commit` publishes the previous tile
    and holds the current one; `tail` releases the last one after the
    epilogue's own `producer_tail`.

    Callers therefore never see the bulk-group wait, the proxy fence, or the
    off-by-one: they say "tile (b, m, n) is issued" and "no more tiles".

    The (idx, valid) pair is threaded through the caller rather than stored on
    this object: it is loop-carried across the epilogue's DSL `while`, and
    loop-carried values must be explicit. Keeping the index expression in ONE
    place is the point — `commit` and `tail` disagreeing about it is the
    tail-publish bug (see the case table above).

    Usage, from the TMA warp only:

        pipe = TileFlagPipeline(sem)
        idx, valid = Int32(0), Boolean(False)
        while work_tile.is_valid_tile:
            ...
            idx, valid = pipe.commit(idx, valid, ptrs, offsets, b, m, n,
                                     n_tiles, this_tile_valid, epi_tile_num)
        epi_store_pipeline.producer_tail()
        pipe.tail(idx, valid, ptrs)
    """

    sem: TileSemaphore

    @cute.jit
    def commit(
        self,
        pending_idx: Int32,
        pending_valid: Boolean,
        peer_ptrs: cute.Tensor,
        offsets,
        batch: Int32,
        m_tile: Int32,
        n_tile: Int32,
        n_tiles,
        valid: Boolean,
        in_flight: cutlass.Constexpr,
    ):
        """Tile (batch, m_tile, n_tile) has issued its D stores. Publish
        whatever is now provably visible and hold this tile.

        in_flight: store groups this tile itself issued (epi_tile_num) — the
        number left outstanding, and hence the depth of the deferral.
        valid: False for a cluster-overhang CTA whose stores were predicated
        away; its flag must not be published at all.

        Returns the new (pending_idx, pending_valid).
        """
        cute.arch.cp_async_bulk_wait_group(in_flight)
        cute.arch.fence_proxy("async")
        self._release(peer_ptrs, pending_idx, pending_valid)
        return self.sem.index(offsets, batch, m_tile, n_tile, n_tiles), valid

    @cute.jit
    def tail(self, pending_idx: Int32, pending_valid: Boolean, peer_ptrs: cute.Tensor):
        """No more tiles. Release the held one; the caller must already have
        drained the store pipeline (`epi_store_pipeline.producer_tail()`)."""
        cute.arch.fence_proxy("async")
        self._release(peer_ptrs, pending_idx, pending_valid)

    @cute.jit
    def _release(self, peer_ptrs: cute.Tensor, idx: Int32, valid: Boolean):
        if valid:
            if cute.arch.lane_idx() == 0:
                self.sem.arrive(peer_ptrs, idx)
