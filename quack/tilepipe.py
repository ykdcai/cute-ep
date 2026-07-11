# Copyright (c) 2026, QuACK team.
"""TilePipe: MoE dispatch <-> grouped-GEMM overlap via per-expert counting
semaphores in symmetric memory (see tilepipe.md). This module owns the
dispatch kernel, the host-side metadata builders, and — critically — the
`TilePipe` class that encapsulates the launch/reset/warm-up discipline.

That discipline encodes hard-won deadlock invariants; tests and examples must
go through this class rather than re-implementing it:

1. Every kernel is warm-up-EXECUTED (not just compiled) at construction time.
   The first launch of a compiled kernel does lazy module load — host-side
   CUDA work that deadlocks against an already-spinning gated GEMM.
2. `reset()` zeroes the expert flags AND the local `seg_done` counters, then
   syncs and barriers. A missed seg_done reset over-publishes and releases
   the GEMM early (silent corruption).
3. `run_overlapped()` / `run_serial()` perform kernel launches ONLY. No
   allocation, no scalar H2D, no sync may happen between the gated-GEMM
   launch and the dispatch launch: any host CUDA call that device-syncs
   deadlocks against the spinning GEMM. All launch arguments are prebuilt.
4. The GEMM is capped via max_active_clusters so `num_comm_sms` SMs remain
   free for dispatch — the SM partition is a launch-time knob. With
   oversubscription (`oversub_sms > 0`) the GEMM grid may cover every SM, so
   the partition is no longer disjoint by construction: correctness then
   relies on dispatch being enqueued FIRST on the high-priority comm stream,
   so its CTAs are resident before the GEMM fills the machine; the GEMM's
   surplus CTAs stay pending and backfill SMs as dispatch retires.

nvshmem is imported lazily so `import quack.tilepipe` works on single-GPU
machines (e.g. for the metadata builders in unit tests).
"""

import os

import numpy as np
import torch
import torch.distributed as dist

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from quack.gemm import gemm as quack_gemm
from quack.tilepipe_sync import ExpertArrivalSemaphore


# ---------------------------------------------------------------------------
# Dispatch kernel: expert-major / rank-minor sends + counting-semaphore raise
# ---------------------------------------------------------------------------
# The send list is host-precomputed and already sorted in the order the sends
# must complete (expert-major, rank-minor, rotated by source rank). Warps
# stride the list front-to-back, so completion order approximately follows
# list order. Per send: warp-cooperative 256-bit copy of the hidden vector
# into the destination rank's recv buffer. Two flag-publish modes:
#
#   PUBLISH_SEGMENT=False (per token): after each copy, lane 0 publishes 1.
#   PUBLISH_SEGMENT=True (per (source, expert) segment): a warp's strided
#     subsequence of the segment-sorted send list is itself segment-sorted,
#     so it copies all its tokens of the current segment back-to-back, then
#     does ONE local atomic_add(seg_done[seg], count, acq_rel, sys). The warp
#     whose increment completes the segment publishes the single remote
#     release; the acquire->release chain through seg_done makes it
#     cumulative over every other warp's remote data stores.


@cute.jit
def _flush_segment(
    cur_seg: Int32,
    count: Int32,
    seg_done: cute.Tensor,        # [num_segs] int32, LOCAL completion counters
    seg_sizes: cute.Tensor,       # [num_segs] int32
    flag_peer_ptrs: cute.Tensor,  # [world_size] int64
    lane_id: Int32,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
):
    cute.arch.sync_warp()
    if lane_id == 0:
        old = cute.arch.atomic_add(
            seg_done.iterator + cur_seg, count, sem="acq_rel", scope="sys")
        seg_size = seg_sizes[cur_seg]
        if old + count == seg_size:
            # Last increment for this segment: publish it to the destination.
            # seg id encodes (local_expert, rotated dst): seg = e * W + rot.
            e = cur_seg // world_size
            dst = (cur_seg % world_size + local_rank) % world_size
            sem = ExpertArrivalSemaphore(peer_ptrs=flag_peer_ptrs)
            sem.arrive(dst, e, seg_size)


@cute.kernel
def tilepipe_dispatch_kernel(
    input_buf: cute.Tensor,       # [num_tokens, hidden] local token data
    recv_peer_ptrs: cute.Tensor,  # [world_size] int64: peer base addr of recv_buf
    flag_peer_ptrs: cute.Tensor,  # [world_size] int64: peer base addr of flags
    send_token: cute.Tensor,      # [total_sends] int32: source token index
    send_slot: cute.Tensor,       # [total_sends] int32: dst recv-buffer row
    send_dst: cute.Tensor,        # [total_sends] int32: destination rank
    send_expert: cute.Tensor,     # [total_sends] int32: dst-local expert index
    send_seg: cute.Tensor,        # [total_sends] int32: segment id (e * W + rot)
    seg_done: cute.Tensor,        # [num_segs] int32: LOCAL completion counters
    seg_sizes: cute.Tensor,       # [num_segs] int32
    total_sends: Int32,
    _hidden: Int32,
    max_recv_tokens: Int32,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    PUBLISH_SEGMENT: cutlass.Constexpr,
):
    hidden = cute.assume(_hidden, divby=32)
    WARP_SIZE = 32
    tidx, _, _ = cute.arch.thread_idx()
    bdimx, _, _ = cute.arch.block_dim()
    gdimx, _, _ = cute.arch.grid_dim()
    bidx, _, _ = cute.arch.block_idx()
    warps_per_cta = bdimx // WARP_SIZE
    warp_id = tidx // WARP_SIZE
    global_warp_id = bidx * warps_per_cta + warp_id
    total_warps = warps_per_cta * gdimx
    lane_id = tidx % WARP_SIZE

    dtype = input_buf.element_type
    COPY_BITS = 256
    elems_per_copy = COPY_BITS // dtype.width
    copy_align = COPY_BITS // 8
    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=COPY_BITS)
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=COPY_BITS,
        memory_scope=cute.nvgpu.common.MemoryScope.SYS,
        memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
    )
    thr_layout = cute.make_ordered_layout((1, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, elems_per_copy), order=(1, 0))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)
    thr_copy_load = tiled_copy_load.get_slice(lane_id)
    thr_copy_store = tiled_copy_store.get_slice(lane_id)

    num_tokens = cute.size(input_buf, mode=[0])
    src_layout = cute.make_ordered_layout((num_tokens, hidden), order=(1, 0))
    src_tensor = cute.make_tensor(input_buf.iterator.align(copy_align), src_layout)
    tSgS = thr_copy_load.partition_S(src_tensor)
    frg = cute.make_fragment_like(tSgS[None, 0, 0])
    hidden_iter = cute.size(tSgS, mode=[2])

    remote_layout = cute.make_ordered_layout((max_recv_tokens, hidden), order=(1, 0))

    cur_seg = Int32(-1)
    count = Int32(0)
    for i in range(global_warp_id, total_sends, total_warps):
        token = send_token[i]
        slot = send_slot[i]
        dst = send_dst[i]
        e = send_expert[i]

        if cutlass.const_expr(PUBLISH_SEGMENT):
            seg = send_seg[i]
            if seg != cur_seg:
                if count > 0:
                    _flush_segment(cur_seg, count, seg_done, seg_sizes,
                                   flag_peer_ptrs, lane_id, local_rank, world_size)
                cur_seg = seg
                count = Int32(0)

        remote_ptr = cute.make_ptr(
            dtype, recv_peer_ptrs[dst], cute.AddressSpace.gmem, assumed_align=copy_align)
        remote_tensor = cute.make_tensor(remote_ptr, remote_layout)
        tDgD = thr_copy_store.partition_D(remote_tensor)
        for k in range(hidden_iter):
            cute.copy(thr_copy_load, tSgS[None, token, k], frg)
            cute.copy(thr_copy_store, frg, tDgD[None, slot, k])

        if cutlass.const_expr(PUBLISH_SEGMENT):
            count += 1
        else:
            # All lanes' stores must be ordered before the flag bump: sync_warp
            # makes them visible to lane 0, whose release/sys add is cumulative.
            cute.arch.sync_warp()
            if lane_id == 0:
                sem = ExpertArrivalSemaphore(peer_ptrs=flag_peer_ptrs)
                sem.arrive(dst, e, Int32(1))

    if cutlass.const_expr(PUBLISH_SEGMENT):
        if count > 0:
            _flush_segment(cur_seg, count, seg_done, seg_sizes,
                           flag_peer_ptrs, lane_id, local_rank, world_size)


@cute.jit
def tilepipe_dispatch(
    input_buf: cute.Tensor,
    recv_peer_ptrs: cute.Tensor,
    flag_peer_ptrs: cute.Tensor,
    send_token: cute.Tensor,
    send_slot: cute.Tensor,
    send_dst: cute.Tensor,
    send_expert: cute.Tensor,
    send_seg: cute.Tensor,
    seg_done: cute.Tensor,
    seg_sizes: cute.Tensor,
    total_sends: Int32,
    hidden: Int32,
    max_recv_tokens: Int32,
    num_ctas: Int32,
    num_warps: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    PUBLISH_SEGMENT: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    tilepipe_dispatch_kernel(
        input_buf,
        recv_peer_ptrs,
        flag_peer_ptrs,
        send_token,
        send_slot,
        send_dst,
        send_expert,
        send_seg,
        seg_done,
        seg_sizes,
        total_sends,
        hidden,
        max_recv_tokens,
        local_rank,
        world_size,
        PUBLISH_SEGMENT,
    ).launch(
        grid=[num_ctas, 1, 1],
        block=[num_warps * 32, 1, 1],
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Host-side metadata (assumed precomputed in a real pipeline; see tilepipe.md)
# ---------------------------------------------------------------------------


def build_send_arrays(all_topk, num_experts, src_rank, world_size):
    """Send list for `src_rank`, sorted expert-major / rank-minor (rotated by
    source rank), with destination recv-buffer slots.

    Receiver layout (per destination rank): rows grouped by local expert,
    within an expert by source rank, within a source by token order — i.e.
    expert segments are contiguous, so the receiver's cu_seqlens_m is just
    the cumsum of its per-expert split sizes.

    all_topk: [world_size, num_tokens, topk] global topk_indices (numpy).
    Returns (send_token, send_slot, send_dst, send_expert, send_seg, seg_sizes)
    int32 numpy arrays; segment id = local_expert * world_size + rotated_dst,
    i.e. exactly the send-order key, so seg ids are ascending in the list.
    """
    epr = num_experts // world_size
    counts = np.zeros((world_size, num_experts), dtype=np.int64)
    for r in range(world_size):
        counts[r] = np.bincount(all_topk[r].reshape(-1), minlength=num_experts)

    # base[dst, e_local, src]: first recv-buffer row on `dst` for tokens of
    # local expert e_local coming from `src`.
    base = np.zeros((world_size, epr, world_size), dtype=np.int64)
    for dst in range(world_size):
        c = 0
        for e in range(epr):
            for src in range(world_size):
                base[dst, e, src] = c
                c += counts[src, dst * epr + e]

    topk_flat = all_topk[src_rank].reshape(-1)  # token-major, then topk slot
    num_tokens, topk = all_topk[src_rank].shape
    tok = np.repeat(np.arange(num_tokens, dtype=np.int64), topk)
    dst = topk_flat // epr
    el = topk_flat % epr

    # Slot within each (dst, e_local) group: source tokens in token order.
    group = dst * epr + el
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    within = np.arange(len(order)) - np.searchsorted(sorted_group, sorted_group, side="left")
    slot = np.empty_like(tok)
    slot[order] = base[dst[order], el[order], src_rank] + within

    # Send order: expert-major, rank-minor, destination rotated by source
    # rank so the W senders don't all hammer rank 0 first.
    rot = (dst - src_rank) % world_size
    key = el * world_size + rot
    order2 = np.argsort(key, kind="stable")
    seg_sizes = np.bincount(key, minlength=epr * world_size).astype(np.int32)
    to_i32 = lambda a: np.ascontiguousarray(a[order2], dtype=np.int32)
    return to_i32(tok), to_i32(slot), to_i32(dst), to_i32(el), to_i32(key), seg_sizes


def build_recv_metadata(all_topk, num_experts, rank, world_size):
    """Receiver-side metadata for `rank`: per-expert split sizes, cu_seqlens,
    and the symmetric max recv-row count. Note recv sizing uses the ACTUAL
    routed counts (max over ranks for the symmetric-alloc requirement), never
    the all-tokens-to-one-expert worst case."""
    epr = num_experts // world_size
    counts = np.zeros((world_size, num_experts), dtype=np.int64)
    for r in range(world_size):
        counts[r] = np.bincount(all_topk[r].reshape(-1), minlength=num_experts)
    split_sizes = counts[:, rank * epr : (rank + 1) * epr].sum(axis=0)  # [epr]
    cu_seqlens = np.concatenate([[0], np.cumsum(split_sizes)])
    totals = [counts[:, d * epr : (d + 1) * epr].sum() for d in range(world_size)]
    max_recv_tokens = max(int(max(totals)), 1)
    return split_sizes.astype(np.int64), cu_seqlens.astype(np.int32), max_recv_tokens


def peer_ptr_tensor(symm_tensor, world_size, device):
    """int64[world_size] tensor of each rank's P2P base address for
    `symm_tensor`, indexed by a runtime rank inside kernels via cute.make_ptr."""
    import nvshmem.core

    return torch.tensor(
        [nvshmem.core.get_peer_tensor(symm_tensor, r).data_ptr()
         for r in range(world_size)],
        device=device, dtype=torch.int64)


def parse_symmetric_size(val):
    """Parse NVSHMEM_SYMMETRIC_SIZE (may carry a K/M/G/T suffix) to bytes."""
    mult = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}.get(val[-1].upper(), 1)
    return int(float(val[:-1]) * mult) if mult != 1 else int(val)


# ---------------------------------------------------------------------------
# TilePipe: the overlapped dispatch + gated grouped GEMM pipeline
# ---------------------------------------------------------------------------


class TilePipe:
    """Owns the streams, symmetric buffers, compiled kernels, and launch/reset
    discipline for TilePipe dispatch+GEMM overlap (invariants in the module
    docstring). Construction allgathers routing, builds metadata, allocates
    symmetric buffers, compiles AND warm-up-executes every kernel, and
    validates the counting-semaphore protocol standalone.

    Collective: all ranks must construct it together (allgather + barriers).
    """

    def __init__(
        self,
        *,
        input_data,          # [tokens, hidden] bf16 cuda tensor (this rank's tokens)
        weights,             # [experts_per_rank, N, K] bf16 (this rank's experts)
        topk_indices,        # [tokens, topk] int32 cuda tensor (global expert ids)
        num_experts,
        tile_m=128,
        tile_n=128,
        comm_warps=16,
        publish="segment",   # "segment" or "token" flag granularity
        warmup_comm_sms=8,
        log=None,            # callable for stage prints (hang diagnostics)
    ):
        import nvshmem.core

        self._nvshmem = nvshmem.core
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.rank, self.world_size = rank, world_size
        device = input_data.device
        self.device = device
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.tile_m, self.tile_n = tile_m, tile_n
        self.comm_warps = comm_warps
        self.publish = publish
        log = log or (lambda msg: None)
        self._freed = False

        tokens, hidden = input_data.shape
        self.hidden = hidden
        assert hidden % 512 == 0, "hidden must be divisible by 512 (256-bit warp copies)"
        epr = num_experts // world_size
        assert weights.shape[0] == epr
        self.experts_per_rank = epr
        self.weights = weights

        # --- Routing allgather + host metadata ---
        all_topk_t = [torch.zeros_like(topk_indices) for _ in range(world_size)]
        dist.all_gather(all_topk_t, topk_indices.contiguous())
        self.all_topk = np.stack([t.cpu().numpy() for t in all_topk_t])
        log("routing allgathered")

        tok, slot, dst_, el, seg, seg_sizes_np = build_send_arrays(
            self.all_topk, num_experts, rank, world_size)
        split_sizes, cu_seqlens, max_recv_tokens = build_recv_metadata(
            self.all_topk, num_experts, rank, world_size)
        self.total_m = int(cu_seqlens[-1])
        self.total_sends = len(tok)
        self.max_recv_tokens = max_recv_tokens

        dev = lambda a: torch.from_numpy(a).to(device)
        self._send_token = dev(tok)
        self._send_slot = dev(slot)
        self._send_dst = dev(dst_)
        self._send_expert = dev(el)
        self._send_seg = dev(seg)
        self._seg_sizes = dev(seg_sizes_np)
        self.seg_done = torch.zeros(len(seg_sizes_np), dtype=torch.int32, device=device)
        self.cu_seqlens_m = dev(cu_seqlens)
        self.split_sizes = dev(split_sizes.astype(np.int32))
        log(f"metadata built: total_m={self.total_m} total_sends={self.total_sends} "
            f"max_recv={max_recv_tokens}")

        # --- Symmetric buffers ---
        symm_bytes = (tokens + max_recv_tokens) * hidden * 2 + epr * 4
        heap_env = os.environ.get("NVSHMEM_SYMMETRIC_SIZE")
        if heap_env is not None and symm_bytes > parse_symmetric_size(heap_env):
            raise RuntimeError(
                f"symmetric buffers ({symm_bytes / 1e9:.3f} GB) exceed the NVSHMEM "
                f"heap ({heap_env}); set NVSHMEM_SYMMETRIC_SIZE higher "
                f"(TilePipe.required_symmetric_bytes underestimated the skew?)")
        log(f"allocating symmetric buffers: {symm_bytes / 1e9:.3f} GB "
            f"(NVSHMEM_SYMMETRIC_SIZE={heap_env or 'default'})")
        self.input_buf = nvshmem.core.tensor((tokens, hidden), dtype=torch.bfloat16)
        self.input_buf.copy_(input_data)
        self.recv_buf = nvshmem.core.tensor(
            (max_recv_tokens, hidden), dtype=torch.bfloat16)
        self.recv_buf.fill_(0)
        self.flags = nvshmem.core.tensor((epr,), dtype=torch.int32)
        self.flags.fill_(0)
        self._recv_peer_ptrs = peer_ptr_tensor(self.recv_buf, world_size, device)
        self._flag_peer_ptrs = peer_ptr_tensor(self.flags, world_size, device)
        log("symmetric buffers allocated")

        self.A = self.recv_buf[: self.total_m]
        self.out = torch.empty(
            (self.total_m, weights.shape[1]), dtype=torch.bfloat16, device=device)

        # High-priority comm stream: with oversubscription the GEMM grid can
        # cover every SM, so dispatch's CTAs must win placement (they are
        # also enqueued first — see run_overlapped).
        self.comm_stream = torch.cuda.Stream(priority=-1)
        self.gemm_stream = torch.cuda.Stream()

        # --- Compile ---
        # Token-dependent dims are marked dynamic so the compiled kernel is
        # reusable across token counts (from_dlpack shapes are static by
        # default and would otherwise force a re-specialization).
        log(f"compiling dispatch kernel (publish={publish})...")
        self._compiled_dispatch = cute.compile(
            tilepipe_dispatch, *self._dispatch_args(), Int32(warmup_comm_sms),
            comm_warps, rank, world_size, publish == "segment",
            cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        log("dispatch kernel compiled")

        # --- Warm-up: EXECUTE every kernel before any overlapped run ---
        # GEMM (gated + ungated) with flags pre-satisfied so nothing spins.
        log("compiling GEMM (gated + ungated)...")
        self.flags.copy_(self.split_sizes)
        self.launch_gemm(gated=True, stream=self.gemm_stream)
        log("gated GEMM compiled + warm-up launched")
        self.launch_gemm(gated=False, stream=self.gemm_stream)
        log("ungated GEMM compiled + warm-up launched")
        torch.cuda.synchronize()
        log("GEMM warm-up synced")
        dist.barrier(device_ids=[rank])

        # Dispatch warm-up execution + standalone protocol validation: flags
        # must land exactly on split_sizes with no gating around.
        self.reset()
        self.launch_dispatch(warmup_comm_sms)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])
        ok = torch.equal(self.flags.cpu(), self.split_sizes.cpu())
        if not ok:
            log(f"dispatch-only FAIL: flags={self.flags.cpu().tolist()} "
                f"expected={self.split_sizes.cpu().tolist()}")
        else:
            log("dispatch-only: flags OK")
        ok_t = torch.tensor([ok], dtype=torch.int32, device=device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        if not bool(ok_t.item()):
            raise RuntimeError(
                f"[rank {rank}] dispatch-only flag validation FAILED")
        self.reset()

    @staticmethod
    def required_symmetric_bytes(tokens, hidden, topk, slack=1.5):
        """NVSHMEM heap size to export as NVSHMEM_SYMMETRIC_SIZE *before*
        nvshmem init: input + expected recv (tokens*topk under uniform
        routing, x slack for skew) + nvshmem internals."""
        exp_recv = tokens * topk
        return (tokens + int(slack * exp_recv)) * hidden * 2 + 128 * 1024 * 1024

    def _dispatch_args(self):
        dyn2d = lambda t: from_dlpack(t, assumed_align=32).mark_layout_dynamic(leading_dim=1)
        dyn1d = lambda t: from_dlpack(t).mark_layout_dynamic(leading_dim=0)
        return (
            dyn2d(self.input_buf), from_dlpack(self._recv_peer_ptrs),
            from_dlpack(self._flag_peer_ptrs),
            dyn1d(self._send_token), dyn1d(self._send_slot), dyn1d(self._send_dst),
            dyn1d(self._send_expert), dyn1d(self._send_seg), from_dlpack(self.seg_done),
            from_dlpack(self._seg_sizes), Int32(self.total_sends), Int32(self.hidden),
            Int32(self.max_recv_tokens))

    def launch_dispatch(self, num_ctas, stream=None):
        stream = stream if stream is not None else self.comm_stream
        self._compiled_dispatch(
            *self._dispatch_args(), Int32(num_ctas),
            cuda.CUstream(stream.cuda_stream))

    def launch_gemm(self, gated, max_clusters=None, stream=None):
        with torch.cuda.stream(stream if stream is not None else self.gemm_stream):
            quack_gemm(
                self.A, self.weights, self.out, C=None, tile_count_semaphore=None,
                tile_M=self.tile_m, tile_N=self.tile_n, cluster_M=1, cluster_N=1,
                persistent=True, cu_seqlens_m=self.cu_seqlens_m,
                expert_ready_flags=self.flags if gated else None,
                max_active_clusters=max_clusters)

    def reset(self):
        """Zero the expert flags AND the local segment counters, then sync +
        barrier. Must run between iterations; a stale seg_done over-publishes."""
        self.flags.fill_(0)
        self.seg_done.zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[self.rank])

    def run_overlapped(self, num_comm_sms, oversub_sms=0):
        """Dispatch on the high-priority comm_stream, gated GEMM on
        gemm_stream. Launches only — call reset() first.

        `oversub_sms` lets the GEMM backfill dispatch's SMs after dispatch
        finishes: the GEMM gets min(num_sms, num_sms + oversub_sms -
        num_comm_sms) clusters, so up to `oversub_sms` of its CTAs start out
        pending behind dispatch and are placed as dispatch CTAs retire —
        removing the tail where the comm partition idles. Dispatch MUST be
        enqueued first: an oversubscribed GEMM placed first would occupy
        every SM spinning on flags and deadlock."""
        max_clusters = min(self.num_sms, self.num_sms + oversub_sms - num_comm_sms)
        self.launch_dispatch(num_comm_sms, self.comm_stream)
        self.launch_gemm(gated=True, max_clusters=max_clusters,
                         stream=self.gemm_stream)

    def run_serial(self):
        """Same kernels, one stream: no SM sharing, no overlap, each phase
        gets the whole GPU. The gate still covers the cross-rank dependency
        (the peer dispatches concurrently)."""
        self.launch_dispatch(self.num_sms, self.comm_stream)
        self.launch_gemm(gated=True, stream=self.comm_stream)

    def free(self):
        if self._freed:
            return
        self._freed = True
        for t in (self.input_buf, self.recv_buf, self.flags):
            self._nvshmem.free_tensor(t)
