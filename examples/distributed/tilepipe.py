"""
TilePipe: MoE dispatch <-> grouped-GEMM overlap via per-expert counting
semaphores in symmetric memory. See tilepipe.md.

Two ordinary kernels on two streams over a disjoint SM partition:

  dispatch (this file)      : walks its sends expert-major / rank-minor
                              (rotated by source rank), copies each token's
                              hidden vector into the destination rank's recv
                              buffer over NVLink, then bumps the destination's
                              per-expert flag with atomic_add(release, sys).
  grouped GEMM (quack)      : quack_gemm(..., cu_seqlens_m=..., expert_ready_flags=flags,
                              max_active_clusters=NUM_SMS - num_comm_sms); its AB-load
                              warps wait for flag[e] == split_sizes[e] before
                              loading expert e's tiles (acquire, sys).

The GEMM consumes experts in increasing local-expert order (varlen batch
order), so dispatch drives experts to completion in that same order; the
rank-minor inner loop keeps every destination rank fed (see tilepipe.md §2).

Run (2 GPUs):
    torchrun --nproc-per-node 2 examples/distributed/tilepipe.py
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

import nvshmem.core

from moe_comm import torchrun_uid_init_bcast, torchrun_finalize, _peer_ptr_tensor

from quack.gemm import gemm as quack_gemm


# ---------------------------------------------------------------------------
# Dispatch kernel: expert-major / rank-minor sends + counting-semaphore raise
# ---------------------------------------------------------------------------
# The send list is host-precomputed and already sorted in the order the sends
# must complete (expert-major, rank-minor, rotated by source rank). Warps
# stride the list front-to-back, so completion order approximately follows
# list order. Per send: warp-cooperative 256-bit copy of the hidden vector
# into the destination rank's recv buffer. Two flag-publish modes:
#
#   PUBLISH_SEGMENT=False (per token): after each copy, lane 0 does
#     atomic_add(dst_flag[e], 1, release, sys). Trivially correct (each warp
#     releases only its own writes) but fires tokens*topk remote atomics at
#     experts_per_rank flag addresses.
#   PUBLISH_SEGMENT=True (per (source, expert) segment): a warp's strided
#     subsequence of the segment-sorted send list is itself segment-sorted,
#     so it copies all its tokens of the current segment back-to-back, then
#     does ONE local atomic_add(seg_done[seg], count, acq_rel, sys). The warp
#     whose increment completes the segment publishes the single remote
#     atomic_add(dst_flag[e], seg_size, release, sys); the acquire->release
#     chain through seg_done makes that release cumulative over every other
#     warp's remote data stores. The GEMM needs the whole expert before it
#     can start, so coarser increments cost no latency.
#
# In both modes sync_warp precedes lane 0's release so all 32 lanes' data
# stores are ordered before the publish.


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
            flag_ptr = cute.make_ptr(
                Int32, flag_peer_ptrs[dst], cute.AddressSpace.gmem, assumed_align=4)
            cute.arch.atomic_add(flag_ptr + e, seg_size, sem="release", scope="sys")


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
                flag_ptr = cute.make_ptr(
                    Int32, flag_peer_ptrs[dst], cute.AddressSpace.gmem, assumed_align=4)
                cute.arch.atomic_add(flag_ptr + e, Int32(1), sem="release", scope="sys")

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
    epr_local = num_experts // world_size
    seg_sizes = np.bincount(key, minlength=epr_local * world_size).astype(np.int32)
    to_i32 = lambda a: np.ascontiguousarray(a[order2], dtype=np.int32)
    return to_i32(tok), to_i32(slot), to_i32(dst), to_i32(el), to_i32(key), seg_sizes


def build_recv_metadata(all_topk, num_experts, rank, world_size):
    """Receiver-side metadata for `rank`: per-expert split sizes, cu_seqlens,
    and the symmetric max recv-row count."""
    epr = num_experts // world_size
    counts = np.zeros((world_size, num_experts), dtype=np.int64)
    for r in range(world_size):
        counts[r] = np.bincount(all_topk[r].reshape(-1), minlength=num_experts)
    split_sizes = counts[:, rank * epr : (rank + 1) * epr].sum(axis=0)  # [epr]
    cu_seqlens = np.concatenate([[0], np.cumsum(split_sizes)])
    totals = [counts[:, d * epr : (d + 1) * epr].sum() for d in range(world_size)]
    max_recv_tokens = max(int(max(totals)), 1)
    return split_sizes.astype(np.int64), cu_seqlens.astype(np.int32), max_recv_tokens


# ---------------------------------------------------------------------------
# Reference check
# ---------------------------------------------------------------------------

def build_recv_reference(all_inputs, all_topk, num_experts, rank, world_size, total_m, device):
    """Each rank rebuilds its expected recv buffer from the (deterministic)
    slot assignment of every source rank."""
    hidden = all_inputs[0].shape[1]
    recv_ref = torch.zeros((total_m, hidden), dtype=all_inputs[0].dtype, device=device)
    for src in range(world_size):
        tok, slot, dst, _, _, _ = build_send_arrays(all_topk, num_experts, src, world_size)
        mine = dst == rank
        recv_ref[torch.from_numpy(slot[mine]).long().to(device)] = all_inputs[src][
            torch.from_numpy(tok[mine]).long().to(device)
        ]
    return recv_ref


def grouped_gemm_reference(recv_ref, weights, cu_seqlens):
    out = torch.empty(
        (recv_ref.shape[0], weights.shape[1]), dtype=torch.float32, device=recv_ref.device
    )
    for e in range(weights.shape[0]):
        lo, hi = int(cu_seqlens[e]), int(cu_seqlens[e + 1])
        out[lo:hi] = recv_ref[lo:hi].float() @ weights[e].float().T
    return out


# ---------------------------------------------------------------------------
# Benchmark / correctness driver
# ---------------------------------------------------------------------------

def run_tilepipe(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    epr = args.experts // world_size
    torch.manual_seed(42 + rank)

    if rank == 0:
        print(f"TilePipe: tokens/rank={args.tokens} hidden={args.hidden} N={args.n} "
              f"experts={args.experts} (local {epr}) topk={args.topk} "
              f"world={world_size} SMs={num_sms}")

    # --- Routing + token data ---
    topk_indices = torch.randint(
        0, args.experts, (args.tokens, args.topk), dtype=torch.int32, device=device)
    input_data = (
        torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device=device)
        / (args.hidden ** 0.5)
    )
    weights_bytes = epr * args.n * args.hidden * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] allocating weights: {weights_bytes / 1e9:.2f} GB "
          f"(free: {free_bytes / 1e9:.2f} GB)")
    weights = torch.randn(
        (epr, args.n, args.hidden), dtype=torch.bfloat16, device=device) / (args.hidden ** 0.25)
    print(f"[rank {rank}] weights allocated")

    all_topk_t = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    dist.all_gather(all_topk_t, topk_indices.contiguous())
    all_topk = np.stack([t.cpu().numpy() for t in all_topk_t])
    print(f"[rank {rank}] routing allgathered")

    # --- Metadata (host-precomputed; see tilepipe.md) ---
    tok, slot, dst, el, seg, seg_sizes_np = build_send_arrays(
        all_topk, args.experts, rank, world_size)
    split_sizes, cu_seqlens, max_recv_tokens = build_recv_metadata(
        all_topk, args.experts, rank, world_size)
    total_m = int(cu_seqlens[-1])
    total_sends = len(tok)

    send_token = torch.from_numpy(tok).to(device)
    send_slot = torch.from_numpy(slot).to(device)
    send_dst = torch.from_numpy(dst).to(device)
    send_expert = torch.from_numpy(el).to(device)
    send_seg = torch.from_numpy(seg).to(device)
    seg_sizes_t = torch.from_numpy(seg_sizes_np).to(device)
    seg_done = torch.zeros(len(seg_sizes_np), dtype=torch.int32, device=device)
    print(f"[rank {rank}] metadata built: total_m={total_m} total_sends={total_sends} "
          f"max_recv={max_recv_tokens}")
    cu_seqlens_m = torch.from_numpy(cu_seqlens).to(device)
    split_sizes_t = torch.from_numpy(split_sizes.astype(np.int32)).to(device)

    # --- Symmetric buffers ---
    symm_bytes = (args.tokens + max_recv_tokens) * args.hidden * 2 + epr * 4
    print(f"[rank {rank}] allocating symmetric buffers: {symm_bytes / 1e9:.2f} GB "
          f"(NVSHMEM_SYMMETRIC_SIZE={os.environ.get('NVSHMEM_SYMMETRIC_SIZE', 'default')})")
    input_buf = nvshmem.core.tensor((args.tokens, args.hidden), dtype=torch.bfloat16)
    input_buf.copy_(input_data)
    recv_buf = nvshmem.core.tensor((max_recv_tokens, args.hidden), dtype=torch.bfloat16)
    recv_buf.fill_(0)
    flags = nvshmem.core.tensor((epr,), dtype=torch.int32)
    flags.fill_(0)
    recv_peer_ptrs = _peer_ptr_tensor(recv_buf, world_size, device)
    flag_peer_ptrs = _peer_ptr_tensor(flags, world_size, device)
    print(f"[rank {rank}] symmetric buffers allocated")

    out = torch.empty((total_m, args.n), dtype=torch.bfloat16, device=device)
    A = recv_buf[:total_m]

    # --- Compile ---
    # Token-dependent dims are marked dynamic so the compiled kernel is
    # reusable across token counts (shapes from from_dlpack are static by
    # default and would otherwise force a re-specialization).
    dyn2d = lambda t: from_dlpack(t, assumed_align=32).mark_layout_dynamic(leading_dim=1)
    dyn1d = lambda t: from_dlpack(t).mark_layout_dynamic(leading_dim=0)
    dispatch_args = lambda: (
        dyn2d(input_buf), from_dlpack(recv_peer_ptrs), from_dlpack(flag_peer_ptrs),
        dyn1d(send_token), dyn1d(send_slot), dyn1d(send_dst),
        dyn1d(send_expert), dyn1d(send_seg), from_dlpack(seg_done),
        from_dlpack(seg_sizes_t), Int32(total_sends), Int32(args.hidden),
        Int32(max_recv_tokens))
    if rank == 0:
        print(f"Compiling dispatch kernel (publish={args.publish})...")
    compiled_dispatch = cute.compile(
        tilepipe_dispatch, *dispatch_args(), Int32(args.comm_sms_list[0]),
        args.comm_warps, rank, world_size, args.publish == "segment",
        cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    print(f"[rank {rank}] dispatch kernel compiled")

    def launch_dispatch(num_ctas, stream):
        compiled_dispatch(
            *dispatch_args(), Int32(num_ctas),
            cuda.CUstream(stream.cuda_stream))

    def launch_gemm(gated, max_clusters=None):
        quack_gemm(
            A, weights, out, C=None, tile_count_semaphore=None,
            tile_M=args.tile_m, tile_N=args.tile_n, cluster_M=1, cluster_N=1,
            persistent=True, cu_seqlens_m=cu_seqlens_m,
            expert_ready_flags=flags if gated else None,
            max_active_clusters=max_clusters)

    # Warm up / compile the GEMM (both gated and ungated variants) with flags
    # pre-satisfied so nothing spins.
    if rank == 0:
        print("Compiling GEMM (gated + ungated)...")
    flags.copy_(split_sizes_t)
    launch_gemm(gated=True)
    print(f"[rank {rank}] gated GEMM compiled + warm-up launched")
    launch_gemm(gated=False)
    print(f"[rank {rank}] ungated GEMM compiled + warm-up launched")
    torch.cuda.synchronize()
    print(f"[rank {rank}] GEMM warm-up synced")
    dist.barrier(device_ids=[rank])

    comm_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()

    def reset():
        flags.fill_(0)
        seg_done.zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    # Dispatch warm-up + standalone validation. The FIRST launch of a compiled
    # kernel does lazy module load — host-side CUDA work that deadlocks if a
    # gated GEMM is already spinning on the GPU — so it must happen here, not
    # inside the overlapped run. This also verifies the counting-semaphore
    # protocol (flags must land exactly on split_sizes) with no gating around.
    def check_dispatch_only():
        reset()
        launch_dispatch(args.comm_sms_list[0], comm_stream)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])
        ok = torch.equal(flags.cpu(), split_sizes_t.cpu())
        if not ok:
            print(f"[rank {rank}] dispatch-only FAIL: flags={flags.cpu().tolist()} "
                  f"expected={split_sizes_t.cpu().tolist()}")
        else:
            print(f"[rank {rank}] dispatch-only: flags OK")
        ok_t = torch.tensor([ok], dtype=torch.int32, device=device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        return bool(ok_t.item())

    if not check_dispatch_only():
        raise SystemExit(f"[rank {rank}] dispatch-only flag check FAILED")

    def overlapped_iter(num_comm_sms):
        # GEMM first (it just spins on the flags), dispatch on the disjoint SMs.
        with torch.cuda.stream(gemm_stream):
            launch_gemm(gated=True, max_clusters=num_sms - num_comm_sms)
        launch_dispatch(num_comm_sms, comm_stream)

    def serial_iter():
        # Same kernels, one stream: no SM sharing, no overlap, each phase gets
        # the whole GPU. The gate still covers the cross-rank dependency (the
        # peer dispatches concurrently).
        launch_dispatch(num_sms, comm_stream)
        with torch.cuda.stream(comm_stream):
            launch_gemm(gated=True)

    # --- Correctness (overlapped) ---
    all_inputs = [torch.zeros_like(input_data) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_data.contiguous())
    recv_ref = build_recv_reference(
        all_inputs, all_topk, args.experts, rank, world_size, total_m, device)
    out_ref = grouped_gemm_reference(recv_ref, weights, cu_seqlens)

    reset()
    out.zero_()
    overlapped_iter(args.comm_sms_list[0])
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])

    ok_flags = torch.equal(flags.cpu(), split_sizes_t.cpu())
    ok_recv = torch.equal(recv_buf[:total_m], recv_ref)
    rel_err = ((out.float() - out_ref).abs().max() /
               out_ref.abs().max().clamp(min=1e-6)).item()
    ok_gemm = rel_err < 2e-2
    ok = ok_flags and ok_recv and ok_gemm
    print(f"[rank {rank}] correctness: flags={'OK' if ok_flags else 'FAIL'} "
          f"recv={'OK' if ok_recv else 'FAIL'} gemm rel_err={rel_err:.2e} "
          f"{'OK' if ok_gemm else 'FAIL'}")
    ok_t = torch.tensor([ok], dtype=torch.int32, device=device)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if not bool(ok_t.item()):
        raise SystemExit(f"[rank {rank}] correctness FAILED")

    if not args.benchmark:
        return

    # --- Benchmark ---
    def time_iters(enqueue, iters, warmup):
        times = []
        for it in range(warmup + iters):
            reset()
            start = torch.cuda.Event(enable_timing=True)
            end_gemm = torch.cuda.Event(enable_timing=True)
            end_comm = torch.cuda.Event(enable_timing=True)
            start.record(comm_stream)
            gemm_stream.wait_event(start)
            enqueue()
            end_comm.record(comm_stream)
            end_gemm.record(gemm_stream)
            torch.cuda.synchronize()
            if it >= warmup:
                times.append(max(start.elapsed_time(end_gemm), start.elapsed_time(end_comm)))
        t = torch.tensor([float(np.median(times))], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank defines the step
        return t.item()

    if rank == 0:
        print("benchmark: pure GEMM / pure dispatch / serial / overlapped sweep...")
    # Ideal lower bounds: each phase alone with all resources.
    flags.copy_(split_sizes_t)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])
    def pure_gemm():
        with torch.cuda.stream(gemm_stream):
            launch_gemm(gated=False)

    t_gemm_pure = time_iters(pure_gemm, args.iters, args.warmup)

    t_disp_pure = {}
    for csms in args.comm_sms_list:
        t_disp_pure[csms] = time_iters(
            lambda: launch_dispatch(csms, comm_stream), args.iters, args.warmup)

    t_serial = time_iters(serial_iter, args.iters, args.warmup)

    results = {}
    for csms in args.comm_sms_list:
        results[csms] = time_iters(
            lambda: overlapped_iter(csms), args.iters, args.warmup)

    if rank == 0:
        gemm_flops = 2 * total_m * args.n * args.hidden
        print(f"\npure GEMM   ({num_sms} SMs): {t_gemm_pure:8.3f} ms "
              f"({gemm_flops / t_gemm_pure / 1e9:.0f} TFLOPS)")
        for csms in args.comm_sms_list:
            print(f"pure dispatch ({csms:3d} SMs): {t_disp_pure[csms]:8.3f} ms")
        print(f"serial (dispatch then GEMM): {t_serial:8.3f} ms")
        print(f"\n{'comm_sms':>8} {'overlapped':>12} {'vs serial':>10} {'vs ideal':>10}")
        for csms, t in results.items():
            ideal = max(t_gemm_pure, t_disp_pure[csms])
            print(f"{csms:>8} {t:>10.3f}ms {t_serial / t:>9.2f}x {t / ideal:>9.2f}x")

    nvshmem.core.free_tensor(input_buf)
    nvshmem.core.free_tensor(recv_buf)
    nvshmem.core.free_tensor(flags)


def main():
    parser = argparse.ArgumentParser(description="TilePipe dispatch + gated grouped GEMM")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--gemm-n", dest="n", type=int, default=2048,
                        help="GEMM N (per-expert FFN width; 2x intermediate for gate+up)")
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--comm-warps", type=int, default=16)
    parser.add_argument("--publish", choices=["token", "segment"], default="segment",
                        help="flag granularity: per token, or one release per "
                             "(source, expert) segment via local completion counters")
    parser.add_argument("--comm-sms", type=str, default="8,16,24,32",
                        help="comma-separated num_comm_sms values to sweep")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    args.comm_sms_list = [int(x) for x in args.comm_sms.split(",")]
    assert args.hidden % 512 == 0, "hidden must be divisible by 512 (256-bit warp copies)"

    # NVSHMEM init is a common silent-hang point (UID broadcast + peer
    # bootstrap): log explicitly on both sides of it.
    print(f"[pre-init] pid={os.getpid()} RANK={os.environ.get('RANK')} "
          f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    torchrun_uid_init_bcast()
    rank = dist.get_rank()
    print(f"[rank {rank}] nvshmem init OK: world={dist.get_world_size()} "
          f"device=cuda:{torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name()})", flush=True)
    try:
        run_tilepipe(args)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        torchrun_finalize()


if __name__ == "__main__":
    main()
