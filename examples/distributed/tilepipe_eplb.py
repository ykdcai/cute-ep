"""
TilePipe-EPLB: unified dispatch kernel — token sends and expert-weight
transfers as ONE work list (see tilepipe.md "Implement Expert Parallel Load
Balancing").

A weight row is a K-vector, exactly like a token's hidden vector, so both
transfers are streams of the same row-copy unit. The kernel walks one flat,
segment-sorted unit list; a segment is either a (source, expert, dst) token
group or one expert weight push, described uniformly by per-segment metadata:

    seg_dst[s]       destination rank
    seg_is_weight[s] source/dest buffer selector (token recv vs weight staging)
    seg_flag[s]      index into the destination's SINGLE flag array
                     ([0, epr) = token arrival counters, [epr, ...) = weight flags)
    seg_pub[s]       value the completing warp publishes (segment size for
                     token counters, 1 for weight-ready flags)

Completion uses the counting protocol from tilepipe.py: warps count their
units of the current segment into local seg_done[s] (acq_rel/sys); the warp
whose increment completes the segment issues the single remote
atomic_add(flag + seg_flag[s], seg_pub[s], release, sys). Counters are
arrival-order-agnostic, so the host may slice weight pushes into interleaved
sub-runs (round-robin across destinations) without kernel changes.

Scheduling is therefore purely a host-side ordering problem. Per tilepipe.md
+ Johnson's rule for the two-stage (comm -> GEMM) pipeline: local-weight
experts first in increasing dependency bytes, transferred experts last in
decreasing token count; per destination the list is deadline-ordered (the
destination GEMM's expert order); with >2 ranks, weight bursts are sliced
round-robin across destinations so no consumer starves.

This file's driver is the weights-only MVP: a synthetic EPLB swap job list
    job[j] = (src_weight_slot, dst_rank, dst_weight_slot, dst_flag_idx)
(any placement — e.g. DeepSeek EPLB phy2log — lowers to these arrays),
correctness check + bandwidth sweep. Token+weight combined lists reuse the
same kernel.

Run (2 GPUs):
    torchrun --nproc-per-node 2 examples/distributed/tilepipe_eplb.py
"""

import argparse
import functools
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

from quack.tilepipe_sync import ExpertArrivalSemaphore

# Stage prints are hang diagnostics — they must not sit in a stdio buffer.
print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Unified dispatch kernel
# ---------------------------------------------------------------------------


@cute.jit
def _flush_segment(
    cur_seg: Int32,
    cnt: Int32,
    seg_done: cute.Tensor,       # [num_segs] int32, LOCAL completion counters
    seg_sizes: cute.Tensor,      # [num_segs] int32
    seg_dst: cute.Tensor,        # [num_segs] int32
    seg_flag: cute.Tensor,       # [num_segs] int32
    seg_pub: cute.Tensor,        # [num_segs] int32
    flag_peer_ptrs: cute.Tensor,  # [world_size] int64
    lane_id: Int32,
):
    cute.arch.sync_warp()
    if lane_id == 0:
        old = cute.arch.atomic_add(
            seg_done.iterator + cur_seg, cnt, sem="acq_rel", scope="sys")
        if old + cnt == seg_sizes[cur_seg]:
            # Last units of this segment: publish to the destination. The
            # acquire->release chain through seg_done makes this release
            # cumulative over every contributing warp's remote stores.
            sem = ExpertArrivalSemaphore(peer_ptrs=flag_peer_ptrs)
            sem.arrive(seg_dst[cur_seg], seg_flag[cur_seg], seg_pub[cur_seg])


@cute.kernel
def unified_dispatch_kernel(
    token_buf: cute.Tensor,        # [num_tokens, K] bf16: local token data
    weight_buf: cute.Tensor,       # [wbuf_rows, K] bf16: local weights (+staging)
    recv_peer_ptrs: cute.Tensor,   # [world] int64: peer token recv buffers
    wstage_peer_ptrs: cute.Tensor,  # [world] int64: peer weight buffers
    flag_peer_ptrs: cute.Tensor,   # [world] int64: peer unified flag arrays
    u_src_row: cute.Tensor,        # [total_units] int32: row in selected src buf
    u_dst_row: cute.Tensor,        # [total_units] int32: row in selected dst buf
    u_seg: cute.Tensor,            # [total_units] int32: ascending segment id
    seg_dst: cute.Tensor,
    seg_flag: cute.Tensor,
    seg_pub: cute.Tensor,
    seg_is_weight: cute.Tensor,    # [num_segs] int32: buffer selector
    seg_sizes: cute.Tensor,
    seg_done: cute.Tensor,
    total_units: Int32,
    _K: Int32,
    recv_rows: Int32,              # peer recv buffer row count (layout)
    wbuf_rows: Int32,              # peer weight buffer row count (layout)
):
    K = cute.assume(_K, divby=32)
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

    dtype = token_buf.element_type
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

    num_tokens = cute.size(token_buf, mode=[0])
    local_wrows = cute.size(weight_buf, mode=[0])
    tok_src = cute.make_tensor(
        token_buf.iterator.align(copy_align),
        cute.make_ordered_layout((num_tokens, K), order=(1, 0)))
    w_src = cute.make_tensor(
        weight_buf.iterator.align(copy_align),
        cute.make_ordered_layout((local_wrows, K), order=(1, 0)))
    tSgS_tok = thr_copy_load.partition_S(tok_src)
    tSgS_w = thr_copy_load.partition_S(w_src)
    frg = cute.make_fragment_like(tSgS_tok[None, 0, 0])
    k_iters = cute.size(tSgS_tok, mode=[2])

    recv_layout = cute.make_ordered_layout((recv_rows, K), order=(1, 0))
    wstage_layout = cute.make_ordered_layout((wbuf_rows, K), order=(1, 0))

    cur_seg = Int32(-1)
    cnt = Int32(0)
    for g in range(global_warp_id, total_units, total_warps):
        s = u_seg[g]
        if s != cur_seg:
            if cnt > 0:
                _flush_segment(cur_seg, cnt, seg_done, seg_sizes, seg_dst,
                               seg_flag, seg_pub, flag_peer_ptrs, lane_id)
            cur_seg = s
            cnt = Int32(0)

        src_row = u_src_row[g]
        dst_row = u_dst_row[g]
        dst = seg_dst[s]
        if seg_is_weight[s] == 0:
            remote_ptr = cute.make_ptr(
                dtype, recv_peer_ptrs[dst], cute.AddressSpace.gmem,
                assumed_align=copy_align)
            tDgD = thr_copy_store.partition_D(
                cute.make_tensor(remote_ptr, recv_layout))
            for k in range(k_iters):
                cute.copy(thr_copy_load, tSgS_tok[None, src_row, k], frg)
                cute.copy(thr_copy_store, frg, tDgD[None, dst_row, k])
        else:
            remote_ptr = cute.make_ptr(
                dtype, wstage_peer_ptrs[dst], cute.AddressSpace.gmem,
                assumed_align=copy_align)
            tDgD = thr_copy_store.partition_D(
                cute.make_tensor(remote_ptr, wstage_layout))
            for k in range(k_iters):
                cute.copy(thr_copy_load, tSgS_w[None, src_row, k], frg)
                cute.copy(thr_copy_store, frg, tDgD[None, dst_row, k])
        cnt += 1

    if cnt > 0:
        _flush_segment(cur_seg, cnt, seg_done, seg_sizes, seg_dst,
                       seg_flag, seg_pub, flag_peer_ptrs, lane_id)


@cute.jit
def unified_dispatch(
    token_buf: cute.Tensor,
    weight_buf: cute.Tensor,
    recv_peer_ptrs: cute.Tensor,
    wstage_peer_ptrs: cute.Tensor,
    flag_peer_ptrs: cute.Tensor,
    u_src_row: cute.Tensor,
    u_dst_row: cute.Tensor,
    u_seg: cute.Tensor,
    seg_dst: cute.Tensor,
    seg_flag: cute.Tensor,
    seg_pub: cute.Tensor,
    seg_is_weight: cute.Tensor,
    seg_sizes: cute.Tensor,
    seg_done: cute.Tensor,
    total_units: Int32,
    K: Int32,
    recv_rows: Int32,
    wbuf_rows: Int32,
    num_ctas: Int32,
    num_warps: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    unified_dispatch_kernel(
        token_buf, weight_buf, recv_peer_ptrs, wstage_peer_ptrs, flag_peer_ptrs,
        u_src_row, u_dst_row, u_seg,
        seg_dst, seg_flag, seg_pub, seg_is_weight, seg_sizes, seg_done,
        total_units, K, recv_rows, wbuf_rows,
    ).launch(
        grid=[num_ctas, 1, 1],
        block=[num_warps * 32, 1, 1],
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Host: unified-list builders
# ---------------------------------------------------------------------------


def weight_jobs_to_segments(job_src, job_dst_rank, job_dst_slot, job_flag, n_rows):
    """Lower an EPLB job list to unified-list arrays: one segment per job,
    n_rows units per segment (unit = one weight row). Weight-ready flags
    publish 1; the GEMM-side target is 1."""
    num_jobs = len(job_src)
    j = np.repeat(np.arange(num_jobs, dtype=np.int64), n_rows)
    row = np.tile(np.arange(n_rows, dtype=np.int64), num_jobs)
    to32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)
    units = dict(
        u_src_row=to32(job_src[j] * n_rows + row),
        u_dst_row=to32(job_dst_slot[j] * n_rows + row),
        u_seg=to32(j),
    )
    segs = dict(
        seg_dst=to32(job_dst_rank),
        seg_flag=to32(job_flag),
        seg_pub=to32(np.ones(num_jobs)),
        seg_is_weight=to32(np.ones(num_jobs)),
        seg_sizes=to32(np.full(num_jobs, n_rows)),
    )
    return units, segs


# ---------------------------------------------------------------------------
# Weights-only MVP driver: synthetic EPLB swap, correctness + bandwidth sweep
# ---------------------------------------------------------------------------


def make_weight(rank, slot, n, k, device):
    """Deterministic per-(owner, slot) weight so the receiver can rebuild the
    reference locally (Philox is counter-based: same seed -> same bits)."""
    g = torch.Generator(device=device)
    g.manual_seed(rank * 100003 + slot)
    return torch.randn((n, k), generator=g, device=device, dtype=torch.float32).to(
        torch.bfloat16)


def run(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "MVP swap config assumes 2 ranks"
    peer = 1 - rank
    device = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    n, k, T = args.n, args.hidden, args.transfer
    weight_bytes = n * k * 2

    if rank == 0:
        print(f"TilePipe-EPLB weight dispatch: N={n} K={k} transfer={T} weights/rank "
              f"({weight_bytes / 1e6:.1f} MB each, {T * weight_bytes / 1e6:.1f} MB total) "
              f"world={world_size} SMs={num_sms}")

    # Symmetric weight buffer: slots [0, T) = this rank's resident ("hot")
    # weights, slots [T, 2T) = staging the peer pushes into. Flags: T weight
    # flags (a combined run would put them at offset epr in the token flag
    # array; the kernel only sees flag indices).
    buf_slots = 2 * T
    symm_bytes = buf_slots * n * k * 2 + T * 4
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] allocating symmetric weight buffer: {symm_bytes / 1e9:.3f} GB "
          f"(free: {free_bytes / 1e9:.2f} GB)")
    wbuf = nvshmem.core.tensor((buf_slots * n, k), dtype=torch.bfloat16)
    wflags = nvshmem.core.tensor((T,), dtype=torch.int32)
    wflags.fill_(0)
    for s in range(T):
        wbuf[s * n:(s + 1) * n] = make_weight(rank, s, n, k, device)
    wbuf[T * n:] = 0
    wstage_peer_ptrs = _peer_ptr_tensor(wbuf, world_size, device)
    flag_peer_ptrs = _peer_ptr_tensor(wflags, world_size, device)
    print(f"[rank {rank}] symmetric buffers allocated")

    # Synthetic EPLB swap: push hot slot s to the peer's staging slot T+s,
    # flag s. Any real placement (DeepSeek phy2log) lowers to the same arrays.
    job_src = np.arange(T, dtype=np.int64)
    job_dst_rank = np.full(T, peer, dtype=np.int64)
    job_dst_slot = np.arange(T, 2 * T, dtype=np.int64)
    job_flag = np.arange(T, dtype=np.int64)
    units, segs = weight_jobs_to_segments(job_src, job_dst_rank, job_dst_slot,
                                          job_flag, n)
    dev = lambda a: torch.from_numpy(a).to(device)
    u_src_row, u_dst_row, u_seg = (dev(units[x]) for x in
                                   ("u_src_row", "u_dst_row", "u_seg"))
    seg_t = {x: dev(segs[x]) for x in segs}
    seg_done = torch.zeros(T, dtype=torch.int32, device=device)
    total_units = len(units["u_seg"])

    # Weights-only run: the token buffer/recv side is unused but the kernel
    # signature is unified — pass 1-row dummies.
    token_dummy = torch.zeros((1, k), dtype=torch.bfloat16, device=device)
    recv_dummy_ptrs = wstage_peer_ptrs.clone()  # never dereferenced (no token segs)

    if rank == 0:
        print(f"Compiling unified dispatch kernel ({args.comm_warps} warps/CTA)...")
    dispatch_args = lambda: (
        from_dlpack(token_dummy, assumed_align=32), from_dlpack(wbuf, assumed_align=32),
        from_dlpack(recv_dummy_ptrs), from_dlpack(wstage_peer_ptrs),
        from_dlpack(flag_peer_ptrs),
        from_dlpack(u_src_row), from_dlpack(u_dst_row), from_dlpack(u_seg),
        from_dlpack(seg_t["seg_dst"]), from_dlpack(seg_t["seg_flag"]),
        from_dlpack(seg_t["seg_pub"]), from_dlpack(seg_t["seg_is_weight"]),
        from_dlpack(seg_t["seg_sizes"]), from_dlpack(seg_done),
        Int32(total_units), Int32(k), Int32(1), Int32(buf_slots * n))
    compiled = cute.compile(
        unified_dispatch, *dispatch_args(), Int32(args.comm_sms_list[0]),
        args.comm_warps, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    print(f"[rank {rank}] unified dispatch kernel compiled")

    stream = torch.cuda.Stream()

    def launch(num_ctas):
        compiled(*dispatch_args(), Int32(num_ctas), cuda.CUstream(stream.cuda_stream))

    def reset():
        wflags.fill_(0)
        seg_done.zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    # Warm-up execution (first launch does lazy module load) doubles as the
    # correctness check: flags land on 1, staging bytes match the peer's
    # weights bit-exactly.
    reset()
    launch(args.comm_sms_list[0])
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])
    ok_flags = torch.equal(wflags.cpu(), torch.ones(T, dtype=torch.int32))
    ok_data = True
    for s in range(T):
        expected = make_weight(peer, s, n, k, device)
        if not torch.equal(wbuf[(T + s) * n:(T + s + 1) * n], expected):
            ok_data = False
            bad = (wbuf[(T + s) * n:(T + s + 1) * n] != expected).sum().item()
            print(f"[rank {rank}] slot {s}: {bad} mismatched elements")
    print(f"[rank {rank}] correctness: "
          f"flags={'OK' if ok_flags else 'FAIL ' + str(wflags.cpu().tolist())} "
          f"data={'OK' if ok_data else 'FAIL'}")
    ok_t = torch.tensor([ok_flags and ok_data], dtype=torch.int32, device=device)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if not bool(ok_t.item()):
        raise SystemExit(f"[rank {rank}] weight dispatch correctness FAILED")

    if not args.benchmark:
        return

    # --- Bandwidth sweep ---
    if rank == 0:
        print(f"\nbenchmark: {T} weights x {weight_bytes / 1e6:.1f} MB pushed per rank")
        print(f"{'comm_sms':>8} {'time':>10} {'GB/s':>8} {'per-weight':>11}")
    total_bytes = T * weight_bytes
    for csms in args.comm_sms_list:
        times = []
        for it in range(args.warmup + args.iters):
            reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            launch(csms)
            end.record(stream)
            torch.cuda.synchronize()
            if it >= args.warmup:
                times.append(start.elapsed_time(end))
        t = torch.tensor([float(np.median(times))], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        t_ms = t.item()
        if rank == 0:
            # Segments complete in list order, so time-to-first-weight ~= t/T:
            # the number that gates the GEMM's first transferred expert.
            print(f"{csms:>8} {t_ms:>8.3f}ms {total_bytes / t_ms / 1e6:>8.0f} "
                  f"{t_ms / T:>9.3f}ms")

    nvshmem.core.free_tensor(wbuf)
    nvshmem.core.free_tensor(wflags)


def main():
    parser = argparse.ArgumentParser(description="TilePipe-EPLB weight dispatch MVP")
    parser.add_argument("--gemm-n", dest="n", type=int, default=4096,
                        help="weight rows N (gate+up FFN width)")
    parser.add_argument("--hidden", type=int, default=7168, help="weight cols K")
    parser.add_argument("--transfer", type=int, default=4,
                        help="weights pushed per rank (EPLB swap size)")
    parser.add_argument("--comm-warps", type=int, default=16)
    parser.add_argument("--comm-sms", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    args.comm_sms_list = [int(x) for x in args.comm_sms.split(",")]
    assert args.hidden % 512 == 0, "K must be divisible by 512 (256-bit warp copies)"

    if "NVSHMEM_SYMMETRIC_SIZE" not in os.environ:
        heap = 2 * args.transfer * args.n * args.hidden * 2 + 128 * 1024 * 1024
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(heap)
    print(f"[pre-init] pid={os.getpid()} RANK={os.environ.get('RANK')} "
          f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"NVSHMEM_SYMMETRIC_SIZE={os.environ['NVSHMEM_SYMMETRIC_SIZE']}")
    torchrun_uid_init_bcast()
    rank = dist.get_rank()
    print(f"[rank {rank}] nvshmem init OK: world={dist.get_world_size()} "
          f"device=cuda:{torch.cuda.current_device()} "
          f"({torch.cuda.get_device_name()})")
    try:
        run(args)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        torchrun_finalize()


if __name__ == "__main__":
    main()
