"""
TilePipe GEMM -> combine overlap (see tilepipe_guide.md): the grouped GEMM is
the PRODUCER — its epilogue publishes per-m-tile completion counters to every
rank (quack_gemm tile_flag_ptrs) — and the gated TMA combine is the CONSUMER,
pulling each token's topk expert-output rows from peers once their tiles are
complete. Two kernels, two streams, disjoint SMs; scoped to one GEMM (in a
real SwiGLU MoE layer this producer is the down-projection).

Inputs are synthetic pre-dispatched activations (A already sits in each
rank's expert-grouped buffer — dispatch overlap is the other, already-built
pipeline). Deterministic per-rank seeds let every rank rebuild the peers'
GEMM outputs for the reference combine.

Run (2 GPUs):
    torchrun --nproc-per-node 2 examples/distributed/tilepipe_gemm_combine.py
"""

import argparse
import functools
import os

import numpy as np
import torch
import torch.distributed as dist


import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda_driver

import nvshmem.core

from moe_comm import (torchrun_uid_init_bcast, torchrun_finalize,
                      CombineTmaKernel, VarlenAllToAllKernel)

from quack.gemm import gemm as quack_gemm
from quack.tilepipe import build_combine_metadata, plan_combine, peer_ptr_tensor
from quack.tilepipe_sync import wait_flag

# Stage prints are hang diagnostics — they must not sit in a stdio buffer.
print = functools.partial(print, flush=True)


def rank_tensor(seed, shape, device, scale=1.0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=torch.float32)
            * scale).to(torch.bfloat16)


def expert_weight(d, e, n, k, device):
    # Per-(rank, expert) seed so peers can rebuild ONE expert at a time: a
    # whole-rank randn((epr, n, k)) needs an fp32 temp of ~4 GB at DSv3 shape,
    # which thrashes the crowded GPUs during the reference build.
    return rank_tensor(900 + d * 4096 + e, (n, k), device, scale=k ** -0.25)


def build_weights(d, epr, n, k, device):
    w = torch.empty((epr, n, k), dtype=torch.bfloat16, device=device)
    for e in range(epr):
        w[e] = expert_weight(d, e, n, k, device)
    return w


def run(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    epr = args.experts // world_size
    n, k = args.n, args.hidden
    n_tiles = (n + args.tile_n - 1) // args.tile_n
    torch.manual_seed(42 + rank)

    if rank == 0:
        print(f"TilePipe GEMM->combine: tokens/rank={args.tokens} K={k} N={n} "
              f"experts={args.experts} (local {epr}) topk={args.topk} "
              f"tile={args.tile_m}x{args.tile_n} (n_tiles={n_tiles}) "
              f"world={world_size} SMs={num_sms}")

    # --- Routing + shared metadata (one derivation for both sides) ---
    topk_indices = torch.randint(
        0, args.experts, (args.tokens, args.topk), dtype=torch.int32, device=device)
    all_topk_t = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    dist.all_gather(all_topk_t, topk_indices.contiguous())
    all_topk = np.stack([t.cpu().numpy() for t in all_topk_t])
    meta = build_combine_metadata(all_topk, args.experts, rank, world_size,
                                  tile_m=args.tile_m)
    total_m = meta["rank_rows"][rank]
    max_rows = max(max(meta["rank_rows"]), 1)
    print(f"[rank {rank}] metadata built: total_m={total_m} max_rows={max_rows} "
          f"total_tiles={meta['total_tiles']}")

    # --- Local GEMM operands (deterministic per rank for references) ---
    weights_bytes = epr * n * k * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] allocating weights {weights_bytes / 1e9:.2f} GB + peer "
          f"references (free: {free_bytes / 1e9:.2f} GB)")
    A = rank_tensor(500 + rank, (total_m, k), device, scale=k ** -0.5)
    weights = build_weights(rank, epr, n, k, device)

    # --- Symmetric buffers: expert output D (peers pull rows) + tile flags ---
    d_buf = nvshmem.core.tensor((max_rows, n), dtype=torch.bfloat16)
    d_buf.fill_(0)
    tile_flags = nvshmem.core.tensor((meta["total_tiles"],), dtype=torch.int32)
    tile_flags.fill_(0)
    d_peer_ptrs = peer_ptr_tensor(d_buf, world_size, device)
    # The GEMM publishes relative to ITS segment of each rank's flag array.
    flag_pub_ptrs = torch.tensor(
        [nvshmem.core.get_peer_tensor(tile_flags, r).data_ptr()
         + int(meta["rank_tile_base"][rank]) * 4
         for r in range(world_size)], device=device, dtype=torch.int64)
    print(f"[rank {rank}] symmetric buffers allocated "
          f"({(max_rows * n * 2 + meta['total_tiles'] * 4) / 1e9:.3f} GB)")

    D = d_buf[:total_m]
    cu_seqlens_m = torch.from_numpy(meta["cu_seqlens"]).to(device)
    tile_offsets_t = torch.from_numpy(meta["tile_offsets"]).to(device)
    scatter_t = torch.from_numpy(meta["scatter"]).to(device)
    flag_idx_t = torch.from_numpy(meta["flag_idx"]).to(device)
    combine_out = torch.zeros((args.tokens, n), dtype=torch.bfloat16, device=device)
    ntok_t = torch.full((world_size,), args.tokens, dtype=torch.int32, device=device)

    def launch_gemm(publish, max_clusters=None, stream=None, local_publish=False):
        # publish=False: plain GEMM. local_publish=True: push-combine mode —
        # flags land only on this rank (1-entry pointer table).
        ptrs = (flag_pub_local if local_publish else flag_pub_ptrs) if publish else None
        with torch.cuda.stream(stream if stream is not None else
                               torch.cuda.current_stream()):
            quack_gemm(
                A, weights, D, C=None, tile_count_semaphore=None,
                tile_M=args.tile_m, tile_N=args.tile_n, cluster_M=1, cluster_N=1,
                persistent=True, cu_seqlens_m=cu_seqlens_m,
                max_active_clusters=max_clusters,
                tile_flag_ptrs=ptrs,
                tile_flag_offsets=tile_offsets_t if publish else None)

    # --- Combine kernel (gated; hchunk must divide N) ---
    assert n % args.hchunk == 0, "--hchunk must divide --gemm-n"
    combine_kernel = CombineTmaKernel(
        cutlass.BFloat16, n, args.topk, hchunk=args.hchunk,
        num_stages=args.combine_stages)
    peer_tensors = [from_dlpack(nvshmem.core.get_peer_tensor(d_buf, r))
                    for r in range(world_size)]
    base_args = lambda: (
        peer_tensors, from_dlpack(combine_out), from_dlpack(topk_indices),
        from_dlpack(scatter_t), from_dlpack(ntok_t))
    gate_args = lambda: (
        from_dlpack(tile_flags), from_dlpack(flag_idx_t), Int32(n_tiles))

    # --- Push-combine (reverse dispatch): this rank PUSHES each of its D rows
    # back to the token's home rank as soon as the row's tile is produced, and
    # the home rank reduces staging[token, topk, N] locally. Row readiness is
    # per-EXPERT (uniform across the GEMM window), unlike the pull combine's
    # per-token readiness (all topk => back-loaded), so overlap is much better.
    # The gate is LOCAL here: rank r waits on rank r's own tile counters.
    pplan = plan_combine(all_topk, args.experts, rank, world_size,
                         tile_m=args.tile_m)
    p_gate = pplan.gate_idx
    dev = lambda a: torch.from_numpy(a).to(device)
    # Push mode's gate space is rank-LOCAL (0-based per-expert m-tiles, same
    # tiling rule the GEMM epilogue publishes with), so it gets its own local
    # flag array and a 1-entry publish pointer table: arrive_all then writes
    # only to this rank — no cross-rank flag traffic at all.
    local_tiles = int(p_gate.max()) + 1 if len(p_gate) else 1
    tile_flags_local = torch.zeros(local_tiles, dtype=torch.int32, device=device)
    flag_pub_local = torch.tensor([tile_flags_local.data_ptr()],
                                  device=device, dtype=torch.int64)
    stage_rows = pplan.dst_rows
    staging = nvshmem.core.tensor((stage_rows, n), dtype=torch.bfloat16)
    staging.fill_(0)
    arrivals = nvshmem.core.tensor((1,), dtype=torch.int32)
    arrivals.fill_(0)
    stage_peer_ptrs = peer_ptr_tensor(staging, world_size, device)
    arriv_peer_ptrs = peer_ptr_tensor(arrivals, world_size, device)
    dplan = pplan.to_device(device)
    push_kernel = VarlenAllToAllKernel(cutlass.BFloat16, n,
                                       num_stages=args.push_stages,
                                       workers=args.push_workers)
    push_args = lambda ctas, strm: dplan.args(
        d_buf, stage_peer_ptrs, arriv_peer_ptrs, ctas, strm,
        gate_flags=tile_flags_local, gate_target=n_tiles)
    print(f"[rank {rank}] push-combine: {pplan.n_rows} rows, staging "
          f"{stage_rows * n * 2 / 1e9:.3f} GB")
    print(f"[rank {rank}] compiling combine (gated + ungated)...")
    comm_stream = torch.cuda.Stream(priority=-1)
    gemm_stream = torch.cuda.Stream()
    cs = lambda s: cuda_driver.CUstream(s.cuda_stream)
    cur = lambda: cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    # NOTE cute launches ignore torch.cuda.stream() contexts — the stream is
    # passed explicitly (a default-stream combine under comm_stream events
    # silently measured nothing).
    combine_gated = {c: cute.compile(combine_kernel, *base_args(), epr, rank,
                                     world_size, c, *gate_args(), cur())
                     for c in args.combine_ctas_list}
    combine_ungated = {c: cute.compile(combine_kernel, *base_args(), epr, rank,
                                       world_size, c, None, None, None, cur())
                       for c in args.combine_ctas_list}
    push_compiled = {c: cute.compile(push_kernel, *dplan.compile_args(
        d_buf, stage_peer_ptrs, arriv_peer_ptrs, c, rank, world_size, cur(),
        gate_flags=tile_flags_local, gate_target=n_tiles))
        for c in args.combine_ctas_list}
    arrivals_target = Int32(stage_rows)
    wait_compiled = cute.compile(
        wait_flag, from_dlpack(arrivals), Int32(0), arrivals_target, cur())
    print(f"[rank {rank}] combine + push kernels compiled")

    def launch_push(c, strm):
        push_compiled[c](*push_args(c, cs(strm)))

    def launch_reduce(strm):
        # Gate the local reduce on arrivals, then a plain torch contraction —
        # both enqueued, no host sync (a spinning gate + host CUDA call would
        # deadlock).
        wait_compiled(from_dlpack(arrivals), Int32(0), arrivals_target, cs(strm))
        with torch.cuda.stream(strm):
            torch.sum(staging.view(args.tokens, args.topk, n), dim=1,
                      out=combine_out)

    def reset():
        tile_flags.fill_(0)
        tile_flags_local.zero_()
        dplan.reset()
        arrivals.fill_(0)
        combine_out.zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    # --- Warm-up + serial correctness (every kernel EXECUTES before overlap:
    # lazy module load deadlocks against a spinning gated kernel) ---
    print(f"[rank {rank}] GEMM warm-up (publishing + plain)...")
    reset()
    launch_gemm(publish=True)
    launch_gemm(publish=False)
    barrier()
    # Every producer publishes to every rank, so the whole array lands on
    # n_tiles.
    exp_flags = torch.full((meta["total_tiles"],), n_tiles, dtype=torch.int32)
    ok_flags = torch.equal(tile_flags.cpu(), exp_flags)
    print(f"[rank {rank}] GEMM tile flags {'OK' if ok_flags else 'FAIL'}")
    if not ok_flags:
        raise SystemExit(f"[rank {rank}] tile-flag publish FAILED: "
                         f"{tile_flags.cpu().tolist()[:16]}...")

    # Reference: rebuild every peer's D and gather (fp32 accumulate). Memory-
    # tight on crowded GPUs: peer weights are rebuilt ONE EXPERT at a time
    # (whole-rank randn((epr, n, k)) needs a ~4 GB fp32 temp at DSv3 shape,
    # which thrashes against co-tenants), with a cache trim per peer.
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] building reference (free: {free_bytes / 1e9:.2f} GB)...")
    ref = torch.zeros((args.tokens, n), dtype=torch.float32, device=device)
    scatter_l = scatter_t.long()
    src_rank_t = torch.from_numpy(meta["src_rank"]).to(device)
    for d in range(world_size):
        if d == rank:
            D_d = D.float()
        else:
            A_d = rank_tensor(500 + d, (meta["rank_rows"][d], k), device,
                              scale=k ** -0.5)
            # per-expert matmul using the producer's cu_seqlens
            meta_d = build_combine_metadata(all_topk, args.experts, d, world_size,
                                            tile_m=args.tile_m)
            D_d = torch.zeros((meta["rank_rows"][d], n), dtype=torch.float32,
                              device=device)
            cu = meta_d["cu_seqlens"]
            for e in range(epr):
                lo, hi = int(cu[e]), int(cu[e + 1])
                if hi > lo:
                    W_e = expert_weight(d, e, n, k, device)
                    D_d[lo:hi] = A_d[lo:hi].float() @ W_e.float().T
                    del W_e
            del A_d
        mask = src_rank_t == d
        ref += (D_d[scatter_l.clamp(0, meta["rank_rows"][d] - 1)]
                * mask.unsqueeze(-1)).sum(dim=1)
        del D_d
        torch.cuda.empty_cache()  # nothing is spinning here; trim fragmentation
        print(f"[rank {rank}] reference: peer {d} done")
    torch.cuda.synchronize()

    def check(tag):
        rel = ((combine_out.float() - ref).abs().max()
               / ref.abs().max().clamp(min=1e-6)).item()
        ok = rel < 3e-2
        print(f"[rank {rank}] {tag}: rel_err={rel:.2e} {'OK' if ok else 'FAIL'}")
        ok_t = torch.tensor([ok], dtype=torch.int32, device=device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        return bool(ok_t.item())

    # Combine warm-up + serial correctness (flags already satisfied).
    c0 = args.combine_ctas_list[0]
    combine_gated[c0](*base_args(), *gate_args(), cur())
    barrier()
    if not check("serial GEMM->combine"):
        raise SystemExit("serial correctness FAILED")
    combine_ungated[c0](*base_args(), cur())
    barrier()

    # --- Overlapped correctness: combine first (high-priority comm stream,
    # it spins on flags), then the capped publishing GEMM ---
    def overlapped(c_ctas):
        combine_gated[c_ctas](*base_args(), *gate_args(), cs(comm_stream))
        launch_gemm(publish=True, max_clusters=num_sms - c_ctas,
                    stream=gemm_stream)

    reset()
    overlapped(c0)
    barrier()
    if not check("overlapped GEMM->combine (pull)"):
        raise SystemExit("overlapped pull correctness FAILED")

    # --- Push-combine: warm-up EXECUTION of every kernel (lazy module load
    # deadlocks against a spinning gate), then serial + overlapped checks ---
    reset()
    launch_gemm(publish=True, local_publish=True)
    launch_push(c0, comm_stream)
    launch_reduce(comm_stream)
    barrier()
    if not check("serial GEMM->push-combine"):
        raise SystemExit("push-combine serial correctness FAILED")

    def overlapped_push(c_ctas):
        # Push kernel first on the high-priority stream: it gates on this
        # rank's own tile counters, so it streams rows out as the GEMM
        # produces them. The reduce follows on the same stream behind the
        # arrival gate.
        launch_push(c_ctas, comm_stream)
        launch_reduce(comm_stream)
        launch_gemm(publish=True, local_publish=True,
                    max_clusters=num_sms - c_ctas, stream=gemm_stream)

    reset()
    overlapped_push(c0)
    barrier()
    if not check("overlapped GEMM->push-combine"):
        raise SystemExit("push-combine overlapped correctness FAILED")

    if not args.benchmark:
        return

    # --- Benchmark ---
    def time_iters(enqueue, pre_reset):
        times = []
        for it in range(args.warmup + args.iters):
            if pre_reset:
                reset()
            else:
                barrier()
            start = torch.cuda.Event(enable_timing=True)
            end_a = torch.cuda.Event(enable_timing=True)
            end_b = torch.cuda.Event(enable_timing=True)
            start.record(comm_stream)
            gemm_stream.wait_event(start)
            torch.cuda.current_stream().wait_event(start)
            enqueue()
            end_a.record(comm_stream)
            end_b.record(gemm_stream)
            torch.cuda.synchronize()
            if it >= args.warmup:
                times.append(max(start.elapsed_time(end_a),
                                 start.elapsed_time(end_b)))
        t = torch.tensor([float(np.median(times))], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    if rank == 0:
        print("\nbenchmark: pure GEMM / pure combine / serial / overlapped...")
    tile_flags.fill_(n_tiles)
    barrier()
    t_gemm_plain = time_iters(
        lambda: launch_gemm(publish=False, stream=gemm_stream), pre_reset=False)
    t_gemm_pub = time_iters(
        lambda: launch_gemm(publish=True, stream=gemm_stream), pre_reset=True)
    t_comb = {c: time_iters(
        lambda: combine_ungated[c](*base_args(), cs(comm_stream)),
        pre_reset=False) for c in args.combine_ctas_list}

    def serial(c):
        launch_gemm(publish=True, stream=comm_stream)
        combine_gated[c](*base_args(), *gate_args(), cs(comm_stream))

    # Fair serial baseline: each phase gets the whole GPU, so the combine
    # runs with the LARGEST CTA count in the sweep (a small-CTA serial
    # combine would pad the baseline and flatter the overlap numbers).
    c_serial = max(args.combine_ctas_list)
    t_serial = time_iters(lambda: serial(c_serial), pre_reset=True)
    t_over = {c: time_iters(lambda: overlapped(c), pre_reset=True)
              for c in args.combine_ctas_list}

    def serial_push(c):
        launch_gemm(publish=True, local_publish=True, stream=comm_stream)
        launch_push(c, comm_stream)
        launch_reduce(comm_stream)

    t_push_serial = time_iters(lambda: serial_push(c_serial), pre_reset=True)
    t_push_over = {c: time_iters(lambda: overlapped_push(c), pre_reset=True)
                   for c in args.combine_ctas_list}

    if rank == 0:
        flops = 2 * total_m * n * k
        print(f"\npure GEMM   ({num_sms} SMs): {t_gemm_plain:8.3f} ms "
              f"({flops / t_gemm_plain / 1e9:.0f} TFLOPS); with publish: "
              f"{t_gemm_pub:8.3f} ms ({(t_gemm_pub / t_gemm_plain - 1) * 100:+.1f}%)")
        for c in args.combine_ctas_list:
            print(f"pure combine ({c:3d} CTAs): {t_comb[c]:8.3f} ms")
        print(f"serial (GEMM then combine @{c_serial} CTAs): {t_serial:8.3f} ms")
        print(f"serial (GEMM then push-combine @{c_serial} CTAs): "
              f"{t_push_serial:8.3f} ms")
        print(f"\n{'ctas':>6} {'pull-ovl':>10} {'vs ser':>8} {'vs ideal':>9}"
              f" | {'push-ovl':>10} {'vs ser':>8} {'vs push-ser':>12}")
        for c in args.combine_ctas_list:
            ideal = max(t_gemm_pub, t_comb[c])
            print(f"{c:>6} {t_over[c]:>8.3f}ms {t_serial / t_over[c]:>7.2f}x "
                  f"{t_over[c] / ideal:>8.2f}x | {t_push_over[c]:>8.3f}ms "
                  f"{t_serial / t_push_over[c]:>7.2f}x "
                  f"{t_push_serial / t_push_over[c]:>11.2f}x")

    nvshmem.core.free_tensor(d_buf)
    nvshmem.core.free_tensor(tile_flags)
    nvshmem.core.free_tensor(staging)
    nvshmem.core.free_tensor(arrivals)


def main():
    parser = argparse.ArgumentParser(description="TilePipe GEMM -> gated combine")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=7168, help="GEMM K")
    parser.add_argument("--gemm-n", dest="n", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=None,
                        help="default: 32 * world_size")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--hchunk", type=int, default=2048,
                        help="combine bulk tile elems (must divide gemm-n)")
    parser.add_argument("--combine-stages", type=int, default=8)
    parser.add_argument("--combine-ctas", type=str, default="8,16,32,64")
    parser.add_argument("--push-stages", type=int, default=12,
                        help="SMEM stage budget for the push-combine kernel")
    parser.add_argument("--push-workers", type=int, default=4,
                        help="producer/consumer warp pairs per push CTA")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    args.combine_ctas_list = [int(x) for x in args.combine_ctas.split(",")]

    if "NVSHMEM_SYMMETRIC_SIZE" not in os.environ:
        # d_buf (~1.5x expected rows) + staging (tokens*topk exactly) + flags
        est_rows = int(1.5 * args.tokens * args.topk)
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(
            (est_rows + args.tokens * args.topk) * args.n * 2 + 256 * 1024 * 1024)
    print(f"[pre-init] pid={os.getpid()} RANK={os.environ.get('RANK')} "
          f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"NVSHMEM_SYMMETRIC_SIZE={os.environ['NVSHMEM_SYMMETRIC_SIZE']}")
    torchrun_uid_init_bcast()
    rank = dist.get_rank()
    if args.experts is None:
        args.experts = 32 * dist.get_world_size()
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
