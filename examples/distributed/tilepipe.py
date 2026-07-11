"""
TilePipe example/benchmark driver: MoE dispatch <-> grouped-GEMM overlap via
per-expert counting semaphores (see tilepipe.md). The kernel, metadata
builders, and the launch/reset/warm-up discipline live in quack.tilepipe —
this file only builds synthetic inputs, checks correctness against
references, and benchmarks.

Run (2 GPUs):
    torchrun --nproc-per-node 2 examples/distributed/tilepipe.py
"""

import argparse
import functools
import os

import numpy as np
import torch
import torch.distributed as dist

from moe_comm import torchrun_uid_init_bcast, torchrun_finalize

from quack.tilepipe import TilePipe, build_send_arrays

# Stage prints are hang diagnostics — they must not sit in a stdio buffer.
print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Reference checks
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

    # --- Memory budget (fail fast instead of thrashing a crowded GPU) ---
    weights_bytes = epr * args.n * args.hidden * 2
    exp_recv = args.tokens * args.topk
    out_bytes = int(1.25 * exp_recv) * args.n * 2
    ref_bytes = int(1.25 * exp_recv) * (args.n * 4 + args.hidden * 2)
    misc_bytes = 512 * 1024 * 1024  # inputs, metadata, NCCL, allocator slack
    torch_need = weights_bytes + out_bytes + ref_bytes + misc_bytes
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] memory budget: weights={weights_bytes / 1e9:.2f} GB "
          f"out+ref={(out_bytes + ref_bytes) / 1e9:.2f} GB misc={misc_bytes / 1e9:.2f} GB "
          f"-> need {torch_need / 1e9:.2f} GB, free {free_bytes / 1e9:.2f} GB "
          f"(symmetric heap already reserved at init)")
    if free_bytes < torch_need:
        raise RuntimeError(
            f"[rank {rank}] insufficient GPU memory: need ~{torch_need / 1e9:.2f} GB, "
            f"only {free_bytes / 1e9:.2f} GB free on device {device} "
            f"(co-tenant processes? check nvidia-smi). Refusing to allocate.")
    print(f"[rank {rank}] allocating weights: {weights_bytes / 1e9:.2f} GB")
    weights = torch.randn(
        (epr, args.n, args.hidden), dtype=torch.bfloat16, device=device) / (args.hidden ** 0.25)
    print(f"[rank {rank}] weights allocated")

    # --- Pipeline (metadata, symmetric buffers, compile, warm-up, validation
    # all happen inside; the deadlock discipline lives in quack.tilepipe) ---
    pipe = TilePipe(
        input_data=input_data,
        weights=weights,
        topk_indices=topk_indices,
        num_experts=args.experts,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        comm_warps=args.comm_warps,
        publish=args.publish,
        warmup_comm_sms=args.comm_sms_list[0],
        log=lambda msg: print(f"[rank {rank}] {msg}"),
    )

    # --- Correctness (overlapped) ---
    all_inputs = [torch.zeros_like(input_data) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_data.contiguous())
    recv_ref = build_recv_reference(
        all_inputs, pipe.all_topk, args.experts, rank, world_size, pipe.total_m, device)
    out_ref = grouped_gemm_reference(recv_ref, weights, pipe.cu_seqlens_m.cpu().numpy())

    pipe.reset()
    pipe.out.zero_()
    # Validate the riskiest oversubscribed path (largest oversub -> GEMM grid
    # most likely to cover all SMs; dispatch-first + high-priority comm
    # stream must prevent deadlock).
    pipe.run_overlapped(args.comm_sms_list[0], oversub_sms=max(args.oversub_sms_list))
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])

    ok_flags = torch.equal(pipe.flags.cpu(), pipe.split_sizes.cpu())
    ok_recv = torch.equal(pipe.recv_buf[:pipe.total_m], recv_ref)
    rel_err = ((pipe.out.float() - out_ref).abs().max() /
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
        pipe.free()
        return

    # --- Benchmark ---
    def time_iters(enqueue, iters, warmup):
        times = []
        for it in range(warmup + iters):
            pipe.reset()
            start = torch.cuda.Event(enable_timing=True)
            end_gemm = torch.cuda.Event(enable_timing=True)
            end_comm = torch.cuda.Event(enable_timing=True)
            start.record(pipe.comm_stream)
            pipe.gemm_stream.wait_event(start)
            enqueue()
            end_comm.record(pipe.comm_stream)
            end_gemm.record(pipe.gemm_stream)
            torch.cuda.synchronize()
            if it >= warmup:
                times.append(max(start.elapsed_time(end_gemm), start.elapsed_time(end_comm)))
        t = torch.tensor([float(np.median(times))], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank defines the step
        return t.item()

    if rank == 0:
        print("benchmark: pure GEMM / pure dispatch / serial / overlapped sweep...")
    # Ideal lower bounds: each phase alone with all resources.
    pipe.flags.copy_(pipe.split_sizes)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])

    t_gemm_pure = time_iters(
        lambda: pipe.launch_gemm(gated=False), args.iters, args.warmup)

    t_disp_pure = {}
    for csms in args.comm_sms_list:
        t_disp_pure[csms] = time_iters(
            lambda: pipe.launch_dispatch(csms), args.iters, args.warmup)

    t_serial = time_iters(pipe.run_serial, args.iters, args.warmup)

    # Sweep comm_sms x oversub_sms (0 = hard partition baseline).
    oversubs = [0] + args.oversub_sms_list
    results = {}
    for csms in args.comm_sms_list:
        for osub in oversubs:
            results[(csms, osub)] = time_iters(
                lambda: pipe.run_overlapped(csms, oversub_sms=osub),
                args.iters, args.warmup)

    if rank == 0:
        gemm_flops = 2 * pipe.total_m * args.n * args.hidden
        print(f"\npure GEMM   ({num_sms} SMs): {t_gemm_pure:8.3f} ms "
              f"({gemm_flops / t_gemm_pure / 1e9:.0f} TFLOPS)")
        for csms in args.comm_sms_list:
            print(f"pure dispatch ({csms:3d} SMs): {t_disp_pure[csms]:8.3f} ms")
        print(f"serial (dispatch then GEMM): {t_serial:8.3f} ms")
        print("\noverlapped, ms(vs-serial s, vs-ideal i) per oversubscription "
              "[ideal = max(pure GEMM, pure dispatch); s>1 good, i->1 good]")
        header = "".join(f"{'+os' + str(o) if o else 'base':>21}" for o in oversubs)
        print(f"{'comm_sms':>8}{header}")
        for csms in args.comm_sms_list:
            ideal = max(t_gemm_pure, t_disp_pure[csms])
            row = "".join(
                f"{results[(csms, o)]:>9.3f}({t_serial / results[(csms, o)]:4.2f}s,"
                f"{results[(csms, o)] / ideal:4.2f}i)"
                for o in oversubs)
            print(f"{csms:>8}{row}")
        best = min(results, key=results.get)
        print(f"\nbest: comm_sms={best[0]} oversub={best[1]} "
              f"{results[best]:.3f} ms = {t_serial / results[best]:.2f}x vs serial")

    pipe.free()


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
    parser.add_argument("--oversub-sms", type=str, default="8,16,24",
                        help="comma-separated GEMM SM oversubscription values to "
                             "sweep: gemm clusters = min(num_sms, num_sms + oversub "
                             "- comm_sms), so the GEMM backfills dispatch's SMs "
                             "once dispatch finishes")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    args.comm_sms_list = [int(x) for x in args.comm_sms.split(",")]
    args.oversub_sms_list = [int(x) for x in args.oversub_sms.split(",")]

    # Size the NVSHMEM symmetric heap from the actual buffer needs (must be
    # set BEFORE init — the whole heap is reserved up front).
    if "NVSHMEM_SYMMETRIC_SIZE" not in os.environ:
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(
            TilePipe.required_symmetric_bytes(args.tokens, args.hidden, args.topk))

    # NVSHMEM init is a common silent-hang point (UID broadcast + peer
    # bootstrap): log explicitly on both sides of it.
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
        run_tilepipe(args)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        torchrun_finalize()


if __name__ == "__main__":
    main()
