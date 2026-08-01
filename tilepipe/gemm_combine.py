"""
TilePipe GEMM -> combine overlap (see tilepipe/docs/guide.md): the grouped GEMM is
the PRODUCER — its epilogue publishes per-m-tile completion counters — and the
gated push-combine is the CONSUMER, streaming each expert-output row back to
its token's home rank as soon as the row's tile is produced. Two kernels, two
streams, disjoint SMs; scoped to one GEMM (in a real SwiGLU MoE layer this
producer is the down-projection).

PUSH ONLY. The pull combine (gated CombineTmaKernel, each rank fetching its
own tokens' topk rows from peers) lost in both roles — serial 8.53 vs 6.81 ms
and best-overlapped 0.90x vs 1.04x at 16K tokens — because readiness is
per-token (all topk, so back-loaded) rather than per-expert, and the gate is a
cross-rank busy-wait. It was removed from this pipeline; the kernel itself
lives on in tilepipe/moe_comm.py as the standalone/serial combine. With pull
gone this benchmark has NO cross-rank flag traffic: the GEMM publishes only to
its own rank.

Inputs are synthetic pre-dispatched activations (A already sits in each
rank's expert-grouped buffer — dispatch overlap is the other, already-built
pipeline). Deterministic per-rank seeds let every rank rebuild the peers'
GEMM outputs for the reference combine.

Run (2 GPUs):
    torchrun --nproc-per-node 2 tilepipe/gemm_combine.py
"""

import argparse
import datetime
import functools
import json
import os
from collections import namedtuple

import numpy as np
import torch
import torch.distributed as dist


import cutlass
import cutlass.cute as cute

import cuda.bindings.driver as cuda_driver

import nvshmem.core

from tilepipe.moe_comm import (torchrun_uid_init_bcast, torchrun_finalize,
                               make_varlen_all_to_all)

from quack.gemm import gemm as quack_gemm
from tilepipe.args import TilePipeArgs
from tilepipe.plan import (build_combine_metadata, plan_combine,
                           peer_ptr_tensor)

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
          f"local m-tiles={len(meta['tile_lo'])}")

    # --- Local GEMM operands (deterministic per rank for references) ---
    weights_bytes = epr * n * k * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] allocating weights {weights_bytes / 1e9:.2f} GB + peer "
          f"references (free: {free_bytes / 1e9:.2f} GB)")
    A = rank_tensor(500 + rank, (total_m, k), device, scale=k ** -0.5)
    weights = build_weights(rank, epr, n, k, device)

    # --- Symmetric buffer: expert output D (rows are pushed FROM here) ---
    d_buf = nvshmem.core.tensor((max_rows, n), dtype=torch.bfloat16)
    d_buf.fill_(0)
    print(f"[rank {rank}] symmetric D allocated ({max_rows * n * 2 / 1e9:.3f} GB)")

    D = d_buf[:total_m]
    cu_seqlens_m = torch.from_numpy(meta["cu_seqlens"]).to(device)
    tile_offsets_t = torch.from_numpy(meta["tile_offsets"]).to(device)
    scatter_t = torch.from_numpy(meta["scatter"]).to(device)
    combine_out = torch.zeros((args.tokens, n), dtype=torch.bfloat16, device=device)

    # Tile flags are rank-LOCAL: the GEMM publishes one counter per local
    # m-tile (per-expert tiling, `tile_offsets`) and the push kernel gates on
    # this rank's own counters -- no cross-rank flag traffic. Sized by the
    # PRODUCER's tile count (len(tile_lo)), not by the consumer's max gate
    # index: a trailing tile with no rows to push would otherwise leave the
    # epilogue publishing one past the end.
    local_tiles = len(meta["tile_lo"])
    tile_flags_local = torch.zeros(local_tiles, dtype=torch.int32, device=device)
    flag_pub_local = torch.tensor([tile_flags_local.data_ptr()],
                                  device=device, dtype=torch.int64)

    def launch_gemm(publish, max_clusters=None, stream=None):
        ptrs = flag_pub_local if publish else None
        with torch.cuda.stream(stream if stream is not None else
                               torch.cuda.current_stream()):
            quack_gemm(
                A, weights, D, C=None, tile_count_semaphore=None,
                tile_M=args.tile_m, tile_N=args.tile_n, cluster_M=1, cluster_N=1,
                persistent=True, cu_seqlens_m=cu_seqlens_m,
                max_active_clusters=max_clusters,
                tilepipe=TilePipeArgs(
                    tile_flag_ptrs=ptrs,
                    tile_flag_offsets=tile_offsets_t if publish else None))

    # --- Push-combine (reverse dispatch): this rank PUSHES each of its D rows
    # back to the token's home rank as soon as the row's tile is produced, and
    # the home rank reduces staging[token, topk, N] locally. Row readiness is
    # per-EXPERT (uniform across the GEMM window), so rows become pushable
    # early and evenly across the GEMM window.
    pplan = plan_combine(all_topk, args.experts, rank, world_size,
                         tile_m=args.tile_m)
    p_gate = pplan.gate_idx
    # Same index space the GEMM epilogue publishes in (plan.py derives both
    # from the per-expert m-tile offsets); assert rather than assume.
    assert len(p_gate) == 0 or int(p_gate.max()) < local_tiles, (
        f"gate index {int(p_gate.max())} outside producer tile range {local_tiles}")
    stage_rows = pplan.dst_rows
    staging = nvshmem.core.tensor((stage_rows, n), dtype=torch.bfloat16)
    staging.fill_(0)
    stage_peer_ptrs = peer_ptr_tensor(staging, world_size, device)
    # No arrival counter: the push kernel is pure data movement (flag_peer_ptrs
    # = None => no segment bookkeeping, no releases). The local reduce is
    # ordered against PEERS' pushes by the pipeline's existing barrier, not by
    # an in-band flag — so it runs after that barrier, outside the overlapped
    # region (it is ~2% of the step).
    dplan = pplan.to_device(device)
    push_kernel = make_varlen_all_to_all(
        cutlass.BFloat16, n, impl=args.comm_impl, num_warps=args.comm_warps,
        num_stages=args.push_stages, workers=args.push_workers)
    push_args = lambda ctas, strm: dplan.args(
        d_buf, stage_peer_ptrs, None, ctas, strm,
        gate_flags=tile_flags_local, gate_target=n_tiles)
    print(f"[rank {rank}] push-combine: {pplan.n_rows} rows, staging "
          f"{stage_rows * n * 2 / 1e9:.3f} GB")
    print(f"[rank {rank}] compiling push-combine (gated + ungated)...")
    comm_stream = torch.cuda.Stream(priority=-1)
    gemm_stream = torch.cuda.Stream()
    cs = lambda s: cuda_driver.CUstream(s.cuda_stream)
    cur = lambda: cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    # NOTE cute launches ignore torch.cuda.stream() contexts — the stream is
    # passed explicitly (a default-stream combine under comm_stream events
    # silently measured nothing).
    push_gated = {c: cute.compile(push_kernel, *dplan.compile_args(
        d_buf, stage_peer_ptrs, None, c, rank, world_size, cur(),
        gate_flags=tile_flags_local, gate_target=n_tiles))
        for c in args.combine_ctas_list}
    # Ungated twin: the comm cost with no producer coupling at all. This is
    # the standalone combine number -- what there is to hide, and the comm
    # half of the speed-of-light bound.
    push_ungated = {c: cute.compile(push_kernel, *dplan.compile_args(
        d_buf, stage_peer_ptrs, None, c, rank, world_size, cur()))
        for c in args.combine_ctas_list}
    print(f"[rank {rank}] push kernels compiled")

    def launch_push(c, strm):
        push_gated[c](*push_args(c, cs(strm)))

    def launch_push_ungated(c, strm):
        push_ungated[c](*dplan.args(d_buf, stage_peer_ptrs, None, c, cs(strm)))

    def launch_reduce(strm):
        # Plain local contraction. CALLER MUST have barriered after the pushes:
        # stream order covers only this rank's push kernel, not the peers'.
        with torch.cuda.stream(strm):
            torch.sum(staging.view(args.tokens, args.topk, n), dim=1,
                      out=combine_out)

    def reset():
        tile_flags_local.zero_()
        dplan.reset()
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
    # One publish per (m-tile, n-tile): every local counter lands on n_tiles.
    # A plain GEMM must add nothing, which the second launch above checks.
    exp_flags = torch.full((local_tiles,), n_tiles, dtype=torch.int32)
    ok_flags = torch.equal(tile_flags_local.cpu(), exp_flags)
    print(f"[rank {rank}] GEMM tile flags {'OK' if ok_flags else 'FAIL'}")
    if not ok_flags:
        raise SystemExit(f"[rank {rank}] tile-flag publish FAILED: "
                         f"{tile_flags_local.cpu().tolist()[:16]}...")

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

    # --- Serial correctness. Also the warm-up EXECUTION of every push kernel
    # (lazy module load device-syncs, which deadlocks against a spinning gate,
    # so nothing may reach the overlapped region uninstantiated) ---
    c0 = args.combine_ctas_list[0]
    reset()
    launch_gemm(publish=True)
    launch_push(c0, comm_stream)
    barrier()          # peers' pushes complete here
    launch_reduce(comm_stream)
    barrier()
    if not check("serial GEMM->push-combine"):
        raise SystemExit("push-combine serial correctness FAILED")

    # Ungated twin, same expected result: it is the standalone comm number, so
    # it must be shown to be a like-for-like substitute for the gated kernel
    # and not merely a faster one. Nothing orders it against the GEMM -- that
    # is the whole point of the ungated form -- so the producer is fenced with
    # a full barrier rather than a flag.
    reset()
    launch_gemm(publish=False)
    barrier()
    launch_push_ungated(c0, comm_stream)
    barrier()
    launch_reduce(comm_stream)
    barrier()
    if not check("serial GEMM->push-combine (ungated)"):
        raise SystemExit("ungated push-combine correctness FAILED")

    def overlapped_push(c_ctas):
        # Push kernel first on the high-priority stream: it gates on this
        # rank's own tile counters, so it streams rows out as the GEMM
        # produces them. The reduce follows after the pipeline's barrier.
        launch_push(c_ctas, comm_stream)
        launch_gemm(publish=True, max_clusters=num_sms - c_ctas,
                    stream=gemm_stream)

    reset()
    overlapped_push(c0)
    barrier()          # peers' pushes complete here
    launch_reduce(comm_stream)
    barrier()
    if not check("overlapped GEMM->push-combine"):
        raise SystemExit("push-combine overlapped correctness FAILED")

    def free_symmetric():
        nvshmem.core.free_tensor(d_buf)
        nvshmem.core.free_tensor(staging)

    if not args.benchmark:
        free_symmetric()
        return None

    # --- Benchmark ---
    # Per-STREAM durations, not just the wall time. In the overlapped run the
    # two streams carry one kernel each, so `comm` is the push's own span and
    # `gemm` the GEMM's own span, both measured from the same start event.
    # Comparing them against the standalone numbers separates "the GEMM got
    # slower from contention" from "the comm ran past the GEMM" -- the wall
    # time alone cannot tell those apart.
    Timing = namedtuple("Timing", "total comm gemm")

    def time_iters(enqueue, pre=None):
        # `pre` runs BEFORE the start event, so per-iteration setup (counter
        # resets, pre-satisfying a gate) never lands inside the timed window.
        tot, t_c, t_g = [], [], []
        for it in range(args.warmup + args.iters):
            if pre is not None:
                pre()
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
                a, b = start.elapsed_time(end_a), start.elapsed_time(end_b)
                tot.append(max(a, b))
                t_c.append(a)
                t_g.append(b)
        med = lambda xs: float(np.median(xs))
        t = torch.tensor([med(tot), med(t_c), med(t_g)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return Timing(*t.tolist())

    if rank == 0:
        print("\nbenchmark: pure GEMM / pure push / capped GEMM / serial / overlapped...")

    # --- Each component alone, then the two compositions. ---
    t_gemm_plain = time_iters(
        lambda: launch_gemm(publish=False, stream=gemm_stream)).total
    t_gemm_pub = time_iters(
        lambda: launch_gemm(publish=True, stream=gemm_stream), pre=reset).total
    # STANDALONE COMM. The push kernel compiled with no gate at all, so this
    # is pure transfer cost with no producer coupling -- the single most
    # important number here: it sets what there is to hide, and it is the
    # comm half of the speed-of-light bound below.
    t_push_alone = {c: time_iters(
        lambda: launch_push_ungated(c, comm_stream), pre=reset).total
        for c in args.combine_ctas_list}
    # The publishing GEMM restricted to the SMs the overlap leaves it. NOT
    # used for the ideal (that stays speed-of-light, see below) -- this is the
    # SM tax on its own, measured rather than modelled, and it is what tells
    # us whether max_active_clusters really frees the SMs we assume.
    t_gemm_capped = {c: time_iters(
        lambda: launch_gemm(publish=True, max_clusters=num_sms - c,
                            stream=gemm_stream), pre=reset).total
        for c in args.combine_ctas_list}

    def serial_push(c):
        # Reduce excluded: it needs a cross-rank barrier, so it is measured
        # separately (t_reduce) and applies equally to serial and overlapped.
        launch_gemm(publish=True, stream=comm_stream)
        launch_push(c, comm_stream)

    # Fair serial baseline: each phase gets the whole GPU, so the push runs
    # with the LARGEST CTA count in the sweep (a small-CTA serial push would
    # pad the baseline and flatter the overlap numbers).
    c_serial = max(args.combine_ctas_list)
    t_push_serial = time_iters(lambda: serial_push(c_serial), pre=reset).total
    t_reduce = time_iters(lambda: launch_reduce(comm_stream)).total
    # Keep the full per-stream split here: `ovl[c].gemm` is the GEMM's own
    # span inside the overlap (vs t_gemm_capped -> contention) and
    # `ovl[c].comm` is the push's own span (vs t_push_alone -> gate stall +
    # drain). This is the only run where the two kernels are concurrent.
    ovl = {c: time_iters(lambda: overlapped_push(c), pre=reset)
           for c in args.combine_ctas_list}
    t_push_over = {c: ovl[c].total for c in args.combine_ctas_list}

    # Speed of light: the full-device GEMM against the comm it has to hide.
    # Deliberately NOT max(t_gemm_capped[c], ...) -- we want the absolute
    # bound the whole approach is aiming at, not the bound conditioned on the
    # SM split we happen to be using. The SM tax shows up separately, as the
    # gap between `gemm@cap` and the full-device GEMM.
    ideal = {c: max(t_gemm_pub, t_push_alone[c]) for c in args.combine_ctas_list}

    if rank == 0:
        flops = 2 * total_m * n * k
        print(f"\npure GEMM   ({num_sms} SMs): {t_gemm_plain:8.3f} ms "
              f"({flops / t_gemm_plain / 1e9:.0f} TFLOPS); with publish: "
              f"{t_gemm_pub:8.3f} ms ({(t_gemm_pub / t_gemm_plain - 1) * 100:+.1f}%)")
        print(f"serial (GEMM then push @{c_serial} CTAs): {t_push_serial:8.3f} ms")
        print(f"local reduce (after barrier, both paths): {t_reduce:8.3f} ms")
        print(f"\n{'ctas':>5} {'push alone':>11} {'gemm@cap':>10} {'SMtax%':>7}"
              f" | {'ideal(SoL)':>11} {'ovl':>9} {'vs ideal':>9} {'vs serial':>10}")
        for c in args.combine_ctas_list:
            print(f"{c:>5} {t_push_alone[c]:>9.3f}ms {t_gemm_capped[c]:>8.3f}ms "
                  f"{(t_gemm_capped[c] / t_gemm_pub - 1) * 100:>6.1f}% | "
                  f"{ideal[c]:>9.3f}ms {t_push_over[c]:>7.3f}ms "
                  f"{t_push_over[c] / ideal[c]:>8.2f}x "
                  f"{t_push_serial / t_push_over[c]:>9.2f}x")
        # Where the overlap's excess actually goes. Both columns are measured
        # INSIDE the same overlapped run, against the standalone baselines.
        print(f"\n{'ctas':>5} | {'gemm in-ovl':>12} {'vs @cap':>8} (contention)"
              f" | {'push in-ovl':>12} {'vs alone':>9} (stall+drain)")
        for c in args.combine_ctas_list:
            g, p = ovl[c].gemm, ovl[c].comm
            print(f"{c:>5} | {g:>10.3f}ms {g - t_gemm_capped[c]:>+7.3f}ms "
                  f"{(g / t_gemm_capped[c] - 1) * 100:>+11.1f}% | "
                  f"{p:>10.3f}ms {p - t_push_alone[c]:>+8.3f}ms "
                  f"{(p / t_push_alone[c] - 1) * 100:>+12.1f}%")

    free_symmetric()
    return dict(
        tokens=args.tokens, total_m=total_m, K=k, N=n, topk=args.topk,
        experts=args.experts, world=world_size, num_sms=num_sms,
        gemm_ms=t_gemm_plain, gemm_publish_ms=t_gemm_pub,
        gemm_tflops=2 * total_m * n * k / t_gemm_plain / 1e9,
        publish_overhead_pct=(t_gemm_pub / t_gemm_plain - 1) * 100,
        serial_push_ms=t_push_serial, reduce_ms=t_reduce, serial_ctas=c_serial,
        push_alone_ms=t_push_alone, gemm_capped_ms=t_gemm_capped, ideal_ms=ideal,
        push_overlapped_ms={c: t_push_over[c] for c in args.combine_ctas_list},
        # Per-stream spans measured inside the overlapped run.
        ovl_gemm_ms={c: ovl[c].gemm for c in args.combine_ctas_list},
        ovl_push_ms={c: ovl[c].comm for c in args.combine_ctas_list},
    )


def write_results(results, args, world_size):
    """Rank 0 writes the sweep to a timestamped run directory: results.json
    (machine-readable) + results.md (the tables as printed)."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"gemm_combine_{world_size}gpu_N{args.n}_K{args.hidden}_"
            f"topk{args.topk}_{stamp}")
    outdir = os.path.join(args.results_dir, name)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args) | {"world_size": world_size},
                   "runs": [{k: (v if not isinstance(v, dict)
                                 else {str(kk): vv for kk, vv in v.items()})
                             for k, v in r.items()} for r in results]},
                  f, indent=2, default=str)

    lines = [f"# TilePipe GEMM -> push-combine ({world_size} GPUs)", "",
             f"- generated: {stamp}",
             f"- K={args.hidden} N={args.n} topk={args.topk} "
             f"experts={args.experts} tile={args.tile_m}x{args.tile_n}",
             f"- comm CTAs swept: {args.combine_ctas_list}", ""]
    lines += ["## Summary (best overlapped vs serial)", "",
              "| tokens/rank | GEMM ms | +publish | push alone (best) | "
              "serial push | best ovl | vs serial | vs ideal |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        bc = min(r["push_alone_ms"].values())
        best_c = min(r["push_overlapped_ms"], key=r["push_overlapped_ms"].get)
        bs = r["push_overlapped_ms"][best_c]
        lines.append(
            f"| {r['tokens']} | {r['gemm_ms']:.3f} | "
            f"{r['publish_overhead_pct']:+.1f}% | {bc:.3f} | "
            f"{r['serial_push_ms']:.3f} | {bs:.3f} (@{best_c}) | "
            f"{r['serial_push_ms'] / bs:.2f}x | "
            f"{bs / r['ideal_ms'][best_c]:.2f}x |")
    for r in results:
        lines += ["", f"## tokens/rank = {r['tokens']} (total_m={r['total_m']})", "",
                  f"pure GEMM {r['gemm_ms']:.3f} ms ({r['gemm_tflops']:.0f} TFLOPS); "
                  f"with publish {r['gemm_publish_ms']:.3f} ms "
                  f"({r['publish_overhead_pct']:+.1f}%)",
                  f"serial @{r['serial_ctas']} CTAs {r['serial_push_ms']:.3f} ms; "
                  f"local reduce {r['reduce_ms']:.3f} ms (excluded from both)", "",
                  "ideal = speed of light = max(full-device publishing GEMM, "
                  "push alone). The SM tax is reported separately rather than "
                  "folded into the bound.", "",
                  "| CTAs | push alone | GEMM @cap | SM tax % | ideal (SoL) | "
                  "overlapped | vs ideal | vs serial |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in sorted(r["push_alone_ms"]):
            lines.append(
                f"| {c} | {r['push_alone_ms'][c]:.3f} "
                f"| {r['gemm_capped_ms'][c]:.3f} "
                f"| {(r['gemm_capped_ms'][c] / r['gemm_publish_ms'] - 1) * 100:+.1f}% "
                f"| {r['ideal_ms'][c]:.3f} | {r['push_overlapped_ms'][c]:.3f} "
                f"| {r['push_overlapped_ms'][c] / r['ideal_ms'][c]:.2f}x "
                f"| {r['serial_push_ms'] / r['push_overlapped_ms'][c]:.2f}x |")
        lines += ["", "Per-stream spans measured inside the overlapped run: "
                  "GEMM vs its standalone capped time isolates contention; push "
                  "vs its standalone time isolates gate stall + drain.", "",
                  "| CTAs | GEMM in-ovl | vs @cap | push in-ovl | vs alone |",
                  "|---|---|---|---|---|"]
        for c in sorted(r["push_alone_ms"]):
            g, p = r["ovl_gemm_ms"][c], r["ovl_push_ms"][c]
            lines.append(
                f"| {c} | {g:.3f} | {(g / r['gemm_capped_ms'][c] - 1) * 100:+.1f}% "
                f"| {p:.3f} | {(p / r['push_alone_ms'][c] - 1) * 100:+.1f}% |")
    with open(os.path.join(outdir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults written to {outdir}/ (results.json, results.md)")


def main():
    parser = argparse.ArgumentParser(description="TilePipe GEMM -> gated combine")
    parser.add_argument("--tokens", type=int, default=None,
                        help="single token count (overrides --token-sweep)")
    parser.add_argument("--token-sweep", type=str, default="2048,4096,8192,16384,32768",
                        help="comma-separated tokens/rank to sweep")
    parser.add_argument("--results-dir", type=str, default="bench_results",
                        help="parent dir for the timestamped run directory")
    parser.add_argument("--hidden", type=int, default=7168, help="GEMM K")
    parser.add_argument("--gemm-n", dest="n", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=None,
                        help="default: 32 * world_size")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256,
                        help="fastest cluster-1x1 config is 128x256; it also "
                             "halves the tile-flag publishes per row")
    parser.add_argument("--combine-ctas", type=str, default="12,24,36,48",
                        help="comma-separated push-combine CTA (= comm SM) counts")
    parser.add_argument("--comm-impl", choices=["simt", "tma"], default="tma",
                        help="push-combine backend. Default tma: measured "
                             "1.2x (N=4096) to 1.7x (N=7168) faster than simt "
                             "at 8-16 CTAs, which is the regime the SM tax "
                             "forces the overlap into. They converge on the "
                             "NVLink roofline above ~24 CTAs, where simt's "
                             "36-CTA point is actually the better one.")
    parser.add_argument("--comm-warps", type=int, default=16,
                        help="[simt] warps per push-combine CTA")
    parser.add_argument("--push-stages", type=int, default=12,
                        help="SMEM stage budget for the push-combine kernel")
    parser.add_argument("--push-workers", type=int, default=4,
                        help="producer/consumer warp pairs per push CTA")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    args.combine_ctas_list = [int(x) for x in args.combine_ctas.split(",")]
    token_list = ([args.tokens] if args.tokens is not None
                  else [int(x) for x in args.token_sweep.split(",")])
    args.tokens = max(token_list)  # heap must cover the largest run

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
        results = []
        for tok in token_list:
            args.tokens = tok
            if rank == 0:
                print(f"\n{'=' * 70}\n=== tokens/rank = {tok} ===\n{'=' * 70}")
            r = run(args)
            if r is not None:
                results.append(r)
            torch.cuda.synchronize()
            dist.barrier()
        if rank == 0 and results:
            write_results(results, args, dist.get_world_size())
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        torchrun_finalize()


if __name__ == "__main__":
    main()
