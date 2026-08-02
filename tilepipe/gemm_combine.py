"""
TilePipe GEMM -> combine overlap (see tilepipe/docs/guide.md): the grouped GEMM is
the PRODUCER — its epilogue publishes per-m-tile completion counters — and the
gated push-combine is the CONSUMER, streaming each expert-output row back to
its token's home rank as soon as the row's tile is produced. Two kernels, two
streams, disjoint SMs; scoped to one GEMM (in a real SwiGLU MoE layer this
producer is the down-projection).

Push only — the pull combine lost in both roles and was removed (rationale in
tilepipe/docs/status.md). Consequence worth knowing here: there is NO
cross-rank flag traffic, the GEMM publishes only to its own rank.

Push-kernel settings come from tilepipe/push_config.py, keyed on tokens/rank;
--push-config sweeps alternatives as paired variants of the same timing loop.

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
                               PushCombineKernel)

from quack.gemm import gemm as quack_gemm
from tilepipe import push_config
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

    # Resolve the push config for THIS token count: tuned table (keyed on
    # tokens/rank, stages clamped to the SMEM this N allows), then any explicit
    # flag overrides, then the extra configs to sweep as paired variants.
    tuned = push_config.pick(args.tokens, n)
    prim = (args.push_workers or tuned.workers,
            args.push_stages or tuned.stages,
            int(args.write_window) if args.write_window else tuned.wwin)
    args.cfg_list = [prim] + [c for c in args.cfg_extra if c != prim]
    for w, st, _ in args.cfg_list:
        push_config.validate(w, st, n)
    # Local, NOT written back to args: run() is called once per token size
    # in a sweep, and the tuned CTA count differs between them.
    ctas_list = args.combine_ctas_list or [tuned.ctas]

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

    # The GEMM is launched exactly as upstream apart from the SM cap and the
    # tile-flag publish. The producer stays untouched deliberately: tile
    # readiness is near-linear in GEMM progress and complete by ~60-70%, so
    # producer tile order is not what limits the overlap (max_swizzle_size was
    # tried and reverted -- details in status.md).
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
    # No arrival counter: PushCombineKernel is pure gated data movement -- no
    # segments, no releases. The local reduce is ordered against PEERS' pushes
    # by the pipeline's existing barrier, not by an in-band flag, so it runs
    # after that barrier, outside the overlapped region (~2% of the step).
    dplan = pplan.to_device(device)
    # `chunk` is INERT: swept at 128/64/32/16 rows and the tail did not move at
    # any token count, so it stays tied to tile_m. (The idea was that the tail
    # is one chunk-time; it is not -- see push_config.py for what does matter.)
    #
    # One kernel per (workers, stages, write_window) under test, compared as
    # variants of the SAME paired loop rather than across runs -- cross-run
    # drift is exactly what corrupts a bandwidth experiment. cfg_list[0] is the
    # primary: it drives correctness and every non-config measurement. The
    # serial baseline is NOT tied to it (see below).
    all_cfgs = list(dict.fromkeys(
        list(args.cfg_list) + [push_config.serial_config(n)]))
    push_kernels = {cfg: PushCombineKernel(
        cutlass.BFloat16, n, num_stages=cfg[1], workers=cfg[0],
        chunk=args.chunk or args.tile_m, write_window=cfg[2])
        for cfg in all_cfgs}
    w0 = args.cfg_list[0]
    push_args = lambda ctas, strm: dplan.push_combine_args(
        d_buf, stage_peer_ptrs, ctas, strm,
        gate_flags=tile_flags_local, gate_target=n_tiles)
    print(f"[rank {rank}] push-combine: {pplan.n_rows} rows, staging "
          f"{stage_rows * n * 2 / 1e9:.3f} GB")
    print(f"[rank {rank}] compiling push-combine...")
    comm_stream = torch.cuda.Stream(priority=-1)
    gemm_stream = torch.cuda.Stream()
    cs = lambda s: cuda_driver.CUstream(s.cuda_stream)
    cur = lambda: cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    # NOTE cute launches ignore torch.cuda.stream() contexts — the stream is
    # passed explicitly (a default-stream combine under comm_stream events
    # silently measured nothing).
    # PushCombineKernel has no Constexpr parameters, so one arg list serves
    # both cute.compile and the compiled callable.
    # The serial baseline's CTA count is pinned at NVLink saturation and is
    # normally NOT in the overlap's sweep list, so it needs compiling too.
    all_ctas = sorted(set(ctas_list) | {
        args.serial_ctas or push_config.SERIAL_CTAS or num_sms})
    push_compiled = {(cfg, c): cute.compile(push_kernels[cfg], *push_args(c, cur()))
                     for cfg in all_cfgs for c in all_ctas}
    print(f"[rank {rank}] push kernels compiled for "
          f"{[f'w{a}:s{b}:win{d}' for a, b, d in args.cfg_list]} "
          f"(primary w{w0[0]}:s{w0[1]}:win{w0[2]})")

    def launch_push(c, strm, w=None):
        push_compiled[(w if w is not None else w0, c)](*push_args(c, cs(strm)))

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

    def reset_flags_satisfied():
        # Reset, then pre-satisfy every gate so no poll ever blocks. Used ONLY
        # to time the overlapped launch with the stall removed -- the push
        # then reads rows the GEMM has not written yet, so the output is
        # garbage by construction. Timing only, never checked.
        reset()
        tile_flags_local.fill_(n_tiles)
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
    c0 = ctas_list[0]
    reset()
    launch_gemm(publish=True)
    launch_push(c0, comm_stream)
    barrier()          # peers' pushes complete here
    launch_reduce(comm_stream)
    barrier()
    if not check("serial GEMM->push-combine"):
        raise SystemExit("push-combine serial correctness FAILED")

    # Standalone-comm form: same kernel, gate PRE-SATISFIED so no poll blocks.
    # This is the "push alone" baseline, and it must produce the same answer as
    # the gated run -- otherwise the baseline is measuring a different kernel.
    # The producer is fenced with a full barrier here instead of by flags.
    reset()
    launch_gemm(publish=False)
    barrier()
    tile_flags_local.fill_(n_tiles)
    barrier()
    launch_push(c0, comm_stream)
    barrier()
    launch_reduce(comm_stream)
    barrier()
    if not check("serial GEMM->push-combine (gate pre-satisfied)"):
        raise SystemExit("pre-satisfied push-combine correctness FAILED")

    def overlapped_push(c_ctas, w=None):
        # Push kernel first on the high-priority stream: it gates on this
        # rank's own tile counters, so it streams rows out as the GEMM
        # produces them. The reduce follows after the pipeline's barrier.
        launch_push(c_ctas, comm_stream, w)
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
    # PAIRED timing. Every variant is launched once per iteration inside ONE
    # loop, so within an iteration they all see the same clock and thermal
    # state; ratios are then formed per-iteration and summarised. Timing each
    # variant in its own loop (the previous scheme) let the two halves of a
    # ratio drift minutes apart -- a `serial` measured early read 7.70 ms and
    # the identical configuration later read 8.26 ms, which alone moved the
    # 16384 speedup from 1.06x to 1.17x. Paired differences cancel that; no
    # amount of extra iterations does.
    Timing = namedtuple("Timing", "total comm gemm")

    def _time_once(enqueue, pre):
        (pre or barrier)()
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
        return start.elapsed_time(end_a), start.elapsed_time(end_b)

    def time_all(variants):
        """variants: name -> (enqueue, pre). Returns name -> Timing of arrays,
        each of shape [iters], already reduced across ranks per iteration (a
        step costs what the SLOWEST rank costs, so MAX is the step time)."""
        names = list(variants)
        arr = np.zeros((3, len(names), args.iters))
        for it in range(args.warmup + args.iters):
            for j, k in enumerate(names):
                enqueue, pre = variants[k]
                a, b = _time_once(enqueue, pre)
                if it >= args.warmup:
                    arr[0, j, it - args.warmup] = max(a, b)
                    arr[1, j, it - args.warmup] = a
                    arr[2, j, it - args.warmup] = b
        t = torch.from_numpy(arr).to(device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        arr = t.cpu().numpy()
        return {k: Timing(arr[0, j], arr[1, j], arr[2, j])
                for j, k in enumerate(names)}

    # Robust summaries. The 16th/84th percentiles are the +-1 sigma band for a
    # normal sample but survive the occasional outlier iteration, which matters
    # once we start benchmarking skewed expert distributions where the spread
    # IS the result rather than noise to be averaged away.
    med = lambda x: float(np.median(x))

    def band(x):
        return float(np.percentile(x, 16)), float(np.percentile(x, 84))

    def ratio(num, den):
        """Per-iteration ratio, then median + band. Formed elementwise because
        num[i] and den[i] were measured in the same iteration."""
        r = np.asarray(num) / np.asarray(den)
        return med(r), *band(r)

    def pm(x):
        """'median +-half-band' for printing a scalar with its spread."""
        lo, hi = band(x)
        return f"{med(x):.3f}+-{(hi - lo) / 2:.3f}"

    if rank == 0:
        print(f"\nbenchmark: paired timing, {args.iters} iters x "
              f"{3 + 5 * len(ctas_list)} variants per iteration...")

    def serial_push(c, w=None):
        # Reduce excluded: it needs a cross-rank barrier, so it is measured
        # separately and applies equally to serial and overlapped.
        launch_gemm(publish=True, stream=comm_stream)
        launch_push(c, comm_stream, w)

    ctas = ctas_list
    # Outside the overlap there is nothing to donate SMs to, so the serial
    # baseline's comm gets the FULL device. Only the overlapped path runs on a
    # reduced CTA count. Its config is likewise not forced to the overlap's:
    # TUNED optimises for the tail, serial is bandwidth-bound, so the baseline
    # picks the best of {what the overlap runs} U {the bandwidth optimum}.
    serial_c = args.serial_ctas or push_config.SERIAL_CTAS or num_sms
    serial_cfgs = list(dict.fromkeys(
        list(args.cfg_list) + [push_config.serial_config(n)]))

    def _serial_key(cfg):
        return f"serial_w{cfg[0]}s{cfg[1]}win{cfg[2]}"

    variants = {
        "gemm_plain": (lambda: launch_gemm(publish=False, stream=gemm_stream), None),
        "gemm_pub": (lambda: launch_gemm(publish=True, stream=gemm_stream), reset),
        "reduce": (lambda: launch_reduce(comm_stream), None),
    }
    for c in ctas:
        # STANDALONE COMM: same kernel, gate pre-satisfied, no GEMM running --
        # pure transfer cost. Sets what there is to hide.
        variants[f"push_alone_{c}"] = (
            (lambda c=c: launch_push(c, comm_stream)), reset_flags_satisfied)
        # The publishing GEMM restricted to the SMs the overlap leaves it: the
        # SM tax measured rather than modelled.
        variants[f"gemm_cap_{c}"] = (
            (lambda c=c: launch_gemm(publish=True, max_clusters=num_sms - c,
                                     stream=gemm_stream)), reset)
        variants[f"ovl_{c}"] = ((lambda c=c: overlapped_push(c)), reset)
        # Same launch, gate PRE-SATISFIED: co-resident but never blocking, so
        # the gap to `push_alone` is contention and the gap to `ovl` is spin.
        # Timing only -- it reads rows the GEMM has not written yet.
        variants[f"ovl_ready_{c}"] = ((lambda c=c: overlapped_push(c)),
                                      reset_flags_satisfied)
        # Every non-primary config, in the SAME loop so all comparisons are
        # paired. Serial is included for the reason above.
        for cfg in args.cfg_list[1:]:
            tag = f"w{cfg[0]}s{cfg[1]}win{cfg[2]}"
            variants[f"push_alone_{tag}_{c}"] = (
                (lambda c=c, cfg=cfg: launch_push(c, comm_stream, cfg)),
                reset_flags_satisfied)
            variants[f"ovl_{tag}_{c}"] = (
                (lambda c=c, cfg=cfg: overlapped_push(c, cfg)), reset)
    # SERIAL over its OWN (config, CTA) grid, not the overlap's. Serial runs
    # each phase on the whole GPU, so its comm wants far more CTAs than the
    # overlap (which is paying an SM tax to donate them). Searching only the
    # overlap's list handicaps the baseline -- with a single tuned CTA count it
    # has nothing to search at all, which read as 1.35x instead of ~1.01x at
    # 8192. Still inside the same paired loop, so serial/ovl stays per-iteration.
    for cfg in serial_cfgs:
        variants[_serial_key(cfg)] = (
            (lambda cfg=cfg: serial_push(serial_c, cfg)), reset)
    # Comm alone at the same CTA count: the sanity check that the baseline is
    # actually saturating NVLink rather than being quietly starved.
    variants["push_alone_serial"] = (
        (lambda: launch_push(serial_c, comm_stream)), reset_flags_satisfied)
    S = time_all(variants)

    t_gemm_plain, t_gemm_pub = S["gemm_plain"].total, S["gemm_pub"].total
    t_reduce = S["reduce"].total
    t_push_alone = {c: S[f"push_alone_{c}"].total for c in ctas}
    t_gemm_capped = {c: S[f"gemm_cap_{c}"].total for c in ctas}
    ovl = {c: S[f"ovl_{c}"] for c in ctas}
    ovl_ready = {c: S[f"ovl_ready_{c}"] for c in ctas}
    t_push_over = {c: ovl[c].total for c in ctas}
    cfg_serial = min(serial_cfgs, key=lambda cfg: med(S[_serial_key(cfg)].total))
    c_serial = serial_c
    t_push_serial = S[_serial_key(cfg_serial)].total
    t_push_serial_comm = S["push_alone_serial"].total
    # Speed of light: full-device GEMM against the comm it has to hide.
    # Deliberately NOT max(gemm@cap, ...) -- the absolute bound the approach is
    # aiming at, not one conditioned on the SM split in use. Elementwise so it
    # stays a per-iteration paired quantity.
    ideal = {c: np.maximum(t_gemm_pub, t_push_alone[c]) for c in ctas}

    if rank == 0:
        flops = 2 * total_m * n * k
        pct = lambda r: (f"{(r[0] - 1) * 100:+.1f}% "
                         f"[{(r[1] - 1) * 100:+.1f},{(r[2] - 1) * 100:+.1f}]")
        xx = lambda r: f"{r[0]:.2f}x [{r[1]:.2f},{r[2]:.2f}]"
        print(f"\n(median +- half the 16-84 pct band over {args.iters} PAIRED "
              f"iters; ratios formed per-iteration, so the band is the spread "
              f"of the ratio itself, not of the two sides separately)")
        print(f"pure GEMM ({num_sms} SMs): {pm(t_gemm_plain)} ms "
              f"({flops / med(t_gemm_plain) / 1e9:.0f} TFLOPS)")
        print(f"  with publish: {pm(t_gemm_pub)} ms  "
              f"({pct(ratio(t_gemm_pub, t_gemm_plain))})")
        print(f"serial (GEMM then push @{c_serial} CTAs, cfg "
              f"{cfg_serial[0]}:{cfg_serial[1]}:{cfg_serial[2]}): {pm(t_push_serial)} ms")
        # SANITY: the baseline is only fair if its comm phase saturates the
        # link. Rows going to this rank never leave the GPU, so only the
        # cross-rank figure is comparable to the NVLink roofline (~900 GB/s per
        # direction on NVLink5; the push plateaus around 790-800). The local
        # share is 1/W under uniform routing, so the total figure runs well
        # above the roofline at small world sizes -- read the second number.
        push_gb = pplan.n_rows * n * 2 / 1e9
        remote_gb = push_gb * float((pplan.dst_rank != rank).mean())
        secs = med(t_push_serial_comm) / 1e3
        print(f"  comm alone @{c_serial} CTAs: {pm(t_push_serial_comm)} ms = "
              f"{push_gb / secs:.0f} GB/s total, {remote_gb / secs:.0f} GB/s "
              f"cross-rank ({100 * remote_gb / push_gb:.0f}% remote, "
              f"{push_gb * 1e3:.0f} MB/rank; NVLink5 roofline ~900)")
        print(f"local reduce (excluded from both paths): {pm(t_reduce)} ms")
        print(f"\n{'ctas':>5} {'push alone':>14} {'gemm@cap':>14} "
              f"{'SM tax':>22} | {'ovl':>14} {'vs ideal(SoL)':>22} "
              f"{'vs serial':>22}")
        for c in ctas:
            print(f"{c:>5} {pm(t_push_alone[c]):>14} {pm(t_gemm_capped[c]):>14} "
                  f"{pct(ratio(t_gemm_capped[c], t_gemm_pub)):>22} | "
                  f"{pm(t_push_over[c]):>14} "
                  f"{xx(ratio(t_push_over[c], ideal[c])):>22} "
                  f"{xx(ratio(t_push_serial, t_push_over[c])):>22}")
        # Where the overlap's excess goes. `ready` is the same concurrent
        # launch with the gate pre-satisfied, so ready-vs-alone is contention
        # and gated-vs-ready is spin waiting.
        print(f"\n{'ctas':>5} | {'gemm in-ovl':>14} {'contention':>20}"
              f" | {'push alone':>14} {'+GEMM':>14} {'+spin':>14}")
        for c in ctas:
            print(f"{c:>5} | {pm(ovl[c].gemm):>14} "
                  f"{pct(ratio(ovl[c].gemm, t_gemm_capped[c])):>20} | "
                  f"{pm(t_push_alone[c]):>14} {pm(ovl_ready[c].comm):>14} "
                  f"{pm(ovl[c].comm):>14}")
        if len(args.cfg_list) > 1:
            # Paired WRITE_WINDOW comparison. `tail` is the diagnostic: if it
            # shrinks with the window the tail is concurrency-bound and short
            # sequences are tunable; if it stays pinned it is a fixed latency
            # and no amount of tuning will fix them.
            gb = total_m * n * 2 / 1e9
            print(f"\n{'ctas':>5} {'w:s:win':>10} {'smem':>7} {'push alone':>14} "
                  f"{'GB/s':>7} {'vs primary':>18} | {'ovl':>14} {'tail':>9} "
                  f"{'vs serial':>20}")
            for c in ctas:
                for cfg in args.cfg_list:
                    tag = f"w{cfg[0]}s{cfg[1]}win{cfg[2]}"
                    pa = (t_push_alone[c] if cfg == w0
                          else S[f"push_alone_{tag}_{c}"].total)
                    ot = (t_push_over[c] if cfg == w0
                          else S[f"ovl_{tag}_{c}"].total)
                    print(f"{c:>5} {f'{cfg[0]}:{cfg[1]}:{cfg[2]}':>10} "
                          f"{cfg[1] * n * 2 / 1024:>6.0f}K {pm(pa):>14} "
                          f"{gb / (med(pa) / 1e3):>7.0f} "
                          f"{pct(ratio(pa, t_push_alone[c])):>18} | "
                          f"{pm(ot):>14} "
                          f"{med(ot) - med(t_gemm_capped[c]):>7.3f}ms "
                          f"{xx(ratio(t_push_serial, ot)):>20}")

    free_symmetric()
    stat = lambda x: dict(med=med(x), lo=band(x)[0], hi=band(x)[1])
    rstat = lambda a, b: dict(zip(("med", "lo", "hi"), ratio(a, b)))
    return dict(
        tokens=args.tokens, total_m=total_m, K=k, N=n, topk=args.topk,
        experts=args.experts, world=world_size, num_sms=num_sms,
        iters=args.iters, serial_ctas=c_serial, serial_cfg=list(cfg_serial),
        gemm=stat(t_gemm_plain), gemm_publish=stat(t_gemm_pub),
        gemm_tflops=2 * total_m * n * k / med(t_gemm_plain) / 1e9,
        publish_overhead=rstat(t_gemm_pub, t_gemm_plain),
        serial_push=stat(t_push_serial), reduce=stat(t_reduce),
        serial_push_comm=stat(t_push_serial_comm),
        push_alone={c: stat(t_push_alone[c]) for c in ctas},
        gemm_capped={c: stat(t_gemm_capped[c]) for c in ctas},
        sm_tax={c: rstat(t_gemm_capped[c], t_gemm_pub) for c in ctas},
        push_overlapped={c: stat(t_push_over[c]) for c in ctas},
        vs_serial={c: rstat(t_push_serial, t_push_over[c]) for c in ctas},
        vs_ideal={c: rstat(t_push_over[c], ideal[c]) for c in ctas},
        ovl_gemm={c: stat(ovl[c].gemm) for c in ctas},
        ovl_push={c: stat(ovl[c].comm) for c in ctas},
        ovl_push_ready={c: stat(ovl_ready[c].comm) for c in ctas},
        # Raw per-iteration samples: paired across variants, so any further
        # comparison (e.g. skewed vs uniform routing) can be done properly
        # rather than from medians.
        samples={k: dict(total=list(v.total), comm=list(v.comm),
                         gemm=list(v.gemm)) for k, v in S.items()},
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

    ms = lambda d: f"{d['med']:.3f}±{(d['hi'] - d['lo']) / 2:.3f}"
    pc = lambda d: (f"{(d['med'] - 1) * 100:+.1f}% "
                    f"[{(d['lo'] - 1) * 100:+.1f},{(d['hi'] - 1) * 100:+.1f}]")
    xx = lambda d: f"{d['med']:.2f}x [{d['lo']:.2f},{d['hi']:.2f}]"

    lines = [f"# TilePipe GEMM -> push-combine ({world_size} GPUs)", "",
             f"- generated: {stamp}",
             f"- K={args.hidden} N={args.n} topk={args.topk} "
             f"experts={args.experts} tile={args.tile_m}x{args.tile_n}",
             f"- comm CTAs: {args.combine_ctas_list or 'tuned per token count'}",
             "- paired timing: every variant launched once per iteration, so "
             "ratios are formed per-iteration. Values are median ± half the "
             "16-84 percentile band; bracketed ranges are that band.", ""]
    lines += ["## Summary (best overlapped vs serial)", "",
              "| tokens/rank | GEMM ms | +publish | push alone (best) | "
              "serial push | best ovl | vs serial | vs ideal |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        best_c = min(r["push_overlapped"], key=lambda c: r["push_overlapped"][c]["med"])
        bc = min(r["push_alone"].values(), key=lambda d: d["med"])
        lines.append(
            f"| {r['tokens']} | {ms(r['gemm'])} | {pc(r['publish_overhead'])} "
            f"| {ms(bc)} | {ms(r['serial_push'])} "
            f"| {ms(r['push_overlapped'][best_c])} (@{best_c}) "
            f"| **{xx(r['vs_serial'][best_c])}** | {xx(r['vs_ideal'][best_c])} |")
    for r in results:
        lines += ["", f"## tokens/rank = {r['tokens']} (total_m={r['total_m']})", "",
                  f"pure GEMM {ms(r['gemm'])} ms ({r['gemm_tflops']:.0f} TFLOPS); "
                  f"with publish {ms(r['gemm_publish'])} ms "
                  f"({pc(r['publish_overhead'])})",
                  f"serial @{r['serial_ctas']} CTAs {ms(r['serial_push'])} ms; "
                  f"local reduce {ms(r['reduce'])} ms (excluded from both)", "",
                  "ideal = speed of light = max(full-device publishing GEMM, "
                  "push alone). The SM tax is reported separately rather than "
                  "folded into the bound.", "",
                  "| CTAs | push alone | GEMM @cap | SM tax | overlapped | "
                  "vs ideal | vs serial |", "|---|---|---|---|---|---|---|"]
        for c in sorted(r["push_alone"], key=int):
            lines.append(
                f"| {c} | {ms(r['push_alone'][c])} | {ms(r['gemm_capped'][c])} "
                f"| {pc(r['sm_tax'][c])} | {ms(r['push_overlapped'][c])} "
                f"| {xx(r['vs_ideal'][c])} | {xx(r['vs_serial'][c])} |")
        lines += ["", "Per-stream spans inside the overlapped run. `+GEMM` is "
                  "the push with the gate pre-satisfied but the GEMM running "
                  "(contention); `+spin` adds the live gate (spin waiting).", "",
                  "| CTAs | GEMM in-ovl | contention | push alone | +GEMM | "
                  "+spin |", "|---|---|---|---|---|---|"]
        for c in sorted(r["push_alone"], key=int):
            g, cap = r["ovl_gemm"][c], r["gemm_capped"][c]["med"]
            cont = {kk: g[kk] / cap for kk in ("med", "lo", "hi")}
            lines.append(
                f"| {c} | {ms(g)} | {pc(cont)} | {ms(r['push_alone'][c])} "
                f"| {ms(r['ovl_push_ready'][c])} | {ms(r['ovl_push'][c])} |")
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
    parser.add_argument("--combine-ctas", type=str, default=None,
                        help="comma-separated push-combine CTA (= comm SM) counts")
    # Push-kernel tunables. Defaults come from tilepipe/push_config.py (keyed
    # on tokens/rank, clamped to the SMEM the row length allows); these flags
    # override it, and --push-config adds extra configs to sweep as paired
    # variants. See push_config.py for what each parameter controls.
    parser.add_argument("--chunk", type=int, default=None,
                        help="rows per round-robin work unit (default tile_m). "
                             "INERT: swept 128/64/32/16, no effect on the tail.")
    parser.add_argument("--serial-ctas", type=int, default=None,
                        help="comm CTAs for the SERIAL baseline (default "
                             "push_config.SERIAL_CTAS). Serial has the whole "
                             "GPU, so this is pinned at NVLink saturation and "
                             "is deliberately NOT the overlap's CTA count.")
    parser.add_argument("--push-config", type=str, default=None,
                        help="extra push configs to sweep as paired variants, "
                             "'workers:stages[:write_window]' comma-separated "
                             "(e.g. 4:24,2:24). In-flight rows per CTA == "
                             "stages; SMEM == stages * N * 2 B.")
    parser.add_argument("--write-window", type=str, default=None,
                        help="in-flight remote writes per worker. INERT: swept "
                             "8/16/32 at both stages//workers = 6 and 12, "
                             "spread <=1.1%% and inside the bands.")
    parser.add_argument("--push-stages", type=int, default=None,
                        help="SMEM pipeline depth == in-flight rows per CTA "
                             "(overrides the tuned table)")
    parser.add_argument("--push-workers", type=int, default=None,
                        help="producer/consumer warp pairs per push CTA "
                             "(overrides the tuned table)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    args.combine_ctas_list = ([int(x) for x in args.combine_ctas.split(",")]
                              if args.combine_ctas else None)
    # Resolved per token count in run(), since the tuned config is keyed on
    # tokens/rank; here we only collect the overrides and the sweep extras.
    args.cfg_extra = push_config.parse_specs(args.push_config)
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
