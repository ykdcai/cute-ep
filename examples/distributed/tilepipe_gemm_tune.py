"""
Single-GPU tuner for the TilePipe producer GEMM.

Two questions, one sweep:

1. **What is the fastest config for the grouped (varlen_m) GEMM we actually
   run?** The overlap drivers hardcode `tile_M x tile_N, cluster 1x1`, while
   `quack.gemm_interface.gemm_tuned` autotunes only the plain GEMM path and
   cannot take TilePipe's extra arguments (`max_active_clusters`,
   `tile_flag_ptrs`). Every millisecond shaved off the GEMM is a millisecond
   the comm kernel no longer has to hide behind.
2. **What does the tile-flag publish cost?** Same config, publish on vs off,
   back to back, so the delta is not confounded by the config choice.

Cluster M=2 turns on the 2-CTA MMA (`use_2cta_instrs`), which halves the CTA
tile M: the flag index space is per *CTA* tile, so the host must size the flag
array with `cta_tile_m()`, not `tile_M`. Getting that wrong is silent — the
counters just never reach their target.

Run (one GPU, no NVSHMEM/torchrun needed):
    CUDA_VISIBLE_DEVICES=0 python examples/distributed/tilepipe_gemm_tune.py
"""

import argparse
import datetime
import functools
import json
import os

import numpy as np
import torch

from quack.gemm import gemm as quack_gemm

print = functools.partial(print, flush=True)


def cta_tile_m(tile_m: int, cluster_m: int) -> int:
    """CTA tile M, which is what the tile-flag index space counts.

    `cluster_m % 2 == 0` with `tile_m in (128, 256)` selects the 2-CTA MMA, so
    the MMA tiler spans two CTAs and each CTA's epilogue writes half the rows
    at its own m-tile coordinate (mirrors GemmSm100.use_2cta_instrs)."""
    return tile_m // 2 if (cluster_m % 2 == 0 and tile_m in (128, 256)) else tile_m


def default_configs():
    # (tile_m, tile_n, cluster_m, cluster_n)
    cfgs = []
    for tile_m, tile_n in [(128, 128), (128, 192), (128, 256), (256, 128),
                           (256, 192), (256, 256)]:
        for cluster_m, cluster_n in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            cfgs.append((tile_m, tile_n, cluster_m, cluster_n))
    return cfgs


def build_problem(args, device):
    """Grouped GEMM with tokens spread over experts the way routing does:
    multinomial around the mean, not equal splits."""
    rng = np.random.default_rng(0)
    total_m = args.tokens * args.topk
    counts = rng.multinomial(total_m, np.ones(args.experts) / args.experts)
    cu = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    A = torch.randn(total_m, args.hidden, device=device, dtype=torch.bfloat16)
    B = torch.randn(args.experts, args.n, args.hidden, device=device,
                    dtype=torch.bfloat16)
    D = torch.empty(total_m, args.n, device=device, dtype=torch.bfloat16)
    return A, B, D, torch.from_numpy(cu).to(device), counts


def flag_metadata(counts, ctile_m, device):
    """Per-expert exclusive cumsum of m-tile counts + total tiles."""
    ntiles = [(int(c) + ctile_m - 1) // ctile_m for c in counts]
    offsets = np.concatenate([[0], np.cumsum(ntiles)[:-1]]).astype(np.int32)
    return torch.from_numpy(offsets).to(device), int(sum(ntiles))


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_ab(launch, warmup, iters, repeats):
    """Publish-off vs publish-on, INTERLEAVED and repeated.

    A single A/B pass cannot resolve the publish cost: the delta is ~2% while
    run-to-run spread on a shared box is 3-5%, which is how an earlier sweep
    produced *negative* overheads. Interleaving within each repeat keeps slow
    drift (clocks, a co-tenant ramping up) common to both arms, and repeating
    turns the answer into a distribution instead of one draw.

    Returns per-repeat lists; the caller reports mean +/- stdev and min. Min is
    the least contaminated single estimate of the machine's capability; the
    stdev says whether the mean is worth quoting at all.
    """
    plain, pub = [], []
    for _ in range(repeats):
        plain.append(bench(lambda: launch(False), warmup, iters))
        pub.append(bench(lambda: launch(True), warmup, iters))
    return plain, pub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts", type=int, default=32,
                   help="local experts (one rank's share)")
    p.add_argument("--hidden", type=int, default=7168, help="K")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--world", type=int, default=1,
                   help="publish fan-out. The push-combine gate is rank-LOCAL "
                        "(1 flag array), which is what the pipeline runs. Use 8 to "
                        "model the pull-combine / gemm+allreduce broadcast, where "
                        "the cost scales with world size.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-publish", dest="publish", action="store_false",
                   help="pure GEMM only, no tile flags anywhere — measures the "
                        "config headroom independently of the publish")
    p.add_argument("--repeats", type=int, default=5,
                   help="interleaved publish-off/on passes; the spread across "
                        "them is what tells you if the delta is real")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--flag-stride", type=int, default=1,
                   help="int32 elements between consecutive tile flags "
                        "(1 = 32 flags per 128B line, 32 = one line each)")
    p.add_argument("--configs", type=str, default=None,
                   help="comma list of tileM x tileN x clusterM x clusterN, "
                        "e.g. 128x256x1x1,256x256x2x1 (default: full sweep)")
    args = p.parse_args()

    if args.configs:
        configs = [tuple(int(v) for v in c.split("x")) for c in args.configs.split(",")]
    else:
        configs = default_configs()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    A, B, D, cu, counts = build_problem(args, device)
    total_m = int(A.shape[0])
    flops = 2 * total_m * args.n * args.hidden
    print(f"grouped GEMM: total_m={total_m} (tokens={args.tokens} x topk={args.topk}) "
          f"K={args.hidden} N={args.n} experts={args.experts} on {props.name} "
          f"({props.multi_processor_count} SMs), publish fan-out world={args.world}")

    rows = []
    for tile_m, tile_n, cluster_m, cluster_n in configs:
        ctm = cta_tile_m(tile_m, cluster_m)
        tile_off, total_tiles = flag_metadata(counts, ctm, device)
        stride = args.flag_stride
        flags = torch.zeros(total_tiles * stride, dtype=torch.int32, device=device)
        # The publish broadcasts to `world` peer arrays; point them all at the
        # same local buffer so a single GPU sees the real per-tile fan-out.
        ptrs = torch.tensor([flags.data_ptr()] * args.world, device=device,
                            dtype=torch.int64)

        def launch(publish):
            quack_gemm(A, B, D, C=None, tile_count_semaphore=None,
                       tile_M=tile_m, tile_N=tile_n,
                       cluster_M=cluster_m, cluster_N=cluster_n,
                       persistent=True, cu_seqlens_m=cu,
                       tile_flag_ptrs=ptrs if publish else None,
                       tile_flag_offsets=tile_off if publish else None,
                       tile_flag_stride=stride)

        tag = f"{tile_m}x{tile_n} c{cluster_m}x{cluster_n}"
        try:
            if args.publish:
                plain_r, pub_r = bench_ab(launch, args.warmup, args.iters, args.repeats)
            else:
                plain_r = [bench(lambda: launch(False), args.warmup, args.iters)
                           for _ in range(args.repeats)]
                pub_r = plain_r  # no publish arm; overhead columns read 0
        except Exception as e:  # unsupported combos (smem, 2-CTA rules, epi_stage)
            print(f"  {tag:<20} SKIP  {type(e).__name__}: {str(e).splitlines()[0][:90]}")
            continue

        # Paired per-repeat deltas, so each overhead reading is formed from two
        # runs taken back to back rather than from two separately-averaged
        # arms; the stdev of these is the honest error bar.
        over_r = [100 * (b - a) / a for a, b in zip(plain_r, pub_r)]
        plain, pub = float(np.mean(plain_r)), float(np.mean(pub_r))
        over_mean, over_sd = float(np.mean(over_r)), float(np.std(over_r, ddof=1))
        # Absolute cost per release atomic, the shape-independent number:
        # one publish per (m-tile, n-tile) per peer.
        n_tiles = (args.n + tile_n - 1) // tile_n
        n_pub = total_tiles * n_tiles * args.world
        ns_per_pub = (pub - plain) * 1e6 / n_pub

        # Correctness of the publish itself: every m-tile must have been
        # counted exactly ceil(N / cta_tile_n) times per launch per peer.
        expect = n_tiles * args.world * (args.warmup + args.iters) * args.repeats
        pub_ok = (not args.publish
                  or bool((flags.view(total_tiles, stride)[:, 0] == expect).all().item()))
        rows.append(dict(flag_stride=stride, tile_m=tile_m, tile_n=tile_n, cluster_m=cluster_m,
                         cluster_n=cluster_n, cta_tile_m=ctm,
                         ms=plain, ms_min=min(plain_r), ms_sd=float(np.std(plain_r, ddof=1)),
                         ms_publish=pub, ms_publish_min=min(pub_r),
                         tflops=flops / plain / 1e9, tflops_peak=flops / min(plain_r) / 1e9,
                         overhead_pct=over_mean, overhead_sd=over_sd,
                         overhead_per_repeat=over_r, ms_per_repeat=plain_r,
                         ms_publish_per_repeat=pub_r,
                         publishes=n_pub, ns_per_publish=ns_per_pub,
                         publish_ok=pub_ok))
        # A delta smaller than its own spread is not a measurement.
        verdict = "" if abs(over_mean) > 2 * over_sd else "  (< 2 sd: noise)"
        print(f"  {tag:<20} cta_m={ctm:<4} {plain:7.3f} +-{np.std(plain_r, ddof=1):5.3f} ms  "
              f"{flops / plain / 1e9:7.1f} TFLOPS | publish {pub:7.3f} ms "
              f"({over_mean:+5.1f}% +-{over_sd:4.1f}, {ns_per_pub:5.1f} ns/pub)"
              f"{verdict}{'' if pub_ok else '   FLAGS WRONG'}")

    if rows:
        best = min(rows, key=lambda r: r["ms"])
        print(f"\nbest: {best['tile_m']}x{best['tile_n']} "
              f"c{best['cluster_m']}x{best['cluster_n']} -> {best['ms']:.3f} ms "
              f"({best['tflops']:.1f} TFLOPS), publish overhead "
              f"{best['overhead_pct']:+.1f}% +-{best['overhead_sd']:.1f} "
              f"({best['ns_per_publish']:.1f} ns per publish over "
              f"{best['publishes']:,} publishes)")
        med = float(np.median([r["overhead_pct"] for r in rows]))
        print(f"median publish overhead across configs: {med:+.1f}%")
        bad = [r for r in rows if not r["publish_ok"]]
        if bad:
            print(f"WARNING: {len(bad)} config(s) published wrong flag counts")

    if args.results_dir:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = (f"gemm_tune_t{args.tokens}_k{args.hidden}_n{args.n}"
                f"_e{args.experts}_w{args.world}_{stamp}")
        d = os.path.join(args.results_dir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump(dict(args=vars(args), rows=rows), f, indent=2)
        print(f"wrote {d}/results.json")


if __name__ == "__main__":
    main()
