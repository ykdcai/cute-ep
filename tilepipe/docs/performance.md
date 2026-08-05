# TilePipe performance: where we stand

Measured 2026-08-04 on 8x B200. DeepSeek-V3 shapes: N=4096, K=7168, topk=8,
experts = 32 x world, uniform routing. All times are **milliseconds**, medians
over 20-30 paired iterations with the 16-84 percentile band; every variant is
launched once per iteration so ratios are formed per-iteration rather than
across runs.

Environment (pinned in `pyproject.toml`; `uv run` needs no `--no-sync`):
torch 2.12.1+cu130, nvidia-cutlass-dsl 4.5.3[cu13], cuDNN 9.24.0.43,
flashinfer 0.6.16.post1, nvshmem4py-cu13 0.3.1.

Path index: `bench_results/final_results.txt`.

---

## 1. Headline

Absolute times at 4 GPUs / 16384 tokens, the size where the combine overlap
is strongest:

| pipeline | TP serial | TP overlap | overlap gain | FlashInfer | TP best vs FI |
|---|---|---|---|---|---|
| GEMM -> combine  | 7.343 | **6.664** | **1.10x** | 10.827 | **1.62x** |
| dispatch -> GEMM | 8.319 | **7.486** | **1.11x** | 12.668 | **1.69x** |

| | verdict |
|---|---|
| GEMM -> combine overlap | break-even to 8K, **wins at 16K-32K** (1.06-1.12x) |
| dispatch -> GEMM overlap | **wins at every size** (1.07-1.16x), with SIMT @32 warps |
| combine collective vs FlashInfer | parity at 4 GPU, **ahead at 8 GPU** |
| dispatch collective vs FlashInfer | **1.5-3.1x behind** |
| grouped GEMM | 1044-1295 TFLOPS |

Three separable claims. End-to-end we are 1.53-1.74x ahead of the FlashInfer
pipeline at every size. The overlap contributes 1.06-1.16x of that; the rest is
the GEMM and the local reduce. Both overlaps now win, but in different regimes:
combine only above 8K tokens, dispatch at every size.

---

## 2. GEMM -> push-combine

TilePipe: `bench_results/sync4/gemm_combine_4gpu_N4096_K7168_topk8_20260804_205220`,
`bench_results/sync8/gemm_combine_8gpu_N4096_K7168_topk8_20260804_205748`.
FlashInfer: `bench_results/sync4/flashinfer_gemm_combine_4gpu_N4096_K7168_topk8_20260804_205652`,
`bench_results/sync8/flashinfer_gemm_combine_8gpu_N4096_K7168_topk8_20260804_205826`.

TilePipe's serial baseline runs its comm phase on the **full device** at the
bandwidth-optimal config; anything less would flatter the overlap.

**4 GPUs (experts=128)** -- ms

| tokens | GEMM | comm@148 | TP serial | TP overlap | comm SMs | **overlap gain** | FI serial | **TP ovl vs FI** |
|---|---|---|---|---|---|---|---|---|
| 2048  | 0.963 | 0.327 | 1.131 | 1.227 | 12 | 0.92x [0.90,0.96] | 1.928 | 1.57x |
| 4096  | 1.631 | 0.456 | 1.926 | 1.931 | 12 | 1.00x [0.98,1.01] | 2.948 | 1.53x |
| 8192  | 3.252 | 0.788 | 3.556 | 3.528 | 8  | 1.01x [0.99,1.02] | 5.486 | 1.56x |
| 16384 | 6.490 | 1.491 | 7.343 | **6.664** | 8 | **1.10x [1.05,1.12]** | 10.827 | **1.62x** |
| 32768 | 12.803 | 2.927 | 14.860 | **13.995** | 8 | **1.06x [1.03,1.11]** | 22.524 | **1.61x** |

**8 GPUs (experts=256)** -- ms

| tokens | GEMM | comm@148 | TP serial | TP overlap | comm SMs | **overlap gain** | FI serial | **TP ovl vs FI** |
|---|---|---|---|---|---|---|---|---|
| 2048  | 0.951 | 0.344 | 1.133 | 1.209 | 12 | 0.94x [0.83,0.96] | 2.002 | 1.66x |
| 4096  | 1.591 | 0.514 | 1.940 | 1.944 | 12 | 1.00x [0.97,1.01] | 2.980 | 1.53x |
| 8192  | 3.187 | 0.863 | 3.619 | 3.527 | 8  | 1.03x [1.01,1.05] | 5.718 | 1.62x |
| 16384 | 6.648 | 1.691 | 7.737 | **7.024** | 8 | **1.10x [1.05,1.12]** | 11.266 | **1.60x** |
| 32768 | 13.195 | 3.362 | 15.808 | **14.181** | 8 | **1.12x [1.05,1.16]** | 23.230 | **1.64x** |

Overlap pays only once there is enough comm to hide: at 2048 the residual tail
is most of the 0.33 ms transfer; at 32768 it is a small part of 2.93 ms.
Optimum is 8-12 comm SMs at every size, SM tax +1.8% to +7.6%.

End-to-end we are **1.53-1.66x** ahead of the FlashInfer pipeline, but see the
caveat below -- most of that margin is the GEMM and the local reduce, not the
overlap, which contributes at most 1.12x.

---

## 3. Dispatch -> GEMM

TilePipe: `bench_results/dsimt32/dispatch_gemm_4gpu_simt_N4096_K7168_topk8_20260804_215121`
(tuned) and
`bench_results/dsimt16/dispatch_gemm_4gpu_simt_N4096_K7168_topk8_20260804_215055`
(default warps). FlashInfer:
`bench_results/sync4/flashinfer_dispatch_gemm_4gpu_N4096_K7168_topk8_20260804_213708`.

**The backend and its warp count decide whether this pipeline wins at all.**
`dispatch_gemm.py` defaults to `--copy simt`; the SIMT kernel's only knob is
`--comm-warps`, and the default 16 is not its optimum.

| tokens | TMA (`--copy tma`) | SIMT, 16 warps (default) | **SIMT, 32 warps (tuned)** |
|---|---|---|---|
| 2048  | 0.93x | 1.00x | **1.11x** |
| 4096  | 0.91x | 0.96x | **1.16x** |
| 8192  | 0.89x | 0.99x | **1.07x** |
| 16384 | 0.95x | 1.08x | **1.11x** |
| 32768 | 0.89x | 1.16x | **1.15x** |

**4 GPUs, SIMT @32 warps** -- ms

| tokens | GEMM | dispatch alone | TP serial | TP overlap | **overlap gain** | FI serial | **TP ovl vs FI** |
|---|---|---|---|---|---|---|---|
| 2048  | 0.915 | 0.636 | 1.338 | **1.204** | **1.11x** | 2.058 | **1.71x** |
| 4096  | 1.626 | 1.193 | 2.318 | **1.999** | **1.16x** | 3.467 | **1.73x** |
| 8192  | 3.096 | 2.190 | 4.204 | **3.926** | **1.07x** | 6.473 | **1.65x** |
| 16384 | 6.107 | 4.198 | 8.319 | **7.486** | **1.11x** | 12.668 | **1.69x** |
| 32768 | 12.552 | 7.027 | 16.710 | **14.513** | **1.15x** | 25.219 | **1.74x** |

Unlike combine, this pipeline wins at **every** size including 2048 -- its
dispatch is proportionally larger relative to the GEMM (0.64 ms against 0.92 ms
at 2048) so there is more to hide even at small batch.

### Why the warp count matters so much

Standalone SIMT dispatch collective, 4 GPUs, 16384 tokens, hidden=7168 (ms):

| `--comm-warps` | 8 CTA | 12 CTA | 16 CTA | 24 CTA | 148 CTA | GB/s per SM @8-16 |
|---|---|---|---|---|---|---|
| 8  | 14.788 | 9.807 | 7.210 | 5.100 | 2.289 | ~16 |
| 16 (default) | 7.693 | 5.381 | 4.140 | 2.899 | 2.223 | ~30 |
| **32** | **5.460** | **3.645** | **2.739** | 2.859 | 2.179 | **~43** |

32 warps is **1.41-1.51x faster than the default at 8-16 CTAs**, which is
exactly the regime the overlap runs in; by 148 CTAs the three converge. This is
the same shape of result as the combine autotune (section 5): the parameter
that matters is the one that raises per-SM efficiency at low CTA counts.

At 148 CTAs SIMT@32 (2.179 ms) and TMA (2.173 ms) are tied, so the standalone
collective comparison in section 4 is unaffected by the backend choice.

## 4. Communication collectives vs FlashInfer

Kernel-to-kernel, GEMM held out. FlashInfer deduplicates per **destination
rank** (one row per (token, rank)); TilePipe sends one row per **(token,
expert)**. The volume penalty is `topk / E[distinct ranks per token]`: 4.02x
(W=2), 2.22x (W=4), 1.52x (W=8), 1.11x (W=32).

### Combine

TilePipe `PushCombineKernel` @148 CTAs (from the gemm_combine dirs above) vs
`moe_a2a_combine` (from
`bench_results/sync4/flashinfer_gemm_combine_4gpu_N4096_K7168_topk8_20260804_205652`,
`bench_results/sync8/flashinfer_gemm_combine_8gpu_N4096_K7168_topk8_20260804_205826`).

| tokens | FI ms (4 GPU) | TP ms | TP/FI | FI ms (8 GPU) | TP ms | TP/FI |
|---|---|---|---|---|---|---|
| 2048  | 0.345 | 0.327 | 0.95x | 0.416 | 0.344 | **0.83x** |
| 4096  | 0.379 | 0.456 | 1.20x | 0.423 | 0.514 | 1.21x |
| 8192  | 0.658 | 0.788 | 1.20x | 0.851 | 0.863 | 1.01x |
| 16384 | 1.330 | 1.491 | 1.12x | 1.740 | 1.691 | 0.97x |
| 32768 | 3.189 | 2.927 | **0.92x** | 3.813 | 3.362 | **0.88x** |

Our kernel is **1.85-2.42x faster per byte** at W=4 and 1.26-1.85x at W=8,
enough to offset the volume penalty at both ends. At 8 GPUs we are at or ahead
of parity at four of five sizes.

This crossover was **predicted before it was measured**: with a ~1.45x per-byte
edge the break-even volume ratio falls between W=4 (2.22x) and W=8 (1.52x).
Measured dedup at W=8: 1.53x against 1.52 predicted.

### Dispatch

TilePipe `VarlenAllToAllKernel(tma)` @148 CTAs
(`bench_results/sync4/tma_dispatch_4gpu_h7168_topk8_e128_tma_20260804_205513`,
`bench_results/sync8/tma_dispatch_8gpu_h7168_topk8_e256_tma_20260804_205900`)
vs `moe_a2a_dispatch`.

| tokens | FI ms (4 GPU) | TP ms | TP/FI | FI ms (8 GPU) | TP ms | TP/FI |
|---|---|---|---|---|---|---|
| 2048  | 0.162 | 0.508 | 3.14x | 0.243 | 0.472 | 1.94x |
| 4096  | 0.279 | 0.774 | 2.78x | 0.443 | 0.750 | 1.69x |
| 8192  | 0.522 | 1.166 | 2.23x | 0.834 | 1.323 | 1.59x |
| 16384 | 0.979 | 2.173 | 2.22x | 1.641 | 2.497 | 1.52x |
| 32768 | 1.905 | 4.171 | 2.19x | 3.178 | 4.899 | 1.54x |

Per-byte ratio is 0.70-1.01 -- **no advantage, and a deficit at small sizes**.

Peak bandwidth: dispatch reaches 901 GB/s total at W=4/32K, essentially the
NVLink5 per-direction roofline (~900). Combine tops out at ~734 GB/s; its rows
are 8 KB (N=4096) against dispatch's 14 KB (K=7168).

### FlashInfer dispatch -> GEMM, full pipeline

`bench_results/sync4/flashinfer_dispatch_gemm_4gpu_N4096_K7168_topk8_20260804_213708`,
`bench_results/sync8/flashinfer_dispatch_gemm_8gpu_N4096_K7168_topk8_20260804_213732`

| tokens | 4 GPU: dispatch / permute / GEMM / serial | 8 GPU: same |
|---|---|---|
| 2048  | 0.161 / 0.841 / 1.081 / 2.058 | 0.257 / 0.882 / 1.095 / 2.197 |
| 4096  | 0.277 / 1.470 / 1.727 / 3.467 | 0.452 / 1.506 / 1.674 / 3.610 |
| 8192  | 0.501 / 2.753 / 3.234 / 6.473 | 0.835 / 2.781 / 3.208 / 6.807 |
| 16384 | 0.956 / 5.537 / 6.153 / 12.668 | 2.104 / 10.548 / 8.246 / 20.322 |
| 32768 | 1.859 / 11.129 / 12.206 / 25.219 | 5.909 / 16.676 / 24.444 / 46.024 |

**Do not read `permute` as FlashInfer's cost.** It is our naive PyTorch
gather/scatter that rebuilds the expert-grouped GEMM operand from the
deduplicated `[src_rank, slot]` recv buffer; TRT-LLM fuses this into the MoE
kernel. TilePipe's dispatch writes straight into expert-grouped slots and pays
nothing here. The `dispatch` and `GEMM` columns are meaningful; the serial
totals inherit the caveat.

The 8-GPU 16384 and 32768 rows are **suspect**: the GEMM reads 24.4 ms at W=8
against 12.2 ms at W=4 for the *same per-rank shape*. Memory pressure from the
MoeAlltoAll workspace is the likely cause; re-run before quoting.

---

## 5. Standalone comm kernel and autotune

`bench_results/pctune/push_combine_4gpu_h4096_topk8_e128_push_20260802_134734`
(config sweep), `bench_results/pcbench/push_combine_4gpu_h4096_topk8_e128_push_20260802_134005`
(dedicated vs old shared kernel). Config is `workers:stages[:write_window]`;
the tuned table is `tilepipe/push_config.py`.

- Config choice moves standalone time by **30-69%** at low CTA counts,
  narrowing to 8-15% at 148 CTAs. `4:24` and `4:28` are within 3% of best in
  72% of cells; `8:16` never is.
- `stages` is the bandwidth knob (12 -> 24 is +12-18% standalone) but buys
  almost nothing for the overlap, which is tail-bound not bandwidth-bound.
- `workers`: 2-4 only. 1 cannot sustain the bandwidth (0.67x at 8192/8 CTAs);
  8 has the worst tail everywhere.
- `write_window` and `chunk` are **inert** -- swept 8/16/32 and 128/64/32/16,
  spread inside the bands at every size.
- Scaling past 24 CTAs is weak: 24 -> 148 (6.2x the SMs) buys 1.02-1.14x, and
  GB/s per SM falls from ~40 to ~5.

---

## 6. Traces

**`bench_results/ncu_gemm_publish_20260730_220807/gemm_long_pub.ncu-rep`**
and **`gemm_long_nopub.ncu-rep`** (same directory) -- 16384 tokens, total_m
131072, 1042 tiles, 128x256 c1x1, one launch each.
Duration **5.42 ms** (pub) vs **5.43 ms** (nopub): the tile-flag publish is
free. Compute (SM) throughput **93.65%**, DRAM 25.30%, SM clock 1.31 GHz,
occupancy limited by Block Limit Shared Mem = 1. The GEMM is compute-bound
here, so donating SMs to comm costs nearly linearly.

**`bench_results/ncu_gemm_publish_20260730_220807/gemm_short_pub.ncu-rep`**
and **`gemm_short_nopub.ncu-rep`** -- 1024 tokens, total_m 8192, 79 tiles.
Duration **491.52 us** (pub) vs **486.24 us** (nopub), +1.1%. Compute 72.27%,
DRAM 57.68% -- weight-bandwidth-bound (1.88 GB of expert weights read for
117 MB of activations), the opposite regime from the long case.

**`bench_results/ncu_gemm_publish_20260730_220807/ncu_gemm.py`** -- the capture
driver. Compiles, allocates and warms up *before* `cudaProfilerStart`, so
`ncu --profile-from-start off` captures exactly one launch.

**`bench_results/ncu_push/push_w1_8192_4x24_148cta.ncu-rep`** -- push kernel,
world=1 (every row local, so pure HBM, no NVLink), 8192 tokens, config 4:24,
148 CTAs. Duration **185.38 us** for `dram__bytes_read` 537.99 MB +
`dram__bytes_write` 491.99 MB = **~5.56 TB/s of HBM traffic (~69% of B200
peak)**. `sm__throughput` **3.11%**, `l1tex__throughput` 34.77% -- confirms a
pure data-movement kernel with no compute being wasted.

**`bench_results/nsys_push/push_w4_8192.nsys-rep`** -- timeline of the push at
world=4. nsys is the only option for concurrency questions: ncu serialises
kernel launches, which both destroys the thing being measured and *deadlocks* a
gated kernel (the push spins on flags the GEMM would set).

Reproduce a GEMM trace:

```
CUDA_VISIBLE_DEVICES=0 ncu --profile-from-start off --target-processes all \
  --set detailed -o out --force-overwrite \
  uv run python bench_results/ncu_gemm_publish_20260730_220807/ncu_gemm.py \
  --tokens 16384 --publish
```

Multi-GPU needs `--replay-mode application` (kernel replay cannot roll back
peer-memory writes) and one `ncu` per rank.

---

## 7. Reproducing

All commands are plain `uv run` -- `pyproject.toml` is fixed so `--no-sync` is
no longer needed. Check `nvidia-smi` first and pick idle GPUs; give every
concurrent run a distinct `--master-port`. `moe_comm.py` does NOT self-size the
NVSHMEM heap, so `NVSHMEM_SYMMETRIC_SIZE` is required there; `gemm_combine.py`
and `dispatch_gemm.py` size it themselves.

For 8 GPUs: `--nproc-per-node 8`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
and `--num_experts`/`--experts 256` (the rule is 32 x world).

### Comm kernels alone (run these first)

Combine collective + config autotune -- section 5:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NVSHMEM_SYMMETRIC_SIZE=32G uv run \
  torchrun --nproc-per-node 4 --master-port 29742 tilepipe/moe_comm.py \
  --test-push-combine --hidden 4096 --num_experts 128 --topk 8 \
  --token-sweep 2048,4096,8192 --dispatch-ctas 8,12,16,24,48,148 \
  --push-config 4:28,4:24,4:16,4:12,2:24,2:28,8:24,8:16 \
  --iterations 20 --warmup_iterations 5 --results-dir bench_results/pctune
```

`--test-push-combine` defaults to `--comm-impl push` (the dedicated
`PushCombineKernel` that `gemm_combine.py` ships); `simt`/`tma` select the
older shared kernel instead. Every config is correctness-checked before it is
timed.

Dispatch collective + warp autotune -- section 3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NVSHMEM_SYMMETRIC_SIZE=48G uv run \
  torchrun --nproc-per-node 4 --master-port 29961 tilepipe/moe_comm.py \
  --test-tma-dispatch --comm-impl simt --comm-warps 32 --hidden 7168 \
  --num_experts 128 --topk 8 --token-sweep 4096,16384 \
  --dispatch-ctas 8,12,16,24,48,148 --iterations 20 --warmup_iterations 5 \
  --results-dir bench_results/dsw
```

Sweep `--comm-warps 8/16/32` to reproduce the warp table; 32 is the optimum and
16 is the (suboptimal) default.

### Fused pipelines

GEMM -> push-combine -- section 2. Config comes from `tilepipe/push_config.py`
keyed on tokens/rank, so no tuning flags are needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 uv run torchrun --nproc-per-node 4 \
  --master-port 29901 tilepipe/gemm_combine.py \
  --token-sweep 2048,4096,8192,16384,32768 --iters 30 --warmup 5 \
  --results-dir bench_results/sync4
```

Dispatch -> GEMM -- section 3. **`--copy simt --comm-warps 32` is essential**:
the default is `simt` at 16 warps, and `--copy tma` turns the win into a loss:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc-per-node 4 \
  --master-port 29971 tilepipe/dispatch_gemm.py \
  --token-sweep 2048,4096,8192,16384,32768 --gemm-n 4096 --experts 128 \
  --topk 8 --copy simt --comm-warps 32 --comm-sms 8,12,16,24 \
  --iters 20 --warmup 5 --results-dir bench_results/dsimt32
```

### FlashInfer baselines -- section 4

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 uv run torchrun --nproc-per-node 4 \
  --master-port 29905 tilepipe/baselines/bench_flashinfer_gemm_combine.py \
  --token-sweep 2048,4096,8192,16384,32768 --experts 128 --iters 30 --warmup 5 \
  --results-dir bench_results/sync4

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc-per-node 4 \
  --master-port 29934 tilepipe/baselines/bench_flashinfer_dispatch_gemm.py \
  --token-sweep 2048,4096,8192,16384,32768 --experts 128 --iters 20 --warmup 5 \
  --results-dir bench_results/sync4
```

### Traces

See section 6 for the `ncu` capture command and the multi-GPU caveats.

---

## 8. Caveats

- **Uniform routing only.** Expert skew concentrates tokens on fewer ranks,
  which *increases* FlashInfer's dedup advantage. Most likely caveat to move
  the combine conclusion.
- **N=4096 only.** The real combine follows the down projection at N=7168,
  where comm volume is 1.75x larger against half the FLOPs. `stages=24` is
  **illegal** at N=7168 (336 KB > 227 KB SMEM); `push_config.pick()` clamps, so
  the tuned table does not transfer unchanged.
- **FlashInfer's GEMM is not autotunable.** `flashinfer.autotuner` is a no-op
  for `grouped_mm_bf16` (profiling cache stays empty, timing identical to
  1.000x) because the cuDNN backend exposes no tactics. Affects serial totals
  only; every collective figure holds the GEMM out.
- **`+publish` reads -9.2% at 8192/16384** in the fused runs, i.e. the
  publishing GEMM measuring *faster* than the plain one. Unphysical; it is an
  ordering artifact, since `gemm_plain` runs first in each paired iteration and
  eats a cold-L2 penalty. True publish overhead is ~0 (confirmed independently
  by the ncu traces above).
- **Single runs on a dependency stack changed the same day.** The combine
  numbers shifted noticeably after the 2026-08-04 sync (our per-byte edge went
  from 1.15-1.57x to 1.85-2.42x at W=4). Re-run W=8 combine before trusting the
  shift over run-to-run drift.
- **Hopper.** `pyproject.toml` is H100-ready (CUDA 13 wheels serve sm90), but
  the tile-flag publish exists only in `quack/gemm_sm100.py`, so the fused path
  needs an sm90 epilogue port before `gemm_combine.py` runs there. The
  standalone collectives should work as-is.
