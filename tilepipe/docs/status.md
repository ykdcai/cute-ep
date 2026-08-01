# TilePipe status: results, problems, next steps

_2026-07-25 · 2 GPUs (B200, 148 SMs), DSv3-ish shapes: K=7168, N=4096,
topk=8, experts = 32 x world_size. Raw data in `bench_results/`._

## Where we are

Two pipelines are functionally correct end to end:

- **dispatch -> gated GEMM** (`tilepipe/dispatch_gemm.py`)
- **GEMM -> combine**, both pull and push variants
  (`tilepipe/gemm_combine.py`)

Kernel-level tests with their own correctness gates live in `tilepipe/moe_comm.py`
(`--test-tma-dispatch`, `--test-push-combine`, `--test-tma-combine`); all
sweep tokens 2048->16384 and write timestamped results.

## Results

**Comm kernel alone (push-combine, N=4096, GB/s per SM):**

| comm SMs | 4096 tok | 16384 tok |
|---|---|---|
| 12 | 25.8 | 30.8 |
| 24 | 22.0 | 28.8 |
| 36 | 19.4 | 26.8 |
| 48 | 15.3 | 20.4 |

Per-SM efficiency *improves* with batch size; peak aggregate ~1 TB/s at 48
SMs.

**SIMT vs TMA A/B, 4 GPUs, N=hidden=7168, topk=8, 128 experts, 16384
tokens/rank** (`moe_comm.py --test-push-combine` / `--test-tma-dispatch`,
`bench_results/{push_combine,tma_dispatch}_4gpu_h7168_topk8_e128_*`):

| CTAs | push simt | push tma | disp simt | disp tma | tma/simt (push) |
|---|---|---|---|---|---|
| 8  | 8.550 ms (27.5 /SM) | 6.535 ms (36.0) | 7.724 ms | 4.975 ms | 0.76x |
| 12 | 5.827 ms (26.9) | **3.424 ms (45.8)** | 5.339 ms | 3.351 ms | **0.59x** |
| 16 | 4.432 ms (26.5) | 3.808 ms (30.9) | 4.079 ms | 2.871 ms | 0.86x |
| 24 | 3.086 ms (25.4) | 3.112 ms (25.2) | 2.855 ms | 2.756 ms | 1.01x |
| 36 | **2.225 ms (23.5)** | 2.490 ms (21.0) | 2.637 ms | 2.409 ms | 1.12x |
| 48 | 2.597 ms (15.1) | 2.378 ms (16.5) | 2.462 ms | 2.345 ms | 0.92x |

**TMA wins decisively where the overlap actually lives.** At 12 CTAs it is
1.70x faster for push-combine and 1.59x for dispatch, at 45.8 GB/s per SM
against SIMT's 26.9. The crossover is ~24 CTAs; above it the two converge on
the NVLink5 roofline (~790-800 GB/s, ~90% of the ~900 GB/s per-direction
limit), and SIMT's 36-CTA point (845 GB/s) is the single best push number.

This contradicts the current default. `--comm-impl` defaults to `simt` and
its help string still claims "faster per SM at TilePipe CTA counts" — true at
36-48 CTAs, false by 1.6-1.7x at 8-16, which is the regime the SM tax forces
us into. **Re-default to TMA for the overlapped role** (dispatch and combine
both) unless the 2-GPU numbers disagree.

Both backends are non-monotonic in CTA count: TMA dips at 16 (3.808 ms, worse
than 12), SIMT dips at 48 (2.597 ms, worse than 36), reproducibly across all
four token counts. Same kernel, same CTA counts, different row plans — points
at segment-to-CTA distribution rather than the transfer engine.

Dispatch and push-combine track each other within 5-11% at >=8192 tokens
(same `VarlenAllToAllKernel`, same bytes); the gap widens at small sizes
(1.29x at 2048/8 CTAs), so the residual is fixed per-row plan overhead.

**Fused GEMM -> combine (16384 tokens):**

```
pure GEMM (148 SMs)     8.36 ms (922 TFLOPS);  with publish  8.64 ms (+3.3%)
pure combine            1.61 ms @64 CTAs ... 12.09 ms @8 CTAs
serial (GEMM then combine)          10.30 ms
best overlapped (push)              10.69 ms   = 0.96x  (i.e. a loss)
local reduce (after barrier)         0.15 ms
```

Overlap is at best break-even across all sizes (0.75x at 2K -> 0.96x at 16K).

## Problems

1. **The SM tax exceeds what we hide.** The GEMM is compute-bound and nearly
   SM-linear, so donating C SMs costs `8.64 x C/148`. At C=24 that is ~1.4 ms
   against 1.1-1.8 ms of comm hidden — a wash by construction, independent of
   any pipeline tuning. Only the low-C regime can win: at C=12 the model
   predicts ~4% net gain (push 3.37 ms hides inside a 9.4 ms slowed GEMM).
2. **A ~0.9 ms model-vs-measured gap at low CTA counts.** Measured overlapped
   is ~0.9 ms worse than `max(slowed GEMM, comm)`. Unresolved; candidates:
   (a) `max_active_clusters` may not free the SMs we assume, (b) gate stall —
   the poll is a serialized memory round-trip before each tile's copies,
   (c) readiness timing (see below).
3. **Gate overhead is real but not where I predicted.** Pull-combine gated vs
   ungated: 13.2% @32 CTAs, 6.5% @64. Replacing the polling atomic RMW with
   `ld.acquire.sys` did **not** help (12.0->13.2%, 4.2->6.5%), so the cost is
   the serialized latency itself, not exclusive-line acquisition. The
   pipelined-gate fix (poll tile t+1 while pushing tile t) is untested.
4. **Latent race in `CombineTmaKernel`** (pre-existing, TODO left in-file):
   1-2 wrong rows out of 1024 at >=1024 tokens, nondeterministic, reproduces
   with static shapes. Likely explains pull's rel_err ~9e-3 vs push's ~3e-3.
   No longer on the pipeline's critical path (pull was removed), but the
   kernel is still the standalone/serial combine, so the bug is still live.
5. **Ratios are measured at N=4096.** The real combine follows the down
   projection (N=hidden=7168) where comm volume is ~1.75x larger against half
   the FLOPs — combine becomes a much bigger fraction. The sweep should be
   re-run at `--gemm-n 7168` before optimizing against current ratios.

## Decisions taken (and why)

- **Push over pull** for the overlapped role: readiness is per-expert
  (uniform across the GEMM window) rather than per-token (all topk, so
  back-loaded); the gate is local to the producer, so no cross-GPU busy-wait
  and no dependence on world size.
- **Pull removed from the pipeline entirely** (`gemm_combine.py` is push-only).
  It lost in *both* roles at 16K/N=4096, not just overlapped: serial 8.53 vs
  6.81 ms, best overlapped 0.90x vs 1.04x. Removing it also deletes the last
  cross-rank flag traffic from this benchmark — the symmetric `tile_flags`
  array, `rank_tile_base`, `flag_idx` and the remote publish pointer table all
  go; the GEMM now publishes only to its own rank. `CombineTmaKernel` itself
  stays in `moe_comm.py` as the standalone/serial combine (and its machinery
  is shared with the TMA dispatch backend), so this is a pipeline decision,
  not a kernel deletion.
- **No arrival counter.** The push kernel is pure data movement
  (`flag_peer_ptrs=None` compiles out all segment bookkeeping); the local
  reduce is ordered by the pipeline's existing barrier and measured
  separately (~2% of the step at 16K).
- **One compile serves every token count.** All token-dependent extents are
  `mark_layout_dynamic`; verified by compiling at 256 tokens and running at
  512/1024/333.
- **Destination rotation reverted.** It broke the invariant that list position
  == D-buffer row, and destroyed gate monotonicity. Any future fairness
  rotation must keep whole tiles contiguous.

## Next steps

0. **Verify the low-overhead publish on the real pipeline** (in flight). The
   epilogue publish now emits `red.release.sys` (was `atom`, an RMW returning
   a dead value) over a fully unrolled per-rank loop (the world size is a
   compile-time constant via `tile_flag_world`; it used to be dynamic, which
   wrapped every publish in an unroll-by-16/8/4 branch ladder over gmem
   loads). Single-GPU A/B at 128x256: ~+2% at 16K tokens, at or below
   run-to-run noise. **Flag padding does not help** — `tile_flag_stride`
   swept at 1/8/32 moves nothing outside noise, because NVIDIA atomics
   execute at the L2 slice rather than migrating a line to the SM, so there
   is no CPU-style ping-pong. What does cost is the *number* of publishes:
   `tile_n=128` doubles the n-tiles and shows a stubborn +13-18%.
   Gate before moving on: overlap runs stay race-free.
1. **Multi-CTA GEMM is deferred, not rejected** (TODO in `gemm_sm100.py`).
   `256x256 cluster 2x1` measures 5.32 ms / 1446 TFLOPS vs 5.66 ms / 1360
   for today's `128x256 c1x1` (+6%). Enabling it found a real bug, now
   fixed: cluster-granular scheduling gives overhanging CTAs an out-of-range
   (m, n), and the publish was unpredicated, so it landed on the *next*
   expert's flags. The predicate is compile-time gated to `cluster != 1x1`,
   so the 1x1 PTX is unchanged. Picking this up also requires every host
   that builds `tile_offsets` to switch to CTA tile M (`tile_M // 2`).
   Sweep both with `tilepipe/gemm_tune.py` (single GPU).
2. **Publish to all peers with ONE instruction (multimem) — NOT for
   gemm+combine.** `distributed_gemm_all_reduce_blackwell.py` publishes with
   `cutlass.utils.distributed.multimem_red_add1(lock_ptr, order="release",
   scope="gpu")` on a *multicast* flag tensor (`barrier_flag_mc`): one
   instruction updates every peer's copy, so its cost is O(1) in world size
   instead of our O(world) loop of `red`s.
   **But push-combine's gate is rank-LOCAL** — every `launch_gemm` in
   `tilepipe_gemm_combine.py` passes `local_publish=True`, a 1-entry pointer
   table — so the fan-out is 1 and multimem has nothing to collapse. It is
   relevant only to **gemm+allreduce** (where every rank must see the
   producer's tile) and to the deprecated pull-combine path.
   Measured consequence of the fan-out: publish overhead at world=1 is
   +2.5% +-3.5 (128x256) and +3.7% +-5.5 (128x128), ~9-10 ns per publish —
   both inside their own error bars. The alarming +13-18% previously
   attributed to publish *count* was mostly the tuner's `--world 8` default,
   i.e. an 8x fan-out the push path never pays. The tuner now defaults to
   world=1 to match the pipeline.
   Also worth knowing: `red_add1`/`multimem_red_add1` already exist upstream
   and are the same instruction as `quack/tilepipe_sync.py:red_release_sys`,
   which only earns its place by taking a variable count.
   Two things NOT to copy from it: it calls `c_pipeline.producer_tail()`
   per output tile to drain stores before flagging (its own comment notes
   this differs from the regular epilogue) — that is the ~10% drain we avoid
   by publishing one tile behind; and it sizes flags per CTA via
   `linear_idx * cluster_size + block_idx_in_cluster()`, which handles the
   cluster overhang by giving overhanging CTAs their own harmless slots
   instead of predicating. Either solution works; ours keeps the flag space
   equal to the consumer's tile space, which the combine gate needs.
3. **push-combine transfer redesign — POSTPONED, analysis complete.**
   See `tilepipe/docs/combine_design.md`. Summary: destination order is free (the
   reduce can permute), which turns each segment into a contiguous ->
   contiguous transfer and lets consecutive rows merge into one bulk copy.
   Full-width 1-D beats 2-D column slices (no tensormaps); SMEM staging — not
   contiguity — bounds op size, so the realistic gain is 8 KB -> ~32 KB per op
   plus the per-row index loads leaving the inner loop. Fragmentation by home
   rank is real but never fatal (worst measured case still beats today's 8 KB).
   Postponed in favour of prototyping gemm+allreduce.
4. **Measure time-to-first-ready-tile** (cheap instrumentation). Settles
   whether combine is starved by readiness or by the SM tax. The raster
   heuristic picks `AlongN` for our shape (blocks_m=1024 >> blocks_n=32), so
   rows should complete early and steadily — but this is unverified.
5. **Verify the SM partition** actually matches `max_active_clusters`, and
   sweep the low-C regime (8/12/16) where the model predicts a win.
   `gemm_combine.py` now *measures* the SM tax instead of modelling it: it
   times the publishing GEMM capped to `num_sms - c` clusters alone and
   reports it as its own column against the full-device GEMM. `ideal` stays
   **speed of light** — `max(full-device publishing GEMM, push alone)` — on
   purpose: we want the bound the approach is aiming at, not one conditioned
   on whichever SM split we happen to be running. The SM tax is visible
   beside it rather than folded into it.
   Caveat: `push alone` runs with the GEMM idle, so it sees full HBM and
   NVLink; the bound ignores shared-resource contention and is optimistic by
   construction, which is what a speed-of-light number is for.
6. **Per-SM timestamp trace** (the real instrument for gate stall). Record the
   exact nanotime at which each GEMM tile completes and each combine transfer
   starts — `%%globaltimer` at the publish site and at the gate's exit, one
   slot per (tile, CTA) in a gmem trace buffer, dumped and aligned after the
   run. That gives readiness-vs-consumption directly per SM, which is what
   actually attributes the ~0.9 ms gap in #2 (gate instruction cost vs stall
   vs SM starvation). Whole-kernel A/Bs cannot separate those; deferred until
   the SM-tax and low-C sweeps are in.
7. **`LayoutSemaphore` refactor.** Producer publishes at its natural (m,n)
   tile coordinate (already true); consumers declare their readiness unit as
   a layout, with `target = |fiber of L|`. Retires `n_tiles`-as-an-argument
   and the hand-maintained index spaces that caused three bugs (trickle hang,
   flag_idx mismatch, tail-publish). This is the reuse vehicle for
   **GEMM+allreduce**, whose consumer unit is one (m,n) tile (target 1) and
   which needs *no epilogue change* under this design.
8. **Pipelined gate**, then re-measure the 12%/6.5% overhead.
9. ~~Dispatch-side SIMT vs TMA A/B at K=7168~~ **done** (see Results): TMA
   is 1.6-1.7x faster at 8-16 CTAs for both dispatch and push-combine, so
   the `simt` default and its help string are wrong for the overlapped
   role. Flip `--comm-impl` to `tma` in `gemm_combine.py` and
   `dispatch_gemm.py` once confirmed at world=2.
10. Fix or retire the `CombineTmaKernel` race. Pull is out of the pipeline, so
   this no longer blocks overlap work, but the kernel is still the
   standalone/serial combine and its machinery backs the TMA dispatch path.

## Not worth doing (measured or reasoned)

- **Pushing every 16/32 tokens**: the MMA emits a whole 128-row tile at once,
  so sub-tile M granularity creates no earliness; only `tile_M=32` would, at
  large GEMM cost.
- **Per-(m,n)-tile pushing in combine**: OPEN, not settled — the earlier
  "32x more transfers" claim in this file was wrong. It assumed a SIMT
  lowering (a column slice becoming 128 strided per-row writes). TMA moves
  2D tiles natively, and no accumulation is involved, so the honest
  arithmetic is: per m-tile today we issue 128 one-dimensional
  `cp.async.bulk` row copies of 8 KB (= 1 MB); tile-granular with SM100
  `cp.async.bulk.tensor.2d.tile::scatter4` would be 16 n-tiles x 32
  scatter4 ops of 4 x 512 B (= the same 1 MB) — **4x more ops, each 4x
  smaller**, not 32x.
  The real blocker is different: the 128 rows of a GEMM m-tile go to 128
  arbitrary token slots (`slot = tok * topk + j`) on possibly different
  ranks, so the destination is a scatter, not an affine 2D tile. That needs
  scatter4 plus a tensormap per destination rank (see
  `quack/tensormap_manager.py`); rows are already sorted by (expert, home),
  so each scatter4's 4 rows do share a destination rank.
  Whether it pays depends entirely on **earliness**, which is unmeasured:
  the raster heuristic picks `AlongN`, so all `n_tiles` of an m-tile
  complete back to back and the wait for a whole row block may already be
  short. Settle with the time-to-first-ready-tile instrumentation before
  building it. (For allreduce none of this arises: source and destination
  layouts match, so a per-(m,n) gate with target 1 is both finer and
  contiguous.)
- **Per-token-block arrival counters**: buys <=0.15 ms, behind a 2 ms problem.
