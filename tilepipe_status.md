# TilePipe status: results, problems, next steps

_2026-07-25 · 2 GPUs (B200, 148 SMs), DSv3-ish shapes: K=7168, N=4096,
topk=8, experts = 32 x world_size. Raw data in `bench_results/`._

## Where we are

Two pipelines are functionally correct end to end:

- **dispatch -> gated GEMM** (`examples/distributed/tilepipe.py`)
- **GEMM -> combine**, both pull and push variants
  (`examples/distributed/tilepipe_gemm_combine.py`)

Kernel-level tests with their own correctness gates live in `moe_comm.py`
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
SMs. TMA backend is ~15% ahead of SIMT at 12-36 SMs for push-combine
(SIMT is the current default per earlier dispatch-side measurements — the
dispatch A/B has not been run yet).

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
5. **Ratios are measured at N=4096.** The real combine follows the down
   projection (N=hidden=7168) where comm volume is ~1.75x larger against half
   the FLOPs — combine becomes a much bigger fraction. The sweep should be
   re-run at `--gemm-n 7168` before optimizing against current ratios.

## Decisions taken (and why)

- **Push over pull** for the overlapped role: readiness is per-expert
  (uniform across the GEMM window) rather than per-token (all topk, so
  back-loaded); the gate is local to the producer, so no cross-GPU busy-wait
  and no dependence on world size.
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

1. **Measure time-to-first-ready-tile** (cheap instrumentation). Settles
   whether combine is starved by readiness or by the SM tax. The raster
   heuristic picks `AlongN` for our shape (blocks_m=1024 >> blocks_n=32), so
   rows should complete early and steadily — but this is unverified.
2. **Verify the SM partition** actually matches `max_active_clusters`, and
   sweep the low-C regime (8/12/16) where the model predicts a win.
3. **`LayoutSemaphore` refactor.** Producer publishes at its natural (m,n)
   tile coordinate (already true); consumers declare their readiness unit as
   a layout, with `target = |fiber of L|`. Retires `n_tiles`-as-an-argument
   and the hand-maintained index spaces that caused three bugs (trickle hang,
   flag_idx mismatch, tail-publish). This is the reuse vehicle for
   **GEMM+allreduce**, whose consumer unit is one (m,n) tile (target 1) and
   which needs *no epilogue change* under this design.
4. **Pipelined gate**, then re-measure the 12%/6.5% overhead.
5. Dispatch-side SIMT vs TMA A/B at K=7168 to confirm the default backend.
6. Fix or retire the `CombineTmaKernel` race (push supersedes it for the
   overlapped role; it remains the serial/standalone combine).

## Not worth doing (measured or reasoned)

- **Pushing every 16/32 tokens**: the MMA emits a whole 128-row tile at once,
  so sub-tile M granularity creates no earliness; only `tile_M=32` would, at
  large GEMM cost.
- **Per-(m,n)-tile pushing in combine**: the destination row is contiguous
  `[N]`, so a column slice becomes 128 x 256 B strided writes instead of one
  8 KB row — 32x more transfers where we are already bandwidth-limited. (For
  allreduce this objection does not apply: source and destination layouts
  match.)
- **Per-token-block arrival counters**: buys <=0.15 ms, behind a 2 ms problem.
