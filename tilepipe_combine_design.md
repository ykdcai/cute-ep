# TilePipe push-combine: transfer redesign (POSTPONED, analysis complete)

_2026-07-26. Design settled to the point of implementation, then postponed in
favour of prototyping gemm+allreduce. Nothing here is built. Everything below
is host-side reasoning plus one measurement script; the GEMM kernel is
untouched._

## The problem this solves

Push-combine is inefficient exactly where the overlap needs it: at low CTA
counts. Measured (`bench_results/push_combine_2gpu_h4096_topk8_e64_20260726_004831`,
SIMT, N=4096, world=2, 16384 tokens):

| CTAs | 12 | 24 | 36 | 48 |
|---|---|---|---|---|
| GB/s | 322 | 598 | 824 | 1037 |
| GB/s per SM | 26.8 | 24.9 | 22.9 | 21.6 |

Per-SM efficiency rises as CTAs drop, but absolute bandwidth collapses — 12
CTAs reach 31% of the 48-CTA figure. That is a latency-bound regime, and the
cause is visible in the inner loop (`moe_comm.py:1203`): every row costs a
chain of dependent scalar gmem loads (`src_row`, `dst_slot`, `dst_rank`, plus
`seg` and `gate_idx` when gating) before its first copy can issue. Combine
suffers ~1.75x more than dispatch because its rows are narrower (N=4096 =
8 KB vs hidden=7168 = 14 KB), so the same fixed cost is amortised over fewer
bytes.

## The unlock: destination order is free

Nothing requires `slot = token * topk + j`. The staging layout is constrained
only by the local reduce, and the reduce is a **summation** — it can permute.

Assign destination slots **in source order** instead: rank r's rows for
`(expert e, home h)` occupy a contiguous block of h's staging buffer, base
offset computed on the host from the already-allgathered routing (so all ranks
agree without communicating).

Source rows are *already* contiguous per segment — `build_push_combine_arrays`
walks expert-major, then home ascending, then token order, and `src_row =
arange`. So making the destination contiguous too turns each segment into a
**contiguous -> contiguous** transfer.

Three things fall out:

1. Consecutive rows merge into one bulk copy instead of one copy per row.
2. `dst_slot[idx]` leaves the inner loop (`slot = base + i`), removing one of
   the dependent per-row loads that cause the low-CTA latency problem.
3. A GEMM output tile maps to a contiguous region of the destination, which is
   what makes tile-granular transfer expressible at all.

Cost: the reduce stops being `staging.view(tokens, topk, N).sum(1)` and becomes
an indexed segment sum over a host-built row->token map. Not optimised — a
one-line `index_add_` is enough to keep correctness checks honest, and it is
timed separately today (~0.15 ms, ~2% of the step).

## 1-D full-width beats 2-D column slices

The column dimension is the only thing that forces 2-D. Options considered:

- **Column slice (m,n) tile, 2-D.** A `(rows x tile_n)` sub-tile is strided
  (row stride N), so it needs `cp.async.bulk.tensor.2d` plus a tensormap per
  destination rank. ~52 KB per op at 16K/8-GPU.
- **Column slice, many small 1-D.** One op per row, 512 B each at tile_n=256 —
  *worse* than today's 8 KB. Rejected.
- **Full-width piece, 1-D.** `h` consecutive full rows are contiguous on both
  sides. One `cp.async.bulk` with a byte count: **no tensormap, no descriptor,
  no per-rank tensormap churn**. The existing `VarlenAllToAllKernel` already
  issues this exact instruction with `tx_count = row_bytes`; the change is a
  run list instead of a row list. **Chosen.**

### SMEM staging is the real bound on op size

Bulk copies must go global -> shared -> global (there is no global-to-global
bulk), so the **stage size bounds the transfer, not contiguity**. An early
estimate of ~832 KB per op (104 rows x 8 KB) was wrong for this reason.

Today: stage = `hidden` elems = 8 KB, 12 stages = 96 KB, comfortable in B200's
228 KB. At 4 rows/stage: 32 KB x 12 = 384 KB, which does **not** fit — depth
would drop to ~6 stages for 192 KB.

So the realistic gain is **8 KB -> ~32 KB per op (~4x)**, bought by trading
pipeline depth for transfer width, plus the per-row index loads leaving the
inner loop. The depth-vs-width trade is a genuine unknown and needs a
2-parameter sweep (`rows_per_stage` x `num_stages`) at 12/24/36/48 CTAs.

## Fragmentation: one GEMM tile can serve several ranks

An m-tile never spans two experts (grouped-GEMM tile offsets are a per-expert
cumsum). But within one expert, rows are ordered by **destination rank**, and
those blocks do not land on 128-row boundaries — so a single GEMM output tile
can hold rows bound for several peers, and its transfer splits into pieces.

Measured with uniform random routing (script in git history of this file's
commit; recomputable in ~1 s of numpy):

| tokens | world | experts | pieces | mean height | % whole 128 | KB/op @tile_n=256 |
|---|---|---|---|---|---|---|
| 16384 | 2 | 64 | 1071 | 122.4 | 91.1% | 61.2 |
| 16384 | 4 | 128 | 1128 | 115.7 | 80.3% | 57.9 |
| 16384 | 8 | 256 | 1262 | 103.9 | 62.1% | 52.0 |
| 16384 | 16 | 512 | 1515 | 86.5 | 35.2% | 43.2 |
| 4096 | 8 | 256 | 497 | 66.5 | 8.2% | 33.2 |
| 2048 | 8 | 256 | 365 | 45.2 | 0.0% | 22.6 |

Driver: rows per `(expert, home)` ~ `tokens * topk / num_experts` — 512 rows at
16K/8-GPU (4 whole m-tiles per block), but 64 rows at 2K tokens (half a tile,
so every tile splits). Fragmentation worsens as batch shrinks and world grows,
the same direction the overlap already struggles in. Even the worst row still
beats today's 8 KB op, so the shape holds.

Two caveats: this assumes **uniform random routing**, and real MoE routing is
skewed (that is what EPLB exists to fix) — the mean survives, the tail gets
worse. And at world=2 fragmentation is nearly absent, so a 2-GPU measurement
**overstates** how well this works at 8.

### Rejected: pad blocks to tile_m

Padding each `(expert, home)` block to a multiple of `tile_m` would give every
m-tile exactly one destination. It costs ~12% extra GEMM rows at 16K/8-GPU,
and at 2K tokens (64-row blocks) padding to 128 nearly **doubles** the GEMM.
Trading a 3-6x transfer win for a 12-100% compute loss is not close.

## Flag design

Two consistent choices, decided by whether transfers are full-width:

- **Full-width pieces (chosen):** flag per m-tile row block, `target =
  n_tiles`, i.e. what exists today. No epilogue change at all.
- **Per-(m,n) tile:** flag per GEMM output tile, `target == 1`. Flag space
  grows from `total_m_tiles` to `total_m_tiles * n_tiles`, and the epilogue's
  index becomes `(offsets[b] + m_tile) * n_tiles + n_tile`. With exactly one
  publisher per flag and target 1, the publish needs no atomic — a
  `st.release.sys` replaces the `red`.

Either way the flag stays **per GEMM output tile, never per comm task**: the
GEMM cannot know the home-boundary decomposition, which changes every step
with the routing. Several comm tasks sharing one flag is trivially correct
under `target == 1` and idempotent polling. Order the task list
`(expert, m-tile, n-tile, piece)` so flag-sharing tasks are adjacent and one
poll serves them all.

## Implementation sketch (not built)

1. Host: `build_push_combine_arrays(..., dst_order="token"|"source")`, plus a
   run list `(gate_flag_idx, src_row_begin, n_rows, dst_rank, dst_slot_begin)`
   merging consecutive rows that share a destination. Keep `"token"` the
   default so the currently-verified path is untouched.
2. Host: staging row->token map for the reduce; `index_add_`, unoptimised.
3. Kernel: extend `VarlenAllToAllKernel` to consume runs and chunk them by
   `rows_per_stage`. No 2-D, no tensormaps, no new copy shape. Name it
   `CombineBlockTMAKernel` if a distinct entry point is wanted.
4. Sweep `rows_per_stage x num_stages` at 12/24/36/48 CTAs.
5. Per-(m,n) 2-D tiles stay shelved unless time-to-first-ready-tile shows real
   earliness to capture. With `AlongN` raster all n-tiles of an m-tile
   complete back to back, so the expected earliness is small.

## Open questions

- Depth vs width in the stage budget (above).
- Run-length distribution under **skewed** routing, not uniform.
- Whether the indexed reduce really stays at ~0.15 ms.
- Whether the low-CTA gap is mostly the per-row index loads or something else;
  a cooperative metadata load (lane L loads row idx+L's indices, then
  `shuffle_sync` broadcasts) tests that hypothesis independently of this whole
  redesign, and is a much smaller change.
