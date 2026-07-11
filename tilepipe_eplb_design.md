# TilePipe-EPLB: fused dispatch + expert-weight transfer — design notes

Companion to `tilepipe.md` ("Implement Expert Parallel Load Balancing") and
`tilepipe_findings.md`. Records the design discussion, the tradeoffs weighed,
and the implementation plan. Code: `examples/distributed/tilepipe_eplb.py`.

## 1. Idea

When EPLB changes expert placement, the rank that will now compute an expert
may not hold its weights — they must be fetched from the previous owner. This
weight transfer is just a **second producer** feeding TilePipe's per-expert
semaphores: an expert is ready when all its tokens *and* all its weight bytes
have arrived. No new synchronization primitive is needed; that is the point
the EPLB extension exists to demonstrate.

The GEMM overlaps with both transfers by computing local-weight experts while
tokens and weights for transferred experts stream in — transferred experts go
last in the GEMM's expert order.

## 2. Design decisions and tradeoffs

### Weight readiness: separate flag (chosen) vs folded counter

- **Chosen: a separate weight-ready flag per transferred expert** (0/1,
  target 1), living in the same physical flag array as the token counters
  (`[0, epr)` = token arrival counters, `[epr, ...)` = weight flags) so the
  kernel needs no second flag table. The GEMM gate polls both.
- Alternative considered: fold weights into the token counter by initializing
  `flag[e] = -weight_chunks[e]`, making the existing `flag[e] == len_m(e)`
  gate correct with **zero** GEMM changes. Elegant, but rejected for
  explicitness: a separate flag keeps token and weight progress observable
  independently (debuggability) and keeps the weight protocol testable
  standalone.

### Push (chosen) vs pull

- **Push** — the owner writes weights into the consumer's staging buffer:
  - P2P **writes are posted**; reads expose round-trip latency and need deep
    outstanding-load ILP per SM to hide. Our 256-bit volatile-store path is
    *measured* at ~19–31 GB/s per SM; pull would be a new, unmeasured path.
  - One protocol everywhere: same `atomic_add(flag, v, release, sys)` publish
    as token dispatch, same local-counter completion trick.
  - The pull argument ("consumer knows its own GEMM priority") evaporates
    because job order is host-precomputed anyway — the host sorts each
    sender's jobs by the destination's GEMM order.
- Recorded weakness of push: a single hot owner replicating one expert to
  many ranks serializes on its own comm SMs. Revisit if we do wide
  replication; irrelevant at world = 2.

### Arbitrary EPLB configurations: the job list

The kernel never knows EPLB's policy. The host lowers any placement change
(swap, replication, migration — e.g. DeepSeek EPLB's `phy2log` table) to a
flat job list:

    job[j] = (src_weight_slot, dst_rank, dst_weight_slot, dst_flag_idx)

A DeepSeek-EPLB adapter is therefore a pure host-side function producing
these arrays; nothing below it changes.

### One weight per SM vs partitioned across SMs (chosen)

Decisive numbers (DSv3 gate+up: N=4096, K=7168, bf16 → **58.7 MB/weight**;
one B200 SM sustains ~20–30 GB/s over NVLink):

- One SM per weight: every weight takes ~2 ms — **2x the entire GEMM** — and
  all weights complete late together. Worst shape for pipelining.
- Partitioned across C SMs, jobs processed in destination-GEMM order: first
  weight ready at ~58.7 MB / (C x 25 GB/s) ≈ **0.15 ms at 16 SMs**, then a
  staircase — exactly what the gated GEMM wants.

The complexity is already paid: partitioning reuses the segment protocol from
token dispatch verbatim (per-warp local counting, single cumulative release
by the completing warp). Chunk = one row of K, so chunk indices are pure
arithmetic — no per-chunk metadata.

### Fused kernel structure: unified work list (chosen) vs warp split

Key observation: **a weight row is a K-vector, identical to a token's hidden
vector** — one weight = N token-equivalents (~32 average experts' token
traffic at DSv3 sizes: 4096 tok x topk 8 / 256 experts ≈ 1.8 MB/expert vs
58.7 MB/weight). Both transfers are streams of the same 14 KB row-copy unit.

So there is no fusion step at all: the dispatch kernel walks **one flat,
segment-sorted unit list**, where a segment is either a (source, expert, dst)
token group or one weight push, described uniformly by per-segment metadata
(`seg_dst`, `seg_is_weight` buffer selector, `seg_flag`, `seg_pub` publish
value — segment size for token counters, 1 for weight flags). Scheduling
becomes a pure host-side ordering problem, which is the TilePipe thesis.

Warp split (dedicated weight warps from t=0) was considered and dropped: it
adds a static ratio to tune, idles weight warps when transfers finish early,
and cannot express deadline-aware interleaving. The unified list strictly
subsumes it.

## 3. Ordering theory: what order maximizes overlap

Model comm → GEMM as a two-machine flow shop: expert `e` needs `D(e)` bytes
delivered (tokens + 58.7 MB if transferred) before the GEMM spends `G(e)`
compute on it (∝ token count). **Johnson's rule** gives the optimal order:

1. Comm-light experts (`D/B < G` — all local-weight experts) first, in
   **increasing D**: cheapest dependencies released first, GEMM never starves
   early.
2. Comm-heavy experts (`D/B > G` — all transferred experts) last, in
   **decreasing G**: hottest transferred expert first within the tail group.

Corollaries:
- The `.md`'s "tokens of no-transfer experts sent first / transferred experts
  computed last" is the coarse form of this rule — but as a *serial* order
  ("all tokens, then weights") it would be wrong: it delays a transferred
  expert to `T_tokens + T_weight` instead of `max(...)`. The correct schedule
  emits each expert's tokens + weight together at its position in the GEMM
  order (**earliest-deadline-first**).
- EPLB moves *hot* experts by definition; hot experts have the largest `G`
  and best tolerate a weight transfer in front of them. The design improves
  on realistic workloads.
- **SwiGLU asymmetry**: gate+up carries 2x the FLOPs of the down projection,
  so the overlap window we target is the fattest phase in the layer. For the
  later `+gemmdown` variant note W2 (K x I) is half W1's bytes but the down
  GEMM is also half the FLOPs — the comm/compute ratio is unchanged, so the
  win comes from W2's *later deadline* (down-GEMM order): schedule W2
  segments into the gate+up window after the W1/token segments. The unified
  EDF list expresses this with no new mechanism.

### Round-robin across destinations (multi-rank)

With W > 2 ranks, a 58.7 MB weight push to one destination is a ~30-expert
burst; strictly serving one destination starves every other consumer for the
whole burst — the §2 (tilepipe.md) starvation argument recurring at chunk
granularity. Fix: conceptually W per-destination EDF queues, drained a few
chunks at a time round-robin. The completion counters are arrival-order-
agnostic, so slicing a weight into interleaved sub-runs needs **no kernel
changes** — only list construction. At world = 2 this degenerates to plain
EDF, so we validate now and the ordering generalizes.

## 4. Implementation plan and status

1. **[DONE] Unified dispatch kernel** (`unified_dispatch_kernel`): flat unit
   list + per-segment metadata; token and weight segments share the copy and
   publish path. Compiles on SM100.
2. **[DONE] Job-list lowering** (`weight_jobs_to_segments`) + weights-only
   MVP driver: synthetic EPLB swap (each rank pushes `--transfer` weights to
   the peer), bit-exact correctness against deterministically regenerated
   references, bandwidth sweep reporting GB/s and per-weight time
   (= time-to-first-ready-weight, the number that sets the comm-SM budget).
   Awaiting the 2-GPU run:
   `torchrun --nproc-per-node 2 examples/distributed/tilepipe_eplb.py`
3. **[TODO] GEMM gate**: add the `wflag` poll next to the existing token-flag
   wait in `gemm_sm100.py` (same lane-0 acquire/sys poll + `ready_batch_idx`
   caching; local-weight experts get their wflag preset to 1 so the gate is
   uniform).
4. **[TODO] Combined token+weight list**: `build_unified_list(routing,
   eplb_jobs, gemm_order)` implementing Johnson ordering + per-destination
   EDF (+ round-robin slicing for W > 2) — pure numpy, unit-testable without
   GPUs. Permute the varlen batch slots (weights buffer, cu_seqlens, flag
   indices) so transferred experts sit last; GEMM code untouched by the
   permutation.
5. **[TODO] End-to-end**: overlapped dispatch+weights+GEMM correctness and
   benchmark vs serial and vs no-EPLB TilePipe on 2 GPUs; then the
   `+gemmdown` variant (W2 transfer scheduled into the gate+up window).
6. **[LATER] DeepSeek EPLB adapter**: lower a real `phy2log`/replica table to
   the job list; wide-replication fan-out may motivate revisiting push-only.

## 5. Open questions

- The EPLB section of `tilepipe.md` ends mid-sentence ("Note we also…") —
  possible missing constraint.
- Whether the GEMM tile scheduler's approximate (not strict) batch order
  weakens the Johnson-order argument enough to matter at small expert counts.
- Real EPLB arguments/config format: to be pulled from DeepSeek EPLB when we
  integrate (job-list interface is designed to absorb it).
