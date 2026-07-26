# TilePipe: a concise guide

## The approach in one paragraph

Overlap MoE communication with the grouped GEMM that depends on it — without a
megakernel. Producer and consumer stay two ordinary kernels on two streams over
a (mostly) disjoint SM partition, synchronized through **counting semaphores in
symmetric memory**: `flag[e]` counts arrived tokens for local expert `e`; the
GEMM's AB-load warp waits `flag[e] == split_sizes[e]` before issuing any TMA
load for expert `e`, and skips the wait for every later tile of the same
expert. All scheduling intelligence is **host-side list construction**; the
kernels just walk precomputed lists and count.

## The five ingredients

1. **Counting semaphore** (`quack/tilepipe_sync.py`, `ExpertArrivalSemaphore`):
   producer publishes `atomic_add(flag, n, release, sys)` after its data
   stores; consumer polls from one elected lane with `atomic_add(flag, 0,
   acquire, sys)` + nanosleep, then `sync_warp` + `fence.proxy.async` (the
   fence is mandatory when the consumer reads via TMA — async proxy). Counters
   are arrival-order-agnostic, which is what makes every other choice free.
2. **Order alignment**: the consumer drains experts in a fixed order (varlen
   batch order), so the producer must complete them in that order. Send lists
   are expert-major / rank-minor with the destination rotated by source rank,
   so all W consumers are fed from t=0 (see tilepipe.md §2).
3. **Segment publish**: one release per (source, expert, dst) segment, not per
   token. Warps count privately into local `seg_done[seg]` (`acq_rel`); the
   warp whose add completes the segment issues the single remote release —
   cumulative over all warps' stores via the acquire→release chain.
4. **SM partition as a launch knob**: the persistent GEMM is capped with
   `max_active_clusters = num_sms - num_comm_sms (+ oversub)`; dispatch is
   enqueued FIRST on a high-priority stream so its CTAs win placement, and
   with oversubscription the GEMM's surplus CTAs backfill SMs as dispatch
   retires.
5. **Launch discipline** (`tilepipe/plan.py`, class `TilePipe`): every kernel
   warm-up-EXECUTED before overlap (lazy module load deadlocks against a
   spinning GEMM); `reset()` zeroes flags AND `seg_done`; the overlapped
   section is launches only — any host CUDA call that device-syncs deadlocks.

## Copy engines

- **SIMT** (`tilepipe_dispatch_kernel`): warp-cooperative 256-bit
  volatile/sys stores. Simple, ~20–30 GB/s per SM — needs ~30 SMs to
  saturate NVLink.
- **TMA bulk** (`moe_comm.DispatchTmaKernel`): per-worker producer/consumer
  warp pairs stream rows through private SMEM stage partitions
  (G2S gather → S2G push). Stage recycle waits only on the SMEM *read*;
  remote-write completion is a per-thread watermark (`cp.async.bulk` groups
  are per-thread — publishes must stay on the issuing thread) that defers the
  segment release, with `fence.proxy.async` before every release. Much higher
  GB/s per SM → fewer comm SMs → more GEMM.

## What generalizes

The semaphore is a *predicate object* handed to the consumer
(`sem.wait_warp(e, target)` in `gemm_sm100.py`) — swapping the class swaps
the readiness condition with zero consumer changes (EPLB: tokens AND weight
arrived; see tilepipe/docs/eplb_design.md). Weight rows are just big token
segments in the same work list. Files: `tilepipe.md` (design),
`tilepipe/docs/findings.md` (measured findings), `tilepipe/plan.py` (pipeline),
`tilepipe/dispatch_gemm.py` (driver), `tilepipe/moe_comm.py --test-tma-dispatch`
(kernel-level test/bench).

## Next: GEMM → combine (epilogue side)

Direction inverts: the GEMM becomes the producer, combine the consumer, and
the natural flag granularity is the GEMM's own M-tile (see plan).

## Design sketch: sync waits as layout algebra

Today every gate hand-computes its flag index (`batch_idx`; `rank_base[d] +
tile_offsets[b] + (slot - cu[b]) // TILE_M`; ...) on both the producer and
consumer sides — and the combine-test hang came exactly from the two sides
deriving *different* index spaces. The generalization: a semaphore is a flag
space plus ONE shared map from data coordinates to flag positions.

    LayoutSemaphore
      flags   : Int32[num_flags] (symmetric)          # the counters
      targets : Int32[num_flags] | scalar             # readiness thresholds
      L       : data_coord -> flag_idx                # THE contract
      wait(coord)         = poll  flags[L(coord)] >= targets[L(coord)]
      arrive(coord, n, r) = add   peer_r.flags[L(coord)] += n

`L` is where the algebra lives, and CuTe layouts express the interesting
structure natively:

- **Granularity is the kernel of `L`** (which coordinates collapse to one
  flag). A zero-stride mode aggregates: `L = (TILE_M, n_tiles):(0, 1)` on the
  row coordinate is tile-level gating ("every TILE_M rows share a flag");
  `(len_e, E):(0, 1)` is expert-level; identity is per-token. Switching wait
  granularity = swapping a layout, no kernel changes.
- **Producers are one more mode**: flag spaces per producer rank compose as
  `L' = (L, world):(1, tiles_per_rank)` — the rank offset the combine gate
  currently hand-codes.
- **Ragged (varlen) segments are the non-affine part**: per-expert bases from
  `cu_seqlens` are a gather, not a stride. So `L = gather ∘ affine`: a small
  host-precomputed index tensor for the ragged hop (today's `tile_offsets` /
  `flag_idx`) composed with an affine CuTe layout inside each segment. The
  object accepts either form; hand-built `flag_idx[t, j]` tables are the
  fully-gathered degenerate case.

Migration is mechanical: `ExpertArrivalSemaphore` is `LayoutSemaphore` with
`L = identity(batch)` and `targets = split_sizes`; the GEMM epilogue publish
is `arrive((b, m_tile), 1, r)` under the tile layout; the combine gate is
`wait((token, j))` under `gather(flag_idx)`. The payoff is the invariant the
hang violated: producer and consumer index spaces cannot diverge because
there is only one `L`, declared once on the host and handed to both kernels.
