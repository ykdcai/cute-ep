# TilePipe — implementation findings (Phase 1: gated grouped GEMM)

## What works

`quack_gemm(..., cu_seqlens_m=..., expert_ready_flags=flags, max_active_clusters=N)` now
gates the SM100 varlen_m grouped GEMM on per-expert token-arrival counters.
Verified on B200: numerically correct, waits key on *per-expert* counters (releasing
experts in reverse order works), partial counts do not release an expert, and the
ungated path is untouched. Test: `tests/test_gemm_tilepipe_gating.py`.

## Key findings

- **Where to gate**: the AB-load warp of `gemm_sm100.py` knows `batch_idx` before
  issuing any TMA load, so the wait goes there — one poll per batch transition, zero
  cost once an expert is released. Wait target is `cu_seqlens_m[b+1] - cu_seqlens_m[b]`,
  so no extra metadata is needed beyond the flags array.
- **Poll from one lane only**: lane 0 does `atomic_add(flag, 0, sem="acquire", scope="sys")`
  in a `nanosleep(256)` loop; `sync_warp()` holds the other 31 lanes. All-lane polling
  would 32x the sys-scope atomic traffic at L2 for no benefit.
- **Memory ordering**: the producer must publish with `atomic_add(flag, n, sem="release",
  scope="sys")` *after* its data writes. A plain store is a weak op — the PTX memory
  model does not guarantee an acquire load pairs with it. `fence.acq_rel.sys`
  (`__threadfence_system`) + relaxed atomics is a valid alternative but orders all prior
  traffic and is not cheaper than a release atomic; it becomes necessary only if the
  data writes go through the async proxy (TMA), which needs `fence.proxy.async` anyway.
- **Streams**: quack GEMM launches on torch's current stream (TVM-FFI env stream), so
  two `torch.cuda.Stream`s give the GEMM/comm split; `max_active_clusters` caps the
  persistent GEMM so SMs remain for the comm kernel (SM partition = launch-time knob).
- **Deadlock trap (test/host side)**: while a persistent kernel spins on flags, *any*
  host CUDA call that device-syncs (allocation, pageable scalar H2D like `flags[e] = int`)
  deadlocks the process. Everything that touches flags mid-flight must run on-device.
  The test uses a 1-warp CuTe trickle kernel with release/sys atomic adds — the same
  protocol the dispatch kernel will use.
- `nanosleep`'s effective delay is far below nominal; treat it as a backoff hint, never
  as a timing primitive.

## Phase 2: dispatch kernel (`examples/distributed/tilepipe.py`) — works on 2 GPUs

Expert-major / rank-minor send list (host-precomputed, destination rotated by source
rank), warp-cooperative 256-bit NVLink copies into the receiver's expert-grouped recv
buffer, flags raised per **(source, expert) segment**, not per token:

- **Publish granularity**: the GEMM can't start expert `e` before all of
  `split_sizes[e]` arrived, so per-token flag increments buy zero latency — they only
  multiply contended sys-scope NVLink atomics. Instead each warp copies its tokens of
  the current segment back-to-back (the segment-sorted send list keeps a warp's strided
  walk segment-contiguous), then does ONE local `atomic_add(seg_done[seg], count,
  acq_rel, sys)`; the warp whose increment completes the segment issues the single
  remote `atomic_add(peer_flag[e], seg_size, release, sys)`. The acquire→release chain
  through `seg_done` makes that release cumulative over every warp's remote stores.
  Remote flag atomics: `world × experts_per_rank` total (vs tokens × topk).
  A `--publish token` mode is kept for A/B comparison.
- **`seg_done` is state**: it must be zeroed between iterations alongside the flags —
  a missed reset over-publishes and releases the GEMM early (silent corruption).
- **Second lazy-init deadlock trap**: the FIRST launch of a compiled kernel does
  module load — host-side CUDA work that deadlocks against an already-spinning gated
  GEMM. Every kernel in the pipeline must be warm-up-executed (not just compiled)
  before the overlapped phase. Verified: dispatch-only flags land exactly on
  `split_sizes`; overlapped dispatch+GEMM correct (recv bit-exact, GEMM rel err ~3e-3).

## Next

- Phase 3: benchmark results (serial vs overlapped, sweep `num_comm_sms`); then
  request full-node access once 2-GPU numbers look good.
