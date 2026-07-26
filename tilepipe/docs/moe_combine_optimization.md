# Saturating NVLink with a Handful of SMs: Optimizing MoE Combine

*How a 250 GB/s all-to-all gather became an 843 GB/s kernel that runs on 32 SMs instead of 148 — and why the copy engine, not more warps, is what made it possible.*

---

## The 3.6× mystery

Modern MoE inference has two all-to-all communication steps per layer. **Dispatch** scatters each token to its top-`k` experts (potentially on other GPUs); **combine** gathers the expert outputs back and sums them. On a single B200 node with NVLink, our dispatch kernel comfortably saturated the fabric at **~900 GB/s**. Our first combine kernel, moving the *exact same volume of data*, managed **~250 GB/s**.

Same bytes, same GPUs, same NVLink. Why is the gather 3.6× slower than the scatter?

That question is the whole post. The answer takes us through Little's law, a null result that taught us more than a win, a reframing of what "fast" even means for a communication kernel, and finally the Tensor Memory Accelerator (TMA) — which let us hit line rate on **32 SMs instead of 148**, freeing the rest of the GPU for expert GEMMs.

All numbers below are from 4× B200, `hidden=7168` (fp32), `topk=8`, `num_tokens=2560` per rank, `world_size=4`.

---

## Reads are not writes

The asymmetry is physical, not a bug.

Dispatch **writes** to remote memory. A remote NVLink write is *posted*: the SM fires the store and moves on. The store is fire-and-forget; the SM never waits for it to land. You can keep issuing writes as fast as the store units accept them.

Combine **reads** from remote memory. A remote NVLink read is *latency-exposed*: the SM issues the load, then the data has to travel across the fabric and back — roughly **1.7 µs** round trip — before the value is usable. If the SM sits idle waiting, throughput collapses.

The governing equation for any latency-bound memory operation is Little's law:

```
bandwidth = bytes_in_flight / latency
```

Latency is fixed by the fabric (~1.7 µs). So the *entire game* for read bandwidth is **maximizing bytes in flight** — the number of outstanding requests × their size. Writes get bytes-in-flight for free (posted). Reads have to work for every byte.

Our 250 GB/s combine had almost no bytes in flight. It read one top-k contribution, waited for it, added it, read the next. One load outstanding at a time. Little's law was punishing us.

Keep this equation in mind — every optimization that follows is just a different term in it.

---

## Round 1: widen the loads, deepen the pipeline

Two independent levers on `bytes_in_flight`:

**Lever A — bigger transactions (the load width).** The original kernel used 128-bit vectorized loads. B200 (SM100) supports 256-bit global loads (`LD.256`). Doubling the transaction size doubles bytes-per-request. In CuTe-DSL this is just the copy atom's `num_bits_per_copy`:

```python
COPY_BITS = 256                      # B200 max; Hopper caps at 128
elems_per_copy = COPY_BITS // dtype.width
load_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=COPY_BITS)
```

256-bit loads need 32-byte alignment, so the peer pointers and output tensor are `.align(32)`.

**Lever B — memory-level parallelism (more outstanding requests).** The original reduction consumed each top-k load immediately:

```python
# latency-exposed: 1 load outstanding
accum = 0
for j in range(topk):
    accum += load(remote[j])   # issue, WAIT, add
```

Restructured to issue all top-k loads *before* consuming any of them:

```python
# all topk loads in flight, then reduce
frgs = [make_fragment_like(...) for _ in range(topk)]
for j in range(topk):
    cute.copy(load_atom, remote_slices[j], frgs[j])   # issue all
for j in range(topk):
    accum += frgs[j].load()                            # then consume
```

Now `topk = 8` loads are outstanding per thread instead of 1.

### The null result that taught us the most

Here is the honest data:

| Change | Combine BW |
|---|---|
| Baseline | ~250 GB/s |
| + MLP restructure (8 loads in flight) | **248 GB/s** (no change!) |
| + 256-bit loads | **568 GB/s** |

The MLP restructure — the change we were *sure* would help — did nothing on its own. The width change nearly doubled throughput.

Why? Because with narrow 128-bit loads consumed immediately, the bottleneck wasn't the *number* of outstanding requests, it was the *width* of each one plus the per-request overhead. The reduction was memory-width-bound, not MLP-bound. The two levers multiply in Little's law, but only once width was addressed did the depth start to matter.

The lesson: **a null result localizes the bottleneck.** It told us combine was transaction-bound, which pointed straight at the next move.

Finally, covering all 148 SMs (148 CTAs × 32 warps = 1024 threads/CTA, 2 CTAs/SM = full occupancy) took us to **~864 GB/s** — matching dispatch.

Problem solved. Except it wasn't.

---

## The reframing: peak bandwidth is the wrong target

We hit 864 GB/s by using **every SM on the GPU**. In a microbenchmark that looks like a win. In a real MoE layer it is a disaster.

Combine does not run in isolation. It overlaps with expert GEMMs — the whole point of a fast comm kernel is *communication/computation overlap*. If combine consumes all 148 SMs to saturate NVLink, there are zero SMs left for the GEMMs it is supposed to overlap with. This is precisely the insight behind DeepEP: **the metric that matters is not GB/s, it is GB/s per SM.** Saturate the link with as *few* SMs as possible, and hand the rest to compute.

By that metric our "864 at 148 SMs" was **5.8 GB/s/SM** — SM-*inefficient*. We had optimized the wrong number.

So we reset the scoreboard. New objective: reach line rate at the minimum SM count. And that objective immediately runs into a wall that no amount of register-kernel tuning can break.

---

## Why the register path can't be SM-efficient

Apply Little's law *per SM* this time. Per-SM read bandwidth is `bytes_in_flight_per_SM / latency`. So: how many bytes can one SM keep outstanding with register loads?

Two hard caps, neither of which is warp count:

1. **MSHRs (Miss Status Handling Registers).** Every outstanding global load occupies an MSHR entry until the data returns. An SM has on the order of a few hundred of them. Once they are full, *the SM cannot issue another load* — it stalls, no matter how many warps are ready. Ballpark: ~256 outstanding × 128 B cache line ≈ 32 KB in flight → ~19 GB/s/SM at 1.7 µs.

2. **Register-file landing space.** Each in-flight 256-bit load needs 8 registers to receive its result, held from issue to consume. With the top-k MLP restructure holding 8 loads per thread, that's ~64 registers/thread *just for the load fragments* — which is why the register kernel can't even reach full 64-warp occupancy. Register pressure caps it first.

Now the structural killer. Suppose you want to concentrate more warps onto *fewer* SMs to push per-SM bytes-in-flight up. **You can't:**

- Max threads per CTA is 1024 = 32 warps. You cannot put more warps in a block.
- To co-resident 2 CTAs (64 warps) on one SM, you must launch more CTAs than SMs — but then the scheduler *spreads* them across all 148 SMs before doubling any up.

There is no launch configuration that says "run at full occupancy on just 32 SMs." Register-kernel parallelism is inherently **spread**, not **concentrated**. The evidence, sweeping CTA count at max warps:

| CTAs (= SMs) | Reg combine BW |
|---|---|
| 20 | 570 |
| 32 | 694 |
| 48 | 797 |
| 80 | 864 (peak) |
| 148 | 864 |

The register kernel needs ~80 SMs to reach peak. It physically cannot do better, because the per-SM ceiling is set by MSHRs and registers — shared resources the copy path is bound by.

To break that ceiling we need a way to keep bytes in flight *without* consuming MSHRs or registers. That is exactly what the copy engine is for.

---

## TMA bulk copy: offloading the gather to the copy engine

`cp.async.bulk` issues an asynchronous bulk transfer that the **Tensor Memory Accelerator** — a dedicated copy engine — carries out into shared memory. The critical property: a bulk copy of a 14 KB tile is *one descriptor-driven transfer managed by the copy engine*. It does not sit in an MSHR. It does not tie up registers waiting for a return value. The data lands in smem and signals an mbarrier when done.

This inverts the per-SM math. Instead of being capped at ~32 KB of MSHR-limited in-flight bytes, one SM can have its **entire shared-memory pipeline** in flight — 8 stages × 14 KB = 112 KB — driven by the copy engine, with the SM's load/store units and register file almost untouched.

### The kernel: warp-specialized producer/consumer

The design is a classic TMA pipeline (mirroring `all_reduce_tma.py`), specialized to the combine gather:

- **Producer (warp 0, one thread issues):** for each `(token, chunk, expert-j)`, resolve which peer owns the expert row at runtime, and fire a bulk copy of that `HCHUNK`-element slice into the next smem stage.
- **Consumer (128 threads):** wait for each staged chunk, accumulate it into registers, and after all `topk` chunks land, write the summed output row.

The pipeline is a `PipelineTmaAsync` with `NUM_STAGES` smem buffers and mbarriers. The producer arms `expect_tx = tile_bytes`; the bulk copy signals the barrier on completion; the consumer drains.

```python
# Producer: stream topk chunks per token into smem via raw 1D bulk copy.
bulk_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), dtype)
while token < num_tokens:
    for k in range(num_chunks):
        for j in range(topk):
            rank_j  = topk_indices[token, j] // experts_per_rank
            slot_j  = scatter_idx[token, j]
            tile_id = slot_j * num_chunks + k
            tma_pipeline.producer_acquire(prod)          # arms expect_tx
            s_tile = staged[None, prod.index]
            for r in range(world_size):                  # constexpr peer select
                if rank_j == r:
                    g_tile = zipped_divide(peer_flat[r], tiler)[None, tile_id]
                    with cute.arch.elect_one():
                        cute.copy(bulk_atom, g_tile, s_tile,
                                  mbar_ptr=tma_pipeline.producer_get_barrier(prod))
            tma_pipeline.producer_commit(prod)
            prod.advance()
```

The consumer side is an ordinary strided smem→register reduction, one store per output chunk — the *write* to the local output row is cheap and local, so it stays a plain store.

### Tensor-tile vs. raw 1D bulk: match the tool to the access pattern

Our first TMA version used `CopyBulkTensorTileG2SOp` — the descriptor-based *tensor-tile* variant, copied wholesale from the all-reduce tutorial. But our access pattern is trivially simple: each tile is a **fully contiguous run** of `HCHUNK` elements at a computed offset in the flattened peer buffer. There is no 2D tiling, no swizzle. That is exactly what raw `cp.async.bulk` (`CopyBulkG2SOp`) is for.

Switching from the tensor-tile atom to the raw 1D form dropped the whole `make_tiled_tma_atom` / `tma_partition` machinery — the producer just slices `peer_flat[r][tile_id*HCHUNK : +HCHUNK]` — and gained **+13%** (523 → 591 GB/s at 20 CTAs) by shedding the descriptor address-generation overhead. Lesson: don't carry tensor-descriptor machinery for a contiguous copy.

---

## Autotuning for SM efficiency

The TMA kernel has two coupled knobs under a fixed shared-memory budget (~227 KB/SM on B200): the tile size `HCHUNK` and the pipeline depth `NUM_STAGES`, with `smem = HCHUNK × NUM_STAGES × 4B`. Bigger tiles amortize the per-tile mbarrier round-trip; more stages give more top-k overlap. They trade against each other for the same smem. So we built a lightweight in-process autotuner: sweep configs, **validate each against the reference before timing** (a fast-but-wrong config can never win), and report GB/s per SM.

**Phase 1 — tile config at fixed 32 CTAs:**

| HCHUNK | stages | smem | GB/s |
|---|---|---|---|
| 1792 | 16 | 112 KB | 820 |
| 3584 | 8 | 112 KB | **843** |
| 7168 | 4 | 112 KB | 814 |
| 1792 | 8 | 56 KB | 660 |
| 3584 | 4 | 56 KB | 643 |
| 7168 | 2 | 56 KB | 586 |

Two clear signals. At a fixed 112 KB budget the three configs cluster within 4% — tile size past ~14 KB barely matters. But halving the budget (fewer stages) costs ~25%. **Pipeline depth dominates tile size**: stages buy the top-k overlap that keeps the copy engine fed. Winner: `(3584, 8)`.

**Phase 2 — CTA sweep at `(3584, 8)`:**

| CTAs (= SMs) | GB/s | GB/s/SM |
|---|---|---|
| 8 | 243 | 30.4 |
| 16 | 480 | 30.0 |
| 24 | 697 | 29.1 |
| **32** | **843** | **26.3** |
| 48 | 866 | 18.0 |
| 64 | 873 (peak) | 13.7 |
| 148 | 863 | 5.8 |

The curve tells the whole story. In the linear region every SM contributes a flat **~30 GB/s** — that is the sustained per-SM throughput of one producer warp feeding the copy engine. Saturating NVLink (~873 GB/s) therefore *fundamentally* needs ~29 SMs. The knee is at **32 CTAs → 843 GB/s (96% of peak)**.

---

## Register vs. TMA, side by side

| | Register (max warps) | TMA bulk |
|---|---|---|
| Bytes in flight / SM | ~32 KB (MSHR-bound) | ~112 KB (smem pipeline) |
| Per-SM throughput | ~5–11 GB/s/SM | ~30 GB/s/SM |
| SMs to reach line rate | ~80 | **32** |
| Peak BW | 864 | 873 |

Both reach the same peak bandwidth. The difference is *how many SMs it costs*. TMA reaches line rate at **32 SMs vs the register kernel's ~80** — a ~2.5× improvement in SM efficiency, which in an MoE layer translates directly into ~48 more SMs available for expert GEMMs during the overlap window.

And crucially, the register kernel *cannot close this gap* by adding warps. Its ceiling is the MSHR/register-file resource, and its parallelism is structurally spread across SMs rather than concentrated. The copy engine is the only mechanism that breaks the per-SM read ceiling.

---

## Takeaways

1. **Little's law is the whole game for latency-bound comm.** Reads need bytes in flight; writes don't. Every optimization here — load width, MLP depth, SM count, TMA — is a different term in `BW = bytes_in_flight / latency`. Derive from the equation, don't guess.

2. **A null result localizes the bottleneck.** The MLP restructure doing nothing told us we were width-bound, which was more useful than a win.

3. **Optimize the right metric.** For a communication kernel that overlaps with compute, peak GB/s is a vanity number. GB/s *per SM* is what buys you overlap. We nearly shipped an SM-inefficient kernel that looked great in a microbenchmark.

4. **Know your per-SM ceilings.** Register loads are capped by MSHRs and register-file landing space — a few tens of KB in flight per SM. When a latency-bound read needs more bytes in flight than that, reach for the copy engine (`cp.async.bulk`): it sustains a full smem pipeline in flight without touching MSHRs or registers.

5. **Match the copy primitive to the access pattern.** Contiguous gather → raw 1D bulk, not tensor-tile descriptors. Free 13%.

The final combine kernel saturates NVLink on 32 of 148 SMs. The other 116 go to the experts.
