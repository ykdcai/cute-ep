"""FlashInfer baseline for the TilePipe GEMM -> combine pipeline.

Serial (non-overlapped) reference point for ``tilepipe/gemm_combine.py``,
measured on the SAME problem: DeepSeek-V3 shapes, the same routing, the same
per-expert varlen GEMM, and the same final answer checked against the same
reference.

The two stacks split the work differently, and the split is the interesting
part of the comparison:

  TilePipe   grouped GEMM (one row per (token, expert)) -> push-combine sends
             all tokens*topk rows to the home rank -> local sum over topk.
  FlashInfer grouped GEMM (identical layout) -> LOCAL pre-reduce of a token's
             expert rows into one row per (token, dst rank) -> MoeAlltoAll
             combine pulls one row per (token, src rank) and sums across ranks.

So FlashInfer moves fewer bytes (deduplicated per rank, not per expert) but
pays a local pre-reduce first; TilePipe moves every expert row but its local
reduce happens after the transfer. Both local reduces are timed separately
here and in gemm_combine.py, so either accounting is reconstructable.

The MoeAlltoAll combine also cannot run standalone: it consumes the send
indices produced by the matching dispatch, so every iteration dispatches
first. The dispatch is timed on its own (it is the baseline for the other
half of the pipeline) and is NOT part of the serial GEMM->combine total.

Run (2 GPUs, defaults are DeepSeek-V3 shapes):
    torchrun --nproc-per-node 2 tilepipe/baselines/bench_flashinfer_gemm_combine.py
"""

import argparse
import datetime
import functools
import json
import os

import numpy as np
import torch
import torch.distributed as dist

from flashinfer.comm.comm_backend import TorchDistBackend
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlConfig
from flashinfer.comm.trtllm_moe_alltoall import (
    MoeAlltoAll,
    moe_a2a_get_workspace_size_per_rank,
)
from flashinfer.grouped_mm import grouped_mm_bf16

from tilepipe.plan import build_combine_metadata

# Stage prints are hang diagnostics — they must not sit in a stdio buffer.
print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Synthetic operands. Byte-identical to tilepipe/gemm_combine.py's, so the GEMM
# and the reference see the same numbers in both benchmarks.
# ---------------------------------------------------------------------------

def rank_tensor(seed, shape, device, scale=1.0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=torch.float32)
            * scale).to(torch.bfloat16)


def expert_weight(d, e, n, k, device):
    # Per-(rank, expert) seed so peers can rebuild ONE expert at a time: a
    # whole-rank randn((epr, n, k)) needs an fp32 temp of ~4 GB at DSv3 shape.
    return rank_tensor(900 + d * 4096 + e, (n, k), device, scale=k ** -0.25)


def build_weights(d, epr, n, k, device):
    w = torch.empty((epr, n, k), dtype=torch.bfloat16, device=device)
    for e in range(epr):
        w[e] = expert_weight(d, e, n, k, device)
    return w


def dist_init():
    """Plain torchrun bootstrap. Deliberately NOT tilepipe's NVSHMEM init: the
    FlashInfer path uses MNNVL (CUDA VMM + POSIX-fd handle exchange over
    torch.distributed), so nothing here needs a symmetric heap."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run(args, a2a, rank, world_size, device):
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    epr = args.experts // world_size
    n, k, T = args.n, args.hidden, args.tokens
    invalid_expert = args.experts  # sanitize marker for empty recv slots
    torch.manual_seed(42 + rank)

    if rank == 0:
        print(f"FlashInfer GEMM->combine: tokens/rank={T} K={k} N={n} "
              f"experts={args.experts} (local {epr}) topk={args.topk} "
              f"world={world_size} SMs={num_sms}")

    # --- Routing, shared with every rank so references can be rebuilt ---
    topk_indices = torch.randint(
        0, args.experts, (T, args.topk), dtype=torch.int32, device=device)
    all_topk_t = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    dist.all_gather(all_topk_t, topk_indices.contiguous())
    all_topk = np.stack([t.cpu().numpy() for t in all_topk_t])

    meta = build_combine_metadata(all_topk, args.experts, rank, world_size)
    total_m = meta["rank_rows"][rank]
    m_indptr = torch.from_numpy(meta["cu_seqlens"]).to(device)
    print(f"[rank {rank}] metadata: total_m={total_m} "
          f"(rows/rank={meta['rank_rows']})")

    # --- GEMM operands (deterministic per rank for the reference) ---
    weights_bytes = epr * n * k * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] allocating weights {weights_bytes / 1e9:.2f} GB "
          f"(free: {free_bytes / 1e9:.2f} GB)")
    A = rank_tensor(500 + rank, (total_m, k), device, scale=k ** -0.5)
    weights = build_weights(rank, epr, n, k, device)
    D = torch.zeros((total_m, n), dtype=torch.bfloat16, device=device)

    # --- Dispatch payloads. The hidden states are the real thing a layer
    # dispatches; the token-id payload is what lets us invert the (racy,
    # atomic-assigned) recv slot order back to source token indices, which the
    # pre-reduce needs to know where each expert row belongs.
    hidden_in = rank_tensor(300 + rank, (T, k), device, scale=k ** -0.5)
    token_ids = torch.arange(T, dtype=torch.int32, device=device).view(T, 1)

    # --- Host-side (token, expert) -> D-row map, per source rank. For every
    # (s, t, j) whose expert lives on THIS rank, pair_row is the row of D
    # holding that expert's output and pair_tok/pair_src say which source
    # token it must be folded into.
    pair_src, pair_tok, pair_row = [], [], []
    for s in range(world_size):
        meta_s = (meta if s == rank
                  else build_combine_metadata(all_topk, args.experts, s, world_size))
        t_idx, j_idx = np.nonzero(meta_s["src_rank"] == rank)
        pair_src.append(np.full(len(t_idx), s, dtype=np.int64))
        pair_tok.append(t_idx.astype(np.int64))
        pair_row.append(meta_s["scatter"][t_idx, j_idx].astype(np.int64))
    pair_src = torch.from_numpy(np.concatenate(pair_src)).to(device)
    pair_tok = torch.from_numpy(np.concatenate(pair_tok)).to(device)
    pair_row = torch.from_numpy(np.concatenate(pair_row)).to(device)
    assert pair_row.numel() == total_m, (
        f"pair count {pair_row.numel()} != total_m {total_m}")

    # --- Byte accounting. FlashInfer deduplicates per destination RANK, so a
    # token hitting 3 experts on the same rank crosses the wire once; TilePipe
    # sends one row per (token, expert). Report both so the GB/s numbers are
    # never compared against mismatched volumes.
    dst_ranks = all_topk[rank] // epr                       # [T, topk]
    pairs_per_token = np.array([len(np.unique(r)) for r in dst_ranks])
    my_pairs = int(pairs_per_token.sum())                   # (token, rank) pairs
    tp_rows = T * args.topk                                 # TilePipe row count
    dispatch_bytes_per_token = k * 2 + args.topk * 4 + args.topk * 4 + 4
    dispatch_bytes = my_pairs * dispatch_bytes_per_token
    combine_bytes = my_pairs * n * 2
    if rank == 0:
        print(f"volume/rank: {my_pairs} (token,rank) pairs vs {tp_rows} "
              f"(token,expert) rows TilePipe moves ({tp_rows / my_pairs:.2f}x); "
              f"dispatch {dispatch_bytes / 1e6:.0f} MB, "
              f"combine {combine_bytes / 1e6:.0f} MB")

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    def do_dispatch():
        return a2a.dispatch(
            topk_indices, [hidden_in, topk_indices, token_ids], T,
            invalid_token_expert_id=invalid_expert, expert_id_payload_index=1)

    def slot_index(recv):
        """Invert the dispatch's recv order: slot_of[(s, t)] -> flat payload row.

        recv slot (s, i) holds source rank s's token recv_ids[s, i]; padding
        slots are the ones the sanitize pass marked with `invalid_expert`.
        The assignment comes from atomics, so it is rebuilt after every
        dispatch rather than cached.
        """
        recv_experts, recv_ids = recv[1], recv[2][..., 0]
        valid = (recv_experts != invalid_expert).any(dim=-1)     # [ep, T]
        s_i, i_i = valid.nonzero(as_tuple=True)
        inv = torch.full((world_size, T), -1, dtype=torch.int64, device=device)
        inv[s_i, recv_ids[s_i, i_i].long()] = i_i
        slot = inv[pair_src, pair_tok]
        assert int(slot.min()) >= 0, "dispatch dropped a token this rank owns"
        return pair_src * T + slot

    def pre_reduce(payload, slot_flat):
        # A token's rows on THIS rank (up to min(topk, epr) experts) collapse
        # into the single row the combine will pull. This is the work TRT-LLM
        # folds into its finalize kernel; it is local, so it is timed on its own.
        # index_add_ needs its targets zeroed first, but only the rows the
        # combine actually reads -- zeroing the whole [ep, T, N] workspace view
        # would charge this baseline for padding slots nobody touches.
        # Accumulation is in bf16 (the payload dtype); summing <= topk rows of
        # like magnitude keeps that well inside the 3e-2 check below.
        flat = payload.view(-1, n)
        flat.index_fill_(0, slot_flat, 0)
        flat.index_add_(0, slot_flat, D)

    # --- Correctness: identical reference to gemm_combine.py's -- for each of
    # my tokens, the fp32 sum over its topk expert outputs, wherever they live.
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"[rank {rank}] building reference (free: {free_bytes / 1e9:.2f} GB)...")
    ref = torch.zeros((T, n), dtype=torch.float32, device=device)
    scatter_l = torch.from_numpy(meta["scatter"]).to(device).long()
    src_rank_t = torch.from_numpy(meta["src_rank"]).to(device)
    for d in range(world_size):
        A_d = A if d == rank else rank_tensor(
            500 + d, (meta["rank_rows"][d], k), device, scale=k ** -0.5)
        meta_d = (meta if d == rank
                  else build_combine_metadata(all_topk, args.experts, d, world_size))
        D_d = torch.zeros((meta["rank_rows"][d], n), dtype=torch.float32, device=device)
        cu = meta_d["cu_seqlens"]
        for e in range(epr):
            lo, hi = int(cu[e]), int(cu[e + 1])
            if hi > lo:
                W_e = expert_weight(d, e, n, k, device)
                D_d[lo:hi] = A_d[lo:hi].float() @ W_e.float().T
                del W_e
        mask = src_rank_t == d
        ref += (D_d[scatter_l.clamp(0, meta["rank_rows"][d] - 1)]
                * mask.unsqueeze(-1)).sum(dim=1)
        del D_d, A_d
        torch.cuda.empty_cache()
        print(f"[rank {rank}] reference: peer {d} done")
    torch.cuda.synchronize()

    # --- Warm-up + correctness. Everything must EXECUTE once (cuDNN plan
    # build, MNNVL handle setup, JIT module load) before it is timed.
    barrier()
    recv = do_dispatch()
    slot_flat = slot_index(recv)
    grouped_mm_bf16(A, weights, m_indptr, out=D)
    payload = a2a.get_combine_payload_tensor_in_workspace(T, n, torch.bfloat16)
    # pre_reduce writes THROUGH this view into the workspace the peers read, so
    # it must be a real view and not a reshape-copy.
    assert payload.is_contiguous(), "combine payload view is not contiguous"
    pre_reduce(payload, slot_flat)
    out = a2a.combine(payload, T, payload_in_workspace=True)
    barrier()

    rel = ((out.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
    ok = rel < 3e-2
    print(f"[rank {rank}] serial GEMM->combine: rel_err={rel:.2e} "
          f"{'OK' if ok else 'FAIL'}")
    ok_t = torch.tensor([ok], dtype=torch.int32, device=device)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if not bool(ok_t.item()):
        raise SystemExit(f"[rank {rank}] correctness FAILED")

    if not args.benchmark:
        return None

    # --- Benchmark. One timed region per iteration, in launch order, so the
    # stage times sum to the serial total by construction. The dispatch is
    # timed separately BEFORE the region because the combine consumes its send
    # indices and the slot inversion needs its result on the host.
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    samples = np.zeros((4, args.iters))
    for it in range(args.warmup + args.iters):
        barrier()
        ev[0].record()
        recv = do_dispatch()
        ev[1].record()
        torch.cuda.synchronize()
        slot_flat = slot_index(recv)
        payload = a2a.get_combine_payload_tensor_in_workspace(T, n, torch.bfloat16)
        barrier()

        ev[2].record()
        grouped_mm_bf16(A, weights, m_indptr, out=D)
        ev[3].record()
        pre_reduce(payload, slot_flat)
        ev[4].record()
        a2a.combine(payload, T, payload_in_workspace=True)
        ev[5].record()
        torch.cuda.synchronize()
        if it >= args.warmup:
            j = it - args.warmup
            samples[0, j] = ev[0].elapsed_time(ev[1])   # dispatch
            samples[1, j] = ev[2].elapsed_time(ev[3])   # gemm
            samples[2, j] = ev[3].elapsed_time(ev[4])   # pre-reduce
            samples[3, j] = ev[4].elapsed_time(ev[5])   # combine

    # A step costs what the SLOWEST rank costs, per iteration.
    t = torch.from_numpy(samples).to(device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    samples = t.cpu().numpy()
    med = lambda x: float(np.median(x))
    band = lambda x: (float(np.percentile(x, 16)), float(np.percentile(x, 84)))
    pm = lambda x: f"{med(x):.3f}±{(band(x)[1] - band(x)[0]) / 2:.3f}"
    t_disp, t_gemm, t_pre, t_comb = samples
    t_serial = samples[1:].sum(axis=0)
    flops = 2 * total_m * n * k

    if rank == 0:
        print(f"\n(median ± half the 16-84 pct band over {args.iters} iters)")
        print(f"grouped GEMM (cuDNN): {pm(t_gemm)} ms "
              f"({flops / med(t_gemm) / 1e9:.0f} TFLOPS)")
        print(f"local pre-reduce:     {pm(t_pre)} ms")
        print(f"a2a combine:          {pm(t_comb)} ms "
              f"({combine_bytes / med(t_comb) / 1e6:.0f} GB/s)")
        print(f"serial GEMM->combine: {pm(t_serial)} ms")
        print(f"a2a dispatch (timed separately, not in the serial total): "
              f"{pm(t_disp)} ms ({dispatch_bytes / med(t_disp) / 1e6:.0f} GB/s)")

    stat = lambda x: dict(med=med(x), lo=band(x)[0], hi=band(x)[1])
    return dict(
        tokens=T, total_m=total_m, K=k, N=n, topk=args.topk,
        experts=args.experts, world=world_size, num_sms=num_sms, iters=args.iters,
        pairs=my_pairs, tilepipe_rows=tp_rows,
        dispatch_mb=dispatch_bytes / 1e6, combine_mb=combine_bytes / 1e6,
        gemm=stat(t_gemm), gemm_tflops=flops / med(t_gemm) / 1e9,
        pre_reduce=stat(t_pre), combine=stat(t_comb),
        combine_gbps=combine_bytes / med(t_comb) / 1e6,
        serial=stat(t_serial),
        dispatch=stat(t_disp), dispatch_gbps=dispatch_bytes / med(t_disp) / 1e6,
        samples=dict(dispatch=list(t_disp), gemm=list(t_gemm),
                     pre_reduce=list(t_pre), combine=list(t_comb)),
    )


def write_results(results, args, world_size):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"flashinfer_gemm_combine_{world_size}gpu_N{args.n}_K{args.hidden}_"
            f"topk{args.topk}_{stamp}")
    outdir = os.path.join(args.results_dir, name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args) | {"world_size": world_size},
                   "runs": results}, f, indent=2, default=str)

    ms = lambda d: f"{d['med']:.3f}±{(d['hi'] - d['lo']) / 2:.3f}"
    lines = [f"# FlashInfer GEMM -> combine baseline ({world_size} GPUs)", "",
             f"- generated: {stamp}",
             f"- K={args.hidden} N={args.n} topk={args.topk} "
             f"experts={args.experts}",
             "- grouped GEMM: `flashinfer.grouped_mm.grouped_mm_bf16` (cuDNN "
             "MoE backend), same varlen per-expert layout as TilePipe.",
             "- comm: `flashinfer.comm.MoeAlltoAll` (TRT-LLM MNNVL throughput "
             "a2a). Combine pulls ONE row per (token, source rank), so it "
             "moves less than TilePipe's one row per (token, expert) — the "
             "`pairs` / `rows` columns give both counts.",
             "- serial total = GEMM + local pre-reduce + combine. Dispatch is "
             "timed separately (the combine needs it, but it is the other "
             "half of the pipeline).", "",
             "| tokens/rank | total_m | pairs | TP rows | GEMM ms | TFLOPS | "
             "pre-reduce ms | combine ms | combine GB/s | serial ms | "
             "dispatch ms | dispatch GB/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['tokens']} | {r['total_m']} | {r['pairs']} | "
            f"{r['tilepipe_rows']} | {ms(r['gemm'])} | {r['gemm_tflops']:.0f} | "
            f"{ms(r['pre_reduce'])} | {ms(r['combine'])} | "
            f"{r['combine_gbps']:.0f} | {ms(r['serial'])} | "
            f"{ms(r['dispatch'])} | {r['dispatch_gbps']:.0f} |")
    with open(os.path.join(outdir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults written to {outdir}/ (results.json, results.md)")


def main():
    parser = argparse.ArgumentParser(
        description="FlashInfer grouped GEMM -> MoE combine baseline")
    parser.add_argument("--tokens", type=int, default=None,
                        help="single token count (overrides --token-sweep)")
    parser.add_argument("--token-sweep", type=str, default="2048,4096,8192,16384",
                        help="comma-separated tokens/rank to sweep")
    parser.add_argument("--results-dir", type=str, default="bench_results")
    parser.add_argument("--hidden", type=int, default=7168, help="GEMM K")
    parser.add_argument("--gemm-n", dest="n", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=None,
                        help="default: 32 * world_size (256 on 8 GPUs = DSv3)")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()
    token_list = ([args.tokens] if args.tokens is not None
                  else [int(x) for x in args.token_sweep.split(",")])

    rank, world_size, device = dist_init()
    if args.experts is None:
        args.experts = 32 * world_size
    assert args.experts % world_size == 0, "experts must divide across ranks"
    print(f"[rank {rank}] torch.distributed OK: world={world_size} "
          f"device=cuda:{device} ({torch.cuda.get_device_name()})")

    # One MNNVL workspace covers the whole sweep: sized for the largest token
    # count and for the WIDEST payload (dispatch carries K, combine carries N).
    max_tokens = max(token_list)
    ws_bytes = moe_a2a_get_workspace_size_per_rank(
        world_size, max_tokens,
        args.hidden * 2 + args.topk * 4 + args.topk * 4 + 4,  # dispatch payloads
        args.n * 2,                                           # combine payload
    )
    mapping = Mapping(world_size=world_size, rank=rank,
                      gpus_per_node=min(world_size, 8),
                      tp_size=world_size, moe_ep_size=world_size)
    print(f"[rank {rank}] MoeAlltoAll workspace {ws_bytes / 1e9:.2f} GB/rank "
          f"(max_tokens={max_tokens})")
    a2a = MoeAlltoAll(mapping, max_num_tokens=max_tokens, top_k=args.topk,
                      num_experts=args.experts,
                      workspace_size_per_rank=ws_bytes,
                      mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend()))
    print(f"[rank {rank}] MoeAlltoAll ready")

    try:
        results = []
        for tok in token_list:
            args.tokens = tok
            if rank == 0:
                print(f"\n{'=' * 70}\n=== tokens/rank = {tok} ===\n{'=' * 70}")
            r = run(args, a2a, rank, world_size, device)
            if r is not None:
                results.append(r)
            torch.cuda.synchronize()
            dist.barrier()
        if rank == 0 and results:
            args.tokens = token_list
            write_results(results, args, world_size)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
