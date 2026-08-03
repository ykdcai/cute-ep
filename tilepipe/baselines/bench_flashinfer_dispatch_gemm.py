"""FlashInfer baseline for the TilePipe dispatch -> grouped-GEMM pipeline.

Serial (non-overlapped) reference point for ``tilepipe/dispatch_gemm.py``,
measured on the same problem, the same routing and the same operands as
``bench_flashinfer_gemm_combine.py`` -- the two baselines together cover both
halves of the MoE layer.

The stacks differ in WHERE the expert grouping happens, and that is the point
of the comparison:

  TilePipe   the dispatch kernel writes each token straight into its slot in
             the expert-grouped buffer, so the GEMM's operand is assembled by
             the transfer itself. Nothing local runs between them.
  FlashInfer MoeAlltoAll.dispatch delivers into [src_rank, slot] order, which
             is NOT expert-grouped, so a local permute has to build the GEMM
             operand afterwards.

That permute is real work the baseline must pay and TilePipe must not, so it
is timed as its own stage and reported separately; the serial total is
dispatch + permute + GEMM. FlashInfer also deduplicates per destination rank
(a token hitting 3 experts on one rank crosses the wire once), so it moves
fewer bytes -- the volume line prints both counts so GB/s figures are never
compared across mismatched volumes.

Run (2 GPUs, defaults are DeepSeek-V3 shapes):
    torchrun --nproc-per-node 2 tilepipe/baselines/bench_flashinfer_dispatch_gemm.py
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
# Same operands and bootstrap as the combine baseline, so a rank's hidden
# states and expert weights are byte-identical across the two.
from tilepipe.baselines.bench_flashinfer_gemm_combine import (
    build_weights, dist_init, expert_weight, rank_tensor,
)

print = functools.partial(print, flush=True)


def run(args, a2a, rank, world_size, device):
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    epr = args.experts // world_size
    n, k, T = args.n, args.hidden, args.tokens
    invalid_expert = args.experts
    torch.manual_seed(42 + rank)

    if rank == 0:
        print(f"FlashInfer dispatch->GEMM: tokens/rank={T} K={k} N={n} "
              f"experts={args.experts} (local {epr}) topk={args.topk} "
              f"world={world_size} SMs={num_sms}")

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

    weights = build_weights(rank, epr, n, k, device)
    # The GEMM operand the permute must assemble, and its output.
    A = torch.zeros((total_m, k), dtype=torch.bfloat16, device=device)
    D = torch.zeros((total_m, n), dtype=torch.bfloat16, device=device)

    # What this rank dispatches. Same seed as the combine baseline's A, so the
    # token a peer receives here is the row that baseline feeds its GEMM.
    hidden_in = rank_tensor(500 + rank, (T, k), device, scale=k ** -0.5)
    token_ids = torch.arange(T, dtype=torch.int32, device=device).view(T, 1)

    # Host-side (src rank, token, slot-in-topk) -> A-row map. Identical
    # construction to the combine baseline's, since dispatch and combine walk
    # the same (token, expert) pairs in opposite directions.
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

    dst_ranks = all_topk[rank] // epr
    my_pairs = int(sum(len(np.unique(r)) for r in dst_ranks))
    tp_rows = T * args.topk
    bytes_per_token = k * 2 + args.topk * 4 + args.topk * 4 + 4
    if rank == 0:
        print(f"volume/rank: {my_pairs} (token,rank) pairs vs {tp_rows} "
              f"(token,expert) rows TilePipe moves ({tp_rows / my_pairs:.2f}x); "
              f"dispatch {my_pairs * bytes_per_token / 1e6:.0f} MB")

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    def do_dispatch():
        return a2a.dispatch(
            topk_indices, [hidden_in, topk_indices, token_ids], T,
            invalid_token_expert_id=invalid_expert, expert_id_payload_index=1)

    def close_dispatch():
        """Return MoeAlltoAll to its idle phase.

        The object is a two-phase state machine -- `dispatch` asserts
        phase=="idle" and only `combine` resets it, with no public abort. A
        dispatch-only benchmark therefore has to issue a matching combine or
        the second iteration dies with "dispatch called twice without
        combine". It is called AFTER the last timing event, so it never lands
        inside a measured window; the payload is workspace-resident garbage
        and its output is discarded, since only the state transition matters.
        """
        a2a.combine(a2a.get_combine_payload_tensor_in_workspace(
            T, n, torch.bfloat16), T, payload_in_workspace=True)

    def permute(recv):
        """Build the expert-grouped GEMM operand from the dispatch's output.

        Recv slot (s, i) holds source rank s's token recv_ids[s, i]; the
        assignment comes from atomics, so the inversion is rebuilt after every
        dispatch. A token landing on several of this rank's experts is
        dispatched ONCE, so this is a one-to-many gather.
        """
        recv_hidden, recv_experts, recv_ids = recv[0], recv[1], recv[2][..., 0]
        valid = (recv_experts != invalid_expert).any(dim=-1)      # [ep, T]
        s_i, i_i = valid.nonzero(as_tuple=True)
        inv = torch.full((world_size, T), -1, dtype=torch.int64, device=device)
        inv[s_i, recv_ids[s_i, i_i].long()] = i_i
        slot = inv[pair_src, pair_tok]
        assert int(slot.min()) >= 0, "dispatch dropped a token this rank owns"
        A[pair_row] = recv_hidden.view(-1, k)[pair_src * T + slot]

    # --- Correctness. The operand the permute must produce is exactly
    # gemm_combine.py's A: row `scatter[t, j]` of the owner rank holds source
    # token (src_rank[t, j], t). Checking A directly localises a failure to the
    # dispatch+permute rather than hiding it in the GEMM's output.
    all_hidden = [torch.zeros_like(hidden_in) for _ in range(world_size)]
    dist.all_gather(all_hidden, hidden_in.contiguous())
    A_ref = torch.zeros_like(A)
    for s in range(world_size):
        meta_s = (meta if s == rank
                  else build_combine_metadata(all_topk, args.experts, s, world_size))
        t_idx, j_idx = np.nonzero(meta_s["src_rank"] == rank)
        rows = torch.from_numpy(meta_s["scatter"][t_idx, j_idx].astype(np.int64)).to(device)
        A_ref[rows] = all_hidden[s][torch.from_numpy(t_idx.astype(np.int64)).to(device)]

    D_ref = torch.zeros((total_m, n), dtype=torch.float32, device=device)
    cu = meta["cu_seqlens"]
    for e in range(epr):
        lo, hi = int(cu[e]), int(cu[e + 1])
        if hi > lo:
            W_e = expert_weight(rank, e, n, k, device)
            D_ref[lo:hi] = A_ref[lo:hi].float() @ W_e.float().T
            del W_e
    torch.cuda.synchronize()

    barrier()
    recv = do_dispatch()
    permute(recv)
    grouped_mm_bf16(A, weights, m_indptr, out=D)
    close_dispatch()
    barrier()
    ok_a = torch.equal(A, A_ref)
    rel = ((D.float() - D_ref).abs().max()
           / D_ref.abs().max().clamp(min=1e-6)).item()
    ok_d = rel < 3e-2
    print(f"[rank {rank}] dispatch->GEMM: operand={'OK' if ok_a else 'FAIL'} "
          f"gemm rel_err={rel:.2e} {'OK' if ok_d else 'FAIL'}")
    ok_t = torch.tensor([ok_a and ok_d], dtype=torch.int32, device=device)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if not bool(ok_t.item()):
        raise SystemExit("correctness FAILED")
    del A_ref, D_ref

    if not args.benchmark:
        return None

    # --- Timing. Stages are timed inside one loop with events between them, so
    # they sum to the serial total by construction (same scheme as the combine
    # baseline and gemm_combine.py's paired timing).
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(4)]
          for _ in range(args.warmup + args.iters)]
    samples = np.zeros((4, args.iters))
    for j in range(args.warmup + args.iters):
        barrier()
        e = ev[j]
        e[0].record()
        recv = do_dispatch()
        e[1].record()
        permute(recv)
        e[2].record()
        grouped_mm_bf16(A, weights, m_indptr, out=D)
        e[3].record()
        close_dispatch()          # untimed: after e[3]
        torch.cuda.synchronize()
        if j >= args.warmup:
            i = j - args.warmup
            samples[0, i] = e[0].elapsed_time(e[1])   # dispatch
            samples[1, i] = e[1].elapsed_time(e[2])   # permute
            samples[2, i] = e[2].elapsed_time(e[3])   # gemm
            samples[3, i] = e[0].elapsed_time(e[3])   # serial total
    # Reduce PER ITERATION (a step costs what the slowest rank costs), so the
    # bands below describe the step rather than one rank's view of it. Same
    # scheme as gemm_combine.py.
    st = torch.from_numpy(samples).to(device)
    dist.all_reduce(st, op=dist.ReduceOp.MAX)
    samples = st.cpu().numpy()
    disp, perm, gemm, serial = np.median(samples, axis=1).tolist()

    def stat(x):
        return dict(med=float(np.median(x)), lo=float(np.percentile(x, 16)),
                    hi=float(np.percentile(x, 84)))

    if rank == 0:
        flops = 2 * total_m * n * k
        print(f"\ndispatch      {disp:7.3f} ms "
              f"({my_pairs * bytes_per_token / disp / 1e6:.0f} GB/s)")
        print(f"permute       {perm:7.3f} ms  (local; TilePipe's dispatch "
              f"does this in-transfer)")
        print(f"grouped GEMM  {gemm:7.3f} ms ({flops / gemm / 1e9:.0f} TFLOPS)")
        print(f"serial total  {serial:7.3f} ms")
    return dict(tokens=T, total_m=total_m, K=k, N=n, topk=args.topk,
                experts=args.experts, world=world_size, num_sms=num_sms,
                iters=args.iters, pairs=my_pairs, tilepipe_rows=tp_rows,
                dispatch_mb=my_pairs * bytes_per_token / 1e6,
                dispatch_gbps=my_pairs * bytes_per_token / disp / 1e6,
                gemm_tflops=2 * total_m * n * k / gemm / 1e9,
                # Same shape as gemm_combine.py's: med/lo/hi per stage plus the
                # raw per-iteration samples, so the two can be compared (and
                # re-analysed) without re-running either.
                dispatch=stat(samples[0]), permute=stat(samples[1]),
                gemm=stat(samples[2]), serial=stat(samples[3]),
                samples=dict(dispatch=list(samples[0]), permute=list(samples[1]),
                             gemm=list(samples[2]), serial=list(samples[3])))


def write_results(results, args, world_size):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"flashinfer_dispatch_gemm_{world_size}gpu_N{args.n}_"
            f"K{args.hidden}_topk{args.topk}_{stamp}")
    outdir = os.path.join(args.results_dir, name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args) | {"world_size": world_size},
                   "runs": results}, f, indent=2, default=str)
    lines = [f"# FlashInfer dispatch -> grouped GEMM ({world_size} GPUs)", "",
             f"- generated: {stamp}",
             f"- K={args.hidden} N={args.n} topk={args.topk} "
             f"experts={args.experts}",
             "- `permute` is the local rearrangement into expert-grouped order "
             "that MoeAlltoAll's [src_rank, slot] output requires and "
             "TilePipe's dispatch does as part of the transfer.", "",
             "- values are median ± half the 16-84 percentile band; raw "
             "per-iteration samples are in results.json.", "",
             "| tokens/rank | dispatch ms | permute ms | GEMM ms | serial ms |"
             " TFLOPS | pairs vs TilePipe rows |",
             "|---|---|---|---|---|---|---|"]
    ms = lambda d: f"{d['med']:.3f}±{(d['hi'] - d['lo']) / 2:.3f}"
    for r in results:
        lines.append(
            f"| {r['tokens']} | {ms(r['dispatch'])} | {ms(r['permute'])} "
            f"| {ms(r['gemm'])} | {ms(r['serial'])} | "
            f"{r['gemm_tflops']:.0f} | {r['pairs']} vs {r['tilepipe_rows']} "
            f"({r['tilepipe_rows'] / r['pairs']:.2f}x) |")
    with open(os.path.join(outdir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults written to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="FlashInfer MoE dispatch -> grouped GEMM baseline")
    parser.add_argument("--tokens", type=int, default=None,
                        help="single token count (overrides --token-sweep)")
    parser.add_argument("--token-sweep", type=str, default="2048,4096,8192,16384")
    parser.add_argument("--results-dir", type=str, default="bench_results")
    parser.add_argument("--hidden", type=int, default=7168, help="GEMM K")
    parser.add_argument("--gemm-n", dest="n", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=None,
                        help="default: 32 * world_size")
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

    # Dispatch-only, so the combine payload width is 0 -- but MoeAlltoAll sizes
    # one workspace for both directions, so keep the combine slot at N to stay
    # byte-comparable with the combine baseline's workspace.
    max_tokens = max(token_list)
    ws_bytes = moe_a2a_get_workspace_size_per_rank(
        world_size, max_tokens,
        args.hidden * 2 + args.topk * 4 + args.topk * 4 + 4,
        args.n * 2,
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
