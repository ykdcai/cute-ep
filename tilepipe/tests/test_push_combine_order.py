"""The push-combine send list must be ordered by the PRODUCING expert -- the
order the GEMM lays rows out in and completes tiles in -- not by destination.

Two independent derivations have to agree on which D-buffer row belongs to
which (token, j):

  build_combine_metadata()      -> the GEMM side: cu_seqlens (per-expert row
                                   ranges) and scatter[t, j] (the row of
                                   (t, j) in the owner rank's buffer)
  build_push_combine_arrays()   -> the comm side: for each row, its producing
                                   expert, home rank and staging slot

If they disagree, gate_idx points at the wrong tile: the consumer either
waits on a counter the producer never reaches (hang) or sends a row before
the MMA wrote it (silently wrong data that a rel_err check can miss when the
row happens to be small). Nothing in the fused benchmark distinguishes those
from a slow kernel, so pin the ordering here, on CPU, where it is cheap.

These are pure numpy derivations -- no GPU, no CUDA context.
"""

import numpy as np
import pytest

from tilepipe.plan import build_combine_metadata, build_push_combine_arrays


def routing(world_size, num_experts, tokens, topk, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_experts, size=(world_size, tokens, topk),
                        dtype=np.int64).astype(np.int32)


CASES = [
    # (world, experts, tokens, topk, tile_m)
    (2, 64, 512, 8, 128),
    (2, 64, 333, 4, 128),      # ragged: not a multiple of tile_m
    (4, 128, 512, 8, 128),
    (4, 128, 97, 2, 128),      # experts with zero rows
    (8, 256, 256, 8, 64),
]
IDS = [f"w{w}_e{e}_t{t}_k{k}_tm{tm}" for w, e, t, k, tm in CASES]


@pytest.mark.parametrize("world,experts,tokens,topk,tile_m", CASES, ids=IDS)
def test_push_order_matches_gemm_expert_order(world, experts, tokens, topk, tile_m):
    all_topk = routing(world, experts, tokens, topk, seed=world * 100 + topk)
    epr = experts // world

    for rank in range(world):
        meta = build_combine_metadata(all_topk, experts, rank, world, tile_m=tile_m)
        row, slot, home, seg, seg_sizes, gate_idx, _ = build_push_combine_arrays(
            all_topk, experts, rank, world, tile_m=tile_m)
        cu = meta["cu_seqlens"].astype(np.int64)
        n_rows = meta["rank_rows"][rank]

        # 1. List position == D-buffer row. Everything else assumes it.
        assert len(row) == n_rows
        assert np.array_equal(row, np.arange(n_rows, dtype=np.int32))

        # 2. Producer order: the expert owning each row, read off the GEMM's
        # cu_seqlens, must be non-decreasing along the send list. This is the
        # property that makes the gate monotone -- and it is exactly "expert
        # order in the GEMM", not destination order.
        expert_of_row = np.searchsorted(cu, np.arange(n_rows), side="right") - 1
        assert np.all(np.diff(expert_of_row) >= 0)

        # 3. The two derivations agree on the (token, j) <-> row bijection.
        # scatter[t, j] is the row in the OWNER's buffer; for rows this rank
        # produces (src_rank == rank), that row must be the one the push plan
        # sends to staging slot t * topk + j.
        expect = np.full(n_rows, -1, dtype=np.int64)
        for src in range(world):
            m = meta if src == rank else build_combine_metadata(
                all_topk, experts, src, world, tile_m=tile_m)
            sc, sr = m["scatter"], m["src_rank"]
            t_idx, j_idx = np.nonzero(sr == rank)
            expect[sc[t_idx, j_idx]] = t_idx.astype(np.int64) * topk + j_idx
        assert not np.any(expect < 0), "some D row is claimed by no (token, j)"
        assert np.array_equal(slot.astype(np.int64), expect), (
            f"rank {rank}: push slot disagrees with the GEMM's scatter map")

        # 4. gate_idx is monotone non-decreasing and inside the producer's
        # tile range (one poll per tile, never past the published array).
        assert np.all(np.diff(gate_idx.astype(np.int64)) >= 0)
        assert gate_idx.max(initial=-1) < len(meta["tile_lo"])

        # 5. gate_idx is the tile the row actually lives in, per the GEMM's
        # own per-expert tiling -- the same arithmetic the epilogue publishes
        # with, recomputed here from the GEMM-side metadata.
        tiles_per_expert = (np.diff(cu) + tile_m - 1) // tile_m
        tile_off = np.concatenate([[0], np.cumsum(tiles_per_expert)])[:-1]
        within = np.arange(n_rows) - cu[expert_of_row]
        assert np.array_equal(gate_idx.astype(np.int64),
                              tile_off[expert_of_row] + within // tile_m)

        # 6. Segments are contiguous runs (one arrival publish each).
        s = seg.astype(np.int64)
        assert np.all(np.diff(s) >= 0), "segments are not contiguous in send order"
        assert np.array_equal(np.bincount(s, minlength=len(seg_sizes)),
                              seg_sizes.astype(np.int64))


@pytest.mark.parametrize("world,experts,tokens,topk,tile_m", CASES, ids=IDS)
def test_push_order_is_not_destination_major(world, experts, tokens, topk, tile_m):
    """Guard against a future 'group by destination' rewrite: home rank must
    NOT be the primary sort key. Destination-major ordering would still pass a
    correctness check -- every row still lands in the right slot -- while
    silently destroying gate monotonicity and the overlap with it."""
    all_topk = routing(world, experts, tokens, topk, seed=7 + world)
    for rank in range(world):
        _, _, home, _, _, gate_idx, _ = build_push_combine_arrays(
            all_topk, experts, rank, world, tile_m=tile_m)
        if len(home) < 2 or world < 2:
            continue
        # Destination-major would make `home` non-decreasing overall. Producer
        # -major revisits every home once per expert, so it must not be.
        assert not np.all(np.diff(home.astype(np.int64)) >= 0), (
            "send list is sorted by destination rank -- gate monotonicity lost")
        # And the gate must advance across the list, not restart per destination.
        assert gate_idx[-1] >= gate_idx[0]
