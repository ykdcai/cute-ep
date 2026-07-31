# Copyright (c) 2026, QuACK team.
# TilePipe GEMM->combine phase 1: epilogue tile-completion publish.
#
# Single-GPU test: run the varlen_m grouped GEMM with tile_flag_ptrs pointing
# at a local flag array (world = 1). After the GEMM, every m-tile's counter
# must equal the N-tile count (each (m, n) work tile bumps its m-tile's flag
# by exactly 1 after its D stores complete), and the output must match the
# reference — the publish must not perturb the epilogue.

import math

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm
from tilepipe.args import TilePipeArgs
from tilepipe.sync import OutputTileSemaphore, RowBlockSemaphore


requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] != 10,
    reason="TilePipe tile-flag test requires SM100",
)


def _reference(A, B, cu_seqlens_m):
    out = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
    for e in range(B.shape[0]):
        lo, hi = cu_seqlens_m[e].item(), cu_seqlens_m[e + 1].item()
        out[lo:hi] = (A[lo:hi].float() @ B[e].float().T).to(A.dtype)
    return out


@requires_sm100
@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("n", [768, 2048])
def test_gemm_varlen_m_tile_flag_publish(n, tile_n):
    if tile_n > n:
        pytest.skip("tile larger than problem")
    device = "cuda"
    torch.random.manual_seed(0)
    num_experts = 8
    tile_m = 128
    k = 512
    seq_lens = torch.randint(64, 512, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    # Flat tile-id space: offsets[b] = cumsum(ceil(len_m / tile_M)) exclusive.
    m_tiles = [(int(s) + tile_m - 1) // tile_m for s in seq_lens.tolist()]
    offsets = torch.tensor(
        [0] + list(torch.tensor(m_tiles).cumsum(0)), dtype=torch.int32, device=device
    )[:-1].contiguous()
    total_tiles = sum(m_tiles)
    flags = torch.zeros(total_tiles, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([flags.data_ptr()], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets),
    )
    torch.cuda.synchronize()

    n_tiles = (n + tile_n - 1) // tile_n
    expected = torch.full((total_tiles,), n_tiles, dtype=torch.int32)
    assert torch.equal(flags.cpu(), expected), (
        f"tile flags {flags.cpu().tolist()} != expected {n_tiles} per m-tile"
    )
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def _flag_metadata(seq_lens, cta_tile_m, device):
    """Flat tile-id space over CTA tiles: offsets[b] = exclusive cumsum of
    ceil(len_m / cta_tile_m). Note CTA tile M, not the MMA tiler M — 2-CTA
    halves it."""
    m_tiles = [(int(s) + cta_tile_m - 1) // cta_tile_m for s in seq_lens.tolist()]
    offsets = torch.tensor(
        [0] + list(torch.tensor(m_tiles).cumsum(0)), dtype=torch.int32, device=device
    )[:-1].contiguous()
    return offsets, sum(m_tiles)


@requires_sm100
@pytest.mark.parametrize("world", [1, 4])
def test_gemm_varlen_m_tile_flag_publish_fanout(world):
    """The publish broadcasts to every peer's flag array, so each m-tile's
    counter must reach n_tiles on ALL of them. Pins the unrolled per-rank
    publish: the peer count is a compile-time constant (tile_flag_world), and
    a miscompiled loop bound shows up as a short count on the later peers."""
    device = "cuda"
    torch.random.manual_seed(2)
    num_experts, tile_m, tile_n, k, n = 4, 128, 128, 256, 512
    seq_lens = torch.randint(96, 320, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    offsets, total_tiles = _flag_metadata(seq_lens, tile_m, device)
    # Separate arrays per "peer" so a publish that skips one is visible.
    peers = [torch.zeros(total_tiles, dtype=torch.int32, device=device) for _ in range(world)]
    flag_ptrs = torch.tensor([p.data_ptr() for p in peers], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets),
    )
    torch.cuda.synchronize()

    n_tiles = (n + tile_n - 1) // tile_n
    expected = torch.full((total_tiles,), n_tiles, dtype=torch.int32)
    for r, p in enumerate(peers):
        assert torch.equal(p.cpu(), expected), f"peer {r}/{world}: {p.cpu().tolist()}"
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_sm100
@pytest.mark.parametrize("stride", [1, 32])
def test_gemm_varlen_m_tile_flag_stride(stride):
    """tile_flag_stride spaces flags out (32 int32 = one 128B line each) to
    spread the release atomics over L2 slices. Only the strided slots may be
    touched; the padding must stay zero."""
    device = "cuda"
    torch.random.manual_seed(3)
    num_experts, tile_m, tile_n, k, n = 4, 128, 128, 256, 512
    seq_lens = torch.randint(96, 320, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    offsets, total_tiles = _flag_metadata(seq_lens, tile_m, device)
    flags = torch.zeros(total_tiles * stride, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([flags.data_ptr()], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
                              tile_semaphore=RowBlockSemaphore(stride=stride)),
    )
    torch.cuda.synchronize()

    n_tiles = (n + tile_n - 1) // tile_n
    grid = flags.cpu().view(total_tiles, stride)
    assert torch.equal(grid[:, 0], torch.full((total_tiles,), n_tiles, dtype=torch.int32))
    assert grid[:, 1:].eq(0).all(), "publish wrote into the padding between flags"
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_sm100
@pytest.mark.parametrize(
    "tile_m,cluster_m,cluster_n", [(128, 1, 1), (256, 2, 1), (128, 1, 2), (256, 2, 2)]
)
def test_gemm_varlen_m_tile_flag_cluster_overhang(tile_m, cluster_m, cluster_n):
    """Regression: work is scheduled per CLUSTER, so a batch whose tile count
    is not a multiple of the cluster gets overhanging CTAs at an out-of-range
    (m, n). Their D stores are predicated away; an unpredicated publish landed
    on the NEXT expert's flags. Every seq_len below gives an ODD number of CTA
    tiles and n gives an odd number of n-tiles, so both overhangs are live.

    Before the fix, cluster 2x1 here gave 4 on the tile after each expert
    boundary instead of n_tiles.
    """
    device = "cuda"
    torch.random.manual_seed(4)
    tile_n, k, n = 128, 256, 384  # 3 n-tiles -> odd, so cluster_n=2 overhangs
    cta_tile_m = tile_m // 2 if (cluster_m % 2 == 0 and tile_m in (128, 256)) else tile_m
    # 5, 3, 7, 5 CTA tiles: every expert boundary lands mid-cluster.
    seq_lens = torch.tensor(
        [5 * cta_tile_m, 3 * cta_tile_m, 7 * cta_tile_m, 5 * cta_tile_m],
        device=device, dtype=torch.int32,
    )
    num_experts = seq_lens.numel()
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    offsets, total_tiles = _flag_metadata(seq_lens, cta_tile_m, device)
    flags = torch.zeros(total_tiles, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([flags.data_ptr()], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=cluster_m, cluster_N=cluster_n,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets),
    )
    torch.cuda.synchronize()

    n_tiles = (n + tile_n - 1) // tile_n
    expected = torch.full((total_tiles,), n_tiles, dtype=torch.int32)
    assert torch.equal(flags.cpu(), expected), (
        f"cluster {cluster_m}x{cluster_n} (cta_tile_m={cta_tile_m}): "
        f"{flags.cpu().tolist()} != {n_tiles} per m-tile"
    )
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_sm100
def test_gemm_varlen_m_tile_flags_with_gating():
    """Tile publish composes with the dispatch-side expert gate (flags
    pre-satisfied): both features active in one launch."""
    device = "cuda"
    torch.random.manual_seed(1)
    num_experts = 4
    tile_m, tile_n = 128, 128
    k, n = 256, 512
    seq_lens = torch.randint(96, 320, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    m_tiles = [(int(s) + tile_m - 1) // tile_m for s in seq_lens.tolist()]
    offsets = torch.tensor(
        [0] + list(torch.tensor(m_tiles).cumsum(0)), dtype=torch.int32, device=device
    )[:-1].contiguous()
    total_tiles = sum(m_tiles)
    tile_flags = torch.zeros(total_tiles, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([tile_flags.data_ptr()], dtype=torch.int64, device=device)
    ready_flags = seq_lens.clone()  # gate pre-satisfied

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(expert_ready_flags=ready_flags,
                              tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets),
    )
    torch.cuda.synchronize()

    n_tiles = (n + tile_n - 1) // tile_n
    assert torch.equal(
        tile_flags.cpu(), torch.full((total_tiles,), n_tiles, dtype=torch.int32)
    )
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# OutputTileSemaphore: one flag per (batch, m_tile, n_tile).
# ---------------------------------------------------------------------------
# The GEMM->allreduce atom. Same producer, same publish site, same deferral as
# the row-block case above — only the flag index rule differs, so these tests
# are about the index space and nothing else.
#
# Scope note: at world = 1 every flag lands on 1, which pins the per-batch
# offset stride (m_tiles * n_tiles, NOT m_tiles), the absence of collisions,
# and the overhang predicate. It cannot distinguish m-major from n-major
# within a batch — both are bijections onto the same range. That ambiguity is
# only observable by a consumer mapping flag -> tile, so it is pinned by the
# 2-GPU allreduce correctness test, where a transposed index reduces a tile
# whose data has not arrived.
#
# multicast=False throughout: `multimem.red` needs a multicast address from
# nvshmem, which a single-GPU test has no business allocating. The index
# arithmetic under test is shared by both publish instructions.


def _tile_flag_metadata_2d(seq_lens, cta_tile_m, n_tiles, device):
    """Flag space for OutputTileSemaphore: each batch owns
    ceil(len_m / tile_M) * n_tiles slots, so the per-batch base is the
    exclusive cumsum of THAT, not of the m-tile count."""
    m_tiles = [(int(s) + cta_tile_m - 1) // cta_tile_m for s in seq_lens.tolist()]
    per_batch = [t * n_tiles for t in m_tiles]
    offsets = torch.tensor(
        [0] + list(torch.tensor(per_batch).cumsum(0)), dtype=torch.int32, device=device
    )[:-1].contiguous()
    return offsets, sum(per_batch)


@requires_sm100
@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("n", [768, 2048])
def test_gemm_varlen_m_output_tile_flag_publish(n, tile_n):
    """Every (m_tile, n_tile) of every expert gets its own counter, bumped
    exactly once."""
    if tile_n > n:
        pytest.skip("tile larger than problem")
    device = "cuda"
    torch.random.manual_seed(0)
    num_experts, tile_m, k = 8, 128, 512
    n_tiles = (n + tile_n - 1) // tile_n
    seq_lens = torch.randint(64, 512, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    offsets, total_flags = _tile_flag_metadata_2d(seq_lens, tile_m, n_tiles, device)
    flags = torch.zeros(total_flags, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([flags.data_ptr()], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
                              tile_semaphore=OutputTileSemaphore(multicast=False)),
    )
    torch.cuda.synchronize()

    expected = torch.ones(total_flags, dtype=torch.int32)
    assert torch.equal(flags.cpu(), expected), (
        f"output-tile flags != 1 everywhere; got "
        f"{flags.cpu().tolist()[:16]}... (total {total_flags})"
    )
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_sm100
@pytest.mark.parametrize("n,tile_n", [(512, 128), (768, 256)])
def test_gemm_dense_output_tile_flag_publish(n, tile_n):
    """The publish lives in the epilogue's shared tile loop, so it works
    without varlen_m — which is the shape GEMM->allreduce actually runs
    (a TP GEMM, one batch). tile_flag_offsets degenerates to a single zero."""
    device = "cuda"
    torch.random.manual_seed(5)
    m, k, tile_m = 1024, 512, 128
    m_tiles, n_tiles = (m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n
    A = torch.randn((1, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((1, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((1, m, n), device=device, dtype=torch.bfloat16)

    offsets = torch.zeros(1, dtype=torch.int32, device=device)
    flags = torch.zeros(m_tiles * n_tiles, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([flags.data_ptr()], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=1, cluster_N=1, persistent=True,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
                              tile_semaphore=OutputTileSemaphore(multicast=False)),
    )
    torch.cuda.synchronize()

    expected = torch.ones(m_tiles * n_tiles, dtype=torch.int32)
    assert torch.equal(flags.cpu(), expected), f"dense: {flags.cpu().tolist()}"
    ref = torch.bmm(A.float(), B.float().mT).to(torch.bfloat16)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@requires_sm100
@pytest.mark.parametrize(
    "tile_m,cluster_m,cluster_n", [(128, 1, 1), (256, 2, 1), (128, 1, 2), (256, 2, 2)]
)
def test_gemm_output_tile_flag_cluster_overhang(tile_m, cluster_m, cluster_n):
    """Cluster overhang is MORE dangerous for a per-(m,n) flag than for a row
    block. With a row-block counter an n-overhang merely overshoots the same
    counter; here it lands on a DIFFERENT tile's counter, leaving one flag at
    2 and its neighbour at 0 — a consumer would then reduce a tile whose data
    was never written. n=384 with tile_n=128 gives 3 n-tiles (odd, so
    cluster_n=2 overhangs) and every seq_len is an odd number of CTA tiles."""
    device = "cuda"
    torch.random.manual_seed(6)
    tile_n, k, n = 128, 256, 384
    n_tiles = (n + tile_n - 1) // tile_n
    cta_tile_m = tile_m // 2 if (cluster_m % 2 == 0 and tile_m in (128, 256)) else tile_m
    seq_lens = torch.tensor(
        [5 * cta_tile_m, 3 * cta_tile_m, 7 * cta_tile_m, 5 * cta_tile_m],
        device=device, dtype=torch.int32,
    )
    num_experts = seq_lens.numel()
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)

    offsets, total_flags = _tile_flag_metadata_2d(seq_lens, cta_tile_m, n_tiles, device)
    flags = torch.zeros(total_flags, dtype=torch.int32, device=device)
    flag_ptrs = torch.tensor([flags.data_ptr()], dtype=torch.int64, device=device)

    quack_gemm(
        A, B, out, C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=cluster_m, cluster_N=cluster_n,
        persistent=True, cu_seqlens_m=cu_seqlens_m,
        tilepipe=TilePipeArgs(tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
                              tile_semaphore=OutputTileSemaphore(multicast=False)),
    )
    torch.cuda.synchronize()

    got = flags.cpu()
    expected = torch.ones(total_flags, dtype=torch.int32)
    assert torch.equal(got, expected), (
        f"cluster {cluster_m}x{cluster_n} (cta_tile_m={cta_tile_m}, "
        f"n_tiles={n_tiles}): overhang leaked, {got.tolist()}"
    )
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
