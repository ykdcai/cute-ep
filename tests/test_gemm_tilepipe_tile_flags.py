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
        tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
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
        tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
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
        tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
        tile_flag_stride=stride,
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
        tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
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
        expert_ready_flags=ready_flags,
        tile_flag_ptrs=flag_ptrs, tile_flag_offsets=offsets,
    )
    torch.cuda.synchronize()

    n_tiles = (n + tile_n - 1) // tile_n
    assert torch.equal(
        tile_flags.cpu(), torch.full((total_tiles,), n_tiles, dtype=torch.int32)
    )
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
