# Copyright (c) 2026, QuACK team.
# TilePipe phase 1: per-expert ready-flag gating of the varlen_m grouped GEMM.
#
# Single-GPU test: the GEMM is launched with all expert flags at 0 and a
# capped number of persistent clusters, so its load warps spin on the flags.
# A tiny CuTe "trickle" kernel (1 block, 1 warp) on a second stream then
# raises each expert's flag to its segment length, one expert at a time, in a
# given order, using release/sys atomic adds — the exact protocol the TilePipe
# dispatch kernel uses. Each flag is raised in two steps (partial, then full)
# so a partial token count provably does not release the expert.
#
# The trickle runs entirely on-device: no host CUDA calls may happen between
# the GEMM launch and the flag raises, because host-side ops (allocations,
# scalar H2D copies) can device-sync and deadlock against the spinning GEMM.

import math

import pytest
import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from quack.cute_dsl_utils import get_device_capacity, nanosleep
from quack.gemm import gemm as quack_gemm


requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] != 10,
    reason="TilePipe gating test requires SM100",
)


@cute.kernel
def _trickle_kernel(
    flags: cute.Tensor,  # (num_experts,) Int32, starts at 0
    targets: cute.Tensor,  # (num_experts,) Int32, per-expert segment length
    order: cute.Tensor,  # (num_experts,) Int32, order in which to release experts
    delay_iters: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        num_experts = cute.size(order.shape)
        for i in cutlass.range(num_experts):
            e = order[i]
            target = targets[e]
            half = target // 2
            for _ in cutlass.range(delay_iters):
                nanosleep(1024)
            # Two-step raise with the dispatch kernel's protocol: release/sys
            # atomic add. A partial count must NOT release the expert.
            cute.arch.atomic_add(flags.iterator + e, half, sem="release", scope="sys")
            for _ in cutlass.range(delay_iters // 4):
                nanosleep(1024)
            cute.arch.atomic_add(flags.iterator + e, target - half, sem="release", scope="sys")


@cute.jit
def _trickle_launch(
    flags: cute.Tensor,
    targets: cute.Tensor,
    order: cute.Tensor,
    stream: cuda.CUstream,
):
    _trickle_kernel(flags, targets, order, 4096).launch(
        grid=[1, 1, 1], block=[32, 1, 1], stream=stream
    )


def _reference(A, B, cu_seqlens_m):
    out = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
    for e in range(B.shape[0]):
        lo, hi = cu_seqlens_m[e].item(), cu_seqlens_m[e + 1].item()
        out[lo:hi] = (A[lo:hi].float() @ B[e].float().T).to(A.dtype)
    return out


def _run_gated_gemm(A, B, out, cu_seqlens_m, flags, max_active_clusters=None):
    quack_gemm(
        A,
        B,
        out,
        C=None,
        tile_count_semaphore=None,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
        persistent=True,
        cu_seqlens_m=cu_seqlens_m,
        expert_ready_flags=flags,
        max_active_clusters=max_active_clusters,
    )


@requires_sm100
@pytest.mark.parametrize("trickle_order", ["forward", "reverse"])
def test_gemm_varlen_m_ready_flag_gating(trickle_order):
    """GEMM waits for each expert's arrival counter to reach its seqlen."""
    device = "cuda"
    torch.random.manual_seed(0)
    num_experts = 8
    k, n = 512, 768
    seq_lens = torch.randint(64, 512, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)
    ref = _reference(A, B, cu_seqlens_m)

    if trickle_order == "forward":
        order = torch.arange(num_experts, device=device, dtype=torch.int32)
    else:
        # Release the *last* expert first to prove the wait keys on
        # per-expert counters, not just "some flag moved".
        order = torch.arange(num_experts - 1, -1, -1, device=device, dtype=torch.int32)
    flags = torch.zeros(num_experts, dtype=torch.int32, device=device)

    # Compile both kernels up front (flags pre-satisfied for the GEMM warm-up,
    # a throwaway flags copy for the trickle) so the gating run enqueues
    # launches only — no compilation or allocation while the GEMM spins.
    flags_warmup = seq_lens.clone()
    _run_gated_gemm(A, B, out, cu_seqlens_m, flags_warmup)
    trickle_compiled = cute.compile(
        _trickle_launch,
        from_dlpack(flags),
        from_dlpack(seq_lens),
        from_dlpack(order),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    # Warm-up execution of the trickle on scratch flags (first call may lazily
    # load the module) — also validates the trickle kernel standalone.
    flags_scratch = torch.zeros_like(flags)
    trickle_compiled(
        from_dlpack(flags_scratch),
        from_dlpack(seq_lens),
        from_dlpack(order),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()
    assert torch.equal(flags_scratch, seq_lens)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    # The real gating run: flags start at zero; the GEMM (capped so SMs remain
    # for the trickle stream) must wait until the trickle kernel raises each
    # expert's counter to its full segment length.
    out.zero_()
    gemm_stream = torch.cuda.Stream()
    trickle_stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    with torch.cuda.stream(gemm_stream):
        _run_gated_gemm(A, B, out, cu_seqlens_m, flags, max_active_clusters=64)
    trickle_compiled(
        from_dlpack(flags),
        from_dlpack(seq_lens),
        from_dlpack(order),
        cuda.CUstream(trickle_stream.cuda_stream),
    )

    torch.cuda.synchronize()
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
    # Flags must have been raised to exactly the targets (untouched by GEMM).
    assert torch.equal(flags, seq_lens)


@requires_sm100
def test_gemm_varlen_m_ungated_still_works():
    """Passing no flags keeps the ungated fast path intact."""
    device = "cuda"
    torch.random.manual_seed(1)
    num_experts = 4
    k, n = 256, 512
    seq_lens = torch.randint(96, 320, (num_experts,), device=device, dtype=torch.int32)
    total_m = int(seq_lens.sum().item())
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), seq_lens.cumsum(0).to(torch.int32)]
    )
    A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn((num_experts, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)
    quack_gemm(
        A,
        B,
        out,
        C=None,
        tile_count_semaphore=None,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
        persistent=True,
        cu_seqlens_m=cu_seqlens_m,
    )
    torch.cuda.synchronize()
    ref = _reference(A, B, cu_seqlens_m)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
