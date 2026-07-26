"""Host-side plumbing for the TilePipe hooks inside quack's GEMM.

Owns everything about the tilepipe GEMM arguments except the kernel code
itself, so ``quack/gemm.py`` carries one optional kwarg instead of four and
``quack/gemm_tvm_ffi_utils.py`` carries no tilepipe knowledge at all.

Imports only ``quack.compile_utils`` (a leaf module), never ``quack.gemm`` —
the dependency runs quack -> tilepipe, one way.
"""

from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
from cutlass import Int32, Int64
from torch import Tensor

from quack.compile_utils import make_fake_tensor as fake_tensor


@dataclass(frozen=True)
class TilePipeArgs:
    """The tilepipe-specific arguments to ``quack.gemm.gemm``.

    All fields default to None/1, so ``TilePipeArgs()`` is the "plain GEMM,
    no tilepipe" case and the kernel compiles exactly as upstream.
    """

    # (l,) int32 token-arrival counters, one per varlen_m batch (expert). The
    # mainloop waits for expert_ready_flags[b] >= seqlen_m(b) before issuing
    # any TMA load for batch b.
    expert_ready_flags: Optional[Tensor] = None

    # GEMM->combine tile-completion publish. tile_flag_ptrs is an (world,)
    # int64 tensor of every rank's tile-flag array base address (symmetric
    # memory); tile_flag_offsets is (l,) int32 with
    # cumsum(ceil(seqlen_m(b) / tile_M)) exclusive — the epilogue bumps
    # flag[offsets[b] + m_tile] by 1 on every rank once the work tile's D
    # stores complete. A row block is ready when its counter reaches
    # ceil(N / tile_N).
    tile_flag_ptrs: Optional[Tensor] = None
    tile_flag_offsets: Optional[Tensor] = None

    # Spacing (in int32 elements) between consecutive tile flags. 1 packs 32
    # flags into a 128B line; 32 gives each its own line, spreading the release
    # atomics over more L2 slices. The consumer must index flags[idx * stride].
    tile_flag_stride: int = 1

    def validate(self, varlen_m: bool, gather_A: bool) -> None:
        if self.expert_ready_flags is not None:
            assert varlen_m, "expert_ready_flags requires varlen_m (grouped GEMM)"
            assert not gather_A, "expert_ready_flags not supported with gather_A"
        if self.tile_flag_ptrs is not None:
            assert varlen_m, "tile_flag_ptrs requires varlen_m (grouped GEMM)"
            assert self.tile_flag_offsets is not None, (
                "tile_flag_ptrs requires tile_flag_offsets"
            )

    @property
    def compile_key(self) -> tuple:
        """The compile-cache key contribution: STATIC scalars only.

        Never include the tensors themselves — this tuple is hashed by
        ``jit_cache``. Positionally matches ``_compile_gemm``'s
        (has_ready_flags, tile_flag_world, tile_flag_stride) tail.
        """
        return (
            self.expert_ready_flags is not None,
            self.tile_flag_ptrs.numel() if self.tile_flag_ptrs is not None else 0,
            self.tile_flag_stride,
        )


def varlen_fields(args: Optional[TilePipeArgs]) -> dict:
    """The tilepipe half of a real ``VarlenArguments``."""
    args = args or TilePipeArgs()
    return dict(
        mReadyFlags=args.expert_ready_flags,
        mTileFlagPtrs=args.tile_flag_ptrs,
        mTileOffsets=args.tile_flag_offsets,
    )


def fake_varlen_fields(has_ready_flags: bool = False, tile_flag_world: int = 0) -> dict:
    """The tilepipe half of a fake (compile-time) ``VarlenArguments``.

    tile_flag_world is the number of peer flag arrays (0 = no tile flags). It
    is STATIC on purpose: the epilogue's publish loop is per-rank, and a
    dynamic extent turns each publish into an unroll ladder over gmem loads.
    """
    has_tile_flags = tile_flag_world > 0
    return dict(
        mReadyFlags=(
            fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
            if has_ready_flags
            else None
        ),
        mTileFlagPtrs=(
            fake_tensor(Int64, (tile_flag_world,), leading_dim=0, divisibility=1)
            if has_tile_flags
            else None
        ),
        mTileOffsets=(
            fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=1)
            if has_tile_flags
            else None
        ),
    )
