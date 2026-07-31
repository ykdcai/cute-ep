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
from tilepipe.sync import OutputTileSemaphore, RowBlockSemaphore, TileSemaphore


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

    # Output-tile completion publish. tile_flag_ptrs is an (world,) int64
    # tensor of the flag-array base addresses to publish to (symmetric memory;
    # a 1-entry table means "publish locally only", or, for a multicast
    # semaphore, holds the multicast address). tile_flag_offsets is (l,) int32,
    # the exclusive cumsum of each batch's flag count.
    tile_flag_ptrs: Optional[Tensor] = None
    tile_flag_offsets: Optional[Tensor] = None

    # WHICH consumer atom the epilogue publishes for — see tilepipe/sync.py.
    # RowBlockSemaphore (default) for GEMM->combine: one counter per
    # (batch, m_tile), target ceil(N / tile_N).
    # OutputTileSemaphore for GEMM->allreduce: one counter per (batch, m_tile,
    # n_tile), target world_size.
    # The SAME object is handed to the consumer kernel, so the two sides
    # cannot disagree about the index space or the target.
    tile_semaphore: Optional[TileSemaphore] = None

    @property
    def semaphore(self) -> TileSemaphore:
        """The publish protocol, defaulting to the combine one."""
        return self.tile_semaphore if self.tile_semaphore is not None else RowBlockSemaphore()

    def validate(self, varlen_m: bool, gather_A: bool) -> None:
        if self.expert_ready_flags is not None:
            assert varlen_m, "expert_ready_flags requires varlen_m (grouped GEMM)"
            assert not gather_A, "expert_ready_flags not supported with gather_A"
        if self.tile_flag_ptrs is not None:
            # The publish lives in the epilogue's shared tile loop, so the
            # dense path works too; tile_flag_offsets is then a 1-element zero.
            assert self.tile_flag_offsets is not None, (
                "tile_flag_ptrs requires tile_flag_offsets"
            )
            if isinstance(self.semaphore, OutputTileSemaphore) and self.semaphore.multicast:
                assert self.tile_flag_ptrs.numel() == 1, (
                    "multicast publish takes a 1-entry table holding the multicast "
                    f"flag address, got {self.tile_flag_ptrs.numel()} entries"
                )
        else:
            assert self.tile_semaphore is None, (
                "tile_semaphore without tile_flag_ptrs publishes nowhere"
            )

    @property
    def compile_key(self) -> tuple:
        """The compile-cache key contribution: STATIC, HASHABLE values only.

        Never include the tensors themselves — this tuple is hashed by
        ``jit_cache``. Positionally matches ``_compile_gemm``'s
        (has_ready_flags, tile_flag_world, tile_semaphore) tail. The semaphore
        is a frozen dataclass, so its type and stride are both in the key: two
        subclasses never share a compiled kernel.
        """
        return (
            self.expert_ready_flags is not None,
            self.tile_flag_ptrs.numel() if self.tile_flag_ptrs is not None else 0,
            self.semaphore if self.tile_flag_ptrs is not None else None,
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
