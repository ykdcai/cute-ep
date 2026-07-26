"""TilePipe: comm/compute overlap for MoE (dispatch -> grouped GEMM -> combine).

Deliberately EMPTY of imports. ``quack.gemm_sm100`` imports ``tilepipe.sync``
at module scope while ``tilepipe.plan`` imports ``quack.gemm``, so eagerly
importing submodules here would re-enter a half-initialized package. Import
submodules directly on both sides:

    from tilepipe.sync import ExpertArrivalSemaphore
    from tilepipe.plan import build_combine_metadata
"""
