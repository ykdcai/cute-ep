"""
MoE Dispatch and Combine kernels ported from Triton-distributed (DeepEP) to CuTeDSL.

Single-1 (intra-node) multi-GPU implementation. All communication is
purely intra-node (P2P over NVLink), so we drop every NVSHMEM *device*
call (put_warp / get_warp / nvshmem_ptr) and instead treat a remote GPU's
symmetric buffer as a plain local pointer: NVSHMEM symmetric allocation +
`nvshmem.core.get_peer_tensor` on the host hand us each peer's base
address, and the kernels move data with ordinary CuTe copy atoms. This
mirrors the `all_reduce_simple.py` example.

Dispatch: routes MoE tokens to target expert ranks by writing the hidden
          vector directly into the remote rank's recv buffer via a
          warp-cooperative `cute.copy` (128-bit). Supports two modes
          controlled by WITH_SCATTER_INDICES:
            True  - read pre-computed scatter indices from a tensor
            False - allocate slots on-the-fly via per-warp atomic_add
                    (matching Triton-distributed kernel_dispatch_token)
Combine:  for each of a token's topk experts, reads the expert result
          slot directly from the owning rank's buffer via a SYS-scope
          VOLATILE `cute.copy` (P2P LDG over NVLink) and accumulates
          across topk to produce the final output.

To run:
    torchrun --nproc-per-node 4 examples/distributed/dispatch_and_combine.py
"""

import os
import argparse
from typing import Optional
import functools

import numpy as np
import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.pipeline as pipeline
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import Int32
import cuda.bindings.driver as cuda_driver
from cuda.core import Device

from quack.cute_dsl_utils import nanosleep
from quack.tilepipe_sync import ExpertArrivalSemaphore

import nvshmem.core

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# NVSHMEM init / finalize / compile helpers
# ---------------------------------------------------------------------------

_KERNEL_OBJECTS: list[nvshmem.core.NvshmemKernelObject] = []


def torchrun_uid_init_bcast():
    """Initialize NVSHMEM using UniqueID. Supports both launchers:

    - torchrun (single-node): reads RANK / LOCAL_RANK / WORLD_SIZE; the
      backend init_method is the c10d default (env://).
    - srun (multi-node): reads SLURM_PROCID / SLURM_LOCALID /
      SLURM_NTASKS / MASTER_ADDR / MASTER_PORT and bootstraps
      torch.distributed via the explicit `tcp://master:port` init
      method. This avoids torchrun's c10d-rendezvous backend, which
      hangs trying to bind MASTER_ADDR:port from non-master nodes.

    Either launcher converges on the same downstream code path: an
    NVSHMEM UID broadcast over the bootstrapped torch.distributed
    process group, then `nvshmem.core.init(initializer_method="uid")`.
    """
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        # torchrun path
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        init_method = None  # env:// default
    elif "SLURM_PROCID" in os.environ:
        # srun path (multi-node)
        global_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        init_method = f"tcp://{master_addr}:{master_port}"
        # The kernel reads LOCAL_WORLD_SIZE for the DeepEP node_id
        # arithmetic. SLURM exports SLURM_NTASKS_PER_NODE; mirror it
        # into LOCAL_WORLD_SIZE here so the kernel's host-side path
        # doesn't need to know about the launcher.
        os.environ.setdefault(
            "LOCAL_WORLD_SIZE",
            os.environ.get("SLURM_NTASKS_PER_NODE", str(world_size)))
    else:
        raise RuntimeError(
            "Cannot bootstrap: neither RANK/LOCAL_RANK (torchrun) nor "
            "SLURM_PROCID (srun) are set in the environment.")

    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()

    if init_method is None:
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    else:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method=init_method,
            rank=global_rank,
            world_size=world_size,
        )

    num_ranks = dist.get_world_size()

    uid = nvshmem.core.get_unique_id(empty=(global_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(device=dev, uid=uid, rank=global_rank, nranks=num_ranks,
                      initializer_method="uid")


def torchrun_finalize():
    nvshmem.core.finalize()
    dist.destroy_process_group()


def _nvshmem_device_bc():
    nvshmem_device_bc = nvshmem.core.find_device_bitcode_library()
    if not os.path.exists(nvshmem_device_bc):
        raise RuntimeError(f"NVSHMEM device bitcode not found at {nvshmem_device_bc}")
    return nvshmem_device_bc


def _compile_kernel(kernel, *example_args):
    nvshmem_device_bc = _nvshmem_device_bc()
    compiled = cute.compile(
        kernel,
        *example_args,
        options=f" --link-libraries={nvshmem_device_bc}",
    )
    compiled = compiled.to(Device().device_id)
    cuda_library = compiled.jit_module.cuda_library
    nvshmem_kernel = nvshmem.core.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.core.library_init(nvshmem_kernel)
    _KERNEL_OBJECTS.append(nvshmem_kernel)
    return compiled


def _finalize_kernels():
    while _KERNEL_OBJECTS:
        nvshmem.core.library_finalize(_KERNEL_OBJECTS.pop())


# ---------------------------------------------------------------------------
# Dispatch kernel: route tokens to remote recv buffers
# ---------------------------------------------------------------------------
# Each warp iterates over tokens assigned to it, for each token iterates topk
# experts, allocates (or reads pre-computed) a slot in the recv buffer, then
# copies the full hidden vector via warp-cooperative tiled copy.

@cute.kernel
def dispatch_kernel(
    input_buf: cute.Tensor, # [num_tokens, hidden_size] float32 (local)
    output_peer_ptrs: cute.Tensor, # [world_size] int64: base addr of each rank's output_buf
    topk_indices_tensor: cute.Tensor, # [num_tokens, topk] int32
    token_dst_scatter_idx: cute.Tensor, # [num_tokens, topk] int32
    recv_buf_offset_per_expert: cute.Pointer, # [world_size, experts_per_rank, world_size] int32
    num_input_tokens_per_rank: cute.Tensor, # [world_size] int32
    max_recv_tokens: cutlass.Int32,
    _hidden_size: cutlass.Int32,
    topk: cutlass.Int32,
    experts_per_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    num_warps: cutlass.Constexpr,
    WITH_SCATTER_INDICES: cutlass.Constexpr,
):
    hidden_size = cute.assume(_hidden_size, divby=16)
    WARP_SIZE = 32
    tidx, tidy, tidz = cute.arch.thread_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()
    bidx, bidy, bidz = cute.arch.block_idx()

    cta_nums = gdimx * gdimy * gdimz
    threads_per_cta = bdimx * bdimy * bdimz
    warps_per_cta = threads_per_cta // WARP_SIZE
    warp_id = tidx // WARP_SIZE
    global_warp_id = bidx * warps_per_cta + warp_id
    total_warps = warps_per_cta * cta_nums

    # Warp-cooperative 128-bit copy of a hidden vector: local input row ->
    # remote output row. The remote row lives on expert_rank's GPU but is
    # P2P-addressable, so we obtain its base pointer from output_peer_ptrs
    # (host-side get_peer_tensor) and store to it with an ordinary CuTe copy
    # atom (SYS scope so the write propagates to the peer before the barrier).
    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), input_buf.element_type, num_bits_per_copy=128)
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        input_buf.element_type,
        num_bits_per_copy=128,
        memory_scope=cute.nvgpu.common.MemoryScope.SYS,
        memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
    )
    thr_layout = cute.make_ordered_layout((1, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, 4), order=(1, 0))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)
    lane_id = tidx % WARP_SIZE
    thr_copy_load = tiled_copy_load.get_slice(lane_id)
    thr_copy_store = tiled_copy_store.get_slice(lane_id)

    num_tokens = num_input_tokens_per_rank[local_rank]

    # Source partition over the local input buffer (reused across tokens).
    src_layout = cute.make_ordered_layout((num_tokens, hidden_size), order=(1, 0))
    src_tensor = cute.make_tensor(input_buf.iterator.align(16), src_layout)
    tSgS = thr_copy_load.partition_S(src_tensor)
    frg = cute.make_fragment_like(tSgS[None, 0, 0])
    hidden_iter = cute.size(tSgS, mode=[2])

    for token_offset in range(global_warp_id, num_tokens, total_warps):
        for j in range(topk):
            expert_idx = topk_indices_tensor[token_offset, j]
            expert_rank = expert_idx // experts_per_rank

            if cutlass.const_expr(WITH_SCATTER_INDICES):
                store_idx = token_dst_scatter_idx[token_offset, j]
            else:
                # Atomic slot allocation (one thread per warp), matching
                # Triton-distributed atomic_add_per_warp pattern.
                expert_idx_intra_rank = expert_idx % experts_per_rank
                offset = (expert_rank * (experts_per_rank * world_size)
                          + expert_idx_intra_rank * world_size + local_rank)
                atomic_ptr = recv_buf_offset_per_expert + offset
                store_idx = Int32(0)
                if lane_id == 0:
                    store_idx = cute.arch.atomic_add(
                        atomic_ptr, Int32(1), sem="relaxed", scope="gpu")
                # Broadcast lane-0 result to all threads in the warp
                store_idx = cute.arch.shuffle_sync(store_idx, 0)
                # Write back so combine kernel can read the scatter index
                token_dst_scatter_idx[token_offset, j] = store_idx

            # Remote recv buffer on expert_rank, addressed as a local pointer.
            remote_ptr = cute.make_ptr(
                input_buf.element_type,
                output_peer_ptrs[expert_rank],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            remote_layout = cute.make_ordered_layout(
                (max_recv_tokens, hidden_size), order=(1, 0))
            remote_tensor = cute.make_tensor(remote_ptr, remote_layout)
            tDgD = thr_copy_store.partition_D(remote_tensor)

            for k in range(hidden_iter):
                cute.copy(thr_copy_load, tSgS[None, token_offset, k], frg)
                cute.copy(thr_copy_store, frg, tDgD[None, store_idx, k])


# ---------------------------------------------------------------------------
# Combine kernel: gather expert results and accumulate per-token
# ---------------------------------------------------------------------------
# Each warp iterates over tokens assigned to it, for each token iterates topk
# experts, loads the expert result from the scatter index, and accumulates
# across topk to produce the final output.

@cute.kernel
def combine_kernel(
    expert_output_peer_ptrs: cute.Tensor, # [world_size] int64: base addr of each rank's expert_output_buf
    combine_output_buf: cute.Pointer, # local [num_tokens, hidden_size] float32
    topk_indices_tensor: cute.Tensor, # [num_tokens, topk] int32
    token_dst_scatter_idx: cute.Tensor, # [num_tokens, topk] int32
    num_input_tokens_per_rank: cute.Tensor, # [world_size] int32
    max_tokens: cutlass.Int32,
    topk: cutlass.Constexpr,
    _hidden_size: cutlass.Int32,
    experts_per_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    num_warps: cutlass.Constexpr,
):
    # Intra-node combine: for each of a token's topk experts, read the
    # expert result slot directly from the owning rank's expert_output_buf
    # (P2P-addressable over NVLink) and accumulate. The remote base pointer
    # for each rank comes from expert_output_peer_ptrs (host-side
    # get_peer_tensor); the read is an ordinary CuTe copy at SYS scope
    # VOLATILE so we observe the peer's post-dispatch write.
    hidden_size = cute.assume(_hidden_size, divby=16)
    WARP_SIZE = 32
    tidx, tidy, tidz = cute.arch.thread_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()
    bidx, bidy, bidz = cute.arch.block_idx()

    cta_nums = gdimx * gdimy * gdimz
    threads_per_cta = bdimx * bdimy * bdimz
    warps_per_cta = threads_per_cta // WARP_SIZE
    warp_id = tidx // WARP_SIZE
    global_warp_id = bidx * warps_per_cta + warp_id
    total_warps = warps_per_cta * cta_nums

    # SYS-scope volatile loads for reading remote GPU memory via a P2P
    # pointer. `CopyUniversalOp` carries the load/store semantics via its
    # `memory_scope` / `memory_order` keyword arguments.
    # B200 (SM100) supports 256-bit vectorized global loads/stores. Widening
    # from 128b doubles the bytes-in-flight per outstanding remote load, which
    # directly helps this latency-bound combine (Little's law). Derive the
    # per-thread element count and alignment from this single width.
    COPY_BITS = 256
    dtype = combine_output_buf.dtype
    elems_per_copy = COPY_BITS // dtype.width
    copy_align = COPY_BITS // 8  # bytes; required alignment for the vector access
    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=COPY_BITS,
        memory_scope=cute.nvgpu.common.MemoryScope.SYS,
        memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=COPY_BITS)
    thr_layout = cute.make_ordered_layout((1, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, elems_per_copy), order=(1, 0))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)
    copy_idx = tidx % WARP_SIZE
    thr_copy_load = tiled_copy_load.get_slice(copy_idx)
    thr_copy_store = tiled_copy_store.get_slice(copy_idx)

    num_tokens = num_input_tokens_per_rank[local_rank]

    # Partition output for stores
    output_layout = cute.make_ordered_layout((num_tokens, hidden_size), order=(1, 0))
    output_tensor = cute.make_tensor(combine_output_buf.align(copy_align), output_layout)
    tDgD = thr_copy_store.partition_D(output_tensor)
    accum = cute.make_fragment_like(tDgD[None, 0, 0])
    # One register fragment per topk expert so all topk remote loads can be
    # issued before any is consumed (memory-level parallelism to hide the
    # NVLink read round-trip; see Little's law).
    frgs = [cute.make_fragment_like(tDgD[None, 0, 0]) for _ in range(topk)]
    hidden_iter = cute.size(tDgD, mode=[2])

    remote_layout = cute.make_ordered_layout((max_tokens, hidden_size), order=(1, 0))

    for token_offset in range(global_warp_id, num_tokens, total_warps):
        # Hoist the per-(token, j) remote addressing out of the hidden loop:
        # expert_rank / scatter_idx / partition depend only on (token, j), so
        # compute them topk times per token instead of hidden_iter * topk.
        remote_slices = []
        for j in cutlass.range_constexpr(topk):
            expert_idx = topk_indices_tensor[token_offset, j]
            expert_rank = expert_idx // experts_per_rank
            scatter_idx = token_dst_scatter_idx[token_offset, j]
            # expert_rank's expert_output_buf, addressed as a local pointer.
            remote_ptr = cute.make_ptr(
                dtype,
                expert_output_peer_ptrs[expert_rank],
                cute.AddressSpace.gmem,
                assumed_align=copy_align,
            )
            remote_tensor = cute.make_tensor(remote_ptr, remote_layout)
            tSgS_remote = thr_copy_load.partition_S(remote_tensor)
            remote_slices.append((tSgS_remote, scatter_idx))

        for k in range(hidden_iter):
            # Issue all topk remote loads first (in flight), then reduce.
            for j in cutlass.range_constexpr(topk):
                tSgS_remote, scatter_idx = remote_slices[j]
                cute.copy(thr_copy_load, tSgS_remote[None, scatter_idx, k], frgs[j])
            accum.fill(0.0)
            for j in cutlass.range_constexpr(topk):
                accum.store(accum.load() + frgs[j].load())
            cute.copy(thr_copy_store, accum, tDgD[None, token_offset, k])


# ---------------------------------------------------------------------------
# Host-side JIT wrappers
# ---------------------------------------------------------------------------

@cute.jit
def dispatch_jit(
    input_buf: cute.Tensor, # [num_tokens, hidden_size] float32 (local)
    output_peer_ptrs: cute.Tensor, # [world_size] int64: peer base addr of output_buf
    topk_indices_tensor: cute.Tensor, # [num_tokens, topk] int32
    token_dst_scatter_idx: cute.Tensor, # [num_tokens, topk] int32
    recv_buf_offset_per_expert: cute.Tensor, # [world_size, experts_per_rank, world_size] int32
    num_input_tokens_per_rank: cute.Tensor, # [world_size] int32
    max_recv_tokens: cutlass.Int32,
    hidden_size: cutlass.Int32,
    topk: cutlass.Int32,
    experts_per_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    WITH_SCATTER_INDICES: cutlass.Constexpr,
):

    num_warps = 32
    cta_nums = 20

    dispatch_kernel(
        input_buf,
        output_peer_ptrs,
        topk_indices_tensor,
        token_dst_scatter_idx,
        recv_buf_offset_per_expert.iterator,
        num_input_tokens_per_rank,
        max_recv_tokens,
        hidden_size,
        topk,
        experts_per_rank,
        local_rank,
        world_size,
        num_warps,
        WITH_SCATTER_INDICES,
    ).launch(
        grid=[cta_nums, 1, 1],
        block=[num_warps * 32, 1, 1],
    )


@cute.jit
def combine_jit(
    expert_output_peer_ptrs: cute.Tensor, # [world_size] int64: peer base addr of expert_output_buf
    combine_output_buf: cute.Tensor, # local [num_tokens, hidden_size] float32
    topk_indices_tensor: cute.Tensor, # [num_tokens, topk] int32
    token_dst_scatter_idx: cute.Tensor, # [num_tokens, topk] int32
    num_input_tokens_per_rank: cute.Tensor, # [world_size] int32
    max_tokens: cutlass.Int32,
    hidden_size: cutlass.Int32,
    topk: cutlass.Constexpr,
    experts_per_rank: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    cta_nums: cutlass.Constexpr,
    num_warps: cutlass.Constexpr,
):
    # Combine is remote-read bound; unlike write-bound dispatch it needs many
    # SMs active to have enough outstanding read requests (MSHRs are per-SM),
    # so cta_nums / num_warps are tunable (see --combine-cta-nums / -num-warps).
    combine_kernel(
        expert_output_peer_ptrs,
        combine_output_buf.iterator,
        topk_indices_tensor,
        token_dst_scatter_idx,
        num_input_tokens_per_rank,
        max_tokens,
        topk,
        hidden_size,
        experts_per_rank,
        local_rank,
        num_warps,
    ).launch(
        grid=[cta_nums, 1, 1],
        block=[num_warps * 32, 1, 1],
    )


# ---------------------------------------------------------------------------
# TMA-bulk combine: SM-efficient variant.
# ---------------------------------------------------------------------------
# The register combine needs ~all SMs to saturate NVLink because per-thread
# LD.256 reads keep too few bytes in flight per SM (MSHR-limited). TMA bulk
# copies whole hidden chunks (peer GMEM -> SMEM) via the copy engine, so a
# single producer warp can keep topk * chunk_bytes in flight from one CTA,
# letting a handful of SMs saturate the link (DeepEP-style). This frees the
# rest of the GPU for the expert GEMMs (comm/compute overlap).
#
# Gather addressing: each peer's expert_output_buf is flattened to 1D and
# tiled by HCHUNK. Expert row `slot` chunk `k` on any peer is tile
# `slot * num_chunks + k` (row-major), a runtime tile id. The owning rank is
# picked by a small constexpr branch over world_size descriptors.
def _per_thread_view(tile_1d, vec_size, consumer_threads, ctid):
    # (HCHUNK,) -> this consumer thread's ((vec,), (1,)) strided slice.
    # Split into groups of vec_size*consumer_threads, then each group into
    # per-thread vec_size chunks; slice out column ctid.
    by_vec = cute.logical_divide(
        cute.zipped_divide(tile_1d, (vec_size * consumer_threads,)),
        (vec_size,))
    return cute.slice_(by_vec, ((None, ctid), None))


class CombineTmaKernel:
    # All knobs are constructor args (below) so the kernel is autotunable.
    # Defaults chosen for hidden=7168 fp32 on B200.
    def __init__(self, dtype, hidden, topk,
                 hchunk: int = 3584,      # hidden elems per bulk tile (must divide hidden)
                 num_stages: int = 8,     # smem pipeline depth (>= topk for full overlap)
                 tma_threads: int = 32,   # producer warp(s); only 1 thread issues
                 consumer_threads: int = 128):
        self.dtype = dtype
        assert hidden % hchunk == 0, f"hidden {hidden} not divisible by hchunk {hchunk}"
        assert hchunk % consumer_threads == 0, (
            f"hchunk {hchunk} not divisible by consumer_threads {consumer_threads}")
        self.HCHUNK = hchunk
        self.NUM_STAGES = num_stages
        self.TMA_THREADS = tma_threads
        self.CONSUMER_THREADS = consumer_threads
        self.num_chunks = hidden // hchunk
        self.topk = topk
        self.vec_size = hchunk // consumer_threads
        self.threads_per_cta = tma_threads + consumer_threads
        self.tma_bytes = (dtype.width // 8) * hchunk

        elems = hchunk
        stages = num_stages

        @cute.struct
        class SharedStorage:
            mbar_array: cute.struct.MemRange[cutlass.Int64, stages * 2]
            smem_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, elems * stages], 128
            ]

        self._SharedStorage = SharedStorage

    @cute.jit
    def __call__(
        self,
        peer_tensors: list[cute.Tensor],   # world_size x [max_tokens, hidden]
        output_tensor: cute.Tensor,        # local [num_tokens, hidden]
        topk_indices: cute.Tensor,         # [num_tokens, topk] int32
        scatter_idx: cute.Tensor,          # [num_tokens, topk] int32
        num_input_tokens: cute.Tensor,     # [world_size] int32
        experts_per_rank: cutlass.Constexpr,
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        cta_nums: cutlass.Constexpr,
        # TilePipe GEMM->combine gate (all-or-none): local mirror of every
        # producer's tile-completion counters, the host-precomputed flat flag
        # index per (token, j) (producer-rank offset included), and the
        # per-m-tile readiness target ceil(N / tile_N).
        tile_flags: Optional[cute.Tensor] = None,
        flag_idx: Optional[cute.Tensor] = None,
        n_tiles: Optional[Int32] = None,
    ):
        # Raw 1D cp.async.bulk: each peer buffer is just flattened to 1D; the
        # producer slices a contiguous HCHUNK run by tile id. No TMA descriptor.
        peer_flat = []
        for i in cutlass.range_constexpr(world_size):
            total = cute.size(peer_tensors[i].layout)
            peer_flat.append(
                cute.make_tensor(peer_tensors[i].iterator, cute.make_layout((total,))))

        out_total = cute.size(output_tensor.layout)
        out_flat = cute.make_tensor(output_tensor.iterator, cute.make_layout((out_total,)))

        self.kernel(
            peer_flat, out_flat,
            topk_indices, scatter_idx, num_input_tokens,
            experts_per_rank, local_rank, world_size, cta_nums,
            tile_flags, flag_idx, n_tiles,
        ).launch(
            grid=[cta_nums, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            smem=self._SharedStorage.size_in_bytes(),
        )

    @cute.kernel
    def kernel(
        self,
        peer_flat: list[cute.Tensor],
        out_flat: cute.Tensor,
        topk_indices: cute.Tensor,
        scatter_idx: cute.Tensor,
        num_input_tokens: cute.Tensor,
        experts_per_rank: cutlass.Constexpr,
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        cta_nums: cutlass.Constexpr,
        tile_flags: Optional[cute.Tensor] = None,
        flag_idx: Optional[cute.Tensor] = None,
        n_tiles: Optional[Int32] = None,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        topk = cutlass.const_expr(self.topk)
        num_chunks = cutlass.const_expr(self.num_chunks)
        HCHUNK = cutlass.const_expr(self.HCHUNK)
        vec_size = cutlass.const_expr(self.vec_size)
        tiler = (HCHUNK,)

        smem = cute_utils.SmemAllocator()
        storage = smem.allocate(self._SharedStorage)
        mbar_ptr = storage.mbar_array.data_ptr()
        staged = storage.smem_buffer.get_tensor(
            cute.make_layout((HCHUNK, self.NUM_STAGES)))

        tma_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.NUM_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.CONSUMER_THREADS),
            tx_count=self.tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        num_tokens = num_input_tokens[local_rank]

        if warp_idx == 0:
            # Producer: stream topk chunks per token into smem via raw 1D bulk.
            bulk_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), self.dtype)
            prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.NUM_STAGES)
            last_flag_idx = Int32(-1)
            token = bidx
            while token < num_tokens:
                for k in cutlass.range_constexpr(num_chunks):
                    for j in cutlass.range_constexpr(topk):
                        expert_idx = topk_indices[token, j]
                        rank_j = expert_idx // experts_per_rank
                        slot_j = scatter_idx[token, j]
                        tile_id = slot_j * num_chunks + k
                        if cutlass.const_expr(tile_flags is not None and k == 0):
                            # TilePipe gate, once per source row: wait until
                            # the producing GEMM's m-tile counter reaches the
                            # N-tile count. wait_warp = elected-lane acquire
                            # poll + sync_warp + fence.proxy.async (the pull
                            # below reads through the async proxy).
                            fidx = flag_idx[token, j]
                            if fidx != last_flag_idx:
                                gate = ExpertArrivalSemaphore(flags=tile_flags)
                                gate.wait_warp(fidx, n_tiles)
                                last_flag_idx = fidx
                        tma_pipeline.producer_acquire(prod)
                        s_tile = cute.slice_(staged, (None, prod.index))
                        for r in cutlass.range_constexpr(world_size):
                            if rank_j == r:
                                g_tiled = cute.zipped_divide(peer_flat[r], tiler)
                                g_tile = g_tiled[(None,), tile_id]
                                with cute.arch.elect_one():
                                    cute.copy(
                                        bulk_atom, g_tile, s_tile,
                                        mbar_ptr=tma_pipeline.producer_get_barrier(prod))
                        tma_pipeline.producer_commit(prod)
                        prod.advance()
                token += cta_nums
        else:
            # Consumer: wait topk chunks, reduce, store local output row chunk.
            ctid = tidx - self.TMA_THREADS
            CONSUMER_THREADS = cutlass.const_expr(self.CONSUMER_THREADS)
            cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.NUM_STAGES)

            out_tiled = cute.zipped_divide(out_flat, tiler)
            token = bidx
            while token < num_tokens:
                for k in cutlass.range_constexpr(num_chunks):
                    accum = cute.make_rmem_tensor(
                        _per_thread_view(
                            cute.slice_(staged, (None, 0)),
                            vec_size, CONSUMER_THREADS, ctid).layout,
                        self.dtype)
                    accum.fill(self.dtype(0.0))
                    for j in cutlass.range_constexpr(topk):
                        tma_pipeline.consumer_wait(cons)
                        s_tile = cute.slice_(staged, (None, cons.index))
                        accum.store(
                            accum.load()
                            + _per_thread_view(
                                s_tile, vec_size, CONSUMER_THREADS, ctid).load())
                        tma_pipeline.sync_object_empty.arrive(
                            cons.index, tma_pipeline.consumer_mask)
                        cons.advance()
                    out_tile = out_tiled[(None,), token * num_chunks + k]
                    _per_thread_view(
                        out_tile, vec_size, CONSUMER_THREADS, ctid).store(accum.load())
                token += cta_nums


# ---------------------------------------------------------------------------
# TilePipe GEMM->combine gating support: tile-trickle producer emulator for
# the gated-combine test (the gate itself lives inside CombineTmaKernel via
# the optional tile_flags/flag_idx/n_tiles arguments).
# ---------------------------------------------------------------------------


@cute.kernel
def _tile_trickle_kernel(
    src: cute.Tensor,             # [rows, hidden] the data tiles should contain
    dst: cute.Tensor,             # [rows, hidden] expert-output buffer (starts garbage)
    flag_peer_ptrs: cute.Tensor,  # [world] int64: peer tile-flag base addrs
    flag_base: Int32,             # local_rank * tiles_per_rank
    rows: Int32,
    n_tiles_target: Int32,        # value to publish per tile (= N-tile count)
    delay_iters: cutlass.Constexpr,
    tile_m: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
):
    """Emulates the GEMM producer for the gated-combine test: fills the
    expert-output buffer tile by tile (data BEFORE flag, release/sys), so the
    combine's output is correct only if its gate actually waits."""
    tidx, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    hidden = cute.size(src, mode=[1])
    num_m_tiles = (rows + tile_m - 1) // tile_m
    for t in range(num_m_tiles):
        lo = t * tile_m
        hi = lo + tile_m
        if hi > rows:
            hi = rows
        # Cooperative fill of this tile's rows.
        for row in range(lo, hi):
            for h in range(tidx, hidden, bdim):
                dst[row, h] = src[row, h]
        cute.arch.barrier()
        if tidx == 0:
            for _ in cutlass.range(delay_iters):
                nanosleep(1024)
            for r in cutlass.range(world_size):
                flag_ptr = cute.make_ptr(
                    Int32, flag_peer_ptrs[r], cute.AddressSpace.gmem, assumed_align=4)
                cute.arch.atomic_add(
                    flag_ptr + flag_base + t, n_tiles_target, sem="release", scope="sys")
        cute.arch.barrier()


@cute.jit
def _tile_trickle_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    flag_peer_ptrs: cute.Tensor,
    flag_base: Int32,
    rows: Int32,
    n_tiles_target: Int32,
    delay_iters: cutlass.Constexpr,
    tile_m: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    _tile_trickle_kernel(
        src, dst, flag_peer_ptrs, flag_base, rows, n_tiles_target,
        delay_iters, tile_m, world_size,
    ).launch(grid=[1, 1, 1], block=[256, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# TMA-bulk dispatch: SM-efficient variant (CombineTmaKernel's machinery with
# the direction inverted: local G2S gather -> remote S2G push).
# ---------------------------------------------------------------------------
# The SIMT dispatch is issue-limited (~20-30 GB/s per SM); the first TMA
# version measured ~14 GB/s/SM, flat in CTA count — a per-row LATENCY
# ceiling on its single worker thread (scalar metadata loads + mbarrier
# handshakes + a stage-recycle wait that was wrongly coupled to remote WRITE
# completion). v3 attacks all three:
#
#   - WORKERS independent pipelines per CTA, each owning a contiguous
#     sub-block of the send list, a private SMEM stage partition, and private
#     bulk-group state (cp.async.bulk groups are PER-THREAD — this is the
#     warp specialization that works; a dedicated flag warp cannot observe
#     another thread's S2G completion, and the publish is one atomic per
#     segment anyway).
#   - Per worker, PRODUCER and CONSUMER are separate warps (combine-kernel
#     structure): the producer thread streams G2S (only needs send_token),
#     the consumer thread waits stages, reads slot/dst/seg, issues S2G, and
#     publishes. Their fixed per-row overheads overlap instead of
#     serializing, and a backpressure stall on one side no longer drains the
#     other.
#   - Stage recycling is gated on the S2G's SMEM READ only
#     (wait_group(read=True), ~free); remote WRITE completion is tracked
#     separately with a wide window that defines the publish watermark.
#
# Publish ordering (consumer thread): S2G stores are ASYNC-PROXY writes, so
# a segment is published only once the write watermark has passed its last
# row, and always behind fence.proxy.async before the generic-proxy release
# (else the flag can beat the data). Boundaries record a pending
# (segment, count, row); tiny segments (two boundaries inside one window)
# fall back to a hard drain. The counting protocol is arrival-order-
# agnostic, so workers/CTAs sharing a boundary segment is fine.


@cute.jit
def _publish_segment_tma(
    seg: Int32,
    count: Int32,
    seg_done: cute.Tensor,
    seg_sizes: cute.Tensor,
    flag_peer_ptrs: cute.Tensor,
    local_rank: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
):
    # Caller must have ensured this thread's S2G writes for the segment are
    # complete (wait_group) AND fenced (fence.proxy.async). Single-thread.
    old = cute.arch.atomic_add(
        seg_done.iterator + seg, count, sem="acq_rel", scope="sys")
    seg_size = seg_sizes[seg]
    if old + count == seg_size:
        e = seg // world_size
        dst = (seg % world_size + local_rank) % world_size
        sem = ExpertArrivalSemaphore(peer_ptrs=flag_peer_ptrs)
        sem.arrive(dst, e, seg_size)


class DispatchTmaKernel:
    """WORKERS producer/consumer warp pairs per CTA, each running a private
    (num_stages // workers)-deep bulk pipeline over a contiguous sub-block of
    the send list. hidden is compile-time (SMEM stage size)."""

    WRITE_WINDOW = 8  # in-flight remote writes per worker (publish watermark)

    def __init__(self, dtype, hidden, num_stages: int = 12, workers: int = 4):
        self.dtype = dtype
        self.hidden = hidden
        self.NUM_STAGES = num_stages
        self.WORKERS = workers
        assert num_stages % workers == 0, "num_stages must divide by workers"
        assert num_stages // workers >= 2, "each worker needs >=2 stages"
        row_bytes = hidden * dtype.width // 8
        assert row_bytes % 16 == 0, "bulk copy needs 16B-aligned rows"
        self.tx_count = row_bytes
        stages = num_stages
        elems = hidden

        @cute.struct
        class SharedStorage:
            mbar_array: cute.struct.MemRange[cutlass.Int64, stages * 2]
            smem_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, elems * stages], 128
            ]

        self._SharedStorage = SharedStorage

    @cute.jit
    def __call__(
        self,
        input_buf: cute.Tensor,       # [num_tokens, hidden] local token data
        recv_peer_ptrs: cute.Tensor,  # [world] int64
        flag_peer_ptrs: cute.Tensor,  # [world] int64
        send_token: cute.Tensor,      # [total_sends] int32
        send_slot: cute.Tensor,       # [total_sends] int32
        send_dst: cute.Tensor,        # [total_sends] int32
        send_seg: cute.Tensor,        # [total_sends] int32
        seg_done: cute.Tensor,        # [num_segs] int32
        seg_sizes: cute.Tensor,       # [num_segs] int32
        total_sends: Int32,
        max_recv_tokens: Int32,
        num_ctas: Int32,
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
    ):
        in_total = cute.size(input_buf.layout)
        in_flat = cute.make_tensor(input_buf.iterator, cute.make_layout((in_total,)))
        self.kernel(
            in_flat, recv_peer_ptrs, flag_peer_ptrs,
            send_token, send_slot, send_dst, send_seg, seg_done, seg_sizes,
            total_sends, max_recv_tokens, local_rank, world_size,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.WORKERS * 64, 1, 1],  # producer warp + consumer warp each
            smem=self._SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        in_flat: cute.Tensor,
        recv_peer_ptrs: cute.Tensor,
        flag_peer_ptrs: cute.Tensor,
        send_token: cute.Tensor,
        send_slot: cute.Tensor,
        send_dst: cute.Tensor,
        send_seg: cute.Tensor,
        seg_done: cute.Tensor,
        seg_sizes: cute.Tensor,
        total_sends: Int32,
        max_recv_tokens: Int32,
        local_rank: cutlass.Constexpr,
        world_size: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        gdim = cute.arch.grid_dim()[0]
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = tidx % 32
        WORKERS = cutlass.const_expr(self.WORKERS)
        SPW = cutlass.const_expr(self.NUM_STAGES // self.WORKERS)
        WWIN = cutlass.const_expr(self.WRITE_WINDOW)
        hidden = cutlass.const_expr(self.hidden)
        tiler = (hidden,)

        smem = cute_utils.SmemAllocator()
        storage = smem.allocate(self._SharedStorage)
        staged_all = storage.smem_buffer.get_tensor(
            cute.make_layout((hidden, self.NUM_STAGES)))
        mbar_base = storage.mbar_array.data_ptr()
        # One private pipeline per worker (created by all threads: create()
        # includes the block-wide init sync).
        pipes = []
        for w in cutlass.range_constexpr(WORKERS):
            pipes.append(pipeline.PipelineTmaAsync.create(
                barrier_storage=mbar_base + w * SPW * 2,
                num_stages=SPW,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.tx_count,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            ))

        src_tiled = cute.zipped_divide(in_flat, tiler)
        g2s_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), self.dtype)
        s2g_atom = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), self.dtype)

        for w in cutlass.range_constexpr(WORKERS):
            # Contiguous sub-block for worker w of this CTA.
            num_workers_total = gdim * WORKERS
            wid = bidx * WORKERS + w
            per_w = (total_sends + num_workers_total - 1) // num_workers_total
            block_start = wid * per_w
            n = total_sends - block_start
            if n > per_w:
                n = per_w
            if n < 0:
                n = Int32(0)
            staged = cute.make_tensor(
                staged_all.iterator + w * SPW * hidden,
                cute.make_layout((hidden, SPW)))

            if warp_id == w:
                # ---- Producer warp: stream rows into SMEM stages. ----
                if lane_id == 0:
                    prod = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, SPW)
                    for i in range(n):
                        pipes[w].producer_acquire(prod)
                        tok = send_token[block_start + i]
                        s_tile = cute.slice_(staged, (None, prod.index))
                        g_tile = src_tiled[(None,), tok]
                        cute.copy(g2s_atom, g_tile, s_tile,
                                  mbar_ptr=pipes[w].producer_get_barrier(prod))
                        pipes[w].producer_commit(prod)
                        prod.advance()

            if warp_id == WORKERS + w:
                # ---- Consumer warp: push staged rows to the destination,
                # recycle stages on SMEM-read completion, publish segments
                # behind the write watermark. ----
                if lane_id == 0:
                    cons = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, SPW)
                    rel = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, SPW)
                    cur_seg = Int32(-1)
                    count = Int32(0)
                    pend_seg = Int32(-1)
                    pend_count = Int32(0)
                    pend_row = Int32(0)
                    for i in range(n):
                        pipes[w].consumer_wait(cons)  # row i staged

                        seg = send_seg[block_start + i]
                        if seg != cur_seg:
                            if count > 0:
                                if pend_seg >= 0:
                                    # Two boundaries inside one write window
                                    # (tiny segments): hard-drain the older.
                                    cute.arch.cp_async_bulk_wait_group(0)
                                    cute.arch.fence_proxy("async")
                                    _publish_segment_tma(
                                        pend_seg, pend_count, seg_done,
                                        seg_sizes, flag_peer_ptrs,
                                        local_rank, world_size)
                                pend_seg = cur_seg
                                pend_count = count
                                pend_row = i - 1
                            cur_seg = seg
                            count = Int32(0)

                        slot = send_slot[block_start + i]
                        dst = send_dst[block_start + i]
                        r_ptr = cute.make_ptr(
                            self.dtype, recv_peer_ptrs[dst],
                            cute.AddressSpace.gmem, assumed_align=16)
                        r_flat = cute.make_tensor(
                            r_ptr, cute.make_layout((max_recv_tokens * hidden,)))
                        d_tile = cute.zipped_divide(r_flat, tiler)[(None,), slot]
                        s_tile = cute.slice_(staged, (None, cons.index))
                        cute.copy(s2g_atom, s_tile, d_tile)
                        cute.arch.cp_async_bulk_commit_group()
                        # Recycle the previous stage as soon as its SMEM read
                        # is done (cheap); remote writes stay in flight.
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                        if i >= 1:
                            pipes[w].consumer_release(rel)
                            rel.advance()
                        # Bound in-flight writes; rows <= i - WWIN complete.
                        cute.arch.cp_async_bulk_wait_group(WWIN)
                        if pend_seg >= 0:
                            if pend_row <= i - WWIN:
                                cute.arch.fence_proxy("async")
                                _publish_segment_tma(
                                    pend_seg, pend_count, seg_done, seg_sizes,
                                    flag_peer_ptrs, local_rank, world_size)
                                pend_seg = Int32(-1)
                        cons.advance()
                        count += 1

                    # Tail: drain everything, publish what's left.
                    if n > 0:
                        cute.arch.cp_async_bulk_wait_group(0)
                        cute.arch.fence_proxy("async")
                        if pend_seg >= 0:
                            _publish_segment_tma(
                                pend_seg, pend_count, seg_done, seg_sizes,
                                flag_peer_ptrs, local_rank, world_size)
                        if count > 0:
                            _publish_segment_tma(
                                cur_seg, count, seg_done, seg_sizes,
                                flag_peer_ptrs, local_rank, world_size)


# ---------------------------------------------------------------------------
# Peer-pointer helper: map an NVSHMEM symmetric tensor to each rank's base
# address so kernels can address remote buffers as plain local pointers.
# ---------------------------------------------------------------------------

def _peer_ptr_tensor(symm_tensor, world_size, device):
    """Return an int64[world_size] tensor of each rank's P2P base address for
    `symm_tensor` (obtained via nvshmem.core.get_peer_tensor). Indexed by a
    runtime rank inside the kernel via cute.make_ptr."""
    return torch.tensor(
        [nvshmem.core.get_peer_tensor(symm_tensor, r).data_ptr()
         for r in range(world_size)],
        device=device, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Reference checks
# ---------------------------------------------------------------------------

# Route classes a token-to-expert hop can fall into. The validator
# counts hits across (source_rank, dest_rank) pairs; a multi-node run
# should hit at least the first three of these (and ideally all four
# for non-trivial cluster shapes — `2x4` hits "self" and three remote
# classes; some `Nx1` shapes have no intra-node remote class).
_ROUTE_CLASSES = (
    "self",                             # peer == rank
    "intra_node_remote",                # peer != rank, same node
    "inter_node_same_local_rank",       # peer's local_rank == my local_rank
    "inter_node_diff_local_rank",       # peer's local_rank != my local_rank
)


def _classify_route(src_rank: int, dst_rank: int, local_world_size: int) -> str:
    """Classify a (src_rank -> dst_rank) hop into one of the four route
    classes used by the route_class_hits validator. The classification
    matches the per-peer routing the kernel performs."""
    if src_rank == dst_rank:
        return "self"
    src_node = src_rank // local_world_size
    dst_node = dst_rank // local_world_size
    if src_node == dst_node:
        return "intra_node_remote"
    if src_rank % local_world_size == dst_rank % local_world_size:
        return "inter_node_same_local_rank"
    return "inter_node_diff_local_rank"


def check_dispatch(input_buf, output_buf, topk_indices, scatter_idx,
                    num_tokens, topk, experts_per_rank, rank, world_size,
                    local_world_size=None, route_class_hits=None):
    """Verify multi-GPU dispatch correctness.
    Allgather inputs/indices from all ranks, then check that each rank's
    output_buf contains the correct data from the source ranks.

    If `local_world_size` and `route_class_hits` are provided, this also
    counts each (src_rank -> dst_rank) hop into the four route classes
    (self, intra_node_remote, inter_node_same_local_rank,
    inter_node_diff_local_rank). The caller asserts the expected
    classes were hit at least once after both dispatch tests run.
    """
    device = input_buf.device

    # Gather inputs from all ranks
    all_inputs = [torch.zeros_like(input_buf) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_buf.contiguous())

    # Gather topk_indices from all ranks
    all_topk = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_indices.contiguous())

    # Gather scatter_idx from all ranks
    all_scatter = [torch.zeros_like(scatter_idx) for _ in range(world_size)]
    dist.all_gather(all_scatter, scatter_idx.contiguous())

    # Gather output_buf from all ranks so GPU0 can verify cross-GPU transfers
    all_outputs = [torch.zeros_like(output_buf) for _ in range(world_size)]
    dist.all_gather(all_outputs, output_buf.contiguous())

    # --- Trace a few tokens from GPU0 to show dispatch paths ---
    if rank == 0:
        src = 0
        for t in [0, num_tokens // 2, num_tokens - 1]:
            experts = all_topk[src][t]
            dests = experts // experts_per_rank
            scatters = all_scatter[src][t]
            print(f"  [trace] GPU{src} token {t}: topk_experts={experts.tolist()}, "
                  f"dest_ranks={dests.tolist()}, scatter_idx={scatters.tolist()}")
            for j in range(topk):
                dst_rank = dests[j].item()
                slot = scatters[j].item()
                src_data = all_inputs[src][t, :4]
                actual_data = all_outputs[dst_rank][slot, :4]
                match = torch.allclose(actual_data, src_data, atol=1e-6, rtol=1e-5)
                print(f"    topk[{j}]: expert {experts[j].item()} -> GPU{dst_rank} "
                      f"output_buf[{slot}]  {'MATCH' if match else 'MISMATCH'}")

    # --- Full correctness check ---
    errors = 0
    for src_rank in range(world_size):
        dest_ranks = all_topk[src_rank] // experts_per_rank  # [num_tokens, topk]

        # Route classification: every (src_rank, dst_rank) hop in the
        # source rank's topk routing fans into one of four classes.
        # We count from rank 0's perspective only — other ranks see the
        # same classification because dest_ranks is allgathered.
        if route_class_hits is not None and local_world_size is not None and rank == 0:
            for j in range(topk):
                for dst in dest_ranks[:, j].tolist():
                    cls = _classify_route(src_rank, int(dst), local_world_size)
                    route_class_hits[cls] = route_class_hits.get(cls, 0) + 1

        for j in range(topk):
            mask = (dest_ranks[:, j] == rank)
            if not mask.any():
                continue

            token_indices = mask.nonzero(as_tuple=True)[0]
            scatter_indices = all_scatter[src_rank][token_indices, j].long()

            expected = all_inputs[src_rank][token_indices]
            actual = output_buf[scatter_indices]

            if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
                mismatch = ~torch.isclose(actual, expected, atol=1e-6, rtol=1e-5)
                num_mismatch = mismatch.any(dim=1).sum().item()
                errors += num_mismatch
                if rank == 0:
                    first_bad = mismatch.any(dim=1).nonzero()[0].item()
                    print(f"  [dispatch] FAIL: src={src_rank} topk={j}, "
                          f"{num_mismatch} mismatches")
                    print(f"    token {token_indices[first_bad].item()}: "
                          f"got {actual[first_bad, :4]}, "
                          f"expected {expected[first_bad, :4]}")

    return errors == 0


def check_combine(combine_output, expert_output_buf, topk_indices,
                   token_dst_scatter_idx, num_tokens, topk, experts_per_rank,
                   rank, world_size):
    """Verify multi-GPU combine correctness:
    combine_output[t] == sum_j(remote_expert_output[expert_rank_j][scatter_idx[t,j]])
    where expert_rank_j = topk_indices[t,j] // experts_per_rank
    """
    # Gather expert_output_buf from all ranks (each GPU's dispatch output)
    all_expert_outputs = [torch.zeros_like(expert_output_buf) for _ in range(world_size)]
    dist.all_gather(all_expert_outputs, expert_output_buf.contiguous())
    all_expert_stacked = torch.stack(all_expert_outputs)  # [world_size, max_tokens, hidden]

    expected = torch.zeros_like(combine_output[:num_tokens])
    for j in range(topk):
        expert_ranks = (topk_indices[:num_tokens, j] // experts_per_rank).long()
        slots = token_dst_scatter_idx[:num_tokens, j].long()
        expected += all_expert_stacked[expert_ranks, slots]

    if torch.allclose(combine_output[:num_tokens], expected, atol=1e-4, rtol=1e-4):
        return True

    mismatch = ~torch.isclose(combine_output[:num_tokens], expected, atol=1e-4, rtol=1e-4)
    num_mismatch = mismatch.any(dim=1).sum().item()
    if rank == 0:
        print(f"  [combine] FAIL: {num_mismatch}/{num_tokens} token mismatches")
        first_bad = mismatch.any(dim=1).nonzero()[0].item()
        print(f"    token {first_bad}: got {combine_output[first_bad, :4]}, "
              f"expected {expected[first_bad, :4]}")
    return False


# ---------------------------------------------------------------------------
# Main: run dispatch -> (simulated expert) -> combine, with ref checks
# ---------------------------------------------------------------------------

def run_moe_dispatch_combine(num_tokens, hidden, num_experts, topk,
                             benchmark=False, warmup_iterations=10,
                             iterations=100, save_baseline_to=None,
                             bench_only="both",
                             combine_cta_nums=148, combine_num_warps=32,
                             combine_impl="reg", combine_tma_cfg=None,
                             autotune=False):
    combine_tma_cfg = combine_tma_cfg or {}
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    experts_per_rank = num_experts // world_size

    # DeepEP convention: local_world_size == nproc_per_node, node_id ==
    # rank // local_world_size. Single-node runs collapse to
    # local_world_size == world_size.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"WORLD_SIZE ({world_size}) must be divisible by LOCAL_WORLD_SIZE "
            f"({local_world_size}); the topology assumes equal-size nodes.")
    nnodes = world_size // local_world_size
    node_id = rank // local_world_size

    if rank == 0:
        print(f"\nMoE Dispatch test (intra-node P2P cute.copy, multi-GPU):")
        print(f"  num_tokens={num_tokens}, hidden={hidden}, "
              f"num_experts={num_experts}, topk={topk}")
        print(f"  world_size={world_size}, experts_per_rank={experts_per_rank}")
        print(f"  nnodes={nnodes}, local_world_size={local_world_size}")

    device = torch.cuda.current_device()
    torch.manual_seed(42 + rank)

    # --- Generate test data ---
    topk_indices_tensor = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)

    # Compute destination rank for each (token, topk_slot)
    dest_ranks = topk_indices_tensor.int() // experts_per_rank  # [num_tokens, topk]

    # Count tokens sent to each destination rank
    send_counts = torch.zeros(world_size, dtype=torch.int64, device=device)
    for d in range(world_size):
        send_counts[d] = (dest_ranks == d).sum()

    # All-gather send_counts to compute scatter indices without collisions
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)

    # Compute max_recv_tokens (must be same on all ranks for nvshmem symmetric alloc)
    max_recv_tokens = 0
    for d in range(world_size):
        total = sum(all_send_counts[s][d].item() for s in range(world_size))
        max_recv_tokens = max(max_recv_tokens, int(total))
    max_recv_tokens = max(max_recv_tokens, 1)

    if rank == 0:
        print(f"  max_recv_tokens={max_recv_tokens}")

    # ------------------------------------------------------------------
    # Compute recv_buf_offset_per_expert
    # Layout: [expert_rank, expert_idx_intra_rank, src_rank]
    # Used by both WITH_SCATTER_INDICES modes
    # ------------------------------------------------------------------
    local_expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    flat_expert_indices = topk_indices_tensor.reshape(-1).long()
    local_expert_counts.scatter_add_(
        0, flat_expert_indices,
        torch.ones(flat_expert_indices.shape[0], dtype=torch.int32, device=device))

    all_expert_counts = [torch.zeros_like(local_expert_counts) for _ in range(world_size)]
    dist.all_gather(all_expert_counts, local_expert_counts.contiguous())

    recv_buf_offset_data = torch.zeros(
        (world_size, experts_per_rank, world_size), dtype=torch.int32, device=device)
    for er in range(world_size):
        cumsum_val = 0
        for ei in range(experts_per_rank):
            expert_global = er * experts_per_rank + ei
            for sr in range(world_size):
                recv_buf_offset_data[er, ei, sr] = cumsum_val
                cumsum_val += all_expert_counts[sr][expert_global].item()

    # Assign scatter indices using the same expert-first layout as recv_buf_offset
    # This makes WITH_SCATTER_INDICES=True produce the same output layout as =False
    token_dst_scatter_idx = torch.zeros(
        (num_tokens, topk), dtype=torch.int32, device=device)
    counters = recv_buf_offset_data.clone()  # [world_size, experts_per_rank, world_size]
    for t in range(num_tokens):
        for j in range(topk):
            expert_idx = topk_indices_tensor[t, j].item()
            expert_rank = expert_idx // experts_per_rank
            expert_idx_intra_rank = expert_idx % experts_per_rank
            token_dst_scatter_idx[t, j] = counters[expert_rank, expert_idx_intra_rank, rank]
            counters[expert_rank, expert_idx_intra_rank, rank] += 1

    # Save random input so both test modes use identical data
    input_data = torch.randn((num_tokens, hidden), dtype=torch.float32, device=device)

    num_input_tokens_per_rank = torch.full(
        (world_size,), num_tokens, dtype=torch.int32, device=device)

    all_pass = True
    # Tally each topk hop into one of the four route classes across both
    # dispatch tests. Multi-node runs should hit at least three of these
    # (or all four for non-trivial topology).
    route_class_hits: dict[str, int] = {cls: 0 for cls in _ROUTE_CLASSES}

    # =====================================================================
    # Test 1: WITH_SCATTER_INDICES=False  (atomic slot allocation)
    # =====================================================================
    if rank == 0:
        print("\n--- WITH_SCATTER_INDICES=False (atomic allocation) ---")

    input_buf = nvshmem.core.tensor((num_tokens, hidden), dtype=torch.float32)
    input_buf.copy_(input_data)
    output_buf = nvshmem.core.tensor((max_recv_tokens, hidden), dtype=torch.float32)
    output_buf.fill_(0)

    scatter_idx_atomic = torch.zeros(
        (num_tokens, topk), dtype=torch.int32, device=device)
    recv_buf_offset = recv_buf_offset_data.clone()

    output_peer_ptrs = _peer_ptr_tensor(output_buf, world_size, device)

    input_cute   = from_dlpack(input_buf)
    output_peer_cute = from_dlpack(output_peer_ptrs)
    topk_idx_cute = from_dlpack(topk_indices_tensor)
    scatter_atomic_cute = from_dlpack(scatter_idx_atomic)
    recv_cute    = from_dlpack(recv_buf_offset)
    ntok_cute    = from_dlpack(num_input_tokens_per_rank)

    if rank == 0:
        print("Compiling dispatch kernel (WITH_SCATTER_INDICES=False)...")
    compiled_atomic = _compile_kernel(
        dispatch_jit,
        input_cute, output_peer_cute, topk_idx_cute, scatter_atomic_cute,
        recv_cute, ntok_cute, max_recv_tokens, hidden,
        topk, experts_per_rank, rank, world_size, False,
    )

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("Running dispatch kernel...")
    compiled_atomic(
        input_cute, output_peer_cute, topk_idx_cute, scatter_atomic_cute,
        recv_cute, ntok_cute, max_recv_tokens, hidden,
        topk,
    )

    torch.cuda.synchronize()
    nvshmem_stream = Device().create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=nvshmem_stream)
    nvshmem_stream.sync()

    ok_atomic = check_dispatch(
        input_buf, output_buf, topk_indices_tensor, scatter_idx_atomic,
        num_tokens, topk, experts_per_rank, rank, world_size,
        local_world_size=local_world_size, route_class_hits=route_class_hits)
    if rank == 0:
        print(f"  Dispatch check: {'PASS' if ok_atomic else 'FAIL'}")
    all_pass = all_pass and ok_atomic

    nvshmem.core.free_tensor(input_buf)
    nvshmem.core.free_tensor(output_buf)

    # =====================================================================
    # Test 2: WITH_SCATTER_INDICES=True  (pre-computed scatter indices)
    # =====================================================================
    if rank == 0:
        print("\n--- WITH_SCATTER_INDICES=True (pre-computed scatter) ---")

    input_buf = nvshmem.core.tensor((num_tokens, hidden), dtype=torch.float32)
    input_buf.copy_(input_data)
    output_buf = nvshmem.core.tensor((max_recv_tokens, hidden), dtype=torch.float32)
    output_buf.fill_(0)

    # Dummy recv_buf_offset (unused in this mode, but must be passed)
    recv_buf_offset_dummy = torch.zeros(
        (world_size, experts_per_rank, world_size), dtype=torch.int32, device=device)

    output_peer_ptrs = _peer_ptr_tensor(output_buf, world_size, device)

    input_cute   = from_dlpack(input_buf)
    output_peer_cute = from_dlpack(output_peer_ptrs)
    topk_idx_cute = from_dlpack(topk_indices_tensor)
    scatter_cute = from_dlpack(token_dst_scatter_idx)
    recv_dummy_cute = from_dlpack(recv_buf_offset_dummy)
    ntok_cute    = from_dlpack(num_input_tokens_per_rank)

    if rank == 0:
        print("Compiling dispatch kernel (WITH_SCATTER_INDICES=True)...")
    compiled_scatter = _compile_kernel(
        dispatch_jit,
        input_cute, output_peer_cute, topk_idx_cute, scatter_cute,
        recv_dummy_cute, ntok_cute, max_recv_tokens, hidden,
        topk, experts_per_rank, rank, world_size, True,
    )

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("Running dispatch kernel...")
    compiled_scatter(
        input_cute, output_peer_cute, topk_idx_cute, scatter_cute,
        recv_dummy_cute, ntok_cute, max_recv_tokens, hidden,
        topk,
    )

    torch.cuda.synchronize()
    nvshmem_stream = Device().create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=nvshmem_stream)
    nvshmem_stream.sync()

    ok_scatter = check_dispatch(
        input_buf, output_buf, topk_indices_tensor, token_dst_scatter_idx,
        num_tokens, topk, experts_per_rank, rank, world_size,
        local_world_size=local_world_size, route_class_hits=route_class_hits)
    if rank == 0:
        print(f"  Dispatch check: {'PASS' if ok_scatter else 'FAIL'}")
    all_pass = all_pass and ok_scatter

    nvshmem.core.free_tensor(input_buf)

    # =====================================================================
    # Test 3: Combine (dispatch -> identity expert -> combine)
    # Reuse output_buf from dispatch as expert_output_buf (identity expert).
    # Combine reads from remote GPUs' output_buf and accumulates per token.
    # Expected: combine_output[t] = sum_j(input_data[t]) = topk * input_data[t]
    # =====================================================================
    if rank == 0:
        print("\n--- Combine (using dispatch scatter_idx, identity expert) ---")

    combine_output_buf = torch.zeros(
        (num_tokens, hidden), dtype=torch.float32, device=device)

    # Peer base addresses of output_buf (the identity expert output, filled
    # by dispatch). Combine reads each expert's slot from the owning rank's
    # buffer directly over NVLink.
    expert_output_peer_ptrs = _peer_ptr_tensor(output_buf, world_size, device)

    expert_output_peer_cute = from_dlpack(expert_output_peer_ptrs)
    combine_cute = from_dlpack(combine_output_buf)
    topk_idx_cute = from_dlpack(topk_indices_tensor)
    scatter_cute = from_dlpack(token_dst_scatter_idx)
    ntok_cute = from_dlpack(num_input_tokens_per_rank)

    if rank == 0:
        print(f"Compiling combine kernel ({combine_impl})...")
    if combine_impl == "tma":
        # TMA path takes peer output tensors directly (descriptors built
        # host-side from each peer's buffer); config is baked as constexpr.
        peer_output_tensors = [
            from_dlpack(nvshmem.core.get_peer_tensor(output_buf, r))
            for r in range(world_size)]
        tma_kernel = CombineTmaKernel(cutlass.Float32, hidden, topk, **combine_tma_cfg)
        combine_run_args = (
            peer_output_tensors, combine_cute,
            topk_idx_cute, scatter_cute, ntok_cute)
        compiled_combine = cute.compile(
            tma_kernel, *combine_run_args,
            experts_per_rank, rank, world_size, combine_cta_nums)
    else:
        combine_run_args = (
            expert_output_peer_cute, combine_cute,
            topk_idx_cute, scatter_cute, ntok_cute, max_recv_tokens, hidden)
        compiled_combine = _compile_kernel(
            combine_jit, *combine_run_args,
            topk, experts_per_rank, rank, combine_cta_nums, combine_num_warps)

    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("Running combine kernel...")
    compiled_combine(*combine_run_args)

    torch.cuda.synchronize()
    nvshmem_stream = Device().create_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=nvshmem_stream)
    nvshmem_stream.sync()

    ok_combine = check_combine(
        combine_output_buf, output_buf, topk_indices_tensor,
        token_dst_scatter_idx, num_tokens, topk, experts_per_rank,
        rank, world_size)
    if rank == 0:
        print(f"  Combine check: {'PASS' if ok_combine else 'FAIL'}")
    all_pass = all_pass and ok_combine

    # Route-class coverage check. The classes that MUST be hit
    # depend on topology:
    #   - nnodes == 1 and local_world_size == 1: only "self".
    #   - nnodes == 1 and local_world_size >= 2: "self" + "intra_node_remote".
    #   - nnodes >= 2: all four classes must be hit.
    # Missing required classes is a hard test failure: if the kernel's
    # per-peer route is under-exercised, a regression in that route
    # would slip past correctness silently. The check runs on rank 0
    # (where route_class_hits is populated) and feeds into all_pass
    # BEFORE the global reduce, so an asymmetric flip propagates
    # cleanly to every rank without deadlocking the next collective.
    if rank == 0:
        print("\nRoute class hits (across both dispatch tests):")
        for cls in _ROUTE_CLASSES:
            n = route_class_hits.get(cls, 0)
            marker = "*" if n > 0 else "-"
            print(f"  {marker} {cls}: {n}")
        expected_classes = ["self"]
        if local_world_size >= 2:
            expected_classes.append("intra_node_remote")
        if nnodes >= 2:
            expected_classes += ["inter_node_same_local_rank",
                                 "inter_node_diff_local_rank"]
        missing = [cls for cls in expected_classes
                   if route_class_hits.get(cls, 0) == 0]
        if missing:
            print(f"\n  ROUTE-CLASS COVERAGE FAIL: nnodes={nnodes} "
                  f"local_world_size={local_world_size}; required classes "
                  f"{expected_classes} not all populated; missing {missing}. "
                  "The kernel's per-peer route is under-exercised — increase "
                  "--num_tokens or --num_experts so cross-node routing fires "
                  "in both the inter_node_same_local_rank and "
                  "inter_node_diff_local_rank dimensions.")
            all_pass = False

    # CRITICAL: `all_pass` is rank-local (each rank computed its own
    # check_dispatch / check_combine result; rank 0 also folds in the
    # route-class coverage gate above). Any subsequent code path that
    # does a collective (dist.all_gather for the baseline writer, or
    # dist.barrier in the benchmark) MUST gate on a global pass —
    # otherwise an asymmetric pass/fail across ranks deadlocks. Reduce
    # the local bool with op=MIN so any single rank's failure (including
    # rank 0's route-class fail) flips the global flag false on every
    # rank.
    pass_tensor = torch.tensor(int(all_pass), dtype=torch.int32, device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    all_pass = bool(pass_tensor.item())

    # =====================================================================
    if rank == 0:
        print(f"\n{'='*40}")
        if all_pass:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
        print(f"{'='*40}")

    # Optional: persist outputs as a reference snapshot for the parity
    # checker. All ranks contribute via dist.all_gather; rank 0 writes
    # the consolidated .npz so the consumer can verify bit-identity
    # of every rank's output_buf / combine_output_buf / scatter_idx
    # against a fresh run. Single-rank-only snapshots are insufficient
    # because non-rank-0 ranks dispatch into THEIR OWN output_buf, and
    # the combine path on rank N can only validate against the
    # all-rank dispatch result.
    if save_baseline_to is not None and all_pass:
        import numpy as np

        def _gather_to_rank0(tensor):
            """All-gather a tensor; only rank 0 returns the stacked
            tensor (shape [world_size, *tensor.shape]). Other ranks
            return None to signal "don't write."""
            tensor = tensor.contiguous()
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
            if rank == 0:
                return torch.stack(gathered, dim=0)
            return None

        output_buf_by_rank = _gather_to_rank0(output_buf)
        combine_buf_by_rank = _gather_to_rank0(combine_output_buf)
        topk_by_rank = _gather_to_rank0(topk_indices_tensor)
        scatter_by_rank = _gather_to_rank0(token_dst_scatter_idx)

        if rank == 0:
            save_dir = os.path.dirname(save_baseline_to)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            # Capture non-data metadata so a consumer can detect
            # baseline / fresh-run shape or topology mismatch up front.
            try:
                hostname = os.uname().nodename
            except Exception:
                hostname = "unknown"
            np.savez(
                save_baseline_to,
                # all-rank arrays
                output_buf_by_rank=output_buf_by_rank.detach().cpu().numpy(),
                combine_output_buf_by_rank=combine_buf_by_rank.detach().cpu().numpy(),
                topk_indices_by_rank=topk_by_rank.detach().cpu().numpy(),
                scatter_idx_by_rank=scatter_by_rank.detach().cpu().numpy(),
                # legacy rank-0 arrays kept for backwards-compat readers
                output_buf=output_buf.detach().cpu().numpy(),
                combine_output_buf=combine_output_buf.detach().cpu().numpy(),
                topk_indices=topk_indices_tensor.detach().cpu().numpy(),
                scatter_idx=token_dst_scatter_idx.detach().cpu().numpy(),
                # shape / topology metadata
                world_size=np.int32(world_size),
                local_world_size=np.int32(local_world_size),
                nnodes=np.int32(nnodes),
                num_tokens=np.int32(num_tokens),
                hidden=np.int32(hidden),
                num_experts=np.int32(num_experts),
                topk=np.int32(topk),
                seed=np.int32(42),  # base seed; per-rank seed = base + rank
                hostname=np.array(hostname),
            )
            print(f"  baseline saved to {save_baseline_to} "
                  f"(all-rank: {output_buf_by_rank.shape[0]} ranks)")

    # =====================================================================
    # Benchmark (reuse compiled kernels and NVSHMEM buffers)
    # =====================================================================
    if benchmark and all_pass:
        # Re-allocate input_buf for dispatch benchmark (was freed earlier)
        input_buf = nvshmem.core.tensor((num_tokens, hidden), dtype=torch.float32)
        input_buf.copy_(input_data)
        output_peer_ptrs = _peer_ptr_tensor(output_buf, world_size, device)
        input_cute = from_dlpack(input_buf)
        output_peer_cute = from_dlpack(output_peer_ptrs)
        topk_idx_cute = from_dlpack(topk_indices_tensor)
        scatter_cute = from_dlpack(token_dst_scatter_idx)
        recv_dummy_cute = from_dlpack(recv_buf_offset_dummy)
        ntok_cute = from_dlpack(num_input_tokens_per_rank)

        # Run dispatch once to fill output_buf for combine benchmark
        compiled_scatter(
            input_cute, output_peer_cute, topk_idx_cute, scatter_cute,
            recv_dummy_cute, ntok_cute, max_recv_tokens, hidden, topk,
        )
        torch.cuda.synchronize()
        nvshmem_stream = Device().create_stream()
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=nvshmem_stream)
        nvshmem_stream.sync()

        combine_output_buf2 = torch.zeros(
            (num_tokens, hidden), dtype=torch.float32, device=device)
        combine_cute2 = from_dlpack(combine_output_buf2)

        dispatch_args = (
            input_cute, output_peer_cute, topk_idx_cute, scatter_cute,
            recv_dummy_cute, ntok_cute, max_recv_tokens, hidden, topk,
        )
        if combine_impl == "tma":
            peer_output_tensors = [
                from_dlpack(nvshmem.core.get_peer_tensor(output_buf, r))
                for r in range(world_size)]
            combine_args = (
                peer_output_tensors, combine_cute2,
                topk_idx_cute, scatter_cute, ntok_cute)
        else:
            expert_output_peer_cute = from_dlpack(
                _peer_ptr_tensor(output_buf, world_size, device))
            combine_args = (
                expert_output_peer_cute, combine_cute2,
                topk_idx_cute, scatter_cute,
                ntok_cute, max_recv_tokens, hidden,
            )
        if autotune:
            autotune_combine_tma(
                peer_output_tensors=peer_output_tensors,
                combine_cute=combine_cute2,
                combine_output_buf=combine_output_buf2,
                output_buf=output_buf,
                topk_idx_cute=topk_idx_cute,
                scatter_cute=scatter_cute,
                ntok_cute=ntok_cute,
                topk_indices_tensor=topk_indices_tensor,
                token_dst_scatter_idx=token_dst_scatter_idx,
                num_tokens=num_tokens, hidden=hidden, topk=topk,
                experts_per_rank=experts_per_rank,
                rank=rank, world_size=world_size, device=device,
                warmup_iterations=warmup_iterations, iterations=iterations)
        else:
            run_moe_benchmark(
                compiled_scatter, compiled_combine,
                dispatch_args, combine_args,
                num_tokens, hidden, topk, rank,
                warmup_iterations, iterations,
                bench_only=bench_only,
            )
        nvshmem.core.free_tensor(input_buf)

    nvshmem.core.free_tensor(output_buf)


def _bench_kernel(kernel_fn, args, warmup, iterations):
    """Simple CUDA event timing for a kernel call."""
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        kernel_fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations  # us


def run_moe_benchmark(compiled_dispatch, compiled_combine,
                      dispatch_args, combine_args,
                      num_tokens, hidden, topk,
                      rank, warmup_iterations=10, iterations=100,
                      bench_only="both"):
    """Benchmark dispatch and combine kernels separately.

    `bench_only` selects which kernel(s) to time: "dispatch", "combine",
    or "both" (default).
    """

    if rank == 0:
        print(f"\n{'='*40}")
        print(f"MoE Benchmark (warmup={warmup_iterations}, iter={iterations})")
        print(f"{'='*40}")

    bytes_moved = num_tokens * topk * hidden * 4  # float32
    dispatch_time = None
    combine_time = None

    # Benchmark Dispatch
    if bench_only in ("both", "dispatch"):
        if rank == 0:
            print("\n--- Benchmark: Dispatch ---")
        dist.barrier()
        dispatch_time = _bench_kernel(
            compiled_dispatch, dispatch_args, warmup_iterations, iterations)
        if rank == 0:
            bw_gbps = bytes_moved / (dispatch_time * 1e-6) / 1e9
            print(f"  Dispatch time: {dispatch_time:.2f} us  "
                  f"BW: {bw_gbps:.2f} GB/s")

    # Benchmark Combine
    if bench_only in ("both", "combine"):
        if rank == 0:
            print("\n--- Benchmark: Combine ---")
        dist.barrier()
        combine_time = _bench_kernel(
            compiled_combine, combine_args, warmup_iterations, iterations)
        if rank == 0:
            bw_gbps = bytes_moved / (combine_time * 1e-6) / 1e9
            print(f"  Combine time:  {combine_time:.2f} us  "
                  f"BW: {bw_gbps:.2f} GB/s")

    if rank == 0:
        print(f"\n  Summary:")
        if dispatch_time is not None:
            print(f"    Dispatch: {dispatch_time:.2f} us")
        if combine_time is not None:
            print(f"    Combine:  {combine_time:.2f} us")
        if dispatch_time is not None and combine_time is not None:
            print(f"    Total:    {dispatch_time + combine_time:.2f} us")


def autotune_combine_tma(*, peer_output_tensors, combine_cute, combine_output_buf,
                         output_buf, topk_idx_cute, scatter_cute, ntok_cute,
                         topk_indices_tensor, token_dst_scatter_idx,
                         num_tokens, hidden, topk, experts_per_rank,
                         rank, world_size, device,
                         warmup_iterations, iterations,
                         num_sm=148, smem_cap_kb=227):
    """Lightweight in-process autotune for the TMA combine kernel.

    SM efficiency is the objective: saturate NVLink with as few CTAs (=SMs)
    as possible so the rest of the GPU can run expert GEMMs. Two phases:
      1. Fix a reference CTA count, sweep (hchunk, num_stages) tile configs.
      2. Take the best tile config, sweep CTA count down to find the knee
         where BW stops improving -> the SM-efficient operating point.

    Every config is validated against the combine reference before timing, so
    a fast-but-wrong config can never win.
    """
    bytes_moved = num_tokens * topk * hidden * 4  # float32, one topk-sum pass
    args = (peer_output_tensors, combine_cute, topk_idx_cute, scatter_cute, ntok_cute)

    def bench_one(cfg, cta):
        smem_b = cfg["hchunk"] * cfg["num_stages"] * 4  # + mbars/align (small)
        if smem_b > smem_cap_kb * 1024:
            return None  # won't fit in smem
        try:
            kernel = CombineTmaKernel(cutlass.Float32, hidden, topk, **cfg)
            compiled = cute.compile(
                kernel, *args, experts_per_rank, rank, world_size, cta)
        except Exception as exc:  # noqa: BLE001 - report and skip bad configs
            if rank == 0:
                print(f"    [skip] cfg={cfg} cta={cta}: compile failed ({exc})")
            return None
        # Correctness gate.
        combine_output_buf.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        compiled(*args)
        torch.cuda.synchronize()
        nstream = Device().create_stream()
        nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=nstream)
        nstream.sync()
        ok = check_combine(
            combine_output_buf, output_buf, topk_indices_tensor,
            token_dst_scatter_idx, num_tokens, topk, experts_per_rank,
            rank, world_size)
        ok_t = torch.tensor(int(ok), dtype=torch.int32, device=device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        ok = bool(ok_t.item())
        dist.barrier()
        us = _bench_kernel(compiled, args, warmup_iterations, iterations)
        bw = bytes_moved / (us * 1e-6) / 1e9
        return us, bw, ok

    # Tile configs: (hchunk, num_stages) under the smem budget. hchunk must
    # divide hidden; larger hchunk => fewer barrier round-trips per byte but
    # forces fewer stages (less topk-load overlap).
    tile_cfgs = [
        dict(hchunk=1792, num_stages=16),
        dict(hchunk=1792, num_stages=8),
        dict(hchunk=3584, num_stages=8),
        dict(hchunk=3584, num_stages=4),
        dict(hchunk=7168, num_stages=4),
        dict(hchunk=7168, num_stages=2),
    ]
    tile_cfgs = [c for c in tile_cfgs if hidden % c["hchunk"] == 0]
    cta_list = [c for c in [8, 12, 16, 20, 24, 32, 48, 64, num_sm] if c <= num_sm]
    REF_CTA = 32 if 32 <= num_sm else num_sm

    cache = {}  # (hchunk, stages, cta) -> (us, bw, ok)

    def key(cfg, cta):
        return (cfg["hchunk"], cfg["num_stages"], cta)

    if rank == 0:
        print(f"\n{'='*60}")
        print("TMA Combine autotune  (objective: GB/s per SM)")
        print(f"  tokens={num_tokens} hidden={hidden} topk={topk} "
              f"bytes={bytes_moved/1e6:.1f} MB  num_sm={num_sm}")
        print(f"{'='*60}")

    # ---- Phase 1: tile sweep at reference CTA count ----
    if rank == 0:
        print(f"\n-- Phase 1: (hchunk, stages) sweep @ {REF_CTA} CTAs --")
        print(f"  {'hchunk':>7} {'stages':>6} {'smem_KB':>7} "
              f"{'us':>8} {'GB/s':>8} {'ok':>4}")
    phase1 = []
    for cfg in tile_cfgs:
        res = bench_one(cfg, REF_CTA)
        if res is None:
            continue
        us, bw, ok = res
        cache[key(cfg, REF_CTA)] = res
        phase1.append((cfg, us, bw, ok))
        if rank == 0:
            print(f"  {cfg['hchunk']:>7} {cfg['num_stages']:>6} "
                  f"{cfg['hchunk']*cfg['num_stages']*4//1024:>7} "
                  f"{us:>8.1f} {bw:>8.1f} {'Y' if ok else 'N':>4}")

    passing = [p for p in phase1 if p[3]]
    if not passing:
        if rank == 0:
            print("  no passing tile config; aborting autotune")
        return
    best_cfg = max(passing, key=lambda p: p[2])[0]
    if rank == 0:
        print(f"  -> best tile: hchunk={best_cfg['hchunk']} "
              f"stages={best_cfg['num_stages']}")

    # ---- Phase 2: CTA sweep at best tile config ----
    if rank == 0:
        print(f"\n-- Phase 2: CTA sweep @ hchunk={best_cfg['hchunk']} "
              f"stages={best_cfg['num_stages']} --")
        print(f"  {'CTAs':>5} {'us':>8} {'GB/s':>8} {'GB/s/SM':>8} {'ok':>4}")
    phase2 = []
    for cta in cta_list:
        res = cache.get(key(best_cfg, cta)) or bench_one(best_cfg, cta)
        if res is None:
            continue
        us, bw, ok = res
        phase2.append((cta, us, bw, ok))
        if rank == 0:
            print(f"  {cta:>5} {us:>8.1f} {bw:>8.1f} {bw/cta:>8.2f} "
                  f"{'Y' if ok else 'N':>4}")

    # ---- SM-efficient pick: fewest CTAs within 5% of peak BW ----
    good = [p for p in phase2 if p[3]]
    if rank == 0 and good:
        peak_bw = max(p[2] for p in good)
        knee = min((p for p in good if p[2] >= 0.95 * peak_bw),
                   key=lambda p: p[0])
        print(f"\n  peak BW: {peak_bw:.1f} GB/s")
        print(f"  SM-efficient pick (>=95% peak at fewest CTAs): "
              f"hchunk={best_cfg['hchunk']} stages={best_cfg['num_stages']} "
              f"cta={knee[0]} -> {knee[2]:.1f} GB/s "
              f"({knee[2]/knee[0]:.2f} GB/s/SM)")


def run_tma_combine(num_tokens, hidden, num_experts, topk,
                    ctas_list, n_tiles_target, tma_cfg,
                    warmup_iterations, iterations):
    """Standalone test + benchmark for the gated CombineTmaKernel: expert outputs
    are synthetic (per-rank seeded randn standing in for GEMM output), the
    producer is emulated by a tile-trickle kernel that fills rows tile by
    tile and publishes the counters data-before-flag — so a correct combine
    output proves the gate actually waits. Benchmark compares gated (flags
    pre-satisfied) against the ungated CombineTmaKernel."""
    from quack.tilepipe import build_recv_metadata

    TILE_M = 128
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    epr = num_experts // world_size
    torch.manual_seed(42 + rank)
    if rank == 0:
        print(f"\nGated TMA combine test: tokens/rank={num_tokens} hidden={hidden} "
              f"experts={num_experts} topk={topk} n_tiles_target={n_tiles_target} "
              f"world={world_size}", flush=True)

    topk_indices = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    all_topk_t = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    dist.all_gather(all_topk_t, topk_indices.contiguous())
    all_topk = np.stack([t.cpu().numpy() for t in all_topk_t])

    # Deterministic recv-slot assignment (same rule as tilepipe dispatch):
    # rows on rank d grouped by local expert, then source rank, then token
    # order. scatter[t, j] = row of (t, j) in rank_j's expert-output buffer.
    counts = np.zeros((world_size, num_experts), dtype=np.int64)
    for r in range(world_size):
        counts[r] = np.bincount(all_topk[r].reshape(-1), minlength=num_experts)
    base = np.zeros((world_size, epr, world_size), dtype=np.int64)
    for d in range(world_size):
        c = 0
        for e in range(epr):
            for src in range(world_size):
                base[d, e, src] = c
                c += counts[src, d * epr + e]
    topk_flat = all_topk[rank].reshape(-1)
    dst = topk_flat // epr
    el = topk_flat % epr
    group = dst * epr + el
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    within = np.arange(len(order)) - np.searchsorted(sorted_group, sorted_group, "left")
    slot = np.empty(len(order), dtype=np.int64)
    slot[order] = base[dst[order], el[order], rank] + within
    scatter_np = slot.reshape(num_tokens, topk)

    # Per-rank tile geometry: rank d has split sizes per local expert;
    # m-tiles per expert = ceil(len/128); rank_tile_base = cumsum of ranks'
    # totals. flag_idx[t, j] fully precomputed (rank offset included).
    per_rank_meta = [build_recv_metadata(all_topk, num_experts, d, world_size)
                     for d in range(world_size)]
    rank_rows = [int(m[1][-1]) for m in per_rank_meta]
    tile_offsets = []
    rank_tiles = []
    for d in range(world_size):
        split = per_rank_meta[d][0]
        mt = (split + TILE_M - 1) // TILE_M
        tile_offsets.append(np.concatenate([[0], np.cumsum(mt)])[:-1])
        rank_tiles.append(int(mt.sum()))
    rank_tile_base = np.concatenate([[0], np.cumsum(rank_tiles)])[:-1]
    total_tiles = int(sum(rank_tiles))
    cu_by_rank = [m[1] for m in per_rank_meta]
    exp_of_slot = [np.searchsorted(cu_by_rank[d], np.arange(rank_rows[d]), "right") - 1
                   for d in range(world_size)]
    flag_idx_np = np.empty((num_tokens, topk), dtype=np.int64)
    for t in range(num_tokens):
        for j in range(topk):
            d, s = int(dst[t * topk + j]), int(scatter_np[t, j])
            b = int(exp_of_slot[d][s])
            tile = int(tile_offsets[d][b] + (s - cu_by_rank[d][b]) // TILE_M)
            flag_idx_np[t, j] = rank_tile_base[d] + tile
    max_rows = max(max(rank_rows), 1)

    # Synthetic expert outputs: rank d's buffer = seeded randn(d), so every
    # rank can rebuild any peer's rows for the reference.
    def rank_data(d):
        g = torch.Generator(device=device)
        g.manual_seed(1000 + d)
        return torch.randn((max_rows, hidden), generator=g, device=device,
                           dtype=torch.float32)

    my_data = rank_data(rank)
    expert_output_buf = nvshmem.core.tensor((max_rows, hidden), dtype=torch.float32)
    expert_output_buf.copy_(my_data)
    tile_flags = nvshmem.core.tensor((total_tiles,), dtype=torch.int32)
    tile_flags.fill_(0)
    flag_peer_ptrs = _peer_ptr_tensor(tile_flags, world_size, device)

    combine_out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=device)
    scatter_t = torch.from_numpy(scatter_np.astype(np.int32)).to(device)
    flag_idx_t = torch.from_numpy(flag_idx_np.astype(np.int32)).to(device)
    ntok_t = torch.full((world_size,), num_tokens, dtype=torch.int32, device=device)

    # Reference: sum over topk of the owning rank's row.
    ref = torch.zeros_like(combine_out)
    for d in range(world_size):
        data_d = my_data if d == rank else rank_data(d)
        mask = torch.from_numpy((dst.reshape(num_tokens, topk) == d)).to(device)
        idx = scatter_t.long().clamp(0, max_rows - 1)
        ref += (data_d[idx] * mask.unsqueeze(-1)).sum(dim=1)
        del data_d

    peer_tensors = [from_dlpack(nvshmem.core.get_peer_tensor(expert_output_buf, r))
                    for r in range(world_size)]
    combine_kernel = CombineTmaKernel(cutlass.Float32, hidden, topk, **tma_cfg)
    base_args = lambda: (
        peer_tensors, from_dlpack(combine_out), from_dlpack(topk_indices),
        from_dlpack(scatter_t), from_dlpack(ntok_t))
    gate_args = lambda: (
        from_dlpack(tile_flags), from_dlpack(flag_idx_t), Int32(n_tiles_target))
    compiled_gated_ctas = {c: cute.compile(combine_kernel, *base_args(),
                                           epr, rank, world_size, c, *gate_args())
                           for c in ctas_list}
    compiled_ungated = {c: cute.compile(combine_kernel, *base_args(),
                                        epr, rank, world_size, c)
                        for c in ctas_list}
    compiled_gated = compiled_gated_ctas[ctas_list[0]]
    print(f"[rank {rank}] combine kernels compiled", flush=True)

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    def check(tag):
        ok = torch.allclose(combine_out, ref, atol=1e-4, rtol=1e-4)
        bad = (~torch.isclose(combine_out, ref, atol=1e-4, rtol=1e-4)).sum().item()
        print(f"[rank {rank}] {tag}: {'OK' if ok else f'FAIL ({bad} elems)'}",
              flush=True)
        ok_t = torch.tensor([ok], dtype=torch.int32, device=device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        return bool(ok_t.item())

    # (A) flags pre-satisfied: warm-up execution + baseline correctness.
    tile_flags.fill_(n_tiles_target)
    combine_out.zero_()
    barrier()
    compiled_gated(*base_args(), *gate_args())
    barrier()
    if not check("gated combine (flags preset)"):
        raise SystemExit("gated combine preset-flags correctness FAILED")

    # Trickle warm-up (module load) standalone: buffer garbage -> filled.
    trickle_src = my_data
    expert_output_buf.fill_(float("nan"))
    tile_flags.fill_(0)
    barrier()
    tstream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled_trickle = cute.compile(
        _tile_trickle_launch, from_dlpack(trickle_src),
        from_dlpack(expert_output_buf), from_dlpack(flag_peer_ptrs),
        Int32(int(rank_tile_base[rank])), Int32(rank_rows[rank]),
        Int32(n_tiles_target), 64, TILE_M, world_size, tstream)
    compiled_trickle(
        from_dlpack(trickle_src), from_dlpack(expert_output_buf),
        from_dlpack(flag_peer_ptrs), Int32(int(rank_tile_base[rank])),
        Int32(rank_rows[rank]), Int32(n_tiles_target), tstream)
    barrier()
    assert torch.equal(expert_output_buf[:rank_rows[rank]],
                       my_data[:rank_rows[rank]]), "trickle fill mismatch"

    # (B) the real gating test: garbage buffer, zero flags; combine starts
    # first and must WAIT for each tile's data-before-flag publish.
    expert_output_buf.fill_(float("nan"))
    tile_flags.fill_(0)
    combine_out.zero_()
    barrier()
    combine_stream = torch.cuda.Stream()
    trickle_stream = torch.cuda.Stream()
    with torch.cuda.stream(combine_stream):
        compiled_gated(*base_args(), *gate_args())
    compiled_trickle(
        from_dlpack(trickle_src), from_dlpack(expert_output_buf),
        from_dlpack(flag_peer_ptrs), Int32(int(rank_tile_base[rank])),
        Int32(rank_rows[rank]), Int32(n_tiles_target),
        cuda_driver.CUstream(trickle_stream.cuda_stream))
    barrier()
    if not check("gated combine (trickled producer)"):
        raise SystemExit("gated combine trickle correctness FAILED")

    # Benchmark: gated (flags preset) vs ungated, CTA sweep.
    pull_bytes = num_tokens * topk * hidden * 4
    tile_flags.fill_(n_tiles_target)
    barrier()
    if rank == 0:
        print(f"\ncombine benchmark ({pull_bytes / 1e6:.0f} MB pulled/rank)")
        print(f"{'ctas':>6} {'ungated':>10} {'gated':>10} {'overhead':>9}")
    for c in ctas_list:
        res = {}
        for name, fn in (("ungated", lambda: compiled_ungated[c](
                              *base_args())),
                         ("gated", lambda: compiled_gated_ctas[c](
                              *base_args(), *gate_args()))):
            times = []
            for it in range(warmup_iterations + iterations):
                barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                if it >= warmup_iterations:
                    times.append(start.elapsed_time(end))
            t = torch.tensor([float(np.median(times))], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            res[name] = t.item()
        if rank == 0:
            print(f"{c:>6} {res['ungated']:>8.3f}ms {res['gated']:>8.3f}ms "
                  f"{(res['gated'] / res['ungated'] - 1) * 100:>8.1f}%", flush=True)

    nvshmem.core.free_tensor(expert_output_buf)
    nvshmem.core.free_tensor(tile_flags)


def run_tma_dispatch(num_tokens, hidden, num_experts, topk,
                     ctas_list, num_stages, workers,
                     warmup_iterations, iterations):
    """Standalone correctness test + CTA-sweep benchmark for
    DispatchTmaKernel (TilePipe-style host-precomputed send list, counting-
    semaphore publish). Reports GB/s and GB/s-per-SM so the SM-efficiency
    claim vs the SIMT dispatch is directly checkable."""
    from quack.tilepipe import build_send_arrays, build_recv_metadata

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    epr = num_experts // world_size
    torch.manual_seed(42 + rank)
    if rank == 0:
        print(f"\nTMA dispatch test: tokens/rank={num_tokens} hidden={hidden} "
              f"experts={num_experts} topk={topk} stages={num_stages} "
              f"workers={workers} world={world_size}", flush=True)

    topk_indices = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    input_data = (torch.randn((num_tokens, hidden), dtype=torch.bfloat16,
                              device=device) / hidden ** 0.5)
    all_topk_t = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    dist.all_gather(all_topk_t, topk_indices.contiguous())
    all_topk = np.stack([t.cpu().numpy() for t in all_topk_t])

    tok, slot, dst_, el, seg, seg_sizes_np = build_send_arrays(
        all_topk, num_experts, rank, world_size)
    split_sizes, cu_seqlens, max_recv = build_recv_metadata(
        all_topk, num_experts, rank, world_size)
    total_sends = len(tok)
    total_m = int(cu_seqlens[-1])

    input_buf = nvshmem.core.tensor((num_tokens, hidden), dtype=torch.bfloat16)
    input_buf.copy_(input_data)
    recv_buf = nvshmem.core.tensor((max_recv, hidden), dtype=torch.bfloat16)
    recv_buf.fill_(0)
    flags = nvshmem.core.tensor((epr,), dtype=torch.int32)
    flags.fill_(0)
    recv_peer_ptrs = _peer_ptr_tensor(recv_buf, world_size, device)
    flag_peer_ptrs = _peer_ptr_tensor(flags, world_size, device)

    dev = lambda a: torch.from_numpy(a).to(device)
    send_token, send_slot, send_dst, send_seg = (
        dev(tok), dev(slot), dev(dst_), dev(seg))
    seg_sizes_t = dev(seg_sizes_np)
    seg_done = torch.zeros(len(seg_sizes_np), dtype=torch.int32, device=device)
    split_sizes_t = dev(split_sizes.astype(np.int32))

    kern = DispatchTmaKernel(cutlass.BFloat16, hidden,
                             num_stages=num_stages, workers=workers)
    dyn1d = lambda t: from_dlpack(t).mark_layout_dynamic(leading_dim=0)
    kargs = lambda: (
        from_dlpack(input_buf, assumed_align=32).mark_layout_dynamic(leading_dim=1),
        from_dlpack(recv_peer_ptrs), from_dlpack(flag_peer_ptrs),
        dyn1d(send_token), dyn1d(send_slot), dyn1d(send_dst), dyn1d(send_seg),
        from_dlpack(seg_done), from_dlpack(seg_sizes_t),
        Int32(total_sends), Int32(max_recv))
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(kern, *kargs(), Int32(ctas_list[0]),
                            rank, world_size, stream)
    print(f"[rank {rank}] TMA dispatch kernel compiled "
          f"(smem={kern._SharedStorage.size_in_bytes() / 1024:.0f} KB)", flush=True)

    def reset():
        flags.fill_(0)
        seg_done.zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    def launch(num_ctas):
        compiled(*kargs(), Int32(num_ctas), stream)

    # Correctness (also the warm-up execution): flags land on split_sizes,
    # recv rows bit-exact against the deterministic slot assignment.
    reset()
    launch(ctas_list[0])
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])
    all_inputs = [torch.zeros_like(input_data) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_data.contiguous())
    recv_ref = torch.zeros((total_m, hidden), dtype=torch.bfloat16, device=device)
    for src in range(world_size):
        tok_s, slot_s, dst_s, _, _, _ = build_send_arrays(
            all_topk, num_experts, src, world_size)
        mine = dst_s == rank
        recv_ref[dev(slot_s[mine]).long()] = all_inputs[src][dev(tok_s[mine]).long()]
    ok_flags = torch.equal(flags.cpu(), split_sizes_t.cpu())
    ok_recv = torch.equal(recv_buf[:total_m], recv_ref)
    print(f"[rank {rank}] TMA dispatch correctness: "
          f"flags={'OK' if ok_flags else 'FAIL ' + str(flags.cpu().tolist()[:8])} "
          f"recv={'OK' if ok_recv else 'FAIL'}", flush=True)
    ok_t = torch.tensor([ok_flags and ok_recv], dtype=torch.int32, device=device)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if not bool(ok_t.item()):
        raise SystemExit(f"[rank {rank}] TMA dispatch correctness FAILED")

    # Benchmark: CTA sweep, median over iterations, slowest rank counts.
    send_bytes = total_sends * hidden * 2
    if rank == 0:
        print(f"\nTMA dispatch benchmark ({send_bytes / 1e6:.0f} MB pushed/rank)")
        print(f"{'ctas':>6} {'time':>10} {'GB/s':>8} {'GB/s/SM':>9}")
    for ctas in ctas_list:
        times = []
        for it in range(warmup_iterations + iterations):
            reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(ctas)
            end.record()
            torch.cuda.synchronize()
            if it >= warmup_iterations:
                times.append(start.elapsed_time(end))
        t = torch.tensor([float(np.median(times))], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        t_ms = t.item()
        if rank == 0:
            gbps = send_bytes / t_ms / 1e6
            print(f"{ctas:>6} {t_ms:>8.3f}ms {gbps:>8.0f} {gbps / ctas:>9.1f}",
                  flush=True)

    nvshmem.core.free_tensor(input_buf)
    nvshmem.core.free_tensor(recv_buf)
    nvshmem.core.free_tensor(flags)


def run(num_tokens, hidden, num_experts, topk, benchmark=False,
        warmup_iterations=10, iterations=100, save_baseline_to=None,
        bench_only="both", combine_cta_nums=148, combine_num_warps=32,
        combine_impl="reg", combine_tma_cfg=None, autotune=False):
    # Autotune is TMA-specific and needs the benchmark harness (dispatch-once
    # fill + timing loop), so it implies both.
    if autotune:
        benchmark = True
        combine_impl = "tma"
        bench_only = "combine"
    torchrun_uid_init_bcast()
    try:
        run_moe_dispatch_combine(num_tokens, hidden, num_experts, topk,
                                 benchmark=benchmark,
                                 warmup_iterations=warmup_iterations,
                                 iterations=iterations,
                                 save_baseline_to=save_baseline_to,
                                 bench_only=bench_only,
                                 combine_cta_nums=combine_cta_nums,
                                 combine_num_warps=combine_num_warps,
                                 combine_impl=combine_impl,
                                 combine_tma_cfg=combine_tma_cfg,
                                 autotune=autotune)
    finally:
        _finalize_kernels()
        torchrun_finalize()


def main():
    parser = argparse.ArgumentParser(
        description="MoE Dispatch + Combine kernels (Triton-distributed -> CuTeDSL)"
    )
    parser.add_argument("--num_tokens", default=2560, type=int)
    parser.add_argument("--hidden", default=7168, type=int)
    parser.add_argument("--num_experts", default=256, type=int,
                        help="Total experts, must be divisible by world_size")
    parser.add_argument("--topk", default=8, type=int)
    # -- Benchmark --
    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks after correctness tests")
    bench.add_argument("--bench-only", default="both",
                       choices=["both", "dispatch", "combine"],
                       help="Benchmark only the dispatch or combine kernel "
                            "(default: both)")
    bench.add_argument("--warmup_iterations", default=10, type=int)
    bench.add_argument("--iterations", default=100, type=int)

    # -- Combine kernel selection + manual config --
    comb = parser.add_argument_group("combine")
    comb.add_argument("--combine-impl", default="reg",
                      choices=["reg", "tma"],
                      help="Combine implementation: 'reg' (register LD.256) or "
                           "'tma' (SM-efficient TMA-bulk peer->smem).")
    comb.add_argument("--combine-cta-nums", default=148, type=int,
                      help="Combine grid size (CTAs).")
    comb.add_argument("--combine-num-warps", default=32, type=int,
                      help="Warps per combine CTA (reg impl only; "
                           "block = num_warps * 32).")
    comb.add_argument("--tma-hchunk", default=3584, type=int,
                      help="[tma] Hidden elems per bulk tile (must divide hidden).")
    comb.add_argument("--tma-stages", default=8, type=int,
                      help="[tma] Smem pipeline depth (>= topk for full overlap).")
    comb.add_argument("--tma-producer-threads", default=32, type=int,
                      help="[tma] Producer threads per CTA (only 1 issues the copy).")
    comb.add_argument("--tma-consumer-threads", default=128, type=int,
                      help="[tma] Consumer threads per CTA (must divide hchunk).")

    # -- Autotune (self-contained mode) --
    auto = parser.add_argument_group("autotune")
    auto.add_argument("--autotune-combine", action="store_true",
                      help="Sweep the TMA combine for SM efficiency and print a "
                           "GB/s-per-SM table. Implies --benchmark --combine-impl "
                           "tma --bench-only combine, and sweeps CTA count + "
                           "(hchunk, stages) itself -- so --combine-cta-nums, "
                           "--tma-hchunk and --tma-stages are ignored in this mode.")

    # -- TilePipe TMA dispatch (self-contained test + benchmark mode) --
    td = parser.add_argument_group("tma-dispatch")
    td.add_argument("--test-tma-dispatch", action="store_true",
                    help="Run only the TilePipe DispatchTmaKernel correctness "
                         "test + CTA-sweep benchmark (GB/s per SM), then exit. "
                         "Uses --num_tokens/--hidden/--num_experts/--topk and "
                         "the iteration counts.")
    td.add_argument("--dispatch-ctas", default="1,2,4,8,16,32",
                    help="[tma-dispatch] comma-separated CTA counts to sweep.")
    td.add_argument("--dispatch-tma-stages", default=12, type=int,
                    help="[tma-dispatch] SMEM stage budget (14 KB each at "
                         "hidden=7168).")
    td.add_argument("--dispatch-tma-workers", default=4, type=int,
                    help="[tma-dispatch] producer/consumer warp pairs per CTA, "
                         "each with a private stage partition and sub-block.")

    # -- TilePipe gated TMA combine (self-contained test + benchmark mode) --
    tc = parser.add_argument_group("tma-combine")
    tc.add_argument("--test-tma-combine", action="store_true",
                    help="Run only the gated CombineTmaKernel correctness test "
                         "(trickled tile-flag producer) + gated-vs-ungated CTA "
                         "sweep, then exit.")
    tc.add_argument("--combine-ctas", default="16,32,64,148",
                    help="[tma-combine] comma-separated CTA counts to sweep.")
    tc.add_argument("--combine-n-tiles-target", default=32, type=int,
                    help="[tma-combine] per-m-tile readiness target "
                         "(= ceil(N / tile_N) of the producing GEMM).")

    parser.add_argument("--save-baseline-to", default=None,
                        help="If set, rank 0 saves output_buf + combine_output_buf "
                             "(plus indices and shape constants) as a .npz file at "
                             "this path. Only writes when both check_dispatch and "
                             "check_combine pass. A subsequent run can compare "
                             "against this file to verify bit-identity.")
    args = parser.parse_args()

    if args.test_tma_combine:
        torchrun_uid_init_bcast()
        try:
            run_tma_combine(
                args.num_tokens, args.hidden, args.num_experts, args.topk,
                ctas_list=[int(x) for x in args.combine_ctas.split(",")],
                n_tiles_target=args.combine_n_tiles_target,
                tma_cfg=dict(
                    hchunk=args.tma_hchunk,
                    num_stages=args.tma_stages,
                    tma_threads=args.tma_producer_threads,
                    consumer_threads=args.tma_consumer_threads,
                ),
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations)
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            torchrun_finalize()
        return

    if args.test_tma_dispatch:
        torchrun_uid_init_bcast()
        try:
            run_tma_dispatch(
                args.num_tokens, args.hidden, args.num_experts, args.topk,
                ctas_list=[int(x) for x in args.dispatch_ctas.split(",")],
                num_stages=args.dispatch_tma_stages,
                workers=args.dispatch_tma_workers,
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations)
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            torchrun_finalize()
        return

    run(args.num_tokens, args.hidden, args.num_experts, args.topk,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        save_baseline_to=args.save_baseline_to,
        bench_only=args.bench_only,
        combine_cta_nums=args.combine_cta_nums,
        combine_num_warps=args.combine_num_warps,
        combine_impl=args.combine_impl,
        combine_tma_cfg=dict(
            hchunk=args.tma_hchunk,
            num_stages=args.tma_stages,
            tma_threads=args.tma_producer_threads,
            consumer_threads=args.tma_consumer_threads,
        ),
        autotune=args.autotune_combine)


if __name__ == "__main__":
    main()