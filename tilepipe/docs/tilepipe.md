# TilePipe: A flexible and modular framework for MoE tile grained communication compute overlap with interkernel pipelining and expert-level semaphore

## 1. Idea

Overlap MoE all-to-all dispatch with the grouped GEMM that consumes it, **without a megakernel**. Dispatch and GEMM are launched as two ordinary kernels on two streams over a disjoint SM partition, and synchronize through a per-expert counting semaphore in symmetric memory.

### The synchronization mechanism

One semaphore per local expert. Its value is **the number of tokens for that expert that have arrived**; its target is `split_sizes[e]`, the total number of tokens routed to that expert from all ranks.

```
flag[e] : one counter per LOCAL expert.   Invariant: 0 <= flag[e] <= split_sizes[e]

  dispatch : increments flag[e] as tokens for expert e land on the receiver.
             Sum of all increments across all senders == split_sizes[e].

  gemm     : before computing any tile of expert e, waits for flag[e] == split_sizes[e].
             That means every token of expert e has arrived, so the tile's rows are all valid.
```

Note that this abstraction works well for prologue communication for the dispatch+gemm case. For epilogue fusion like gemm+combine, we might need to do tile level flag or preaggregate per token, however this is less important because usually there is no operation to further fuse into combine. Also, gemm+combine is out of the scope.

### Requirements

- **Disjoint SMs.** Dispatch gets `num_comm_sms` blocks; GEMM gets the rest. If the spinning GEMM occupied every SM, dispatch could never run and the semaphore would never reach its target — deadlock.
- **Overlap.** With this approach, the dispatch communication should be hidden into the gemm, aside from the first wave of experts.

### Consequence: order alignment

Since GEMM blocks on expert `e` until `e` is *complete*, dispatch must drive experts to completion in the order GEMM consumes them. Otherwise GEMM stalls on expert `e` while the tokens in flight belong to `e+3`.

We call this sequence the **`expert_order`**: the order in which a kernel processes its tokens, grouped by expert. Both kernels already have such an order. A grouped GEMM iterates expert by expert so that an expert's weight matrix stays resident in L2 across all of that expert's tiles; dispatch groups its sends by destination expert because an expert's tokens occupy contiguous slots in the receiver's buffer. Neither loop structure is imposed by TilePipe — it is the structure these kernels already have, for reasons that have nothing to do with overlap.

TilePipe therefore does not add a loop. It only fixes the *permutation* that loop walks, and requires both kernels to walk the same one. That is the entire coupling between them, and it is why an existing grouped GEMM can join the pipeline without restructuring.

But there is a second, sharper constraint on which permutations are admissible, because **each rank runs its own GEMM against its own semaphore array.** Dispatch is not feeding one consumer; it is feeding `W` independent consumers concurrently. Getting this wrong starves entire GPUs.

## 2. EP-aware expert ordering


Take 4 experts over 2 ranks: GPU 0 owns global experts `{0, 1}`, GPU 1 owns `{2, 3}`.

If every sender walks its destinations in **global expert order** `0, 1, 2, 3`, then for the entire first half of dispatch every send is addressed to GPU 0. GPU 1's counters stay at zero. GPU 1's GEMM spins on an empty semaphore across the whole first half of the communication phase, and only starts once experts `2, 3` begin to arrive. Half the cluster's compute is idle for half of dispatch — the overlap the design exists to create is destroyed for that rank.

Interleave across destinations instead — `0, 2, 1, 3`:

```
global order   0 1 2 3      GPU0: ####----      GPU1: ----####      GPU1 starts late
                            GPU1 idle here ^^^^

interleaved    0 2 1 3      GPU0: ##--##--      GPU1: --##--##      both start early
```

Now expert `0` (on GPU 0) and expert `2` (on GPU 1) complete at roughly the same time, so both GEMMs unblock together and both stay fed.

The rule this expresses is that dispatch must touch **every destination rank** before advancing to its next expert. So the send order is a nested loop, expert-major and rank-minor:

```
for i in expert_order:      # permutation of LOCAL expert indices, agreed by all ranks
    for r in ranks:         # every destination, before advancing i
        send this rank's tokens for (dst_rank=r, local_expert=i)
```

`expert_order` is thus a permutation over *local* expert indices, not global ones — a flat order over global experts cannot express the inner loop, which is exactly why global order starves ranks. In the example, `0, 2, 1, 3` is what the nesting produces: local index `0` visits GPU 0 then GPU 1 (global `0`, `2`), then local index `1` does the same (global `1`, `3`). Equivalently, sort global experts by `e % experts_per_rank` with a stable tie-break on rank.

## Implement Expert Parallel Load Balancing
To illustrate the flexibility of tilepipe, we implement EPLB with it. We first build a fused dispatch-weight transfer kernel. The high level idea here is in the GEMM kernel,
we perform gemm for the tokens whose weight will be fetched the last, so that the token dispatches and weight transfer can overlap with the gemm of the tokens whose experts
are local as much as possible. Note that this implies the tokens whose experts on the remote rank does not require transfer should be send first in the dispatch kernel.
There can be two version of this tilepipe-eplb: elpb+dispatch+gemm and elpb+dispatch+gemm+gemmdown, where the fused kernel also transfer the second weight matrix.
Note we also 