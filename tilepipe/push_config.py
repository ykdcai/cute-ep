"""Tuned configuration for the GEMM -> push-combine overlap.

One place that answers "what settings should the push kernel use here", so the
benchmark, the autotuner and any production caller agree. `pick()` is the
lookup; `parse_specs()` is the CLI form the autotuner sweeps with.

TUNABLES, and what each actually controls:

  workers  Producer/consumer warp pairs per CTA (block = workers * 64 threads).
           More workers means more parallel issue streams, but each one ends
           with its own `cp_async_bulk_wait_group(0)` drain, so the fixed tail
           grows with worker count. Measured: 1 worker cannot sustain the
           bandwidth (0.67x at 8192/8 CTAs) and 8 workers has the worst tail
           and the worst overlap at every size. 2 and 4 are the useful range.

  stages   SMEM pipeline depth == in-flight rows per CTA. This is the standalone
           bandwidth knob: 12 -> 24 stages is +12-18% on the comm kernel alone.
           It COSTS SMEM -- `stages * N * 2` bytes against ~228 KB per SM on
           B200 -- so the feasible maximum shrinks as N grows. 24 stages fits
           N=4096 (192 KB) but NOT N=7168 (336 KB); `pick()` clamps for you.

  wwin     In-flight remote writes per worker (the `cp_async_bulk_wait_group`
           watermark). Only binds when `stages // workers > wwin`; below that
           the producer blocks in `producer_acquire` first, which is why an
           early 8/16/32 sweep at stages//workers == 3 found nothing.

  ctas     Comm SMs; the GEMM gets `num_sms - ctas`. This matters more than the
           three above -- it trades comm bandwidth against the SM tax, and it
           is the only parameter whose optimum clearly moves with token count.

Deliberately NOT tunable here: `chunk`, the round-robin work unit. Swept at
128/64/32/16 rows and the tail did not move at any token count, so it stays
tied to tile_m.

The table is measured at N=4096, K=7168, topk=8, 4 GPUs, uniform routing.
Skewed routing and N=7168 are not yet covered -- see tilepipe/docs/status.md.
"""

from collections import namedtuple

PushConfig = namedtuple("PushConfig", "workers stages wwin ctas")

# B200 shared memory per SM available to a block (opt-in). Used only to clamp
# `stages`; pass an explicit budget to pick() to override.
DEFAULT_SMEM_BUDGET = 227 * 1024

# tokens/rank -> PushConfig. Lookup takes the entry for the largest key <=
# tokens, so intermediate and larger sizes fall back to the nearest measured
# point rather than to a default.
TUNED = {
    2048: PushConfig(workers=2, stages=24, wwin=8, ctas=12),
    4096: PushConfig(workers=2, stages=24, wwin=8, ctas=12),
    8192: PushConfig(workers=4, stages=24, wwin=8, ctas=8),
}


def max_stages(n, workers, dtype_bytes=2, budget=DEFAULT_SMEM_BUDGET):
    """Largest legal `stages` for a row length of `n`: SMEM is
    stages * n * dtype_bytes, and the kernel needs stages % workers == 0 and
    stages // workers >= 2."""
    cap = budget // (n * dtype_bytes)
    cap -= cap % workers                      # keep stages % workers == 0
    return max(cap, 2 * workers)              # the kernel's own floor


def pick(tokens, n, dtype_bytes=2, budget=DEFAULT_SMEM_BUDGET, table=TUNED):
    """Config for `tokens` per rank at row length `n`, with `stages` clamped to
    what SMEM allows. Clamping is silent by design: an N the table was not
    measured at should still run, just not necessarily at the optimum."""
    key = max((k for k in table if k <= tokens), default=min(table))
    cfg = table[key]
    return cfg._replace(stages=min(cfg.stages, max_stages(
        n, cfg.workers, dtype_bytes, budget)))


def parse_specs(spec, default_wwin=8):
    """'workers:stages[:wwin]' comma-separated -> [(workers, stages, wwin)].
    The autotuner's sweep form; `pick()` is what callers should use."""
    out = []
    for s in (spec or "").split(","):
        if not s.strip():
            continue
        w, st, win = (s.strip().split(":") + [str(default_wwin)])[:3]
        out.append((int(w), int(st), int(win)))
    return out


def validate(workers, stages, n, dtype_bytes=2, budget=DEFAULT_SMEM_BUDGET):
    """Raise with a useful message rather than failing at kernel launch."""
    assert stages % workers == 0, f"stages {stages} must divide by workers {workers}"
    assert stages // workers >= 2, f"each worker needs >=2 stages ({workers}:{stages})"
    need = stages * n * dtype_bytes
    assert need <= budget, (
        f"config {workers}:{stages} needs {need / 1024:.0f} KB of SMEM at N={n}, "
        f"budget is {budget / 1024:.0f} KB -- max stages here is "
        f"{max_stages(n, workers, dtype_bytes, budget)}")
