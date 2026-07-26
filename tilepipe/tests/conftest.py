"""pytest configuration for the TilePipe tests.

These tests used to live under ``tests/`` and inherited ``tests/conftest.py``.
That conftest does NOT apply here (pytest only loads conftest files from the
rootdir down to the test file), so the reusable plugin is opted into
explicitly — the mechanism ``tests/conftest.py`` documents for downstream
projects.

What does NOT follow from ``tests/conftest.py``: the xdist worker/GPU
round-robin, the OOM-to-skip translation, and the crash-item handling. The
TilePipe tests run single-GPU without ``-n``, so this costs nothing today; if
you ever run them under xdist, set CUDA_VISIBLE_DEVICES yourself.
"""

pytest_plugins = ["quack.testing.pytest_plugin"]
