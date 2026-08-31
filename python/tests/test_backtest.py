from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

try:
    import pytest
except ImportError:  # allow `python3 python/tests/test_backtest.py`
    pytest = None


def _have_lib() -> bool:
    try:
        import conix as cx

        cx.lib()
        return True
    except FileNotFoundError:
        return False


if pytest is not None:
    pytestmark = pytest.mark.skipif(not _have_lib(), reason="libconix not built")


def test_cvar_sequence():
    import conix as cx

    r1 = [
        [0.01, 0.02],
        [-0.03, 0.01],
        [0.00, -0.02],
        [0.02, 0.00],
    ]
    r2 = [
        [0.00, 0.01],
        [-0.02, 0.01],
        [0.01, 0.00],
        [0.02, -0.01],
    ]
    l = [0.0, 0.0]
    u = [1.0, 1.0]
    with cx.cvar(r1, 0.8, l, u) as ws:
        s1 = ws.solve()
        assert s1.status == "Solved"
        assert abs(sum(s1.x[:2]) - 1.0) < 1e-4
        assert s1.residuals["pri"] <= 1e-6
        ws.update_cvar(r2, 0.8, l, u)
        s2 = ws.solve()
        assert s2.status == "Solved"
        assert abs(sum(s2.x[:2]) - 1.0) < 1e-4
        assert s2.residuals["pri"] <= 1e-6


def test_evar_checked():
    import conix as cx

    r = [
        [0.01, 0.00],
        [-0.02, 0.01],
        [0.00, 0.02],
        [0.01, -0.01],
        [-0.03, 0.02],
        [0.02, -0.02],
        [0.00, 0.01],
        [-0.01, 0.00],
        [0.015, -0.01],
        [-0.005, 0.02],
    ]
    p = [0.1] * 10
    l = [0.0, 0.0]
    u = [1.0, 1.0]
    with cx.evar(r, p, 0.8, l, u, engine=cx.IPM) as ws:
        sol = ws.solve()
        assert sol.status == "Solved"
        assert abs(sum(sol.x[:2]) - 1.0) < 1e-4
        assert sol.residuals["pri"] <= 1e-6
        assert sol.residuals["dual"] <= 1e-6


def test_mean_variance_sequence():
    import conix as cx
    import numpy as np

    rng = np.random.default_rng(0)
    r1 = rng.standard_normal((8, 3)) * 0.02
    r2 = rng.standard_normal((8, 3)) * 0.02
    l = [0.0, 0.0, 0.0]
    u = [1.0, 1.0, 1.0]
    sigma1 = np.cov(r1, rowvar=False)
    mu1 = r1.mean(axis=0)
    with cx.mean_variance(sigma1, mu1, l, u, 1.0) as ws:
        s1 = ws.solve()
        assert s1.status == "Solved"
        sigma2 = np.cov(r2, rowvar=False)
        mu2 = r2.mean(axis=0)
        ws.update_mean_variance(sigma2, mu2, l, u, 1.0)
        s2 = ws.solve()
        assert s2.status == "Solved"


if __name__ == "__main__":
    if not _have_lib():
        raise SystemExit("libconix not found; cargo build --release and set CONIX_LIB")
    test_cvar_sequence()
    test_evar_checked()
    print("ok")
