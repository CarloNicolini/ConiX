#!/usr/bin/env python3
"""Standard CVXPY benchmarks: ConiX vs Clarabel (correctness + speed).

Runs LP / QP / SOCP families through CVXPY with both solvers and reports
objective agreement plus wall-clock (setup+solve of ``problem.solve``).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

_default_lib = ROOT / "target" / "release" / "libconix.so"
if _default_lib.exists() and "CONIX_LIB" not in os.environ:
    os.environ["CONIX_LIB"] = str(_default_lib)

import cvxpy as cp  # noqa: E402
from conix.cvxpy_interface import register  # noqa: E402

register()


def _time_solve(prob: cp.Problem, solver: str, reps: int, warmups: int) -> list[float]:
    times: list[float] = []
    for i in range(warmups + reps):
        # Force a fresh solve; clear values so DPP cache does not skip work.
        for v in prob.variables():
            v.value = None
        t0 = time.perf_counter()
        prob.solve(solver=solver, warm_start=False)
        dt = time.perf_counter() - t0
        if i >= warmups:
            times.append(dt)
    return times


def _check(name: str, x_cx, x_cl, obj_cx, obj_cl, atol: float = 5e-3) -> bool:
    ok = True
    if x_cx is None or x_cl is None:
        print(f"  FAIL {name}: missing solution cx={x_cx is not None} cl={x_cl is not None}")
        return False
    if not np.allclose(x_cx, x_cl, atol=atol, rtol=1e-3):
        print(f"  FAIL {name}: x mismatch max|Δ|={np.max(np.abs(x_cx - x_cl)):.3e}")
        ok = False
    if abs(obj_cx - obj_cl) > atol and abs(obj_cx - obj_cl) / max(1.0, abs(obj_cl)) > 1e-3:
        print(f"  FAIL {name}: obj cx={obj_cx:.6g} cl={obj_cl:.6g}")
        ok = False
    return ok


def make_qp(n: int, m_ineq: int, seed: int) -> cp.Problem:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    P = B.T @ B + np.eye(n)
    q = rng.normal(size=n)
    G = rng.normal(size=(m_ineq, n))
    h = rng.random(size=m_ineq) + 0.1
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [G @ x <= h])


def make_lp(n: int, m: int, seed: int) -> cp.Problem:
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    # Feasible RHS
    x0 = rng.random(size=n)
    b = A @ x0 + rng.random(size=m)
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= 0])


def make_socp(n: int, seed: int) -> cp.Problem:
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(c @ x + cp.norm(x, 2)), [cp.sum(x) == 1, x >= -1])


def make_portfolio(n: int, seed: int) -> cp.Problem:
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((max(2 * n, 40), n)) * 0.02
    mu = r.mean(axis=0)
    sigma = np.cov(r, rowvar=False) + 1e-6 * np.eye(n)
    w = cp.Variable(n)
    return cp.Problem(
        cp.Minimize(cp.quad_form(w, sigma) - mu @ w),
        [cp.sum(w) == 1, w >= 0],
    )


def run_case(name: str, factory, reps: int, warmups: int) -> dict:
    prob_cx = factory()
    prob_cl = factory()
    t_cx = _time_solve(prob_cx, "CONIX", reps, warmups)
    t_cl = _time_solve(prob_cl, cp.CLARABEL, reps, warmups)
    x_cx = np.array(prob_cx.variables()[0].value, dtype=float)
    x_cl = np.array(prob_cl.variables()[0].value, dtype=float)
    ok = _check(name, x_cx, x_cl, float(prob_cx.value), float(prob_cl.value))
    med_cx = statistics.median(t_cx)
    med_cl = statistics.median(t_cl)
    speedup = med_cl / med_cx if med_cx > 0 else float("nan")
    print(
        f"{name:18s}  status_cx={prob_cx.status:18s} status_cl={prob_cl.status:18s}  "
        f"conix={med_cx*1e3:8.2f} ms  clarabel={med_cl*1e3:8.2f} ms  "
        f"speedup={speedup:6.2f}x  ok={ok}"
    )
    return {
        "name": name,
        "ok": ok,
        "conix_ms": med_cx * 1e3,
        "clarabel_ms": med_cl * 1e3,
        "speedup": speedup,
        "obj_cx": float(prob_cx.value),
        "obj_cl": float(prob_cl.value),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cases = [
        ("lp_small", lambda: make_lp(20, 40, 0)),
        ("qp_small", lambda: make_qp(20, 40, 1)),
        ("socp_small", lambda: make_socp(15, 2)),
        ("portfolio_20", lambda: make_portfolio(20, 3)),
    ]
    if not args.smoke:
        cases.extend(
            [
                ("lp_med", lambda: make_lp(80, 160, 4)),
                ("qp_med", lambda: make_qp(60, 120, 5)),
                ("portfolio_50", lambda: make_portfolio(50, 6)),
            ]
        )

    print(
        f"CVXPY ConiX vs Clarabel  reps={args.reps} warmups={args.warmups}",
        flush=True,
    )
    results = [run_case(name, factory, args.reps, args.warmups) for name, factory in cases]
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} cases within tolerance", flush=True)
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
