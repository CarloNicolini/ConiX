#!/usr/bin/env python3
"""skfolio MeanRisk walk-forward: ConiX vs Clarabel (CVXPY solver).

Uses the same problem family as skfolio-accelerate's canonical benchmark
(boxed MeanRisk on synthetic factor returns), but forces the CVXPY solver so
ConiX and Clarabel are compared head-to-head. Compact OSQP/HiGHS engines are
intentionally bypassed by setting ``solver`` to CONIX / CLARABEL with an
option that keeps the estimator on the native / sequential CVXPY path.

Install::

    pip install skfolio
    # optional: pip install -e /path/to/skfolio-accelerate

Run (after ``cargo build --release``)::

    CONIX_LIB=target/release/libconix.so python python/conix/bench_skfolio.py --quick
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

from conix.cvxpy_interface import register  # noqa: E402

register()


def _factor_returns(n_obs: int, n_assets: int, n_factors: int, seed: int) -> np.ndarray:
    """Match skfolio-accelerate ``flagship.factor_returns`` construction."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, size=(n_obs, n_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_factors, n_assets))
    idio = rng.normal(0.0, 0.005, size=(n_obs, n_assets))
    return factors @ loadings + idio


def _path_sharpes(prediction) -> np.ndarray:
    try:
        from skfolio_accelerate.scoring import path_sharpes

        return np.asarray(path_sharpes(prediction), dtype=float)
    except Exception:
        # Fallback: per-portfolio Sharpe when accelerate is not installed.
        rets = []
        try:
            for ptf in prediction:
                r = np.asarray(ptf.returns, dtype=float)
                if r.size == 0:
                    continue
                mu = r.mean()
                sd = r.std(ddof=1)
                rets.append(mu / sd if sd > 1e-12 else 0.0)
        except TypeError:
            r = np.asarray(prediction.returns, dtype=float)
            mu = r.mean()
            sd = r.std(ddof=1)
            rets.append(mu / sd if sd > 1e-12 else 0.0)
        return np.asarray(rets, dtype=float)


def _weights_matrix(prediction) -> np.ndarray | None:
    try:
        rows = [np.asarray(ptf.weights, dtype=float) for ptf in prediction]
        return np.vstack(rows)
    except Exception:
        return None


def run_walk_forward(
    X: np.ndarray,
    *,
    solver: str,
    train_size: int,
    test_size: int,
    risk_measure: str,
    l2_coef: float,
    use_accelerate: bool,
) -> tuple[object, float]:
    from skfolio.model_selection import WalkForward, cross_val_predict as sk_cvp
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.measures import RiskMeasure

    # Ensure CONIX is visible even if skfolio was imported elsewhere first.
    register()

    rm = getattr(RiskMeasure, risk_measure)
    solver_params = None
    if solver == "CONIX":
        from conix import ADMM, AUTO

        # VARIANCE (QP): AUTO/IPM is fine. Scenario LPs (CVaR/MAD/…): force
        # ADMM — Auto's cold IPM can certify a suboptimal point on skfolio's
        # scaled CVXPY graph.
        engine = ADMM if risk_measure not in {"VARIANCE", "STANDARD_DEVIATION"} else AUTO
        solver_params = dict(
            engine=engine,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=25_000,
        )
    est = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=rm,
        l2_coef=l2_coef,
        solver=solver,
        solver_params=solver_params,
    )
    cv = WalkForward(train_size=train_size, test_size=test_size)

    if use_accelerate:
        from skfolio_accelerate import cross_val_predict as acc_cvp

        # Force sequential CVXPY so ConiX and Clarabel share the same path
        # (compact OSQP/HiGHS/Clarabel engines would otherwise dominate CLARABEL).
        t0 = time.perf_counter()
        pred = acc_cvp(est, X, cv=cv, backend="cvxpy-sequential")
        return pred, time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = sk_cvp(est, X, cv=cv, n_jobs=1)
    return pred, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-obs", type=int, default=252)
    p.add_argument("--n-assets", type=int, default=12)
    p.add_argument("--n-factors", type=int, default=8)
    p.add_argument("--train-size", type=int, default=126)
    p.add_argument("--test-size", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--l2-coef", type=float, default=1e-5)
    p.add_argument(
        "--risk",
        action="append",
        default=None,
        help="RiskMeasure name (repeatable). Default: VARIANCE CVAR",
    )
    p.add_argument("--accelerate", action="store_true", help="Use skfolio_accelerate.cross_val_predict")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_obs, args.n_assets = 120, 6
        args.train_size, args.test_size = 40, 20
        args.reps, args.warmups = 1, 1

    risks = args.risk or ["VARIANCE", "CVAR"]
    X = _factor_returns(args.n_obs, args.n_assets, args.n_factors, args.seed)
    mode = "skfolio-accelerate" if args.accelerate else "skfolio"
    print(
        f"skfolio MeanRisk ({mode})  X={X.shape} train={args.train_size} "
        f"test={args.test_size} reps={args.reps}",
        flush=True,
    )

    all_ok = True
    for risk in risks:
        times_cx: list[float] = []
        times_cl: list[float] = []
        pred_cx = pred_cl = None
        for i in range(args.warmups + args.reps):
            pred_cx, t_cx = run_walk_forward(
                X,
                solver="CONIX",
                train_size=args.train_size,
                test_size=args.test_size,
                risk_measure=risk,
                l2_coef=args.l2_coef,
                use_accelerate=args.accelerate,
            )
            pred_cl, t_cl = run_walk_forward(
                X,
                solver="CLARABEL",
                train_size=args.train_size,
                test_size=args.test_size,
                risk_measure=risk,
                l2_coef=args.l2_coef,
                use_accelerate=args.accelerate,
            )
            if i >= args.warmups:
                times_cx.append(t_cx)
                times_cl.append(t_cl)

        sharpe_cx = _path_sharpes(pred_cx)
        sharpe_cl = _path_sharpes(pred_cl)
        w_cx = _weights_matrix(pred_cx)
        w_cl = _weights_matrix(pred_cl)
        max_err = float("nan")
        weight_ok = True
        if w_cx is not None and w_cl is not None and w_cx.shape == w_cl.shape:
            max_err = float(np.max(np.abs(w_cx - w_cl)))
            tol = 1e-2 if risk == "VARIANCE" else 5e-2
            weight_ok = max_err < tol
            if not weight_ok:
                print(f"  WARN {risk}: max |Δw|={max_err:.3e} (tol={tol})")
                all_ok = False
        mean_s_cx = float(np.nanmean(sharpe_cx)) if sharpe_cx.size else float("nan")
        mean_s_cl = float(np.nanmean(sharpe_cl)) if sharpe_cl.size else float("nan")
        # Path-Sharpe agreement is the skfolio-accelerate ranking metric.
        sharpe_ok = True
        if np.isfinite(mean_s_cx) and np.isfinite(mean_s_cl):
            denom = max(1e-8, abs(mean_s_cl))
            sharpe_ok = abs(mean_s_cx - mean_s_cl) / denom < 0.25 or abs(
                mean_s_cx - mean_s_cl
            ) < 5e-2
            if not sharpe_ok:
                print(f"  WARN {risk}: mean Sharpe disagree")
                all_ok = False
        med_cx = statistics.median(times_cx)
        med_cl = statistics.median(times_cl)
        speedup = med_cl / med_cx if med_cx > 0 else float("nan")
        print(
            f"{risk:12s}  conix={med_cx:7.3f}s  clarabel={med_cl:7.3f}s  "
            f"speedup={speedup:6.2f}x  mean_sharpe_cx={mean_s_cx: .4f}  "
            f"mean_sharpe_cl={mean_s_cl: .4f}  Δsharpe={mean_s_cx - mean_s_cl: .4e}  "
            f"|Δw|_∞={max_err:.3e}  weights_ok={weight_ok} sharpe_ok={sharpe_ok}",
            flush=True,
        )

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
