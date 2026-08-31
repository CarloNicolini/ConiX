#!/usr/bin/env python3
"""Walk-forward benchmark: ConiX vs Clarabel vs OSQP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

# Default library path when built in-tree.
_default_lib = ROOT / "target" / "release" / "libconix.so"
if _default_lib.exists() and "CONIX_LIB" not in os.environ:
    os.environ["CONIX_LIB"] = str(_default_lib)

from conix.compare import (  # noqa: E402
    bench_cvar,
    bench_mad,
    bench_mean_variance,
    format_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description="ConiX vs Clarabel/OSQP walk-forward benchmark")
    p.add_argument("--t", type=int, default=10_000, help="scenario rows per rebalance")
    p.add_argument("--n", type=int, default=100, help="number of assets")
    p.add_argument("--dates", type=int, default=5, help="rebalance dates (warm/cold sequence length)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.95, help="CVaR confidence level")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0, help="mean-variance risk aversion")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="small problem for a quick sanity check (T=200, N=20, dates=3)",
    )
    args = p.parse_args()

    t, n, dates = args.t, args.n, args.dates
    if args.smoke:
        t, n, dates = 200, 20, 3

    print(f"ConiX walk-forward benchmark  T={t}  N={n}  dates={dates}  seed={args.seed}", flush=True)
    print("Returns: numpy.standard_normal only\n", flush=True)

    for label, runner in (
        ("mean-variance", lambda: bench_mean_variance(t, n, dates, args.seed, args.lam)),
        ("MAD", lambda: bench_mad(t, n, dates, args.seed)),
        ("CVaR", lambda: bench_cvar(t, n, dates, args.seed, args.beta)),
    ):
        print(format_report(label, t, n, dates, runner()), flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
