# ConiX

A sequential conic optimizer for repeated, structure-preserving convex programs.

The intended workload is a long chain of related problems — especially quantitative-finance backtests — where LP, QP, second-order, exponential, and power cones appear (CVaR, MAD, EVaR, CDaR, mean-variance, log/power utilities). The design goal is not a new marketing name for ADMM or IPM. It is **minimum end-to-end time over a sequence**, with independently checked correctness, while carrying solver state forward.

The mathematical proposal is in [docs/mathematics.md](docs/mathematics.md). Implementation and numerical experiments come after that argument, not before it.

## Thesis

One unmodified algorithm cannot simultaneously dominate broad cone coverage, high-accuracy certificates, huge sparse instances, and low-latency rolling solves. ConiX therefore uses a **finite-switch hybrid** behind one canonical quadratic cone form:

1. **Cached homogeneous Douglas–Rachford** as the sequential fast path (factor reuse when \(P,A\) are fixed; true primal/dual/embedding warm starts).
2. **Proximal ADMM** as a polyhedral/QP specialist (CVaR, MAD, CDaR, box-constrained Markowitz).
3. **Homogeneous IPM** as the certifying high-accuracy fallback (Andersen–Ye \((\tau,\kappa)\) Newton on a sparse cone-block KKT; Nesterov–Todd \(H_s\) for SOC and PSD; Clarabel \(H_s\) for exp/power/genpower; polyhedral NT as the diagonal special case). Symbolic AMD is reused on R1.
4. **Safeguarded Anderson / limited-memory Broyden** only on a fixed splitting map (disabled on tiny \(n+m\) maps where the extra saxpy is a net loss).
5. **Independent residual and ray verification** in original coordinates.

State that is persisted is typed: symbolic sparsity, numeric factorization, iterates, acceleration history, and finance auxiliaries are not interchangeable vectors.

## Numerical substrate

Sparse CSC storage and QDLDL (AMD + \(LDL^\top\)) come from [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs), the same substrate used by [COSMO.rs](https://github.com/CarloNicolini/COSMO.rs). The ConiX ADMM / DR / IPM algorithms are unchanged; only the linear-algebra backend was replaced with Clarabel's professionally tested primitives (with Clarabel-style static diagonal regularization before IPM refactors).

## Status

Rust kernel is in tree: sequential workspace, Clarabel QDLDL KKT factors with in-place numeric refactor, COSMO-style ADMM with sparse active-set polish, homogeneous DR with safeguarded Anderson, homogeneous sparse-KKT IPM (Andersen–Ye \((\tau,\kappa)\), cone-block \(H_s\), Nesterov–Todd on SOC/PSD), cone projections (zero, nonnegative, SOC, exponential, power, genpower, PSD), finance builders (mean-variance, CVaR, MAD, CDaR, EVaR), and a **maturin / PyO3** Python package (`conix`) for both the library API and the CVXPY solver.

`EngineKind::Auto` on polyhedral problems uses ADMM when a numeric factor is still valid (R0) and the homogeneous NT-IPM when \(P\) or \(A\) must be numerically refactored (setup / R1). Non-polyhedral Auto (EVaR, exp, power, SOC) runs the barrier IPM first and keeps ADMM only if the independent checker prefers it. Every accepted `Solved` status is re-checked in original coordinates (\(r_p\), \(r_d\), \(r_K\), gap, complementarity). Finance slacks are reconstructed from \(x\) after R0/R1 data changes, not copied blindly. IPM infeasibility statuses are independently checked Farkas rays.

```bash
# Rust
cargo test
cargo test --release --test compare -- --nocapture

# Python package (primary path — same layout as COSMO.rs)
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy scipy
maturin develop --release --features python
python -c "import conix; print(conix.ConixSolver)"
```

## Python library

```python
import numpy as np
from scipy import sparse
import conix

P = sparse.eye(2, format="csc")
q = np.array([-1.0, -1.0])
A = sparse.eye(2, format="csc")
b = np.array([1.0, 1.0])
sol = conix.solve(P, q, A, b, [("nonnegative", 2)])
print(sol.status, sol.x)

# Sequential finance builders (workspace reuse across dates)
ws = conix.cvar(returns, beta=0.9, l=[0, 0], u=[1, 1])
sol = ws.solve()
ws.update_cvar(returns2, 0.9, [0, 0], [1, 1])
sol2 = ws.solve()
```

## CVXPY interface

ConiX registers as a CVXPY conic solver (same Clarabel form \(Ax + s = b\), \(s \in \mathcal K\)), adapted from the COSMO.rs Python API:

```python
from conix.cvxpy_interface import register
register()

import cvxpy as cp
x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0, cp.sum(x) == 1])
prob.solve(solver="CONIX")
```

```bash
source .venv/bin/activate
pip install 'cvxpy>=1.5' clarabel pytest
pytest python/tests -q
python python/conix/bench_cvxpy.py --smoke
```

skfolio / [skfolio-accelerate](https://github.com/CarloNicolini/skfolio-accelerate) MeanRisk walk-forward (forces the CVXPY path so ConiX and Clarabel are compared head-to-head; with `--accelerate`, uses `backend="cvxpy-sequential"`):

```bash
pip install skfolio  # and optionally: pip install -e /path/to/skfolio-accelerate
python python/conix/bench_skfolio.py --quick
python python/conix/bench_skfolio.py --quick --accelerate
```

`register()` also refreshes skfolio's import-time `INSTALLED_SOLVERS` snapshot so `MeanRisk(solver="CONIX")` works even if skfolio was imported first.

Sequence-level timings versus Clarabel 0.11, OSQP 0.6, and SCS 3.2 are in [docs/benchmarks.md](docs/benchmarks.md). Rolling CVaR is independently `Solved` at \(10^{-6}\) on every date in those tests. Non-degenerate EVaR (exponential cones) is independently `Solved` at \(10^{-6}\) by the barrier IPM and matches Clarabel's objective. OSQP remains faster on small bound QPs.
