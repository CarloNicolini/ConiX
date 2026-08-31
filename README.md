# ConiX

A sequential conic optimizer for repeated, structure-preserving convex programs.

The intended workload is a long chain of related problems — especially quantitative-finance backtests — where LP, QP, second-order, exponential, and power cones appear (CVaR, MAD, EVaR, CDaR, mean-variance, log/power utilities). The design goal is not a new marketing name for ADMM or IPM. It is **minimum end-to-end time over a sequence**, with independently checked correctness, while carrying solver state forward.

The mathematical proposal is in [docs/mathematics.md](docs/mathematics.md). Implementation and numerical experiments come after that argument, not before it.

## Thesis

One unmodified algorithm cannot simultaneously dominate broad cone coverage, high-accuracy certificates, huge sparse instances, and low-latency rolling solves. ConiX therefore uses a **finite-switch hybrid** behind one canonical quadratic cone form:

1. **Cached homogeneous Douglas–Rachford** as the sequential fast path (factor reuse when \(P,A\) are fixed; true primal/dual/embedding warm starts).
2. **Proximal ADMM** as a polyhedral/QP specialist (CVaR, MAD, CDaR, box-constrained Markowitz).
3. **Homogeneous primal-dual IPM** as the certifying high-accuracy fallback (Clarabel-class cones, recentered hot starts).
4. **Safeguarded Anderson / limited-memory Broyden** only on a fixed splitting map.
5. **Independent residual and ray verification** in original coordinates.

State that is persisted is typed: symbolic sparsity, numeric factorization, iterates, acceleration history, and finance auxiliaries are not interchangeable vectors.

## Status

Rust kernel is in tree: sequential workspace, cached quasi-definite LDL, COSMO-style ADMM with sparse active-set polish, homogeneous DR, IPM fallback, cone projections (zero, nonnegative, SOC, exponential, power, genpower, PSD), and finance builders (mean-variance, CVaR, MAD, CDaR, EVaR).

Every accepted `Solved` status is re-checked in original coordinates (`r_p`, `r_d`, `r_K`, gap, complementarity). R0 updates reuse the numeric KKT factor. Finance slacks are reconstructed from `x` after R0/R1 data changes, not copied blindly.

```bash
cargo test
cargo test --test compare -- --nocapture
```

Sequence-level timings vs Clarabel 0.9 (same QCP form, persistent `update_q` / `update_A`) are in [docs/benchmarks.md](docs/benchmarks.md). OSQP and SCS are not linked here; they are C libraries, not a same-language cone peer of Clarabel.rs.

Remaining work toward the full objective: uniform 1e-6 on every rolling CVaR date, production Anderson, a stronger IPM, OSQP/SCS C harnesses, and a Python backtest API.

