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

Rust kernel is in tree: sequential workspace, cached quasi-definite LDL, COSMO-style ADMM with sparse active-set polish, homogeneous DR, polyhedral NT-IPM on that same KKT, cone projections (zero, nonnegative, SOC, exponential, power, genpower, PSD), and finance builders (mean-variance, CVaR, MAD, CDaR, EVaR).

`EngineKind::Auto` on polyhedral problems uses ADMM when a numeric factor is still valid (R0) and NT-IPM when \(P\) or \(A\) must be numerically refactored (setup / R1). Every accepted `Solved` status is re-checked in original coordinates (\(r_p\), \(r_d\), \(r_K\), gap, complementarity). Finance slacks are reconstructed from \(x\) after R0/R1 data changes, not copied blindly.

```bash
cargo test
cargo test --test compare -- --nocapture
cargo test --release --test compare -- --nocapture
python3 scripts/scs_sequence.py
```

Sequence-level timings versus Clarabel 0.9, OSQP 0.6, and SCS 3.2 are in [docs/benchmarks.md](docs/benchmarks.md). Rolling CVaR is independently `Solved` at \(10^{-6}\) on every date in those tests; ConiX sequence time on that LP is Clarabel-class. OSQP remains faster on small bound QPs. EVaR (exponential cones) still uses ADMM projections rather than a production nonsymmetric IPM.

Remaining work toward the full objective: production Anderson, a stronger IPM for exponential/power/PSD, and a Python backtest API.
