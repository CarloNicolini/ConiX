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

Mathematics-first design. No solver kernel yet.
