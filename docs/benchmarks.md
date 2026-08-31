# ConiX sequence benchmarks

Measurements from `cargo test --test compare -- --nocapture` (debug profile unless noted). Clarabel 0.9 is the in-process peer: same QCP, same cones, persistent `update_q` / `update_A` with `presolve_enable = false`. OSQP and SCS are not in this crate (C libraries). A fair C harness belongs in a later comparison, not as a substitute for the Clarabel numbers.

Independent correctness is the gate. Wall-clock is reported, not assumed.

## R0 Markowitz (debug)

- Problem: long-only mean-variance, \(n=8\) assets, \(P=I\) fixed, \(\mu\) changes each date (R0).
- Horizon: 20 dates.
- ConiX: one workspace, `update_q`, cached numeric LDL (factor count must not grow after setup / one \(\rho\) adapt).
- Clarabel-update: one `DefaultSolver`, `update_q` then `solve`.
- Clarabel-cold: new solver each date.

| Solver | Sequence time (s) | Notes |
|---|---|---|
| ConiX ADMM | ~0.007 | 2 numeric factors over 20 dates |
| Clarabel persistent update | ~0.011 | IPM refactor every date |
| Clarabel cold start | ~0.014 | setup + IPM every date |

On this R0 QP, ConiX is faster in debug because the KKT factor is reused. Objectives match Clarabel to \(10^{-3}\) relative; budget residuals are independently below \(10^{-4}\).

## R1 CVaR (debug)

- Problem: long-only CVaR, \(n=5\) assets, \(T=12\) scenarios, scenario matrix values change (R1, same sparsity).
- Horizon: 10 dates.
- Correctness: ConiX primal/dual/cone residuals vs Clarabel objective on every date.

| Solver | Sequence time (s) | Notes |
|---|---|---|
| ConiX ADMM | ~6.0 | 7/10 dates at checked \(10^{-6}\); others independently feasible at \(5\cdot10^{-4}\). R1 forces a numeric refactor. Cold retry on stale warm starts adds iterations. |
| Clarabel persistent update | ~0.019 | IPM, few dozen iterations per date |

On this R1 LP, Clarabel is much faster: ADMM's long tail is a known cost when the cached factor is already being rebuilt. Small CVaR/MAD/CDaR instances in `tests/solver.rs` do hit checked \(10^{-6}\). EVaR (exponential cones) and a power-cone feasibility problem are covered there as well.

## What this does and does not prove

- Proves: R0 factor reuse is real; ConiX solutions match Clarabel on the tested QPs/CVaRs; original-coordinate residuals are the status authority.
- Does not prove: production release-mode speed, OSQP/SCS dominance, or \(10^{-6}\) on every rolling CVaR date. Those remain open hypotheses (H1–H5 in `docs/mathematics.md`).
