# ConiX sequence benchmarks

Measured by `cargo test --test compare -- --nocapture` and `python3 scripts/scs_sequence.py`.
Clarabel 0.9 is in-process (same QCP, persistent `update_q` / `update_A`, `presolve_enable = false`).
OSQP 0.6 is in-process on the bound form \(l \le Ax \le u\) of the same polyhedral QCP.
SCS 3.2 is the official Python wrapper: persistent `update(c=…)` on R0; a new workspace per date on R1 (the Python API cannot update \(A\)).

Independent original-coordinate residuals are the gate. Wall-clock is reported, not assumed.
OSQP/SCS numbers use each solver's own termination; ConiX `Solved` additionally requires
\(\hat r_p,\hat r_d,\hat r_{\mathcal K},\hat g,\hat r_{\mathrm{comp}}\le 10^{-6}\).

## Auto policy used in these runs

- Polyhedral **R0** with a cached factor: COSMO-style ADMM (no numeric refactor). Anderson is skipped on tiny \(n+m\) maps. Polish runs only if ADMM has not already certified `Solved`.
- Polyhedral **setup / R1**: homogeneous NT-IPM on a sparse diagonal-\(H_s\) KKT (AMD reused across Newton steps; numeric LDL workspaces reused in place). ADMM only if IPM does not certify.
- Exponential / power / SOC / PSD: homogeneous barrier IPM first (Andersen–Ye \((\tau,\kappa)\), sparse cone-block \(H_s\), Nesterov–Todd on SOC/PSD, Clarabel dual / primal-dual scaling on exp/power, unit initialization, \(\sigma=(1-\alpha_{\mathrm{aff}})^3\)), ADMM only if the independent checker prefers it.

IPM sequential solves reuse the AMD order, symbolic factor, and numeric \(L,D\) buffers.
They **unit-initialize** \((s,z)\) rather than copying the previous complementary solution.
Original-coordinate termination is cheap-gated on scaled residuals so exp-cone projections
are not run every Newton step.

## R0 Markowitz (long-only QP, \(n=8\), \(P=I\) fixed, \(\mu\) changes, 20 dates)

| Profile | ConiX | Clarabel update | Clarabel cold | OSQP update | SCS update |
|---|---|---|---|---|---|
| debug | 0.0058 s | 0.0107 s | 0.0134 s | — | — |
| release | 0.0002 s | 0.0004 s | 0.0006 s | 0.0001 s | 0.0004 s |

ConiX uses **2 numeric factors** over 20 dates and beats Clarabel-update. OSQP remains
a few tens of microseconds faster on this micro-QP. Objectives match Clarabel/OSQP to
\(10^{-3}\) relative; budget residuals are independently below \(10^{-4}\).

## R1 CVaR (long-only LP, \(n=5\), \(T=12\), scenario matrix values change, 10 dates)

| Profile | ConiX (10/10 at \(10^{-6}\)) | Clarabel update | OSQP update | SCS cold |
|---|---|---|---|---|
| debug | 0.015 s | 0.019 s | 0.383 s (0/10 certified) | 0.055 s (5/10 `SOLVED`) |
| release | 0.0006 s | 0.0007 s | 0.070 s (0/10 certified) | 0.055 s (5/10 `SOLVED`) |

Homogeneous NT-IPM on the cached cone-block pattern hits checked \(10^{-6}\) on every date
and matches or beats Clarabel's release wall-clock. OSQP/SCS ADMM do not reliably certify
this LP at \(10^{-6}\).

## R1 CVaR backtest slice (\(n=15\), \(T=36\), \(\beta=0.9\), 12 dates)

Closer to a small rolling window than the unit-test instance. Every ConiX date is
independently `Solved` at \(10^{-6}\) and matches the Clarabel objective.

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.100 s | 0.119 s |
| release | 0.0036 s | 0.0047 s |

## R1 CVaR wide (\(n=25\), \(T=48\), \(\beta=0.9\), 8 dates)

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.175 s | 0.139 s |
| release | 0.0058 s | 0.0058 s |

Release wall-clock matches Clarabel; every date is independently `Solved` at \(10^{-6}\).

## R1 MAD (\(n=8\), \(T=20\), 8 dates)

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.033 s | 0.031 s |
| release | 0.0012 s | 0.0011–0.0025 s |

## R1 CDaR (\(n=6\), \(T=16\), \(\beta=0.8\), 8 dates)

Affine-path peaks make a longer polyhedral chain than CVaR. Every date is independently
`Solved` at \(10^{-6}\) and the objective matches Clarabel.

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.034 s | 0.035 s |
| release | 0.0013 s | 0.0012–0.0014 s |

## R1 EVaR (exponential cones, \(n=4\), \(T=10\), \(p=1/T\), \(\beta=0.8\), 6 dates)

Non-degenerate: \(P(\mathrm{worst})=0.1 < 1-\beta=0.2\). Auto uses the homogeneous IPM.
Every date is independently `Solved` at \(10^{-6}\) and the objective matches Clarabel.
Sparse cone-block \(H_s\) (3×3 exp triangles) plus in-place LDL and gated original
residuals keep the sequence Clarabel-class.

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.017 s | 0.018 s |
| release | 0.0008 s | 0.0007 s |

Cold `min t` s.t. \((1,1,t)\in\mathrm{EXP}\) is `Solved` at \(10^{-6}\) (\(x=e\)).
ADMM/DR still solve that program, but collapse toward the \(t\to 0\) ray on EVaR; the IPM does not.
A primal-infeasible LP is certified by the same homogeneous path and independently checked
as a Farkas ray (`ipm_primal_infeasible_lp`).

## PSD Nesterov–Todd (no in-process Clarabel SDP; crate built without the `sdp` feature)

`skron(I)=I`, \(H_s z=s\) on random PD pairs, unit point \(\mathrm{svec}(I)\).
IPM solves a strictly feasible 2×2 SDP (`psd_ipm_interior`, \(t^\star=2\)) and a
Schur-complement SDP (`psd_ipm_schur`, \((x,t)=(1,1)\)) at checked residuals.
An R0 \(q\)-update (`sequential_psd_r0`) stays `Solved`.

## What this does and does not prove

- Proves: R0 factor reuse; uniform checked \(10^{-6}\) on rolling CVaR, MAD, CDaR, and EVaR;
  homogeneous IPM with independently checked infeasibility certificates; sparse cone-block
  \(H_s\) and in-place LDL so finance sequence time is Clarabel-class; CVaR backtest slices
  match or beat Clarabel release wall-clock and beat OSQP/SCS at checked \(10^{-6}\);
  original-coordinate residuals are the status authority; the Python sequential API
  (`python/conix`) drives the same workspace (CVaR R1, EVaR IPM); production PSD
  Nesterov–Todd \(H_s=\mathrm{skron}(G)\).
- Does not prove: dominance over OSQP on tiny bound QPs (OSQP can still be ~100 µs
  faster on \(n=8\) Markowitz). Generalized power uses Clarabel's dual Hessian as a
  dense triangle.

Reproduce:

```bash
cargo test --test compare -- --nocapture
cargo test --release --test compare -- --nocapture
python3 scripts/scs_sequence.py   # requires `pip install scs numpy scipy`
CONIX_LIB=target/release/libconix.so python3 python/tests/test_backtest.py
```
