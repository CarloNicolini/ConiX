# ConiX sequence benchmarks

Measured by `cargo test --test compare -- --nocapture` and `python3 scripts/scs_sequence.py`.
Clarabel 0.9 is in-process (same QCP, persistent `update_q` / `update_A`, `presolve_enable = false`).
OSQP 0.6 is in-process on the bound form \(l \le Ax \le u\) of the same polyhedral QCP.
SCS 3.2 is the official Python wrapper: persistent `update(c=…)` on R0; a new workspace per date on R1 (the Python API cannot update \(A\)).

Independent original-coordinate residuals are the gate. Wall-clock is reported, not assumed.
OSQP/SCS numbers use each solver's own termination; ConiX `Solved` additionally requires
\(\hat r_p,\hat r_d,\hat r_{\mathcal K},\hat g,\hat r_{\mathrm{comp}}\le 10^{-6}\).

## Auto policy used in these runs

- Polyhedral **R0** with a cached factor: COSMO-style ADMM (no numeric refactor). Anderson is skipped on tiny \(n+m\) maps.
- Polyhedral **setup / R1**: homogeneous NT-IPM on a sparse diagonal-\(H_s\) KKT (AMD reused across Newton steps), then ADMM only if IPM does not certify.
- Exponential / power / SOC: homogeneous barrier IPM first (Andersen–Ye \((\tau,\kappa)\), sparse cone-block \(H_s\), Clarabel dual / primal-dual scaling, unit initialization, \(\sigma=(1-\alpha_{\mathrm{aff}})^3\)), ADMM only if the independent checker prefers it.

## R0 Markowitz (long-only QP, \(n=8\), \(P=I\) fixed, \(\mu\) changes, 20 dates)

| Profile | ConiX | Clarabel update | Clarabel cold | OSQP update | SCS update |
|---|---|---|---|---|---|
| debug | 0.0073 s | 0.0108 s | 0.0137 s | 0.0005 s | 0.0004 s |
| release | 0.0004 s | 0.0004 s | 0.0007 s | 0.0001 s | 0.0005 s |

ConiX uses **2 numeric factors** over 20 dates. Skipping Anderson on this tiny map recovers the
pre-Anderson 0.0004 s release figure and matches Clarabel-update; OSQP remains the QP specialist.
Objectives match Clarabel/OSQP to \(10^{-3}\) relative; budget residuals are independently below \(10^{-4}\).

## R1 CVaR (long-only LP, \(n=5\), \(T=12\), scenario matrix values change, 10 dates)

| Profile | ConiX (10/10 at \(10^{-6}\)) | Clarabel update | OSQP update | SCS cold |
|---|---|---|---|---|
| debug | 0.015 s | 0.019 s | 0.383 s (0/10 certified) | 0.055 s (5/10 `SOLVED`) |
| release | 0.0007 s | 0.0007 s | 0.073 s (0/10 certified) | 0.057 s (5/10 `SOLVED`) |

This is the hypothesis that used to fail: ADMM-only ConiX spent ~6 s in debug and missed
\(10^{-6}\) on some dates. Homogeneous NT-IPM on the cached cone-block pattern hits checked
\(10^{-6}\) on every date and matches Clarabel's release wall-clock. OSQP/SCS ADMM do not
reliably certify this LP at \(10^{-6}\).

## R1 CVaR backtest slice (\(n=15\), \(T=36\), \(\beta=0.9\), 12 dates)

Closer to a small rolling window than the unit-test instance. Every ConiX date is
independently `Solved` at \(10^{-6}\) and matches the Clarabel objective.

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.108 s | 0.121 s |
| release | 0.0041 s | 0.0047 s |

## R1 EVaR (exponential cones, \(n=4\), \(T=10\), \(p=1/T\), \(\beta=0.8\), 6 dates)

Non-degenerate: \(P(\mathrm{worst})=0.1 < 1-\beta=0.2\). Auto uses the homogeneous IPM.
Every date is independently `Solved` at \(10^{-6}\) and the objective matches Clarabel.
Sparse cone-block \(H_s\) (3×3 exp triangles, not a dense \(m\times m\) Hessian) is what
makes the sequence Clarabel-class.

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.018 s | 0.018 s |
| release | 0.0010 s | 0.0007 s |

Cold `min t` s.t. \((1,1,t)\in\mathrm{EXP}\) is `Solved` at \(10^{-6}\) (\(x=e\)).
ADMM/DR still solve that program, but collapse toward the \(t\to 0\) ray on EVaR; the IPM does not.
A primal-infeasible LP is certified by the same homogeneous path and independently checked
as a Farkas ray (`ipm_primal_infeasible_lp`).

## What this does and does not prove

- Proves: R0 factor reuse; uniform checked \(10^{-6}\) on rolling CVaR and rolling EVaR;
  homogeneous IPM with independently checked infeasibility certificates; sparse cone-block
  \(H_s\) so EVaR sequence time is Clarabel-class; polyhedral finance LPs match or beat
  Clarabel-update and beat OSQP/SCS at checked \(10^{-6}\); original-coordinate residuals
  are the status authority; the Python sequential API (`python/conix`) drives the same
  workspace (CVaR R1, EVaR IPM).
- Does not prove: dominance over OSQP on small bound QPs (OSQP still wins R0 Markowitz);
  a production genpower sparse-expansion / PSD Nesterov–Todd Hessian (those cones are
  projected, but CVaR/MAD/EVaR/CDaR do not require them).

Reproduce:

```bash
cargo test --test compare -- --nocapture
cargo test --release --test compare -- --nocapture
python3 scripts/scs_sequence.py   # requires `pip install scs numpy scipy`
CONIX_LIB=target/release/libconix.so python3 python/tests/test_backtest.py
```
