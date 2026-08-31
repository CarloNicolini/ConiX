# ConiX sequence benchmarks

Measured by `cargo test --test compare -- --nocapture` and `python3 scripts/scs_sequence.py`.
Clarabel 0.9 is in-process (same QCP, persistent `update_q` / `update_A`, `presolve_enable = false`).
OSQP 0.6 is in-process on the bound form \(l \le Ax \le u\) of the same polyhedral QCP.
SCS 3.2 is the official Python wrapper: persistent `update(c=…)` on R0; a new workspace per date on R1 (the Python API cannot update \(A\)).

Independent original-coordinate residuals are the gate. Wall-clock is reported, not assumed.
OSQP/SCS numbers use each solver's own termination; ConiX `Solved` additionally requires
\(\hat r_p,\hat r_d,\hat r_{\mathcal K},\hat g,\hat r_{\mathrm{comp}}\le 10^{-6}\).

## Auto policy used in these runs

- Polyhedral **R0** with a cached factor: COSMO-style ADMM (no numeric refactor), Anderson memory 5 with a residual-decrease safeguard.
- Polyhedral **setup / R1**: Nesterov–Todd IPM on the same AMD-ordered KKT
  (\(H=\mathrm{diag}(s./z)\) is \(\rho_i=z_i/s_i\)), then ADMM only if IPM does not certify.
- Exponential / power / SOC: barrier IPM first (Clarabel dual Hessian / primal-dual \(H_s\),
  unit initialization, \(\sigma=(1-\alpha_{\mathrm{aff}})^3\)), ADMM only if the independent checker prefers it.

## R0 Markowitz (long-only QP, \(n=8\), \(P=I\) fixed, \(\mu\) changes, 20 dates)

| Profile | ConiX | Clarabel update | Clarabel cold | OSQP update | SCS update |
|---|---|---|---|---|---|
| debug | 0.007 s | 0.011 s | 0.014 s | 0.0005 s | 0.0004 s |
| release | 0.0008 s | 0.0004 s | 0.0006 s | 0.0001 s | 0.0004 s |

ConiX uses **2 numeric factors** over 20 dates. OSQP is the QP specialist and wins this R0
micro-QP; Anderson on the tiny ADMM map adds a little overhead versus the pre-Anderson 0.0004 s
figure. Objectives match Clarabel/OSQP to \(10^{-3}\) relative; budget residuals are independently below \(10^{-4}\).

## R1 CVaR (long-only LP, \(n=5\), \(T=12\), scenario matrix values change, 10 dates)

| Profile | ConiX (10/10 at \(10^{-6}\)) | Clarabel update | OSQP update | SCS cold |
|---|---|---|---|---|
| debug | 0.015 s | 0.019 s | 0.344 s (0/10 certified) | 0.055 s (5/10 `SOLVED`) |
| release | 0.0011 s | 0.0007 s | 0.067 s (0/10 certified) | 0.056 s (5/10 `SOLVED`) |

This is the hypothesis that used to fail: ADMM-only ConiX spent ~6 s in debug and missed
\(10^{-6}\) on some dates. NT-IPM on the cached KKT pattern hits checked \(10^{-6}\) on every date
and is **Clarabel-class in sequence time** (same order of magnitude; Clarabel still slightly
ahead in release). OSQP/SCS ADMM do not reliably certify this LP at \(10^{-6}\).

## R1 CVaR backtest slice (\(n=15\), \(T=36\), \(\beta=0.9\), 12 dates)

Closer to a small rolling window than the unit-test instance. Every ConiX date is
independently `Solved` at \(10^{-6}\) and matches the Clarabel objective.

| Profile | ConiX | Clarabel update |
|---|---|---|
| debug | 0.120 s | 0.119 s |
| release | 0.0065 s | 0.0046 s |

## R1 EVaR (exponential cones, \(n=4\), \(T=10\), \(p=1/T\), \(\beta=0.8\), 6 dates)

Non-degenerate: \(P(\mathrm{worst})=0.1 < 1-\beta=0.2\). Auto uses the barrier IPM.
Every date is independently `Solved` at \(10^{-6}\) and the objective matches Clarabel.

| Profile | ConiX | Clarabel update |
|---|---|---|
| release | 0.0073 s | 0.0015 s |

Cold `min t` s.t. \((1,1,t)\in\mathrm{EXP}\) is `Solved` at \(10^{-6}\) in 11 IPM steps (\(x=e\)).
ADMM/DR still solve that program, but collapse toward the \(t\to 0\) ray on EVaR; the IPM does not.
Sequence time is slower than Clarabel here because each Newton step rebuilds a dense cone Hessian;
correctness at the declared tolerance is the gate that ADMM could not pass.

## What this does and does not prove

- Proves: R0 factor reuse; uniform checked \(10^{-6}\) on rolling CVaR and rolling EVaR;
  ConiX sequence time on polyhedral finance LPs is in Clarabel's class and faster than OSQP/SCS
  at that tolerance; original-coordinate residuals are the status authority; the Python
  sequential API (`python/conix`) drives the same workspace (CVaR R1, EVaR IPM).
- Does not prove: dominance over OSQP on small bound QPs (OSQP wins R0 Markowitz);
  Clarabel-class wall-clock on exponential R1 (Clarabel is faster; ConiX certifies);
  a full Andersen–Ye \((\tau,\kappa)\) embedding on the IPM path (homogeneous DR remains
  the embedding engine; the IPM is a primal-dual barrier Newton);
  a production PSD/genpower IPM Hessian (those cones are not required by CVaR/MAD/EVaR/CDaR).

Reproduce:

```bash
cargo test --test compare -- --nocapture
cargo test --release --test compare -- --nocapture
python3 scripts/scs_sequence.py   # requires `pip install scs numpy scipy`
CONIX_LIB=target/release/libconix.so python3 python/tests/test_backtest.py
```
