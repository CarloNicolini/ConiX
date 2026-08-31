# ConiX: a sequential hybrid for quadratic cone programs

This note is a mathematical design, not a claim of a finished solver and not a claim that a single new iteration dominates every convex program. The use case is a long sequence of *related* conic programs, as in a vectorized portfolio backtest, where correctness and amortized wall-clock time both matter.

COSMO is a Julia ADMM solver (`COSMO.jl`). There is no official `COSMO.rs`. Clarabel is the Oxford/Goulart–Chen homogeneous IPM with a first-class Rust implementation. OSQP is a QP-only ADMM solver whose sequential API is the practical gold standard for bound-constrained QPs. ConiX is built from what those methods share, and from what they refuse to share.

---

## 1. Canonical problem

Every model below is reduced to a convex quadratic cone program (QCP)

\[
\begin{aligned}
(\mathrm{P})\quad
\min_{x,s}\quad
&\tfrac12 x^\top P x + q^\top x \\
\mathrm{s.t.}\quad
&Ax + s = b,\qquad s\in\mathcal K,\qquad P\succeq 0,
\end{aligned}
\]

with dual

\[
\begin{aligned}
(\mathrm{D})\quad
\max_{x,z}\quad
&-\tfrac12 x^\top P x - b^\top z \\
\mathrm{s.t.}\quad
&Px + A^\top z + q = 0,\qquad z\in\mathcal K^\ast.
\end{aligned}
\]

Here \(\mathcal K\) is a closed convex cone, written as a product of primitive cones (zero, nonnegative, second-order, exponential, power, generalized power, PSD). Under constraint qualification, optimality is equivalent to

\[
Ax+s=b,\quad
Px+A^\top z+q=0,\quad
s\in\mathcal K,\ z\in\mathcal K^\ast,\quad
\langle s,z\rangle=0.
\]

This is the native form of Clarabel, COSMO (when \(\mathcal C\) is a cone), and SCS v3. OSQP is the special case \(\mathcal K = \mathbb R^n_+\) after splitting two-sided bounds. ECOS uses a linear objective and must epigraph a quadratic; ConiX will not.

A primal infeasibility certificate is \(z\neq 0\) with

\[
A^\top z = 0,\qquad z\in\mathcal K^\ast,\qquad b^\top z < 0.
\]

A dual infeasibility (primal unboundedness) certificate is \(d\neq 0\) with

\[
Pd = 0,\qquad -Ad\in\mathcal K,\qquad q^\top d < 0.
\]

Weak infeasibility, where two convex sets have positive distance zero, need not admit a strict Farkas certificate. No finite-precision method should invent one.

---

## 2. Sequential contract

A backtest is a path of data \(\theta_t\) in a *family* of QCPs with a slowly changing structure key. Write

\[
\min_x\ \tfrac12 x^\top P(\theta_t)x + q(\theta_t)^\top x
\quad\mathrm{s.t.}\quad
A(\theta_t)x + s = b(\theta_t),\quad s\in\mathcal K(\theta_t).
\]

The only quantity that matters for the stated goal is total sequence time

\[
T_N = T_{\mathrm{setup}} + \sum_{t=1}^N \bigl(T_{\mathrm{update}}(\theta_t) + T_{\mathrm{solve}}(\theta_t)\bigr),
\]

subject to independently checked residuals at a declared tolerance. Minimizing IPM iteration count on a cold instance is a different, strictly easier problem.

Updates fall into three mathematical classes.

| Class | What changes | What may be reused |
|---|---|---|
| **R0** | \(q,b\) only (and polyhedral bounds) | Numeric factorization of a *fixed* proximal/resolvent matrix; all iterates |
| **R1** | numeric \(P,A\) with identical sparsity and cone layout | Symbolic analysis, elimination tree, allocations, mapped iterates |
| **R2** | dimensions, sparsity, cone order, cone parameters | Only identifiers that can be remapped (assets, dated constraints) |

Finance maps onto this taxonomy almost exactly:

- rolling means, cash, turnover reference \(x^0\), CVaR/MAD RHS: **R0**;
- rolling covariance or scenario-matrix values at fixed window length: **R1**;
- assets entering/leaving, changing scenario count, changing power exponents: **R2**.

A previous optimizer \(x^\star(\theta_{t-1})\) is a good *predictor*. It is not a reusable numeric KKT factor when \(P\) or cone scaling changed. Conflating those two facts is the main reason sequential IPM APIs disappoint.

---

## 3. What the source methods actually share

### 3.1 The same saddle skeleton

COSMO/OSQP, SCS, and Clarabel all invert a matrix of the form

\[
\begin{bmatrix}
P + D_x & A^\top \\
A & -D_z
\end{bmatrix}.
\]

- COSMO/OSQP: \(D_x = \sigma I\), \(D_z = \rho^{-1}I\), **constant** while penalties are frozen.
- SCS: \(D_x = R_x\), \(D_z = R_y\), constant while the DR metric is frozen.
- Clarabel: \(D_z = H(s,z)\), the barrier/scaling Hessian, **different every Newton step**.

That is the entire sequential story in one line. Splitting methods can amortize a numeric \(LDL^\top\) over hundreds of backtest dates. Interior-point methods cannot, because \(H\) is the point of the algorithm.

Vanderbei’s theorem says a symmetric quasi-definite matrix admits \(LDL^\top\) for every symmetric permutation. It does not say the factorization is stable in finite precision; static/dynamic regularization and iterative refinement remain mandatory.

### 3.2 Projection versus barrier

For a closed convex cone,

\[
\Pi_{\mathcal K} = (I + N_{\mathcal K})^{-1}.
\]

ADMM/DR evaluate the resolvent of the normal cone (a Euclidean projection). IPM replaces complementarity \(s\perp z\) by a barrier gradient \(z + \mu\nabla F(s) = 0\) and Newton-linearizes it. Both are regularizations of the same monotone inclusion. They are not interchangeable half-steps. Replacing a barrier corrector by a projection, or Nesterov–Todd scaling by an exponential-cone Euclidean projection, voids the convergence argument.

### 3.3 Homogeneous embeddings versus drifting iterates

Clarabel and SCS v3 start from the same Andersen–Ye homogeneous monotone complementarity problem for the QCP KKT map. For \(P\neq 0\) the embedding is homogeneous but **not** the classical linear self-dual embedding.

- SCS applies Douglas–Rachford to a maximal monotone extension, including the \(\tau=0\) boundary.
- Clarabel applies a primal-dual barrier Newton method to the same embedding.

COSMO/OSQP do not homogenize. They detect infeasibility from limiting one-step differences (Banjac–Stellato–Boyd). That theory is for a specific ADMM operator. It does not automatically survive arbitrary acceleration, metric changes, or extra blocks.

### 3.4 Warm starts are algorithm-specific

Operator splitting lives on the boundary of \(\mathcal K\) and *wants* to. Interior-point methods live in the strict interior and *cannot start on the boundary*: the Hessian blows up and steps collapse. That is why Clarabel and ECOS still cold-start iterates even when they reuse symbolic factorizations. A sequential solver that copies \(x^\star\) into an IPM without recentering is not “warm.” It is often worse than a default interior point.

---

## 4. Proposed method

ConiX is a **typed sequential operator** with three engines and a one-way switch, not a single unproved hybrid iteration.

```text
canonicalize once -> classify update (R0/R1/R2)
                 -> choose engine
                 -> solve with persisted state
                 -> verify in original coordinates
                 -> accept or fallback
```

### 4.1 Engine S — cached homogeneous Douglas–Rachford (default sequential path)

Follow SCS v3. Form the monotone QCP operator whose symmetric part is determined by \(P\succeq 0\), homogenize with \((\tau,\kappa)\), and apply DR:

\[
\tilde u^{k+1} = (I+\mathcal Q)^{-1} w^k,\qquad
u^{k+1} = \Pi_{\mathcal C\times\mathbb R_+}(2\tilde u^{k+1}-w^k),\qquad
w^{k+1} = w^k + u^{k+1} - \tilde u^{k+1}.
\]

The resolvent \((I+\mathcal Q)^{-1}\) reduces to one quasi-definite solve

\[
\begin{bmatrix}
R_x+P & A^\top \\
A & -R_y
\end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} r_x \\ r_y \end{bmatrix}
\]

plus a scalar quadratic equation for the homogeneous variables. If \(P,A,R_x,R_y\) are frozen, the numeric factor is a constant of the entire **R0** sequence.

**Why this is the default.** It has (i) general cone projections, including exponential and power cones; (ii) true iterate warm starts; (iii) exact numeric-factor reuse on rolling \(q,b\); (iv) the same certificate geometry as Clarabel’s embedding.

**Why it is not the only engine.** High-accuracy tails are slow. Adaptive metrics that change \(R\) destroy the factor cache. Weakly infeasible problems leave \((\tau,\kappa)\) ambiguous for a long time.

**Cone oracle.** Euclidean projection, to a controlled inner tolerance. Dual projection via Moreau: \(\Pi_{\mathcal K}(v) + \Pi_{\mathcal K^\ast}(-v) = v\).

### 4.2 Engine Q — proximal ADMM for polyhedral QCPs

When \(\mathcal K\) is a product of zero and nonnegative cones (CVaR, MAD, CDaR, long-short boxes, linear transaction costs), drop the embedding and run COSMO/OSQP-style two-block ADMM:

\[
\begin{bmatrix}
P+\sigma I & A^\top \\
A & -\rho^{-1}I
\end{bmatrix}
\begin{bmatrix} \tilde x^{k+1} \\ \nu^{k+1} \end{bmatrix}
=
\begin{bmatrix}
-q+\sigma x^k \\
b-s^k+\rho^{-1} y^k
\end{bmatrix},
\qquad
s^{k+1}=\Pi_{\mathcal K}(\cdots).
\]

Projection is clipping. For **R0**, the factor is constant. Over-relaxation and residual-based \(\rho\) updates are allowed only with the usual refactor accounting: a \(\rho\) change that saves ten cheap iterations and costs one sparse factor is a net loss on a short horizon and a net win on a long one.

Infeasibility uses Banjac rays from iterate differences, then the independent verifier of §6. Polishing is OSQP-style equality QP on an identified active set, accepted only if original residuals improve.

### 4.3 Engine I — homogeneous predictor-corrector IPM (fallback)

Use Clarabel’s quadratic homogeneous embedding. Linearization reduces every Newton step to one factorization of

\[
\begin{bmatrix} P & A^\top \\ A & -H \end{bmatrix}
\]

and three triangular solves (affine predictor, centering/corrector, combined). Symmetric cones use Nesterov–Todd scaling. On **polyhedral** cones the NT Hessian is diagonal, \(H=\mathrm{diag}(s./z)\), which is the ADMM KKT of §3.1 with \(\rho_i=z_i/s_i\). ConiX therefore runs the polyhedral IPM by rewriting only those diagonals on the cached AMD-ordered factor; \(\sigma\) and \(\rho\) are restored afterwards so a later R0 ADMM step still matches the sequential contract. Nonsymmetric cones use a barrier Hessian plus a *derived* primal-dual scaling (Clarabel’s low-rank BFGS-type map, or Dahl–Andersen third-order exponential-cone corrections). NT scaling is **not** used on exponential or power cones.

**Hot start, not a copied optimum.** Keep two points from a previous IPM solve: the last strictly interior iterate at barrier parameter \(\mu_{\mathrm{anchor}}>0\), and the accepted solution. For a new \(\theta_t\),

1. optionally apply a KKT sensitivity predictor if the previous active face looks stable;
2. take a convex combination with a default centered point (Skajaa–Andersen–Ye);
3. enforce strict cone interiority and a centrality ceiling;
4. accept the start only if scaled residual and centrality beat the default initializer.

A near-optimal splitting point from Engine S is first *interiorized and recentered* before it is handed to Engine I. The switch is one-way. Alternating S and I inside one solve does not inherit either convergence theorem.

Clarabel’s paper notes that for \(P\neq 0\) with nonsymmetric cones, existence of the proposed central path is not theorem-covered. The conservative modelling move is always available: epigraph the quadratic into an SOC and recover a standard conic IPM. Engine I should expose that lift as a robustness option, not as the default (the lift can destroy sparsity).

### 4.4 Acceleration, and what is forbidden

Let \(T\) be the DR or ADMM map of the *current* data, \(R(w)=w-T(w)\).

**Allowed, with SuperMann / Type-I Anderson safeguards.** Short-memory Anderson on \(R\); restarted limited-memory Broyden on \(R\); Barzilai–Borwein *only* as a candidate relaxation inside the averaged-operator interval, or as an infrequent \(\rho\) proposal that must pass a correlation test and a refactor-cost test.

Acceptance: finite candidate, sufficient decrease of a current-problem residual merit, else a plain Krasnosel’skiĭ–Mann / ADMM step. Certificates are never read from an accelerated ghost iterate.

**Forbidden.**

- L-BFGS as an inverse Jacobian of a nonsymmetric nonsmooth residual.
- Carrying Anderson/Broyden history across a data, scaling, or penalty change.
- Claiming global convergence for vanilla AA or BB.
- Reusing numeric factors after \(P,A,H,\rho,R\) changed.
- Treating exponential/power boundaries as linear active sets.

---

## 5. Persisted state (typed, not a blob)

State is a product of caches with explicit dirty flags.

1. **Pattern.** Dimensions, CSC index arrays, cone sequence, finite-bound mask. Hash *and* compare; a 64-bit fingerprint is not authority.
2. **Analysis.** AMD/ordering, elimination tree, scatter maps from \((P,A)\) and cone blocks into KKT slots, chordal clique tree if used, allocated workspaces.
3. **Numeric factor.** Valid only for a recorded \((P,A,D_x,D_z)\) checksum.
4. **Equilibration.** Ruiz-style diagonal maps. Reused by default on nearby instances; recomputed when a cheap drift metric fires. Certificates and warm starts live in *unscaled* coordinates and are mapped in/out.
5. **Engine iterates.**
   - S: embedding vector \(w\), \((x,s,y,\tau,\kappa)\).
   - Q: \((x,s,y)\) and frozen \((\rho,\sigma)\).
   - I: strictly interior \((x,s,z,\tau,\kappa,\mu)\), plus an anchor point.
6. **Acceleration memory.** Bound to one residual map. Reset on any map change.
7. **Finance auxiliaries.** Rebuilt from the proposed primal \(x\), never copied blindly: transaction slacks from \(x-x^0\), CVaR slacks from new losses, CDaR peaks from the new path, exponential/power slacks pushed strictly into their domains.

An update is transactional. A rejected partial write cannot leave a stale factor marked valid.

### 5.1 Thought experiment: low-rank **R1**

A rolling factor-covariance update \(P_t = P_{t-1} + UU^\top - VV^\top\) (window add/drop) is a signed low-rank modification of the KKT matrix when \(A\) is fixed. Candidate mechanisms: sparse \(LDL\) update/downdate, or Woodbury against the old factor.

This is *not* in the first implementation. It is mathematically attractive and numerically treacherous (indefinite KKT, downdate instability, accumulating rank). The rule, when tried later: attempt the update, check solve residual, pivot growth, and small-system condition; otherwise rebase with a full numeric factor. Idiosyncratic diagonal changes are full rank and skip this path.

---

## 6. Independent correctness layer

Every accepted status is recomputed in original, unscaled coordinates.

\[
\hat r_p = \frac{\|Ax+s-b\|}{1+\|b\|},\quad
\hat r_d = \frac{\|Px+q+A^\top z\|}{1+\|q\|},
\]

\[
\hat r_{\mathcal K}
=
\max\Bigl\{
\tfrac{\mathrm{dist}(s,\mathcal K)}{1+\|s\|},
\tfrac{\mathrm{dist}(z,\mathcal K^\ast)}{1+\|z\|}
\Bigr\},
\quad
\hat g = \frac{|p-d|}{1+|p|+|d|}.
\]

Success at tolerance \(\varepsilon\) requires all four \(\le\varepsilon\). Infeasibility rays are normalized to \(b^\top z=-1\) or \(q^\top d=-1\) and checked for cone membership and the linear conditions of §1. Homogeneous residuals must exclude the trivial origin. Weakly infeasible instances may return `indeterminate`.

Acceleration, polishing, and engine switches may change the incumbent only when this checker improves it. A backtest must not trade on `almost solved`, `closest feasible`, or an uncertified iterate.

---

## 7. Portfolio models and cones

All of the following are standard convex reductions. Exact cardinality, fixed costs, and true constant-rebalanced compounded drawdown are not convex; they are out of scope unless explicitly relaxed.

| Model | Cone product | Typical sequential class |
|---|---|---|
| Mean-variance | QP, or SOC/RSOC epigraph | R0 if only \(\mu\) moves; R1 if \(\Sigma\) moves |
| Linear book / leverage / piecewise costs | LP | R0 |
| CVaR / Rockafellar–Uryasev | LP | R0/R1 at fixed scenario count |
| MAD / Konno–Yamazaki | LP | R0/R1 at fixed window |
| CDaR / path drawdown (affine wealth) | LP | R1 (most path coefficients move) |
| EVaR | one exponential cone per scenario | R0/R1 at fixed support |
| Ellipsoidal robust mean | SOC | R0/R1 |
| Log / Kelly utility | exponential | R0/R1 |
| CRRA / \(p\)-impact | power or generalized power | cone parameter change is R2 |
| Entropy / KL | exponential | R0/R1 |
| SDP relaxations / some DRO | PSD | structure is expensive; chordal when sparse |

Numerical modelling rules that dominate algorithm choice:

- Prefer factor loadings \(G\) with \(\Sigma=G^\top G\) over a formed dense covariance.
- Keep returns, weights, and SOC volatilities in commensurate units; do not mix dollar expected return with variance of size \(10^{-4}\) without scaling.
- For CVaR near \(\beta\to 1\), multiply through by \(1-\beta\) rather than forming a huge tail weight.
- Never materialize \(\exp(\ell_s/t)\) for EVaR; use the exponential-cone perspective.
- Affine-path CDaR does **not** extend automatically to compounded \(\prod(1+r^\top x)\).

Engine Q will win on CVaR/MAD/CDaR if implemented well. Engine S/I exist so EVaR, utilities, robust SOC, and mixed models stay inside one API.

---

## 8. Implementation skeleton (when code starts)

Not part of the mathematical claim, but the mathematics constrains the code:

- Rust, permissive license, CSC with documented triangle and structural-zero policy.
- Separate `Pattern`, `Analysis`, `Values`, `SolveState` types.
- Direct quasi-definite \(LDL\) (QDLDL-class simplicial for small fill; supernodal faer/FERAL when arithmetic intensity warrants it).
- Allocation-free update/solve in the R0 steady state.
- Cone kernels specialized by type and size; group work before threading.
- Parallelism first across independent dates/strategies, not inside one sparse factor.
- Presolve and chordal decomposition must expose update maps or they are disabled on sequential workspaces (Clarabel’s current rejection contract).

---

## 9. Hypotheses to test later (not tested here)

These are empirical questions. They are not assumed true.

1. **H1 (factor reuse).** On R0 sequences at checked \(10^{-6}\), Engine S or Q with a cached factor beats persistent Clarabel and matches or beats SCS/OSQP on their own domains by a material margin in geometric-mean steady-state latency.
2. **H2 (Anderson).** Safeguarded Anderson reduces time-to-tolerance \(\ge 20\%\) without increasing false statuses.
3. **H3 (polyhedral polish).** Active-set polishing reaches \(10^{-8}\) cheaper than extra ADMM steps on at least half of eligible LP/QP dates, with zero accepted degradations.
4. **H4 (IPM recenter).** Recentered Engine I hot starts reduce *numeric factorizations*, not merely iteration cosmetics, and reject themselves near active-face changes.
5. **H5 (hybrid headroom).** The per-date lower envelope of {S, Q, I} is at least \(25\%\) faster than the best single engine on mixed finance sequences; a fixed switch policy recovers most of that envelope.
6. **H6 (low-rank).** Guarded Woodbury/LDL updates help rolling factor-covariance R1 enough to beat numeric refactor after residual/stability checks. Expected to fail on full-rank diagonal risk updates.

Stop conditions: if H1 fails on the intended finance track, the project is not “a faster kernel,” it is a modelling/API project, and the design must be revised rather than decorated with more accelerators.

---

## 10. What “new” means here

The ingredients are known: homogeneous DR (SCS), quadratic IPM (Clarabel), sequential ADMM (OSQP/COSMO), SuperMann/Type-I Anderson, Skajaa–Andersen–Ye IPM warm starts, Banjac infeasibility rays, Newton-ADMM/semismooth refinement.

The design that is *not* already a product is the **typed sequential QCP machine**:

- one canonical form covering the finance cone list;
- an update classifier that makes factor reuse a theorem of the data, not a hope;
- engine-specific state and a one-way certifying switch;
- finance auxiliary reconstruction;
- an independent checker as part of the algorithm, not a benchmark script.

That is the method. A later kernel may add low-rank KKT updates or semismooth proximal-point refinement. Those are optional, guarded, and not the correctness backbone.

---

## 11. Primary sources

- Garstka, Cannon, Goulart, COSMO, *J. Optim. Theory Appl.*, 2021. [arXiv:1901.10887](https://arxiv.org/abs/1901.10887)
- Goulart, Chen, Clarabel, *Math. Prog. Comp.*, 2026. [arXiv:2405.12762](https://arxiv.org/abs/2405.12762)
- Stellato et al., OSQP, *Math. Prog. Comp.*, 2020. [arXiv:1711.08013](https://arxiv.org/abs/1711.08013)
- O’Donoghue, SCS QCP embedding, *SIAM J. Optim.*, 2021. [arXiv:2004.02177](https://arxiv.org/abs/2004.02177)
- O’Donoghue et al., original SCS, 2016. [arXiv:1312.3039](https://arxiv.org/abs/1312.3039)
- Andersen, Ye, homogeneous MCP. *Math. Prog.*, 1999.
- Banjac, Stellato, Boyd, ADMM infeasibility. 2019.
- Eckstein, Bertsekas, DR / ADMM / proximal point. 1992.
- Vanderbei, symmetric quasi-definite matrices. 1995.
- Skajaa, Andersen, Ye, warm-starting homogeneous IPMs. 2013.
- Zhang, O’Donoghue, Boyd, Type-I Anderson. 2020.
- Themelis, Patrinos, SuperMann. 2019.
- Rockafellar, Uryasev, CVaR. 2000.
- Konno, Yamazaki, MAD. 1991.
- Ahmadi-Javid, EVaR. 2012.
- Chekhlov, Uryasev, Zabarankin, CDaR. 2005.
