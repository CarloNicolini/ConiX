//! Canonical finance QCP builders. Auxiliaries are reconstructed from `x`.

use crate::algebra::CscMatrix;
use crate::cones::{CompositeCone, Cone};
use crate::workspace::Qcp;

/// Mean-variance: `min λ/2 x'Σx - μ'x` s.t. `1'x = 1`, `l ≤ x ≤ u`.
pub fn mean_variance(sigma: &CscMatrix, mu: &[f64], l: &[f64], u: &[f64], lambda: f64) -> Qcp {
    let n = mu.len();
    let mut p = sigma.clone();
    for v in p.x.iter_mut() {
        *v *= lambda;
    }
    let q: Vec<f64> = mu.iter().map(|v| -v).collect();
    let mut trips = Vec::new();
    // 1'x = 1
    for j in 0..n {
        trips.push((0, j, 1.0));
    }
    // x ≤ u  →  x + s = u, s ≥ 0
    for j in 0..n {
        trips.push((1 + j, j, 1.0));
    }
    // -x ≤ -l → -x + s = -l
    for j in 0..n {
        trips.push((1 + n + j, j, -1.0));
    }
    let a = CscMatrix::from_triplets_keep_zeros(1 + 2 * n, n, &trips);
    let mut b = vec![1.0];
    b.extend_from_slice(u);
    for &li in l {
        b.push(-li);
    }
    let cones = CompositeCone::new(vec![
        Cone::Zero { dim: 1 },
        Cone::Nonnegative { dim: 2 * n },
    ]);
    Qcp { p, q, a, b, cones }
}

/// CVaR of scenario losses `ℓ_s = -r_s' x` at level `β`.
///
/// `min η + 1/((1-β)T) Σ z_s` s.t. `z ≥ -R x - η 1`, `z ≥ 0`, `1'x=1`, bounds.
pub fn cvar(returns: &[Vec<f64>], beta: f64, l: &[f64], u: &[f64]) -> Qcp {
    let t = returns.len();
    let n = l.len();
    let nv = n + 1 + t; // x, η, z
    let p = CscMatrix::zeros(nv, nv);
    let tail = 1.0 / ((1.0 - beta) * t as f64);
    let mut q = vec![0.0; nv];
    q[n] = 1.0;
    for i in 0..t {
        q[n + 1 + i] = tail;
    }
    let mut trips = Vec::new();
    let mut b = Vec::new();
    let mut row = 0usize;
    // 1'x = 1
    for j in 0..n {
        trips.push((row, j, 1.0));
    }
    b.push(1.0);
    row += 1;
    // x ≤ u
    for j in 0..n {
        trips.push((row, j, 1.0));
        b.push(u[j]);
        row += 1;
    }
    // -x ≤ -l
    for j in 0..n {
        trips.push((row, j, -1.0));
        b.push(-l[j]);
        row += 1;
    }
    // z_s + η + r_s' x ≥ 0  →  -r'x - η - z ≤ 0
    for s in 0..t {
        for j in 0..n {
            trips.push((row, j, -returns[s][j]));
        }
        trips.push((row, n, -1.0));
        trips.push((row, n + 1 + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    // z ≥ 0 → -z ≤ 0
    for s in 0..t {
        trips.push((row, n + 1 + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    let a = CscMatrix::from_triplets_keep_zeros(row, nv, &trips);
    let cones = CompositeCone::new(vec![
        Cone::Zero { dim: 1 },
        Cone::Nonnegative { dim: row - 1 },
    ]);
    Qcp { p, q, a, b, cones }
}

/// MAD / Konno–Yamazaki: `min Σ p_s d_s` s.t. `d ≥ ±(r_s - r̄)'x`, budget, bounds.
pub fn mad(returns: &[Vec<f64>], probs: &[f64], l: &[f64], u: &[f64]) -> Qcp {
    let t = returns.len();
    let n = l.len();
    let nv = n + t;
    let p = CscMatrix::zeros(nv, nv);
    let mut q = vec![0.0; nv];
    for s in 0..t {
        q[n + s] = probs[s];
    }
    let mut rbar = vec![0.0; n];
    for s in 0..t {
        for j in 0..n {
            rbar[j] += probs[s] * returns[s][j];
        }
    }
    let mut trips = Vec::new();
    let mut b = Vec::new();
    let mut row = 0usize;
    for j in 0..n {
        trips.push((row, j, 1.0));
    }
    b.push(1.0);
    row += 1;
    for j in 0..n {
        trips.push((row, j, 1.0));
        b.push(u[j]);
        row += 1;
    }
    for j in 0..n {
        trips.push((row, j, -1.0));
        b.push(-l[j]);
        row += 1;
    }
    for s in 0..t {
        for j in 0..n {
            let c = returns[s][j] - rbar[j];
            trips.push((row, j, c));
        }
        trips.push((row, n + s, -1.0));
        b.push(0.0);
        row += 1;
        for j in 0..n {
            let c = returns[s][j] - rbar[j];
            trips.push((row, j, -c));
        }
        trips.push((row, n + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    let a = CscMatrix::from_triplets_keep_zeros(row, nv, &trips);
    let cones = CompositeCone::new(vec![
        Cone::Zero { dim: 1 },
        Cone::Nonnegative { dim: row - 1 },
    ]);
    Qcp { p, q, a, b, cones }
}

/// Affine-path CDaR: peak `h_t`, drawdown `h_t - Y_t`, CVaR of drawdowns.
pub fn cdar(path_returns: &[Vec<f64>], beta: f64, l: &[f64], u: &[f64]) -> Qcp {
    let t = path_returns.len();
    let n = l.len();
    // x (n), h (t), η (1), z (t)
    let nv = n + t + 1 + t;
    let p = CscMatrix::zeros(nv, nv);
    let tail = 1.0 / ((1.0 - beta) * t as f64);
    let mut q = vec![0.0; nv];
    q[n + t] = 1.0;
    for s in 0..t {
        q[n + t + 1 + s] = tail;
    }
    let h0 = n;
    let eta = n + t;
    let z0 = n + t + 1;
    let mut trips = Vec::new();
    let mut b = Vec::new();
    let mut row = 0usize;
    for j in 0..n {
        trips.push((row, j, 1.0));
    }
    b.push(1.0);
    row += 1;
    for j in 0..n {
        trips.push((row, j, 1.0));
        b.push(u[j]);
        row += 1;
    }
    for j in 0..n {
        trips.push((row, j, -1.0));
        b.push(-l[j]);
        row += 1;
    }
    // Y_t = sum_{k=0..t} r_k' x  (cumulative)
    // h_t >= h_{t-1}: h_{t-1} - h_t ≤ 0
    for s in 1..t {
        trips.push((row, h0 + s - 1, 1.0));
        trips.push((row, h0 + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    // h_t >= Y_t: Y_t - h_t ≤ 0
    let mut cum = vec![vec![0.0; n]; t];
    for s in 0..t {
        for j in 0..n {
            cum[s][j] = path_returns[s][j];
            if s > 0 {
                cum[s][j] += cum[s - 1][j];
            }
            trips.push((row, j, cum[s][j]));
        }
        trips.push((row, h0 + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    // z_t >= h_t - Y_t - η  →  -Y_t + h_t - η - z_t ≤ 0 wait:
    // z >= D - η, D = h - Y → z - h + Y + η >= 0 → -z + h - Y - η ≤ 0
    for s in 0..t {
        for j in 0..n {
            trips.push((row, j, -cum[s][j]));
        }
        trips.push((row, h0 + s, 1.0));
        trips.push((row, eta, -1.0));
        trips.push((row, z0 + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    for s in 0..t {
        trips.push((row, z0 + s, -1.0));
        b.push(0.0);
        row += 1;
    }
    let a = CscMatrix::from_triplets_keep_zeros(row, nv, &trips);
    let cones = CompositeCone::new(vec![
        Cone::Zero { dim: 1 },
        Cone::Nonnegative { dim: row - 1 },
    ]);
    Qcp { p, q, a, b, cones }
}

/// EVaR with one exponential cone per scenario (Ahmadi-Javid).
///
/// `min z - t log(1-β)` s.t. `Σ p_s u_s ≤ t` and
/// `(ℓ_s(x)-z, t, u_s) ∈ EXP`, i.e. `t exp((ℓ_s-z)/t) ≤ u_s`.
/// Together these are `E[exp((ℓ-z)/t)] ≤ 1`.
pub fn evar(returns: &[Vec<f64>], probs: &[f64], beta: f64, l: &[f64], u: &[f64]) -> Qcp {
    let t = returns.len();
    let n = l.len();
    // x, z_var, t_persp, u[T]
    let nv = n + 2 + t;
    let p = CscMatrix::zeros(nv, nv);
    let mut q = vec![0.0; nv];
    q[n] = 1.0; // z
    q[n + 1] = -(1.0 - beta).ln(); // -t log(1-β)
    let zvar = n;
    let tper = n + 1;
    let u0 = n + 2;
    let mut trips = Vec::new();
    let mut b = Vec::new();
    let mut cones = Vec::new();
    let mut row = 0usize;
    for j in 0..n {
        trips.push((row, j, 1.0));
    }
    b.push(1.0);
    cones.push(Cone::Zero { dim: 1 });
    row += 1;
    for j in 0..n {
        trips.push((row, j, 1.0));
        b.push(u[j]);
        row += 1;
    }
    for j in 0..n {
        trips.push((row, j, -1.0));
        b.push(-l[j]);
        row += 1;
    }
    cones.push(Cone::Nonnegative { dim: 2 * n });
    // Σ p u - t ≤ 0
    for s in 0..t {
        trips.push((row, u0 + s, probs[s]));
    }
    trips.push((row, tper, -1.0));
    b.push(0.0);
    cones.push(Cone::Nonnegative { dim: 1 });
    row += 1;
    // EXP: (ℓ_s - z, t, u_s) with ℓ_s = -r_s'x, i.e. y exp(x/y) ≤ z on (x,y,z).
    // Then t exp((ℓ-z)/t) ≤ u and Σ p u ≤ t  ⇒  E[exp((ℓ-z)/t)] ≤ 1.
    // Ax + s = b:
    //   s0 = ℓ - z = -r'x - z  →  r'x + z + s0 = 0
    //   s1 = t                 →  -t + s1 = 0
    //   s2 = u                 →  -u + s2 = 0
    for s in 0..t {
        for j in 0..n {
            trips.push((row, j, returns[s][j]));
        }
        trips.push((row, zvar, 1.0));
        b.push(0.0);
        trips.push((row + 1, tper, -1.0));
        b.push(0.0);
        trips.push((row + 2, u0 + s, -1.0));
        b.push(0.0);
        cones.push(Cone::Exponential);
        row += 3;
    }
    let a = CscMatrix::from_triplets_keep_zeros(row, nv, &trips);
    Qcp {
        p,
        q,
        a,
        b,
        cones: CompositeCone::new(cones),
    }
}

/// Reconstruct CVaR slacks from a primal portfolio `x`.
pub fn cvar_slacks(returns: &[Vec<f64>], x: &[f64], eta: f64) -> (Vec<f64>, f64) {
    let mut z = Vec::with_capacity(returns.len());
    let mut losses = Vec::new();
    for r in returns {
        let mut ret = 0.0;
        for (j, &xj) in x.iter().enumerate() {
            ret += r[j] * xj;
        }
        losses.push(-ret);
        z.push((-ret - eta).max(0.0));
    }
    let eta = eta.max(*losses.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&0.0));
    (z, eta)
}
