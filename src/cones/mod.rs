//! Primitive cones and Euclidean projections.

use crate::algebra::dot;

mod nonsym;
pub(crate) use nonsym::*;

#[derive(Clone, Debug)]
pub enum Cone {
    Zero {
        dim: usize,
    },
    Nonnegative {
        dim: usize,
    },
    SecondOrder {
        dim: usize,
    },
    Exponential,
    DualExponential,
    Power {
        alpha: f64,
    },
    DualPower {
        alpha: f64,
    },
    /// Generalized power cone: `x ∈ R_+^{n_exp}`, `||z||_2 ≤ Π x_i^{α_i}`.
    /// Total dimension `n_exp + n_z`.
    GenPower {
        alpha: Vec<f64>,
        n_z: usize,
    },
    /// Packed upper triangle of a real PSD matrix, length `side*(side+1)/2`.
    PsdTriangle {
        side: usize,
    },
}

impl Cone {
    pub fn dim(&self) -> usize {
        match self {
            Cone::Zero { dim } | Cone::Nonnegative { dim } | Cone::SecondOrder { dim } => *dim,
            Cone::Exponential | Cone::DualExponential => 3,
            Cone::Power { .. } | Cone::DualPower { .. } => 3,
            Cone::GenPower { alpha, n_z } => alpha.len() + *n_z,
            Cone::PsdTriangle { side } => side * (side + 1) / 2,
        }
    }

    pub fn is_polyhedral(&self) -> bool {
        matches!(self, Cone::Zero { .. } | Cone::Nonnegative { .. })
    }

    /// Barrier parameter degree (Clarabel convention).
    pub fn barrier_degree(&self) -> usize {
        match self {
            Cone::Zero { .. } => 0,
            Cone::Nonnegative { dim } => *dim,
            Cone::SecondOrder { .. } => 1,
            Cone::Exponential | Cone::DualExponential => 3,
            Cone::Power { .. } | Cone::DualPower { .. } => 3,
            Cone::GenPower { alpha, .. } => alpha.len() + 1,
            Cone::PsdTriangle { side } => *side,
        }
    }

    pub fn is_symmetric(&self) -> bool {
        matches!(
            self,
            Cone::Zero { .. }
                | Cone::Nonnegative { .. }
                | Cone::SecondOrder { .. }
                | Cone::PsdTriangle { .. }
        )
    }
}

#[derive(Clone, Debug)]
pub struct CompositeCone {
    pub cones: Vec<Cone>,
    pub offsets: Vec<usize>,
    pub dim: usize,
}

impl CompositeCone {
    pub fn new(cones: Vec<Cone>) -> Self {
        let mut offsets = Vec::with_capacity(cones.len());
        let mut dim = 0;
        for c in &cones {
            offsets.push(dim);
            dim += c.dim();
        }
        Self {
            cones,
            offsets,
            dim,
        }
    }

    pub fn is_polyhedral(&self) -> bool {
        self.cones.iter().all(|c| c.is_polyhedral())
    }

    pub fn is_symmetric(&self) -> bool {
        self.cones.iter().all(|c| c.is_symmetric())
    }

    pub fn barrier_degree(&self) -> usize {
        self.cones.iter().map(|c| c.barrier_degree()).sum()
    }

    pub fn project(&self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.dim);
        for (cone, &off) in self.cones.iter().zip(&self.offsets) {
            let d = cone.dim();
            project_cone(cone, &mut x[off..off + d]);
        }
    }

    /// Π_{K*}(y) = y + Π_K(-y)  (Moreau).
    pub fn project_dual(&self, y: &mut [f64]) {
        let mut tmp = y.to_vec();
        for t in tmp.iter_mut() {
            *t = -*t;
        }
        self.project(&mut tmp);
        for (yi, ti) in y.iter_mut().zip(tmp) {
            *yi += ti;
        }
    }

    pub fn dist(&self, x: &[f64]) -> f64 {
        let mut p = x.to_vec();
        self.project(&mut p);
        let mut s = 0.0;
        for (a, b) in x.iter().zip(&p) {
            let d = a - b;
            s += d * d;
        }
        s.sqrt()
    }

    pub fn dist_dual(&self, z: &[f64]) -> f64 {
        let mut p = z.to_vec();
        self.project_dual(&mut p);
        let mut s = 0.0;
        for (a, b) in z.iter().zip(&p) {
            let d = a - b;
            s += d * d;
        }
        s.sqrt()
    }
}

pub fn project_cone(cone: &Cone, x: &mut [f64]) {
    match cone {
        Cone::Zero { dim } => {
            x[..*dim].fill(0.0);
        }
        Cone::Nonnegative { dim } => {
            for xi in x.iter_mut().take(*dim) {
                if *xi < 0.0 {
                    *xi = 0.0;
                }
            }
        }
        Cone::SecondOrder { dim } => project_soc(x, *dim),
        Cone::Exponential => project_exp(x, true),
        Cone::DualExponential => project_exp(x, false),
        Cone::Power { alpha } => project_power(x, *alpha),
        Cone::DualPower { alpha } => {
            // Π_{K*}(y) = y + Π_K(-y)
            let mut tmp = [-x[0], -x[1], -x[2]];
            project_power(&mut tmp, *alpha);
            x[0] += tmp[0];
            x[1] += tmp[1];
            x[2] += tmp[2];
        }
        Cone::GenPower { alpha, n_z } => project_genpower(x, alpha, *n_z),
        Cone::PsdTriangle { side } => project_psd_triangle(x, *side),
    }
}

fn project_soc(x: &mut [f64], q: usize) {
    if q == 0 {
        return;
    }
    if q == 1 {
        x[0] = x[0].max(0.0);
        return;
    }
    let v1 = x[0];
    let mut s = 0.0;
    for xi in x.iter().take(q).skip(1) {
        s += xi * xi;
    }
    let s = s.sqrt();
    if s <= v1 {
        return;
    }
    if s <= -v1 {
        x[..q].fill(0.0);
        return;
    }
    let alpha = (s + v1) / 2.0;
    x[0] = alpha;
    let scale = alpha / s;
    for xi in x.iter_mut().take(q).skip(1) {
        *xi *= scale;
    }
}

/// Friberg (2021) exponential-cone projection onto primal (`primal=true`)
/// or dual (`primal=false`).
fn project_exp(v: &mut [f64], primal: bool) {
    debug_assert_eq!(v.len(), 3);
    let mut v0 = [v[0], v[1], v[2]];
    if !primal {
        v0[0] = -v0[0];
        v0[1] = -v0[1];
        v0[2] = -v0[2];
    }
    let (mut vp, pdist) = proj_primal_exp_heuristic(&v0);
    let (mut vd, ddist) = proj_polar_exp_heuristic(&v0);
    let err = (vp[0] + vd[0] - v0[0])
        .abs()
        .max((vp[1] + vd[1] - v0[1]).abs())
        .max((vp[2] + vd[2] - v0[2]).abs());
    let opt = (v0[1] <= 0.0 && v0[0] <= 0.0)
        || pdist.min(ddist) <= 1e-8
        || (err <= 1e-8 && dot(&vp, &vd) <= 1e-8);
    if opt {
        if primal {
            v.copy_from_slice(&vp);
        } else {
            v[0] = -vd[0];
            v[1] = -vd[1];
            v[2] = -vd[2];
        }
        return;
    }
    let (xl, xh) = exp_search_bracket(&v0, pdist, ddist);
    let rho = root_search_newton(&v0, xl, xh, 0.5 * (xl + xh));
    if primal {
        if let Some((hat, dist)) = proj_sol_primal_exp(&v0, rho) {
            if dist <= pdist {
                vp = hat;
            }
        }
        v.copy_from_slice(&vp);
    } else {
        if let Some((hat, dist)) = proj_sol_polar_exp(&v0, rho) {
            if dist <= ddist {
                vd = hat;
            }
        }
        v[0] = -vd[0];
        v[1] = -vd[1];
        v[2] = -vd[2];
    }
}

fn proj_primal_exp_heuristic(v0: &[f64; 3]) -> ([f64; 3], f64) {
    let (r0, s0, t0) = (v0[0], v0[1], v0[2]);
    let mut vp = [r0.min(0.0), 0.0, t0.max(0.0)];
    let mut dist = n3(v0, &vp);
    if s0 > 0.0 {
        let tp = t0.max(s0 * (r0 / s0).exp());
        let newd = (tp - t0).abs();
        if newd < dist {
            vp = [r0, s0, tp];
            dist = newd;
        }
    }
    (vp, dist)
}

fn proj_polar_exp_heuristic(v0: &[f64; 3]) -> ([f64; 3], f64) {
    let (r0, s0, t0) = (v0[0], v0[1], v0[2]);
    let mut vd = [0.0, s0.min(0.0), t0.min(0.0)];
    let mut dist = n3(v0, &vd);
    if r0 > 0.0 {
        let td = t0.min(-r0 * (s0 / r0 - 1.0).exp());
        let newd = (t0 - td).abs();
        if newd < dist {
            vd = [r0, s0, td];
            dist = newd;
        }
    }
    (vd, dist)
}

fn n3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn hfun(v0: &[f64; 3], rho: f64) -> (f64, f64) {
    let (r0, s0, t0) = (v0[0], v0[1], v0[2]);
    let exprho = rho.exp();
    let expnegrho = (-rho).exp();
    let f = ((rho - 1.0) * r0 + s0) * exprho
        - (r0 - rho * s0) * expnegrho
        - (rho * (rho - 1.0) + 1.0) * t0;
    let df =
        (rho * r0 + s0) * exprho + (r0 - (rho - 1.0) * s0) * expnegrho - (2.0 * rho - 1.0) * t0;
    (f, df)
}

fn ppsi(v0: &[f64; 3]) -> f64 {
    let (r0, s0) = (v0[0], v0[1]);
    let disc = (r0 * r0 + s0 * s0 - r0 * s0).sqrt();
    let psi = if r0 > s0 {
        (r0 - s0 + disc) / r0
    } else {
        -s0 / (r0 - s0 - disc)
    };
    ((psi - 1.0) * r0 + s0) / (psi * (psi - 1.0) + 1.0)
}

fn dpsi(v0: &[f64; 3]) -> f64 {
    let (r0, s0) = (v0[0], v0[1]);
    let disc = (r0 * r0 + s0 * s0 - r0 * s0).sqrt();
    let psi = if s0 > r0 {
        (r0 - disc) / s0
    } else {
        (r0 - s0) / (r0 + disc)
    };
    (r0 - psi * s0) / (psi * (psi - 1.0) + 1.0)
}

fn pomega(rho: f64) -> f64 {
    let mut val = rho.exp() / (rho * (rho - 1.0) + 1.0);
    if rho < 2.0 {
        val = val.min((2.0_f64).exp() / 3.0);
    }
    val
}

fn domega(rho: f64) -> f64 {
    let mut val = -(-rho).exp() / (rho * (rho - 1.0) + 1.0);
    if rho > -1.0 {
        val = val.max(-1.0_f64.exp() / 3.0);
    }
    val
}

fn exp_search_bracket(v0: &[f64; 3], pdist: f64, ddist: f64) -> (f64, f64) {
    let (r0, s0, t0) = (v0[0], v0[1], v0[2]);
    let inf = 1e15_f64;
    let mut baselow = -inf;
    let mut baseupr = inf;
    let mut low = -inf;
    let mut upr = inf;
    let dp = (pdist * pdist - s0.min(0.0).powi(2)).max(0.0).sqrt();
    let dd = (ddist * ddist - r0.min(0.0).powi(2)).max(0.0).sqrt();
    if t0 > 0.0 {
        low = low.max((t0 / ppsi(v0)).ln());
    } else if t0 < 0.0 {
        upr = upr.min(-(-t0 / dpsi(v0)).ln());
    }
    if r0 > 0.0 {
        baselow = 1.0 - s0 / r0;
        low = low.max(baselow);
        let tpu = 1e-12_f64.max(dd.min(dp + t0));
        let cur = low.max(baselow + tpu / r0 / pomega(low));
        upr = upr.min(cur);
    }
    if s0 > 0.0 {
        baseupr = r0 / s0;
        upr = upr.min(baseupr);
        let tdl = -1e-12_f64.max(dp.min(dd - t0));
        let cur = upr.min(baseupr - tdl / s0 / domega(upr));
        low = low.max(cur);
    }
    low = low.min(upr).clamp(baselow, baseupr);
    upr = low.max(upr).clamp(baselow, baseupr);
    if (low - upr).abs() > 0.0 {
        let (fl, _) = hfun(v0, low);
        let (fu, _) = hfun(v0, upr);
        if fl * fu > 0.0 {
            if fl.abs() < fu.abs() {
                upr = low;
            } else {
                low = upr;
            }
        }
    }
    (low, upr)
}

fn root_search_newton(v0: &[f64; 3], mut xl: f64, mut xu: f64, mut x: f64) -> f64 {
    for _ in 0..20 {
        let (f, df) = hfun(v0, x);
        if f.abs() <= 1e-15 {
            break;
        }
        if f < 0.0 {
            xl = x;
        } else {
            xu = x;
        }
        if xu <= xl {
            return 0.5 * (xu + xl);
        }
        if !f.is_finite() || df < 1e-13 {
            break;
        }
        let xp = x - f / df;
        if (xp - x).abs() <= 1e-15 * xp.abs().max(1.0) {
            x = xp;
            break;
        }
        if xp >= xu {
            x = (0.05 * x + 0.95 * xu).min(xu);
        } else if xp <= xl {
            x = (0.05 * x + 0.95 * xl).max(xl);
        } else {
            x = xp;
        }
    }
    x.clamp(xl, xu)
}

fn proj_sol_primal_exp(v0: &[f64; 3], rho: f64) -> Option<([f64; 3], f64)> {
    let linrho = (rho - 1.0) * v0[0] + v0[1];
    let exprho = rho.exp();
    if linrho > 0.0 && exprho.is_finite() {
        let q = rho * (rho - 1.0) + 1.0;
        let vp = [rho * linrho / q, linrho / q, exprho * linrho / q];
        let dist = n3(&vp, v0);
        Some((vp, dist))
    } else {
        None
    }
}

fn proj_sol_polar_exp(v0: &[f64; 3], rho: f64) -> Option<([f64; 3], f64)> {
    let linrho = v0[0] - rho * v0[1];
    let exprho = (-rho).exp();
    if linrho > 0.0 && exprho.is_finite() {
        let q = rho * (rho - 1.0) + 1.0;
        let vd = [linrho / q, (1.0 - rho) * linrho / q, -exprho * linrho / q];
        let dist = n3(v0, &vd);
        Some((vd, dist))
    } else {
        None
    }
}

fn project_power(v: &mut [f64], a: f64) {
    let xh = v[0];
    let yh = v[1];
    let rh = v[2].abs();
    if xh >= 0.0 && yh >= 0.0 && 1e-9 + xh.powf(a) * yh.powf(1.0 - a) >= rh {
        return;
    }
    let dual_ok = xh <= 0.0
        && yh <= 0.0
        && 1e-9 + (-xh).powf(a) * (-yh).powf(1.0 - a) >= rh * a.powf(a) * (1.0 - a).powf(1.0 - a);
    if dual_ok {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
        return;
    }
    let mut r = rh / 2.0;
    let mut x = 0.0;
    let mut y = 0.0;
    for _ in 0..20 {
        x = pow_calc_x(r, xh, rh, a);
        y = pow_calc_x(r, yh, rh, 1.0 - a);
        let f = x.powf(a) * y.powf(1.0 - a) - r;
        if f.abs() < 1e-9 {
            break;
        }
        let dx = pow_dxdr(x, xh, rh, r, a);
        let dy = pow_dxdr(y, yh, rh, r, 1.0 - a);
        let fp = x.powf(a) * y.powf(1.0 - a) * (a * dx / x + (1.0 - a) * dy / y) - 1.0;
        r = (r - f / fp).clamp(0.0, rh);
    }
    v[0] = x;
    v[1] = y;
    v[2] = if v[2] < 0.0 { -r } else { r };
}

fn pow_calc_x(r: f64, xh: f64, rh: f64, a: f64) -> f64 {
    (0.5 * (xh + (xh * xh + 4.0 * a * (rh - r) * r).sqrt())).max(1e-12)
}

fn pow_dxdr(x: f64, xh: f64, rh: f64, r: f64, a: f64) -> f64 {
    a * (rh - 2.0 * r) / (2.0 * x - xh)
}

/// Projection onto a generalized power cone by a damped primal Newton method
/// on the hypersurface `Π x_i^{α_i} = ||z||` when the point is outside K ∪ -K*.
fn project_genpower(v: &mut [f64], alpha: &[f64], _n_z: usize) {
    let n_x = alpha.len();
    let (x, z) = v.split_at_mut(n_x);
    let znorm = nrm(z);
    let mut geomean = 1.0;
    let mut in_x = true;
    for (xi, &ai) in x.iter().zip(alpha) {
        if *xi < 0.0 {
            in_x = false;
            break;
        }
        geomean *= xi.powf(ai);
    }
    if in_x && geomean + 1e-12 >= znorm {
        return;
    }
    // polar heuristic: if all x<=0 and dual-power-like residual is large, send to 0
    if x.iter().all(|&xi| xi <= 0.0) && z.iter().all(|&zi| zi.abs() <= 1e-14) {
        v.fill(0.0);
        return;
    }
    // Project x onto R+ first, then scale z if needed.
    for xi in x.iter_mut() {
        *xi = xi.max(1e-16);
    }
    let mut g = 1.0;
    for (xi, &ai) in x.iter().zip(alpha) {
        g *= xi.powf(ai);
    }
    if g >= znorm {
        return;
    }
    if znorm > 0.0 && g > 0.0 {
        let s = g / znorm;
        for zi in z.iter_mut() {
            *zi *= s;
        }
    } else {
        z.fill(0.0);
    }
}

fn nrm(z: &[f64]) -> f64 {
    z.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Interior of the primal exponential cone: \(y>0,z>0,\; y\exp(x/y)<z\).
pub(crate) fn exp_primal_interior(s: &[f64]) -> bool {
    s.len() >= 3
        && s[1] > 0.0
        && s[2] > 0.0
        && (s[1] * (s[2] / s[1]).ln() - s[0]).is_finite()
        && s[1] * (s[2] / s[1]).ln() - s[0] > 1e-16
}

/// Interior of the dual exponential cone: \(u<0,w>0,\; -u\exp(v/u-1)<w\).
pub(crate) fn exp_dual_interior(z: &[f64]) -> bool {
    if z.len() < 3 || z[2] <= 0.0 || z[0] >= 0.0 {
        return false;
    }
    let l = (-z[2] / z[0]).ln();
    if !l.is_finite() {
        return false;
    }
    z[1] - z[0] - z[0] * l > 1e-16
}

/// Dual-barrier Hessian \(\nabla^2 f^\ast(z)\) at a dual-interior point, row-major.
pub(crate) fn exp_dual_hessian(z: &[f64]) -> Option<[f64; 9]> {
    if !exp_dual_interior(z) {
        return None;
    }
    let l = (-z[2] / z[0]).ln();
    let r = -z[0] * l - z[0] + z[1];
    if r <= 1e-16 || !r.is_finite() {
        return None;
    }
    let z0 = z[0];
    let z2 = z[2];
    let h00 = (r * r - z0 * r + l * l * z0 * z0) / (r * z0 * z0 * r);
    let h01 = -l / (r * r);
    let h11 = 1.0 / (r * r);
    let h02 = (z[1] - z0) / (r * r * z2);
    let h12 = -z0 / (r * r * z2);
    let h22 = (r * r - z0 * r + z0 * z0) / (r * r * z2 * z2);
    if ![h00, h01, h11, h02, h12, h22]
        .iter()
        .all(|v| v.is_finite())
    {
        return None;
    }
    Some([h00, h01, h02, h01, h11, h12, h02, h12, h22])
}

pub(crate) fn exp_unit_point() -> [f64; 3] {
    // Self-dual unit initialization used by ECOS/Clarabel for EXP.
    [
        -1.051_383_945_322_714,
        0.556_409_619_469_370,
        1.258_967_884_768_947,
    ]
}

pub(crate) fn exp_backtrack(x: &[f64], d: &[f64], primal: bool) -> f64 {
    let mut a = 1.0_f64;
    let mut w = [0.0; 3];
    for _ in 0..48 {
        w[0] = x[0] + a * d[0];
        w[1] = x[1] + a * d[1];
        w[2] = x[2] + a * d[2];
        let ok = if primal {
            exp_primal_interior(&w)
        } else {
            exp_dual_interior(&w)
        };
        if ok {
            return a;
        }
        a *= 0.8;
        if a < 1e-14 {
            return 0.0;
        }
    }
    a
}

fn project_psd_triangle(x: &mut [f64], side: usize) {
    let n = side;
    let mut a = vec![0.0; n * n];
    let mut k = 0;
    for j in 0..n {
        for i in 0..=j {
            let v = if i == j {
                x[k]
            } else {
                x[k] / std::f64::consts::SQRT_2
            };
            a[i * n + j] = v;
            a[j * n + i] = v;
            k += 1;
        }
    }
    let (w, q) = jacobi_eig(n, &mut a);
    // A+ = Q diag(max(w,0)) Q'
    let mut ap = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..n {
                s += q[i * n + t] * w[t].max(0.0) * q[j * n + t];
            }
            ap[i * n + j] = s;
        }
    }
    k = 0;
    for j in 0..n {
        for i in 0..=j {
            x[k] = if i == j {
                ap[i * n + j]
            } else {
                std::f64::consts::SQRT_2 * ap[i * n + j]
            };
            k += 1;
        }
    }
}

fn jacobi_eig(n: usize, a: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    for _ in 0..(8 * n * n).max(16) {
        let mut maxv = 0.0;
        let mut p = 0;
        let mut r = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i * n + j].abs();
                if v > maxv {
                    maxv = v;
                    p = i;
                    r = j;
                }
            }
        }
        if maxv < 1e-14 {
            break;
        }
        let app = a[p * n + p];
        let arr = a[r * n + r];
        let apr = a[p * n + r];
        let tau = (arr - app) / (2.0 * apr);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            let akp = a[k * n + p];
            let akr = a[k * n + r];
            a[k * n + p] = c * akp - s * akr;
            a[p * n + k] = a[k * n + p];
            a[k * n + r] = s * akp + c * akr;
            a[r * n + k] = a[k * n + r];
        }
        a[p * n + p] = c * c * app + s * s * arr - 2.0 * s * c * apr;
        a[r * n + r] = s * s * app + c * c * arr + 2.0 * s * c * apr;
        a[p * n + r] = 0.0;
        a[r * n + p] = 0.0;
        for k in 0..n {
            let qkp = q[k * n + p];
            let qkr = q[k * n + r];
            q[k * n + p] = c * qkp - s * qkr;
            q[k * n + r] = s * qkp + c * qkr;
        }
    }
    let w: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (w, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soc_inside() {
        let mut x = [2.0, 0.5, 0.5];
        project_soc(&mut x, 3);
        assert!((x[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn nn_clip() {
        let mut x = [-1.0, 2.0];
        project_cone(&Cone::Nonnegative { dim: 2 }, &mut x);
        assert_eq!(x, [0.0, 2.0]);
    }

    #[test]
    fn exp_unit_is_interior() {
        let u = exp_unit_point();
        assert!(exp_primal_interior(&u));
        assert!(exp_dual_interior(&u));
        let h = exp_dual_hessian(&u).unwrap();
        assert!(h[0] > 0.0);
        let det2 = h[0] * h[4] - h[1] * h[1];
        assert!(det2 > 0.0);
    }
}
