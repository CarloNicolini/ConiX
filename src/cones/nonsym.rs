//! Barrier gradients and Hessians for exponential and 3D power cones.
//!
//! Formulas follow Clarabel 0.9 (`expcone.rs`, `powcone.rs`, `nonsymmetric_common.rs`).
//! The centrality condition is \(s = -\mu\nabla f^\ast(z)\), linearized as
//! \(\Delta s + H_s\Delta z = -(s + \sigma\mu\nabla f^\ast(z))\).

use super::{exp_dual_hessian, exp_dual_interior, exp_primal_interior};

/// Dual-barrier gradient \(\nabla f^\ast(z)\) at a dual-interior exponential point.
pub(crate) fn exp_dual_grad(z: &[f64]) -> Option<[f64; 3]> {
    if !exp_dual_interior(z) {
        return None;
    }
    let l = (-z[2] / z[0]).ln();
    let r = -z[0] * l - z[0] + z[1];
    if r.abs() < 1e-16 || !r.is_finite() {
        return None;
    }
    let c2 = 1.0 / r;
    let g0 = c2 * l - 1.0 / z[0];
    let g1 = -c2;
    let g2 = (c2 * z[0] - 1.0) / z[2];
    if ![g0, g1, g2].iter().all(|v| v.is_finite()) {
        return None;
    }
    Some([g0, g1, g2])
}

/// Wright-Ω: solve \(y + \ln y = z\) for \(z \ge 0\).
pub(crate) fn wright_omega(z: f64) -> f64 {
    if z < 0.0 {
        return f64::NAN;
    }
    let mut w;
    if z < 1.0 + std::f64::consts::PI {
        let zm1 = z - 1.0;
        let mut p = zm1;
        w = 1.0 + p * 0.5;
        p *= zm1;
        w += p * (1.0 / 16.0);
        p *= zm1;
        w -= p * (1.0 / 192.0);
        p *= zm1;
        w -= p * (1.0 / 3072.0);
        p *= zm1;
        w += p * (13.0 / 61440.0);
    } else {
        let logz = z.ln();
        let zinv = 1.0 / z;
        w = z - logz;
        let mut q = logz * zinv;
        w += q;
        q *= zinv;
        w += q * (logz / 2.0 - 1.0);
        q *= zinv;
        w += q * (logz * logz / 3.0 - logz * 1.5 + 1.0);
    }
    let mut r = z - w - w.ln();
    for _ in 0..2 {
        let wp1 = w + 1.0;
        let t = wp1 * (wp1 + r * 2.0 / 3.0);
        w *= 1.0 + (r / wp1) * (t - r * 0.5) / (t - r);
        let r4 = r * r * r * r;
        let wp16 = wp1 * wp1 * wp1 * wp1 * wp1 * wp1;
        r = (w * w * 2.0 - w * 8.0 - 1.0) / (wp16 * 72.0) * r4;
    }
    w
}

/// Primal barrier gradient \(\nabla f(s)\) at a primal-interior exponential point.
pub(crate) fn exp_primal_grad(s: &[f64]) -> Option<[f64; 3]> {
    if !exp_primal_interior(s) {
        return None;
    }
    let arg = 1.0 - s[0] / s[1] - (s[1] / s[2]).ln();
    if arg < 0.0 || !arg.is_finite() {
        return None;
    }
    let omega = wright_omega(arg);
    if !omega.is_finite() || (omega - 1.0).abs() < 1e-16 {
        return None;
    }
    let g0 = 1.0 / ((omega - 1.0) * s[1]);
    let g1 = g0 + g0 * (omega * s[1] / s[2]).ln() - 1.0 / s[1];
    let g2 = omega / ((1.0 - omega) * s[2]);
    if ![g0, g1, g2].iter().all(|v| v.is_finite()) {
        return None;
    }
    Some([g0, g1, g2])
}

fn mul3(h: &[f64; 9], x: &[f64; 3]) -> [f64; 3] {
    [
        h[0] * x[0] + h[1] * x[1] + h[2] * x[2],
        h[3] * x[0] + h[4] * x[1] + h[5] * x[2],
        h[6] * x[0] + h[7] * x[1] + h[8] * x[2],
    ]
}

fn fro3(h: &[f64; 9]) -> f64 {
    h.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Scaling matrix \(H_s\) for the exponential cone.
/// Dual: \(H_s = \mu\nabla^2 f^\ast(z)\). Primal-dual: Clarabel low-rank map.
pub(crate) fn exp_hs(s: &[f64], z: &[f64], mu: f64, primal_dual: bool) -> Option<[f64; 9]> {
    let h_dual = exp_dual_hessian(z)?;
    if !primal_dual {
        return Some(scale9(&h_dual, mu));
    }
    let zt = exp_primal_grad(s)?;
    let st = exp_dual_grad(z)?;
    let mut dot_sz = 0.0_f64;
    for i in 0..3 {
        dot_sz += s[i] * z[i];
    }
    let mu_local = dot_sz / 3.0;
    let mut_t = (st[0] * zt[0] + st[1] * zt[1] + st[2] * zt[2]) / 3.0;
    let mut ds = [0.0; 3];
    let mut dz = [0.0; 3];
    for i in 0..3 {
        ds[i] = s[i] + mu_local * st[i];
        dz[i] = z[i] + mu_local * zt[i];
    }
    let mut dot_dsz = 0.0_f64;
    for i in 0..3 {
        dot_dsz += ds[i] * dz[i];
    }
    let hzt = mul3(&h_dual, &zt);
    let de1 = mu_local * mut_t - 1.0;
    let de2 = zt[0] * hzt[0] + zt[1] * hzt[1] + zt[2] * hzt[2] - 3.0 * mut_t * mut_t;
    let eps = f64::EPSILON.sqrt();
    if de1.abs() > eps && de2.abs() > f64::EPSILON && dot_sz > 0.0 && dot_dsz > 0.0 {
        let mut tmp = [0.0; 3];
        for i in 0..3 {
            tmp[i] = mut_t * st[i] - hzt[i];
        }
        let mut hs = h_dual;
        for i in 0..3 {
            for j in i..3 {
                let v = hs[i * 3 + j] - st[i] * st[j] / 3.0 - tmp[i] * tmp[j] / de2;
                hs[i * 3 + j] = v;
                hs[j * 3 + i] = v;
            }
        }
        let t = mu_local * fro3(&hs);
        let mut axis = [
            z[1] * zt[2] - z[2] * zt[1],
            z[2] * zt[0] - z[0] * zt[2],
            z[0] * zt[1] - z[1] * zt[0],
        ];
        let an = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if an < 1e-16 {
            return Some(scale9(&h_dual, mu_local));
        }
        for a in axis.iter_mut() {
            *a /= an;
        }
        for i in 0..3 {
            for j in i..3 {
                let v = s[i] * s[j] / dot_sz
                    + ds[i] * ds[j] / dot_dsz
                    + t * axis[i] * axis[j];
                hs[i * 3 + j] = v;
                hs[j * 3 + i] = v;
            }
        }
        if hs.iter().all(|v| v.is_finite()) {
            return Some(hs);
        }
    }
    Some(scale9(&h_dual, mu_local.max(mu)))
}

fn scale9(h: &[f64; 9], mu: f64) -> [f64; 9] {
    let mut o = [0.0; 9];
    for i in 0..9 {
        o[i] = mu * h[i];
    }
    o
}

pub(crate) fn power_primal_interior(s: &[f64], alpha: f64) -> bool {
    if s.len() < 3 || s[0] <= 0.0 || s[1] <= 0.0 {
        return false;
    }
    let phi = s[0].powf(2.0 * alpha) * s[1].powf(2.0 - 2.0 * alpha);
    phi.is_finite() && phi - s[2] * s[2] > 1e-16
}

pub(crate) fn power_dual_interior(z: &[f64], alpha: f64) -> bool {
    if z.len() < 3 || z[0] <= 0.0 || z[1] <= 0.0 {
        return false;
    }
    let a = alpha.clamp(1e-8, 1.0 - 1e-8);
    let phi = (z[0] / a).powf(2.0 * a) * (z[1] / (1.0 - a)).powf(2.0 - 2.0 * a);
    phi.is_finite() && phi - z[2] * z[2] > 1e-16
}

pub(crate) fn power_unit_point(alpha: f64) -> [f64; 3] {
    let a = alpha.clamp(1e-8, 1.0 - 1e-8);
    [(1.0 + a).sqrt(), (2.0 - a).sqrt(), 0.0]
}

pub(crate) fn power_dual_grad_h(z: &[f64], alpha: f64) -> Option<([f64; 3], [f64; 9])> {
    if !power_dual_interior(z, alpha) {
        return None;
    }
    let a = alpha.clamp(1e-8, 1.0 - 1e-8);
    let phi = (z[0] / a).powf(2.0 * a) * (z[1] / (1.0 - a)).powf(2.0 - 2.0 * a);
    let psi = phi - z[2] * z[2];
    if psi <= 1e-16 || !psi.is_finite() {
        return None;
    }
    let gpsi = [
        2.0 * a * phi / (z[0] * psi),
        2.0 * (1.0 - a) * phi / (z[1] * psi),
        -2.0 * z[2] / psi,
    ];
    let h00 = gpsi[0] * gpsi[0] - 2.0 * a * (2.0 * a - 1.0) * phi / (z[0] * z[0] * psi)
        + (1.0 - a) / (z[0] * z[0]);
    let h01 = gpsi[0] * gpsi[1] - 4.0 * a * (1.0 - a) * phi / (z[0] * z[1] * psi);
    let h11 = gpsi[1] * gpsi[1]
        - 2.0 * (1.0 - a) * (1.0 - 2.0 * a) * phi / (z[1] * z[1] * psi)
        + a / (z[1] * z[1]);
    let h02 = gpsi[0] * gpsi[2];
    let h12 = gpsi[1] * gpsi[2];
    let h22 = gpsi[2] * gpsi[2] + 2.0 / psi;
    let grad = [
        -2.0 * a * phi / (z[0] * psi) - (1.0 - a) / z[0],
        -2.0 * (1.0 - a) * phi / (z[1] * psi) - a / z[1],
        2.0 * z[2] / psi,
    ];
    if ![h00, h01, h11, h02, h12, h22, grad[0], grad[1], grad[2]]
        .iter()
        .all(|v| v.is_finite())
    {
        return None;
    }
    Some((grad, [h00, h01, h02, h01, h11, h12, h02, h12, h22]))
}

pub(crate) fn power_backtrack(x: &[f64], d: &[f64], alpha: f64, primal: bool) -> f64 {
    let mut a = 1.0_f64;
    let mut w = [0.0; 3];
    for _ in 0..48 {
        w[0] = x[0] + a * d[0];
        w[1] = x[1] + a * d[1];
        w[2] = x[2] + a * d[2];
        let ok = if primal {
            power_primal_interior(&w, alpha)
        } else {
            power_dual_interior(&w, alpha)
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

pub(crate) fn soc_interior(x: &[f64]) -> bool {
    if x.is_empty() || x[0] <= 0.0 {
        return false;
    }
    let mut n2 = 0.0_f64;
    for &v in &x[1..] {
        n2 += v * v;
    }
    x[0] * x[0] > n2 + 1e-16
}

/// NT scaling \(H = \eta^2(2ww^\top - J)\) for a second-order cone.
pub(crate) fn soc_nt_hessian(s: &[f64], z: &[f64]) -> Option<Vec<f64>> {
    let d = s.len();
    if z.len() != d || d < 2 || !soc_interior(s) || !soc_interior(z) {
        return None;
    }
    let sscale = soc_sqrt_res(s)?;
    let zscale = soc_sqrt_res(z)?;
    let eta2 = sscale / zscale;
    let mut w = vec![0.0; d];
    for i in 0..d {
        w[i] = s[i] / sscale;
    }
    w[0] += z[0] / zscale;
    for i in 1..d {
        w[i] -= z[i] / zscale;
    }
    let wscale = soc_sqrt_res(&w)?;
    for wi in w.iter_mut() {
        *wi /= wscale;
    }
    let mut w1sq = 0.0_f64;
    for &wi in &w[1..] {
        w1sq += wi * wi;
    }
    w[0] = (1.0 + w1sq).sqrt();
    let mut h = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut v = 2.0 * w[i] * w[j];
            if i == j {
                v += if i == 0 { -1.0 } else { 1.0 };
            }
            h[i * d + j] = eta2 * v;
        }
    }
    if h.iter().all(|v| v.is_finite()) {
        Some(h)
    } else {
        None
    }
}

fn soc_sqrt_res(x: &[f64]) -> Option<f64> {
    let mut r = x[0] * x[0];
    for &v in &x[1..] {
        r -= v * v;
    }
    if r <= 0.0 {
        None
    } else {
        Some(r.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wright_omega_identity() {
        for z in [1e-7, 1e-3, 0.1, 1.0, 10.0, 1e3] {
            let y = wright_omega(z);
            let err = (z - (y + y.ln())).abs();
            assert!(err / z.max(1e-12) < 1e-9, "z={z} y={y} err={err}");
        }
    }

    #[test]
    fn exp_unit_central() {
        let u = crate::cones::exp_unit_point();
        let g = exp_dual_grad(&u).unwrap();
        // At the self-dual unit, μ = 1 and s = -∇f*(z).
        for i in 0..3 {
            assert!((u[i] + g[i]).abs() < 1e-8, "i={i} u={} g={}", u[i], g[i]);
        }
        let hs = exp_hs(&u, &u, 1.0, false).unwrap();
        assert!(hs[0] > 0.0);
        let det2 = hs[0] * hs[4] - hs[1] * hs[1];
        assert!(det2 > 0.0);
        let pd = exp_hs(&u, &u, 1.0, true).unwrap();
        assert!(pd[0] > 0.0);
    }

    #[test]
    fn power_unit_interior() {
        let u = power_unit_point(0.5);
        assert!(power_primal_interior(&u, 0.5));
        assert!(power_dual_interior(&u, 0.5));
        let (g, h) = power_dual_grad_h(&u, 0.5).unwrap();
        assert!(h[0] > 0.0);
        assert!(g.iter().all(|v| v.is_finite()));
    }
}
