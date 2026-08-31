//! PSD triangle cone: svec packing, Euclidean projection, and Nesterov–Todd \(H_s\).
//!
//! Packed storage is Clarabel's col-major upper triangle with \(\sqrt{2}\) off-diagonals
//! so that \(\langle\mathrm{svec}(A),\mathrm{svec}(B)\rangle=\langle A,B\rangle_F\).
//! The NT matrix \(G\) satisfies \(G Z G = S\), and \(H_s=\mathrm{skron}(G)\) so
//! \(H_s\,\mathrm{svec}(X)=\mathrm{svec}(G X G)\) and \(H_s z = s\).

use std::f64::consts::SQRT_2;

const EIG_TOL: f64 = 1e-14;

pub(crate) fn psd_tri_len(side: usize) -> usize {
    side * (side + 1) / 2
}

/// Packed svec(\(I\)): ones on the matrix diagonal, zeros on off-diagonals.
pub(crate) fn psd_unit_svec(side: usize) -> Vec<f64> {
    let mut s = vec![0.0; psd_tri_len(side)];
    let mut k = 0usize;
    for j in 0..side {
        for i in 0..=j {
            if i == j {
                s[k] = 1.0;
            }
            k += 1;
        }
    }
    s
}

pub(crate) fn svec_to_mat(x: &[f64], side: usize) -> Vec<f64> {
    let n = side;
    let mut a = vec![0.0; n * n];
    let mut k = 0usize;
    for j in 0..n {
        for i in 0..=j {
            let v = if i == j { x[k] } else { x[k] / SQRT_2 };
            a[i * n + j] = v;
            a[j * n + i] = v;
            k += 1;
        }
    }
    a
}

pub(crate) fn mat_to_svec(a: &[f64], side: usize) -> Vec<f64> {
    let n = side;
    let mut x = vec![0.0; psd_tri_len(n)];
    let mut k = 0usize;
    for j in 0..n {
        for i in 0..=j {
            x[k] = if i == j {
                a[i * n + j]
            } else {
                SQRT_2 * 0.5 * (a[i * n + j] + a[j * n + i])
            };
            k += 1;
        }
    }
    x
}

pub(crate) fn project_psd_triangle(x: &mut [f64], side: usize) {
    if side == 0 {
        return;
    }
    let n = side;
    let mut a = svec_to_mat(x, n);
    let (w, q) = jacobi_eig(n, &mut a);
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
    let packed = mat_to_svec(&ap, n);
    x[..packed.len()].copy_from_slice(&packed);
}

/// NT Hessian as a dense row-major \(t\times t\) matrix, \(t=n(n+1)/2\).
pub(crate) fn psd_nt_hessian(s: &[f64], z: &[f64], side: usize) -> Option<Vec<f64>> {
    let t = psd_tri_len(side);
    if side == 0 {
        return Some(Vec::new());
    }
    if s.len() != t || z.len() != t {
        return None;
    }
    let g = nt_g(s, z, side)?;
    let hs = skron(&g, side);
    if hs.iter().all(|v| v.is_finite()) {
        Some(hs)
    } else {
        None
    }
}

pub(crate) fn psd_svec_inv(z: &[f64], side: usize) -> Option<Vec<f64>> {
    if side == 0 {
        return Some(Vec::new());
    }
    let mut zmat = svec_to_mat(z, side);
    let (w, q) = jacobi_eig(side, &mut zmat);
    if w.iter().any(|&lam| lam <= EIG_TOL) {
        return None;
    }
    let inv = eig_fun(side, &w, &q, |lam| 1.0 / lam);
    Some(mat_to_svec(&inv, side))
}

/// Jordan product \(\mathrm{svec}((YZ+ZY)/2)\).
pub(crate) fn psd_jordan(y: &[f64], z: &[f64], side: usize) -> Vec<f64> {
    let ymat = svec_to_mat(y, side);
    let zmat = svec_to_mat(z, side);
    let yz = mat_mul(side, &ymat, &zmat);
    let zy = mat_mul(side, &zmat, &ymat);
    let mut x = vec![0.0; side * side];
    for i in 0..x.len() {
        x[i] = 0.5 * (yz[i] + zy[i]);
    }
    mat_to_svec(&x, side)
}

/// Largest \(\alpha\in[0,1]\) with \(X+\alpha\Delta X\succ 0\).
pub(crate) fn psd_max_step(x: &[f64], dx: &[f64], side: usize) -> f64 {
    if side == 0 {
        return 1.0;
    }
    let mut xmat = svec_to_mat(x, side);
    let (w, q) = jacobi_eig(side, &mut xmat);
    if w.iter().any(|&lam| lam <= EIG_TOL) {
        return backtrack_pd(x, dx, side);
    }
    let dmat = svec_to_mat(dx, side);
    let qt = transpose(side, &q);
    let tmp = mat_mul(side, &qt, &dmat);
    let mut m = mat_mul(side, &tmp, &q);
    for i in 0..side {
        let si = 1.0 / w[i].sqrt();
        for j in 0..side {
            m[i * side + j] *= si / w[j].sqrt();
        }
    }
    let (gw, _) = jacobi_eig(side, &mut m);
    let mut gmin = gw[0];
    for &g in &gw {
        if g < gmin {
            gmin = g;
        }
    }
    let a = if gmin < 0.0 {
        (-1.0 / gmin).clamp(0.0, 1.0)
    } else {
        1.0
    };
    if pd_axpy(x, dx, a, side) {
        a
    } else {
        backtrack_pd(x, dx, side)
    }
}

fn backtrack_pd(x: &[f64], dx: &[f64], side: usize) -> f64 {
    let mut a = 1.0_f64;
    for _ in 0..48 {
        if pd_axpy(x, dx, a, side) {
            return a;
        }
        a *= 0.8;
        if a < 1e-14 {
            return 0.0;
        }
    }
    a
}

fn pd_axpy(x: &[f64], dx: &[f64], a: f64, side: usize) -> bool {
    let t = psd_tri_len(side);
    let mut y = vec![0.0; t];
    for i in 0..t {
        y[i] = x[i] + a * dx[i];
    }
    let mut ymat = svec_to_mat(&y, side);
    let (w, _) = jacobi_eig(side, &mut ymat);
    w.iter().all(|&lam| lam > EIG_TOL)
}

/// \(G = S^{1/2}(S^{1/2} Z S^{1/2})^{-1/2} S^{1/2}\) so that \(G Z G = S\).
fn nt_g(s: &[f64], z: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut smat = svec_to_mat(s, n);
    let zmat = svec_to_mat(z, n);
    let (sw, sq) = jacobi_eig(n, &mut smat);
    if sw.iter().any(|&lam| lam <= EIG_TOL) {
        return None;
    }
    let ssqrt = eig_fun(n, &sw, &sq, |lam| lam.sqrt());
    let mid = mat_mul(n, &mat_mul(n, &ssqrt, &zmat), &ssqrt);
    let mut mid_s = symmetrize(n, &mid);
    let (mw, mq) = jacobi_eig(n, &mut mid_s);
    if mw.iter().any(|&lam| lam <= EIG_TOL) {
        return None;
    }
    let mid_isqrt = eig_fun(n, &mw, &mq, |lam| 1.0 / lam.sqrt());
    let g = mat_mul(n, &mat_mul(n, &ssqrt, &mid_isqrt), &ssqrt);
    Some(symmetrize(n, &g))
}

/// Upper triangle of \(\mathrm{skron}(A)=A\otimes_s A\) as a full row-major \(t\times t\).
fn skron(a: &[f64], n: usize) -> Vec<f64> {
    let t = psd_tri_len(n);
    let mut out = vec![0.0; t * t];
    let mut col = 0usize;
    for l in 0..n {
        for k in 0..=l {
            let mut row = 0usize;
            let kl_eq = k == l;
            for j in 0..n {
                let ajl = a[j * n + l];
                let ajk = a[j * n + k];
                for i in 0..=j {
                    if row > col {
                        break;
                    }
                    let ij_eq = i == j;
                    let v = match (ij_eq, kl_eq) {
                        (false, false) => a[i * n + k] * ajl + a[i * n + l] * ajk,
                        (true, false) => SQRT_2 * ajl * ajk,
                        (false, true) => SQRT_2 * a[i * n + l] * ajk,
                        (true, true) => ajl * ajl,
                    };
                    out[row * t + col] = v;
                    out[col * t + row] = v;
                    row += 1;
                }
            }
            col += 1;
        }
    }
    out
}

fn eig_fun(n: usize, w: &[f64], q: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..n {
                s += q[i * n + t] * f(w[t]) * q[j * n + t];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn mat_mul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

fn transpose(n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

fn symmetrize(n: usize, a: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    s
}

pub(crate) fn jacobi_eig(n: usize, a: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    if n <= 1 {
        let w: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        return (w, q);
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

    fn mul_full(t: usize, h: &[f64], x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; t];
        for i in 0..t {
            for j in 0..t {
                y[i] += h[i * t + j] * x[j];
            }
        }
        y
    }

    #[test]
    fn skron_identity_is_identity() {
        let n = 3;
        let t = psd_tri_len(n);
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let h = skron(&eye, n);
        for i in 0..t {
            for j in 0..t {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (h[i * t + j] - want).abs() < 1e-12,
                    "Hs[{i},{j}]={}",
                    h[i * t + j]
                );
            }
        }
    }

    #[test]
    fn nt_at_identity() {
        let n = 2;
        let s = psd_unit_svec(n);
        let h = psd_nt_hessian(&s, &s, n).unwrap();
        let t = s.len();
        for i in 0..t {
            for j in 0..t {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((h[i * t + j] - want).abs() < 1e-10, "{h:?}");
            }
        }
    }

    #[test]
    fn nt_hs_z_equals_s() {
        // S = 2I, Z = I ⇒ G = √2 I, Hs = 2 I, Hs z = s.
        let n = 2;
        let z = psd_unit_svec(n);
        let s: Vec<f64> = z.iter().map(|v| 2.0 * v).collect();
        let h = psd_nt_hessian(&s, &z, n).unwrap();
        let hz = mul_full(z.len(), &h, &z);
        for i in 0..s.len() {
            assert!((hz[i] - s[i]).abs() < 1e-9, "hz={hz:?} s={s:?}");
        }
    }

    #[test]
    fn nt_random_pd_pair() {
        let n = 3;
        let g = [
            1.2, 0.3, -0.1, 0.3, 0.9, 0.2, -0.1, 0.2, 1.1, 0.8, -0.2, 0.4, -0.2, 1.0, 0.1, 0.4,
            0.1, 0.7,
        ];
        let mut s_mat = vec![0.0; n * n];
        let mut z_mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    s_mat[i * n + j] += g[i * n + k] * g[j * n + k];
                    z_mat[i * n + j] += g[n * n + i * n + k] * g[n * n + j * n + k];
                }
            }
            s_mat[i * n + i] += 0.5;
            z_mat[i * n + i] += 0.5;
        }
        let s = mat_to_svec(&s_mat, n);
        let z = mat_to_svec(&z_mat, n);
        let h = psd_nt_hessian(&s, &z, n).unwrap();
        let hz = mul_full(s.len(), &h, &z);
        for i in 0..s.len() {
            assert!((hz[i] - s[i]).abs() < 1e-8, "i={i} hz={} s={}", hz[i], s[i]);
        }
    }

    #[test]
    fn max_step_blocks_exit() {
        let n = 2;
        let s = psd_unit_svec(n);
        // dS = -2 I takes us through 0 at α=0.5.
        let ds: Vec<f64> = s.iter().map(|v| -2.0 * v).collect();
        let a = psd_max_step(&s, &ds, n);
        assert!(a > 0.4 && a <= 0.5 + 1e-9, "{a}");
        assert!(pd_axpy(&s, &ds, a, n));
        assert!(!pd_axpy(&s, &ds, 0.51, n));
    }
}
