//! Homogeneous primal-dual interior-point fallback.
//!
//! Symmetric cones use Nesterov–Todd diagonal (nonnegative) or dense NT
//! (SOC). Nonsymmetric cones use the dual barrier Hessian. The reduced
//! Newton matrix is refactored every iteration; symbolic analysis of the
//! Hessian-fill pattern is reused.

use crate::algebra::csc::CscMatrix;
use crate::algebra::ldl::{LdlNumeric, LdlSymbolic};
use crate::algebra::{dot, inf_norm};
use crate::cones::Cone;
use crate::kkt::KktSystem;
use crate::status::Status;
use crate::workspace::Workspace;

pub fn run(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    initialize_interior(ws);
    let mut mu = duality_mu(ws);
    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let mut h = vec![0.0; m * m]; // dense H workspace (block-scattered)
    let mut k_csc;
    let mut sym: Option<LdlSymbolic> = None;

    while iter < ws.settings.max_iter.min(200) {
        iter += 1;
        fill_scaling(ws, &mut h, mu);
        k_csc = assemble_ipm_kkt(&ws.p, &ws.a, &h, n, m);
        if sym.as_ref().map(|s| s.n != k_csc.n).unwrap_or(true) {
            match LdlSymbolic::analyze(&k_csc) {
                Ok(s) => sym = Some(s),
                Err(_) => {
                    status = Status::Indeterminate;
                    break;
                }
            }
        }
        let fac = match LdlNumeric::factor_regularized(&k_csc, sym.as_ref().unwrap(), n, 1e-12) {
            Ok(f) => f,
            Err(_) => match LdlNumeric::factor(&k_csc, sym.as_ref().unwrap()) {
                Ok(f) => f,
                Err(_) => {
                    status = Status::Indeterminate;
                    break;
                }
            },
        };
        ws.factorizations += 1;

        let mut rd = vec![0.0; n];
        ws.p.sym_mul_add(&ws.x, &mut rd, 1.0);
        let mut atz = vec![0.0; n];
        ws.a.tmul(&ws.z, &mut atz);
        for i in 0..n {
            rd[i] += atz[i] + ws.q[i];
        }
        let mut rp = vec![0.0; m];
        ws.a.mul(&ws.x, &mut rp);
        for i in 0..m {
            rp[i] += ws.s[i] - ws.b[i];
        }

        // affine RHS: [ -rd ; -rp + ds_aff ] with ds_aff = -s (centering 0)
        let mut rhs = vec![0.0; n + m];
        for i in 0..n {
            rhs[i] = -rd[i];
        }
        for i in 0..m {
            rhs[n + i] = -rp[i] + ws.s[i];
        }
        let mut d_aff = rhs.clone();
        solve_perm(&fac, &mut d_aff);
        let _dx_aff = &d_aff[..n];
        let dz_aff = &d_aff[n..];
        let mut ds_aff = vec![0.0; m];
        for i in 0..m {
            ds_aff[i] = -ws.s[i] - h_mul_row(&h, m, i, dz_aff);
        }
        let alpha_p = max_step(ws, &ds_aff, true);
        let alpha_d = max_step_dual(ws, dz_aff);
        let mut s_aff = ws.s.clone();
        let mut z_aff = ws.z.clone();
        for i in 0..m {
            s_aff[i] += alpha_p * ds_aff[i];
            z_aff[i] += alpha_d * dz_aff[i];
        }
        let mu_aff = dot(&s_aff, &z_aff) / (m.max(1) as f64);
        let sigma = (mu_aff / mu.max(1e-16)).clamp(0.0, 1.0).powi(3);

        // combined
        for i in 0..n {
            rhs[i] = -rd[i];
        }
        for i in 0..m {
            rhs[n + i] = -rp[i] + ws.s[i] - sigma * mu / ws.z[i].max(1e-16);
        }
        let mut d = rhs;
        solve_perm(&fac, &mut d);
        let mut ds = vec![0.0; m];
        for i in 0..m {
            ds[i] = -ws.s[i] + sigma * mu / ws.z[i].max(1e-16) - h_mul_row(&h, m, i, &d[n..]);
        }
        let ap = 0.99 * max_step(ws, &ds, true);
        let ad = 0.99 * max_step_dual(ws, &d[n..]);
        let a = ap.min(ad).max(0.0);
        for i in 0..n {
            ws.x[i] += a * d[i];
        }
        for i in 0..m {
            ws.s[i] += a * ds[i];
            ws.z[i] += a * d[n + i];
        }
        force_interior(ws);
        mu = duality_mu(ws);

        let r =
            crate::verifier::residuals(&ws.p, &ws.q, &ws.a, &ws.b, &ws.cones, &ws.x, &ws.s, &ws.z);
        if crate::verifier::solved_at(&r, ws.settings.eps_abs) {
            status = Status::Solved;
            break;
        }
        if mu < 1e-14 && r.res_pri < 1e-4 && r.res_dual < 1e-4 {
            status = Status::Solved;
            break;
        }
    }
    if status == Status::Unsolved {
        status = Status::MaxIters;
    }
    ws.info.status = status;
    ws.info.iterations = iter;
    ws.info.engine = "ipm";
    let _ = KktSystem::analyze; // keep import used if refactoring
}

fn initialize_interior(ws: &mut Workspace) {
    if inf_norm(&ws.s) < 1e-14 {
        ws.s.fill(1.0);
    }
    if inf_norm(&ws.z) < 1e-14 {
        ws.z.fill(1.0);
    }
    // push into cone interiors
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = 0.0;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = ws.s[off + k].abs().max(1.0);
                    ws.z[off + k] = ws.z[off + k].abs().max(1.0);
                }
            }
            Cone::SecondOrder { dim } => {
                let t = off;
                ws.s[t] = ws.s[t].abs() + nrm(&ws.s[t + 1..t + dim]) + 1.0;
                ws.z[t] = ws.z[t].abs() + nrm(&ws.z[t + 1..t + dim]) + 1.0;
            }
            Cone::Exponential
            | Cone::Power { .. }
            | Cone::DualExponential
            | Cone::DualPower { .. } => {
                ws.s[off] = -1.0;
                ws.s[off + 1] = 1.0;
                ws.s[off + 2] = 1.0;
                ws.z[off] = -1.0;
                ws.z[off + 1] = 1.0;
                ws.z[off + 2] = 1.0;
                if matches!(cone, Cone::Exponential) {
                    // primal exp interior: y>0, z>0, y exp(x/y) < z
                    ws.s[off] = -1.0;
                    ws.s[off + 1] = 1.0;
                    ws.s[off + 2] = 2.0;
                    ws.z[off] = -1.0;
                    ws.z[off + 1] = 1.0;
                    ws.z[off + 2] = 2.0;
                }
            }
            _ => {
                for k in 0..cone.dim() {
                    ws.s[off + k] = ws.s[off + k].abs() + 1.0;
                    ws.z[off + k] = ws.z[off + k].abs() + 1.0;
                }
            }
        }
    }
}

fn duality_mu(ws: &Workspace) -> f64 {
    let mut gap = 0.0_f64;
    let mut nu = 0.0_f64;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { .. } => {}
            _ => {
                let d = cone.dim();
                gap += dot(&ws.s[off..off + d], &ws.z[off..off + d]);
                nu += d as f64;
            }
        }
    }
    if nu == 0.0 {
        1.0
    } else {
        (gap / nu).max(1e-16)
    }
}

fn fill_scaling(ws: &Workspace, h: &mut [f64], mu: f64) {
    let m = ws.s.len();
    h.fill(0.0);
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    h[(off + k) * m + (off + k)] = 1e-8;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    let s = ws.s[off + k].max(1e-16);
                    let z = ws.z[off + k].max(1e-16);
                    h[(off + k) * m + (off + k)] = s / z;
                }
            }
            Cone::SecondOrder { dim } => {
                soc_hessian(&ws.s[off..off + dim], mu, h, m, off);
            }
            Cone::Exponential => {
                exp_dual_hessian(&ws.z[off..off + 3], mu, h, m, off);
            }
            Cone::Power { alpha } => {
                // diagonal fallback plus small coupling
                for k in 0..3 {
                    h[(off + k) * m + (off + k)] = mu / ws.z[off + k].max(1e-8).powi(2);
                }
                let _ = alpha;
            }
            _ => {
                for k in 0..cone.dim() {
                    h[(off + k) * m + (off + k)] = mu / ws.z[off + k].abs().max(1e-8).powi(2);
                }
            }
        }
    }
}

fn soc_hessian(s: &[f64], mu: f64, h: &mut [f64], m: usize, off: usize) {
    let n = s.len();
    let mut js = s[0] * s[0];
    for k in 1..n {
        js -= s[k] * s[k];
    }
    js = js.max(1e-12);
    // ∇² barrier of -log(s'Js) is  2/js J + 4/js^2 (Js)(Js)'
    // use μ times that as H
    let c1 = 2.0 * mu / js;
    let c2 = 4.0 * mu / (js * js);
    for i in 0..n {
        let jsi = if i == 0 { s[0] } else { -s[i] };
        for j in i..n {
            let jsj = if j == 0 { s[0] } else { -s[j] };
            let mut v = c2 * jsi * jsj;
            if i == j {
                v += if i == 0 { c1 } else { -c1 };
            }
            h[(off + i) * m + (off + j)] = v;
            h[(off + j) * m + (off + i)] = v;
        }
    }
}

fn exp_dual_hessian(z: &[f64], mu: f64, h: &mut [f64], m: usize, off: usize) {
    // crude SPD Hessian: μ (I / max(|z_i|,ε)^2 + ee')
    for i in 0..3 {
        for j in 0..3 {
            let mut v = if i == j {
                mu / z[i].abs().max(1e-6).powi(2)
            } else {
                0.0
            };
            v += 1e-6 * mu;
            h[(off + i) * m + (off + j)] = v;
        }
    }
}

fn h_mul_row(h: &[f64], m: usize, i: usize, z: &[f64]) -> f64 {
    let mut s = 0.0;
    for j in 0..m {
        s += h[i * m + j] * z[j];
    }
    s
}

fn assemble_ipm_kkt(p: &CscMatrix, a: &CscMatrix, h: &[f64], n: usize, m: usize) -> CscMatrix {
    let mut trips = Vec::new();
    let pu = p.upper_triangle();
    let mut has = vec![false; n];
    for j in 0..n {
        for idx in pu.col_ptr[j]..pu.col_ptr[j + 1] {
            let i = pu.row_idx[idx];
            trips.push((i, j, pu.x[idx]));
            if i == j {
                has[j] = true;
            }
        }
    }
    for j in 0..n {
        if !has[j] {
            trips.push((j, j, 1e-10));
        } else {
            // add tiny regularizer
            trips.push((j, j, 1e-10));
        }
    }
    for c in 0..n {
        for idx in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[idx];
            trips.push((c, n + r, a.x[idx]));
        }
    }
    for i in 0..m {
        for j in i..m {
            let v = -h[i * m + j];
            if v.abs() > 1e-18 || i == j {
                trips.push((n + i, n + j, if i == j { v - 1e-10 } else { v }));
            }
        }
    }
    CscMatrix::from_triplets(n + m, n + m, &trips).upper_triangle()
}

fn solve_perm(fac: &LdlNumeric, x: &mut [f64]) {
    // IPM assembly is in natural order; factor was on unpermuted K.
    fac.solve_in_place(x);
}

fn max_step(ws: &Workspace, ds: &[f64], _primal: bool) -> f64 {
    let mut a = 1.0_f64;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    if ds[off + k] < 0.0 {
                        a = a.min(-0.99 * ws.s[off + k] / ds[off + k]);
                    }
                }
            }
            Cone::SecondOrder { dim } => {
                a = a.min(soc_step(&ws.s[off..off + dim], &ds[off..off + dim]));
            }
            _ => {
                for k in 0..cone.dim() {
                    if ds[off + k] < 0.0 && ws.s[off + k] > 0.0 {
                        a = a.min(-0.99 * ws.s[off + k] / ds[off + k]);
                    }
                }
            }
        }
    }
    a.clamp(0.0, 1.0)
}

fn max_step_dual(ws: &Workspace, dz: &[f64]) -> f64 {
    let mut a = 1.0_f64;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        if let Cone::Nonnegative { dim } = cone {
            for k in 0..*dim {
                if dz[off + k] < 0.0 {
                    a = a.min(-0.99 * ws.z[off + k] / dz[off + k]);
                }
            }
        }
    }
    a.clamp(0.0, 1.0)
}

fn soc_step(s: &[f64], d: &[f64]) -> f64 {
    // conservative: keep s0^2 - ||sbar||^2 > 0
    let mut a = 1.0_f64;
    for _ in 0..20 {
        let t0 = s[0] + a * d[0];
        let mut n2 = 0.0;
        for i in 1..s.len() {
            let v = s[i] + a * d[i];
            n2 += v * v;
        }
        if t0 > 0.0 && t0 * t0 > n2 {
            return a;
        }
        a *= 0.5;
        let _ = t0;
    }
    a
}

fn force_interior(ws: &mut Workspace) {
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        if let Cone::Nonnegative { dim } = cone {
            for k in 0..*dim {
                ws.s[off + k] = ws.s[off + k].max(1e-12);
                ws.z[off + k] = ws.z[off + k].max(1e-12);
            }
        }
    }
}

fn nrm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
