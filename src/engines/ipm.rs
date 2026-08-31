//! Homogeneous primal-dual interior-point fallback.
//!
//! Polyhedral cones (zero + nonnegative) use Nesterov–Todd scaling
//! \(H=\mathrm{diag}(s./z)\), which is the ADMM KKT with \(\rho_i=z_i/s_i\).
//! The cached AMD order and symbolic factor are reused; only the numeric
//! diagonals change. After the run, \(\rho\) and \(\sigma\) are restored so a
//! later ADMM step still matches the sequential contract.
//!
//! Non-polyhedral cones keep a dense Hessian Newton matrix (SOC / exp).

use crate::algebra::csc::CscMatrix;
use crate::algebra::ldl::{LdlNumeric, LdlSymbolic};
use crate::algebra::{dot, inf_norm};
use crate::cones::Cone;
use crate::status::Status;
use crate::workspace::Workspace;

pub fn run(ws: &mut Workspace) {
    if ws.cones.is_polyhedral() {
        run_polyhedral(ws);
    } else {
        run_dense(ws);
    }
}

fn run_polyhedral(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    let rho_saved = ws.rho.clone();
    let sigma_saved = ws.kkt.sigma;
    interiorize_polyhedral(ws);

    let max_iter = ws.settings.ipm_max_iter.min(ws.settings.max_iter).max(1);
    let mut n_tiny = 0usize;
    let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
    let sigma_ipm = 1e-10_f64;
    let ir = ws.settings.iterative_refinement.max(2);
    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let mut rd = vec![0.0; n];
    let mut rp = vec![0.0; m];
    let mut atz = vec![0.0; n];
    let mut rhs = vec![0.0; n + m];
    let mut d_aff = vec![0.0; n + m];
    let mut d = vec![0.0; n + m];
    let mut ds_aff = vec![0.0; m];
    let mut ds = vec![0.0; m];
    let mut nn_idx = Vec::new();
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        if let Cone::Nonnegative { dim } = cone {
            for k in 0..*dim {
                nn_idx.push(off + k);
            }
        }
    }
    let nu = nn_idx.len().max(1) as f64;

    while iter < max_iter {
        iter += 1;
        set_nt_rho(ws);
        if ws.kkt.update_nt(sigma_ipm, &ws.rho).is_err() {
            status = Status::Indeterminate;
            break;
        }
        ws.factorizations += 1;

        fill_rd_rp(ws, &mut rd, &mut rp, &mut atz);
        // affine: rhs_comp = -s◦z  →  A dx - H dz = -rp + s  (zero rows: s=0)
        for i in 0..n {
            rhs[i] = -rd[i];
        }
        for i in 0..m {
            rhs[n + i] = -rp[i] + ws.s[i];
        }
        d_aff.copy_from_slice(&rhs);
        ws.kkt.solve(&rhs, &mut d_aff, ir);
        slack_step(ws, &d_aff[n..], 0.0, None, &mut ds_aff);

        let (ap0, ad0) = frac_to_bound(ws, &ds_aff, &d_aff[n..]);
        let mu = duality_mu_nn(ws, &nn_idx);
        let mut mu_aff = 0.0_f64;
        for &i in &nn_idx {
            mu_aff += (ws.s[i] + ap0 * ds_aff[i]) * (ws.z[i] + ad0 * d_aff[n + i]);
        }
        mu_aff = (mu_aff / nu).max(0.0);
        let sigma = (mu_aff / mu.max(1e-16)).clamp(0.0, 1.0).powi(3);

        for i in 0..n {
            rhs[i] = -rd[i];
        }
        for i in 0..m {
            rhs[n + i] = -rp[i] + ws.s[i];
        }
        for &i in &nn_idx {
            let zi = ws.z[i].max(1e-16);
            rhs[n + i] += -sigma * mu / zi + ds_aff[i] * d_aff[n + i] / zi;
        }
        ws.kkt.solve(&rhs, &mut d, ir);
        slack_step(
            ws,
            &d[n..],
            sigma * mu,
            Some((&ds_aff, &d_aff[n..])),
            &mut ds,
        );

        let (ap, ad) = frac_to_bound(ws, &ds, &d[n..]);
        let a = 0.99_f64 * ap.min(ad);
        if a < 1e-12 {
            n_tiny += 1;
            if n_tiny > 4 {
                break;
            }
            recenter_polyhedral(ws, 1.0);
            continue;
        }
        n_tiny = 0;
        for i in 0..n {
            ws.x[i] += a * d[i];
        }
        for i in 0..m {
            ws.s[i] += a * ds[i];
            ws.z[i] += a * d[n + i];
        }
        force_polyhedral_interior(ws);

        let r = ws.original_residuals();
        if crate::verifier::solved_at(&r, eps) {
            status = Status::Solved;
            break;
        }
        if duality_mu_nn(ws, &nn_idx) < 1e-14 && r.res_pri < eps && r.res_dual < eps {
            status = Status::Solved;
            break;
        }
    }

    let _ = ws.kkt.update_nt(sigma_saved, &rho_saved);
    ws.factorizations += 1;
    ws.rho = rho_saved;
    ws.sync_w();
    if status == Status::Unsolved {
        status = Status::MaxIters;
    }
    ws.info.status = status;
    ws.info.iterations = iter;
    ws.info.engine = "ipm";
}

fn interiorize_polyhedral(ws: &mut Workspace) {
    let cold = inf_norm(&ws.x) + inf_norm(&ws.s) + inf_norm(&ws.z) < 1e-12;
    let floor = if cold {
        1.0_f64
    } else {
        // μ must not be orders of magnitude below the residual, or the
        // affine step immediately hits the cone boundary (α ≈ 0).
        let mut ax = vec![0.0; ws.s.len()];
        ws.a.mul(&ws.x, &mut ax);
        let mut rp = 0.0_f64;
        for i in 0..ax.len() {
            rp = rp.max((ax[i] + ws.s[i] - ws.b[i]).abs());
        }
        let mut px = vec![0.0; ws.x.len()];
        ws.p.sym_mul_add(&ws.x, &mut px, 1.0);
        let mut atz = vec![0.0; ws.x.len()];
        ws.a.tmul(&ws.z, &mut atz);
        let mut rd = 0.0_f64;
        for i in 0..px.len() {
            rd = rd.max((px[i] + atz[i] + ws.q[i]).abs());
        }
        (rp.max(rd)).clamp(1e-8, 1.0)
    };
    recenter_polyhedral(ws, floor);
}

fn recenter_polyhedral(ws: &mut Workspace, floor: f64) {
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = 0.0;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = ws.s[off + k].abs().max(floor);
                    ws.z[off + k] = ws.z[off + k].abs().max(floor);
                }
            }
            _ => {}
        }
    }
}

fn set_nt_rho(ws: &mut Workspace) {
    let rho_eq = (ws.settings.rho * 1e6).clamp(1e4, 1e8);
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    ws.rho[off + k] = rho_eq;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    ws.rho[off + k] = (ws.z[off + k] / ws.s[off + k]).clamp(1e-8, 1e8);
                }
            }
            _ => {}
        }
    }
}

fn fill_rd_rp(ws: &Workspace, rd: &mut [f64], rp: &mut [f64], atz: &mut [f64]) {
    rd.fill(0.0);
    ws.p.sym_mul_add(&ws.x, rd, 1.0);
    ws.a.tmul(&ws.z, atz);
    for i in 0..rd.len() {
        rd[i] += atz[i] + ws.q[i];
    }
    ws.a.mul(&ws.x, rp);
    for i in 0..rp.len() {
        rp[i] += ws.s[i] - ws.b[i];
    }
}

fn slack_step(
    ws: &Workspace,
    dz: &[f64],
    sigma_mu: f64,
    mehrotra: Option<(&[f64], &[f64])>,
    ds: &mut [f64],
) {
    ds.fill(0.0);
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        if let Cone::Nonnegative { dim } = cone {
            for k in 0..*dim {
                let i = off + k;
                let zi = ws.z[i].max(1e-16);
                let mut rhs = -ws.s[i] + sigma_mu / zi;
                if let Some((dsa, dza)) = mehrotra {
                    rhs -= dsa[i] * dza[i] / zi;
                }
                ds[i] = rhs - dz[i] / ws.rho[i];
            }
        }
    }
}

fn frac_to_bound(ws: &Workspace, ds: &[f64], dz: &[f64]) -> (f64, f64) {
    let mut ap = 1.0_f64;
    let mut ad = 1.0_f64;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        if let Cone::Nonnegative { dim } = cone {
            for k in 0..*dim {
                let i = off + k;
                if ds[i] < 0.0 {
                    ap = ap.min(-ws.s[i] / ds[i]);
                }
                if dz[i] < 0.0 {
                    ad = ad.min(-ws.z[i] / dz[i]);
                }
            }
        }
    }
    (ap.clamp(0.0, 1.0), ad.clamp(0.0, 1.0))
}

fn duality_mu_nn(ws: &Workspace, nn_idx: &[usize]) -> f64 {
    if nn_idx.is_empty() {
        return 1e-16;
    }
    let mut g = 0.0_f64;
    for &i in nn_idx {
        g += ws.s[i] * ws.z[i];
    }
    (g / nn_idx.len() as f64).max(1e-16)
}

fn force_polyhedral_interior(ws: &mut Workspace) {
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = 0.0;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = ws.s[off + k].max(1e-12);
                    ws.z[off + k] = ws.z[off + k].max(1e-12);
                }
            }
            _ => {}
        }
    }
}

fn run_dense(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    initialize_interior(ws);
    let mut mu = duality_mu(ws);
    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let mut h = vec![0.0; m * m];
    let mut sym: Option<LdlSymbolic> = None;
    let max_iter = ws.settings.ipm_max_iter.min(ws.settings.max_iter).max(1);

    while iter < max_iter {
        iter += 1;
        fill_scaling(ws, &mut h, mu);
        let k_csc = assemble_ipm_kkt(&ws.p, &ws.a, &h, n, m);
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

        let mut rhs = vec![0.0; n + m];
        for i in 0..n {
            rhs[i] = -rd[i];
        }
        for i in 0..m {
            rhs[n + i] = -rp[i] + ws.s[i];
        }
        let mut d_aff = rhs.clone();
        solve_perm(&fac, &mut d_aff);
        let dz_aff = &d_aff[n..];
        let mut ds_aff = vec![0.0; m];
        for i in 0..m {
            ds_aff[i] = -ws.s[i] - h_mul_row(&h, m, i, dz_aff);
        }
        let alpha_p = max_step(ws, &ds_aff);
        let alpha_d = max_step_dual(ws, dz_aff);
        let mut s_aff = ws.s.clone();
        let mut z_aff = ws.z.clone();
        for i in 0..m {
            s_aff[i] += alpha_p * ds_aff[i];
            z_aff[i] += alpha_d * dz_aff[i];
        }
        let mu_aff = dot(&s_aff, &z_aff) / (m.max(1) as f64);
        let sigma = (mu_aff / mu.max(1e-16)).clamp(0.0, 1.0).powi(3);

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
        let ap = 0.99 * max_step(ws, &ds);
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
}

fn initialize_interior(ws: &mut Workspace) {
    if inf_norm(&ws.s) < 1e-14 {
        ws.s.fill(1.0);
    }
    if inf_norm(&ws.z) < 1e-14 {
        ws.z.fill(1.0);
    }
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
                ws.s[off + 2] = 2.0;
                ws.z[off] = -1.0;
                ws.z[off + 1] = 1.0;
                ws.z[off + 2] = 2.0;
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
        trips.push((j, j, 1e-10));
        let _ = has[j];
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
    fac.solve_in_place(x);
}

fn max_step(ws: &Workspace, ds: &[f64]) -> f64 {
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
