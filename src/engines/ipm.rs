//! Homogeneous primal-dual interior-point fallback (Clarabel embedding).
//!
//! Newton linearization of the Andersen–Ye \((\tau,\kappa)\) QCP embedding
//! reduces to one factorization of the sparse cone-block KKT
//! \(K=[P, A'; A, -H_s]\) and a 2×2 reduction for \(\Delta\tau\). Symmetric
//! cones use Nesterov–Todd \(H_s\); exponential/power use Clarabel \(H_s\)
//! and \(\Delta s + H_s\Delta z = -(s+\sigma\mu\nabla f^\ast(z))\).
//! Polyhedral NT is the diagonal special case \(H=\mathrm{diag}(s./z)\).
//! AMD order and the symbolic factor persist across Newton steps and R1.

use crate::algebra::{dot, inf_norm, CscMatrix};
use crate::cones::Cone;
use crate::ipm_kkt::IpmKkt;
use crate::status::Status;
use crate::workspace::Workspace;

pub fn run(ws: &mut Workspace) {
    run_homog(ws);
}

fn run_homog(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    if !ensure_kkt(ws) {
        ws.info.status = Status::Indeterminate;
        ws.info.iterations = 0;
        ws.info.engine = "ipm";
        return;
    }
    unit_initialize(ws);
    let mut tau = 1.0_f64;
    let mut kappa = 1.0_f64;

    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let mut primal_dual = true;
    let max_iter = ws.settings.ipm_max_iter.min(ws.settings.max_iter).max(1);
    let ir = ws.settings.iterative_refinement.max(1);
    let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
    let eps_inf = ws.settings.eps_infeas;
    let mut n_tiny = 0usize;

    let mut packed = vec![0.0; packed_len(&ws.cones)];
    let mut c = vec![0.0; m];
    let mut ds = vec![0.0; m];
    let mut ds_aff = vec![0.0; m];
    let mut rx = vec![0.0; n];
    let mut rz = vec![0.0; m];
    let mut px = vec![0.0; n];
    let mut atz = vec![0.0; n];
    let mut ax = vec![0.0; m];
    let mut x1 = vec![0.0; n];
    let mut z1 = vec![0.0; m];
    let mut x2 = vec![0.0; n];
    let mut z2 = vec![0.0; m];
    let mut dx = vec![0.0; n];
    let mut dz = vec![0.0; m];
    let mut dx_aff = vec![0.0; n];
    let mut dz_aff = vec![0.0; m];
    let mut rhs_x = vec![0.0; n];
    let mut rhs_z = vec![0.0; m];
    let mut xi = vec![0.0; n];
    let mut hs_dz = vec![0.0; m];
    let mut ptmp = vec![0.0; n];

    while iter < max_iter {
        iter += 1;
        let (rtau, mu) =
            homog_residuals(ws, tau, kappa, &mut px, &mut atz, &mut ax, &mut rx, &mut rz);

        if let Some(st) = homog_status(ws, tau, kappa, eps, eps_inf) {
            status = st;
            break;
        }

        if !pack_hs(ws, &mut packed, mu, primal_dual) {
            unit_initialize(ws);
            tau = 1.0;
            kappa = 1.0;
            primal_dual = false;
            continue;
        }
        {
            let kkt = ws.ipm_kkt.as_mut().unwrap();
            if kkt.update_hs(&packed).is_err() {
                status = Status::Indeterminate;
                break;
            }
            ws.factorizations += 1;
            kkt.solve_constant(&ws.q, &ws.b, ir);
            x2.copy_from_slice(&kkt.x2);
            z2.copy_from_slice(&kkt.z2);
        }

        affine_c(ws, &mut c);
        for i in 0..n {
            rhs_x[i] = rx[i];
        }
        for i in 0..m {
            rhs_z[i] = c[i] - rz[i];
        }
        {
            let kkt = ws.ipm_kkt.as_mut().unwrap();
            kkt.solve_split(&rhs_x, &rhs_z, &mut x1, &mut z1, ir);
        }
        let rhs_kappa_aff = tau * kappa;
        let (dtau_aff, dkappa_aff) = finish_direction(
            ws,
            &x1,
            &z1,
            &x2,
            &z2,
            &c,
            tau,
            kappa,
            rtau,
            rhs_kappa_aff,
            &mut xi,
            &mut dx_aff,
            &mut dz_aff,
            &mut ds_aff,
            &mut hs_dz,
            &mut ptmp,
        );
        if !dtau_aff.is_finite() {
            unit_initialize(ws);
            tau = 1.0;
            kappa = 1.0;
            primal_dual = false;
            continue;
        }

        let ap_aff = max_step(ws, &ds_aff);
        let ad_aff = max_step_dual(ws, &dz_aff);
        let at_aff = scalar_step(tau, dtau_aff).min(scalar_step(kappa, dkappa_aff));
        let alpha_aff = ap_aff.min(ad_aff).min(at_aff).clamp(0.0, 1.0);
        let sigma = (1.0 - alpha_aff).clamp(0.0, 1.0).powi(3);
        let m_corr = if iter > 1 { 1.0 } else { alpha_aff };

        combined_c(ws, sigma, mu, &ds_aff, &dz_aff, &mut c);
        let om = 1.0 - sigma;
        for i in 0..n {
            rhs_x[i] = om * rx[i];
        }
        for i in 0..m {
            rhs_z[i] = c[i] - om * rz[i];
        }
        let rhs_tau = om * rtau;
        let rhs_kappa = -sigma * mu + m_corr * dtau_aff * dkappa_aff + tau * kappa;
        {
            let kkt = ws.ipm_kkt.as_mut().unwrap();
            kkt.solve_split(&rhs_x, &rhs_z, &mut x1, &mut z1, ir);
        }
        let (dtau, dkappa) = finish_direction(
            ws, &x1, &z1, &x2, &z2, &c, tau, kappa, rhs_tau, rhs_kappa, &mut xi, &mut dx, &mut dz,
            &mut ds, &mut hs_dz, &mut ptmp,
        );
        if !dtau.is_finite() {
            n_tiny += 1;
            if n_tiny > 4 {
                break;
            }
            unit_initialize(ws);
            tau = 1.0;
            kappa = 1.0;
            primal_dual = false;
            continue;
        }

        let ap = max_step(ws, &ds);
        let ad = max_step_dual(ws, &dz);
        let at = scalar_step(tau, dtau).min(scalar_step(kappa, dkappa));
        let a = 0.99_f64 * ap.min(ad).min(at).max(0.0);
        if primal_dual && a < 1e-2 {
            primal_dual = false;
            iter = iter.saturating_sub(1);
            continue;
        }
        if ws.settings.verbose {
            eprintln!(
                "ipm {iter} a={a:.3e} mu={mu:.3e} tau={tau:.3e} kap={kappa:.3e} rp={:.3e} rd={:.3e}",
                inf_norm(&rz),
                inf_norm(&rx)
            );
        }
        if a < 1e-12 {
            n_tiny += 1;
            if n_tiny > 4 {
                break;
            }
            unit_initialize(ws);
            tau = 1.0;
            kappa = 1.0;
            primal_dual = false;
            continue;
        }
        n_tiny = 0;
        for i in 0..n {
            ws.x[i] += a * dx[i];
        }
        for i in 0..m {
            ws.s[i] += a * ds[i];
            ws.z[i] += a * dz[i];
        }
        tau += a * dtau;
        kappa += a * dkappa;
        if tau < 1e-16 {
            tau = 1e-16;
        }
        if kappa < 1e-16 {
            kappa = 1e-16;
        }
        snap_zero_cone(ws);
        rescale(ws, &mut tau, &mut kappa);
    }

    match status {
        Status::PrimalInfeasible | Status::DualInfeasible => {
            let inv = 1.0 / kappa.max(1e-16);
            for xi in ws.x.iter_mut() {
                *xi *= inv;
            }
            for si in ws.s.iter_mut() {
                *si *= inv;
            }
            for zi in ws.z.iter_mut() {
                *zi *= inv;
            }
        }
        _ => {
            let inv = 1.0 / tau.max(1e-16);
            for xi in ws.x.iter_mut() {
                *xi *= inv;
            }
            for si in ws.s.iter_mut() {
                *si *= inv;
            }
            for zi in ws.z.iter_mut() {
                *zi *= inv;
            }
            let r = ws.original_residuals();
            if crate::verifier::solved_at(&r, eps) {
                status = Status::Solved;
            }
        }
    }

    if status == Status::Unsolved {
        status = Status::MaxIters;
    }
    ws.info.status = status;
    ws.info.iterations = iter;
    ws.info.engine = "ipm";
    ws.sync_w();
}

fn ensure_kkt(ws: &mut Workspace) -> bool {
    let n = ws.x.len();
    let m = ws.s.len();
    let reuse = ws.ipm_kkt.as_ref().is_some_and(|k| k.n == n && k.m == m);
    if reuse {
        return true;
    }
    match IpmKkt::analyze(&ws.p, &ws.a, &ws.cones) {
        Ok(k) => {
            ws.ipm_kkt = Some(k);
            ws.factorizations += 1;
            true
        }
        Err(_) => {
            ws.ipm_kkt = None;
            false
        }
    }
}

fn packed_len(cones: &crate::cones::CompositeCone) -> usize {
    cones.cones.iter().map(|c| c.hs_packed_len()).sum()
}

fn homog_residuals(
    ws: &Workspace,
    tau: f64,
    kappa: f64,
    px: &mut [f64],
    atz: &mut [f64],
    ax: &mut [f64],
    rx: &mut [f64],
    rz: &mut [f64],
) -> (f64, f64) {
    px.fill(0.0);
    ws.p.sym_mul_add(&ws.x, px, 1.0);
    ws.a.tmul(&ws.z, atz);
    for i in 0..rx.len() {
        rx[i] = -px[i] - atz[i] - ws.q[i] * tau;
    }
    ws.a.mul(&ws.x, ax);
    for i in 0..rz.len() {
        rz[i] = ax[i] + ws.s[i] - ws.b[i] * tau;
    }
    let xpx = dot(&ws.x, px);
    let rtau = dot(&ws.q, &ws.x) + dot(&ws.b, &ws.z) + kappa + xpx / tau.max(1e-16);
    let nu = (ws.cones.barrier_degree() + 1) as f64;
    let mu = (dot(&ws.s, &ws.z) + tau * kappa) / nu.max(1.0);
    (rtau, mu.max(1e-16))
}

fn homog_status(ws: &Workspace, tau: f64, kappa: f64, eps: f64, eps_inf: f64) -> Option<Status> {
    let ktr = kappa / tau.max(1e-16);
    if ktr <= 1.0 {
        let inv = 1.0 / tau.max(1e-16);
        let mut x = ws.x.clone();
        let mut s = ws.s.clone();
        let mut z = ws.z.clone();
        for v in x.iter_mut() {
            *v *= inv;
        }
        for v in s.iter_mut() {
            *v *= inv;
        }
        for v in z.iter_mut() {
            *v *= inv;
        }
        crate::scale::unscale_solution(&ws.eq, &mut x, &mut s, &mut z);
        let r = crate::verifier::residuals(
            &ws.orig.p,
            &ws.orig.q,
            &ws.orig.a,
            &ws.orig.b,
            &ws.orig.cones,
            &x,
            &s,
            &z,
        );
        if crate::verifier::solved_at(&r, eps) {
            return Some(Status::Solved);
        }
    }
    if ktr > 1e4 {
        let inv = 1.0 / kappa.max(1e-16);
        let mut x = ws.x.clone();
        let mut s = ws.s.clone();
        let mut z = ws.z.clone();
        for v in x.iter_mut() {
            *v *= inv;
        }
        for v in s.iter_mut() {
            *v *= inv;
        }
        for v in z.iter_mut() {
            *v *= inv;
        }
        crate::scale::unscale_solution(&ws.eq, &mut x, &mut s, &mut z);
        if crate::verifier::check_primal_ray(&ws.orig.a, &ws.orig.b, &ws.orig.cones, &z, eps_inf) {
            return Some(Status::PrimalInfeasible);
        }
        if crate::verifier::check_dual_ray(
            &ws.orig.p,
            &ws.orig.q,
            &ws.orig.a,
            &ws.orig.cones,
            &x,
            eps_inf,
        ) {
            return Some(Status::DualInfeasible);
        }
    }
    None
}

fn pquad(p: &CscMatrix, x: &[f64], y: &[f64], tmp: &mut [f64]) -> f64 {
    tmp.fill(0.0);
    p.sym_mul_add(y, tmp, 1.0);
    dot(x, tmp)
}

fn finish_direction(
    ws: &Workspace,
    x1: &[f64],
    z1: &[f64],
    x2: &[f64],
    z2: &[f64],
    c: &[f64],
    tau: f64,
    kappa: f64,
    rhs_tau: f64,
    rhs_kappa: f64,
    xi: &mut [f64],
    dx: &mut [f64],
    dz: &mut [f64],
    ds: &mut [f64],
    hs_dz: &mut [f64],
    ptmp: &mut [f64],
) -> (f64, f64) {
    let n = ws.x.len();
    let tinv = 1.0 / tau.max(1e-16);
    for i in 0..n {
        xi[i] = ws.x[i] * tinv;
    }
    let q_x1 = dot(&ws.q, x1);
    let b_z1 = dot(&ws.b, z1);
    let q_x2 = dot(&ws.q, x2);
    let b_z2 = dot(&ws.b, z2);
    let two_xi_p_x1 = 2.0 * pquad(&ws.p, xi, x1, ptmp);
    let tau_num = rhs_tau - rhs_kappa * tinv + q_x1 + b_z1 + two_xi_p_x1;
    for i in 0..n {
        xi[i] -= x2[i];
    }
    let q_xi = pquad(&ws.p, xi, xi, ptmp);
    let q_x2x2 = pquad(&ws.p, x2, x2, ptmp);
    let tau_den = kappa * tinv - q_x2 - b_z2 + q_xi - q_x2x2;
    if tau_den.abs() < 1e-30 {
        return (f64::NAN, f64::NAN);
    }
    let dtau = tau_num / tau_den;
    let dkappa = -(rhs_kappa + kappa * dtau) / tau.max(1e-16);
    for i in 0..n {
        dx[i] = x1[i] + dtau * x2[i];
    }
    for i in 0..ws.z.len() {
        dz[i] = z1[i] + dtau * z2[i];
    }
    ws.ipm_kkt.as_ref().unwrap().mul_hs(dz, hs_dz);
    for i in 0..ws.s.len() {
        ds[i] = -c[i] - hs_dz[i];
    }
    (dtau, dkappa)
}

fn scalar_step(v: f64, dv: f64) -> f64 {
    if dv < 0.0 {
        (-v / dv).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn rescale(ws: &mut Workspace, tau: &mut f64, kappa: &mut f64) {
    let scale = tau.max(*kappa);
    if scale > 1e4 || scale < 1e-4 {
        let inv = 1.0 / scale.max(1e-16);
        for v in ws.x.iter_mut() {
            *v *= inv;
        }
        for v in ws.s.iter_mut() {
            *v *= inv;
        }
        for v in ws.z.iter_mut() {
            *v *= inv;
        }
        *tau *= inv;
        *kappa *= inv;
    }
}

fn affine_c(ws: &Workspace, c: &mut [f64]) {
    c.fill(0.0);
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { .. } => {}
            _ => {
                let d = cone.dim();
                c[off..off + d].copy_from_slice(&ws.s[off..off + d]);
            }
        }
    }
}

fn combined_c(ws: &Workspace, sigma: f64, mu: f64, ds_aff: &[f64], dz_aff: &[f64], c: &mut [f64]) {
    affine_c(ws, c);
    let sigu = sigma * mu;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { .. } => {}
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    let i = off + k;
                    let zi = ws.z[i].max(1e-16);
                    c[i] += -sigu / zi + ds_aff[i] * dz_aff[i] / zi;
                }
            }
            Cone::Exponential | Cone::DualExponential => {
                if let Some(g) = crate::cones::exp_dual_grad(&ws.z[off..off + 3]) {
                    for k in 0..3 {
                        c[off + k] += sigu * g[k];
                    }
                }
            }
            Cone::Power { alpha } | Cone::DualPower { alpha } => {
                if let Some((g, _)) = crate::cones::power_dual_grad_h(&ws.z[off..off + 3], *alpha) {
                    for k in 0..3 {
                        c[off + k] += sigu * g[k];
                    }
                }
            }
            Cone::GenPower { alpha, n_z } => {
                let d = cone.dim();
                if let Some((g, _)) =
                    crate::cones::genpower_dual_grad_h(&ws.z[off..off + d], alpha, *n_z)
                {
                    for k in 0..d {
                        c[off + k] += sigu * g[k];
                    }
                }
            }
            Cone::SecondOrder { dim } => {
                c[off] -= sigu;
                for k in 0..*dim {
                    c[off + k] += ds_aff[off + k] * dz_aff[off + k];
                }
            }
            _ => {
                for k in 0..cone.dim() {
                    let i = off + k;
                    let zi = ws.z[i].abs().max(1e-16);
                    c[i] += -sigu / zi + ds_aff[i] * dz_aff[i] / zi;
                }
            }
        }
    }
}

fn pack_hs(ws: &Workspace, packed: &mut [f64], mu: f64, primal_dual: bool) -> bool {
    packed.fill(0.0);
    let mut po = 0usize;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for _ in 0..*dim {
                    packed[po] = 1e-12;
                    po += 1;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    let s = ws.s[off + k].max(1e-16);
                    let z = ws.z[off + k].max(1e-16);
                    packed[po] = s / z;
                    po += 1;
                }
            }
            Cone::GenPower { alpha, n_z } => {
                let d = cone.dim();
                let Some((_, hd)) =
                    crate::cones::genpower_dual_grad_h(&ws.z[off..off + d], alpha, *n_z)
                else {
                    return false;
                };
                let hs: Vec<f64> = hd.iter().map(|v| mu * v).collect();
                po = pack_dense(packed, po, d, &hs);
            }
            Cone::SecondOrder { dim } => {
                if let Some(hs) =
                    crate::cones::soc_nt_hessian(&ws.s[off..off + dim], &ws.z[off..off + dim])
                {
                    po = pack_dense(packed, po, *dim, &hs);
                } else {
                    po = pack_dense_diag(packed, po, *dim, 1.0);
                }
            }
            Cone::Exponential | Cone::DualExponential => {
                let Some(hs) =
                    crate::cones::exp_hs(&ws.s[off..off + 3], &ws.z[off..off + 3], mu, primal_dual)
                else {
                    return false;
                };
                po = pack_dense(packed, po, 3, &hs);
            }
            Cone::Power { alpha } | Cone::DualPower { alpha } => {
                let Some((_, hd)) = crate::cones::power_dual_grad_h(&ws.z[off..off + 3], *alpha)
                else {
                    return false;
                };
                let mut hs = [0.0; 9];
                for i in 0..9 {
                    hs[i] = mu * hd[i];
                }
                po = pack_dense(packed, po, 3, &hs);
            }
            Cone::PsdTriangle { side } => {
                let d = side * (side + 1) / 2;
                po = pack_dense_diag(packed, po, d, mu.max(1e-8));
            }
        }
    }
    true
}

fn pack_dense(packed: &mut [f64], mut po: usize, dim: usize, h: &[f64]) -> usize {
    for j in 0..dim {
        for i in 0..=j {
            packed[po] = h[i * dim + j];
            po += 1;
        }
    }
    po
}

fn pack_dense_diag(packed: &mut [f64], mut po: usize, dim: usize, v: f64) -> usize {
    for j in 0..dim {
        for i in 0..=j {
            packed[po] = if i == j { v } else { 0.0 };
            po += 1;
        }
    }
    po
}

fn unit_initialize(ws: &mut Workspace) {
    ws.x.fill(0.0);
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = 0.0;
                    ws.z[off + k] = 0.0;
                }
            }
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = 1.0;
                    ws.z[off + k] = 1.0;
                }
            }
            Cone::SecondOrder { dim } => {
                for k in 0..*dim {
                    ws.s[off + k] = 0.0;
                    ws.z[off + k] = 0.0;
                }
                ws.s[off] = 1.0;
                ws.z[off] = 1.0;
            }
            Cone::Exponential | Cone::DualExponential => {
                let u = crate::cones::exp_unit_point();
                ws.s[off..off + 3].copy_from_slice(&u);
                ws.z[off..off + 3].copy_from_slice(&u);
            }
            Cone::Power { alpha } | Cone::DualPower { alpha } => {
                let u = crate::cones::power_unit_point(*alpha);
                ws.s[off..off + 3].copy_from_slice(&u);
                ws.z[off..off + 3].copy_from_slice(&u);
            }
            Cone::GenPower { alpha, n_z } => {
                let u = crate::cones::genpower_unit_point(alpha, *n_z);
                ws.s[off..off + u.len()].copy_from_slice(&u);
                ws.z[off..off + u.len()].copy_from_slice(&u);
            }
            _ => {
                for k in 0..cone.dim() {
                    ws.s[off + k] = 1.0;
                    ws.z[off + k] = 1.0;
                }
            }
        }
    }
}

fn snap_zero_cone(ws: &mut Workspace) {
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        if let Cone::Zero { dim } = cone {
            for k in 0..*dim {
                ws.s[off + k] = 0.0;
            }
        }
    }
}

fn max_step(ws: &Workspace, ds: &[f64]) -> f64 {
    let mut a = 1.0_f64;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    if ds[off + k] < 0.0 {
                        a = a.min(-ws.s[off + k] / ds[off + k]);
                    }
                }
            }
            Cone::SecondOrder { dim } => {
                a = a.min(soc_step(&ws.s[off..off + dim], &ds[off..off + dim]));
            }
            Cone::Exponential => {
                a = a.min(crate::cones::exp_backtrack(
                    &ws.s[off..off + 3],
                    &ds[off..off + 3],
                    true,
                ));
            }
            Cone::DualExponential => {
                a = a.min(crate::cones::exp_backtrack(
                    &ws.s[off..off + 3],
                    &ds[off..off + 3],
                    false,
                ));
            }
            Cone::Power { alpha } => {
                a = a.min(crate::cones::power_backtrack(
                    &ws.s[off..off + 3],
                    &ds[off..off + 3],
                    *alpha,
                    true,
                ));
            }
            Cone::DualPower { alpha } => {
                a = a.min(crate::cones::power_backtrack(
                    &ws.s[off..off + 3],
                    &ds[off..off + 3],
                    *alpha,
                    false,
                ));
            }
            Cone::GenPower { alpha, n_z } => {
                let d = cone.dim();
                a = a.min(crate::cones::genpower_backtrack(
                    &ws.s[off..off + d],
                    &ds[off..off + d],
                    alpha,
                    *n_z,
                    true,
                ));
            }
            _ => {
                for k in 0..cone.dim() {
                    if ds[off + k] < 0.0 && ws.s[off + k] > 0.0 {
                        a = a.min(-ws.s[off + k] / ds[off + k]);
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
        match cone {
            Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    if dz[off + k] < 0.0 {
                        a = a.min(-ws.z[off + k] / dz[off + k]);
                    }
                }
            }
            Cone::SecondOrder { dim } => {
                a = a.min(soc_step(&ws.z[off..off + dim], &dz[off..off + dim]));
            }
            Cone::Exponential => {
                a = a.min(crate::cones::exp_backtrack(
                    &ws.z[off..off + 3],
                    &dz[off..off + 3],
                    false,
                ));
            }
            Cone::DualExponential => {
                a = a.min(crate::cones::exp_backtrack(
                    &ws.z[off..off + 3],
                    &dz[off..off + 3],
                    true,
                ));
            }
            Cone::Power { alpha } => {
                a = a.min(crate::cones::power_backtrack(
                    &ws.z[off..off + 3],
                    &dz[off..off + 3],
                    *alpha,
                    false,
                ));
            }
            Cone::DualPower { alpha } => {
                a = a.min(crate::cones::power_backtrack(
                    &ws.z[off..off + 3],
                    &dz[off..off + 3],
                    *alpha,
                    true,
                ));
            }
            Cone::GenPower { alpha, n_z } => {
                let d = cone.dim();
                a = a.min(crate::cones::genpower_backtrack(
                    &ws.z[off..off + d],
                    &dz[off..off + d],
                    alpha,
                    *n_z,
                    false,
                ));
            }
            _ => {}
        }
    }
    a.clamp(0.0, 1.0)
}

fn soc_step(s: &[f64], d: &[f64]) -> f64 {
    let mut a = 1.0_f64;
    for _ in 0..24 {
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
