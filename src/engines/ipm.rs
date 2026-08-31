//! Primal-dual interior-point fallback.
//!
//! Polyhedral cones (zero + nonnegative) use Nesterov–Todd scaling
//! \(H=\mathrm{diag}(s./z)\), which is the ADMM KKT with \(\rho_i=z_i/s_i\).
//! The cached AMD order and symbolic factor are reused; only the numeric
//! diagonals change. After the run, \(\sigma\) and \(\rho\) are restored so a
//! later ADMM step still matches the sequential contract.
//!
//! Exponential / power / SOC use a dense barrier Newton system with Clarabel
//! \(H_s\) and linearized centrality \(s=-\mu\nabla f^\ast(z)\).

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
    unit_initialize(ws);
    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let mut h = vec![0.0; m * m];
    let mut kkt: Option<DenseKkt> = None;
    let mut primal_dual = true;
    let max_iter = ws.settings.ipm_max_iter.min(ws.settings.max_iter).max(1);
    let ir = ws.settings.iterative_refinement.max(1);
    let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
    let mut n_tiny = 0usize;
    let mut c = vec![0.0; m];
    let mut ds = vec![0.0; m];
    let mut ds_aff = vec![0.0; m];
    let mut rhs = vec![0.0; n + m];
    let mut d_aff = vec![0.0; n + m];
    let mut d = vec![0.0; n + m];
    let mut rd = vec![0.0; n];
    let mut rp = vec![0.0; m];
    let mut atz = vec![0.0; n];

    while iter < max_iter {
        iter += 1;
        let mu = duality_mu(ws);
        if !fill_scaling(ws, &mut h, mu, primal_dual) {
            unit_initialize(ws);
            primal_dual = false;
            continue;
        }
        let k_csc = assemble_ipm_kkt(&ws.p, &ws.a, &h, n, m);
        match kkt.as_mut() {
            Some(k) if k.upper.same_pattern(&k_csc) => {
                k.upper = k_csc;
                k.perm_mat = k.upper.permute_sym_upper(&k.perm);
                if k.refactor().is_err() {
                    status = Status::Indeterminate;
                    break;
                }
            }
            _ => match DenseKkt::new(k_csc, n) {
                Ok(k) => kkt = Some(k),
                Err(_) => {
                    status = Status::Indeterminate;
                    break;
                }
            },
        }
        let ksys = kkt.as_mut().unwrap();
        ws.factorizations += 1;

        fill_rd_rp(ws, &mut rd, &mut rp, &mut atz);
        affine_c(ws, &mut c);
        pack_rhs(&mut rhs, &rd, &rp, &c, n, m);
        ksys.solve(&rhs, &mut d_aff, ir);
        recover_ds(&mut ds_aff, &c, &h, m, &d_aff[n..]);

        let ap_aff = max_step(ws, &ds_aff);
        let ad_aff = max_step_dual(ws, &d_aff[n..]);
        let alpha_aff = ap_aff.min(ad_aff).clamp(0.0, 1.0);
        let sigma = (1.0 - alpha_aff).clamp(0.0, 1.0).powi(3);

        combined_c(ws, sigma, mu, &ds_aff, &d_aff[n..], &mut c);
        pack_rhs(&mut rhs, &rd, &rp, &c, n, m);
        ksys.solve(&rhs, &mut d, ir);
        recover_ds(&mut ds, &c, &h, m, &d[n..]);

        let ap = max_step(ws, &ds);
        let ad = max_step_dual(ws, &d[n..]);
        let a = 0.99_f64 * ap.min(ad).max(0.0);
        if primal_dual && a < 1e-2 {
            primal_dual = false;
            iter = iter.saturating_sub(1);
            continue;
        }
        if ws.settings.verbose {
            eprintln!(
                "ipm {iter} a={a:.3e} ap={ap:.3e} ad={ad:.3e} mu={mu:.3e} rp={:.3e} rd={:.3e} pd={primal_dual}",
                inf_norm(&rp),
                inf_norm(&rd)
            );
        }
        if a < 1e-12 {
            n_tiny += 1;
            if n_tiny > 4 {
                break;
            }
            unit_initialize(ws);
            primal_dual = false;
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
        snap_zero_cone(ws);

        let r = ws.original_residuals();
        if crate::verifier::solved_at(&r, eps) {
            status = Status::Solved;
            break;
        }
        if duality_mu(ws) < 1e-14 && r.res_pri < eps && r.res_dual < eps {
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
    ws.sync_w();
}

struct DenseKkt {
    n: usize,
    upper: CscMatrix,
    perm: Vec<usize>,
    perm_mat: CscMatrix,
    sym: LdlSymbolic,
    fac: LdlNumeric,
    work: Vec<f64>,
    work2: Vec<f64>,
}

impl DenseKkt {
    fn new(upper: CscMatrix, n_pos: usize) -> Result<Self, ()> {
        let perm = crate::algebra::amd::order_upper(&upper);
        let perm_mat = upper.permute_sym_upper(&perm);
        let sym = LdlSymbolic::analyze(&perm_mat).map_err(|_| ())?;
        let fac = LdlNumeric::factor_regularized(&perm_mat, &sym, n_pos, 1e-12)
            .or_else(|_| LdlNumeric::factor(&perm_mat, &sym))
            .map_err(|_| ())?;
        let dim = upper.n;
        Ok(Self {
            n: n_pos,
            upper,
            perm,
            perm_mat,
            sym,
            fac,
            work: vec![0.0; dim],
            work2: vec![0.0; dim],
        })
    }

    fn refactor(&mut self) -> Result<(), ()> {
        self.fac = LdlNumeric::factor_regularized(&self.perm_mat, &self.sym, self.n, 1e-12)
            .or_else(|_| LdlNumeric::factor(&self.perm_mat, &self.sym))
            .map_err(|_| ())?;
        Ok(())
    }

    fn solve(&mut self, rhs: &[f64], sol: &mut [f64], refinement: usize) {
        crate::algebra::permute(&self.perm, rhs, &mut self.work);
        self.fac.solve_in_place(&mut self.work);
        crate::algebra::inv_permute(&self.perm, &self.work, sol);
        for _ in 0..refinement {
            self.work2.fill(0.0);
            self.upper.sym_mul_add(sol, &mut self.work2, 1.0);
            for i in 0..rhs.len() {
                self.work2[i] = rhs[i] - self.work2[i];
            }
            crate::algebra::permute(&self.perm, &self.work2, &mut self.work);
            self.fac.solve_in_place(&mut self.work);
            crate::algebra::inv_permute(&self.perm, &self.work, &mut self.work2);
            for i in 0..sol.len() {
                sol[i] += self.work2[i];
            }
        }
    }
}

fn pack_rhs(rhs: &mut [f64], rd: &[f64], rp: &[f64], c: &[f64], n: usize, m: usize) {
    for i in 0..n {
        rhs[i] = -rd[i];
    }
    for i in 0..m {
        rhs[n + i] = c[i] - rp[i];
    }
}

fn recover_ds(ds: &mut [f64], c: &[f64], h: &[f64], m: usize, dz: &[f64]) {
    for i in 0..m {
        ds[i] = -c[i] - h_mul_row(h, m, i, dz);
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

fn combined_c(
    ws: &Workspace,
    sigma: f64,
    mu: f64,
    ds_aff: &[f64],
    dz_aff: &[f64],
    c: &mut [f64],
) {
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
            Cone::Exponential => {
                if let Some(g) = crate::cones::exp_dual_grad(&ws.z[off..off + 3]) {
                    for k in 0..3 {
                        c[off + k] += sigu * g[k];
                    }
                }
            }
            Cone::Power { alpha } => {
                if let Some((g, _)) = crate::cones::power_dual_grad_h(&ws.z[off..off + 3], *alpha) {
                    for k in 0..3 {
                        c[off + k] += sigu * g[k];
                    }
                }
            }
            Cone::SecondOrder { dim } => {
                // Mehrotra on the leading SOC residual, plus σμ on the identity.
                let i0 = off;
                c[i0] -= sigu;
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

fn duality_mu(ws: &Workspace) -> f64 {
    let mut gap = 0.0_f64;
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        let d = cone.dim();
        gap += dot(&ws.s[off..off + d], &ws.z[off..off + d]);
    }
    let nu = ws.cones.barrier_degree().max(1) as f64;
    (gap / nu).max(1e-16)
}

fn fill_scaling(ws: &Workspace, h: &mut [f64], mu: f64, primal_dual: bool) -> bool {
    let m = ws.s.len();
    h.fill(0.0);
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            Cone::Zero { dim } => {
                for k in 0..*dim {
                    h[(off + k) * m + (off + k)] = 1e-12;
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
                if let Some(hs) =
                    crate::cones::soc_nt_hessian(&ws.s[off..off + dim], &ws.z[off..off + dim])
                {
                    for i in 0..*dim {
                        for j in 0..*dim {
                            h[(off + i) * m + (off + j)] = hs[i * dim + j];
                        }
                    }
                } else {
                    for k in 0..*dim {
                        h[(off + k) * m + (off + k)] = 1.0;
                    }
                }
            }
            Cone::Exponential => {
                let Some(hs) =
                    crate::cones::exp_hs(&ws.s[off..off + 3], &ws.z[off..off + 3], mu, primal_dual)
                else {
                    return false;
                };
                for i in 0..3 {
                    for j in 0..3 {
                        h[(off + i) * m + (off + j)] = hs[i * 3 + j];
                    }
                }
            }
            Cone::Power { alpha } => {
                let Some((_, hd)) = crate::cones::power_dual_grad_h(&ws.z[off..off + 3], *alpha)
                else {
                    return false;
                };
                for i in 0..3 {
                    for j in 0..3 {
                        h[(off + i) * m + (off + j)] = mu * hd[i * 3 + j];
                    }
                }
            }
            _ => {
                for k in 0..cone.dim() {
                    h[(off + k) * m + (off + k)] = mu / ws.z[off + k].abs().max(1e-8).powi(2);
                }
            }
        }
    }
    true
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
    for j in 0..n {
        for idx in pu.col_ptr[j]..pu.col_ptr[j + 1] {
            trips.push((pu.row_idx[idx], j, pu.x[idx]));
        }
        trips.push((j, j, 1e-10));
    }
    for c in 0..n {
        for idx in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[idx];
            trips.push((c, n + r, a.x[idx]));
        }
    }
    // Always write every H entry (including zeros) so the AMD pattern is
    // iteration-invariant. H is block-diagonal across cones, but a dense
    // lower-right block is cheap at the finance sizes we target.
    for i in 0..m {
        for j in i..m {
            let v = -h[i * m + j];
            trips.push((n + i, n + j, if i == j { v - 1e-10 } else { v }));
        }
    }
    CscMatrix::from_triplets_keep_zeros(n + m, n + m, &trips)
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
            Cone::Power { alpha } => {
                a = a.min(crate::cones::power_backtrack(
                    &ws.s[off..off + 3],
                    &ds[off..off + 3],
                    *alpha,
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
            Cone::Power { alpha } => {
                a = a.min(crate::cones::power_backtrack(
                    &ws.z[off..off + 3],
                    &dz[off..off + 3],
                    *alpha,
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
