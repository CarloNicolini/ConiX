//! COSMO-style two-block ADMM with cached quasi-definite factors.
//!
//! COSMO stores the primal `x` as a view of `w_prev`, so residuals are
//! evaluated on the same `w` that was just projected onto `K`. We copy that
//! convention: after `w ← w`, `s = Π_K(w_s)`, termination uses `x = w_prev[1:n]`.

use crate::algebra::{copy_from, inf_norm};
use crate::kkt::KktSystem;
use crate::status::{Status, UpdateClass};
use crate::workspace::Workspace;

pub fn run(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    let alpha = ws.settings.alpha;
    let sigma = ws.settings.sigma;
    let mut ls = vec![0.0; n + m];
    let mut sol = vec![0.0; n + m];
    let mut s_tl = vec![0.0; m];
    let mut ax = vec![0.0; m];
    let mut px = vec![0.0; n];
    let mut atz = vec![0.0; n];

    // w = [x; s - z/ρ]  because z = ρ(s - w_s) = -μ_COSMO ∈ K*.
    ws.w[..n].copy_from_slice(&ws.x);
    for i in 0..m {
        ws.w[n + i] = ws.s[i] - ws.z[i] / ws.rho[i];
    }

    // COSMO does one affine step so the loop's first projection matches ADMM.
    // Skip it only when re-solving unchanged data from an accepted solution.
    let reuse = ws.has_solution && matches!(ws.last_update, UpdateClass::Setup);
    if !reuse {
        x_update(ws, &mut ls, &mut sol, &mut s_tl, sigma, n, m);
        w_update(ws, &sol[..n], &s_tl, alpha, n, m);
    }

    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let aa_mem = if n + m < 32 {
        0
    } else {
        ws.settings.anderson_memory
    };
    let mut aa = crate::engines::anderson::Anderson::new(aa_mem, n + m);
    let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
    let mut n_rho = 0usize;

    while iter < ws.settings.max_iter {
        iter += 1;
        copy_from(&mut ws.w_prev, &ws.w);
        if aa_mem > 0 {
            aa.capture_in(&ws.w);
        }

        // s = Π_K(w[n:]); x for residuals is w_prev[1:n] == current w[1:n] here.
        ws.s.copy_from_slice(&ws.w[n..]);
        ws.cones.project(&mut ws.s);

        if ws.settings.adaptive_rho
            && n_rho < ws.settings.adaptive_rho_max_adaptions
            && iter % ws.settings.adaptive_rho_interval.max(1) == 0
            && iter > 1
        {
            recover_z(ws, n);
            ws.x.copy_from_slice(&ws.w_prev[..n]);
            if adapt_rho(ws, &mut ax, &mut px, &mut atz, eps) {
                n_rho += 1;
                ws.kkt.update_rho(&ws.rho).ok();
                ws.factorizations += 1;
                aa.reset();
                for i in 0..m {
                    ws.w[n + i] = ws.s[i] - ws.z[i] / ws.rho[i];
                }
            }
        }

        x_update(ws, &mut ls, &mut sol, &mut s_tl, sigma, n, m);
        w_update(ws, &sol[..n], &s_tl, alpha, n, m);

        if aa_mem > 0 {
            aa.maybe_replace(&mut ws.w);
        }

        if iter % ws.settings.check_termination == 0 || iter == 1 || iter == ws.settings.max_iter {
            recover_z(ws, n);
            // COSMO: x is a view of w_prev.
            ws.x.copy_from_slice(&ws.w_prev[..n]);
            let r = ws.original_residuals();
            if crate::verifier::solved_at(&r, eps) {
                status = Status::Solved;
                break;
            }
        }
    }

    recover_z(ws, n);
    ws.x.copy_from_slice(&ws.w_prev[..n]);
    if status == Status::Unsolved && iter >= ws.settings.max_iter {
        status = Status::MaxIters;
    }
    if ws.settings.polish
        && ws.cones.is_polyhedral()
        && status != Status::PrimalInfeasible
        && status != Status::DualInfeasible
    {
        sparse_polish(ws);
        let r = ws.original_residuals();
        if crate::verifier::solved_at(&r, eps) {
            status = Status::Solved;
        }
    }
    ws.info.status = status;
    ws.info.iterations = iter;
    ws.info.engine = "admm";
    ws.info.update_class = if matches!(ws.last_update, UpdateClass::Setup) {
        UpdateClass::Setup
    } else {
        ws.last_update
    };
}

fn x_update(
    ws: &mut Workspace,
    ls: &mut [f64],
    sol: &mut [f64],
    s_tl: &mut [f64],
    sigma: f64,
    n: usize,
    m: usize,
) {
    for i in 0..n {
        ls[i] = sigma * ws.w[i] - ws.q[i];
    }
    for i in 0..m {
        ls[n + i] = ws.b[i] - 2.0 * ws.s[i] + ws.w[n + i];
    }
    ws.kkt.solve(ls, sol, ws.settings.iterative_refinement);
    for i in 0..m {
        s_tl[i] = 2.0 * ws.s[i] - ws.w[n + i] - sol[n + i] / ws.rho[i];
    }
}

fn w_update(ws: &mut Workspace, x_tl: &[f64], s_tl: &[f64], alpha: f64, n: usize, m: usize) {
    for i in 0..n {
        ws.w[i] += alpha * (x_tl[i] - ws.w[i]);
    }
    for i in 0..m {
        ws.w[n + i] += alpha * (s_tl[i] - ws.s[i]);
    }
}

fn recover_z(ws: &mut Workspace, n: usize) {
    let m = ws.s.len();
    // Moreau: μ_COSMO = ρ (w_s - s) ∈ -K*, so z = -μ = ρ (s - w_s) ∈ K*.
    for i in 0..m {
        ws.z[i] = ws.rho[i] * (ws.s[i] - ws.w_prev[n + i]);
    }
}

fn scaled_residuals(ws: &Workspace, ax: &mut [f64], px: &mut [f64], atz: &mut [f64]) -> (f64, f64) {
    ws.a.mul(&ws.x, ax);
    let mut rp = 0.0_f64;
    for i in 0..ax.len() {
        rp = rp.max((ax[i] + ws.s[i] - ws.b[i]).abs());
    }
    px.fill(0.0);
    ws.p.sym_mul_add(&ws.x, px, 1.0);
    ws.a.tmul(&ws.z, atz);
    let mut rd = 0.0_f64;
    for i in 0..px.len() {
        rd = rd.max((px[i] + atz[i] + ws.q[i]).abs());
    }
    let rp = rp / (1.0 + inf_norm(&ws.b));
    let rd = rd / (1.0 + inf_norm(&ws.q));
    (rp, rd)
}

fn adapt_rho(
    ws: &mut Workspace,
    ax: &mut [f64],
    px: &mut [f64],
    atz: &mut [f64],
    eps: f64,
) -> bool {
    let (rp, rd) = scaled_residuals(ws, ax, px, atz);
    if rp < 10.0 * eps && rd < 10.0 * eps {
        return false;
    }
    if rp < 1e-18 && rd < 1e-18 {
        return false;
    }
    let ratio = (rp / rd.max(1e-18)).sqrt();
    if !(0.2..=5.0).contains(&ratio) {
        let new = (ws.rho[0] * ratio).clamp(1e-6, 1e6);
        if (new / ws.rho[0] - 1.0).abs() > 0.2 {
            let factor = new / ws.rho[0];
            for r in ws.rho.iter_mut() {
                *r *= factor;
            }
            return true;
        }
    }
    false
}

/// OSQP-style equality QP on an identified active set, using a throwaway
/// reduced KKT so the sequential factor cache is not dirtied.
fn sparse_polish(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
    let nn_rows: Vec<usize> = {
        let mut v = Vec::new();
        for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
            if let crate::cones::Cone::Nonnegative { dim } = cone {
                for k in 0..*dim {
                    v.push(off + k);
                }
            }
        }
        v
    };
    for &thresh in &[1e-4_f64, 1e-3, 5e-3] {
        let mut eq_rows: Vec<usize> = Vec::new();
        let mut is_eq = vec![false; m];
        for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
            match cone {
                crate::cones::Cone::Zero { dim } => {
                    for k in 0..*dim {
                        is_eq[off + k] = true;
                        eq_rows.push(off + k);
                    }
                }
                crate::cones::Cone::Nonnegative { dim } => {
                    for k in 0..*dim {
                        let i = off + k;
                        if ws.s[i].abs() <= thresh || ws.z[i] > 10.0 * thresh {
                            is_eq[i] = true;
                            eq_rows.push(i);
                        }
                    }
                }
                _ => {}
            }
        }
        if eq_rows.is_empty() {
            continue;
        }
        let mut accepted = false;
        for _ in 0..12 {
            let a_eq = ws.a.select_rows(&eq_rows);
            let mut b_eq = vec![0.0; eq_rows.len()];
            for (t, &row) in eq_rows.iter().enumerate() {
                b_eq[t] = ws.b[row];
            }
            let Some(sol) = solve_eq_qp(&ws.p, &a_eq, &ws.q, &b_eq) else {
                break;
            };
            let xnew = sol[..n].to_vec();
            let mut ax = vec![0.0; m];
            ws.a.mul(&xnew, &mut ax);
            let mut sraw = vec![0.0; m];
            for i in 0..m {
                sraw[i] = ws.b[i] - ax[i];
            }
            let mut changed = false;
            for &i in &nn_rows {
                if !is_eq[i] && sraw[i] < -1e-12 {
                    is_eq[i] = true;
                    eq_rows.push(i);
                    changed = true;
                }
            }
            if changed {
                continue;
            }
            let mut snew = vec![0.0; m];
            let mut znew = vec![0.0; m];
            for i in 0..m {
                snew[i] = if is_eq[i] { 0.0 } else { sraw[i].max(0.0) };
            }
            for (t, &row) in eq_rows.iter().enumerate() {
                znew[row] = sol[n + t];
            }
            let mut dropped = false;
            for &i in &nn_rows {
                if is_eq[i] && znew[i] < -1e-12 {
                    is_eq[i] = false;
                    dropped = true;
                } else if znew[i] < 0.0 {
                    znew[i] = 0.0;
                }
            }
            if dropped {
                eq_rows.retain(|&r| is_eq[r]);
                continue;
            }
            let r_old = ws.original_residuals();
            let r_new = {
                let mut x = xnew.clone();
                let mut s = snew.clone();
                let mut z = znew.clone();
                crate::scale::unscale_solution(&ws.eq, &mut x, &mut s, &mut z);
                crate::verifier::residuals(
                    &ws.orig.p,
                    &ws.orig.q,
                    &ws.orig.a,
                    &ws.orig.b,
                    &ws.orig.cones,
                    &x,
                    &s,
                    &z,
                )
            };
            if crate::verifier::merit(&r_new) <= crate::verifier::merit(&r_old) {
                ws.x = xnew;
                ws.s = snew;
                ws.z = znew;
                ws.sync_w();
                accepted = true;
                if crate::verifier::solved_at(&r_new, eps) {
                    return;
                }
            }
            break;
        }
        if accepted && crate::verifier::solved_at(&ws.original_residuals(), eps) {
            return;
        }
    }
}

fn solve_eq_qp(
    p: &crate::algebra::CscMatrix,
    a_eq: &crate::algebra::CscMatrix,
    q: &[f64],
    b_eq: &[f64],
) -> Option<Vec<f64>> {
    let n = p.n;
    let ne = a_eq.m;
    let dim = n + ne;
    if dim == 0 {
        return None;
    }
    if dim <= 128 {
        let mut k = vec![0.0; dim * dim];
        let mut rhs = vec![0.0; dim];
        for j in 0..n {
            for idx in p.col_ptr[j]..p.col_ptr[j + 1] {
                let i = p.row_idx[idx];
                k[i * dim + j] += p.x[idx];
                if i != j {
                    k[j * dim + i] += p.x[idx];
                }
            }
            k[j * dim + j] += 1e-8;
            rhs[j] = -q[j];
        }
        for c in 0..n {
            for idx in a_eq.col_ptr[c]..a_eq.col_ptr[c + 1] {
                let r = a_eq.row_idx[idx];
                let v = a_eq.x[idx];
                k[(n + r) * dim + c] = v;
                k[c * dim + (n + r)] = v;
            }
        }
        for t in 0..ne {
            k[(n + t) * dim + (n + t)] = -1e-8;
            rhs[n + t] = b_eq[t];
        }
        if let Some(sol) = solve_dense(&k, &rhs, dim) {
            return Some(sol);
        }
    }
    let rho_eq = vec![1e6; ne];
    let mut kkt = KktSystem::analyze(p, a_eq, 1e-6, &rho_eq).ok()?;
    let mut rhs = vec![0.0; dim];
    for i in 0..n {
        rhs[i] = -q[i];
    }
    rhs[n..].copy_from_slice(b_eq);
    let mut sol = vec![0.0; dim];
    kkt.solve(&rhs, &mut sol, 8);
    Some(sol)
}

fn solve_dense(k: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = k.to_vec();
    let mut b = rhs.to_vec();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for i in col + 1..n {
            let v = a[i * n + col].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-14 {
            a[col * n + col] = if a[col * n + col] >= 0.0 {
                1e-12
            } else {
                -1e-12
            };
            piv = col;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        let diag = a[col * n + col];
        for i in col + 1..n {
            let f = a[i * n + col] / diag;
            b[i] -= f * b[col];
            for j in col..n {
                a[i * n + j] -= f * a[col * n + j];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    Some(x)
}
