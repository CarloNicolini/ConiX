//! COSMO-style two-block ADMM with cached quasi-definite factors.

use crate::algebra::{axpy, copy_from, dot, inf_norm};
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
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m];

    // w = [x; s - z/ρ]  because z = ρ(s - w_s)
    ws.w[..n].copy_from_slice(&ws.x);
    for i in 0..m {
        ws.w[n + i] = ws.s[i] - ws.z[i] / ws.rho[i];
    }

    // initialization step so iterates match standard ADMM
    x_update(ws, &mut ls, &mut sol, &mut s_tl, sigma, n, m);
    w_update(ws, &sol[..n], &s_tl, alpha, n, m);

    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let mut aa = Anderson::new(ws.settings.anderson_memory, n + m);

    while iter < ws.settings.max_iter {
        iter += 1;
        copy_from(&mut ws.w_prev, &ws.w);
        if ws.settings.anderson_memory > 0 {
            aa.capture_in(&ws.w);
        }

        // z-update: s = Π_K(w[n:])
        ws.s.copy_from_slice(&ws.w[n..]);
        ws.cones.project(&mut ws.s);

        if ws.settings.adaptive_rho
            && iter % ws.settings.adaptive_rho_interval.max(1) == 0
            && iter > 1
        {
            recover_mu(ws, n);
            if adapt_rho(ws) {
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

        if ws.settings.anderson_memory > 0 {
            aa.maybe_replace(&mut ws.w);
        }

        if iter % ws.settings.check_termination == 0 || iter == 1 {
            recover_mu(ws, n);
            ws.x.copy_from_slice(&ws.w[..n]);
            let (rp, rd) = scaled_residuals(ws);
            if rp <= ws.settings.eps_abs + ws.settings.eps_rel
                && rd <= ws.settings.eps_abs + ws.settings.eps_rel
            {
                status = Status::Solved;
                break;
            }
        }

        if iter % ws.settings.check_infeasibility == 0 {
            recover_mu(ws, n);
            for i in 0..n {
                dx[i] = ws.w[i] - ws.w_prev[i];
            }
            for i in 0..m {
                dy[i] = ws.z[i] - dy[i]; // filled below
            }
            // δy from μ difference: store previous μ in dy at last check
            // Use current z vs previous recovered from w_prev
            let mut z_prev = vec![0.0; m];
            for i in 0..m {
                z_prev[i] = ws.rho[i] * (ws.s[i] - ws.w_prev[n + i]);
                dy[i] = ws.z[i] - z_prev[i];
            }
            if crate::verifier::check_primal_ray(
                &ws.a,
                &ws.b,
                &ws.cones,
                &dy,
                ws.settings.eps_infeas,
            ) {
                status = Status::PrimalInfeasible;
                break;
            }
            if crate::verifier::check_dual_ray(
                &ws.p,
                &ws.q,
                &ws.a,
                &ws.cones,
                &dx,
                ws.settings.eps_infeas,
            ) {
                status = Status::DualInfeasible;
                break;
            }
        }
    }

    recover_mu(ws, n);
    ws.x.copy_from_slice(&ws.w[..n]);
    if status == Status::Unsolved && iter >= ws.settings.max_iter {
        status = Status::MaxIters;
    }
    if ws.settings.polish
        && ws.cones.is_polyhedral()
        && status != Status::PrimalInfeasible
        && status != Status::DualInfeasible
    {
        polish(ws);
        let (rp, rd) = scaled_residuals(ws);
        if rp <= ws.settings.eps_abs + ws.settings.eps_rel
            && rd <= ws.settings.eps_abs + ws.settings.eps_rel
        {
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

fn recover_mu(ws: &mut Workspace, n: usize) {
    let m = ws.s.len();
    // Moreau: w_s - s ∈ -K*, so z = ρ (s - w_s) ∈ K*.
    for i in 0..m {
        ws.z[i] = ws.rho[i] * (ws.s[i] - ws.w_prev[n + i]);
    }
}

fn scaled_residuals(ws: &Workspace) -> (f64, f64) {
    let n = ws.x.len();
    let m = ws.s.len();
    let mut ax = vec![0.0; m];
    ws.a.mul(&ws.x, &mut ax);
    let mut rp = 0.0_f64;
    for i in 0..m {
        rp = rp.max((ax[i] + ws.s[i] - ws.b[i]).abs());
    }
    let mut px = vec![0.0; n];
    ws.p.sym_mul_add(&ws.x, &mut px, 1.0);
    let mut atz = vec![0.0; n];
    ws.a.tmul(&ws.z, &mut atz);
    let mut rd = 0.0_f64;
    for i in 0..n {
        rd = rd.max((px[i] + atz[i] + ws.q[i]).abs());
    }
    let rp = rp / (1.0 + inf_norm(&ws.b));
    let rd = rd / (1.0 + inf_norm(&ws.q));
    (rp, rd)
}

fn adapt_rho(ws: &mut Workspace) -> bool {
    let (rp, rd) = scaled_residuals(ws);
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

fn polish(ws: &mut Workspace) {
    // Identify nearly-active nonnegative rows and freeze them as equalities.
    // For zero cones they are already equalities.
    let n = ws.x.len();
    let mut eq_rows: Vec<usize> = Vec::new();
    for (cone, &off) in ws.cones.cones.iter().zip(&ws.cones.offsets) {
        match cone {
            crate::cones::Cone::Zero { dim } => {
                for k in 0..*dim {
                    eq_rows.push(off + k);
                }
            }
            crate::cones::Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    let i = off + k;
                    if ws.s[i].abs() < 1e-4 && ws.z[i].abs() > 1e-8 {
                        eq_rows.push(i);
                    }
                }
            }
            _ => {}
        }
    }
    if eq_rows.is_empty() {
        return;
    }
    // Dense equality QP on the identified set, accepted only if residuals drop.
    let ne = eq_rows.len();
    let dim = n + ne;
    let mut k = vec![0.0; dim * dim];
    let mut rhs = vec![0.0; dim];
    // P block
    let pd = ws.p.to_dense();
    for i in 0..n {
        for j in 0..n {
            let v = if i < pd.len() && j < pd[i].len() {
                pd[i][j]
            } else {
                0.0
            };
            k[i * dim + j] += v;
            if i != j {
                // dense from upper-only P: fill from CSC via sym
            }
        }
        rhs[i] = -ws.q[i];
        k[i * dim + i] += 1e-10;
    }
    // Use CSC symmetric multiply to fill P properly
    k[..dim * dim].fill(0.0);
    for j in 0..n {
        for p in ws.p.col_ptr[j]..ws.p.col_ptr[j + 1] {
            let i = ws.p.row_idx[p];
            k[i * dim + j] += ws.p.x[p];
            if i != j {
                k[j * dim + i] += ws.p.x[p];
            }
        }
        k[j * dim + j] += 1e-10;
        rhs[j] = -ws.q[j];
    }
    let ad = ws.a.to_dense();
    for (t, &row) in eq_rows.iter().enumerate() {
        for j in 0..n {
            let v = ad[row][j];
            k[n + t] += 0.0;
            k[(n + t) * dim + j] = v;
            k[j * dim + (n + t)] = v;
        }
        rhs[n + t] = ws.b[row];
    }
    if let Some(sol) = solve_dense(&k, &rhs, dim) {
        let xnew = sol[..n].to_vec();
        let mut snew = vec![0.0; ws.s.len()];
        let mut ax = vec![0.0; ws.s.len()];
        ws.a.mul(&xnew, &mut ax);
        for i in 0..ws.s.len() {
            snew[i] = ws.b[i] - ax[i];
        }
        ws.cones.project(&mut snew);
        let mut znew = ws.z.clone();
        for (t, &row) in eq_rows.iter().enumerate() {
            znew[row] = sol[n + t];
        }
        let r_old =
            crate::verifier::residuals(&ws.p, &ws.q, &ws.a, &ws.b, &ws.cones, &ws.x, &ws.s, &ws.z);
        let r_new =
            crate::verifier::residuals(&ws.p, &ws.q, &ws.a, &ws.b, &ws.cones, &xnew, &snew, &znew);
        if r_new.res_pri + r_new.res_dual + r_new.res_gap
            < r_old.res_pri + r_old.res_dual + r_old.res_gap
        {
            ws.x = xnew;
            ws.s = snew;
            ws.z = znew;
        }
    }
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
        if best < 1e-16 {
            return None;
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

struct Anderson {
    m: usize,
    dim: usize,
    f_hist: Vec<Vec<f64>>,
    x_hist: Vec<Vec<f64>>,
}

impl Anderson {
    fn new(m: usize, dim: usize) -> Self {
        Self {
            m,
            dim,
            f_hist: Vec::new(),
            x_hist: Vec::new(),
        }
    }
    fn reset(&mut self) {
        self.f_hist.clear();
        self.x_hist.clear();
    }
    fn capture_in(&mut self, x: &[f64]) {
        if self.m == 0 {
            return;
        }
        self.x_hist.push(x.to_vec());
        if self.x_hist.len() > self.m + 1 {
            self.x_hist.remove(0);
        }
    }
    fn maybe_replace(&mut self, x: &mut [f64]) {
        if self.m == 0 || self.x_hist.len() < 3 {
            return;
        }
        let k = self.x_hist.len();
        let f: Vec<f64> = (0..self.dim)
            .map(|i| x[i] - self.x_hist[k - 1][i])
            .collect();
        self.f_hist.push(f);
        if self.f_hist.len() > self.m {
            self.f_hist.remove(0);
        }
        if self.f_hist.len() < 2 {
            return;
        }
        let mk = self.f_hist.len() - 1;
        // Type-I least squares on ΔF α ≈ f_k
        let mut gram = vec![0.0; mk * mk];
        let mut rhs = vec![0.0; mk];
        let fk = &self.f_hist[mk];
        for i in 0..mk {
            let di: Vec<f64> = (0..self.dim)
                .map(|t| self.f_hist[i + 1][t] - self.f_hist[i][t])
                .collect();
            rhs[i] = dot(&di, fk);
            for j in 0..mk {
                let dj: Vec<f64> = (0..self.dim)
                    .map(|t| self.f_hist[j + 1][t] - self.f_hist[j][t])
                    .collect();
                gram[i * mk + j] = dot(&di, &dj);
            }
            gram[i * mk + i] += 1e-8;
        }
        if let Some(alpha) = solve_dense(&gram, &rhs, mk) {
            let mut cand = x.to_vec();
            let nrm_a: f64 = alpha.iter().map(|v| v.abs()).sum();
            if nrm_a > 1e3 {
                return;
            }
            for i in 0..mk {
                axpy(&mut cand, -alpha[i], &{
                    (0..self.dim)
                        .map(|t| self.x_hist[self.x_hist.len() - mk + i][t])
                        .collect::<Vec<_>>()
                });
            }
            // merit: ||cand - x_prev|| vs ||x - x_prev||
            let prev = &self.x_hist[k - 1];
            let mut n1 = 0.0;
            let mut n2 = 0.0;
            for i in 0..self.dim {
                n1 += (x[i] - prev[i]).powi(2);
                n2 += (cand[i] - prev[i]).powi(2);
            }
            if n2 < 0.9 * n1 && cand.iter().all(|v| v.is_finite()) {
                x.copy_from_slice(&cand);
            }
        }
    }
}
