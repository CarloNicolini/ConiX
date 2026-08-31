//! Homogeneous Douglas–Rachford (SCS v3 geometry) with a cached resolvent.

use crate::algebra::{dot, inf_norm};
use crate::kkt::KktSystem;
use crate::status::Status;
use crate::workspace::Workspace;

pub fn run(ws: &mut Workspace) {
    let n = ws.x.len();
    let m = ws.s.len();
    let l = n + m + 1;
    let rho = vec![1.0; m];
    // Resolvent metric: P+I and -I. Reuse the ADMM factor only when sigma=1 and ρ=1.
    let need_new =
        (ws.settings.sigma - 1.0).abs() > 1e-14 || ws.rho.iter().any(|&r| (r - 1.0).abs() > 1e-12);
    if need_new {
        if let Ok(k) = KktSystem::analyze(&ws.p, &ws.a, 1.0, &rho) {
            ws.kkt = k;
            ws.factorizations += 1;
            ws.rho = rho.clone();
        }
    }

    // h = [q; b], g = K^{-1}[q; -b]
    let mut rhs = vec![0.0; n + m];
    rhs[..n].copy_from_slice(&ws.q);
    for i in 0..m {
        rhs[n + i] = -ws.b[i];
    }
    ws.kkt
        .solve(&rhs, &mut ws.g_embed, ws.settings.iterative_refinement);

    let mut v = vec![0.0; l];
    if inf_norm(&ws.v_embed) < 1e-16 {
        v[l - 1] = 1.0;
    } else {
        v.copy_from_slice(&ws.v_embed);
    }
    let mut u = vec![0.0; l];
    let mut ut = vec![0.0; l];
    let mut rsk = vec![0.0; l];
    let mut aa = crate::engines::anderson::Anderson::new(ws.settings.anderson_memory, l);
    let mut status = Status::Unsolved;
    let mut iter = 0usize;
    let alpha = ws.settings.alpha.min(1.8).max(1.0);

    while iter < ws.settings.max_iter {
        iter += 1;
        if ws.settings.anderson_memory > 0 {
            aa.capture_in(&v);
        }
        // linear resolvent
        ut[..n + m].copy_from_slice(&v[..n + m]);
        for i in n..n + m {
            ut[i] = -ut[i];
        }
        let mut sol = vec![0.0; n + m];
        ws.kkt
            .solve(&ut[..n + m], &mut sol, ws.settings.iterative_refinement);
        ut[..n + m].copy_from_slice(&sol);
        let tau = if iter < 2 {
            1.0
        } else {
            root_plus(&ws.g_embed, &sol, &v, v[l - 1], n + m)
        };
        for i in 0..n + m {
            ut[i] -= tau * ws.g_embed[i];
        }
        ut[l - 1] = tau;

        // cone projection of 2ũ - v onto R^n × K* × R+
        for i in 0..l {
            u[i] = 2.0 * ut[i] - v[i];
        }
        ws.cones.project_dual(&mut u[n..n + m]);
        u[l - 1] = u[l - 1].max(0.0);

        for i in 0..l {
            rsk[i] = v[i] + u[i] - 2.0 * ut[i];
        }
        for i in 0..l {
            v[i] += alpha * (u[i] - ut[i]);
        }
        let vn = crate::algebra::nrm2(&v).max(1e-16);
        let target = (l as f64).sqrt();
        for vi in v.iter_mut() {
            *vi *= target / vn;
        }
        if ws.settings.anderson_memory > 0 {
            aa.maybe_replace(&mut v);
        }

        if iter % ws.settings.check_termination == 0 || iter == 1 {
            let tau = u[l - 1].abs().max(1e-16);
            for i in 0..n {
                ws.x[i] = u[i] / tau;
            }
            for i in 0..m {
                ws.z[i] = u[n + i] / tau;
                ws.s[i] = rsk[n + i] / tau;
            }
            let r = ws.original_residuals();
            let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
            if crate::verifier::solved_at(&r, eps) {
                status = Status::Solved;
                break;
            }
            // infeasibility: tau ~ 0, kappa = rsk[end]
            let kappa = rsk[l - 1];
            if tau < ws.settings.eps_infeas && kappa > 0.0 {
                if crate::verifier::check_primal_ray(
                    &ws.a,
                    &ws.b,
                    &ws.cones,
                    &ws.z,
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
                    &ws.x,
                    ws.settings.eps_infeas,
                ) {
                    status = Status::DualInfeasible;
                    break;
                }
            }
        }
    }
    ws.v_embed.copy_from_slice(&v);
    if status == Status::Unsolved && iter >= ws.settings.max_iter {
        status = Status::MaxIters;
    }
    ws.info.status = status;
    ws.info.iterations = iter;
    ws.info.engine = "splitting";
}

fn root_plus(g: &[f64], p: &[f64], mu: &[f64], eta: f64, nm: usize) -> f64 {
    let a = 1.0 + dot(&g[..nm], &g[..nm]);
    let b = dot(&mu[..nm], g) - 2.0 * dot(&p[..nm], g) - eta;
    let c = dot(&p[..nm], &p[..nm]) - dot(&p[..nm], &mu[..nm]);
    let rad = (b * b - 4.0 * a * c).max(0.0).sqrt();
    (-b + rad) / (2.0 * a).max(1e-16)
}

