//! Independent KKT residual and ray checks in original coordinates.

use crate::algebra::{dot, inf_norm, CscMatrix, CscExt};
use crate::cones::CompositeCone;
use crate::status::Status;

#[derive(Clone, Debug)]
pub struct Residuals {
    pub res_pri: f64,
    pub res_dual: f64,
    pub res_gap: f64,
    pub res_cone: f64,
    pub res_comp: f64,
    pub obj_p: f64,
    pub obj_d: f64,
}

pub fn merit(r: &Residuals) -> f64 {
    r.res_pri + r.res_dual + r.res_gap + r.res_cone + r.res_comp
}

pub fn residuals(
    p: &CscMatrix,
    q: &[f64],
    a: &CscMatrix,
    b: &[f64],
    cones: &CompositeCone,
    x: &[f64],
    s: &[f64],
    z: &[f64],
) -> Residuals {
    let n = x.len();
    let m = s.len();
    let mut ax = vec![0.0; m];
    a.mul(x, &mut ax);
    let mut rp = vec![0.0; m];
    for i in 0..m {
        rp[i] = ax[i] + s[i] - b[i];
    }
    let mut px = vec![0.0; n];
    p.sym_mul_add(x, &mut px, 1.0);
    let mut atz = vec![0.0; n];
    a.tmul(z, &mut atz);
    let mut rd = vec![0.0; n];
    for i in 0..n {
        rd[i] = px[i] + atz[i] + q[i];
    }
    let xtpx = dot(x, &px);
    let obj_p = 0.5 * xtpx + dot(q, x);
    let obj_d = -0.5 * xtpx - dot(b, z);
    let gap = (obj_p - obj_d).abs();
    let comp = dot(s, z).abs();
    Residuals {
        res_pri: inf_norm(&rp) / (1.0 + inf_norm(b)),
        res_dual: inf_norm(&rd) / (1.0 + inf_norm(q)),
        res_gap: gap / (1.0 + obj_p.abs() + obj_d.abs()),
        res_cone: cones.dist(s).max(cones.dist_dual(z)) / (1.0 + inf_norm(s).max(inf_norm(z))),
        res_comp: comp / (1.0 + obj_p.abs() + obj_d.abs()),
        obj_p,
        obj_d,
    }
}

pub fn solved_at(r: &Residuals, eps: f64) -> bool {
    r.res_pri <= eps
        && r.res_dual <= eps
        && r.res_gap <= eps
        && r.res_cone <= eps
        && r.res_comp <= eps
}

pub fn check_primal_ray(
    a: &CscMatrix,
    b: &[f64],
    cones: &CompositeCone,
    z: &[f64],
    eps: f64,
) -> bool {
    let mut atz = vec![0.0; a.n];
    a.tmul(z, &mut atz);
    let bz = dot(b, z);
    inf_norm(&atz) <= eps && cones.dist_dual(z) <= eps && bz < -eps
}

pub fn check_dual_ray(
    p: &CscMatrix,
    q: &[f64],
    a: &CscMatrix,
    cones: &CompositeCone,
    d: &[f64],
    eps: f64,
) -> bool {
    let mut pd = vec![0.0; p.n];
    p.sym_mul_add(d, &mut pd, 1.0);
    let mut ad = vec![0.0; a.m];
    a.mul(d, &mut ad);
    for v in ad.iter_mut() {
        *v = -*v;
    }
    inf_norm(&pd) <= eps && cones.dist(&ad) <= eps && dot(q, d) < -eps
}

pub fn classify(
    r: &Residuals,
    eps: f64,
    pri_ray: bool,
    dual_ray: bool,
    iters: usize,
    max_iter: usize,
) -> Status {
    if pri_ray {
        return Status::PrimalInfeasible;
    }
    if dual_ray {
        return Status::DualInfeasible;
    }
    if solved_at(r, eps) {
        return Status::Solved;
    }
    if iters >= max_iter {
        return Status::MaxIters;
    }
    Status::Unsolved
}
