//! Block-respecting Ruiz equilibration.

use crate::algebra::{CscExt, CscMatrix};
use crate::cones::{CompositeCone, Cone};

#[derive(Clone, Debug)]
pub struct Equilibration {
    pub d: Vec<f64>,
    pub e: Vec<f64>,
    pub c: f64,
}

impl Equilibration {
    pub fn identity(n: usize, m: usize) -> Self {
        Self {
            d: vec![1.0; n],
            e: vec![1.0; m],
            c: 1.0,
        }
    }
}

pub fn ruiz(
    p0: &CscMatrix,
    q0: &[f64],
    a0: &CscMatrix,
    b0: &[f64],
    cones: &CompositeCone,
    iters: usize,
) -> (Equilibration, CscMatrix, Vec<f64>, CscMatrix, Vec<f64>) {
    let n = p0.n;
    let m = a0.m;
    let mut d = vec![1.0; n];
    let mut e = vec![1.0; m];
    let mut p = p0.clone();
    let mut a = a0.clone();
    let mut q = q0.to_vec();
    let mut b = b0.to_vec();
    if iters == 0 {
        return (Equilibration { d, e, c: 1.0 }, p, q, a, b);
    }
    for _ in 0..iters {
        let mut coln = vec![0.0_f64; n];
        for j in 0..n {
            for idx in p.colptr[j]..p.colptr[j + 1] {
                let i = p.rowval[idx];
                let v = p.nzval[idx].abs();
                coln[j] = coln[j].max(v);
                coln[i] = coln[i].max(v);
            }
            for idx in a.colptr[j]..a.colptr[j + 1] {
                coln[j] = coln[j].max(a.nzval[idx].abs());
            }
        }
        let rown = a.row_inf_norms();
        let mut dstep = vec![1.0; n];
        for j in 0..n {
            dstep[j] = 1.0 / coln[j].max(1e-12).sqrt();
        }
        let mut estep = vec![1.0; m];
        for (cone, &off) in cones.cones.iter().zip(&cones.offsets) {
            let dim = cone.dim();
            match cone {
                Cone::Zero { .. } | Cone::Nonnegative { .. } => {
                    for k in 0..dim {
                        estep[off + k] = 1.0 / rown[off + k].max(1e-12).sqrt();
                    }
                }
                _ => {
                    let mut mx = 1e-12_f64;
                    for k in 0..dim {
                        mx = mx.max(rown[off + k]);
                    }
                    let s = 1.0 / mx.sqrt();
                    for k in 0..dim {
                        estep[off + k] = s;
                    }
                }
            }
        }
        for j in 0..p.n {
            for idx in p.colptr[j]..p.colptr[j + 1] {
                let i = p.rowval[idx];
                p.nzval[idx] *= dstep[i] * dstep[j];
            }
        }
        for j in 0..a.n {
            for idx in a.colptr[j]..a.colptr[j + 1] {
                let i = a.rowval[idx];
                a.nzval[idx] *= estep[i] * dstep[j];
            }
        }
        for i in 0..n {
            q[i] *= dstep[i];
            d[i] *= dstep[i];
        }
        for i in 0..m {
            b[i] *= estep[i];
            e[i] *= estep[i];
        }
    }
    let pmax = p
        .inf_norm()
        .max(q.iter().fold(0.0_f64, |m, &v| m.max(v.abs())))
        .max(1.0);
    let c = 1.0 / pmax;
    for v in p.nzval.iter_mut() {
        *v *= c;
    }
    for v in q.iter_mut() {
        *v *= c;
    }
    (Equilibration { d, e, c }, p, q, a, b)
}

pub fn unscale_solution(eq: &Equilibration, x: &mut [f64], s: &mut [f64], z: &mut [f64]) {
    for i in 0..x.len() {
        x[i] *= eq.d[i];
    }
    for i in 0..s.len() {
        s[i] /= eq.e[i].max(1e-16);
    }
    for i in 0..z.len() {
        z[i] *= eq.e[i] / eq.c.max(1e-16);
    }
}

/// Map an unscaled iterate into scaled coordinates.
pub fn scale_iterate(eq: &Equilibration, x: &mut [f64], s: &mut [f64], z: &mut [f64]) {
    for i in 0..x.len() {
        x[i] /= eq.d[i].max(1e-16);
    }
    for i in 0..s.len() {
        s[i] *= eq.e[i];
    }
    for i in 0..z.len() {
        z[i] *= eq.c / eq.e[i].max(1e-16);
    }
}
