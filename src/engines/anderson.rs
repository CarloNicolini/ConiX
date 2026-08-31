//! Safeguarded Anderson acceleration on a fixed splitting map.

use crate::algebra::{axpy, dot};

pub(crate) struct Anderson {
    m: usize,
    dim: usize,
    f_hist: Vec<Vec<f64>>,
    x_hist: Vec<Vec<f64>>,
}

impl Anderson {
    pub(crate) fn new(m: usize, dim: usize) -> Self {
        Self {
            m,
            dim,
            f_hist: Vec::new(),
            x_hist: Vec::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.f_hist.clear();
        self.x_hist.clear();
    }

    pub(crate) fn capture_in(&mut self, x: &[f64]) {
        if self.m == 0 {
            return;
        }
        self.x_hist.push(x.to_vec());
        if self.x_hist.len() > self.m + 1 {
            self.x_hist.remove(0);
        }
    }

    pub(crate) fn maybe_replace(&mut self, x: &mut [f64]) {
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
