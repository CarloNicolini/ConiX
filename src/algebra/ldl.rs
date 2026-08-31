//! Simplicial LDL' factorization for symmetric quasi-definite matrices.
//!
//! The numeric algorithm follows Davis, Algorithm 849, in the QDLDL form used
//! for operator-splitting KKT systems. Input is the *upper* triangle in CSC.

use super::csc::CscMatrix;

const UNKNOWN: i64 = -1;

#[derive(Clone, Debug)]
pub struct LdlSymbolic {
    pub n: usize,
    pub etree: Vec<i64>,
    pub lnz: Vec<usize>,
    pub lp: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct LdlNumeric {
    pub sym: LdlSymbolic,
    pub li: Vec<usize>,
    pub lx: Vec<f64>,
    pub d: Vec<f64>,
    pub dinv: Vec<f64>,
    y_markers: Vec<bool>,
    y_vals: Vec<f64>,
    y_idx: Vec<usize>,
    elim: Vec<usize>,
    l_next: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct LdlError {
    pub msg: &'static str,
}

impl LdlSymbolic {
    pub fn analyze(a: &CscMatrix) -> Result<Self, LdlError> {
        assert_eq!(a.m, a.n);
        let n = a.n;
        let mut work = vec![0i64; n];
        let mut lnz = vec![0usize; n];
        let mut etree = vec![UNKNOWN; n];

        for i in 0..n {
            work[i] = 0;
            lnz[i] = 0;
            etree[i] = UNKNOWN;
            if a.col_ptr[i] == a.col_ptr[i + 1] {
                return Err(LdlError {
                    msg: "empty column in upper CSC",
                });
            }
        }

        for j in 0..n {
            work[j] = j as i64;
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                let mut i = a.row_idx[p];
                if i > j {
                    return Err(LdlError {
                        msg: "entry below diagonal",
                    });
                }
                while work[i] != j as i64 {
                    if etree[i] == UNKNOWN {
                        etree[i] = j as i64;
                    }
                    lnz[i] += 1;
                    work[i] = j as i64;
                    if etree[i] == UNKNOWN {
                        break;
                    }
                    i = etree[i] as usize;
                }
            }
        }

        let mut lp = vec![0usize; n + 1];
        for i in 0..n {
            lp[i + 1] = lp[i] + lnz[i];
        }
        Ok(Self { n, etree, lnz, lp })
    }
}

impl LdlNumeric {
    fn allocate(sym: LdlSymbolic) -> Self {
        let n = sym.n;
        let nnz_l = sym.lp[n];
        Self {
            sym,
            li: vec![0; nnz_l],
            lx: vec![0.0; nnz_l],
            d: vec![0.0; n],
            dinv: vec![0.0; n],
            y_markers: vec![false; n],
            y_vals: vec![0.0; n],
            y_idx: vec![0; n],
            elim: vec![0; n],
            l_next: vec![0; n],
        }
    }

    pub fn factor_with_reg(
        a: &CscMatrix,
        sym: &LdlSymbolic,
        n_pos: Option<usize>,
        eps: f64,
    ) -> Result<Self, LdlError> {
        let mut fac = Self::allocate(sym.clone());
        fac.numeric(a, n_pos, eps)?;
        Ok(fac)
    }

    pub fn factor(a: &CscMatrix, sym: &LdlSymbolic) -> Result<Self, LdlError> {
        Self::factor_with_reg(a, sym, None, 0.0)
    }

    pub fn factor_regularized(
        a: &CscMatrix,
        sym: &LdlSymbolic,
        n_pos: usize,
        eps: f64,
    ) -> Result<Self, LdlError> {
        Self::factor_with_reg(a, sym, Some(n_pos), eps)
    }

    /// Recompute \(L\) and \(D\) in place. Pattern and symbolic analysis stay fixed.
    pub fn refactor(
        &mut self,
        a: &CscMatrix,
        n_pos: Option<usize>,
        eps: f64,
    ) -> Result<(), LdlError> {
        self.numeric(a, n_pos, eps)
    }

    pub fn refactor_qd(&mut self, a: &CscMatrix, n_pos: usize, eps: f64) -> Result<(), LdlError> {
        self.refactor(a, Some(n_pos), eps)
            .or_else(|_| self.refactor(a, None, 0.0))
    }

    fn numeric(
        &mut self,
        a: &CscMatrix,
        n_pos: Option<usize>,
        eps: f64,
    ) -> Result<(), LdlError> {
        let n = self.sym.n;
        debug_assert_eq!(a.n, n);
        self.y_markers.fill(false);
        self.y_vals.fill(0.0);
        for i in 0..n {
            self.l_next[i] = self.sym.lp[i];
        }

        self.d[0] = a.x[a.col_ptr[0]];
        self.d[0] = regularize_pivot(self.d[0], 0, n_pos, eps)?;
        self.dinv[0] = 1.0 / self.d[0];

        for k in 1..n {
            let mut nnz_y = 0usize;
            self.d[k] = 0.0;
            for i in a.col_ptr[k]..a.col_ptr[k + 1] {
                let bidx = a.row_idx[i];
                if bidx == k {
                    self.d[k] = a.x[i];
                    continue;
                }
                self.y_vals[bidx] = a.x[i];
                if !self.y_markers[bidx] {
                    self.y_markers[bidx] = true;
                    self.elim[0] = bidx;
                    let mut nnz_e = 1usize;
                    let mut next = if self.sym.etree[bidx] == UNKNOWN {
                        None
                    } else {
                        Some(self.sym.etree[bidx] as usize)
                    };
                    while let Some(nx) = next {
                        if nx >= k || self.y_markers[nx] {
                            break;
                        }
                        self.y_markers[nx] = true;
                        self.elim[nnz_e] = nx;
                        nnz_e += 1;
                        next = if self.sym.etree[nx] == UNKNOWN {
                            None
                        } else {
                            Some(self.sym.etree[nx] as usize)
                        };
                    }
                    while nnz_e > 0 {
                        nnz_e -= 1;
                        self.y_idx[nnz_y] = self.elim[nnz_e];
                        nnz_y += 1;
                    }
                }
            }

            for ii in (0..nnz_y).rev() {
                let cidx = self.y_idx[ii];
                let tmp = self.l_next[cidx];
                let yv = self.y_vals[cidx];
                for j in self.sym.lp[cidx]..tmp {
                    self.y_vals[self.li[j]] -= self.lx[j] * yv;
                }
                self.li[tmp] = k;
                self.lx[tmp] = yv * self.dinv[cidx];
                self.d[k] -= yv * self.lx[tmp];
                self.l_next[cidx] += 1;
                self.y_vals[cidx] = 0.0;
                self.y_markers[cidx] = false;
            }

            self.d[k] = regularize_pivot(self.d[k], k, n_pos, eps)?;
            self.dinv[k] = 1.0 / self.d[k];
        }
        Ok(())
    }

    pub fn solve_in_place(&self, x: &mut [f64]) {
        let n = self.sym.n;
        debug_assert_eq!(x.len(), n);
        for i in 0..n {
            let val = x[i];
            for j in self.sym.lp[i]..self.sym.lp[i + 1] {
                x[self.li[j]] -= self.lx[j] * val;
            }
        }
        for i in 0..n {
            x[i] *= self.dinv[i];
        }
        for i in (0..n).rev() {
            let mut val = x[i];
            for j in self.sym.lp[i]..self.sym.lp[i + 1] {
                val -= self.lx[j] * x[self.li[j]];
            }
            x[i] = val;
        }
    }
}

fn regularize_pivot(d: f64, _k: usize, n_pos: Option<usize>, eps: f64) -> Result<f64, LdlError> {
    if n_pos.is_none() {
        if d == 0.0 {
            return Err(LdlError { msg: "zero pivot" });
        }
        return Ok(d);
    }
    let floor = eps.abs().max(1e-16);
    if d.abs() < floor {
        Ok(if d >= 0.0 { floor } else { -floor })
    } else {
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::CscMatrix;

    #[test]
    fn refactor_matches_fresh_factor() {
        let a = CscMatrix::from_triplets(
            3,
            3,
            &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0), (1, 2, 0.5), (2, 2, 2.0)],
        );
        let sym = LdlSymbolic::analyze(&a).unwrap();
        let mut fac = LdlNumeric::factor(&a, &sym).unwrap();
        let d0 = fac.d.clone();
        fac.refactor(&a, None, 0.0).unwrap();
        for i in 0..3 {
            assert!((fac.d[i] - d0[i]).abs() < 1e-12, "{:?} vs {d0:?}", fac.d);
        }
        let mut b = vec![1.0, 2.0, 3.0];
        fac.solve_in_place(&mut b);
        let mut b2 = vec![1.0, 2.0, 3.0];
        LdlNumeric::factor(&a, &sym)
            .unwrap()
            .solve_in_place(&mut b2);
        for i in 0..3 {
            assert!((b[i] - b2[i]).abs() < 1e-10);
        }
    }
}
