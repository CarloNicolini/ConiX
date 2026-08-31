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
    pub fn factor_with_reg(
        a: &CscMatrix,
        sym: &LdlSymbolic,
        n_pos: Option<usize>,
        eps: f64,
    ) -> Result<Self, LdlError> {
        let n = a.n;
        let nnz_l = sym.lp[n];
        let mut li = vec![0usize; nnz_l];
        let mut lx = vec![0.0; nnz_l];
        let mut d = vec![0.0; n];
        let mut dinv = vec![0.0; n];
        let mut y_markers = vec![false; n];
        let mut y_vals = vec![0.0; n];
        let mut y_idx = vec![0usize; n];
        let mut elim = vec![0usize; n];
        let mut l_next = vec![0usize; n];

        for i in 0..n {
            l_next[i] = sym.lp[i];
        }

        d[0] = a.x[a.col_ptr[0]];
        d[0] = regularize_pivot(d[0], 0, n_pos, eps)?;
        dinv[0] = 1.0 / d[0];

        for k in 1..n {
            let mut nnz_y = 0usize;
            d[k] = 0.0;
            for i in a.col_ptr[k]..a.col_ptr[k + 1] {
                let bidx = a.row_idx[i];
                if bidx == k {
                    d[k] = a.x[i];
                    continue;
                }
                y_vals[bidx] = a.x[i];
                if !y_markers[bidx] {
                    y_markers[bidx] = true;
                    elim[0] = bidx;
                    let mut nnz_e = 1usize;
                    let mut next = if sym.etree[bidx] == UNKNOWN {
                        None
                    } else {
                        Some(sym.etree[bidx] as usize)
                    };
                    while let Some(nx) = next {
                        if nx >= k || y_markers[nx] {
                            break;
                        }
                        y_markers[nx] = true;
                        elim[nnz_e] = nx;
                        nnz_e += 1;
                        next = if sym.etree[nx] == UNKNOWN {
                            None
                        } else {
                            Some(sym.etree[nx] as usize)
                        };
                    }
                    while nnz_e > 0 {
                        nnz_e -= 1;
                        y_idx[nnz_y] = elim[nnz_e];
                        nnz_y += 1;
                    }
                }
            }

            for ii in (0..nnz_y).rev() {
                let cidx = y_idx[ii];
                let tmp = l_next[cidx];
                let yv = y_vals[cidx];
                for j in sym.lp[cidx]..tmp {
                    y_vals[li[j]] -= lx[j] * yv;
                }
                li[tmp] = k;
                lx[tmp] = yv * dinv[cidx];
                d[k] -= yv * lx[tmp];
                l_next[cidx] += 1;
                y_vals[cidx] = 0.0;
                y_markers[cidx] = false;
            }

            d[k] = regularize_pivot(d[k], k, n_pos, eps)?;
            dinv[k] = 1.0 / d[k];
        }

        Ok(Self {
            sym: sym.clone(),
            li,
            lx,
            d,
            dinv,
        })
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
