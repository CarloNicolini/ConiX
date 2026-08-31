//! Sparse CSC and vector kernels used by ConiX engines.
//!
//! [`CscMatrix`] and [`QDLDL`] factorisation are reused from Clarabel.rs (the
//! same numerical substrate as COSMO.rs). Clarabel's `gemv`/`symv` trait methods
//! are crate-private, so the kernels ConiX needs are implemented here on top of
//! the public CSC storage. The ConiX ADMM / DR / IPM algorithms are unchanged.

#![allow(non_snake_case)]

pub use clarabel::algebra::{
    MatrixMath, MatrixMathMut, ScalarMath, TriangularMatrixChecks, VectorMath,
};
pub use clarabel::qdldl::{
    QDLDLError, QDLDLFactorisation, QDLDLSettings, QDLDLSettingsBuilder,
};

/// Clarabel CSC storage specialised to `f64` (same substrate as COSMO.rs).
pub type CscMatrix = clarabel::algebra::CscMatrix<f64>;

/// Build CSC from `(row, col, value)` triplets, dropping exact zeros.
pub fn from_triplets(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> CscMatrix {
    let mut I = Vec::with_capacity(trips.len());
    let mut J = Vec::with_capacity(trips.len());
    let mut V = Vec::with_capacity(trips.len());
    for &(i, j, v) in trips {
        if v != 0.0 {
            I.push(i);
            J.push(j);
            V.push(v);
        }
    }
    CscMatrix::new_from_triplets(m, n, I, J, V)
}

/// Build CSC from triplets, keeping structural zeros (needed for R1 update slots).
pub fn from_triplets_keep_zeros(
    m: usize,
    n: usize,
    trips: &[(usize, usize, f64)],
) -> CscMatrix {
    let mut I = Vec::with_capacity(trips.len());
    let mut J = Vec::with_capacity(trips.len());
    let mut V = Vec::with_capacity(trips.len());
    for &(i, j, v) in trips {
        I.push(i);
        J.push(j);
        V.push(v);
    }
    CscMatrix::new_from_triplets(m, n, I, J, V)
}

/// Extension methods used throughout ConiX on Clarabel CSC matrices.
pub trait CscExt {
    fn from_triplets(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> Self;
    fn from_triplets_keep_zeros(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> Self;
    fn upper_triangle(&self) -> Self;
    fn same_pattern(&self, other: &Self) -> bool;
    fn mul(&self, x: &[f64], y: &mut [f64]);
    fn mul_add(&self, x: &[f64], y: &mut [f64], alpha: f64);
    fn tmul(&self, x: &[f64], y: &mut [f64]);
    fn tmul_add(&self, x: &[f64], y: &mut [f64], alpha: f64);
    fn sym_mul_add(&self, x: &[f64], y: &mut [f64], alpha: f64);
    fn select_row_indices(&self, rows: &[usize]) -> Self;
    fn to_dense(&self) -> Vec<Vec<f64>>;
    fn inf_norm(&self) -> f64;
    fn col_inf_norms(&self) -> Vec<f64>;
    fn row_inf_norms(&self) -> Vec<f64>;
}

impl CscExt for CscMatrix {
    fn from_triplets(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> Self {
        from_triplets(m, n, trips)
    }

    fn from_triplets_keep_zeros(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> Self {
        from_triplets_keep_zeros(m, n, trips)
    }

    fn upper_triangle(&self) -> Self {
        assert_eq!(self.m, self.n);
        if self.is_triu() {
            self.clone()
        } else {
            self.to_triu()
        }
    }

    fn same_pattern(&self, other: &Self) -> bool {
        self.m == other.m
            && self.n == other.n
            && self.colptr == other.colptr
            && self.rowval == other.rowval
    }

    fn mul(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.m);
        y.fill(0.0);
        self.mul_add(x, y, 1.0);
    }

    fn mul_add(&self, x: &[f64], y: &mut [f64], alpha: f64) {
        for j in 0..self.n {
            let xj = alpha * x[j];
            if xj == 0.0 {
                continue;
            }
            for p in self.colptr[j]..self.colptr[j + 1] {
                y[self.rowval[p]] += self.nzval[p] * xj;
            }
        }
    }

    fn tmul(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.m);
        assert_eq!(y.len(), self.n);
        for j in 0..self.n {
            let mut s = 0.0;
            for p in self.colptr[j]..self.colptr[j + 1] {
                s += self.nzval[p] * x[self.rowval[p]];
            }
            y[j] = s;
        }
    }

    fn tmul_add(&self, x: &[f64], y: &mut [f64], alpha: f64) {
        for j in 0..self.n {
            let mut s = 0.0;
            for p in self.colptr[j]..self.colptr[j + 1] {
                s += self.nzval[p] * x[self.rowval[p]];
            }
            y[j] += alpha * s;
        }
    }

    fn sym_mul_add(&self, x: &[f64], y: &mut [f64], alpha: f64) {
        debug_assert_eq!(self.m, self.n);
        for j in 0..self.n {
            let xj = x[j];
            for p in self.colptr[j]..self.colptr[j + 1] {
                let i = self.rowval[p];
                let a = alpha * self.nzval[p];
                if i == j {
                    y[i] += a * xj;
                } else {
                    y[i] += a * xj;
                    y[j] += a * x[i];
                }
            }
        }
    }

    fn select_row_indices(&self, rows: &[usize]) -> Self {
        let mut inv = vec![None; self.m];
        for (k, &r) in rows.iter().enumerate() {
            inv[r] = Some(k);
        }
        let mut trips = Vec::new();
        for j in 0..self.n {
            for p in self.colptr[j]..self.colptr[j + 1] {
                if let Some(k) = inv[self.rowval[p]] {
                    trips.push((k, j, self.nzval[p]));
                }
            }
        }
        from_triplets_keep_zeros(rows.len(), self.n, &trips)
    }

    fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; self.n]; self.m];
        for j in 0..self.n {
            for p in self.colptr[j]..self.colptr[j + 1] {
                d[self.rowval[p]][j] = self.nzval[p];
            }
        }
        d
    }

    fn inf_norm(&self) -> f64 {
        self.nzval.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
    }

    fn col_inf_norms(&self) -> Vec<f64> {
        let mut nrm = vec![0.0_f64; self.n];
        for j in 0..self.n {
            for p in self.colptr[j]..self.colptr[j + 1] {
                nrm[j] = nrm[j].max(self.nzval[p].abs());
            }
        }
        nrm
    }

    fn row_inf_norms(&self) -> Vec<f64> {
        let mut nrm = vec![0.0_f64; self.m];
        for p in 0..self.nzval.len() {
            nrm[self.rowval[p]] = nrm[self.rowval[p]].max(self.nzval[p].abs());
        }
        nrm
    }
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn nrm2(a: &[f64]) -> f64 {
    a.norm()
}

#[inline]
pub fn inf_norm(a: &[f64]) -> f64 {
    a.norm_inf()
}

#[inline]
pub fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[inline]
pub fn scale(x: &mut [f64], a: f64) {
    for xi in x {
        *xi *= a;
    }
}

#[inline]
pub fn copy_from(dst: &mut [f64], src: &[f64]) {
    dst.copy_from_slice(src);
}

/// Push every nonzero index of `k` into `ldl` and numerically refactor.
pub fn qdldl_sync_values(ldl: &mut QDLDLFactorisation<f64>, k: &CscMatrix) -> Result<(), String> {
    let n = k.nnz();
    let indices: Vec<usize> = (0..n).collect();
    ldl.update_values(&indices, &k.nzval);
    ldl.refactor().map_err(|e| e.to_string())
}

/// Factor a quasi-definite upper-triangular KKT matrix with Clarabel QDLDL.
pub fn qdldl_factor_qd(
    k: &CscMatrix,
    n_pos: usize,
    regularize_eps: f64,
) -> Result<QDLDLFactorisation<f64>, String> {
    let ntot = k.n;
    let mut dsigns = vec![1i8; ntot];
    for s in dsigns.iter_mut().skip(n_pos) {
        *s = -1;
    }
    let opts = QDLDLSettingsBuilder::<f64>::default()
        .Dsigns(dsigns)
        .regularize_enable(true)
        .regularize_eps(regularize_eps)
        .regularize_delta(regularize_eps.max(1e-14))
        // Clarabel's own KKT path uses 1.5; the QDLDL default of 1.0 gave
        // poorer AMD orderings on larger QP/CVaR KKTs in practice.
        .amd_dense_scale(1.5)
        .build()
        .map_err(|e| e.to_string())?;
    QDLDLFactorisation::new(k, Some(opts)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_identity() {
        let a = CscMatrix::identity(3);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        a.mul(&x, &mut y);
        assert_eq!(y, x);
        let mut z = [0.0; 3];
        a.tmul(&x, &mut z);
        assert_eq!(z, x);
    }

    #[test]
    fn symv_triu() {
        let p = clarabel::algebra::CscMatrix::<f64>::from(&[[4.0, 1.0], [0.0, 2.0]]);
        let x = [1.0, 2.0];
        let mut y = [0.0; 2];
        p.sym_mul_add(&x, &mut y, 1.0);
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn qdldl_solve_small() {
        let k = from_triplets(
            2,
            2,
            &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0)],
        );
        let mut ldl = qdldl_factor_qd(&k, 2, 1e-14).unwrap();
        let mut b = vec![1.0, 2.0];
        ldl.solve(&mut b);
        // [4,1;1,3]^{-1} [1,2] = (1/11)[1,7]
        assert!((b[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((b[1] - 7.0 / 11.0).abs() < 1e-10);
    }
}
