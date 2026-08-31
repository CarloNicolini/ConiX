//! Compressed sparse column matrices.

#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix {
    pub m: usize,
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub x: Vec<f64>,
}

impl CscMatrix {
    pub fn zeros(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            col_ptr: vec![0; n + 1],
            row_idx: Vec::new(),
            x: Vec::new(),
        }
    }

    pub fn identity(n: usize) -> Self {
        Self {
            m: n,
            n,
            col_ptr: (0..=n).collect(),
            row_idx: (0..n).collect(),
            x: vec![1.0; n],
        }
    }

    pub fn nnz(&self) -> usize {
        self.x.len()
    }

    /// Build from triplets `(row, col, value)`, summing duplicates.
    /// Explicit zeros are dropped unless `keep_zeros` is set (needed for R1 update slots).
    pub fn from_triplets(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> Self {
        Self::from_triplets_ex(m, n, trips, false)
    }

    pub fn from_triplets_keep_zeros(m: usize, n: usize, trips: &[(usize, usize, f64)]) -> Self {
        Self::from_triplets_ex(m, n, trips, true)
    }

    fn from_triplets_ex(
        m: usize,
        n: usize,
        trips: &[(usize, usize, f64)],
        keep_zeros: bool,
    ) -> Self {
        let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(i, j, v) in trips {
            if i >= m || j >= n {
                panic!("triplet ({i},{j}) out of bounds {m}x{n}");
            }
            if keep_zeros || v != 0.0 {
                buckets[j].push((i, v));
            }
        }
        let mut col_ptr = vec![0; n + 1];
        let mut row_idx = Vec::new();
        let mut x = Vec::new();
        for j in 0..n {
            let mut col = std::mem::take(&mut buckets[j]);
            col.sort_by_key(|t| t.0);
            let mut k = 0;
            while k < col.len() {
                let row = col[k].0;
                let mut acc = col[k].1;
                k += 1;
                while k < col.len() && col[k].0 == row {
                    acc += col[k].1;
                    k += 1;
                }
                if keep_zeros || acc != 0.0 {
                    row_idx.push(row);
                    x.push(acc);
                }
            }
            col_ptr[j + 1] = row_idx.len();
        }
        Self {
            m,
            n,
            col_ptr,
            row_idx,
            x,
        }
    }

    pub fn from_dense(data: &[Vec<f64>]) -> Self {
        let m = data.len();
        let n = if m == 0 { 0 } else { data[0].len() };
        let mut trips = Vec::new();
        for (i, row) in data.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    trips.push((i, j, v));
                }
            }
        }
        Self::from_triplets(m, n, &trips)
    }

    /// Keep the upper triangle (including diagonal) of a square matrix.
    pub fn upper_triangle(&self) -> Self {
        assert_eq!(self.m, self.n);
        let mut trips = Vec::new();
        for j in 0..self.n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                if i <= j {
                    trips.push((i, j, self.x[p]));
                } else {
                    trips.push((j, i, self.x[p]));
                }
            }
        }
        Self::from_triplets(self.n, self.n, &trips)
    }

    pub fn mul(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.m);
        y.fill(0.0);
        self.mul_add(x, y, 1.0);
    }

    pub fn mul_add(&self, x: &[f64], y: &mut [f64], alpha: f64) {
        for j in 0..self.n {
            let xj = alpha * x[j];
            if xj == 0.0 {
                continue;
            }
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                y[self.row_idx[p]] += self.x[p] * xj;
            }
        }
    }

    pub fn tmul(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.m);
        assert_eq!(y.len(), self.n);
        for j in 0..self.n {
            let mut s = 0.0;
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                s += self.x[p] * x[self.row_idx[p]];
            }
            y[j] = s;
        }
    }

    pub fn tmul_add(&self, x: &[f64], y: &mut [f64], alpha: f64) {
        for j in 0..self.n {
            let mut s = 0.0;
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                s += self.x[p] * x[self.row_idx[p]];
            }
            y[j] += alpha * s;
        }
    }

    /// Symmetric multiply using only the upper triangle: y = alpha * A * x + y.
    pub fn sym_mul_add(&self, x: &[f64], y: &mut [f64], alpha: f64) {
        debug_assert_eq!(self.m, self.n);
        for j in 0..self.n {
            let xj = x[j];
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                let a = alpha * self.x[p];
                if i == j {
                    y[i] += a * xj;
                } else {
                    y[i] += a * xj;
                    y[j] += a * x[i];
                }
            }
        }
    }

    pub fn same_pattern(&self, other: &Self) -> bool {
        self.m == other.m
            && self.n == other.n
            && self.col_ptr == other.col_ptr
            && self.row_idx == other.row_idx
    }

    /// Extract a subset of rows, preserving column order and explicit zeros.
    pub fn select_rows(&self, rows: &[usize]) -> Self {
        let mut inv = vec![None; self.m];
        for (k, &r) in rows.iter().enumerate() {
            inv[r] = Some(k);
        }
        let mut trips = Vec::new();
        for j in 0..self.n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                if let Some(k) = inv[self.row_idx[p]] {
                    trips.push((k, j, self.x[p]));
                }
            }
        }
        Self::from_triplets_keep_zeros(rows.len(), self.n, &trips)
    }

    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; self.n]; self.m];
        for j in 0..self.n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                d[self.row_idx[p]][j] = self.x[p];
            }
        }
        d
    }

    pub fn inf_norm(&self) -> f64 {
        self.x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
    }

    pub fn col_inf_norms(&self) -> Vec<f64> {
        let mut nrm = vec![0.0_f64; self.n];
        for j in 0..self.n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                nrm[j] = nrm[j].max(self.x[p].abs());
            }
        }
        nrm
    }

    pub fn row_inf_norms(&self) -> Vec<f64> {
        let mut nrm = vec![0.0_f64; self.m];
        for p in 0..self.x.len() {
            nrm[self.row_idx[p]] = nrm[self.row_idx[p]].max(self.x[p].abs());
        }
        nrm
    }

    /// Symmetric permutation of an upper-triangular square CSC: B = P A P^T.
    pub fn permute_sym_upper(&self, perm: &[usize]) -> Self {
        assert_eq!(self.m, self.n);
        assert_eq!(perm.len(), self.n);
        let mut pinv = vec![0; self.n];
        for (k, &i) in perm.iter().enumerate() {
            pinv[i] = k;
        }
        let mut trips = Vec::with_capacity(self.nnz());
        for j in 0..self.n {
            let pj = pinv[j];
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p];
                let pi = pinv[i];
                let (r, c) = if pi <= pj { (pi, pj) } else { (pj, pi) };
                trips.push((r, c, self.x[p]));
            }
        }
        Self::from_triplets_keep_zeros(self.n, self.n, &trips)
    }
}

/// y = P x for a permutation vector `perm` (new[k] = old[perm[k]]).
pub fn permute(perm: &[usize], x: &[f64], y: &mut [f64]) {
    for k in 0..perm.len() {
        y[k] = x[perm[k]];
    }
}

/// x = P^T y
pub fn inv_permute(perm: &[usize], y: &[f64], x: &mut [f64]) {
    for k in 0..perm.len() {
        x[perm[k]] = y[k];
    }
}
