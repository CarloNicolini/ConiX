//! Quasi-definite KKT assembly with Clarabel QDLDL (AMD + LDLᵀ).
//!
//! K = [ P+σI , A' ]
//!     [ A    , -R ]
//! stored as the upper triangle. Factorisation, AMD ordering, and numeric
//! refactorisation come from Clarabel's QDLDL — the same substrate as COSMO.rs.
//! The ConiX ADMM / splitting iteration that *calls* this solver is unchanged.

use crate::algebra::{qdldl_factor_qd, qdldl_sync_values, CscExt, CscMatrix, QDLDLFactorisation};

#[derive(Debug)]
pub struct KktSystem {
    pub n: usize,
    pub m: usize,
    pub sigma: f64,
    pub k_upper: CscMatrix,
    pub p_map: Vec<usize>,
    pub a_map: Vec<usize>,
    pub diag_map: Vec<usize>,
    pub rho_map: Vec<usize>,
    ldl: QDLDLFactorisation<f64>,
    work: Vec<f64>,
    work2: Vec<f64>,
}

impl KktSystem {
    pub fn analyze(
        p: &CscMatrix,
        a: &CscMatrix,
        sigma: f64,
        rho: &[f64],
    ) -> Result<Self, String> {
        let n = p.n;
        let m = a.m;
        assert_eq!(p.m, n);
        assert_eq!(a.n, n);
        assert_eq!(rho.len(), m);

        let pu = p.upper_triangle();
        let (k_upper, p_map, a_map, diag_map, rho_map) = assemble_upper(&pu, a, sigma, rho);
        let ldl = qdldl_factor_qd(&k_upper, n, 1e-14)?;
        Ok(Self {
            n,
            m,
            sigma,
            k_upper,
            p_map,
            a_map,
            diag_map,
            rho_map,
            ldl,
            work: vec![0.0; n + m],
            work2: vec![0.0; n + m],
        })
    }

    pub fn refactor(&mut self) -> Result<(), String> {
        qdldl_sync_values(&mut self.ldl, &self.k_upper)
    }

    pub fn solve(&mut self, rhs: &[f64], sol: &mut [f64], refinement: usize) {
        let ntot = self.n + self.m;
        assert_eq!(rhs.len(), ntot);
        sol.copy_from_slice(rhs);
        self.ldl.solve(sol);
        for _ in 0..refinement {
            kkt_mul(&self.k_upper, sol, &mut self.work2);
            for i in 0..ntot {
                self.work[i] = rhs[i] - self.work2[i];
            }
            self.ldl.solve(&mut self.work);
            for i in 0..ntot {
                sol[i] += self.work[i];
            }
        }
    }

    pub fn update_vectors_only(&self) {
        // numeric factor stays valid for R0
    }

    pub fn update_rho(&mut self, rho: &[f64]) -> Result<(), String> {
        self.update_nt(self.sigma, rho)
    }

    /// Rewrite the quasi-definite diagonals in place: \(P+\sigma I\) and \(-1/\rho\).
    /// Pattern and Clarabel AMD order stay valid (polyhedral NT-IPM).
    pub fn update_nt(&mut self, sigma: f64, rho: &[f64]) -> Result<(), String> {
        let ds = sigma - self.sigma;
        if ds.abs() > 0.0 {
            for j in 0..self.n {
                self.k_upper.nzval[self.diag_map[j]] += ds;
            }
            self.sigma = sigma;
        }
        for i in 0..self.m {
            self.k_upper.nzval[self.rho_map[i]] = -1.0 / rho[i];
        }
        // Only the changed diagonal entries need syncing.
        let mut idx = Vec::with_capacity(self.n + self.m);
        let mut vals = Vec::with_capacity(self.n + self.m);
        for j in 0..self.n {
            let p = self.diag_map[j];
            idx.push(p);
            vals.push(self.k_upper.nzval[p]);
        }
        for i in 0..self.m {
            let p = self.rho_map[i];
            idx.push(p);
            vals.push(self.k_upper.nzval[p]);
        }
        self.ldl.update_values(&idx, &vals);
        self.ldl.refactor().map_err(|e| e.to_string())
    }

    pub fn update_pa(
        &mut self,
        p: &CscMatrix,
        a: &CscMatrix,
        sigma: f64,
        rho: &[f64],
    ) -> Result<(), String> {
        let pu = p.upper_triangle();
        let (k_upper, p_map, a_map, diag_map, rho_map) = assemble_upper(&pu, a, sigma, rho);
        if !k_upper.same_pattern(&self.k_upper) {
            return Err("sparsity pattern changed (R2); re-analyze".into());
        }
        self.k_upper = k_upper;
        self.p_map = p_map;
        self.a_map = a_map;
        self.diag_map = diag_map;
        self.rho_map = rho_map;
        self.sigma = sigma;
        self.refactor()
    }
}

fn kkt_mul(k_upper: &CscMatrix, x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    k_upper.sym_mul_add(x, y, 1.0);
}

fn assemble_upper(
    p_upper: &CscMatrix,
    a: &CscMatrix,
    sigma: f64,
    rho: &[f64],
) -> (
    CscMatrix,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
) {
    let n = p_upper.n;
    let m = a.m;
    let ntot = n + m;
    let mut trips: Vec<(usize, usize, f64)> = Vec::new();

    let mut has_diag = vec![false; n];
    for j in 0..n {
        for p in p_upper.colptr[j]..p_upper.colptr[j + 1] {
            let i = p_upper.rowval[p];
            let mut v = p_upper.nzval[p];
            if i == j {
                v += sigma;
                has_diag[j] = true;
            }
            trips.push((i, j, v));
        }
    }
    for j in 0..n {
        if !has_diag[j] {
            trips.push((j, j, sigma));
        }
    }

    for c in 0..n {
        for p in a.colptr[c]..a.colptr[c + 1] {
            let r = a.rowval[p];
            trips.push((c, n + r, a.nzval[p]));
        }
    }

    for r in 0..m {
        trips.push((n + r, n + r, -1.0 / rho[r]));
    }

    let k = CscMatrix::from_triplets_keep_zeros(ntot, ntot, &trips);

    let mut p_map = Vec::with_capacity(p_upper.nnz());
    for j in 0..n {
        for p in p_upper.colptr[j]..p_upper.colptr[j + 1] {
            let i = p_upper.rowval[p];
            p_map.push(find_entry(&k, i, j));
        }
    }
    let mut a_map = Vec::with_capacity(a.nnz());
    for c in 0..n {
        for p in a.colptr[c]..a.colptr[c + 1] {
            let r = a.rowval[p];
            a_map.push(find_entry(&k, c, n + r));
        }
    }
    let mut diag_map = vec![0; n];
    for j in 0..n {
        diag_map[j] = find_entry(&k, j, j);
    }
    let mut rho_map = vec![0; m];
    for r in 0..m {
        rho_map[r] = find_entry(&k, n + r, n + r);
    }
    (k, p_map, a_map, diag_map, rho_map)
}

fn find_entry(k: &CscMatrix, row: usize, col: usize) -> usize {
    for p in k.colptr[col]..k.colptr[col + 1] {
        if k.rowval[p] == row {
            return p;
        }
    }
    panic!("KKT entry ({row},{col}) missing");
}
