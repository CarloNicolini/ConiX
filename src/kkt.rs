//! Quasi-definite KKT assembly, symbolic analysis, and cached numeric factors.

use crate::algebra::amd::order_upper;
use crate::algebra::csc::{inv_permute, permute, CscMatrix};
use crate::algebra::ldl::{LdlNumeric, LdlSymbolic};

/// K = [ P+σI , A' ]
///     [ A    , -R ]
/// stored as the upper triangle, then AMD-permuted.
#[derive(Clone, Debug)]
pub struct KktSystem {
    pub n: usize,
    pub m: usize,
    pub sigma: f64,
    pub perm: Vec<usize>,
    pub k_upper: CscMatrix,
    pub k_perm: CscMatrix,
    pub p_map: Vec<usize>,
    pub a_map: Vec<usize>,
    pub diag_map: Vec<usize>,
    pub rho_map: Vec<usize>,
    pub sym: LdlSymbolic,
    pub fac: Option<LdlNumeric>,
    work: Vec<f64>,
    work2: Vec<f64>,
}

impl KktSystem {
    pub fn analyze(p: &CscMatrix, a: &CscMatrix, sigma: f64, rho: &[f64]) -> Result<Self, String> {
        let n = p.n;
        let m = a.m;
        assert_eq!(p.m, n);
        assert_eq!(a.n, n);
        assert_eq!(rho.len(), m);

        let pu = p.upper_triangle();
        let (k_upper, p_map, a_map, diag_map, rho_map) = assemble_upper(&pu, a, sigma, rho);
        let perm = order_upper(&k_upper);
        let k_perm = k_upper.permute_sym_upper(&perm);
        let sym = LdlSymbolic::analyze(&k_perm).map_err(|e| e.msg.to_string())?;
        let mut sys = Self {
            n,
            m,
            sigma,
            perm,
            k_upper,
            k_perm,
            p_map,
            a_map,
            diag_map,
            rho_map,
            sym,
            fac: None,
            work: vec![0.0; n + m],
            work2: vec![0.0; n + m],
        };
        sys.refactor()?;
        Ok(sys)
    }

    pub fn refactor(&mut self) -> Result<(), String> {
        let fac = LdlNumeric::factor_regularized(&self.k_perm, &self.sym, self.n, 1e-14)
            .or_else(|_| LdlNumeric::factor(&self.k_perm, &self.sym))
            .map_err(|e| e.msg.to_string())?;
        self.fac = Some(fac);
        Ok(())
    }

    pub fn solve(&mut self, rhs: &[f64], sol: &mut [f64], refinement: usize) {
        let ntot = self.n + self.m;
        assert_eq!(rhs.len(), ntot);
        permute(&self.perm, rhs, &mut self.work);
        self.fac.as_ref().unwrap().solve_in_place(&mut self.work);
        inv_permute(&self.perm, &self.work, sol);
        for _ in 0..refinement {
            // r = K sol - rhs, using unpermuted upper triangle
            kkt_mul(&self.k_upper, sol, &mut self.work2);
            for i in 0..ntot {
                self.work2[i] = rhs[i] - self.work2[i];
            }
            permute(&self.perm, &self.work2, &mut self.work);
            self.fac.as_ref().unwrap().solve_in_place(&mut self.work);
            inv_permute(&self.perm, &self.work, &mut self.work2);
            for i in 0..ntot {
                sol[i] += self.work2[i];
            }
        }
    }

    pub fn update_vectors_only(&self) {
        // numeric factor stays valid for R0
    }

    pub fn update_rho(&mut self, rho: &[f64]) -> Result<(), String> {
        for i in 0..self.m {
            let val = -1.0 / rho[i];
            self.k_upper.x[self.rho_map[i]] = val;
        }
        self.k_perm = self.k_upper.permute_sym_upper(&self.perm);
        self.refactor()
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
        self.k_perm = self.k_upper.permute_sym_upper(&self.perm);
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
) -> (CscMatrix, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = p_upper.n;
    let m = a.m;
    let ntot = n + m;
    let mut trips: Vec<(usize, usize, f64)> = Vec::new();

    // P + σI
    let mut has_diag = vec![false; n];
    for j in 0..n {
        for p in p_upper.col_ptr[j]..p_upper.col_ptr[j + 1] {
            let i = p_upper.row_idx[p];
            let mut v = p_upper.x[p];
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

    // A' in the (1,2) block: column n+r, row c, value A[r,c]
    for c in 0..n {
        for p in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[p];
            trips.push((c, n + r, a.x[p]));
        }
    }

    // -1/ρ on the (2,2) diagonal
    for r in 0..m {
        trips.push((n + r, n + r, -1.0 / rho[r]));
    }

    let k = CscMatrix::from_triplets_keep_zeros(ntot, ntot, &trips);

    // Maps: locate stored values. For P we map each upper nnz onto K.
    let mut p_map = Vec::with_capacity(p_upper.nnz());
    for j in 0..n {
        for p in p_upper.col_ptr[j]..p_upper.col_ptr[j + 1] {
            let i = p_upper.row_idx[p];
            p_map.push(find_entry(&k, i, j));
        }
    }
    let mut a_map = Vec::with_capacity(a.nnz());
    for c in 0..n {
        for p in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[p];
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
    for p in k.col_ptr[col]..k.col_ptr[col + 1] {
        if k.row_idx[p] == row {
            return p;
        }
    }
    panic!("KKT entry ({row},{col}) missing");
}
