//! Sparse cone-block KKT for the homogeneous IPM, factored with Clarabel QDLDL.
//!
//! \[
//! K = \begin{bmatrix} P+\sigma I & A^\top \\ A & -H_s \end{bmatrix}
//! \]
//!
//! \(H_s\) is block-diagonal across cones: a diagonal for Zero/NN, and a packed
//! upper triangle for SOC / exp / power / genpower / PSD. Clarabel AMD order and
//! the QDLDL symbolic factor stay valid across Newton steps and across sequential
//! R1 updates that keep the \((P,A)\) pattern. The ConiX IPM algorithm that
//! drives this linear system is unchanged.

use crate::algebra::{qdldl_factor_qd, CscExt, CscMatrix, QDLDLFactorisation};
use crate::cones::CompositeCone;

const SIGMA: f64 = 1e-10;
const HS_REG: f64 = 1e-12;

#[derive(Clone, Debug)]
enum HsBlock {
    Diagonal {
        off: usize,
        indices: Vec<usize>,
    },
    DenseUpper {
        off: usize,
        dim: usize,
        indices: Vec<usize>,
    },
}

pub struct IpmKkt {
    pub n: usize,
    pub m: usize,
    k_upper: CscMatrix,
    #[allow(dead_code)]
    p_map: Vec<usize>,
    #[allow(dead_code)]
    a_map: Vec<usize>,
    #[allow(dead_code)]
    diag_p: Vec<usize>,
    hs_blocks: Vec<HsBlock>,
    packed_len: usize,
    packed: Vec<f64>,
    ldl: QDLDLFactorisation<f64>,
    work: Vec<f64>,
    work2: Vec<f64>,
    rhs: Vec<f64>,
    sol: Vec<f64>,
    pub x2: Vec<f64>,
    pub z2: Vec<f64>,
}

impl IpmKkt {
    pub fn analyze(
        p: &CscMatrix,
        a: &CscMatrix,
        cones: &CompositeCone,
    ) -> Result<Self, String> {
        let n = p.n;
        let m = a.m;
        assert_eq!(p.m, n);
        assert_eq!(a.n, n);
        assert_eq!(cones.dim, m);

        let packed_len: usize = cones.cones.iter().map(|c| c.hs_packed_len()).sum();
        let packed = vec![0.0; packed_len];
        let (k_upper, p_map, a_map, diag_p, hs_blocks) = assemble(p, a, cones, &packed);
        let ldl = qdldl_factor_qd(&k_upper, n, 1e-12)?;
        let dim = n + m;
        Ok(Self {
            n,
            m,
            k_upper,
            p_map,
            a_map,
            diag_p,
            hs_blocks,
            packed_len,
            packed,
            ldl,
            work: vec![0.0; dim],
            work2: vec![0.0; dim],
            rhs: vec![0.0; dim],
            sol: vec![0.0; dim],
            x2: vec![0.0; n],
            z2: vec![0.0; m],
        })
    }

    pub fn packed_len(&self) -> usize {
        self.packed_len
    }

    pub fn k_nnz(&self) -> usize {
        self.k_upper.nnz()
    }

    fn refactor(&mut self) -> Result<(), String> {
        // Clarabel-style static diagonal regularization before numeric refactor:
        // shift diag by ±ε according to Dsigns so quasi-definite pivots stay
        // away from zero; restore the unshifted K for residual matvecs / IR.
        let ntot = self.n + self.m;
        let mut diag_idx = Vec::with_capacity(ntot);
        let mut diag_true = Vec::with_capacity(ntot);
        let mut diag_shift = Vec::with_capacity(ntot);
        for j in 0..ntot {
            let p = find_diag(&self.k_upper, j);
            let v = self.k_upper.nzval[p];
            diag_idx.push(p);
            diag_true.push(v);
            let sign = if j < self.n { 1.0 } else { -1.0 };
            diag_shift.push(v + sign * 1e-10_f64.max(1e-8 * v.abs()));
        }
        self.ldl.update_values(&diag_idx, &diag_shift);
        // Also sync any off-diagonal Hs / PA values already written into k_upper.
        let all: Vec<usize> = (0..self.k_upper.nnz()).collect();
        self.ldl.update_values(&all, &self.k_upper.nzval);
        // Re-apply the shifted diagonal on top of the full sync.
        self.ldl.update_values(&diag_idx, &diag_shift);
        self.ldl.refactor().map_err(|e| e.to_string())?;
        // Restore unshifted diagonal in the kept K (IR matvecs use this copy).
        for (&p, &v) in diag_idx.iter().zip(diag_true.iter()) {
            self.k_upper.nzval[p] = v;
        }
        Ok(())
    }

    /// Write packed \(H_s\) into the lower-right blocks, refactor.
    pub fn update_hs(&mut self, packed: &[f64]) -> Result<(), String> {
        assert_eq!(packed.len(), self.packed_len);
        self.packed.copy_from_slice(packed);
        write_hs_into(&mut self.k_upper, &self.hs_blocks, packed);
        self.refactor()
    }

    pub fn update_pa(&mut self, p: &CscMatrix, a: &CscMatrix) -> Result<(), String> {
        let (k_upper, p_map, a_map, diag_p, hs_blocks) =
            assemble_with_blocks(p, a, &self.hs_blocks, &self.packed);
        if !k_upper.same_pattern(&self.k_upper) {
            return Err("IPM KKT pattern changed (R2)".into());
        }
        self.k_upper = k_upper;
        self.p_map = p_map;
        self.a_map = a_map;
        self.diag_p = diag_p;
        self.hs_blocks = hs_blocks;
        self.refactor()
    }

    pub fn solve(&mut self, rhs: &[f64], sol: &mut [f64], refinement: usize) {
        let ntot = self.n + self.m;
        sol.copy_from_slice(rhs);
        self.ldl.solve(sol);
        for _ in 0..refinement {
            self.work2.fill(0.0);
            self.k_upper.sym_mul_add(sol, &mut self.work2, 1.0);
            for i in 0..ntot {
                self.work[i] = rhs[i] - self.work2[i];
            }
            self.ldl.solve(&mut self.work);
            for i in 0..sol.len() {
                sol[i] += self.work[i];
            }
        }
    }

    pub fn solve_split(
        &mut self,
        rhs_x: &[f64],
        rhs_z: &[f64],
        x: &mut [f64],
        z: &mut [f64],
        refinement: usize,
    ) {
        let n = self.n;
        let m = self.m;
        self.rhs[..n].copy_from_slice(rhs_x);
        self.rhs[n..n + m].copy_from_slice(rhs_z);
        self.sol.fill(0.0);
        for i in 0..n + m {
            self.work2[i] = self.rhs[i];
        }
        self.solve_into(refinement);
        x.copy_from_slice(&self.sol[..n]);
        z.copy_from_slice(&self.sol[n..]);
    }

    fn solve_into(&mut self, refinement: usize) {
        let ntot = self.n + self.m;
        self.sol[..ntot].copy_from_slice(&self.work2[..ntot]);
        self.ldl.solve(&mut self.sol[..ntot]);
        for _ in 0..refinement {
            self.rhs.fill(0.0);
            self.k_upper.sym_mul_add(&self.sol, &mut self.rhs, 1.0);
            for i in 0..ntot {
                self.work[i] = self.work2[i] - self.rhs[i];
            }
            self.ldl.solve(&mut self.work[..ntot]);
            for i in 0..ntot {
                self.sol[i] += self.work[i];
            }
        }
    }

    /// \(K\begin{bmatrix}x_2\\z_2\end{bmatrix}=\begin{bmatrix}-q\\b\end{bmatrix}\).
    pub fn solve_constant(&mut self, q: &[f64], b: &[f64], refinement: usize) {
        for i in 0..self.n {
            self.work2[i] = -q[i];
        }
        self.work2[self.n..].copy_from_slice(b);
        self.sol.fill(0.0);
        self.solve_into(refinement);
        self.x2.copy_from_slice(&self.sol[..self.n]);
        self.z2.copy_from_slice(&self.sol[self.n..]);
    }

    pub fn mul_hs(&self, x: &[f64], y: &mut [f64]) {
        mul_hs_packed(&self.hs_blocks, &self.packed, x, y);
    }
}

fn assemble(
    p: &CscMatrix,
    a: &CscMatrix,
    cones: &CompositeCone,
    packed: &[f64],
) -> (
    CscMatrix,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<HsBlock>,
) {
    let blocks = hs_block_placeholders(cones);
    assemble_with_blocks(p, a, &blocks, packed)
}

fn hs_block_placeholders(cones: &CompositeCone) -> Vec<HsBlock> {
    cones
        .cones
        .iter()
        .zip(&cones.offsets)
        .map(|(cone, &off)| {
            let d = cone.dim();
            if cone.hs_is_diagonal() {
                HsBlock::Diagonal {
                    off,
                    indices: vec![0; d],
                }
            } else {
                HsBlock::DenseUpper {
                    off,
                    dim: d,
                    indices: vec![0; d * (d + 1) / 2],
                }
            }
        })
        .collect()
}

fn assemble_with_blocks(
    p: &CscMatrix,
    a: &CscMatrix,
    blocks: &[HsBlock],
    packed: &[f64],
) -> (
    CscMatrix,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<HsBlock>,
) {
    let n = p.n;
    let m = a.m;
    let ntot = n + m;
    let pu = p.upper_triangle();
    let mut trips: Vec<(usize, usize, f64)> = Vec::new();

    let mut has_diag = vec![false; n];
    for j in 0..n {
        for idx in pu.colptr[j]..pu.colptr[j + 1] {
            let i = pu.rowval[idx];
            let mut v = pu.nzval[idx];
            if i == j {
                v += SIGMA;
                has_diag[j] = true;
            }
            trips.push((i, j, v));
        }
    }
    for j in 0..n {
        if !has_diag[j] {
            trips.push((j, j, SIGMA));
        }
    }

    for c in 0..n {
        for idx in a.colptr[c]..a.colptr[c + 1] {
            let r = a.rowval[idx];
            trips.push((c, n + r, a.nzval[idx]));
        }
    }

    let mut po = 0usize;
    for block in blocks {
        match block {
            HsBlock::Diagonal { off, indices } => {
                for k in 0..indices.len() {
                    let h = packed.get(po + k).copied().unwrap_or(0.0);
                    trips.push((n + off + k, n + off + k, -h - HS_REG));
                }
                po += indices.len();
            }
            HsBlock::DenseUpper { off, dim, indices } => {
                let mut t = 0usize;
                for j in 0..*dim {
                    for i in 0..=j {
                        let h = packed.get(po + t).copied().unwrap_or(0.0);
                        let v = if i == j { -h - HS_REG } else { -h };
                        trips.push((n + off + i, n + off + j, v));
                        t += 1;
                    }
                }
                po += indices.len();
            }
        }
    }

    let k = CscMatrix::from_triplets_keep_zeros(ntot, ntot, &trips);

    let mut p_map = Vec::with_capacity(pu.nnz());
    for j in 0..n {
        for idx in pu.colptr[j]..pu.colptr[j + 1] {
            p_map.push(find_entry(&k, pu.rowval[idx], j));
        }
    }
    let mut a_map = Vec::with_capacity(a.nnz());
    for c in 0..n {
        for idx in a.colptr[c]..a.colptr[c + 1] {
            let r = a.rowval[idx];
            a_map.push(find_entry(&k, c, n + r));
        }
    }
    let mut diag_p = vec![0; n];
    for j in 0..n {
        diag_p[j] = find_entry(&k, j, j);
    }

    let mut out_blocks = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            HsBlock::Diagonal { off, indices } => {
                let mut idx = Vec::with_capacity(indices.len());
                for ki in 0..indices.len() {
                    idx.push(find_entry(&k, n + off + ki, n + off + ki));
                }
                out_blocks.push(HsBlock::Diagonal {
                    off: *off,
                    indices: idx,
                });
            }
            HsBlock::DenseUpper { off, dim, .. } => {
                let mut idx = Vec::with_capacity(*dim * (*dim + 1) / 2);
                for j in 0..*dim {
                    for i in 0..=j {
                        idx.push(find_entry(&k, n + off + i, n + off + j));
                    }
                }
                out_blocks.push(HsBlock::DenseUpper {
                    off: *off,
                    dim: *dim,
                    indices: idx,
                });
            }
        }
    }

    (k, p_map, a_map, diag_p, out_blocks)
}

fn write_hs_into(k: &mut CscMatrix, blocks: &[HsBlock], packed: &[f64]) {
    let mut po = 0usize;
    for block in blocks {
        match block {
            HsBlock::Diagonal { indices, .. } => {
                for (k_i, &slot) in indices.iter().enumerate() {
                    k.nzval[slot] = -packed[po + k_i] - HS_REG;
                }
                po += indices.len();
            }
            HsBlock::DenseUpper { dim, indices, .. } => {
                let mut t = 0usize;
                for j in 0..*dim {
                    for i in 0..=j {
                        let h = packed[po + t];
                        k.nzval[indices[t]] = if i == j { -h - HS_REG } else { -h };
                        t += 1;
                    }
                }
                po += indices.len();
            }
        }
    }
}

fn mul_hs_packed(blocks: &[HsBlock], packed: &[f64], x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    let mut po = 0usize;
    for block in blocks {
        match block {
            HsBlock::Diagonal { off, indices } => {
                for k in 0..indices.len() {
                    y[off + k] = packed[po + k] * x[off + k];
                }
                po += indices.len();
            }
            HsBlock::DenseUpper { off, dim, indices } => {
                let mut t = 0usize;
                for j in 0..*dim {
                    for i in 0..=j {
                        let v = packed[po + t];
                        y[off + i] += v * x[off + j];
                        if i != j {
                            y[off + j] += v * x[off + i];
                        }
                        t += 1;
                    }
                }
                po += indices.len();
            }
        }
    }
}

fn find_entry(k: &CscMatrix, row: usize, col: usize) -> usize {
    for p in k.colptr[col]..k.colptr[col + 1] {
        if k.rowval[p] == row {
            return p;
        }
    }
    panic!("IPM KKT entry ({row},{col}) missing");
}

fn find_diag(k: &CscMatrix, col: usize) -> usize {
    find_entry(k, col, col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::CscMatrix;
    use crate::cones::{CompositeCone, Cone};

    #[test]
    fn exp_block_is_not_dense_m() {
        let p = CscMatrix::zeros((1, 1));
        let a = CscMatrix::from_triplets(6, 1, &[(0, 0, 1.0), (3, 0, -1.0)]);
        let cones = CompositeCone::new(vec![Cone::Nonnegative { dim: 3 }, Cone::Exponential]);
        let k = IpmKkt::analyze(&p, &a, &cones).unwrap();
        let dense_hs = 6 * 7 / 2;
        assert!(
            k.k_nnz() < 1 + 2 + dense_hs,
            "nnz={} should be cone-block, not dense Hs",
            k.k_nnz()
        );
        assert_eq!(k.packed_len(), 3 + 6);
    }

    #[test]
    fn nn_only_diagonal_hs() {
        let p = CscMatrix::identity(2);
        let a = CscMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let cones = CompositeCone::new(vec![Cone::Nonnegative { dim: 2 }]);
        let k = IpmKkt::analyze(&p, &a, &cones).unwrap();
        assert_eq!(k.packed_len(), 2);
    }
}
