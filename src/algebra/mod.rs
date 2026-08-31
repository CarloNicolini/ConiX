pub mod amd;
pub mod csc;
pub mod ldl;

pub use csc::{inv_permute, permute, CscMatrix};
pub use ldl::{LdlNumeric, LdlSymbolic};

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn nrm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
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
