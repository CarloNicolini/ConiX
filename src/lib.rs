//! ConiX: sequential conic optimizer.
//!
//! Sparse CSC storage and QDLDL factorisation are reused from Clarabel.rs
//! (same numerical substrate as COSMO.rs). The ConiX hybrid ADMM / DR / IPM
//! algorithms are independent of that substrate.

pub mod algebra;
pub mod capi;
pub mod cones;
pub mod engines;
pub mod ipm_kkt;
pub mod kkt;
pub mod models;
pub mod scale;
pub mod settings;
pub mod status;
pub mod verifier;
pub mod workspace;

pub use algebra::{CscExt, CscMatrix};
pub use cones::{CompositeCone, Cone};
pub use settings::{EngineKind, Settings};
pub use status::{Solution, SolveInfo, Status, UpdateClass};
pub use workspace::{solve, Qcp, Workspace};

pub fn setup(problem: Qcp, settings: Settings) -> Result<Workspace, String> {
    Workspace::setup(problem, settings)
}

pub fn solve_once(problem: Qcp, settings: Settings) -> Result<Solution, String> {
    let mut ws = Workspace::setup(problem, settings)?;
    Ok(solve(&mut ws))
}

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _conix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
