//! ConiX: sequential conic optimizer.

pub mod algebra;
pub mod capi;
pub mod cones;
pub mod engines;
pub mod kkt;
pub mod models;
pub mod scale;
pub mod settings;
pub mod status;
pub mod verifier;
pub mod workspace;

pub use algebra::CscMatrix;
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
