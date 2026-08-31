#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Unsolved,
    Solved,
    MaxIters,
    PrimalInfeasible,
    DualInfeasible,
    Indeterminate,
}

#[derive(Clone, Debug)]
pub struct SolveInfo {
    pub status: Status,
    pub iterations: usize,
    pub obj_primal: f64,
    pub obj_dual: f64,
    pub res_pri: f64,
    pub res_dual: f64,
    pub res_gap: f64,
    pub res_cone: f64,
    pub factorizations: usize,
    pub engine: &'static str,
    pub update_class: UpdateClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateClass {
    Setup,
    R0,
    R1,
    R2,
}

impl Default for SolveInfo {
    fn default() -> Self {
        Self {
            status: Status::Unsolved,
            iterations: 0,
            obj_primal: f64::NAN,
            obj_dual: f64::NAN,
            res_pri: f64::INFINITY,
            res_dual: f64::INFINITY,
            res_gap: f64::INFINITY,
            res_cone: f64::INFINITY,
            factorizations: 0,
            engine: "none",
            update_class: UpdateClass::Setup,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub z: Vec<f64>,
    pub info: SolveInfo,
}
