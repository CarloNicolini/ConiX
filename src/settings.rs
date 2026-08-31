#[derive(Clone, Debug)]
pub struct Settings {
    pub max_iter: usize,
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub eps_infeas: f64,
    pub alpha: f64,
    pub rho: f64,
    pub sigma: f64,
    pub adaptive_rho: bool,
    pub adaptive_rho_interval: usize,
    pub check_termination: usize,
    pub check_infeasibility: usize,
    pub iterative_refinement: usize,
    pub scaling_iter: usize,
    pub verbose: bool,
    pub anderson_memory: usize,
    pub polish: bool,
    pub engine: EngineKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    Auto,
    Admm,
    Splitting,
    Ipm,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            eps_abs: 1e-6,
            eps_rel: 1e-6,
            eps_infeas: 1e-8,
            alpha: 1.6,
            rho: 1.0,
            sigma: 1e-6,
            adaptive_rho: true,
            adaptive_rho_interval: 25,
            check_termination: 25,
            check_infeasibility: 40,
            iterative_refinement: 1,
            scaling_iter: 10,
            verbose: false,
            anderson_memory: 0,
            polish: true,
            engine: EngineKind::Auto,
        }
    }
}
