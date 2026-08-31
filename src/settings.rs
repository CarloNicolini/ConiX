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
    pub adaptive_rho_max_adaptions: usize,
    pub check_termination: usize,
    pub check_infeasibility: usize,
    pub iterative_refinement: usize,
    pub scaling_iter: usize,
    pub verbose: bool,
    /// Type-I Anderson memory on the ADMM/DR map. `0` disables. Safeguard:
    /// accept the candidate only if it shortens the last residual.
    pub anderson_memory: usize,
    pub polish: bool,
    pub engine: EngineKind,
    /// ADMM iteration cap used by `EngineKind::Auto` on polyhedral problems
    /// before the NT-IPM fallback. Non-polyhedral Auto uses `max_iter`.
    pub auto_admm_max_iter: usize,
    /// Newton steps for the homogeneous sparse-KKT IPM.
    pub ipm_max_iter: usize,
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
            adaptive_rho_max_adaptions: 8,
            check_termination: 25,
            check_infeasibility: 40,
            iterative_refinement: 2,
            scaling_iter: 10,
            verbose: false,
            anderson_memory: 5,
            polish: true,
            engine: EngineKind::Auto,
            auto_admm_max_iter: 50,
            ipm_max_iter: 80,
        }
    }
}
