use crate::algebra::{CscExt, CscMatrix};
use crate::cones::CompositeCone;
use crate::kkt::KktSystem;
use crate::scale::Equilibration;
use crate::settings::{EngineKind, Settings};
use crate::status::{Solution, SolveInfo, UpdateClass};

#[derive(Clone, Debug)]
pub struct Qcp {
    pub p: CscMatrix,
    pub q: Vec<f64>,
    pub a: CscMatrix,
    pub b: Vec<f64>,
    pub cones: CompositeCone,
}

impl Qcp {
    pub fn validate(&self) -> Result<(), String> {
        if self.p.m != self.p.n {
            return Err("P must be square".into());
        }
        if self.q.len() != self.p.n {
            return Err("q dimension".into());
        }
        if self.a.n != self.p.n {
            return Err("A column dimension".into());
        }
        if self.a.m != self.b.len() {
            return Err("b dimension".into());
        }
        if self.cones.dim != self.a.m {
            return Err("cone dimension != m".into());
        }
        Ok(())
    }
}

pub struct Workspace {
    pub orig: Qcp,
    pub p: CscMatrix,
    pub q: Vec<f64>,
    pub a: CscMatrix,
    pub b: Vec<f64>,
    pub cones: CompositeCone,
    pub settings: Settings,
    pub eq: Equilibration,
    pub kkt: KktSystem,
    pub rho: Vec<f64>,
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub z: Vec<f64>,
    pub w: Vec<f64>,
    pub w_prev: Vec<f64>,
    pub factorizations: usize,
    pub last_update: UpdateClass,
    pub info: SolveInfo,
    pub last_engine: EngineKind,
    pub has_solution: bool,
    // splitting embedding
    pub v_embed: Vec<f64>,
    pub g_embed: Vec<f64>,
    pub h_embed: Vec<f64>,
    /// Sparse cone-block IPM KKT (AMD + symbolic reused on R1).
    pub ipm_kkt: Option<crate::ipm_kkt::IpmKkt>,
}

impl Workspace {
    pub fn setup(problem: Qcp, settings: Settings) -> Result<Self, String> {
        problem.validate()?;
        let n = problem.p.n;
        let m = problem.a.m;
        let cones = problem.cones.clone();
        let (eq, p, q, a, b) = crate::scale::ruiz(
            &problem.p,
            &problem.q,
            &problem.a,
            &problem.b,
            &cones,
            settings.scaling_iter,
        );
        let rho = default_rho(&cones, settings.rho);
        let kkt = KktSystem::analyze(&p, &a, settings.sigma, &rho)?;
        let nm = n + m;
        Ok(Self {
            orig: problem,
            p,
            q,
            a,
            b,
            cones,
            settings,
            eq,
            kkt,
            rho,
            x: vec![0.0; n],
            s: vec![0.0; m],
            z: vec![0.0; m],
            w: vec![0.0; nm],
            w_prev: vec![0.0; nm],
            factorizations: 1,
            last_update: UpdateClass::Setup,
            info: SolveInfo::default(),
            last_engine: EngineKind::Admm,
            has_solution: false,
            v_embed: vec![0.0; nm + 1],
            g_embed: vec![0.0; nm],
            h_embed: vec![0.0; nm],
            ipm_kkt: None,
        })
    }

    pub fn update_q(&mut self, q: &[f64]) -> Result<(), String> {
        if q.len() != self.orig.q.len() {
            return Err("q size".into());
        }
        self.orig.q.copy_from_slice(q);
        self.reapply_scale_vectors();
        bump_update(&mut self.last_update, UpdateClass::R0);
        Ok(())
    }

    pub fn update_b(&mut self, b: &[f64]) -> Result<(), String> {
        if b.len() != self.orig.b.len() {
            return Err("b size".into());
        }
        self.orig.b.copy_from_slice(b);
        self.reapply_scale_vectors();
        self.reconstruct_slacks();
        bump_update(&mut self.last_update, UpdateClass::R0);
        Ok(())
    }

    pub fn update_p(&mut self, p: &CscMatrix) -> Result<(), String> {
        if !p.same_pattern(&self.orig.p)
            && !p
                .upper_triangle()
                .same_pattern(&self.orig.p.upper_triangle())
        {
            return Err("P pattern changed (R2)".into());
        }
        self.orig.p = p.clone();
        self.reassemble_r1()?;
        bump_update(&mut self.last_update, UpdateClass::R1);
        Ok(())
    }

    pub fn update_a(&mut self, a: &CscMatrix) -> Result<(), String> {
        if !a.same_pattern(&self.orig.a) {
            return Err("A pattern changed (R2)".into());
        }
        self.orig.a = a.clone();
        self.reassemble_r1()?;
        bump_update(&mut self.last_update, UpdateClass::R1);
        Ok(())
    }

    pub fn warm_start(&mut self, x: Option<&[f64]>, s: Option<&[f64]>, z: Option<&[f64]>) {
        if let Some(x) = x {
            self.x.copy_from_slice(x);
        }
        if let Some(s) = s {
            self.s.copy_from_slice(s);
        }
        if let Some(z) = z {
            self.z.copy_from_slice(z);
        }
        crate::scale::scale_iterate(&self.eq, &mut self.x, &mut self.s, &mut self.z);
        let n = self.x.len();
        self.w[..n].copy_from_slice(&self.x);
        for i in 0..self.s.len() {
            self.w[n + i] = self.s[i] - self.z[i] / self.rho[i];
        }
    }

    fn reapply_scale_vectors(&mut self) {
        let (_, _, q, _, b) = crate::scale::ruiz(
            &self.orig.p,
            &self.orig.q,
            &self.orig.a,
            &self.orig.b,
            &self.cones,
            0,
        );
        // apply stored D,E,c
        let mut qs = self.orig.q.clone();
        let mut bs = self.orig.b.clone();
        for i in 0..qs.len() {
            qs[i] *= self.eq.c * self.eq.d[i];
        }
        for i in 0..bs.len() {
            bs[i] *= self.eq.e[i];
        }
        self.q = qs;
        self.b = bs;
        let _ = q;
        let _ = b;
    }

    fn reassemble_r1(&mut self) -> Result<(), String> {
        let (eq, p, q, a, b) = crate::scale::ruiz(
            &self.orig.p,
            &self.orig.q,
            &self.orig.a,
            &self.orig.b,
            &self.cones,
            self.settings.scaling_iter,
        );
        self.eq = eq;
        self.p = p;
        self.q = q;
        self.a = a;
        self.b = b;
        self.rho = default_rho(&self.cones, self.settings.rho);
        self.kkt
            .update_pa(&self.p, &self.a, self.settings.sigma, &self.rho)?;
        self.factorizations += 1;
        if let Some(ipm) = self.ipm_kkt.as_mut() {
            if ipm.update_pa(&self.p, &self.a).is_err() {
                self.ipm_kkt = None;
            } else {
                self.factorizations += 1;
            }
        }
        self.z.fill(0.0);
        self.reconstruct_slacks();
        Ok(())
    }

    /// Rebuild cone slacks from the current primal: `s = Π_K(b - Ax)`.
    /// Finance auxiliaries (CVaR/MAD/CDaR z, peaks) are not copied; they are
    /// implied by this projection after an R0/R1 data change.
    pub fn reconstruct_slacks(&mut self) {
        let m = self.s.len();
        let mut ax = vec![0.0; m];
        self.a.mul(&self.x, &mut ax);
        for i in 0..m {
            self.s[i] = self.b[i] - ax[i];
        }
        self.cones.project(&mut self.s);
        self.sync_w();
    }

    /// COSMO `w = [x; s - z/ρ]` from the current unscaled-in-workspace iterate.
    pub fn sync_w(&mut self) {
        let n = self.x.len();
        let m = self.s.len();
        self.w[..n].copy_from_slice(&self.x);
        self.w_prev[..n].copy_from_slice(&self.x);
        for i in 0..m {
            let wi = self.s[i] - self.z[i] / self.rho[i];
            self.w[n + i] = wi;
            self.w_prev[n + i] = wi;
        }
    }

    /// Independent residuals in the caller's original coordinates.
    pub fn original_residuals(&self) -> crate::verifier::Residuals {
        let mut x = self.x.clone();
        let mut s = self.s.clone();
        let mut z = self.z.clone();
        crate::scale::unscale_solution(&self.eq, &mut x, &mut s, &mut z);
        crate::verifier::residuals(
            &self.orig.p,
            &self.orig.q,
            &self.orig.a,
            &self.orig.b,
            &self.orig.cones,
            &x,
            &s,
            &z,
        )
    }
}

fn bump_update(cur: &mut UpdateClass, incoming: UpdateClass) {
    *cur = match (*cur, incoming) {
        (UpdateClass::R2, _) | (_, UpdateClass::R2) => UpdateClass::R2,
        (UpdateClass::R1, _) | (_, UpdateClass::R1) => UpdateClass::R1,
        (UpdateClass::R0, _) | (_, UpdateClass::R0) => UpdateClass::R0,
        _ => UpdateClass::Setup,
    };
}

fn default_rho(cones: &CompositeCone, rho0: f64) -> Vec<f64> {
    let mut rho = vec![rho0; cones.dim];
    for (cone, &off) in cones.cones.iter().zip(&cones.offsets) {
        if let crate::cones::Cone::Zero { dim } = cone {
            for k in 0..*dim {
                rho[off + k] = rho0 * 1e3;
            }
        }
    }
    rho
}

fn run_engines(ws: &mut Workspace) {
    let engine = ws.settings.engine;
    ws.last_engine = engine;
    match engine {
        EngineKind::Ipm => crate::engines::ipm::run(ws),
        EngineKind::Splitting => crate::engines::splitting::run(ws),
        EngineKind::Admm => crate::engines::admm::run(ws),
        EngineKind::Auto => run_auto(ws),
    }
}

fn run_auto(ws: &mut Workspace) {
    let poly = ws.cones.is_polyhedral();
    let r0_reuse = ws.has_solution && ws.last_update == crate::status::UpdateClass::R0;
    if poly && r0_reuse {
        crate::engines::admm::run(ws);
        return;
    }

    // Setup/R1 polyhedral, and every nonsymmetric problem: IPM first.
    crate::engines::ipm::run(ws);
    if matches!(
        ws.info.status,
        crate::status::Status::Solved
            | crate::status::Status::PrimalInfeasible
            | crate::status::Status::DualInfeasible
    ) {
        ws.last_engine = EngineKind::Ipm;
        return;
    }
    let bak_x = ws.x.clone();
    let bak_s = ws.s.clone();
    let bak_z = ws.z.clone();
    let bak_info = ws.info.clone();
    let ipm_iters = ws.info.iterations;
    let saved_max = ws.settings.max_iter;
    let admm_cap = if poly {
        saved_max.min(ws.settings.auto_admm_max_iter)
    } else {
        saved_max
    };
    ws.settings.max_iter = admm_cap;
    crate::engines::admm::run(ws);
    ws.settings.max_iter = saved_max;
    ws.info.iterations += ipm_iters;
    let r_ipm = {
        let mut x = bak_x.clone();
        let mut s = bak_s.clone();
        let mut z = bak_z.clone();
        crate::scale::unscale_solution(&ws.eq, &mut x, &mut s, &mut z);
        crate::verifier::residuals(
            &ws.orig.p,
            &ws.orig.q,
            &ws.orig.a,
            &ws.orig.b,
            &ws.orig.cones,
            &x,
            &s,
            &z,
        )
    };
    let r_admm = ws.original_residuals();
    if crate::verifier::merit(&r_admm) > crate::verifier::merit(&r_ipm) {
        ws.x = bak_x;
        ws.s = bak_s;
        ws.z = bak_z;
        ws.info = bak_info;
        ws.sync_w();
        ws.last_engine = EngineKind::Ipm;
    } else {
        ws.last_engine = EngineKind::Admm;
    }
}

fn finalize(ws: &mut Workspace) -> Solution {
    let mut x = ws.x.clone();
    let mut s = ws.s.clone();
    let mut z = ws.z.clone();
    crate::scale::unscale_solution(&ws.eq, &mut x, &mut s, &mut z);
    let r = crate::verifier::residuals(
        &ws.orig.p,
        &ws.orig.q,
        &ws.orig.a,
        &ws.orig.b,
        &ws.orig.cones,
        &x,
        &s,
        &z,
    );
    ws.info.res_pri = r.res_pri;
    ws.info.res_dual = r.res_dual;
    ws.info.res_gap = r.res_gap;
    ws.info.res_cone = r.res_cone;
    ws.info.res_comp = r.res_comp;
    ws.info.obj_primal = r.obj_p;
    ws.info.obj_dual = r.obj_d;
    ws.info.factorizations = ws.factorizations;
    ws.info.update_class = ws.last_update;
    let eps = ws.settings.eps_abs.max(ws.settings.eps_rel);
    match ws.info.status {
        crate::status::Status::PrimalInfeasible => {
            if !crate::verifier::check_primal_ray(
                &ws.orig.a,
                &ws.orig.b,
                &ws.orig.cones,
                &z,
                ws.settings.eps_infeas,
            ) {
                ws.info.status = crate::status::Status::MaxIters;
            }
        }
        crate::status::Status::DualInfeasible => {
            if !crate::verifier::check_dual_ray(
                &ws.orig.p,
                &ws.orig.q,
                &ws.orig.a,
                &ws.orig.cones,
                &x,
                ws.settings.eps_infeas,
            ) {
                ws.info.status = crate::status::Status::MaxIters;
            }
        }
        _ => {
            if crate::verifier::solved_at(&r, eps) {
                ws.info.status = crate::status::Status::Solved;
            } else if ws.info.status == crate::status::Status::Solved {
                ws.info.status = crate::status::Status::MaxIters;
            }
        }
    }
    ws.has_solution = ws.info.status == crate::status::Status::Solved;
    Solution {
        x,
        s,
        z,
        info: ws.info.clone(),
    }
}

pub fn solve(ws: &mut Workspace) -> Solution {
    run_engines(ws);
    let sol = finalize(ws);
    ws.last_update = UpdateClass::Setup;
    sol
}
