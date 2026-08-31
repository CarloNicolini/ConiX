use crate::algebra::CscMatrix;
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
    // splitting embedding
    pub v_embed: Vec<f64>,
    pub g_embed: Vec<f64>,
    pub h_embed: Vec<f64>,
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
            v_embed: vec![0.0; nm + 1],
            g_embed: vec![0.0; nm],
            h_embed: vec![0.0; nm],
        })
    }

    pub fn update_q(&mut self, q: &[f64]) -> Result<(), String> {
        if q.len() != self.orig.q.len() {
            return Err("q size".into());
        }
        self.orig.q.copy_from_slice(q);
        self.reapply_scale_vectors();
        self.last_update = UpdateClass::R0;
        Ok(())
    }

    pub fn update_b(&mut self, b: &[f64]) -> Result<(), String> {
        if b.len() != self.orig.b.len() {
            return Err("b size".into());
        }
        self.orig.b.copy_from_slice(b);
        self.reapply_scale_vectors();
        self.last_update = UpdateClass::R0;
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
        self.last_update = UpdateClass::R1;
        Ok(())
    }

    pub fn update_a(&mut self, a: &CscMatrix) -> Result<(), String> {
        if !a.same_pattern(&self.orig.a) {
            return Err("A pattern changed (R2)".into());
        }
        self.orig.a = a.clone();
        self.reassemble_r1()?;
        self.last_update = UpdateClass::R1;
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
        self.kkt
            .update_pa(&self.p, &self.a, self.settings.sigma, &self.rho)?;
        self.factorizations += 1;
        Ok(())
    }
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

pub fn solve(ws: &mut Workspace) -> Solution {
    let engine = ws.settings.engine;
    ws.last_engine = engine;
    match engine {
        EngineKind::Ipm => crate::engines::ipm::run(ws),
        EngineKind::Splitting => crate::engines::splitting::run(ws),
        EngineKind::Admm => crate::engines::admm::run(ws),
        EngineKind::Auto => {
            if ws.cones.is_polyhedral() {
                crate::engines::admm::run(ws);
            } else {
                crate::engines::splitting::run(ws);
            }
            if ws.info.status == crate::status::Status::MaxIters {
                let bak_x = ws.x.clone();
                let bak_s = ws.s.clone();
                let bak_z = ws.z.clone();
                let bak_info = ws.info.clone();
                crate::engines::ipm::run(ws);
                let r_old = crate::verifier::residuals(
                    &ws.p, &ws.q, &ws.a, &ws.b, &ws.cones, &bak_x, &bak_s, &bak_z,
                );
                let r_new = crate::verifier::residuals(
                    &ws.p, &ws.q, &ws.a, &ws.b, &ws.cones, &ws.x, &ws.s, &ws.z,
                );
                let m_old = r_old.res_pri + r_old.res_dual + r_old.res_gap + r_old.res_cone;
                let m_new = r_new.res_pri + r_new.res_dual + r_new.res_gap + r_new.res_cone;
                if m_new > m_old {
                    ws.x = bak_x;
                    ws.s = bak_s;
                    ws.z = bak_z;
                    ws.info = bak_info;
                }
            }
        }
    }
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
    ws.info.obj_primal = r.obj_p;
    ws.info.obj_dual = r.obj_d;
    ws.info.factorizations = ws.factorizations;
    ws.info.update_class = ws.last_update;
    if crate::verifier::solved_at(&r, ws.settings.eps_abs.max(ws.settings.eps_rel))
        && ws.info.status != crate::status::Status::PrimalInfeasible
        && ws.info.status != crate::status::Status::DualInfeasible
    {
        ws.info.status = crate::status::Status::Solved;
    }
    Solution {
        x,
        s,
        z,
        info: ws.info.clone(),
    }
}
