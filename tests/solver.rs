use conix::algebra::{CscMatrix, LdlNumeric, LdlSymbolic};
use conix::cones::CompositeCone;
use conix::cones::Cone;
use conix::models;
use conix::settings::{EngineKind, Settings};
use conix::{setup, solve, solve_once, Qcp, Status};

fn dense_sym_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

#[test]
fn ldl_matches_dense_spd() {
    let a = CscMatrix::from_triplets(
        3,
        3,
        &[
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 1, 3.0),
            (1, 2, 0.5),
            (2, 2, 2.0),
        ],
    );
    let sym = LdlSymbolic::analyze(&a).unwrap();
    let fac = LdlNumeric::factor(&a, &sym).unwrap();
    let mut b = vec![1.0, 2.0, 3.0];
    fac.solve_in_place(&mut b);
    let dense = a.to_dense();
    // complete symmetric dense
    let mut s = vec![vec![0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = dense[i][j] + dense[j][i];
            if i == j {
                s[i][j] = dense[i][j];
            }
        }
    }
    let y = dense_sym_mul(&s, &b);
    for i in 0..3 {
        assert!((y[i] - [1.0, 2.0, 3.0][i]).abs() < 1e-8, "{y:?}");
    }
}

#[test]
fn equality_qp() {
    let p = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]);
    let b = vec![1.0];
    let cones = CompositeCone::new(vec![Cone::Zero { dim: 1 }]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.eps_abs = 1e-6;
    st.eps_rel = 1e-6;
    st.adaptive_rho = false;
    let sol = solve_once(Qcp { p, q, a, b, cones }, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
    assert!((sol.x[0] - 0.5).abs() < 5e-4);
    assert!((sol.x[1] - 0.5).abs() < 5e-4);
}

#[test]
fn bound_qp() {
    // min 1/2 x^2  s.t. x >= 1
    let p = CscMatrix::identity(1);
    let q = vec![0.0];
    let a = CscMatrix::from_triplets(1, 1, &[(0, 0, -1.0)]);
    let b = vec![-1.0];
    let cones = CompositeCone::new(vec![Cone::Nonnegative { dim: 1 }]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = false;
    let sol = solve_once(Qcp { p, q, a, b, cones }, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
    assert!((sol.x[0] - 1.0).abs() < 1e-3, "{:?}", sol.x);
}

#[test]
fn sequential_r0_reuses_factor() {
    let p = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]);
    let b = vec![1.0];
    let cones = CompositeCone::new(vec![Cone::Zero { dim: 1 }]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = false;
    let mut ws = setup(Qcp { p, q, a, b, cones }, st).unwrap();
    let s1 = solve(&mut ws);
    let f1 = s1.info.factorizations;
    ws.update_q(&[-1.0, 0.0]).unwrap();
    let s2 = solve(&mut ws);
    assert_eq!(s2.info.factorizations, f1, "R0 must not refactor");
    assert_eq!(s2.info.status, Status::Solved, "{:?}", s2.info);
}

#[test]
fn cvar_long_only() {
    let returns = vec![
        vec![0.01, 0.02],
        vec![-0.03, 0.01],
        vec![0.00, -0.02],
        vec![0.02, 0.00],
    ];
    let qcp = models::cvar(&returns, 0.75, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = true;
    st.max_iter = 20_000;
    let sol = solve_once(qcp, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
    assert!(sol.info.res_pri <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_dual <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_cone <= 1e-6, "{:?}", sol.info);
    let x = &sol.x[..2];
    let s = x[0] + x[1];
    assert!((s - 1.0).abs() < 1e-4, "budget {x:?}");
}

#[test]
fn soc_projection_problem() {
    // min 0 s.t. ||(x1,x2)|| <= x0, x0 = 1, which forces ||x||<=1, pick x=0 via qp
    let p = CscMatrix::identity(3);
    let q = vec![0.0, 0.0, 0.0];
    // s in SOC: Ax + s = b with A = -I, b = 0 → s = x, x in SOC
    // plus x0 = 1 (zero cone on first extra row): actually encode [1,0,0]x = 1
    let a = CscMatrix::from_triplets(
        4,
        3,
        &[(0, 0, 1.0), (1, 0, -1.0), (2, 1, -1.0), (3, 2, -1.0)],
    );
    let b = vec![1.0, 0.0, 0.0, 0.0];
    let cones = CompositeCone::new(vec![Cone::Zero { dim: 1 }, Cone::SecondOrder { dim: 3 }]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = false;
    st.max_iter = 8_000;
    let sol = solve_once(Qcp { p, q, a, b, cones }, st).unwrap();
    assert!(sol.x[0].abs() < 2.0);
}

#[test]
fn mad_and_cdar_build() {
    let r = vec![vec![0.01, 0.0], vec![-0.01, 0.02], vec![0.0, 0.01]];
    let p = vec![1.0 / 3.0; 3];
    let mad = models::mad(&r, &p, &[0.0, 0.0], &[1.0, 1.0]);
    assert_eq!(mad.cones.dim, mad.a.m);
    let cdar = models::cdar(&r, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    assert_eq!(cdar.cones.dim, cdar.a.m);
    let evar = models::evar(&r, &p, 0.9, &[0.0, 0.0], &[1.0, 1.0]);
    assert_eq!(evar.cones.dim, evar.a.m);
}

#[test]
fn mean_variance_and_mad_solve() {
    let sigma = CscMatrix::identity(2);
    let mv = models::mean_variance(&sigma, &[0.1, 0.05], &[0.0, 0.0], &[1.0, 1.0], 1.0);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = true;
    let sol = solve_once(mv, st.clone()).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
    assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-3);

    let r = vec![vec![0.02, 0.00], vec![0.00, 0.01], vec![-0.01, 0.02]];
    let pr = vec![1.0 / 3.0; 3];
    let mad = models::mad(&r, &pr, &[0.0, 0.0], &[1.0, 1.0]);
    let sol = solve_once(mad, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
}

#[test]
fn cdar_solves() {
    let r = vec![
        vec![0.01, 0.0],
        vec![-0.02, 0.01],
        vec![0.0, -0.01],
        vec![0.01, 0.02],
    ];
    let qcp = models::cdar(&r, 0.75, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    st.adaptive_rho = true;
    st.max_iter = 8_000;
    st.auto_admm_max_iter = 50;
    let sol = solve_once(qcp, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
    assert!(
        sol.info.res_pri <= 1e-6 && sol.info.res_dual <= 1e-6,
        "{:?}",
        sol.info
    );
    assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-4, "{:?}", sol.x);
}

#[test]
fn ipm_solves_cvar() {
    let returns = vec![
        vec![0.01, 0.02],
        vec![-0.03, 0.01],
        vec![0.00, -0.02],
        vec![0.02, 0.00],
    ];
    let qcp = models::cvar(&returns, 0.75, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Ipm;
    let sol = solve_once(qcp, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "{:?}", sol.info);
    assert!(sol.info.res_pri <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_dual <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_gap <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_comp <= 1e-6, "{:?}", sol.info);
    assert!(
        (sol.x[0] + sol.x[1] - 1.0).abs() < 1e-4,
        "budget {:?}",
        sol.x
    );
}

#[test]
fn ipm_and_splitting_equality() {
    let p = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]);
    let b = vec![1.0];
    let cones = CompositeCone::new(vec![Cone::Zero { dim: 1 }]);
    let mut st = Settings::default();
    st.adaptive_rho = false;
    st.engine = EngineKind::Ipm;
    let ipm = solve_once(
        Qcp {
            p: p.clone(),
            q: q.clone(),
            a: a.clone(),
            b: b.clone(),
            cones: cones.clone(),
        },
        st.clone(),
    )
    .unwrap();
    assert!(
        ipm.info.status == Status::Solved || ipm.info.res_pri < 1e-3,
        "ipm {:?}",
        ipm.info
    );

    st.engine = EngineKind::Splitting;
    st.max_iter = 5_000;
    let sp = solve_once(Qcp { p, q, a, b, cones }, st).unwrap();
    assert!(
        sp.info.status == Status::Solved || (sp.x[0] + sp.x[1] - 1.0).abs() < 5e-2,
        "split {:?}",
        sp.info
    );
}

#[test]
fn sequential_cvar_r1() {
    let r1 = vec![vec![0.01, 0.02], vec![-0.01, 0.00], vec![0.02, -0.01]];
    let r2 = vec![vec![0.00, 0.01], vec![-0.02, 0.01], vec![0.01, 0.00]];
    let q1 = models::cvar(&r1, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    let q2 = models::cvar(&r2, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = true;
    let mut ws = setup(q1, st).unwrap();
    let s1 = solve(&mut ws);
    assert_eq!(s1.info.status, Status::Solved, "{:?}", s1.info);
    ws.update_a(&q2.a).unwrap();
    ws.update_b(&q2.b).unwrap();
    ws.update_q(&q2.q).unwrap();
    let s2 = solve(&mut ws);
    assert_eq!(s2.info.status, Status::Solved, "{:?}", s2.info);
    assert_eq!(s2.info.update_class, conix::UpdateClass::R1);
    assert!(s2.info.factorizations >= s1.info.factorizations);
}

#[test]
fn exp_cone_log() {
    // min t  s.t. (1, 1, t) ∈ EXP  ⇒  t ≥ e
    let p = CscMatrix::zeros(1, 1);
    let q = vec![1.0];
    let a = CscMatrix::from_triplets(3, 1, &[(2, 0, -1.0)]);
    let b = vec![1.0, 1.0, 0.0];
    let cones = CompositeCone::new(vec![Cone::Exponential]);
    for engine in [EngineKind::Splitting, EngineKind::Admm, EngineKind::Ipm] {
        let mut st = Settings::default();
        st.engine = engine;
        st.max_iter = 8_000;
        st.ipm_max_iter = 40;
        st.adaptive_rho = false;
        let sol = solve_once(
            Qcp {
                p: p.clone(),
                q: q.clone(),
                a: a.clone(),
                b: b.clone(),
                cones: cones.clone(),
            },
            st,
        )
        .unwrap();
        assert_eq!(sol.info.status, Status::Solved, "{engine:?} {:?}", sol.info);
        assert!(sol.info.res_pri <= 1e-6, "{engine:?} {:?}", sol.info);
        assert!(sol.info.res_dual <= 1e-6, "{engine:?} {:?}", sol.info);
        assert!(sol.info.res_gap <= 1e-6, "{engine:?} {:?}", sol.info);
        assert!(
            (sol.x[0] - std::f64::consts::E).abs() < 1e-4,
            "{engine:?} {:?}",
            sol.x
        );
    }
}

#[test]
fn evar_solves() {
    // T=10, p=0.1, β=0.8 so P(worst)=0.1 < 1-β=0.2 (non-degenerate EVaR).
    let r = vec![
        vec![0.01, 0.00],
        vec![-0.02, 0.01],
        vec![0.00, 0.02],
        vec![0.01, -0.01],
        vec![-0.03, 0.02],
        vec![0.02, -0.02],
        vec![0.00, 0.01],
        vec![-0.01, 0.00],
        vec![0.015, -0.01],
        vec![-0.005, 0.02],
    ];
    let p = vec![0.1; 10];
    let qcp = models::evar(&r, &p, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Ipm;
    st.ipm_max_iter = 80;
    let sol = solve_once(qcp, st).unwrap();
    assert_eq!(sol.info.status, Status::Solved, "evar {:?}", sol.info);
    assert!(sol.info.res_pri <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_dual <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_gap <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_comp <= 1e-6, "{:?}", sol.info);
    assert!(sol.info.res_cone <= 1e-6, "{:?}", sol.info);
    assert!(
        (sol.x[0] + sol.x[1] - 1.0).abs() < 1e-4,
        "budget {:?}",
        sol.x
    );
    // Perspective t must stay off the t→0 essential-supremum ray.
    assert!(sol.x[3] > 1e-5, "t_persp degenerate {:?}", sol.x);
}

#[test]
fn sequential_evar_r1() {
    let r1 = vec![
        vec![0.01, 0.00],
        vec![-0.02, 0.01],
        vec![0.00, 0.02],
        vec![0.01, -0.01],
        vec![-0.03, 0.02],
        vec![0.02, -0.02],
        vec![0.00, 0.01],
        vec![-0.01, 0.00],
        vec![0.015, -0.01],
        vec![-0.005, 0.02],
    ];
    let r2: Vec<Vec<f64>> = r1
        .iter()
        .map(|row| vec![row[1] * 0.8, row[0] * 1.1])
        .collect();
    let p = vec![0.1; 10];
    let q1 = models::evar(&r1, &p, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    let q2 = models::evar(&r2, &p, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    st.ipm_max_iter = 80;
    let mut ws = setup(q1, st).unwrap();
    let s1 = solve(&mut ws);
    assert_eq!(s1.info.status, Status::Solved, "{:?}", s1.info);
    ws.update_a(&q2.a).unwrap();
    ws.update_b(&q2.b).unwrap();
    ws.update_q(&q2.q).unwrap();
    let s2 = solve(&mut ws);
    assert_eq!(s2.info.status, Status::Solved, "{:?}", s2.info);
    assert_eq!(s2.info.update_class, conix::UpdateClass::R1);
    assert!(
        s2.info.res_pri <= 1e-6 && s2.info.res_dual <= 1e-6,
        "{:?}",
        s2.info
    );
}

#[test]
fn power_cone_bound() {
    // min 1/2 (x^2+y^2)  s.t. (x, y, 1) ∈ POW(0.5) and x+y=2, feasible at (1,1).
    let p = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(
        4,
        2,
        &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, -1.0), (2, 1, -1.0)],
    );
    let b = vec![2.0, 0.0, 0.0, 1.0];
    let cones = CompositeCone::new(vec![Cone::Zero { dim: 1 }, Cone::Power { alpha: 0.5 }]);
    for engine in [EngineKind::Admm, EngineKind::Ipm] {
        let mut st = Settings::default();
        st.engine = engine;
        st.max_iter = 8_000;
        st.ipm_max_iter = 40;
        let sol = solve_once(
            Qcp {
                p: p.clone(),
                q: q.clone(),
                a: a.clone(),
                b: b.clone(),
                cones: cones.clone(),
            },
            st,
        )
        .unwrap();
        assert_eq!(sol.info.status, Status::Solved, "{engine:?} {:?}", sol.info);
        assert!(
            (sol.x[0] + sol.x[1] - 2.0).abs() < 1e-4,
            "{engine:?} {:?}",
            sol.x
        );
    }
}

#[test]
fn verifier_independent() {
    let p = CscMatrix::identity(1);
    let q = vec![-1.0];
    let a = CscMatrix::from_triplets(1, 1, &[(0, 0, 1.0)]);
    let b = vec![0.0];
    let cones = CompositeCone::new(vec![Cone::Nonnegative { dim: 1 }]);
    // x <= 0, min 1/2 x^2 - x → unconstrained min x=1 infeasible, solution x=0
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = false;
    let sol = solve_once(
        Qcp {
            p: p.clone(),
            q: q.clone(),
            a: a.clone(),
            b: b.clone(),
            cones: cones.clone(),
        },
        st,
    )
    .unwrap();
    let r = conix::verifier::residuals(&p, &q, &a, &b, &cones, &sol.x, &sol.s, &sol.z);
    assert!(r.res_cone < 1e-6);
}

#[test]
fn ipm_primal_infeasible_lp() {
    // x >= 1 and x <= 0.
    let p = CscMatrix::zeros(1, 1);
    let q = vec![0.0];
    let a = CscMatrix::from_triplets(2, 1, &[(0, 0, -1.0), (1, 0, 1.0)]);
    let b = vec![-1.0, 0.0];
    let cones = CompositeCone::new(vec![Cone::Nonnegative { dim: 2 }]);
    let mut st = Settings::default();
    st.engine = EngineKind::Ipm;
    st.ipm_max_iter = 80;
    let sol = solve_once(Qcp { p, q, a, b, cones }, st).unwrap();
    assert_eq!(sol.info.status, Status::PrimalInfeasible, "{:?}", sol.info);
}

#[test]
fn ipm_kkt_cone_blocks_not_dense() {
    let p = CscMatrix::zeros(1, 1);
    let a = CscMatrix::from_triplets(9, 1, &[(0, 0, 1.0), (3, 0, -1.0), (6, 0, 0.5)]);
    let cones = CompositeCone::new(vec![
        Cone::Nonnegative { dim: 3 },
        Cone::Exponential,
        Cone::Exponential,
    ]);
    let k = conix::ipm_kkt::IpmKkt::analyze(&p, &a, &cones).unwrap();
    let dense_hs = 9 * 10 / 2;
    assert!(
        k.k_nnz() < dense_hs,
        "nnz={} should beat a dense m×m Hs triangle ({dense_hs})",
        k.k_nnz()
    );
    assert_eq!(k.packed_len(), 3 + 6 + 6);
}
