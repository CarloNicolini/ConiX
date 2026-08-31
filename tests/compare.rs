//! Correctness and sequence-level timing versus Clarabel 0.9 and OSQP 0.6.
//!
//! Fair sequential comparison is ConiX workspace reuse vs persistent
//! `update_q`/`update_A` on Clarabel and OSQP. SCS is a C library without a
//! first-class same-language QCP crate on this toolchain; the Python/C harness
//! is documented in `docs/benchmarks.md`.

#![allow(non_snake_case)]

use clarabel::algebra::CscMatrix as ClarCsc;
use clarabel::solver::*;
use conix::algebra::CscMatrix;
use conix::models;
use conix::settings::{EngineKind, Settings};
use conix::{setup, solve, solve_once, Qcp, Status};
use std::borrow::Cow;
use std::time::Instant;

fn to_clarabel(a: &CscMatrix) -> ClarCsc<f64> {
    ClarCsc::new(a.m, a.n, a.col_ptr.clone(), a.row_idx.clone(), a.x.clone())
}

fn cones_clarabel(q: &Qcp) -> Vec<SupportedConeT<f64>> {
    q.cones
        .cones
        .iter()
        .map(|c| match c {
            conix::Cone::Zero { dim } => SupportedConeT::ZeroConeT(*dim),
            conix::Cone::Nonnegative { dim } => SupportedConeT::NonnegativeConeT(*dim),
            conix::Cone::SecondOrder { dim } => SupportedConeT::SecondOrderConeT(*dim),
            conix::Cone::Exponential => SupportedConeT::ExponentialConeT(),
            conix::Cone::Power { alpha } => SupportedConeT::PowerConeT(*alpha),
            conix::Cone::GenPower { alpha, n_z } => {
                SupportedConeT::GenPowerConeT(alpha.clone(), *n_z)
            }
            other => panic!("no Clarabel mapping for {other:?}"),
        })
        .collect()
}

fn clarabel_settings() -> clarabel::solver::DefaultSettings<f64> {
    DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(50)
        .equilibrate_enable(true)
        .presolve_enable(false)
        .build()
        .unwrap()
}

fn solve_clarabel(q: &Qcp) -> (Vec<f64>, f64, SolverStatus) {
    let P = to_clarabel(&q.p.upper_triangle());
    let A = to_clarabel(&q.a);
    let cones = cones_clarabel(q);
    let settings = clarabel_settings();
    let mut solver = DefaultSolver::new(&P, &q.q, &A, &q.b, &cones, settings);
    solver.solve();
    (
        solver.solution.x.clone(),
        solver.solution.obj_val,
        solver.solution.status,
    )
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

fn returns(t: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..t)
        .map(|_| (0..n).map(|_| 0.02 * (lcg(&mut s) - 0.5)).collect())
        .collect()
}

#[test]
fn clarabel_matches_equality_qp() {
    let p = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(1, 2, &[(0, 0, 1.0), (0, 1, 1.0)]);
    let b = vec![1.0];
    let cones = conix::CompositeCone::new(vec![conix::Cone::Zero { dim: 1 }]);
    let qcp = Qcp { p, q, a, b, cones };
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = false;
    let ours = solve_once(
        Qcp {
            p: qcp.p.clone(),
            q: qcp.q.clone(),
            a: qcp.a.clone(),
            b: qcp.b.clone(),
            cones: qcp.cones.clone(),
        },
        st,
    )
    .unwrap();
    let (cx, cobj, cstat) = solve_clarabel(&qcp);
    assert_eq!(cstat, SolverStatus::Solved, "clarabel {cstat:?}");
    assert_eq!(ours.info.status, Status::Solved, "{:?}", ours.info);
    assert!((ours.x[0] - cx[0]).abs() < 1e-4);
    assert!((ours.x[1] - cx[1]).abs() < 1e-4);
    assert!((ours.info.obj_primal - cobj).abs() < 1e-4);
}

#[test]
fn clarabel_matches_cvar() {
    let r = vec![
        vec![0.01, 0.02],
        vec![-0.03, 0.01],
        vec![0.00, -0.02],
        vec![0.02, 0.00],
        vec![0.01, -0.01],
        vec![-0.01, 0.02],
    ];
    let qcp = models::cvar(&r, 0.8, &[0.0, 0.0], &[1.0, 1.0]);
    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.max_iter = 20_000;
    let ours = solve_once(
        Qcp {
            p: qcp.p.clone(),
            q: qcp.q.clone(),
            a: qcp.a.clone(),
            b: qcp.b.clone(),
            cones: qcp.cones.clone(),
        },
        st,
    )
    .unwrap();
    let (cx, cobj, cstat) = solve_clarabel(&qcp);
    assert_eq!(cstat, SolverStatus::Solved, "clarabel {cstat:?}");
    assert_eq!(ours.info.status, Status::Solved, "{:?}", ours.info);
    let n = 2;
    let bgt_c: f64 = cx[..n].iter().sum();
    let bgt_o: f64 = ours.x[..n].iter().sum();
    assert!((bgt_c - 1.0).abs() < 1e-6);
    assert!((bgt_o - 1.0).abs() < 1e-4);
    assert!(
        (ours.info.obj_primal - cobj).abs() < 1e-3_f64.max(1e-3 * cobj.abs()),
        "obj ours={} clarabel={}",
        ours.info.obj_primal,
        cobj
    );
}

#[test]
fn clarabel_matches_evar() {
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
    let ours = solve_once(
        Qcp {
            p: qcp.p.clone(),
            q: qcp.q.clone(),
            a: qcp.a.clone(),
            b: qcp.b.clone(),
            cones: qcp.cones.clone(),
        },
        st,
    )
    .unwrap();
    let (cx, cobj, cstat) = solve_clarabel(&qcp);
    assert_eq!(cstat, SolverStatus::Solved, "clarabel {cstat:?}");
    assert_eq!(ours.info.status, Status::Solved, "{:?}", ours.info);
    let bgt_c: f64 = cx[..2].iter().sum();
    let bgt_o: f64 = ours.x[..2].iter().sum();
    assert!((bgt_c - 1.0).abs() < 1e-6);
    assert!((bgt_o - 1.0).abs() < 1e-4);
    assert!(
        (ours.info.obj_primal - cobj).abs() < 1e-3_f64.max(1e-3 * cobj.abs()),
        "obj ours={} clarabel={}",
        ours.info.obj_primal,
        cobj
    );
}

#[test]
fn sequence_r1_evar_vs_clarabel() {
    let n = 4usize;
    let t = 10usize;
    let dates = 6usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let p = vec![1.0 / t as f64; t];
    let r0 = returns(t, n, 11);
    let q0 = models::evar(&r0, &p, 0.8, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    st.ipm_max_iter = 80;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar = 0.0_f64;
    let t0 = Instant::now();
    let s = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s.info.status, Status::Solved, "date0 {:?}", s.info);
    clar.solve();
    assert_eq!(clar.solution.status, SolverStatus::Solved);

    for d in 1..dates {
        let r = returns(t, n, 11 + d as u64 * 17);
        let q = models::evar(&r, &p, 0.8, &l, &u);
        ws.update_a(&q.a).unwrap();
        ws.update_b(&q.b).unwrap();
        ws.update_q(&q.q).unwrap();
        let t0 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t0.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "date{d} {:?}", s.info);
        assert!(
            s.info.res_pri <= 1e-6 && s.info.res_dual <= 1e-6 && s.info.res_gap <= 1e-6,
            "date{d} {:?}",
            s.info
        );

        let t1 = Instant::now();
        clar.update_A(&to_clarabel(&q.a)).unwrap();
        clar.update_b(&q.b).unwrap();
        clar.update_q(&q.q).unwrap();
        clar.solve();
        t_clar += t1.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - clar.solution.obj_val).abs()
                < 1e-3_f64.max(1e-3 * clar.solution.obj_val.abs()),
            "date{d} obj ours={} clarabel={}",
            s.info.obj_primal,
            clar.solution.obj_val
        );
    }
    println!("R1 EVaR n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar:.4}s");
}

#[test]
fn sequence_r0_markowitz_vs_clarabel() {
    let n = 8usize;
    let dates = 20usize;
    let p = CscMatrix::identity(n);
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let mu0 = vec![0.01; n];
    let q0 = models::mean_variance(&p, &mu0, &l, &u, 1.0);

    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = true;
    st.max_iter = 8_000;

    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar_upd = 0.0_f64;
    let mut t_clar_cold = 0.0_f64;
    let mut seed = 1u64;
    let f0 = {
        let t0 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t0.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "date0 {:?}", s.info);
        s.info.factorizations
    };
    clar.solve();
    for d in 1..dates {
        let mu: Vec<f64> = (0..n).map(|_| 0.005 + 0.02 * lcg(&mut seed)).collect();
        let q: Vec<f64> = mu.iter().map(|v| -v).collect();
        ws.update_q(&q).unwrap();
        let t0 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t0.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "date{d} {:?}", s.info);
        assert_eq!(
            s.info.factorizations, f0,
            "R0 must not refactor the cached KKT"
        );
        assert!((s.x[..n].iter().sum::<f64>() - 1.0).abs() < 1e-4);

        let t1 = Instant::now();
        clar.update_q(&q).unwrap();
        clar.solve();
        t_clar_upd += t1.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);

        let qcp = models::mean_variance(&p, &mu, &l, &u, 1.0);
        let t2 = Instant::now();
        let (_cx, cobj, cstat) = solve_clarabel(&qcp);
        t_clar_cold += t2.elapsed().as_secs_f64();
        assert_eq!(cstat, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - cobj).abs() < 5e-3_f64.max(1e-3 * cobj.abs()),
            "date{d} obj ours={} clarabel={cobj}",
            s.info.obj_primal
        );
    }
    println!(
        "R0 Markowitz n={n} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar_upd:.4}s  Clarabel-cold={t_clar_cold:.4}s  factors={f0}"
    );
}

#[test]
fn sequence_r1_cvar_vs_clarabel() {
    let n = 5usize;
    let t = 12usize;
    let dates = 10usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let r0 = returns(t, n, 7);
    let q0 = models::cvar(&r0, 0.8, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    st.adaptive_rho = true;
    st.max_iter = 4_000;
    st.auto_admm_max_iter = 50;

    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar_upd = 0.0_f64;
    let mut seed = 11u64;
    let mut n_solved = 0usize;
    let t0 = Instant::now();
    let s0 = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s0.info.status, Status::Solved, "cvar0 {:?}", s0.info);
    assert!(
        s0.info.res_pri <= 1e-6
            && s0.info.res_dual <= 1e-6
            && s0.info.res_cone <= 1e-6
            && s0.info.res_gap <= 1e-6
            && s0.info.res_comp <= 1e-6,
        "cvar0 {:?}",
        s0.info
    );
    n_solved += 1;
    clar.solve();
    let f0 = s0.info.factorizations;

    for d in 1..dates {
        let r = returns(t, n, 100 + d as u64 + ((seed as usize) as u64));
        seed = seed.wrapping_add(1);
        let q1 = models::cvar(&r, 0.8, &l, &u);
        ws.update_a(&q1.a).unwrap();
        ws.update_b(&q1.b).unwrap();
        ws.update_q(&q1.q).unwrap();
        let t1 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t1.elapsed().as_secs_f64();
        assert_eq!(s.info.update_class, conix::UpdateClass::R1);
        assert!(s.info.factorizations >= f0);
        assert!((s.x[..n].iter().sum::<f64>() - 1.0).abs() < 2e-3);
        assert_eq!(s.info.status, Status::Solved, "cvar date{d} {:?}", s.info);
        assert!(
            s.info.res_pri <= 1e-6
                && s.info.res_dual <= 1e-6
                && s.info.res_cone <= 1e-6
                && s.info.res_gap <= 1e-6
                && s.info.res_comp <= 1e-6,
            "cvar date{d} {:?}",
            s.info
        );
        n_solved += 1;

        let t2 = Instant::now();
        clar.update_A(&to_clarabel(&q1.a)).unwrap();
        clar.update_b(&q1.b).unwrap();
        clar.update_q(&q1.q).unwrap();
        clar.solve();
        t_clar_upd += t2.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - clar.solution.obj_val).abs()
                < 1e-2_f64.max(5e-3 * clar.solution.obj_val.abs()),
            "date{d} obj ours={} clarabel={}",
            s.info.obj_primal,
            clar.solution.obj_val
        );
    }
    println!(
        "R1 CVaR n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar_upd:.4}s  solved_1e-6={n_solved}/{dates}  factors0={f0} factorsN={}",
        ws.factorizations
    );
    assert_eq!(
        n_solved, dates,
        "expected every date at checked 1e-6, got {n_solved}/{dates}"
    );
}

fn to_osqp(a: &CscMatrix) -> osqp::CscMatrix<'static> {
    osqp::CscMatrix {
        nrows: a.m,
        ncols: a.n,
        indptr: Cow::Owned(a.col_ptr.clone()),
        indices: Cow::Owned(a.row_idx.clone()),
        data: Cow::Owned(a.x.clone()),
    }
}

fn osqp_bounds(q: &Qcp) -> (Vec<f64>, Vec<f64>) {
    let m = q.b.len();
    let mut l = vec![0.0; m];
    let mut u = vec![0.0; m];
    for (cone, &off) in q.cones.cones.iter().zip(&q.cones.offsets) {
        match cone {
            conix::Cone::Zero { dim } => {
                for k in 0..*dim {
                    l[off + k] = q.b[off + k];
                    u[off + k] = q.b[off + k];
                }
            }
            conix::Cone::Nonnegative { dim } => {
                for k in 0..*dim {
                    l[off + k] = f64::NEG_INFINITY;
                    u[off + k] = q.b[off + k];
                }
            }
            other => panic!("OSQP is QP-only; got {other:?}"),
        }
    }
    (l, u)
}

fn osqp_settings() -> osqp::Settings {
    osqp::Settings::default()
        .verbose(false)
        .warm_start(true)
        .polish(true)
        .eps_abs(1e-6)
        .eps_rel(1e-6)
        .max_iter(20_000)
}

fn qp_obj(p: &CscMatrix, q: &[f64], x: &[f64]) -> f64 {
    let mut px = vec![0.0; x.len()];
    p.sym_mul_add(x, &mut px, 1.0);
    0.5 * px.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
        + q.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
}

fn osqp_primal<'a>(st: &'a osqp::Status<'a>) -> Option<&'a [f64]> {
    match st {
        osqp::Status::Solved(s)
        | osqp::Status::SolvedInaccurate(s)
        | osqp::Status::MaxIterationsReached(s)
        | osqp::Status::TimeLimitReached(s) => Some(s.x()),
        _ => None,
    }
}

#[test]
fn sequence_r0_markowitz_vs_osqp() {
    let n = 8usize;
    let dates = 20usize;
    let p = CscMatrix::identity(n);
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let mu0 = vec![0.01; n];
    let q0 = models::mean_variance(&p, &mu0, &l, &u, 1.0);

    let mut st = Settings::default();
    st.engine = EngineKind::Admm;
    st.adaptive_rho = true;
    st.max_iter = 8_000;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let (lb, ub) = osqp_bounds(&q0);
    let mut osqp = osqp::Problem::new(
        to_osqp(&q0.p.upper_triangle()),
        &q0.q,
        to_osqp(&q0.a),
        &lb,
        &ub,
        &osqp_settings(),
    )
    .expect("osqp setup");

    let mut t_conix = 0.0_f64;
    let mut t_osqp = 0.0_f64;
    let mut seed = 1u64;
    let t0 = Instant::now();
    let s = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s.info.status, Status::Solved, "date0 {:?}", s.info);
    let t1 = Instant::now();
    let r = osqp.solve();
    t_osqp += t1.elapsed().as_secs_f64();
    assert!(r.solution().is_some(), "osqp date0 {:?}", r);

    for d in 1..dates {
        let mu: Vec<f64> = (0..n).map(|_| 0.005 + 0.02 * lcg(&mut seed)).collect();
        let q: Vec<f64> = mu.iter().map(|v| -v).collect();
        ws.update_q(&q).unwrap();
        let t0 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t0.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "date{d} {:?}", s.info);

        osqp.update_lin_cost(&q);
        let t1 = Instant::now();
        let r = osqp.solve();
        t_osqp += t1.elapsed().as_secs_f64();
        assert!(r.solution().is_some(), "osqp date{d} {:?}", r);
        let ox = r.x().expect("osqp x").to_vec();
        let oobj = qp_obj(&q0.p, &q, &ox);
        assert!(
            (s.info.obj_primal - oobj).abs() < 5e-3_f64.max(1e-3 * oobj.abs()),
            "date{d} obj ours={} osqp={oobj}",
            s.info.obj_primal
        );
    }
    println!("R0 Markowitz n={n} dates={dates}: ConiX={t_conix:.4}s  OSQP-update={t_osqp:.4}s");
}

#[test]
fn sequence_r1_cvar_vs_osqp() {
    let n = 5usize;
    let t = 12usize;
    let dates = 10usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let r0 = returns(t, n, 7);
    let q0 = models::cvar(&r0, 0.8, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let (lb0, ub0) = osqp_bounds(&q0);
    let mut osqp = osqp::Problem::new(
        to_osqp(&q0.p.upper_triangle()),
        &q0.q,
        to_osqp(&q0.a),
        &lb0,
        &ub0,
        &osqp_settings(),
    )
    .expect("osqp cvar setup");

    let mut t_conix = 0.0_f64;
    let mut t_osqp = 0.0_f64;
    let mut seed = 11u64;
    let t0 = Instant::now();
    let s0 = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s0.info.status, Status::Solved, "cvar0 {:?}", s0.info);
    let t1 = Instant::now();
    let r0s = osqp.solve();
    t_osqp += t1.elapsed().as_secs_f64();
    let mut n_osqp_solved = if r0s.solution().is_some() { 1 } else { 0 };

    for d in 1..dates {
        let r = returns(t, n, 100 + d as u64 + seed);
        seed = seed.wrapping_add(1);
        let q1 = models::cvar(&r, 0.8, &l, &u);
        ws.update_a(&q1.a).unwrap();
        ws.update_b(&q1.b).unwrap();
        ws.update_q(&q1.q).unwrap();
        let t1 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t1.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "cvar date{d} {:?}", s.info);
        assert!(s.info.res_pri <= 1e-6 && s.info.res_dual <= 1e-6);

        let (lb, ub) = osqp_bounds(&q1);
        osqp.update_A(to_osqp(&q1.a));
        osqp.update_lin_cost(&q1.q);
        osqp.update_bounds(&lb, &ub);
        let t2 = Instant::now();
        let rs = osqp.solve();
        t_osqp += t2.elapsed().as_secs_f64();
        if rs.solution().is_some() {
            n_osqp_solved += 1;
        }
        if let Some(ox) = osqp_primal(&rs) {
            let oobj = qp_obj(&q1.p, &q1.q, ox);
            assert!(
                (s.info.obj_primal - oobj).abs() < 5e-2_f64.max(5e-2 * oobj.abs().max(1e-6)),
                "date{d} obj ours={} osqp={oobj} status={:?}",
                s.info.obj_primal,
                rs
            );
        }
    }
    println!(
        "R1 CVaR n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  OSQP-update={t_osqp:.4}s  osqp_solved_1e-6={n_osqp_solved}/{dates}"
    );
}

#[test]
fn sequence_r1_cvar_backtest_vs_clarabel() {
    let n = 15usize;
    let t = 36usize;
    let dates = 12usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let r0 = returns(t, n, 3);
    let q0 = models::cvar(&r0, 0.9, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar = 0.0_f64;
    let mut seed = 21u64;
    let t0 = Instant::now();
    let s0 = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s0.info.status, Status::Solved, "backtest0 {:?}", s0.info);
    let t_c0 = Instant::now();
    clar.solve();
    t_clar += t_c0.elapsed().as_secs_f64();
    assert_eq!(clar.solution.status, SolverStatus::Solved);

    for d in 1..dates {
        let r = returns(t, n, 200 + d as u64 + seed);
        seed = seed.wrapping_add(1);
        let q1 = models::cvar(&r, 0.9, &l, &u);
        ws.update_a(&q1.a).unwrap();
        ws.update_b(&q1.b).unwrap();
        ws.update_q(&q1.q).unwrap();
        let t1 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t1.elapsed().as_secs_f64();
        assert_eq!(
            s.info.status,
            Status::Solved,
            "backtest date{d} {:?}",
            s.info
        );
        assert!(s.info.res_pri <= 1e-6 && s.info.res_dual <= 1e-6 && s.info.res_comp <= 1e-6);

        let t2 = Instant::now();
        clar.update_A(&to_clarabel(&q1.a)).unwrap();
        clar.update_b(&q1.b).unwrap();
        clar.update_q(&q1.q).unwrap();
        clar.solve();
        t_clar += t2.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - clar.solution.obj_val).abs()
                < 1e-2_f64.max(5e-3 * clar.solution.obj_val.abs()),
            "date{d} obj ours={} clarabel={}",
            s.info.obj_primal,
            clar.solution.obj_val
        );
    }
    println!(
        "R1 CVaR backtest n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar:.4}s"
    );
}

#[test]
fn sequence_r1_mad_vs_clarabel() {
    let n = 8usize;
    let t = 20usize;
    let dates = 8usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let p = vec![1.0 / t as f64; t];
    let r0 = returns(t, n, 13);
    let q0 = models::mad(&r0, &p, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar = 0.0_f64;
    let mut seed = 31u64;
    let t0 = Instant::now();
    let s0 = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s0.info.status, Status::Solved, "mad0 {:?}", s0.info);
    clar.solve();
    assert_eq!(clar.solution.status, SolverStatus::Solved);

    for d in 1..dates {
        let r = returns(t, n, 300 + d as u64 + seed);
        seed = seed.wrapping_add(1);
        let q1 = models::mad(&r, &p, &l, &u);
        ws.update_a(&q1.a).unwrap();
        ws.update_b(&q1.b).unwrap();
        ws.update_q(&q1.q).unwrap();
        let t1 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t1.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "mad date{d} {:?}", s.info);
        assert!(s.info.res_pri <= 1e-6 && s.info.res_dual <= 1e-6 && s.info.res_comp <= 1e-6);

        let t2 = Instant::now();
        clar.update_A(&to_clarabel(&q1.a)).unwrap();
        clar.update_b(&q1.b).unwrap();
        clar.update_q(&q1.q).unwrap();
        clar.solve();
        t_clar += t2.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - clar.solution.obj_val).abs()
                < 1e-2_f64.max(5e-3 * clar.solution.obj_val.abs()),
            "date{d} obj ours={} clarabel={}",
            s.info.obj_primal,
            clar.solution.obj_val
        );
    }
    println!("R1 MAD n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar:.4}s");
}

#[test]
fn sequence_r1_cdar_vs_clarabel() {
    let n = 6usize;
    let t = 16usize;
    let dates = 8usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let r0 = returns(t, n, 17);
    let q0 = models::cdar(&r0, 0.8, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar = 0.0_f64;
    let mut seed = 41u64;
    let t0 = Instant::now();
    let s0 = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s0.info.status, Status::Solved, "cdar0 {:?}", s0.info);
    clar.solve();
    assert_eq!(clar.solution.status, SolverStatus::Solved);

    for d in 1..dates {
        let r = returns(t, n, 400 + d as u64 + seed);
        seed = seed.wrapping_add(1);
        let q1 = models::cdar(&r, 0.8, &l, &u);
        ws.update_a(&q1.a).unwrap();
        ws.update_b(&q1.b).unwrap();
        ws.update_q(&q1.q).unwrap();
        let t1 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t1.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "cdar date{d} {:?}", s.info);
        assert!(s.info.res_pri <= 1e-6 && s.info.res_dual <= 1e-6 && s.info.res_comp <= 1e-6);

        let t2 = Instant::now();
        clar.update_A(&to_clarabel(&q1.a)).unwrap();
        clar.update_b(&q1.b).unwrap();
        clar.update_q(&q1.q).unwrap();
        clar.solve();
        t_clar += t2.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - clar.solution.obj_val).abs()
                < 1e-2_f64.max(5e-3 * clar.solution.obj_val.abs()),
            "date{d} obj ours={} clarabel={}",
            s.info.obj_primal,
            clar.solution.obj_val
        );
    }
    println!("R1 CDaR n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar:.4}s");
}

#[test]
fn sequence_r1_cvar_wide_vs_clarabel() {
    let n = 25usize;
    let t = 48usize;
    let dates = 8usize;
    let l = vec![0.0; n];
    let u = vec![1.0; n];
    let r0 = returns(t, n, 5);
    let q0 = models::cvar(&r0, 0.9, &l, &u);

    let mut st = Settings::default();
    st.engine = EngineKind::Auto;
    let mut ws = setup(
        Qcp {
            p: q0.p.clone(),
            q: q0.q.clone(),
            a: q0.a.clone(),
            b: q0.b.clone(),
            cones: q0.cones.clone(),
        },
        st,
    )
    .unwrap();

    let P = to_clarabel(&q0.p.upper_triangle());
    let A = to_clarabel(&q0.a);
    let cones = cones_clarabel(&q0);
    let mut clar = DefaultSolver::new(&P, &q0.q, &A, &q0.b, &cones, clarabel_settings());

    let mut t_conix = 0.0_f64;
    let mut t_clar = 0.0_f64;
    let mut seed = 51u64;
    let t0 = Instant::now();
    let s0 = solve(&mut ws);
    t_conix += t0.elapsed().as_secs_f64();
    assert_eq!(s0.info.status, Status::Solved, "wide0 {:?}", s0.info);
    clar.solve();
    assert_eq!(clar.solution.status, SolverStatus::Solved);

    for d in 1..dates {
        let r = returns(t, n, 500 + d as u64 + seed);
        seed = seed.wrapping_add(1);
        let q1 = models::cvar(&r, 0.9, &l, &u);
        ws.update_a(&q1.a).unwrap();
        ws.update_b(&q1.b).unwrap();
        ws.update_q(&q1.q).unwrap();
        let t1 = Instant::now();
        let s = solve(&mut ws);
        t_conix += t1.elapsed().as_secs_f64();
        assert_eq!(s.info.status, Status::Solved, "wide date{d} {:?}", s.info);
        assert!(s.info.res_pri <= 1e-6 && s.info.res_dual <= 1e-6 && s.info.res_comp <= 1e-6);

        let t2 = Instant::now();
        clar.update_A(&to_clarabel(&q1.a)).unwrap();
        clar.update_b(&q1.b).unwrap();
        clar.update_q(&q1.q).unwrap();
        clar.solve();
        t_clar += t2.elapsed().as_secs_f64();
        assert_eq!(clar.solution.status, SolverStatus::Solved);
        assert!(
            (s.info.obj_primal - clar.solution.obj_val).abs()
                < 1e-2_f64.max(5e-3 * clar.solution.obj_val.abs()),
            "date{d} obj ours={} clarabel={}",
            s.info.obj_primal,
            clar.solution.obj_val
        );
    }
    println!(
        "R1 CVaR wide n={n} T={t} dates={dates}: ConiX={t_conix:.4}s  Clarabel-update={t_clar:.4}s"
    );
}

#[test]
fn sequence_vs_scs_python() {
    let script = format!("{}/scripts/scs_sequence.py", env!("CARGO_MANIFEST_DIR"));
    let out = std::process::Command::new("python3")
        .arg(&script)
        .output()
        .expect("python3");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("{stdout}{stderr}");
    if !out.status.success() {
        let combined = format!("{stdout}{stderr}");
        if combined.contains("ModuleNotFoundError") {
            return;
        }
        panic!("scs harness failed: {stderr}");
    }
    assert!(
        stdout.contains("SCS-update") && stdout.contains("SCS-cold"),
        "unexpected SCS harness output: {stdout}"
    );
}
