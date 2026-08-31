//! One-off residual dump for exponential-cone programs. Not part of the test gate.
use conix::algebra::CscExt;
use conix::models;
use conix::settings::{EngineKind, Settings};
use conix::solve_once;
use conix::{CompositeCone, Cone, CscMatrix, Qcp};

fn dump(label: &str, st: Settings, qcp: Qcp) {
    let sol = solve_once(qcp, st).unwrap();
    println!(
        "{label:18} status={:?} eng={} it={} pri={:.3e} dual={:.3e} gap={:.3e} cone={:.3e} comp={:.3e} x={:?}",
        sol.info.status,
        sol.info.engine,
        sol.info.iterations,
        sol.info.res_pri,
        sol.info.res_dual,
        sol.info.res_gap,
        sol.info.res_cone,
        sol.info.res_comp,
        sol.x
    );
}

fn main() {
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
    let beta = 0.9;
    for (name, eng, iters, anderson) in [
        ("evar-admm", EngineKind::Admm, 12_000, 0),
        ("evar-admm-aa5", EngineKind::Admm, 12_000, 5),
        ("evar-split", EngineKind::Splitting, 12_000, 0),
        ("evar-auto", EngineKind::Auto, 12_000, 0),
        ("evar-ipm", EngineKind::Ipm, 120, 0),
    ] {
        let mut st = Settings::default();
        st.engine = eng;
        st.max_iter = iters;
        st.ipm_max_iter = iters;
        st.anderson_memory = anderson;
        dump(name, st, models::evar(&r, &p, beta, &[0.0, 0.0], &[1.0, 1.0]));
    }

    let qcp = Qcp {
        p: CscMatrix::zeros((1, 1)),
        q: vec![1.0],
        a: CscMatrix::from_triplets(3, 1, &[(2, 0, -1.0)]),
        b: vec![1.0, 1.0, 0.0],
        cones: CompositeCone::new(vec![Cone::Exponential]),
    };
    for (name, eng, iters, aa) in [
        ("log-admm", EngineKind::Admm, 8_000, 0),
        ("log-split", EngineKind::Splitting, 8_000, 0),
        ("log-ipm", EngineKind::Ipm, 40, 0),
    ] {
        let mut st = Settings::default();
        st.engine = eng;
        st.max_iter = iters;
        st.ipm_max_iter = iters;
        st.anderson_memory = aa;
        st.verbose = name == "log-ipm" || name == "evar-ipm";
        dump(
            name,
            st,
            Qcp {
                p: qcp.p.clone(),
                q: qcp.q.clone(),
                a: qcp.a.clone(),
                b: qcp.b.clone(),
                cones: qcp.cones.clone(),
            },
        );
    }
}
