//! PyO3 bindings. The numerical core does not depend on this module.
//!
//! Packaging mirrors COSMO.rs: maturin builds `conix._conix` as the native
//! extension; the pure-Python package under `python/conix` wraps it for the
//! library API and the CVXPY solver.

#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
use crate::cones::{CompositeCone, Cone};
use crate::models;
use crate::settings::{EngineKind, Settings};
use crate::status::Status;
use crate::workspace::{solve as solve_ws, Qcp, Workspace};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct PyCsc(CscMatrix);

fn py_index_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(v) = obj.extract::<Vec<usize>>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<Vec<i64>>() {
        return v
            .into_iter()
            .map(|i| {
                usize::try_from(i)
                    .map_err(|_| PyValueError::new_err(format!("negative CSC index {i}")))
            })
            .collect();
    }
    if let Ok(v) = obj.extract::<Vec<i32>>() {
        return v
            .into_iter()
            .map(|i| {
                usize::try_from(i)
                    .map_err(|_| PyValueError::new_err(format!("negative CSC index {i}")))
            })
            .collect();
    }
    Err(PyValueError::new_err(
        "CSC indices/indptr must be an integer array (int32/int64)",
    ))
}

impl<'a> FromPyObject<'a> for PyCsc {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> PyResult<Self> {
        // Dense ndarray → CSC via scipy if available.
        if obj.hasattr("ndim")? {
            let ndim: usize = obj.getattr("ndim")?.extract().unwrap_or(0);
            if ndim == 2 && !obj.hasattr("tocsc")? {
                let sp = pyo3::types::PyModule::import(obj.py(), "scipy.sparse")?;
                let csc_cls = sp.getattr("csc_matrix")?;
                let csc = csc_cls.call1((obj,))?;
                return Self::extract_bound(&csc);
            }
        }
        let csc = if obj.hasattr("tocsc")? {
            let fmt: String = obj
                .getattr("format")
                .and_then(|f| f.extract())
                .unwrap_or_default();
            if fmt == "csc" {
                obj.clone()
            } else {
                obj.call_method0("tocsc")?
            }
        } else {
            obj.clone()
        };
        // Force a contiguous copy of data to avoid numpy sub-view issues.
        let data_obj = csc.getattr("data")?;
        let nzval: Vec<f64> = if data_obj.hasattr("copy")? {
            data_obj.call_method0("copy")?.extract()?
        } else {
            data_obj.extract()?
        };
        let rowval = py_index_vec(&csc.getattr("indices")?)?;
        let colptr = py_index_vec(&csc.getattr("indptr")?)?;
        let shape: Vec<usize> = csc.getattr("shape")?.extract()?;
        if shape.len() != 2 {
            return Err(PyValueError::new_err("matrix shape must be (m, n)"));
        }
        Ok(PyCsc(CscMatrix::new(
            shape[0], shape[1], colptr, rowval, nzval,
        )))
    }
}

fn cones_from_list(obj: &Bound<'_, PyAny>) -> PyResult<CompositeCone> {
    let mut cones = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let kind: String = if let Ok(s) = item.get_item(0) {
            s.extract()?
        } else {
            item.getattr("kind")?.extract()?
        };
        let kind = kind.to_lowercase();
        match kind.as_str() {
            "zero" | "z" | "eq" => {
                let dim: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::Zero { dim });
            }
            "nonnegative" | "nonneg" | "l" | "nn" => {
                let dim: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::Nonnegative { dim });
            }
            "soc" | "q" | "secondorder" => {
                let dim: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::SecondOrder { dim });
            }
            "exp" | "exponential" | "ep" => cones.push(Cone::Exponential),
            "dualexp" | "ed" => cones.push(Cone::DualExponential),
            "power" | "pow" | "p" => {
                let alpha: f64 = item.get_item(1)?.extract()?;
                cones.push(Cone::Power { alpha });
            }
            "dualpower" => {
                let alpha: f64 = item.get_item(1)?.extract()?;
                cones.push(Cone::DualPower { alpha });
            }
            "psd" | "sdp" => {
                let side: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::PsdTriangle { side });
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown cone kind '{other}'"
                )));
            }
        }
    }
    Ok(CompositeCone::new(cones))
}

fn engine_from_i(v: i32) -> EngineKind {
    match v {
        1 => EngineKind::Admm,
        2 => EngineKind::Splitting,
        3 => EngineKind::Ipm,
        _ => EngineKind::Auto,
    }
}

fn status_str(s: Status) -> &'static str {
    match s {
        Status::Unsolved => "Unsolved",
        Status::Solved => "Solved",
        Status::MaxIters => "MaxIters",
        Status::PrimalInfeasible => "PrimalInfeasible",
        Status::DualInfeasible => "DualInfeasible",
        Status::Indeterminate => "Indeterminate",
    }
}

fn settings_from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Settings> {
    let mut s = Settings::default();
    s.verbose = false;
    let Some(kwargs) = kwargs else {
        return Ok(s);
    };
    for (k, v) in kwargs.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "rho" => s.rho = v.extract()?,
            "sigma" => s.sigma = v.extract()?,
            "alpha" => s.alpha = v.extract()?,
            "eps_abs" => s.eps_abs = v.extract()?,
            "eps_rel" => s.eps_rel = v.extract()?,
            "max_iter" => s.max_iter = v.extract()?,
            "verbose" => s.verbose = v.extract()?,
            "polish" => s.polish = v.extract()?,
            "engine" => {
                let eng: i32 = v.extract()?;
                s.engine = engine_from_i(eng);
            }
            "adaptive_rho" => s.adaptive_rho = v.extract()?,
            "ipm_max_iter" => s.ipm_max_iter = v.extract()?,
            "anderson_memory" => s.anderson_memory = v.extract()?,
            "scaling" | "scaling_iter" => s.scaling_iter = v.extract()?,
            "use_quad_obj" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "unrecognized ConiX setting '{other}'"
                )));
            }
        }
    }
    Ok(s)
}

#[pyclass(name = "Solution")]
struct PySolution {
    #[pyo3(get)]
    x: Vec<f64>,
    #[pyo3(get)]
    y: Vec<f64>,
    #[pyo3(get)]
    s: Vec<f64>,
    #[pyo3(get)]
    obj_val: f64,
    #[pyo3(get)]
    obj: f64,
    #[pyo3(get)]
    iter: usize,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    r_prim: f64,
    #[pyo3(get)]
    r_dual: f64,
    #[pyo3(get)]
    r_gap: f64,
    #[pyo3(get)]
    r_cone: f64,
    #[pyo3(get)]
    r_comp: f64,
    #[pyo3(get)]
    setup_time: f64,
    #[pyo3(get)]
    solve_time: f64,
    #[pyo3(get)]
    engine: String,
}

impl PySolution {
    fn from_ws(ws: &Workspace) -> Self {
        let info = &ws.info;
        let status = status_str(info.status).to_string();
        let mut x = ws.x.clone();
        let mut s = ws.s.clone();
        let mut z = ws.z.clone();
        crate::scale::unscale_solution(&ws.eq, &mut x, &mut s, &mut z);
        Self {
            x,
            y: z,
            s,
            obj_val: info.obj_primal,
            obj: info.obj_primal,
            iter: info.iterations,
            iterations: info.iterations,
            status,
            r_prim: info.res_pri,
            r_dual: info.res_dual,
            r_gap: info.res_gap,
            r_cone: info.res_cone,
            r_comp: info.res_comp,
            setup_time: 0.0,
            solve_time: 0.0,
            engine: info.engine.to_string(),
        }
    }
}

#[pymethods]
impl PySolution {
    #[getter]
    fn residuals(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        d.set_item("pri", self.r_prim)?;
        d.set_item("dual", self.r_dual)?;
        d.set_item("gap", self.r_gap)?;
        d.set_item("cone", self.r_cone)?;
        d.set_item("comp", self.r_comp)?;
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Solution(status={:?}, obj={:.6e}, iters={})",
            self.status, self.obj_val, self.iter
        )
    }
}

#[pyclass(name = "ConixSolver")]
struct PyConixSolver {
    inner: Workspace,
}

#[pymethods]
impl PyConixSolver {
    #[new]
    #[pyo3(signature = (P, q, A, b, cones, **kwargs))]
    fn new(
        P: PyCsc,
        q: Vec<f64>,
        A: PyCsc,
        b: Vec<f64>,
        cones: Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let settings = settings_from_kwargs(kwargs)?;
        let cones = cones_from_list(&cones)?;
        let problem = Qcp {
            p: P.0,
            q,
            a: A.0,
            b,
            cones,
        };
        let inner = Workspace::setup(problem, settings)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(Self { inner })
    }

    fn solve(&mut self, py: Python<'_>) -> PyResult<PySolution> {
        py.allow_threads(|| {
            solve_ws(&mut self.inner);
        });
        Ok(PySolution::from_ws(&self.inner))
    }

    fn update_q(&mut self, q: Vec<f64>) -> PyResult<()> {
        self.inner
            .update_q(&q)
            .map_err(|e| PyValueError::new_err(e))
    }

    fn update_b(&mut self, b: Vec<f64>) -> PyResult<()> {
        self.inner
            .update_b(&b)
            .map_err(|e| PyValueError::new_err(e))
    }

    fn update_p(&mut self, P: PyCsc) -> PyResult<()> {
        self.inner
            .update_p(&P.0)
            .map_err(|e| PyValueError::new_err(e))
    }

    fn update_a(&mut self, A: PyCsc) -> PyResult<()> {
        self.inner
            .update_a(&A.0)
            .map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(signature = (x = None, s = None, z = None))]
    fn warm_start(
        &mut self,
        x: Option<Vec<f64>>,
        s: Option<Vec<f64>>,
        z: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .warm_start(x.as_deref(), s.as_deref(), z.as_deref());
        Ok(())
    }

    #[pyo3(signature = (**kwargs))]
    fn configure(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let Some(kwargs) = kwargs else {
            return Ok(());
        };
        for (k, v) in kwargs.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "max_iter" => self.inner.settings.max_iter = v.extract()?,
                "eps_abs" => self.inner.settings.eps_abs = v.extract()?,
                "eps_rel" => self.inner.settings.eps_rel = v.extract()?,
                "verbose" => self.inner.settings.verbose = v.extract()?,
                "polish" => self.inner.settings.polish = v.extract()?,
                "engine" => {
                    let eng: i32 = v.extract()?;
                    self.inner.settings.engine = engine_from_i(eng);
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unrecognized configure key '{other}'"
                    )));
                }
            }
        }
        Ok(())
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.x.len()
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.s.len()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Ok(false)
    }

    fn close(&mut self) {}
}

#[pyfunction]
#[pyo3(signature = (P, q, A, b, cones, **kwargs))]
fn solve(
    py: Python<'_>,
    P: PyCsc,
    q: Vec<f64>,
    A: PyCsc,
    b: Vec<f64>,
    cones: Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySolution> {
    let mut solver = PyConixSolver::new(P, q, A, b, cones, kwargs)?;
    solver.solve(py)
}

fn rowmajor_returns(returns: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Vec<f64>)> {
    // Accept list-of-lists or 2d numpy.
    if let Ok(arr) = returns.extract::<Vec<Vec<f64>>>() {
        let t = arr.len();
        let n = if t == 0 { 0 } else { arr[0].len() };
        let mut flat = Vec::with_capacity(t * n);
        for row in arr {
            if row.len() != n {
                return Err(PyValueError::new_err("ragged returns matrix"));
            }
            flat.extend(row);
        }
        return Ok((t, n, flat));
    }
    let shape: Vec<usize> = returns.getattr("shape")?.extract()?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("returns must be 2-d"));
    }
    let flat: Vec<f64> = returns
        .call_method0("ravel")
        .or_else(|_| returns.call_method1("reshape", (-1,)))?
        .extract()?;
    Ok((shape[0], shape[1], flat))
}

fn settings_engine(engine: i32) -> Settings {
    let mut s = Settings::default();
    s.engine = engine_from_i(engine);
    s
}

fn apply_qcp(ws: &mut Workspace, q: &Qcp) -> Result<(), String> {
    if !q.a.same_pattern(&ws.orig.a) || !q.p.same_pattern(&ws.orig.p) {
        return Err("pattern change is R2; construct a new workspace".into());
    }
    use crate::algebra::CscExt;
    if q.p.nzval != ws.orig.p.nzval {
        ws.update_p(&q.p)?;
    }
    ws.update_a(&q.a)?;
    ws.update_b(&q.b)?;
    ws.update_q(&q.q)?;
    Ok(())
}

#[pyclass(name = "Workspace")]
struct PyWorkspace {
    inner: Workspace,
}

#[pymethods]
impl PyWorkspace {
    fn solve(&mut self, py: Python<'_>) -> PyResult<PySolution> {
        py.allow_threads(|| {
            solve_ws(&mut self.inner);
        });
        Ok(PySolution::from_ws(&self.inner))
    }

    fn set_engine(&mut self, engine: i32) -> PyResult<()> {
        self.inner.settings.engine = engine_from_i(engine);
        Ok(())
    }

    fn update_q(&mut self, q: Vec<f64>) -> PyResult<()> {
        self.inner
            .update_q(&q)
            .map_err(|e| PyValueError::new_err(e))
    }

    fn update_cvar(
        &mut self,
        returns: Bound<'_, PyAny>,
        beta: f64,
        l: Vec<f64>,
        u: Vec<f64>,
    ) -> PyResult<()> {
        let (t, n, flat) = rowmajor_returns(&returns)?;
        let mut rows = Vec::with_capacity(t);
        for i in 0..t {
            rows.push(flat[i * n..(i + 1) * n].to_vec());
        }
        let q = models::cvar(&rows, beta, &l, &u);
        apply_qcp(&mut self.inner, &q).map_err(|e| PyRuntimeError::new_err(e))
    }

    fn update_evar(
        &mut self,
        returns: Bound<'_, PyAny>,
        probs: Vec<f64>,
        beta: f64,
        l: Vec<f64>,
        u: Vec<f64>,
    ) -> PyResult<()> {
        let (t, n, flat) = rowmajor_returns(&returns)?;
        let mut rows = Vec::with_capacity(t);
        for i in 0..t {
            rows.push(flat[i * n..(i + 1) * n].to_vec());
        }
        let q = models::evar(&rows, &probs, beta, &l, &u);
        apply_qcp(&mut self.inner, &q).map_err(|e| PyRuntimeError::new_err(e))
    }

    fn update_mad(
        &mut self,
        returns: Bound<'_, PyAny>,
        probs: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
    ) -> PyResult<()> {
        let (t, n, flat) = rowmajor_returns(&returns)?;
        let mut rows = Vec::with_capacity(t);
        for i in 0..t {
            rows.push(flat[i * n..(i + 1) * n].to_vec());
        }
        let q = models::mad(&rows, &probs, &l, &u);
        apply_qcp(&mut self.inner, &q).map_err(|e| PyRuntimeError::new_err(e))
    }

    fn update_mean_variance(
        &mut self,
        sigma: PyCsc,
        mu: Vec<f64>,
        l: Vec<f64>,
        u: Vec<f64>,
        lam: f64,
    ) -> PyResult<()> {
        let q = models::mean_variance(&sigma.0, &mu, &l, &u, lam);
        apply_qcp(&mut self.inner, &q).map_err(|e| PyRuntimeError::new_err(e))
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.x.len()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Ok(false)
    }

    fn close(&mut self) {}
}

#[pyfunction]
#[pyo3(signature = (returns, beta, l, u, engine = 0))]
fn cvar(
    returns: Bound<'_, PyAny>,
    beta: f64,
    l: Vec<f64>,
    u: Vec<f64>,
    engine: i32,
) -> PyResult<PyWorkspace> {
    let (t, n, flat) = rowmajor_returns(&returns)?;
    if l.len() != n || u.len() != n {
        return Err(PyValueError::new_err("bounds length must match n"));
    }
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        rows.push(flat[i * n..(i + 1) * n].to_vec());
    }
    let q = models::cvar(&rows, beta, &l, &u);
    let inner = Workspace::setup(q, settings_engine(engine))
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(PyWorkspace { inner })
}

#[pyfunction]
#[pyo3(signature = (returns, probs, beta, l, u, engine = 0))]
fn evar(
    returns: Bound<'_, PyAny>,
    probs: Vec<f64>,
    beta: f64,
    l: Vec<f64>,
    u: Vec<f64>,
    engine: i32,
) -> PyResult<PyWorkspace> {
    let (t, n, flat) = rowmajor_returns(&returns)?;
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        rows.push(flat[i * n..(i + 1) * n].to_vec());
    }
    let q = models::evar(&rows, &probs, beta, &l, &u);
    let inner = Workspace::setup(q, settings_engine(engine))
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(PyWorkspace { inner })
}

#[pyfunction]
#[pyo3(signature = (returns, probs, l, u, engine = 0))]
fn mad(
    returns: Bound<'_, PyAny>,
    probs: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    engine: i32,
) -> PyResult<PyWorkspace> {
    let (t, n, flat) = rowmajor_returns(&returns)?;
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        rows.push(flat[i * n..(i + 1) * n].to_vec());
    }
    let q = models::mad(&rows, &probs, &l, &u);
    let inner = Workspace::setup(q, settings_engine(engine))
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(PyWorkspace { inner })
}

#[pyfunction]
#[pyo3(signature = (returns, beta, l, u, engine = 0))]
fn cdar(
    returns: Bound<'_, PyAny>,
    beta: f64,
    l: Vec<f64>,
    u: Vec<f64>,
    engine: i32,
) -> PyResult<PyWorkspace> {
    let (t, n, flat) = rowmajor_returns(&returns)?;
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        rows.push(flat[i * n..(i + 1) * n].to_vec());
    }
    let q = models::cdar(&rows, beta, &l, &u);
    let inner = Workspace::setup(q, settings_engine(engine))
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(PyWorkspace { inner })
}

#[pyfunction]
#[pyo3(signature = (sigma, mu, l, u, lam, engine = 0))]
fn mean_variance(
    sigma: PyCsc,
    mu: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    lam: f64,
    engine: i32,
) -> PyResult<PyWorkspace> {
    let q = models::mean_variance(&sigma.0, &mu, &l, &u, lam);
    let inner = Workspace::setup(q, settings_engine(engine))
        .map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(PyWorkspace { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConixSolver>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PyWorkspace>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(cvar, m)?)?;
    m.add_function(wrap_pyfunction!(evar, m)?)?;
    m.add_function(wrap_pyfunction!(mad, m)?)?;
    m.add_function(wrap_pyfunction!(cdar, m)?)?;
    m.add_function(wrap_pyfunction!(mean_variance, m)?)?;
    m.add("AUTO", 0_i32)?;
    m.add("ADMM", 1_i32)?;
    m.add("SPLITTING", 2_i32)?;
    m.add("IPM", 3_i32)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
