//! C ABI for the sequential Python backtest API.

use std::cell::RefCell;
use std::ffi::{c_char, c_double, c_int, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use crate::algebra::{CscExt, CscMatrix};
use crate::cones::{CompositeCone, Cone};
use crate::models;
use crate::settings::{EngineKind, Settings};
use crate::status::Status;
use crate::workspace::{solve, Qcp, Workspace};

thread_local! {
    static LAST_ERR: RefCell<CString> = RefCell::new(CString::new("ok").unwrap());
}

fn set_err(msg: &str) {
    let s = CString::new(msg.replace('\0', "")).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERR.with(|e| *e.borrow_mut() = s);
}

unsafe fn csc_from_raw(
    m: usize,
    n: usize,
    col_ptr: *const usize,
    row_idx: *const usize,
    x: *const c_double,
    nnz: usize,
    keep_zeros: bool,
) -> Result<CscMatrix, String> {
    if col_ptr.is_null() || (nnz > 0 && (row_idx.is_null() || x.is_null())) {
        return Err("null CSC pointer".into());
    }
    let cp = slice::from_raw_parts(col_ptr, n + 1);
    let ri = slice::from_raw_parts(row_idx, nnz);
    let xv = slice::from_raw_parts(x, nnz);
    if cp[0] != 0 || cp[n] != nnz {
        return Err("CSC col_ptr must start at 0 and end at nnz".into());
    }
    let mut trips = Vec::with_capacity(nnz);
    for j in 0..n {
        for p in cp[j]..cp[j + 1] {
            trips.push((ri[p], j, xv[p]));
        }
    }
    Ok(if keep_zeros {
        CscMatrix::from_triplets_keep_zeros(m, n, &trips)
    } else {
        CscMatrix::from_triplets(m, n, &trips)
    })
}

unsafe fn cones_from_raw(
    kind: *const c_int,
    dim: *const usize,
    alpha: *const c_double,
    n_cones: usize,
) -> Result<CompositeCone, String> {
    if n_cones == 0 || kind.is_null() || dim.is_null() {
        return Err("null cone pointer".into());
    }
    let kinds = slice::from_raw_parts(kind, n_cones);
    let dims = slice::from_raw_parts(dim, n_cones);
    let alphas = if alpha.is_null() {
        &[][..]
    } else {
        slice::from_raw_parts(alpha, n_cones)
    };
    let mut cones = Vec::with_capacity(n_cones);
    for i in 0..n_cones {
        let a = if i < alphas.len() { alphas[i] } else { 0.5 };
        cones.push(match kinds[i] {
            0 => Cone::Zero { dim: dims[i] },
            1 => Cone::Nonnegative { dim: dims[i] },
            2 => Cone::SecondOrder { dim: dims[i] },
            3 => Cone::Exponential,
            4 => Cone::Power { alpha: a },
            5 => Cone::DualExponential,
            6 => Cone::DualPower { alpha: a },
            8 => Cone::PsdTriangle { side: dims[i] },
            k => return Err(format!("unknown cone kind {k}")),
        });
    }
    Ok(CompositeCone::new(cones))
}

fn engine_from_i(v: c_int) -> EngineKind {
    match v {
        1 => EngineKind::Admm,
        2 => EngineKind::Splitting,
        3 => EngineKind::Ipm,
        _ => EngineKind::Auto,
    }
}

fn box_ws(ws: Workspace) -> *mut Workspace {
    Box::into_raw(Box::new(ws))
}

unsafe fn ws_mut<'a>(p: *mut Workspace) -> Result<&'a mut Workspace, String> {
    if p.is_null() {
        Err("null workspace".into())
    } else {
        Ok(&mut *p)
    }
}

unsafe fn returns_from_raw(r: *const c_double, t: usize, n: usize) -> Result<Vec<Vec<f64>>, String> {
    if r.is_null() {
        return Err("null returns".into());
    }
    let flat = slice::from_raw_parts(r, t * n);
    Ok((0..t).map(|s| flat[s * n..(s + 1) * n].to_vec()).collect())
}

fn catch<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err("panic in ConiX".into()),
    }
}

#[no_mangle]
pub extern "C" fn conix_last_error() -> *const c_char {
    LAST_ERR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn conix_free(p: *mut Workspace) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_setup(
    n: usize,
    m: usize,
    p_col: *const usize,
    p_row: *const usize,
    p_x: *const c_double,
    p_nnz: usize,
    q: *const c_double,
    a_col: *const usize,
    a_row: *const usize,
    a_x: *const c_double,
    a_nnz: usize,
    b: *const c_double,
    cone_kind: *const c_int,
    cone_dim: *const usize,
    cone_alpha: *const c_double,
    n_cones: usize,
    engine: c_int,
) -> *mut Workspace {
    match catch(|| {
        let p = csc_from_raw(n, n, p_col, p_row, p_x, p_nnz, false)?;
        let a = csc_from_raw(m, n, a_col, a_row, a_x, a_nnz, true)?;
        if q.is_null() || b.is_null() {
            return Err("null q/b".into());
        }
        let qv = slice::from_raw_parts(q, n).to_vec();
        let bv = slice::from_raw_parts(b, m).to_vec();
        let cones = cones_from_raw(cone_kind, cone_dim, cone_alpha, n_cones)?;
        let mut st = Settings::default();
        st.engine = engine_from_i(engine);
        Workspace::setup(
            Qcp {
                p,
                q: qv,
                a,
                b: bv,
                cones,
            },
            st,
        )
    }) {
        Ok(ws) => box_ws(ws),
        Err(e) => {
            set_err(&e);
            ptr::null_mut()
        }
    }
}

unsafe fn setup_model(qcp: Qcp, engine: c_int) -> Result<Workspace, String> {
    let mut st = Settings::default();
    st.engine = engine_from_i(engine);
    Workspace::setup(qcp, st)
}

#[no_mangle]
pub unsafe extern "C" fn conix_mean_variance(
    n: usize,
    sigma_col: *const usize,
    sigma_row: *const usize,
    sigma_x: *const c_double,
    sigma_nnz: usize,
    mu: *const c_double,
    l: *const c_double,
    u: *const c_double,
    lambda: c_double,
    engine: c_int,
) -> *mut Workspace {
    match catch(|| {
        let sigma = csc_from_raw(n, n, sigma_col, sigma_row, sigma_x, sigma_nnz, false)?;
        if mu.is_null() || l.is_null() || u.is_null() {
            return Err("null mean_variance vector".into());
        }
        let qcp = models::mean_variance(
            &sigma,
            slice::from_raw_parts(mu, n),
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
            lambda,
        );
        setup_model(qcp, engine)
    }) {
        Ok(ws) => box_ws(ws),
        Err(e) => {
            set_err(&e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_cvar(
    t: usize,
    n: usize,
    returns: *const c_double,
    beta: c_double,
    l: *const c_double,
    u: *const c_double,
    engine: c_int,
) -> *mut Workspace {
    match catch(|| {
        if l.is_null() || u.is_null() {
            return Err("null bounds".into());
        }
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::cvar(
            &r,
            beta,
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        setup_model(qcp, engine)
    }) {
        Ok(ws) => box_ws(ws),
        Err(e) => {
            set_err(&e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_mad(
    t: usize,
    n: usize,
    returns: *const c_double,
    probs: *const c_double,
    l: *const c_double,
    u: *const c_double,
    engine: c_int,
) -> *mut Workspace {
    match catch(|| {
        if l.is_null() || u.is_null() || probs.is_null() {
            return Err("null mad input".into());
        }
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::mad(
            &r,
            slice::from_raw_parts(probs, t),
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        setup_model(qcp, engine)
    }) {
        Ok(ws) => box_ws(ws),
        Err(e) => {
            set_err(&e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_cdar(
    t: usize,
    n: usize,
    returns: *const c_double,
    beta: c_double,
    l: *const c_double,
    u: *const c_double,
    engine: c_int,
) -> *mut Workspace {
    match catch(|| {
        if l.is_null() || u.is_null() {
            return Err("null bounds".into());
        }
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::cdar(
            &r,
            beta,
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        setup_model(qcp, engine)
    }) {
        Ok(ws) => box_ws(ws),
        Err(e) => {
            set_err(&e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_evar(
    t: usize,
    n: usize,
    returns: *const c_double,
    probs: *const c_double,
    beta: c_double,
    l: *const c_double,
    u: *const c_double,
    engine: c_int,
) -> *mut Workspace {
    match catch(|| {
        if l.is_null() || u.is_null() || probs.is_null() {
            return Err("null evar input".into());
        }
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::evar(
            &r,
            slice::from_raw_parts(probs, t),
            beta,
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        setup_model(qcp, engine)
    }) {
        Ok(ws) => box_ws(ws),
        Err(e) => {
            set_err(&e);
            ptr::null_mut()
        }
    }
}

fn apply_qcp(ws: &mut Workspace, q: &Qcp) -> Result<(), String> {
    if !q.a.same_pattern(&ws.orig.a) || !q.p.same_pattern(&ws.orig.p) {
        return Err("pattern change is R2; construct a new workspace".into());
    }
    if q.p.nzval != ws.orig.p.nzval {
        ws.update_p(&q.p)?;
    }
    ws.update_a(&q.a)?;
    ws.update_b(&q.b)?;
    ws.update_q(&q.q)?;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_cvar(
    p: *mut Workspace,
    t: usize,
    n: usize,
    returns: *const c_double,
    beta: c_double,
    l: *const c_double,
    u: *const c_double,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::cvar(
            &r,
            beta,
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        apply_qcp(ws, &qcp)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_evar(
    p: *mut Workspace,
    t: usize,
    n: usize,
    returns: *const c_double,
    probs: *const c_double,
    beta: c_double,
    l: *const c_double,
    u: *const c_double,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::evar(
            &r,
            slice::from_raw_parts(probs, t),
            beta,
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        apply_qcp(ws, &qcp)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_mean_variance(
    p: *mut Workspace,
    n: usize,
    sigma_col: *const usize,
    sigma_row: *const usize,
    sigma_x: *const c_double,
    sigma_nnz: usize,
    mu: *const c_double,
    l: *const c_double,
    u: *const c_double,
    lambda: c_double,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let sigma = csc_from_raw(n, n, sigma_col, sigma_row, sigma_x, sigma_nnz, false)?;
        if mu.is_null() || l.is_null() || u.is_null() {
            return Err("null mean_variance vector".into());
        }
        let qcp = models::mean_variance(
            &sigma,
            slice::from_raw_parts(mu, n),
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
            lambda,
        );
        apply_qcp(ws, &qcp)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_mad(
    p: *mut Workspace,
    t: usize,
    n: usize,
    returns: *const c_double,
    probs: *const c_double,
    l: *const c_double,
    u: *const c_double,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::mad(
            &r,
            slice::from_raw_parts(probs, t),
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        apply_qcp(ws, &qcp)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_cdar(
    p: *mut Workspace,
    t: usize,
    n: usize,
    returns: *const c_double,
    beta: c_double,
    l: *const c_double,
    u: *const c_double,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let r = returns_from_raw(returns, t, n)?;
        let qcp = models::cdar(
            &r,
            beta,
            slice::from_raw_parts(l, n),
            slice::from_raw_parts(u, n),
        );
        apply_qcp(ws, &qcp)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_q(p: *mut Workspace, q: *const c_double, n: usize) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        if q.is_null() {
            return Err("null q".into());
        }
        ws.update_q(slice::from_raw_parts(q, n))
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_b(p: *mut Workspace, b: *const c_double, m: usize) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        if b.is_null() {
            return Err("null b".into());
        }
        ws.update_b(slice::from_raw_parts(b, m))
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_solve(p: *mut Workspace) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        solve(ws);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_n(p: *const Workspace) -> usize {
    if p.is_null() {
        0
    } else {
        (*p).x.len()
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_m(p: *const Workspace) -> usize {
    if p.is_null() {
        0
    } else {
        (*p).s.len()
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_x(p: *const Workspace, out: *mut c_double, n: usize) -> c_int {
    if p.is_null() || out.is_null() {
        set_err("null");
        return -1;
    }
    let mut x = (*p).x.clone();
    let mut s = (*p).s.clone();
    let mut z = (*p).z.clone();
    crate::scale::unscale_solution(&(*p).eq, &mut x, &mut s, &mut z);
    let k = n.min(x.len());
    ptr::copy_nonoverlapping(x.as_ptr(), out, k);
    0
}

#[no_mangle]
pub unsafe extern "C" fn conix_status(p: *const Workspace) -> c_int {
    if p.is_null() {
        return -1;
    }
    match (*p).info.status {
        Status::Solved => 1,
        Status::MaxIters => 2,
        Status::PrimalInfeasible => 3,
        Status::DualInfeasible => 4,
        Status::Indeterminate => 5,
        Status::Unsolved => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_obj(p: *const Workspace) -> c_double {
    if p.is_null() {
        return f64::NAN;
    }
    (*p).info.obj_primal
}

#[no_mangle]
pub unsafe extern "C" fn conix_iterations(p: *const Workspace) -> usize {
    if p.is_null() {
        0
    } else {
        (*p).info.iterations
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_residuals(
    p: *const Workspace,
    pri: *mut c_double,
    dual: *mut c_double,
    gap: *mut c_double,
    cone: *mut c_double,
    comp: *mut c_double,
) -> c_int {
    if p.is_null() {
        return -1;
    }
    let i = &(*p).info;
    if !pri.is_null() {
        *pri = i.res_pri;
    }
    if !dual.is_null() {
        *dual = i.res_dual;
    }
    if !gap.is_null() {
        *gap = i.res_gap;
    }
    if !cone.is_null() {
        *cone = i.res_cone;
    }
    if !comp.is_null() {
        *comp = i.res_comp;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn conix_set_engine(p: *mut Workspace, engine: c_int) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        ws.settings.engine = engine_from_i(engine);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn conix_version() -> *const c_char {
    static V: &[u8] = b"0.1.0\0";
    V.as_ptr() as *const c_char
}

/// Copy unscaled dual cone multiplier `z` (Clarabel / CVXPY dual `y`).
#[no_mangle]
pub unsafe extern "C" fn conix_z(p: *const Workspace, out: *mut c_double, m: usize) -> c_int {
    if p.is_null() || out.is_null() {
        set_err("null");
        return -1;
    }
    let mut x = (*p).x.clone();
    let mut s = (*p).s.clone();
    let mut z = (*p).z.clone();
    crate::scale::unscale_solution(&(*p).eq, &mut x, &mut s, &mut z);
    let k = m.min(z.len());
    ptr::copy_nonoverlapping(z.as_ptr(), out, k);
    0
}

/// Copy unscaled slack `s`.
#[no_mangle]
pub unsafe extern "C" fn conix_s(p: *const Workspace, out: *mut c_double, m: usize) -> c_int {
    if p.is_null() || out.is_null() {
        set_err("null");
        return -1;
    }
    let mut x = (*p).x.clone();
    let mut s = (*p).s.clone();
    let mut z = (*p).z.clone();
    crate::scale::unscale_solution(&(*p).eq, &mut x, &mut s, &mut z);
    let k = m.min(s.len());
    ptr::copy_nonoverlapping(s.as_ptr(), out, k);
    0
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_p(
    p: *mut Workspace,
    n: usize,
    p_col: *const usize,
    p_row: *const usize,
    p_x: *const c_double,
    p_nnz: usize,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let mat = csc_from_raw(n, n, p_col, p_row, p_x, p_nnz, false)?;
        ws.update_p(&mat)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn conix_update_a(
    p: *mut Workspace,
    m: usize,
    n: usize,
    a_col: *const usize,
    a_row: *const usize,
    a_x: *const c_double,
    a_nnz: usize,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let mat = csc_from_raw(m, n, a_col, a_row, a_x, a_nnz, true)?;
        ws.update_a(&mat)
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

/// Warm-start from unscaled `(x, s, z)`. Null pointers skip that vector.
#[no_mangle]
pub unsafe extern "C" fn conix_warm_start(
    p: *mut Workspace,
    x: *const c_double,
    s: *const c_double,
    z: *const c_double,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        let xv = if x.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(x, ws.x.len()))
        };
        let sv = if s.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(s, ws.s.len()))
        };
        let zv = if z.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(z, ws.z.len()))
        };
        ws.warm_start(xv, sv, zv);
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}

/// Configure common solver settings. Pass `-1` / `NaN` to leave a field unchanged.
#[no_mangle]
pub unsafe extern "C" fn conix_configure(
    p: *mut Workspace,
    max_iter: c_int,
    eps_abs: c_double,
    eps_rel: c_double,
    verbose: c_int,
    engine: c_int,
    polish: c_int,
) -> c_int {
    match catch(|| {
        let ws = ws_mut(p)?;
        if max_iter >= 0 {
            ws.settings.max_iter = max_iter as usize;
        }
        if eps_abs.is_finite() && eps_abs > 0.0 {
            ws.settings.eps_abs = eps_abs;
        }
        if eps_rel.is_finite() && eps_rel > 0.0 {
            ws.settings.eps_rel = eps_rel;
        }
        if verbose >= 0 {
            ws.settings.verbose = verbose != 0;
        }
        if engine >= 0 {
            ws.settings.engine = engine_from_i(engine);
        }
        if polish >= 0 {
            ws.settings.polish = polish != 0;
        }
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_err(&e);
            -1
        }
    }
}
