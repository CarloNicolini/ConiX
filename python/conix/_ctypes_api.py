"""ctypes fallback for ConiX when the maturin extension is not installed.

Requires a built ``libconix`` (``cargo build --release``). Set ``CONIX_LIB``
to the shared library path if it is not under ``target/release``.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_void_p,
)
from pathlib import Path

AUTO, ADMM, SPLITTING, IPM = 0, 1, 2, 3
UNSOLVED, SOLVED, MAX_ITERS, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, INDETERMINATE = (
    0,
    1,
    2,
    3,
    4,
    5,
)

_STATUS = {
    0: "Unsolved",
    1: "Solved",
    2: "MaxIters",
    3: "PrimalInfeasible",
    4: "DualInfeasible",
    5: "Indeterminate",
}


def _lib_path() -> Path:
    env = os.environ.get("CONIX_LIB")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    names = ("libconix.so", "libconix.dylib", "conix.dll")
    roots = [Path.cwd(), here.parent, here.parents[1]]
    if len(here.parents) > 2:
        roots.append(here.parents[2])
    for root in roots:
        for name in names:
            for sub in (root, root / "target" / "release", root / "target" / "debug"):
                cand = sub / name
                if cand.exists():
                    return cand
    raise FileNotFoundError(
        "libconix not found; build with `cargo build --release` or set CONIX_LIB"
    )


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_lib_path()))
    lib.conix_last_error.restype = c_char_p
    lib.conix_version.restype = c_char_p
    lib.conix_free.argtypes = [c_void_p]
    lib.conix_cvar.restype = c_void_p
    lib.conix_cvar.argtypes = [
        c_size_t,
        c_size_t,
        POINTER(c_double),
        c_double,
        POINTER(c_double),
        POINTER(c_double),
        c_int,
    ]
    lib.conix_evar.restype = c_void_p
    lib.conix_evar.argtypes = [
        c_size_t,
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        c_double,
        POINTER(c_double),
        POINTER(c_double),
        c_int,
    ]
    lib.conix_mad.restype = c_void_p
    lib.conix_mad.argtypes = [
        c_size_t,
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_int,
    ]
    lib.conix_cdar.restype = c_void_p
    lib.conix_cdar.argtypes = lib.conix_cvar.argtypes
    lib.conix_mean_variance.restype = c_void_p
    lib.conix_mean_variance.argtypes = [
        c_size_t,
        POINTER(c_size_t),
        POINTER(c_size_t),
        POINTER(c_double),
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_double,
        c_int,
    ]
    lib.conix_update_mean_variance.argtypes = [
        c_void_p,
        c_size_t,
        POINTER(c_size_t),
        POINTER(c_size_t),
        POINTER(c_double),
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_double,
    ]
    lib.conix_update_mean_variance.restype = c_int
    lib.conix_update_mad.argtypes = [
        c_void_p,
        c_size_t,
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
    ]
    lib.conix_update_mad.restype = c_int
    lib.conix_solve.argtypes = [c_void_p]
    lib.conix_solve.restype = c_int
    lib.conix_n.argtypes = [c_void_p]
    lib.conix_n.restype = c_size_t
    lib.conix_m.argtypes = [c_void_p]
    lib.conix_m.restype = c_size_t
    lib.conix_x.argtypes = [c_void_p, POINTER(c_double), c_size_t]
    lib.conix_x.restype = c_int
    lib.conix_status.argtypes = [c_void_p]
    lib.conix_status.restype = c_int
    lib.conix_obj.argtypes = [c_void_p]
    lib.conix_obj.restype = c_double
    lib.conix_iterations.argtypes = [c_void_p]
    lib.conix_iterations.restype = c_size_t
    lib.conix_residuals.argtypes = [
        c_void_p,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
    ]
    lib.conix_update_cvar.argtypes = [
        c_void_p,
        c_size_t,
        c_size_t,
        POINTER(c_double),
        c_double,
        POINTER(c_double),
        POINTER(c_double),
    ]
    lib.conix_update_cvar.restype = c_int
    lib.conix_update_evar.argtypes = [
        c_void_p,
        c_size_t,
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        c_double,
        POINTER(c_double),
        POINTER(c_double),
    ]
    lib.conix_update_evar.restype = c_int
    lib.conix_update_q.argtypes = [c_void_p, POINTER(c_double), c_size_t]
    lib.conix_update_q.restype = c_int
    lib.conix_set_engine.argtypes = [c_void_p, c_int]
    lib.conix_set_engine.restype = c_int
    return lib


_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def last_error() -> str:
    raw = lib().conix_last_error()
    return raw.decode() if raw else ""


def _dptr(xs: list[float]):
    arr = (c_double * len(xs))(*[float(v) for v in xs])
    return arr, arr


def _rowmajor(rows: list[list[float]]) -> ctypes.Array:
    t = len(rows)
    n = len(rows[0]) if t else 0
    flat = [float(v) for row in rows for v in row]
    return (c_double * (t * n))(*flat)


def _sigma_ptrs(sigma):
    """CSC upper triangle arrays for ``conix_mean_variance`` (numpy or nested lists)."""
    try:
        import numpy as np
        from conix.qcp import upper_triplet

        if hasattr(sigma, "shape"):
            col_ptr, row_idx, x = upper_triplet(np.asarray(sigma, dtype=np.float64))
            n = int(col_ptr.size - 1)
            nnz = int(x.size)
            col_arr = (c_size_t * (n + 1))(*[int(v) for v in col_ptr])
            row_arr = (c_size_t * nnz)(*[int(v) for v in row_idx])
            x_arr = (c_double * nnz)(*[float(v) for v in x])
            return n, col_arr, row_arr, x_arr, nnz
    except ImportError:
        pass
    raise TypeError("sigma must be a numpy array; install numpy for mean-variance API")


def _as_f64_list(v) -> list[float]:
    if hasattr(v, "tolist"):
        return [float(x) for x in v.tolist()]
    return [float(x) for x in v]


class Solution:
    def __init__(
        self,
        x: list[float],
        status: str,
        obj: float,
        iterations: int,
        residuals: dict[str, float],
    ):
        self.x = x
        self.status = status
        self.obj = obj
        self.iterations = iterations
        self.residuals = residuals

    def __repr__(self) -> str:
        return f"Solution(status={self.status!r}, obj={self.obj:.6g}, iters={self.iterations})"


class Workspace:
    """Persistent sequential solver. R0/R1 updates reuse factorizations."""

    def __init__(self, ptr):
        if not ptr:
            raise RuntimeError(last_error() or "ConiX setup failed")
        self._ptr = ptr

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            lib().conix_free(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def n(self) -> int:
        return int(lib().conix_n(self._ptr))

    def set_engine(self, engine: int) -> None:
        if lib().conix_set_engine(self._ptr, int(engine)) != 0:
            raise RuntimeError(last_error())

    def solve(self) -> Solution:
        if lib().conix_solve(self._ptr) != 0:
            raise RuntimeError(last_error())
        n = self.n
        xbuf = (c_double * n)()
        lib().conix_x(self._ptr, xbuf, n)
        pri = c_double()
        dual = c_double()
        gap = c_double()
        cone = c_double()
        comp = c_double()
        lib().conix_residuals(
            self._ptr,
            ctypes.byref(pri),
            ctypes.byref(dual),
            ctypes.byref(gap),
            ctypes.byref(cone),
            ctypes.byref(comp),
        )
        st = int(lib().conix_status(self._ptr))
        return Solution(
            list(xbuf),
            _STATUS.get(st, str(st)),
            float(lib().conix_obj(self._ptr)),
            int(lib().conix_iterations(self._ptr)),
            {
                "pri": pri.value,
                "dual": dual.value,
                "gap": gap.value,
                "cone": cone.value,
                "comp": comp.value,
            },
        )

    def update_cvar(
        self,
        returns: list[list[float]],
        beta: float,
        l: list[float],
        u: list[float],
    ) -> None:
        t = len(returns)
        n = len(l)
        r = _rowmajor(returns)
        lb, _ = _dptr(l)
        ub, _ = _dptr(u)
        rc = lib().conix_update_cvar(self._ptr, t, n, r, float(beta), lb, ub)
        if rc != 0:
            raise RuntimeError(last_error())

    def update_evar(
        self,
        returns: list[list[float]],
        probs: list[float],
        beta: float,
        l: list[float],
        u: list[float],
    ) -> None:
        t = len(returns)
        n = len(l)
        r = _rowmajor(returns)
        pr, _ = _dptr(probs)
        lb, _ = _dptr(l)
        ub, _ = _dptr(u)
        rc = lib().conix_update_evar(self._ptr, t, n, r, pr, float(beta), lb, ub)
        if rc != 0:
            raise RuntimeError(last_error())

    def update_q(self, q: list[float]) -> None:
        arr, _ = _dptr(q)
        if lib().conix_update_q(self._ptr, arr, len(q)) != 0:
            raise RuntimeError(last_error())

    def update_mean_variance(
        self,
        sigma,
        mu,
        l: list[float],
        u: list[float],
        lam: float,
    ) -> None:
        n, col_arr, row_arr, x_arr, nnz = _sigma_ptrs(sigma)
        mu_arr, _ = _dptr(_as_f64_list(mu))
        lb, _ = _dptr(l)
        ub, _ = _dptr(u)
        rc = lib().conix_update_mean_variance(
            self._ptr,
            n,
            col_arr,
            row_arr,
            x_arr,
            nnz,
            mu_arr,
            lb,
            ub,
            float(lam),
        )
        if rc != 0:
            raise RuntimeError(last_error())

    def update_mad(
        self,
        returns: list[list[float]],
        probs: list[float],
        l: list[float],
        u: list[float],
    ) -> None:
        t = len(returns)
        n = len(l)
        r = _rowmajor(returns)
        pr, _ = _dptr(probs)
        lb, _ = _dptr(l)
        ub, _ = _dptr(u)
        rc = lib().conix_update_mad(self._ptr, t, n, r, pr, lb, ub)
        if rc != 0:
            raise RuntimeError(last_error())


def cvar(
    returns: list[list[float]],
    beta: float,
    l: list[float],
    u: list[float],
    engine: int = AUTO,
) -> Workspace:
    t = len(returns)
    n = len(l)
    r = _rowmajor(returns)
    lb, _ = _dptr(l)
    ub, _ = _dptr(u)
    return Workspace(lib().conix_cvar(t, n, r, float(beta), lb, ub, int(engine)))


def evar(
    returns: list[list[float]],
    probs: list[float],
    beta: float,
    l: list[float],
    u: list[float],
    engine: int = AUTO,
) -> Workspace:
    t = len(returns)
    n = len(l)
    r = _rowmajor(returns)
    pr, _ = _dptr(probs)
    lb, _ = _dptr(l)
    ub, _ = _dptr(u)
    return Workspace(lib().conix_evar(t, n, r, pr, float(beta), lb, ub, int(engine)))


def mad(
    returns: list[list[float]],
    probs: list[float],
    l: list[float],
    u: list[float],
    engine: int = AUTO,
) -> Workspace:
    t = len(returns)
    n = len(l)
    r = _rowmajor(returns)
    pr, _ = _dptr(probs)
    lb, _ = _dptr(l)
    ub, _ = _dptr(u)
    return Workspace(lib().conix_mad(t, n, r, pr, lb, ub, int(engine)))


def mean_variance(
    sigma,
    mu,
    l: list[float],
    u: list[float],
    lam: float,
    engine: int = AUTO,
) -> Workspace:
    n, col_arr, row_arr, x_arr, nnz = _sigma_ptrs(sigma)
    mu_arr, _ = _dptr(_as_f64_list(mu))
    lb, _ = _dptr(l)
    ub, _ = _dptr(u)
    return Workspace(
        lib().conix_mean_variance(
            n,
            col_arr,
            row_arr,
            x_arr,
            nnz,
            mu_arr,
            lb,
            ub,
            float(lam),
            int(engine),
        )
    )


def cdar(
    returns: list[list[float]],
    beta: float,
    l: list[float],
    u: list[float],
    engine: int = AUTO,
) -> Workspace:
    t = len(returns)
    n = len(l)
    r = _rowmajor(returns)
    lb, _ = _dptr(l)
    ub, _ = _dptr(u)
    return Workspace(lib().conix_cdar(t, n, r, float(beta), lb, ub, int(engine)))


