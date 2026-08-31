"""General QCP solver.

When the maturin extension ``conix._conix`` is installed, re-export the native
PyO3 ``ConixSolver``. Otherwise provide a ctypes ABI wrapper around
``libconix.so`` (see the remainder of this file).
"""

from __future__ import annotations

try:
    from conix._conix import ConixSolver, Solution as SolverSolution, solve

    __all__ = ["ConixSolver", "SolverSolution", "solve"]
except ImportError:
    import ctypes
    from ctypes import POINTER, c_double, c_int, c_size_t, c_void_p
    from dataclasses import dataclass

    import numpy as np
    from scipy import sparse

    # Cone kind codes matching ``src/capi.rs`` ``cones_from_raw``.
    _CONE_KIND = {
        "zero": 0,
        "z": 0,
        "eq": 0,
        "nonnegative": 1,
        "nonneg": 1,
        "nn": 1,
        "l": 1,
        "secondorder": 2,
        "soc": 2,
        "q": 2,
        "exponential": 3,
        "exp": 3,
        "power": 4,
        "pow": 4,
        "dualexponential": 5,
        "dualexp": 5,
        "dualpower": 6,
        "psd": 8,
        "sdp": 8,
    }

    _STATUS = {
        0: "Unsolved",
        1: "Solved",
        2: "MaxIters",
        3: "PrimalInfeasible",
        4: "DualInfeasible",
        5: "Indeterminate",
    }

    def _pkg():
        """Late import avoids circular dependency with ``conix.__init__``."""
        import conix as cx

        return cx

    def _bind_general(lib_) -> None:
        if getattr(lib_, "_conix_general_bound", False):
            return
        lib_.conix_setup.restype = c_void_p
        lib_.conix_setup.argtypes = [
            c_size_t,
            c_size_t,
            POINTER(c_size_t),
            POINTER(c_size_t),
            POINTER(c_double),
            c_size_t,
            POINTER(c_double),
            POINTER(c_size_t),
            POINTER(c_size_t),
            POINTER(c_double),
            c_size_t,
            POINTER(c_double),
            POINTER(c_int),
            POINTER(c_size_t),
            POINTER(c_double),
            c_size_t,
            c_int,
        ]
        lib_.conix_z.argtypes = [c_void_p, POINTER(c_double), c_size_t]
        lib_.conix_z.restype = c_int
        lib_.conix_s.argtypes = [c_void_p, POINTER(c_double), c_size_t]
        lib_.conix_s.restype = c_int
        lib_.conix_update_p.argtypes = [
            c_void_p,
            c_size_t,
            POINTER(c_size_t),
            POINTER(c_size_t),
            POINTER(c_double),
            c_size_t,
        ]
        lib_.conix_update_p.restype = c_int
        lib_.conix_update_a.argtypes = [
            c_void_p,
            c_size_t,
            c_size_t,
            POINTER(c_size_t),
            POINTER(c_size_t),
            POINTER(c_double),
            c_size_t,
        ]
        lib_.conix_update_a.restype = c_int
        lib_.conix_warm_start.argtypes = [
            c_void_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        lib_.conix_warm_start.restype = c_int
        lib_.conix_configure.argtypes = [
            c_void_p,
            c_int,
            c_double,
            c_double,
            c_int,
            c_int,
            c_int,
        ]
        lib_.conix_configure.restype = c_int
        lib_._conix_general_bound = True

    def _csc_ptrs(mat: sparse.spmatrix):
        csc = sparse.csc_matrix(mat, dtype=np.float64)
        csc.sort_indices()
        col = (c_size_t * len(csc.indptr))(*[int(v) for v in csc.indptr])
        row = (c_size_t * len(csc.indices))(*[int(v) for v in csc.indices])
        data = (c_double * len(csc.data))(*[float(v) for v in csc.data])
        return csc.shape[0], csc.shape[1], col, row, data, int(csc.nnz), csc

    def _cones_ptrs(cones: list):
        kinds: list[int] = []
        dims: list[int] = []
        alphas: list[float] = []
        for cone in cones:
            if isinstance(cone, tuple):
                kind = str(cone[0]).lower()
                rest = cone[1:]
            elif isinstance(cone, dict):
                kind = str(cone["kind"]).lower()
                rest = [cone.get("dim", cone.get("side", cone.get("alpha", 0)))]
            else:
                raise TypeError(f"unsupported cone specifier: {cone!r}")
            code = _CONE_KIND.get(kind)
            if code is None:
                raise ValueError(f"unknown cone kind '{kind}'")
            kinds.append(code)
            if code in (3, 5):
                dims.append(3)
                alphas.append(0.5)
            elif code in (4, 6):
                alpha = float(rest[0]) if rest else 0.5
                dims.append(3)
                alphas.append(alpha)
            elif code == 8:
                dims.append(int(rest[0]))
                alphas.append(0.5)
            else:
                dims.append(int(rest[0]))
                alphas.append(0.5)
        n = len(kinds)
        return (c_int * n)(*kinds), (c_size_t * n)(*dims), (c_double * n)(*alphas), n

    @dataclass
    class SolverSolution:
        x: np.ndarray
        y: np.ndarray
        s: np.ndarray
        obj_val: float
        iter: int
        status: str
        r_prim: float
        r_dual: float
        solve_time: float = 0.0
        setup_time: float = 0.0

        @property
        def z(self) -> np.ndarray:
            return self.y

    class ConixSolver:
        """Persistent ConiX workspace for a fixed sparsity pattern (ctypes)."""

        def __init__(
            self,
            P,
            q,
            A,
            b,
            cones: list,
            *,
            engine: int | None = None,
            max_iter: int | None = None,
            eps_abs: float | None = None,
            eps_rel: float | None = None,
            verbose: bool = False,
            polish: bool | None = None,
            **_ignored,
        ):
            cx = _pkg()
            lib_ = cx.lib()
            _bind_general(lib_)
            if engine is None:
                engine = cx.AUTO
            p_m, p_n, p_col, p_row, p_x, p_nnz, p_csc = _csc_ptrs(sparse.triu(P))
            if p_m != p_n:
                raise ValueError("P must be square")
            a_m, a_n, a_col, a_row, a_x, a_nnz, a_csc = _csc_ptrs(A)
            if a_n != p_n:
                raise ValueError("A column dimension must match P")
            qv = np.asarray(q, dtype=np.float64).reshape(-1)
            bv = np.asarray(b, dtype=np.float64).reshape(-1)
            if qv.size != p_n:
                raise ValueError("q dimension must match P")
            if bv.size != a_m:
                raise ValueError("b dimension must match A rows")
            q_arr = (c_double * p_n)(*[float(v) for v in qv])
            b_arr = (c_double * a_m)(*[float(v) for v in bv])
            kind_arr, dim_arr, alpha_arr, n_cones = _cones_ptrs(cones)
            ptr = lib_.conix_setup(
                p_n,
                a_m,
                p_col,
                p_row,
                p_x,
                p_nnz,
                q_arr,
                a_col,
                a_row,
                a_x,
                a_nnz,
                b_arr,
                kind_arr,
                dim_arr,
                alpha_arr,
                n_cones,
                int(engine),
            )
            if not ptr:
                raise RuntimeError(cx.last_error() or "conix_setup failed")
            self._lib = lib_
            self._cx = cx
            self._ptr = ptr
            self._n = p_n
            self._m = a_m
            self._P = p_csc
            self._A = a_csc
            self.configure(
                max_iter=max_iter,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                verbose=verbose,
                polish=polish,
            )

        def close(self) -> None:
            if getattr(self, "_ptr", None):
                self._lib.conix_free(self._ptr)
                self._ptr = None

        def __del__(self) -> None:
            self.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

        def configure(
            self,
            *,
            max_iter: int | None = None,
            eps_abs: float | None = None,
            eps_rel: float | None = None,
            verbose: bool | None = None,
            engine: int | None = None,
            polish: bool | None = None,
            **_ignored,
        ) -> None:
            rc = self._lib.conix_configure(
                self._ptr,
                -1 if max_iter is None else int(max_iter),
                float("nan") if eps_abs is None else float(eps_abs),
                float("nan") if eps_rel is None else float(eps_rel),
                -1 if verbose is None else (1 if verbose else 0),
                -1 if engine is None else int(engine),
                -1 if polish is None else (1 if polish else 0),
            )
            if rc != 0:
                raise RuntimeError(self._cx.last_error())

        def solve(self) -> SolverSolution:
            import time

            t0 = time.perf_counter()
            if self._lib.conix_solve(self._ptr) != 0:
                raise RuntimeError(self._cx.last_error())
            dt = time.perf_counter() - t0
            n, m = self._n, self._m
            xbuf = (c_double * n)()
            ybuf = (c_double * m)()
            sbuf = (c_double * m)()
            self._lib.conix_x(self._ptr, xbuf, n)
            self._lib.conix_z(self._ptr, ybuf, m)
            self._lib.conix_s(self._ptr, sbuf, m)
            pri = c_double()
            dual = c_double()
            gap = c_double()
            cone = c_double()
            comp = c_double()
            self._lib.conix_residuals(
                self._ptr,
                ctypes.byref(pri),
                ctypes.byref(dual),
                ctypes.byref(gap),
                ctypes.byref(cone),
                ctypes.byref(comp),
            )
            st = int(self._lib.conix_status(self._ptr))
            return SolverSolution(
                x=np.array(xbuf, dtype=np.float64),
                y=np.array(ybuf, dtype=np.float64),
                s=np.array(sbuf, dtype=np.float64),
                obj_val=float(self._lib.conix_obj(self._ptr)),
                iter=int(self._lib.conix_iterations(self._ptr)),
                status=_STATUS.get(st, str(st)),
                r_prim=float(pri.value),
                r_dual=float(dual.value),
                solve_time=dt,
            )

        def update_p(self, P) -> None:
            _, n, col, row, data, nnz, csc = _csc_ptrs(sparse.triu(P))
            if n != self._n:
                raise ValueError("P size mismatch")
            if self._lib.conix_update_p(self._ptr, n, col, row, data, nnz) != 0:
                raise RuntimeError(self._cx.last_error())
            self._P = csc

        def update_a(self, A) -> None:
            m, n, col, row, data, nnz, csc = _csc_ptrs(A)
            if n != self._n or m != self._m:
                raise RuntimeError(self._cx.last_error() or "A size/pattern change is R2")
            if self._lib.conix_update_a(self._ptr, m, n, col, row, data, nnz) != 0:
                raise RuntimeError(self._cx.last_error())
            self._A = csc

        def update_q(self, q) -> None:
            qv = np.asarray(q, dtype=np.float64).reshape(-1)
            if qv.size != self._n:
                raise ValueError("q size mismatch")
            arr = (c_double * self._n)(*[float(v) for v in qv])
            if self._lib.conix_update_q(self._ptr, arr, self._n) != 0:
                raise RuntimeError(self._cx.last_error())

        def update_b(self, b) -> None:
            bv = np.asarray(b, dtype=np.float64).reshape(-1)
            if bv.size != self._m:
                raise ValueError("b size mismatch")
            # C ABI update_b may live on update path via configure-less update_q style;
            # fall back to re-setup is not available — require matching ctypes export.
            if not hasattr(self._lib, "conix_update_b"):
                raise NotImplementedError("conix_update_b not in this libconix build")
            arr = (c_double * self._m)(*[float(v) for v in bv])
            if self._lib.conix_update_b(self._ptr, arr, self._m) != 0:
                raise RuntimeError(self._cx.last_error())

        def warm_start(self, x=None, s=None, z=None) -> None:
            def _ptr(v, n):
                if v is None:
                    return None
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
                if arr.size != n:
                    raise ValueError("warm_start size mismatch")
                return (c_double * n)(*[float(t) for t in arr])

            xp = _ptr(x, self._n)
            sp_ = _ptr(s, self._m)
            zp = _ptr(z, self._m)
            if self._lib.conix_warm_start(self._ptr, xp, sp_, zp) != 0:
                raise RuntimeError(self._cx.last_error())

    def solve(P, q, A, b, cones, **kwargs) -> SolverSolution:
        with ConixSolver(P, q, A, b, cones, **kwargs) as sol:
            return sol.solve()

    __all__ = ["ConixSolver", "SolverSolution", "solve"]
