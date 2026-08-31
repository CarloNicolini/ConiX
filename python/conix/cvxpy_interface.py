"""CVXPY adapter for ConiX (adapted from COSMO.rs ``cvxpy_interface``).

Extracts CVXPY canonical conic data and calls ConiX through the ctypes ABI.
Form matches Clarabel: ``Ax + s = b``, ``s ∈ K``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    import cvxpy as cp
    import cvxpy.settings as s
    from cvxpy.constraints import ExpCone, PowCone3D, SOC
    from cvxpy.reductions.solution import Solution, failure_solution
    from cvxpy.reductions.solvers import utilities
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
except ImportError as exc:  # pragma: no cover
    raise ImportError("cvxpy is required for conix.cvxpy_interface") from exc

try:
    from cvxpy.constraints import SvecPSD
except ImportError:  # older cvxpy
    SvecPSD = None

from .solver import ConixSolver


class CONIX(ConicSolver):
    """CVXPY interface for the ConiX sequential conic optimizer."""

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D]
    if SvecPSD is not None:
        SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS + [SvecPSD]
    REQUIRED_MODULES = ("conix",)
    EXP_CONE_ORDER = [0, 1, 2]
    # Match Clarabel / ConiX packed PSD convention.
    try:
        from cvxpy.utilities.psd_utils import TriangleKind

        PSD_TRIANGLE_KIND = TriangleKind.UPPER
        PSD_SQRT2_SCALING = True
    except Exception:  # pragma: no cover
        pass

    STATUS_MAP = {
        "Solved": s.OPTIMAL,
        "PrimalInfeasible": s.INFEASIBLE,
        "DualInfeasible": s.UNBOUNDED,
        "MaxIters": s.OPTIMAL_INACCURATE,
        "Indeterminate": s.SOLVER_ERROR,
        "Unsolved": s.SOLVER_ERROR,
    }

    # If MaxIters residuals are still large, demote to USER_LIMIT in invert.
    _INACCURATE_RES_GATE = 1e-3

    def name(self):
        return "CONIX"

    def import_solver(self) -> None:
        import conix  # noqa: F401

    def supports_quad_obj(self) -> bool:
        return True

    def invert(self, solution, inverse_data):
        attr = {
            s.SOLVE_TIME: getattr(solution, "solve_time", 0.0),
            s.SETUP_TIME: getattr(solution, "setup_time", 0.0),
            s.NUM_ITERS: getattr(solution, "iter", 0),
            s.EXTRA_STATS: {
                "r_prim": getattr(solution, "r_prim", None),
                "r_dual": getattr(solution, "r_dual", None),
            },
        }
        raw = str(solution.status)
        status = self.STATUS_MAP.get(raw, s.SOLVER_ERROR)
        if raw == "MaxIters":
            rp = float(getattr(solution, "r_prim", 1.0) or 1.0)
            rd = float(getattr(solution, "r_dual", 1.0) or 1.0)
            if max(rp, rd) > self._INACCURATE_RES_GATE:
                status = s.USER_LIMIT
            elif solution.x is None:
                status = s.USER_LIMIT
        y = np.array(solution.y, dtype=float) if solution.y is not None else None
        dual_vars = {}
        if y is not None:
            zero_idx = inverse_data[ConicSolver.DIMS].zero
            eq_dual_vars = utilities.get_dual_values(
                y[:zero_idx],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual_vars = utilities.get_dual_values(
                y[zero_idx:],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            dual_vars = eq_dual_vars | ineq_dual_vars

        if status in s.SOLUTION_PRESENT:
            primal_val = float(solution.obj_val)
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: np.array(solution.x, dtype=float)}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        return failure_solution(status, attr, dual_vars)

    @staticmethod
    def dims_to_cones(dims):
        cones = []
        if dims.zero > 0:
            cones.append(("zero", int(dims.zero)))
        if dims.nonneg > 0:
            cones.append(("nonnegative", int(dims.nonneg)))
        for dim in dims.soc:
            cones.append(("soc", int(dim)))
        if getattr(dims, "psd", None):
            for side in dims.psd:
                cones.append(("psd", int(side)))
        for _ in range(int(dims.exp)):
            cones.append(("exp",))
        for alpha in dims.p3d:
            cones.append(("power", float(alpha)))
        if getattr(dims, "pnd", None) and len(dims.pnd) > 0:
            raise ValueError(
                "ConiX CVXPY interface does not expose ND power cones yet; "
                "use the native GenPower C API / finance builders"
            )
        return cones

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        A = sp.csc_matrix(data[s.A])
        b = np.array(data[s.B], dtype=float)
        q = np.array(data[s.C], dtype=float)
        if s.P in data:
            P = sp.csc_matrix(sp.triu(data[s.P]))
        else:
            P = sp.csc_matrix((q.size, q.size))

        cones = self.dims_to_cones(data[self.DIMS])
        opts = dict(solver_opts)
        opts.pop("use_quad_obj", None)
        opts["verbose"] = bool(verbose)
        # Engine defaults to AUTO (IPM on cold polyhedral setup). Pass
        # ``engine=conix.ADMM`` for CVXPY CVaR/MAD graphs where ADMM is more
        # reliable on the canonicalized form.

        # Reuse a cached workspace when the sparsity pattern is unchanged.
        cached = None
        if warm_start and solver_cache is not None and self.name() in solver_cache:
            cached = solver_cache[self.name()]

        solver = None
        if cached is not None and isinstance(cached, dict) and "solver" in cached:
            solver = cached["solver"]
            try:
                solver.update_q(q)
                solver.update_b(b)
                if P.nnz or cached.get("had_p", False):
                    solver.update_p(P)
                solver.update_a(A)
                solver.configure(**{k: v for k, v in opts.items() if k in {
                    "max_iter", "eps_abs", "eps_rel", "verbose", "engine", "polish"
                }})
                if warm_start and "result" in cached:
                    old = cached["result"]
                    solver.warm_start(x=old.x, y=old.y, s=old.s)
            except Exception:
                solver.close()
                solver = None

        if solver is None:
            solver = ConixSolver(P, q, A, b, cones, **opts)
            if warm_start and cached is not None and isinstance(cached, dict) and "result" in cached:
                old = cached["result"]
                try:
                    solver.warm_start(x=list(old.x), y=list(old.y), s=list(old.s))
                except Exception:
                    pass

        result = solver.solve()
        if solver_cache is not None:
            solver_cache[self.name()] = {
                "solver": solver,
                "result": result,
                "had_p": bool(P.nnz),
            }
        return result

    def cite(self, data):
        return (
            "@misc{ConiX,\n"
            "  title  = {ConiX: Sequential conic optimizer for rolling convex programs},\n"
            "  author = {Carlo Nicolini},\n"
            "  year   = {2026},\n"
            "  url    = {https://github.com/CarloNicolini/ConiX},\n"
            "}"
        )


def register() -> None:
    """Register CONIX so ``problem.solve(solver='CONIX')`` works.

    Also refreshes skfolio's import-time ``INSTALLED_SOLVERS`` snapshot when
    skfolio was imported before this call (common with skfolio-accelerate).
    """
    import cvxpy.reductions.solvers.defines as ds

    name = "CONIX"
    s.CONIX = name
    inst = CONIX()
    ds.SOLVER_MAP_CONIC[name] = inst
    if name not in ds.INSTALLED_SOLVERS:
        ds.INSTALLED_SOLVERS.append(name)
    if name not in ds.CONIC_SOLVERS:
        ds.CONIC_SOLVERS.append(name)
    if not hasattr(cp, "CONIX"):
        setattr(cp, "CONIX", name)

    # skfolio freezes ``cp.installed_solvers()`` at import time.
    try:
        import skfolio.optimization.convex._base as sk_base

        if name not in sk_base.INSTALLED_SOLVERS:
            sk_base.INSTALLED_SOLVERS = list(sk_base.INSTALLED_SOLVERS) + [name]
    except ImportError:
        pass
