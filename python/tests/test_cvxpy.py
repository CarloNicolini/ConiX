"""Correctness tests: ConiX vs Clarabel (native + CVXPY)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

_default_lib = ROOT / "target" / "release" / "libconix.so"
if _default_lib.exists() and "CONIX_LIB" not in os.environ:
    os.environ["CONIX_LIB"] = str(_default_lib)


def _have_lib() -> bool:
    try:
        import conix as cx

        cx.lib()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _have_lib(), reason="libconix not built")


def textbook_qp():
    P = sparse.triu(sparse.csc_matrix([[6.0, 0.0], [0.0, 4.0]])).tocsc()
    q = np.array([-1.0, -4.0])
    A = sparse.csc_matrix(
        [
            [1.0, -2.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    b = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    cones = [("zero", 1), ("nonnegative", 4)]
    return P, q, A, b, cones


def _clarabel_solve(P, q, A, b, cones):
    import clarabel

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 20_000
    settings.tol_gap_abs = 1e-6
    settings.tol_gap_rel = 1e-6
    settings.tol_feas = 1e-6
    settings.presolve_enable = False
    ccones = []
    for c in cones:
        kind = c[0]
        if kind == "zero":
            ccones.append(clarabel.ZeroConeT(c[1]))
        elif kind == "nonnegative":
            ccones.append(clarabel.NonnegativeConeT(c[1]))
        elif kind == "soc":
            ccones.append(clarabel.SecondOrderConeT(c[1]))
        elif kind == "exp":
            ccones.append(clarabel.ExponentialConeT())
        elif kind == "power":
            ccones.append(clarabel.PowerConeT(c[1]))
        elif kind == "psd":
            ccones.append(clarabel.PSDTriangleConeT(c[1]))
        else:
            raise ValueError(kind)
    solver = clarabel.DefaultSolver(P, q, A, b, ccones, settings)
    return solver.solve()


def test_native_qp_matches_clarabel():
    from conix.solver import ConixSolver

    P, q, A, b, cones = textbook_qp()
    with ConixSolver(P, q, A, b, cones, eps_abs=1e-6, eps_rel=1e-6) as sol:
        cx = sol.solve()
    assert cx.status == "Solved"
    cl = _clarabel_solve(P, q, A, b, cones)
    np.testing.assert_allclose(cx.x, cl.x, atol=1e-3)
    assert cx.obj_val == pytest.approx(cl.obj_val, abs=1e-3)


def test_box_qp():
    from conix.solver import ConixSolver

    P = sparse.eye(2, format="csc")
    q = np.array([-1.0, -1.0])
    A = sparse.csc_matrix(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    b = np.array([1.0, 1.0, 0.0, 0.0])
    cones = [("nonnegative", 4)]
    with ConixSolver(P, q, A, b, cones) as s:
        sol = s.solve()
    assert sol.status == "Solved"
    np.testing.assert_allclose(sol.x, [1.0, 1.0], atol=1e-3)


def test_update_q_sequence():
    from conix.solver import ConixSolver

    P, q, A, b, cones = textbook_qp()
    with ConixSolver(P, q, A, b, cones) as s:
        s1 = s.solve()
        assert s1.status == "Solved"
        q2 = q * 1.1
        s.update_q(q2)
        s2 = s.solve()
        assert s2.status == "Solved"
        fresh = ConixSolver(P, q2, A, b, cones).solve()
        np.testing.assert_allclose(s2.x, fresh.x, atol=2e-3)


def test_cvxpy_registration_vs_clarabel():
    pytest.importorskip("cvxpy")
    import cvxpy as cp
    from conix.cvxpy_interface import CONIX, register

    register()
    x = cp.Variable(2)
    P = np.array([[6.0, 0.0], [0.0, 4.0]])
    q = np.array([-1.0, -4.0])
    constraints = [x[0] - 2 * x[1] == 0, x >= -1, x <= 1]
    obj = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)

    prob_cx = cp.Problem(obj, constraints)
    prob_cx.solve(solver="CONIX")
    assert prob_cx.status in ("optimal", "optimal_inaccurate")

    x2 = cp.Variable(2)
    prob_cl = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x2, P) + q @ x2),
        [x2[0] - 2 * x2[1] == 0, x2 >= -1, x2 <= 1],
    )
    prob_cl.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(x.value, x2.value, atol=2e-3)
    assert prob_cx.value == pytest.approx(prob_cl.value, abs=2e-3)

    # Also accept solver instance
    x3 = cp.Variable(2)
    prob3 = cp.Problem(cp.Minimize(cp.sum_squares(x3)), [x3 >= 0, cp.sum(x3) == 1])
    prob3.solve(solver=CONIX())
    np.testing.assert_allclose(x3.value, [0.5, 0.5], atol=2e-2)


def test_cvxpy_socp_vs_clarabel():
    pytest.importorskip("cvxpy")
    import cvxpy as cp
    from conix.cvxpy_interface import register

    register()
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [cp.sum(x) == 1, x >= 0])
    x.value = None
    prob.solve(solver="CONIX")
    assert prob.status in ("optimal", "optimal_inaccurate")
    x_cl = cp.Variable(3)
    prob_cl = cp.Problem(cp.Minimize(cp.norm(x_cl, 2)), [cp.sum(x_cl) == 1, x_cl >= 0])
    prob_cl.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(x.value, x_cl.value, atol=5e-3)
