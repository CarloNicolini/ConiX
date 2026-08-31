"""CVXPY QP solved with CONIX (Clarabel-compatible form)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import cvxpy as cp
import numpy as np
from conix.cvxpy_interface import register

register()

P = np.array([[6.0, 0.0], [0.0, 4.0]])
q = np.array([-1.0, -4.0])
x = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
    [x[0] - 2 * x[1] == 0, x >= -1, x <= 1],
)
prob.solve(solver="CONIX", verbose=False)
print(f"status = {prob.status}")
print(f"x      = {x.value}")
print(f"obj    = {prob.value}")
