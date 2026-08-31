"""Sequential ConiX optimizer — Python package.

Primary path: the maturin-built native extension ``conix._conix`` (PyO3),
matching COSMO.rs packaging. A ctypes fallback remains for a plain
``cargo build --release`` shared library when the extension is absent.
"""

from __future__ import annotations

AUTO, ADMM, SPLITTING, IPM = 0, 1, 2, 3

try:
    from . import _conix as _native

    _HAS_NATIVE = True
except ImportError:  # pragma: no cover
    _native = None
    _HAS_NATIVE = False


def lib():
    """Return the native extension module, or the ctypes CDLL fallback."""
    if _HAS_NATIVE:
        return _native
    from . import _ctypes_api as ct

    return ct.lib()


def last_error() -> str:
    if _HAS_NATIVE:
        return ""
    from . import _ctypes_api as ct

    return ct.last_error()


if _HAS_NATIVE:
    ConixSolver = _native.ConixSolver
    Solution = _native.Solution
    Workspace = _native.Workspace
    solve = _native.solve
    cvar = _native.cvar
    evar = _native.evar
    mad = _native.mad
    cdar = _native.cdar
    mean_variance = _native.mean_variance
    AUTO = int(_native.AUTO)
    ADMM = int(_native.ADMM)
    SPLITTING = int(_native.SPLITTING)
    IPM = int(_native.IPM)
    SolverSolution = Solution
else:
    from ._ctypes_api import (  # noqa: F401
        Workspace,
        Solution,
        cvar,
        evar,
        mad,
        cdar,
        mean_variance,
    )
    from .solver import ConixSolver, SolverSolution, solve  # noqa: F401

try:
    from .cvxpy_interface import CONIX, register
except Exception:  # cvxpy is optional
    CONIX = None

    def register():
        raise ImportError("cvxpy is required to register CONIX")

__all__ = [
    "AUTO",
    "ADMM",
    "SPLITTING",
    "IPM",
    "Workspace",
    "Solution",
    "ConixSolver",
    "SolverSolution",
    "solve",
    "CONIX",
    "register",
    "cvar",
    "evar",
    "mad",
    "cdar",
    "mean_variance",
    "lib",
    "last_error",
]
