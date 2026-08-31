"""Build finance QCPs in SciPy CSC form (matches Rust ``models.rs``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Qcp:
    p: sparse.csc_matrix
    q: np.ndarray
    a: sparse.csc_matrix
    b: np.ndarray
    n: int
    m: int


def _csc_from_triplets(m: int, n: int, rows: list[int], cols: list[int], data: list[float]) -> sparse.csc_matrix:
    if not data:
        return sparse.csc_matrix((m, n), dtype=np.float64)
    coo = sparse.coo_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float64)
    return coo.tocsc()


def mean_variance(
    sigma: np.ndarray,
    mu: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    lam: float,
) -> Qcp:
    n = mu.size
    p = sparse.triu(lam * sigma, format="csc")
    q = -mu.astype(np.float64, copy=False)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b: list[float] = [1.0]
    for j in range(n):
        rows.append(0)
        cols.append(j)
        data.append(1.0)
    for j in range(n):
        rows.append(1 + j)
        cols.append(j)
        data.append(1.0)
        b.append(float(u[j]))
    for j in range(n):
        rows.append(1 + n + j)
        cols.append(j)
        data.append(-1.0)
        b.append(float(-l[j]))
    a = _csc_from_triplets(1 + 2 * n, n, rows, cols, data)
    return Qcp(p=p, q=q, a=a, b=np.asarray(b, dtype=np.float64), n=n, m=a.shape[0])


def cvar(returns: np.ndarray, beta: float, l: np.ndarray, u: np.ndarray) -> Qcp:
    t, n = returns.shape
    nv = n + 1 + t
    tail = 1.0 / ((1.0 - beta) * t)
    q = np.zeros(nv, dtype=np.float64)
    q[n] = 1.0
    q[n + 1 :] = tail
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b: list[float] = []
    row = 0
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(1.0)
    b.append(1.0)
    row += 1
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(1.0)
        b.append(float(u[j]))
        row += 1
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(-1.0)
        b.append(float(-l[j]))
        row += 1
    for s in range(t):
        for j in range(n):
            rows.append(row)
            cols.append(j)
            data.append(-float(returns[s, j]))
        rows.append(row)
        cols.append(n)
        data.append(-1.0)
        rows.append(row)
        cols.append(n + 1 + s)
        data.append(-1.0)
        b.append(0.0)
        row += 1
    for s in range(t):
        rows.append(row)
        cols.append(n + 1 + s)
        data.append(-1.0)
        b.append(0.0)
        row += 1
    p = sparse.csc_matrix((nv, nv), dtype=np.float64)
    a = _csc_from_triplets(row, nv, rows, cols, data)
    return Qcp(p=p, q=q, a=a, b=np.asarray(b, dtype=np.float64), n=nv, m=row)


def mad(returns: np.ndarray, probs: np.ndarray, l: np.ndarray, u: np.ndarray) -> Qcp:
    t, n = returns.shape
    nv = n + t
    q = np.zeros(nv, dtype=np.float64)
    q[n:] = probs
    rbar = probs @ returns
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b: list[float] = []
    row = 0
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(1.0)
    b.append(1.0)
    row += 1
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(1.0)
        b.append(float(u[j]))
        row += 1
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(-1.0)
        b.append(float(-l[j]))
        row += 1
    for s in range(t):
        c = returns[s] - rbar
        for j in range(n):
            rows.append(row)
            cols.append(j)
            data.append(float(c[j]))
        rows.append(row)
        cols.append(n + s)
        data.append(-1.0)
        b.append(0.0)
        row += 1
        for j in range(n):
            rows.append(row)
            cols.append(j)
            data.append(float(-c[j]))
        rows.append(row)
        cols.append(n + s)
        data.append(-1.0)
        b.append(0.0)
        row += 1
    p = sparse.csc_matrix((nv, nv), dtype=np.float64)
    a = _csc_from_triplets(row, nv, rows, cols, data)
    return Qcp(p=p, q=q, a=a, b=np.asarray(b, dtype=np.float64), n=nv, m=row)


def osqp_bounds(qcp: Qcp) -> tuple[np.ndarray, np.ndarray]:
    """Box form ``l <= A x <= u`` for polyhedral ConiX models."""
    m = qcp.m
    lo = np.zeros(m, dtype=np.float64)
    hi = np.zeros(m, dtype=np.float64)
    # budget row
    lo[0] = hi[0] = qcp.b[0]
    # remaining rows: one-sided NN cones → -inf <= (Ax)_i <= b_i
    lo[1:] = -np.inf
    hi[1:] = qcp.b[1:]
    return lo, hi


def clarabel_cones(m: int) -> list:
    """Zero + nonnegative split used by all finance builders in ``models.rs``."""
    import clarabel

    return [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(m - 1)]


def upper_triangle(p: sparse.csc_matrix) -> sparse.csc_matrix:
    """Symmetrize to upper triangle for Clarabel / OSQP."""
    return sparse.triu(p, format="csc")


def upper_triplet(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper triangle CSC arrays for ``conix_mean_variance`` C API."""
    n = sigma.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for j in range(n):
        for i in range(j + 1):
            rows.append(i)
            cols.append(j)
            data.append(float(sigma[i, j]))
    col_ptr = np.zeros(n + 1, dtype=np.uint64)
    row_idx = np.asarray(rows, dtype=np.uint64)
    x = np.asarray(data, dtype=np.float64)
    k = 0
    for j in range(n):
        col_ptr[j] = k
        while k < len(cols) and cols[k] == j:
            k += 1
    col_ptr[n] = len(data)
    return col_ptr, row_idx, x
