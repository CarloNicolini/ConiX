#!/usr/bin/env python3
"""Sequence timings for SCS 3.x on the same R0 Markowitz / R1 CVaR families
used in tests/compare.rs. Persistent `update` covers R0 (c/b only). R1 CVaR
rebuilds the workspace because the Python SCS API cannot update A.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse

import scs

U32_MAX = float(0xFFFFFFFF)


def lcg(state: int) -> tuple[int, float]:
    state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return state, float(state >> 33) / U32_MAX


def returns(t: int, n: int, seed: int) -> np.ndarray:
    s = seed
    out = np.zeros((t, n))
    for i in range(t):
        for j in range(n):
            s, u = lcg(s)
            out[i, j] = 0.02 * (u - 0.5)
    return out


def mean_variance(n: int, mu: np.ndarray):
    p = sparse.eye(n, format="csc")
    q = -mu
    rows, cols, data = [], [], []
    b = [1.0]
    for j in range(n):
        rows.append(0)
        cols.append(j)
        data.append(1.0)
    for j in range(n):
        rows.append(1 + j)
        cols.append(j)
        data.append(1.0)
        b.append(1.0)
    for j in range(n):
        rows.append(1 + n + j)
        cols.append(j)
        data.append(-1.0)
        b.append(0.0)
    a = sparse.csc_matrix((data, (rows, cols)), shape=(1 + 2 * n, n))
    cone = {"z": 1, "l": 2 * n}
    return p, q, a, np.array(b, dtype=float), cone


def cvar(r: np.ndarray, beta: float = 0.8):
    t, n = r.shape
    nv = n + 1 + t
    p = sparse.csc_matrix((nv, nv))
    tail = 1.0 / ((1.0 - beta) * t)
    q = np.zeros(nv)
    q[n] = 1.0
    q[n + 1 :] = tail
    rows, cols, data = [], [], []
    b = []
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
        b.append(1.0)
        row += 1
    for j in range(n):
        rows.append(row)
        cols.append(j)
        data.append(-1.0)
        b.append(0.0)
        row += 1
    for s in range(t):
        for j in range(n):
            rows.append(row)
            cols.append(j)
            data.append(-float(r[s, j]))
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
    a = sparse.csc_matrix((data, (rows, cols)), shape=(row, nv))
    cone = {"z": 1, "l": row - 1}
    return p, q, a, np.array(b, dtype=float), cone


def settings():
    return dict(eps_abs=1e-6, eps_rel=1e-6, verbose=False, max_iters=10_000)


def r0_markowitz():
    n, dates = 8, 20
    mu0 = np.full(n, 0.01)
    p, q, a, b, cone = mean_variance(n, mu0)
    data = {"P": p, "A": a, "b": b, "c": q}
    solver = scs.SCS(data, cone, **settings())
    t0 = time.perf_counter()
    sol = solver.solve()
    elapsed = time.perf_counter() - t0
    assert sol["info"]["status_val"] == scs.SOLVED, sol["info"]
    seed = 1
    for _ in range(1, dates):
        mu = np.zeros(n)
        for j in range(n):
            seed, u = lcg(seed)
            mu[j] = 0.005 + 0.02 * u
        q = -mu
        solver.update(c=q)
        t1 = time.perf_counter()
        sol = solver.solve()
        elapsed += time.perf_counter() - t1
        assert sol["info"]["status_val"] == scs.SOLVED, sol["info"]
    print(f"R0 Markowitz n={n} dates={dates}: SCS-update={elapsed:.4f}s")


def r1_cvar():
    n, t, dates = 5, 12, 10
    r0 = returns(t, n, 7)
    p, q, a, b, cone = cvar(r0)
    data = {"P": p, "A": a, "b": b, "c": q}
    solver = scs.SCS(data, cone, **settings())
    t0 = time.perf_counter()
    sol = solver.solve()
    elapsed = time.perf_counter() - t0
    n_solved = int(sol["info"]["status_val"] == scs.SOLVED)
    seed = 11
    for d in range(1, dates):
        r = returns(t, n, 100 + d + seed)
        seed = (seed + 1) & 0xFFFFFFFFFFFFFFFF
        p, q, a, b, cone = cvar(r)
        data = {"P": p, "A": a, "b": b, "c": q}
        solver = scs.SCS(data, cone, **settings())
        t1 = time.perf_counter()
        sol = solver.solve()
        elapsed += time.perf_counter() - t1
        if sol["info"]["status_val"] == scs.SOLVED:
            n_solved += 1
    print(
        f"R1 CVaR n={n} T={t} dates={dates}: SCS-cold={elapsed:.4f}s  solved={n_solved}/{dates}"
    )


if __name__ == "__main__":
    r0_markowitz()
    r1_cvar()
