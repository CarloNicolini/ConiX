"""Compare ConiX against Clarabel and OSQP on rolling finance QCPs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import clarabel
import numpy as np
import osqp

import conix as cx
from conix import qcp as qcp_mod


@dataclass
class TimingResult:
    conix_warm: float
    conix_cold: float
    clarabel_warm: float
    clarabel_cold: float
    osqp_warm: float
    osqp_cold: float

    @staticmethod
    def speedup(baseline: float, conix: float) -> float:
        if conix <= 0.0:
            return float("nan")
        return baseline / conix


def _clarabel_settings() -> clarabel.DefaultSettings:
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.max_iter = 20_000
    s.tol_gap_abs = 1e-6
    s.tol_gap_rel = 1e-6
    s.tol_feas = 1e-6
    s.equilibrate_enable = True
    s.presolve_enable = False
    return s


def _osqp_settings() -> dict:
    return {
        "verbose": False,
        "warm_start": True,
        "polish": True,
        "eps_abs": 1e-6,
        "eps_rel": 1e-6,
        "max_iter": 20_000,
    }


def _solve_clarabel_cold(q: qcp_mod.Qcp) -> None:
    p = qcp_mod.upper_triangle(q.p)
    cones = qcp_mod.clarabel_cones(q.m)
    solver = clarabel.DefaultSolver(
        p, q.q, q.a, q.b, cones, _clarabel_settings()
    )
    solver.solve()


def _solve_osqp_cold(q: qcp_mod.Qcp) -> None:
    p = qcp_mod.upper_triangle(q.p)
    lo, hi = qcp_mod.osqp_bounds(q)
    prob = osqp.OSQP()
    prob.setup(p, q.q, q.a, lo, hi, **_osqp_settings())
    prob.solve()


def _bench_sequence(
    build_qcp: Callable[[int], qcp_mod.Qcp],
    conix_open_at: Callable[[int], cx.Workspace],
    conix_update: Callable[[cx.Workspace, int], None],
    dates: int,
) -> TimingResult:
    q0 = build_qcp(0)
    p0 = qcp_mod.upper_triangle(q0.p)
    lo0, hi0 = qcp_mod.osqp_bounds(q0)
    cones0 = qcp_mod.clarabel_cones(q0.m)

    t_conix_warm = 0.0
    t_conix_cold = 0.0
    t_clar_warm = 0.0
    t_clar_cold = 0.0
    t_osqp_warm = 0.0
    t_osqp_cold = 0.0

    with conix_open_at(0) as ws:
        t0 = time.perf_counter()
        ws.solve()
        t_conix_warm += time.perf_counter() - t0

        clar = clarabel.DefaultSolver(
            p0, q0.q, q0.a, q0.b, cones0, _clarabel_settings()
        )
        t0 = time.perf_counter()
        clar.solve()
        t_clar_warm += time.perf_counter() - t0

        osqp_prob = osqp.OSQP()
        osqp_prob.setup(p0, q0.q, q0.a, lo0, hi0, **_osqp_settings())
        t0 = time.perf_counter()
        osqp_prob.solve()
        t_osqp_warm += time.perf_counter() - t0

        for d in range(1, dates):
            q = build_qcp(d)
            p = qcp_mod.upper_triangle(q.p)
            lo, hi = qcp_mod.osqp_bounds(q)

            t0 = time.perf_counter()
            conix_update(ws, d)
            ws.solve()
            t_conix_warm += time.perf_counter() - t0

            t0 = time.perf_counter()
            with conix_open_at(d) as cold_ws:
                cold_ws.solve()
            t_conix_cold += time.perf_counter() - t0

            t0 = time.perf_counter()
            _solve_clarabel_cold(q)
            t_clar_cold += time.perf_counter() - t0

            t0 = time.perf_counter()
            clar.update(A=q.a, b=q.b, q=q.q)
            if q.p.nnz:
                clar.update(P=p)
            clar.solve()
            t_clar_warm += time.perf_counter() - t0

            t0 = time.perf_counter()
            _solve_osqp_cold(q)
            t_osqp_cold += time.perf_counter() - t0

            t0 = time.perf_counter()
            osqp_prob.update(
                q=q.q,
                Ax=q.a.data,
                Ap=q.a.indptr,
                Ai=q.a.indices,
                l=lo,
                u=hi,
            )
            if q.p.nnz:
                osqp_prob.update(Px=p.data, Pp=p.indptr, Pi=p.indices)
            osqp_prob.solve()
            t_osqp_warm += time.perf_counter() - t0

    t0 = time.perf_counter()
    with conix_open_at(0) as cold_ws:
        cold_ws.solve()
    t_conix_cold += time.perf_counter() - t0

    t0 = time.perf_counter()
    _solve_clarabel_cold(q0)
    t_clar_cold += time.perf_counter() - t0

    t0 = time.perf_counter()
    _solve_osqp_cold(q0)
    t_osqp_cold += time.perf_counter() - t0

    return TimingResult(
        conix_warm=t_conix_warm,
        conix_cold=t_conix_cold,
        clarabel_warm=t_clar_warm,
        clarabel_cold=t_clar_cold,
        osqp_warm=t_osqp_warm,
        osqp_cold=t_osqp_cold,
    )


def _panels(t: int, n: int, dates: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((t, n)) * 0.02 for _ in range(dates)]


def bench_mean_variance(
    t: int,
    n: int,
    dates: int,
    seed: int = 42,
    lam: float = 1.0,
) -> TimingResult:
    panels = _panels(t, n, dates, seed)
    l = np.zeros(n)
    u = np.ones(n)

    def stats(d: int) -> tuple[np.ndarray, np.ndarray]:
        r = panels[d]
        return np.cov(r, rowvar=False), r.mean(axis=0)

    def build_q(d: int) -> qcp_mod.Qcp:
        sigma, mu = stats(d)
        return qcp_mod.mean_variance(sigma, mu, l, u, lam)

    def open_at(d: int) -> cx.Workspace:
        sigma, mu = stats(d)
        return cx.mean_variance(sigma, mu, l.tolist(), u.tolist(), lam)

    def update_ws(ws: cx.Workspace, d: int) -> None:
        sigma, mu = stats(d)
        ws.update_mean_variance(sigma, mu, l.tolist(), u.tolist(), lam)

    return _bench_sequence(build_q, open_at, update_ws, dates)


def bench_cvar(
    t: int,
    n: int,
    dates: int,
    seed: int = 42,
    beta: float = 0.95,
) -> TimingResult:
    panels = _panels(t, n, dates, seed)
    l = np.zeros(n)
    u = np.ones(n)

    def build_q(d: int) -> qcp_mod.Qcp:
        return qcp_mod.cvar(panels[d], beta, l, u)

    def open_at(d: int) -> cx.Workspace:
        return cx.cvar(panels[d].tolist(), beta, l.tolist(), u.tolist())

    def update_ws(ws: cx.Workspace, d: int) -> None:
        ws.update_cvar(panels[d].tolist(), beta, l.tolist(), u.tolist())

    return _bench_sequence(build_q, open_at, update_ws, dates)


def bench_mad(
    t: int,
    n: int,
    dates: int,
    seed: int = 42,
) -> TimingResult:
    panels = _panels(t, n, dates, seed)
    l = np.zeros(n)
    u = np.ones(n)
    probs = np.full(t, 1.0 / t)

    def build_q(d: int) -> qcp_mod.Qcp:
        return qcp_mod.mad(panels[d], probs, l, u)

    def open_at(d: int) -> cx.Workspace:
        return cx.mad(panels[d].tolist(), probs.tolist(), l.tolist(), u.tolist())

    def update_ws(ws: cx.Workspace, d: int) -> None:
        ws.update_mad(panels[d].tolist(), probs.tolist(), l.tolist(), u.tolist())

    return _bench_sequence(build_q, open_at, update_ws, dates)


def format_report(
    name: str,
    t: int,
    n: int,
    dates: int,
    r: TimingResult,
) -> str:
    lines = [
        f"=== {name}  T={t}  N={n}  dates={dates} ===",
        f"  ConiX warm:     {r.conix_warm:.4f}s",
        f"  ConiX cold:     {r.conix_cold:.4f}s",
        f"  Clarabel warm:  {r.clarabel_warm:.4f}s",
        f"  Clarabel cold:  {r.clarabel_cold:.4f}s",
        f"  OSQP warm:      {r.osqp_warm:.4f}s",
        f"  OSQP cold:      {r.osqp_cold:.4f}s",
        "  Speedup vs Clarabel (warm): "
        f"{TimingResult.speedup(r.clarabel_warm, r.conix_warm):.2f}x",
        "  Speedup vs Clarabel (cold): "
        f"{TimingResult.speedup(r.clarabel_cold, r.conix_cold):.2f}x",
        "  Speedup vs OSQP (warm): "
        f"{TimingResult.speedup(r.osqp_warm, r.conix_warm):.2f}x",
        "  Speedup vs OSQP (cold): "
        f"{TimingResult.speedup(r.osqp_cold, r.conix_cold):.2f}x",
    ]
    return "\n".join(lines)
