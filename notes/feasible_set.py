"""The feasible-set study: where chladni stops being accurate, and which stage limits it.  (2026-11-06)

Plan: `~/projects/python/FEASIBLE_SET_PLAN.md`, Part 2. Polygon families pushed toward hardness,
one task per (family, parameter, prec). Each task writes one JSON:

  base      `Eigenproblem.plan_and_solve` and its `SolveRecord.report(2)`: everything the solve
            computed, per level and per run (`chladni.report`), plus the warnings it raised.
            When the planner declines on the clearance gate (`plan.CLEARANCE_FLOOR`, a constant
            set from one 583 s trial, not a measured limit), every rung is solved directly
            instead and the best kept (`gate_bypassed`), so the sweep measures where solves fail.
  truth     the closed form or table where one exists (rectangles, the isosceles table, N = 3, 4),
            else one-sided bounds from domain monotonicity against the unit square
            (`truth_upper` / `truth_lower`), else nothing.
  fd        the derivative along the family parameter, two ways: Hadamard (vertex hat fields,
            chained through d(vertices)/dp) and central differences on the FROZEN base plan,
            Richardson over a step ladder with an error bar (noise from the levels' bounds,
            truncation from consecutive steps), as `hadamard_hard_shapes.py` FDref. Compared per
            UNIT: a level, or a whole chladni cluster (its sum).
  knobs     (`--knobs`, the pilot) one change at a time on the base plan: basis x0.6 and x1.5,
            boundary and interior collocation x2, three 1e-14 vertex jitters. What moves the
            values, and what does not, says which stage limits them.

Families (`FAMILIES`; `ntasks`, `pilot` list them):
  F1  iso      isosceles triangle, base 2, apex angle 60 -> 2 deg, plus the table heights
  F2  flat     triangle (0, 1, 0.5 + i t), t -> 0: two thin wedges and an angle near 180
  F3  split    rectangle 1 x (1 + 10^-s): a near-degenerate pair, closed form
  F4  aspect   rectangle a x 1, a = 1 -> 20, closed form
  F6  needle   unit square + an outward isosceles needle of length 0.5, base w -> 5e-4
  F6b finger   unit square + a rectangular finger, same w and length (no thin wedge)
  F7  ngon     regular N-gons, N = 3 -> 64
  F8  hexrad / hextan  a regular hexagon, one vertex moved radially (keeps a reflection) or
               tangentially (breaks every symmetry) by 10^-s
  F9  rightish rectangle 1 x 0.7 with one corner at 90 + delta deg, delta = +-10^-s and 0
  F10 chamfer  unit square, one corner cut by c -> 1e-4
  F11 dumbbell two unit squares joined by a neck of width w, length 0.5
  F12 notch    unit square with a V-notch, interior angle 190 -> 355 deg
  F13 shifted  an F1 triangle and an F6 needle translated by 10^t (exact invariance)
  F14 highk    60 levels on the square, iso h = 4, the 12-gon, the needle w = 0.05

Run:
  python notes/feasible_set.py ntasks [pilot]
  python notes/feasible_set.py run <family> <index> <prec> [--knobs]    JSON to stdout
  python notes/feasible_set.py task <i> --out DIR [--pilot]             the Slurm entry point
  python notes/feasible_set.py analyze DIR
"""
import json
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRECS = (1e-8, 1e-10, 1e-12)
N_EIGS = 15
FD_STEPS = (1e-3, 3e-4, 1e-4)          # relative to the parameter's scale (`Family.fd_scale`)
KNOB_JITTER = 1e-14


# --- families -------------------------------------------------------------------------------

def _rect(a, b):
    return np.array([0, a, a + 1j * b, 1j * b])


def _rect_truth(a, b, n):
    m = np.arange(1, 4 * n + 2)
    lam = np.sort((np.pi ** 2 * (m[:, None] ** 2 / a ** 2 + m[None, :] ** 2 / b ** 2)).ravel())
    return lam[:n]


SQUARE = _rect_truth(1.0, 1.0, 4 * N_EIGS)


def _iso(theta_deg):
    h = 1.0 / math.tan(math.radians(theta_deg) / 2)
    return np.array([1, 1j * h, -1])


def _iso_h(theta_deg):
    return 1.0 / math.tan(math.radians(theta_deg) / 2)


ISO_TABLE_H = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0)


def _iso_truth(theta, n):
    from chladni.reference.tables import iso_tri_eigs
    h = _iso_h(theta)
    for ht in ISO_TABLE_H:
        if abs(h - ht) < 1e-12 * ht:
            return np.array([iso_tri_eigs(k, ht) for k in range(1, 11)])[:n]
    return None


def _needle(w, L=0.5):
    return np.array([0, 1, 1 + 1j * (0.5 - w / 2), 1 + L + 0.5j, 1 + 1j * (0.5 + w / 2),
                     1 + 1j, 1j])


def _finger(w, L=0.5):
    return np.array([0, 1, 1 + 1j * (0.5 - w / 2), 1 + L + 1j * (0.5 - w / 2),
                     1 + L + 1j * (0.5 + w / 2), 1 + 1j * (0.5 + w / 2), 1 + 1j, 1j])


def _ngon(N):
    return np.exp(2j * np.pi * np.arange(int(N)) / int(N))


def _hex(delta, how):
    z = _ngon(6)
    z[0] = z[0] + (delta if how == "radial" else 1j * delta)
    return z


def _rightish(delta_deg):
    d = math.radians(delta_deg)
    return np.array([0, 1, 1 + 0.7 * np.exp(1j * (np.pi / 2 - d)), 0.7j])


def _chamfer(c):
    return np.array([0, 1, 1 + 1j * (1 - c), 1 - c + 1j, 1j])


def _dumbbell(w):
    lo, hi = 0.5 - w / 2, 0.5 + w / 2
    return np.array([0, 1, 1 + 1j * lo, 1.5 + 1j * lo, 1.5, 2.5, 2.5 + 1j, 1.5 + 1j,
                     1.5 + 1j * hi, 1 + 1j * hi, 1 + 1j, 1j])


def _notch(omega_deg, a_max=0.25, d_max=0.5):
    half = (2 * np.pi - math.radians(omega_deg)) / 2
    d = a_max / math.tan(half)
    a = a_max
    if d > d_max:
        d, a = d_max, d_max * math.tan(half)
    return np.array([0, 1, 1 + 1j, 0.5 + a + 1j, 0.5 + 1j * (1 - d), 0.5 - a + 1j, 1j])


@dataclass(frozen=True)
class Family:
    name: str
    params: tuple
    verts: object                       # param -> complex vertices, counter-clockwise
    truth: object = None                # (param, n) -> values, or None
    upper: object = None                # (param, n) -> upper bounds (domain monotonicity)
    lower: object = None
    fd_scale: object = "rel"            # "rel": step x |param|; a float: absolute; None: no FD
    n_eigs: int = N_EIGS
    precs: tuple = PRECS
    pilot: tuple = ()                   # indices into params for the pilot


def _two_squares(n):
    return np.sort(np.concatenate([SQUARE, SQUARE]))[:n]


def _log(a, b, k):
    return tuple(float(x) for x in np.geomspace(a, b, k))


_ISO_THETAS = tuple(sorted(set(_log(60.0, 2.0, 30)) | {
    float(2 * math.degrees(math.atan(1 / h))) for h in ISO_TABLE_H}, reverse=True))
_NEEDLE_W = tuple(sorted(set(_log(0.5, 0.005, 25)) | set(_log(0.005, 5e-4, 7)), reverse=True))
_S = tuple(range(1, 13))

FAMILIES = {f.name: f for f in [
    Family("iso", _ISO_THETAS, _iso, truth=_iso_truth, pilot=(0, 18, 36)),
    Family("flat", _log(0.5, 0.005, 20), lambda t: np.array([0, 1, 0.5 + 1j * t]),
           pilot=(0, 10, 19)),
    Family("split", tuple(1.0 + 10.0 ** -s for s in _S), lambda b: _rect(1.0, b),
           truth=lambda b, n: _rect_truth(1.0, b, n), pilot=(1, 5, 9)),
    Family("aspect", _log(1.0, 20.0, 20), lambda a: _rect(a, 1.0),
           truth=lambda a, n: _rect_truth(a, 1.0, n), pilot=(0, 10, 19)),
    Family("needle", _NEEDLE_W, _needle, upper=lambda w, n: SQUARE[:n], pilot=(0, 20, 30)),
    Family("finger", _NEEDLE_W, _finger, upper=lambda w, n: SQUARE[:n], pilot=(0, 20, 30)),
    Family("ngon", (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64),
           _ngon, truth=lambda N, n: (_rect_truth(2 ** .5, 2 ** .5, n) if N == 4 else None),
           fd_scale=None, pilot=(0, 9, 19)),
    Family("hexrad", tuple(10.0 ** -s for s in _S), lambda d: _hex(d, "radial"),
           pilot=(1, 5, 9)),
    Family("hextan", tuple(10.0 ** -s for s in _S), lambda d: _hex(d, "tangential"),
           pilot=(1, 5, 9)),
    Family("rightish", tuple(sorted({0.0} | {s * 10.0 ** -k for k in range(1, 15)
                                             for s in (-1.0, 1.0)})),
           _rightish, truth=lambda d, n: (_rect_truth(1.0, 0.7, n) if d == 0 else None),
           fd_scale=None, pilot=(0, 14, 28)),
    Family("chamfer", _log(0.3, 1e-4, 20), _chamfer, lower=lambda c, n: SQUARE[:n],
           pilot=(0, 10, 19)),
    Family("dumbbell", _log(0.5, 5e-3, 15), _dumbbell, upper=lambda w, n: _two_squares(n),
           pilot=(0, 7, 14)),
    Family("notch", tuple(float(x) for x in np.linspace(190.0, 355.0, 15)), _notch,
           lower=lambda w, n: SQUARE[:n], fd_scale=1e-2, pilot=(0, 7, 14)),
    Family("shifted", tuple((s, t) for s in ("iso20", "needle05") for t in range(0, 9)),
           lambda st: ((_iso(20.0) if st[0] == "iso20" else _needle(0.05)) + 10.0 ** st[1]),
           fd_scale=None, precs=(1e-10,), pilot=(0, 4, 8)),
    Family("highk", ("square", "iso4", "ngon12", "needle05"),
           lambda s: {"square": _rect(1.0, 1.0), "iso4": np.array([1, 4j, -1]),
                      "ngon12": _ngon(12), "needle05": _needle(0.05)}[s],
           truth=lambda s, n: (_rect_truth(1.0, 1.0, n) if s == "square" else None),
           fd_scale=None, n_eigs=60, precs=(1e-10,), pilot=(0, 1)),
]}
FAMILY_ORDER = ("iso", "flat", "split", "aspect", "needle", "finger", "ngon", "hexrad", "hextan",
                "rightish", "chamfer", "dumbbell", "notch", "shifted", "highk")


def tasks(pilot=False):
    out = []
    for name in FAMILY_ORDER:
        f = FAMILIES[name]
        idx = f.pilot if pilot else range(len(f.params))
        precs = (1e-10, 1e-12) if pilot and len(f.precs) > 1 else f.precs
        out += [(name, int(i), float(p)) for i in idx for p in precs]
    return out


# --- one task -------------------------------------------------------------------------------

def _domain(z, prec):
    from chladni.geometry import Polygon
    return Polygon(np.asarray(z, dtype=complex), bc="dir", prec=prec)


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x


def _levels_brief(rec):
    return [dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound), met=bool(L.met),
                 cluster=L.cluster) for L in rec.spectrum.levels]


def _unfolded(rec):
    return np.array([float(L.lam) for L in rec.spectrum.levels for _ in range(int(L.mult))])


def _evals(rec):
    """Pencil factorizations the record's solver paid for: the deterministic cost."""
    from chladni.eigenproblem import _n_evals
    return _n_evals(getattr(rec, "solver", None))


def _solve_base(dom, n, prec):
    """`(record, info)`: the black-box solve, or, when the planner declines on the geometry
    alone (`plan.shape_refusal`: the clearance gate, or float64 precision short of `prec`),
    every rung solved on its own plan and the best kept. Both refusals are what this study
    tests, so a decline would answer nothing."""
    from chladni.eigenproblem import Eigenproblem
    from chladni.plan import LADDER, plan_for
    ep = Eigenproblem(dom)
    _, rec = ep.plan_and_solve(n, prec)
    info = dict(gate_bypassed=False)
    reason = rec.spectrum.reason if rec.spectrum.declined else ""
    if not ("near self-intersection" in reason or "geometry supports" in reason):
        return rec, info
    info.update(gate_bypassed=True, gate_reason=rec.spectrum.reason, rungs=[])
    best, best_key = None, None
    for rung in range(len(LADDER)):
        t0 = time.perf_counter()
        try:
            r = ep.solve(n, prec, plan=plan_for(dom, n, prec, rung))
        except Exception as err:                          # noqa: BLE001
            info["rungs"].append(dict(rung=rung, error=f"{type(err).__name__}: {err}"))
            continue
        levels = r.spectrum.levels
        met = sum(bool(L.met) for L in levels)
        info["rungs"].append(dict(rung=rung, seconds=time.perf_counter() - t0,
                                  evaluations=_evals(r),
                                  declined=bool(r.spectrum.declined), n_levels=len(levels),
                                  n_met=met, flags=list(r.spectrum.flags)))
        key = (not r.spectrum.declined, met)
        if best is None or key > best_key:
            best, best_key = r, key
        if not r.spectrum.declined and levels and met == len(levels):
            break
    return (best if best is not None else rec), info


def _units(rec):
    """Level index lists differentiated as one: a level, or a whole chladni cluster."""
    spec, seen, out = rec.spectrum, set(), []
    for i, L in enumerate(spec.levels):
        if i in seen:
            continue
        u = list(spec.clusters[L.cluster].levels) if L.cluster is not None else [i]
        seen.update(u)
        out.append(u)
    return out


def _vertex_fields(dom, z):
    """`V(curve, t)`: each vertex's hat field in x then in y (`hadamard_hard_shapes.fields`
    without its four identity fields)."""
    segs = [s for b in dom.boundaries for s in b.segments]

    def V(curve, t):
        s = segs[curve]
        t = np.asarray(t, float)
        p, nrm = s.p(t), s.N(t)
        a, b = s.p(np.array([0.0]))[0], s.p(np.array([1.0]))[0]
        out = []
        for v in z:
            for w in (1.0, 1j):
                d = np.zeros_like(p)
                if abs(a - v) < 1e-12 * max(1.0, abs(v)):
                    d = d + w * (1 - t)
                if abs(b - v) < 1e-12 * max(1.0, abs(v)):
                    d = d + w * t
                out.append(d)
        return np.array([(f * nrm.conjugate()).real for f in out])
    return V


def _hadamard_dp(rec, dom, z, dz):
    """Per unit, d(sum of the unit's values)/dp from Hadamard, and the unit's `attained`."""
    from chladni._core import PrecisionWarning
    V = _vertex_fields(dom, z)
    w = np.ravel(np.column_stack([dz.real, dz.imag]))        # x0, y0, x1, y1, ...
    out = []
    for u in _units(rec):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PrecisionWarning)
                H, rep = rec.hadamard(u if len(u) > 1 else u[0], V, report=True)
            tr = np.trace(H, axis1=1, axis2=2)
            lam_sum = float(sum(float(rec.spectrum.levels[j].lam) * int(rec.spectrum.levels[j].mult)
                                for j in u))
            # `attained` is relative to lam per unit field, so on this combination of hat fields
            # it bounds d_sum to about attained * (sum of the unit's values) * ||w||_1
            out.append(dict(levels=u, d_sum=float(tr @ w), attained=float(rep.attained),
                            attained_abs=float(rep.attained) * lam_sum * float(np.abs(w).sum()),
                            identity_residual=float(rep.identity_residual),
                            eigenfunction_bound=float(rep.eigenfunction_bound)))
        except Exception as err:                          # noqa: BLE001
            out.append(dict(levels=u, error=f"{type(err).__name__}: {err}"))
    return out


def _fd_dp(fam, p, rec, n, prec):
    """Central differences of each unit's sum on the frozen base plan, Richardson-combined from
    +-h and +-2h, per step in `FD_STEPS`; the error bar is noise (bounds / h) plus truncation
    (the change to the next step). Values are matched to the base by unfolded position."""
    from chladni.eigenproblem import Eigenproblem
    scale = abs(p) if fam.fd_scale == "rel" else float(fam.fd_scale)
    units = _units(rec)
    base_pos = np.cumsum([0] + [int(L.mult) for L in rec.spectrum.levels])
    n_vals = int(base_pos[-1])

    def sums(q):
        r = Eigenproblem(_domain(fam.verts(q), prec)).solve(n, prec, plan=rec.plan)
        vals = _unfolded(r)
        bnds = np.array([float(L.bound) for L in r.spectrum.levels for _ in range(int(L.mult))])
        if len(vals) < n_vals:
            raise RuntimeError(f"{len(vals)} values at p = {q!r}, base had {n_vals}")
        s = [float(vals[base_pos[u[0]]:base_pos[u[-1] + 1]].sum()) for u in units]
        b = [float(bnds[base_pos[u[0]]:base_pos[u[-1] + 1]].sum()) for u in units]
        return np.array(s), np.array(b), [L.met for L in r.spectrum.levels]

    steps = []
    for hr in FD_STEPS:
        h = hr * scale
        try:
            vals = {k: sums(p + k * h) for k in (-2, -1, 1, 2)}
        except Exception as err:                          # noqa: BLE001
            steps.append(dict(h=h, error=f"{type(err).__name__}: {err}"))
            continue
        d1 = (vals[1][0] - vals[-1][0]) / (2 * h)
        d2 = (vals[2][0] - vals[-2][0]) / (4 * h)
        rich = (4 * d1 - d2) / 3
        noise = (vals[1][1] + vals[-1][1]) / (2 * h)
        steps.append(dict(h=h, d=rich.tolist(), noise=noise.tolist(),
                          all_met=bool(all(all(v[2]) for v in vals.values()))))
    good = [s for s in steps if "d" in s]
    for a, b in zip(good, good[1:]):
        a["trunc"] = np.abs(np.array(a["d"]) - np.array(b["d"])).tolist()
    if good:
        good[-1]["trunc"] = good[-2]["trunc"] if len(good) > 1 else [math.nan] * len(units)
    best = []
    for j in range(len(units)):
        cand = [(s["noise"][j] + s["trunc"][j], s["d"][j], s["h"]) for s in good
                if math.isfinite(s["noise"][j] + s["trunc"][j])]
        best.append(min(cand) if cand else (math.nan, math.nan, math.nan))
    return dict(units=units, steps=steps,
                best=[dict(d=d, bar=bar, h=h) for bar, d, h in best])


def _knobs(dom, rec, n, prec):
    """One change at a time on the base plan; each row: values, bounds, met, and the largest
    relative change against the base."""
    from chladni.eigenproblem import Eigenproblem
    plan, base = rec.plan, _unfolded(rec)
    variants = {
        "basis_x0.6": replace(plan, corner_orders=tuple(int(round(0.6 * o))
                                                        for o in plan.corner_orders),
                              n_sources=int(round(0.6 * plan.n_sources))),
        "basis_x1.5": replace(plan, corner_orders=tuple(int(round(1.5 * o))
                                                        for o in plan.corner_orders),
                              n_sources=int(round(1.5 * plan.n_sources))),
        "bdry_x2": replace(plan, bdry_ratio=2 * plan.bdry_ratio),
        "int_x2": replace(plan, int_ratio=2 * plan.int_ratio),
    }
    out = {}
    z = np.array([c.z for c in dom.corners])
    rng = np.random.default_rng(7)
    jobs = [(k, dom, v) for k, v in variants.items()]
    for j in range(3):
        zj = z * (1 + KNOB_JITTER * rng.normal(size=len(z)))
        jobs.append((f"jitter_{j}", _domain(zj, prec), plan))
    for name, d, pl in jobs:
        t0 = time.perf_counter()
        try:
            r = Eigenproblem(d).solve(n, prec, plan=pl)
            vals = _unfolded(r)
            k = min(len(vals), len(base))
            out[name] = dict(levels=_levels_brief(r), n_cols=int(pl.n_cols),
                             seconds=time.perf_counter() - t0, evaluations=_evals(r),
                             max_rel_change=float(np.max(np.abs(vals[:k] - base[:k]) / base[:k]))
                             if k else None,
                             rel_change=(np.abs(vals[:k] - base[:k]) / base[:k]).tolist())
        except Exception as err:                          # noqa: BLE001
            out[name] = dict(error=f"{type(err).__name__}: {err}")
    return out


def _richardson(f, p, eps):
    """d f / dp of an exact function (vertices, a closed form): central differences at eps and
    2 eps, Richardson-combined -- truncation O(eps^4), rounding O(1e-16 / eps)."""
    d1 = (f(p + eps) - f(p - eps)) / (2 * eps)
    d2 = (f(p + 2 * eps) - f(p - 2 * eps)) / (4 * eps)
    return (4 * d1 - d2) / 3


def run(family, index, prec, knobs=False):
    fam = FAMILIES[family]
    p = fam.params[int(index)]
    prec = float(prec)
    n = fam.n_eigs
    z = np.asarray(fam.verts(p), dtype=complex)
    row = dict(family=family, index=int(index), param=p, prec=prec, n=n,
               vertices=[[float(c.real), float(c.imag)] for c in z])
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            dom = _domain(z, prec)
            row["achievable_prec"] = float(dom.achievable_prec)
            rec, info = _solve_base(dom, n, prec)
            row.update(info)
            row["report"] = rec.report(2).as_dict()
            vals = _unfolded(rec)
            row["values"] = vals.tolist()
            for key, fn in (("truth", fam.truth), ("truth_upper", fam.upper),
                            ("truth_lower", fam.lower)):
                t = None if fn is None else fn(p, n)
                if t is not None:
                    k = min(len(t), len(vals))
                    row[key] = np.asarray(t, float)[:k].tolist()
                    if key == "truth":
                        row["err_rel"] = (np.abs(vals[:k] - t[:k]) / t[:k]).tolist()
                    elif key == "truth_upper":
                        row["violation_rel"] = np.maximum(vals[:k] - t[:k], 0).tolist()
                    else:
                        row["violation_rel"] = np.maximum(t[:k] - vals[:k], 0).tolist()
            if fam.fd_scale is not None and not rec.spectrum.declined and rec.plan is not None:
                eps = 1e-4 * (abs(p) if fam.fd_scale == "rel" else float(fam.fd_scale))
                dz = _richardson(lambda q: np.asarray(fam.verts(q), dtype=complex), p, eps)
                row["hadamard_dp"] = _hadamard_dp(rec, dom, z, dz)
                row["fd_dp"] = _fd_dp(fam, p, rec, n, prec)
                if fam.truth is not None and fam.truth(p, n) is not None:
                    tdp = _richardson(lambda q: np.asarray(fam.truth(q, n), float), p, eps)
                    pos = np.cumsum([0] + [int(L.mult) for L in rec.spectrum.levels])
                    row["truth_dp"] = [float(tdp[pos[u[0]]:pos[u[-1] + 1]].sum())
                                       if pos[u[-1] + 1] <= len(tdp) else None
                                       for u in _units(rec)]
            if knobs and not rec.spectrum.declined and rec.plan is not None:
                row["knobs"] = _knobs(dom, rec, n, prec)
            row["status"] = "ok"
        except Exception as err:                          # noqa: BLE001
            row.update(status="error", error=f"{type(err).__name__}: {err}",
                       traceback=traceback.format_exc().strip().splitlines()[-4:])
    row["seconds"] = time.perf_counter() - t0
    counts = {}
    for w in caught:
        key = f"{w.category.__name__}: {str(w.message)[:120]}"
        counts[key] = counts.get(key, 0) + 1
    row["warnings"] = counts
    return _plain(row)


def task(i, out, pilot=False):
    family, index, prec = tasks(pilot)[int(i)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{family}_{index:03d}_{prec:.0e}.json"
    if path.exists():
        return
    row = run(family, index, prec, knobs=pilot)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.rename(path)


# --- analysis -------------------------------------------------------------------------------

def _worst(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return max(xs) if xs else math.nan


def analyze(d):
    """One line per task: family, parameter, prec, status, rung, levels met, the worst error
    against truth (or bound violation), the worst Hadamard attained, the worst tension floor,
    the worst FD-vs-Hadamard disagreement against the FD bar (`fd/bar`), the part of it beyond
    that bar against Hadamard's own estimate (`H/att`: above 1 means `attained` is silent), and
    the knob that moved values most."""
    rows = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]
    print(f"{'family':9s} {'param':>11s} {'prec':>6s} {'st':>3s} {'rung':>4s} {'met':>6s} "
          f"{'err':>8s} {'attnd':>8s} {'floor':>8s} {'fd/bar':>8s} {'H/att':>8s} {'s':>6s}  knob")
    for r in rows:
        rep = r.get("report", {})
        L = rep.get("levels", [])
        met = f"{sum(bool(x.get('met')) for x in L)}/{len(L)}"
        rung = rep.get("run", {}).get("plan", {}).get("rung", "-")
        err = _worst(r.get("err_rel", []) or r.get("violation_rel", []))
        att = _worst([x.get("hadamard_attained") for x in L])
        flo = _worst([x.get("tension_floor_rel") for x in L])
        ratio = hratio = math.nan
        if "fd_dp" in r and "hadamard_dp" in r:
            qs, hs = [], []
            for hd, fb in zip(r["hadamard_dp"], r["fd_dp"]["best"]):
                if "d_sum" in hd and isinstance(fb.get("bar"), float) and fb["bar"] > 0:
                    gap = abs(hd["d_sum"] - fb["d"])
                    qs.append(gap / fb["bar"])
                    if isinstance(hd.get("attained_abs"), float) and hd["attained_abs"] > 0:
                        # the disagreement beyond FD's own bar, against Hadamard's estimate
                        hs.append(max(gap - fb["bar"], 0.0) / hd["attained_abs"])
            ratio, hratio = _worst(qs), _worst(hs)
        knob = ""
        if "knobs" in r:
            moved = {k: v.get("max_rel_change") for k, v in r["knobs"].items()
                     if isinstance(v.get("max_rel_change"), float)}
            if moved:
                k = max(moved, key=moved.get)
                knob = f"{k} {moved[k]:.1e}"
        p = r["param"]
        ps = f"{p:.4g}" if isinstance(p, float) else str(p)
        tag = "G" if r.get("gate_bypassed") else ""
        print(f"{r['family']:9s} {ps:>11s} {r['prec']:6.0e} {r['status'][:3]:>3s} "
              f"{str(rung) + tag:>4s} {met:>6s} {err:8.1e} {att:8.1e} {flo:8.1e} {ratio:8.1e} {hratio:8.1e} "
              f"{r['seconds']:6.0f}  {knob}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ntasks":
        print(len(tasks(pilot=(sys.argv[2:] == ["pilot"]))))
    elif cmd == "run":
        a = sys.argv[2:]
        print(json.dumps(run(a[0], int(a[1]), float(a[2]), knobs="--knobs" in a)))
    elif cmd == "task":
        a = sys.argv[3:]
        task(int(sys.argv[2]), a[a.index("--out") + 1], pilot="--pilot" in a)
    elif cmd == "analyze":
        analyze(sys.argv[2])
    else:
        raise SystemExit(__doc__)
