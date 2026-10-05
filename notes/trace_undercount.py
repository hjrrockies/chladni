"""Trace one cell of the sporadic cluster UNDER-COUNT, stage by stage.

`notes/smooth_filter.md` section 10.7 is the record this reproduces. The failure: `solve` declares
a cluster one member short -- 2 of 3, 3 of 4 -- at 4-12x nominal resolution margin, sometimes
silently. Seen under both penalty orders and at five settings; its mechanism was guessed twice and
wrong both times, so this script does not guess. It wraps the names `chladni.mps.search.solve`
calls -- `bracket_interval`, `refine`, `multiplicity`, `_slope_from_cache`, `_claim` -- and the
objective's evaluation, and prints what each stage did against the true members.

Nothing in `src/` is modified; the wrapping is monkeypatching, undone on exit. Run from the root:

    .venv/bin/python notes/trace_undercount.py                   # the 1e-10 target cell
    .venv/bin/python notes/trace_undercount.py 1e-12 0.1 65 1.0  # prec, C (tau = C*prec), N, band
    .venv/bin/python notes/trace_undercount.py 1e-6 auto 50 0.3  # C = auto: the default pencil
"""
from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager

sys.path.insert(0, "tests")
sys.path.insert(0, "notes")
warnings.simplefilter("ignore")

import numpy as np

import bracket_score as bs
import rect_suite as rs
from basis_sufficiency import sufficient
from chladni.mps import search as S
from chladni.mps.objective import Objective
from smooth_filter_tau import pencil_for

LEVELS = {50: 3, 65: 4}


@contextmanager
def patched(log):
    """Wrap the stage entry points `solve` resolves from its own module namespace."""
    saved = {k: getattr(S, k) for k in
             ("bracket_interval", "refine", "multiplicity", "_slope_from_cache", "_claim")}

    def bracket_interval(obj, **kw):
        res = saved["bracket_interval"](obj, **kw)
        log.append(("bracket", res))
        return res

    def refine(obj, lo, hi, **kw):
        r = saved["refine"](obj, lo, hi, **kw)
        log.append(("refine", (lo, hi, kw.get("lam_hint")), r))
        return r

    def multiplicity(tens, lam, **kw):
        m = saved["multiplicity"](tens, lam, **kw)
        log.append(("mult", lam, kw.get("bracket"), kw.get("c"), m))
        return m

    def slope(lams, vs, lam, floor=0.0):
        c = saved["_slope_from_cache"](lams, vs, lam, floor=floor)
        log.append(("slope", lam, floor, c))
        return c

    def claim(branches, lam_root, claimants, bias_lo, bias_hi, **kw):
        out = saved["_claim"](branches, lam_root, claimants, bias_lo, bias_hi, **kw)
        log.append(("claim", lam_root, list(claimants), out))
        return out

    for k, f in (("bracket_interval", bracket_interval), ("refine", refine),
                 ("multiplicity", multiplicity), ("_slope_from_cache", slope), ("_claim", claim)):
        setattr(S, k, f)
    try:
        yield
    finally:
        for k, f in saved.items():
            setattr(S, k, f)


def cell(prec, C, N, band):
    (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: LEVELS[N]})
    if C == "auto":                  # the pencil that ships: no regularization knob at all
        from chladni.mps import MPSPencil
        q = MPSPencil(case.pencil.bdry, case.pencil.interior, prec=prec)
    else:
        q = pencil_for(case.pencil, prec, None if C is None else C * prec)
    base = bs.objective_for(case, prec)
    evals = []

    def f(x):
        t = q.tensions(float(x))
        evals.append((float(x), float(np.atleast_1d(t)[0])))
        return t

    obj = Objective(f, base.lo, base.hi, prec=prec, variable=case.variable,
                    expected_count=getattr(base, "expected_count", None))
    return case, obj, evals


def near(truth, lam):
    """`(index, offset in licences)` of the true member nearest `lam`."""
    i = int(np.argmin(np.abs(truth - lam)))
    return i, (lam - truth[i]) / TRUTH_LICENCE


def main(prec=1e-10, C=0.02, N=50, band=0.3):
    global TRUTH_LICENCE
    case, obj, evals = cell(prec, C, N, band)
    truth = np.array([t for t in np.asarray(case.spectrum, dtype=float)
                      if abs(t - N * np.pi ** 2) < 0.01 * N * np.pi ** 2])
    TRUTH_LICENCE = prec * float(truth.max())
    ok = sufficient(case, prec).multiplicity
    print(f"cell N={N} band={band} prec={prec:g} C={C} sufficient={ok}")
    print("true members:", ", ".join(f"{t:.15f}" for t in truth))
    print("gaps (licences):", np.round(np.diff(truth) / TRUTH_LICENCE, 3))

    log = []
    with patched(log):
        res = S.solve(obj, prec=prec)

    for e in log:
        kind = e[0]
        if kind == "bracket":
            print(f"\n[stage 1] declined={e[1].declined}  {len(e[1].brackets)} brackets")
            for b in e[1].brackets:
                lo, hi = obj.in_lam_bracket(b.lo, b.hi)
                print(f"    [{lo:.12f}, {hi:.12f}]  holds members "
                      f"{[i for i, t in enumerate(truth) if lo <= t <= hi]}")
        elif kind == "refine":
            (lo, hi, hint), r = e[1], e[2]
            i, off = near(truth, r.lam)
            print(f"[refine] [{lo:.12f}, {hi:.12f}] -> {r.reason:>13} accepted={r.accepted} "
                  f"lam={r.lam:.15f} sigma={r.sigma:.2e} delta_res={r.delta_res:.1e}  "
                  f"nearest member {i} at {off:+.3f} lic")
        elif kind == "slope":
            print(f"[slope] at {e[1]:.12f} floor={e[2]:.2e} c={e[3]}")
        elif kind == "mult":
            _, lam, br, c, m = e
            print(f"[mult] root {lam:.15f} c={c} mu={m.mu} ambiguous={m.ambiguous}")
            for b in m.branches:
                print(f"        branch {b.index}: sigma={b.sigma:.2e} d_est={b.d_est:.2e} "
                      f"({b.d_est / TRUTH_LICENCE:.3f} lic) side={b.side} verdict={b.verdict}")
        elif kind == "claim":
            print(f"[claim] root {e[1]:.12f} claimants={['%.12f' % x for x in e[2]]} -> {e[3]}")

    d = res.diagnostics
    tot = sum(x.mult for x in res.levels)
    print(f"\nRESULT: {tot} of {len(truth)}  declined={res.declined}  "
          f"n_uncertain={getattr(d, 'n_uncertain', None)}  evals={obj.n_evals}")
    for L in res.levels:
        i, off = near(truth, L.lam)
        print(f"    level {L.lam:.15f} mult={L.mult} sigma={L.sigma:.2e} "
              f"uncertain={L.count_uncertain}  nearest member {i} at {off:+.3f} lic")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(float(a[0]), a[1] if a[1] == "auto" else float(a[1]), int(a[2]), float(a[3]))
         if a else ())
