"""Does `tau = C eps sqrt(n)` serve the posterior penalty across basis sizes and domains?

`notes/posterior_penalty.md` found a FIXED `tau = 2e-14`, read through posterior tensions, clean on
the cluster grid at every rung -- on one basis size (126 plane-wave columns) of one family. The
noise jaw it sits above is `~3 eps sqrt(n)`, so the rule to test is `tau = C eps sqrt(n)`, and the
question has two halves, each against the threshold on the same pencil:

    size     the 50/65*pi^2 cluster grid rebuilt at three basis sizes (`pad`), at 1e-12 and 1e-14
    roster   the hard roster -- disk, equilateral triangle, and the L-shape on its pure corner
             Fourier-Bessel basis -- scored by total count and worst relative error against truth

`C = 8` is the measured `tau = 2e-14` at n = 126. The sweep brackets it, 3 (at the jaw) to 100.

EXPERIMENTAL; nothing in `src/` uses it. Run from the repo root:

    .venv/bin/python notes/posterior_tau_sweep.py size
    .venv/bin/python notes/posterior_tau_sweep.py roster
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")
sys.path.insert(0, "notes")
warnings.simplefilter("ignore")

import numpy as np

import bracket_score as bs
import rect_suite as rs
from basis_sufficiency import sufficient
from chladni.mps import MPSPencil
from chladni.mps.objective import Objective
from chladni.mps.search import solve
from posterior_penalty import BANDS, CLUSTERS, posterior_tensions

EPS = np.finfo(float).eps
CS = (3.0, 8.0, 30.0, 100.0)
PADS = (1.2, 1.6, 2.4)


def pencil(p, prec, C):
    """`C=None` is the threshold baseline at its derived rtol."""
    if C is None:
        return MPSPencil(p.bdry, p.interior, reg="svd", prec=prec)
    tau = C * EPS * np.sqrt(p.n_cols)
    return MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec,
                     rtol=tau / np.sqrt(p.n_cols))


def _solve(case, q, prec, posterior):
    f = ((lambda x: posterior_tensions(q, x)) if posterior
         else (lambda x: q.tensions(float(x))))
    base = bs.objective_for(case, prec)
    obj = Objective(f, base.lo, base.hi, prec=prec, variable=case.variable,
                    expected_count=getattr(base, "expected_count", None))
    return solve(obj, prec=prec), obj.n_evals


def _cluster_case(N, m, band, prec, pad):
    """`rect_suite.split_degeneracy_cases`, with the basis padding exposed."""
    lam = N * rs.PI2
    a, b = lam * 0.97, lam * 1.03
    tp = rs.tightest_pair(a, b)
    L = 1.0 + band * prec / tp[0]
    return rs.make_case(f"X:split{m}/b={band:g}/p={prec:.0e}/pad={pad}", L, 1.0, a, b,
                        modes=rs.modes_in_window(a, b), pad=pad)


def size():
    for pad in PADS:
        for prec in (1e-12, 1e-14):
            cases = []
            for N, m in CLUSTERS:
                for band in BANDS:
                    c = _cluster_case(N, m, band, prec, pad)
                    if sufficient(c, prec).multiplicity:
                        cases.append((N, m, band, c))
            n = cases[0][3].pencil.n_cols if cases else 0
            print(f"\npad {pad}  n = {n}  prec {prec:.0e}  ({len(cases)} exercisable cells)",
                  flush=True)
            for C in (None,) + CS:
                t0 = time.time()
                right = flagged = silent = declined = 0
                bad = []
                for N, m, band, c in cases:
                    res, _ = _solve(c, pencil(c.pencil, prec, C), prec, C is not None)
                    tot = sum(x.mult for x in res.levels)
                    if res.declined:
                        declined += 1
                    elif tot == m:
                        right += 1
                    elif res.diagnostics.n_uncertain:
                        flagged += 1
                        bad.append((N, band, f"{tot}/{m} flagged"))
                    else:
                        silent += 1
                        bad.append((N, band, f"{tot}/{m} silent"))
                label = "threshold" if C is None else f"C = {C:g} (tau {C * EPS * np.sqrt(n):.1e})"
                print(f"  {label:>26}  {right:2d} right {flagged:2d} flagged {silent:2d} silent "
                      f"{declined:2d} declined  ({time.time() - t0:.0f}s)  {bad}", flush=True)


def roster():
    import hard_roster as hr

    for name, build in hr.ROSTER.items():
        case = build()
        truth = np.asarray(case.truth, dtype=float)
        precs = (1e-10, 1e-12) if name == "L_shape" else (1e-10, 1e-12, 1e-14)
        for prec in precs:
            print(f"\n{name}  n = {case.pencil.n_cols}  prec {prec:.0e}  "
                  f"({len(truth)} true eigenvalues in [{case.a:g}, {case.b:g}])", flush=True)
            for C in (None,) + CS:
                t0 = time.time()
                res, evals = _solve(case, pencil(case.pencil, prec, C), prec, C is not None)
                lams = np.array([x.lam for x in res.levels])
                tot = sum(x.mult for x in res.levels)
                worst = (max(np.min(np.abs(truth - l)) / l for l in lams)
                         if len(lams) else float("nan"))
                label = "threshold" if C is None else f"C = {C:g}"
                print(f"  {label:>10}  total {tot:3d} of {len(truth):3d}  "
                      f"declined={res.declined}  uncertain={res.diagnostics.n_uncertain}  "
                      f"worst rel err {worst:.1e}  {evals} evals  ({time.time() - t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    {"size": size, "roster": roster}[sys.argv[1] if len(sys.argv) > 1 else "size"]()
