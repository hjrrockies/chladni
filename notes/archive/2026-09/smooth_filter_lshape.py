"""`bracket_interval` on the L-SHAPE, against filtered objectives.

The corner-domain gap. Every search-level measurement so far is a rectangle with a plane-wave
basis, and `CLAUDE.md` puts corner singularities at the centre of what `chladni` is for.

Two facts established before running anything, and they shape what this can show:

* The L-shape's truth table carries an EXACT degeneracy at 128.3049 (the 24th and 25th
  tabulated levels are equal to all 14 digits), so the window holds a genuine tight pair.
* With a pure corner-FB basis the threshold rank is n_cols at n_fb = 80 and 160 whatever the
  window -- truncation is INERT -- and only becomes active at n_fb = 240, where it runs
  232..234 of 240 over [5, 130]. A spread of 2, against 29..63 on the rectangle's 60x window.
  So the rank barely moves with lam here: the dilemma this whole note is about is a
  plane-wave phenomenon, and a corner-adapted basis mostly does not have it.

n_fb = 160 is therefore run as a CONTROL (truncation inert -> every policy must agree) and
n_fb = 240 as the live case.
"""
from __future__ import annotations

import sys, time, warnings
sys.path.insert(0, "tests"); sys.path.insert(0, "notes")
warnings.simplefilter("ignore")

import numpy as np
import hard_roster as hr
import bracket_score as bsc
from chladni.reference import L_shape_eigs
from chladni.mps.bracket import bracket_interval
from chladni.mps.objective import Objective
from chladni.mps import MPSPencil
from smooth_filter import tension_hybrid, tension_tikhonov

EPS = np.finfo(float).eps
LAM = np.asarray(L_shape_eigs(25), dtype=float)


def objectives(p, a, b, prec):
    frozen = MPSPencil.from_basis(p.basis, p.bdry, p.interior, reg=p.reg, rtol=p.rtol,
                                  lam_ref=(a, b))
    c = float(np.sqrt(prec * EPS))
    return [("threshold (current)", lambda x: p.tensions(float(x))),
            ("frozen rank", lambda x: frozen.tensions(float(x))),
            ("hybrid c=sqrt(prec*eps)",
             lambda x: tension_hybrid(*p.pencil(float(x)), c, 2.0)),
            ("tikhonov mu=sqrt(prec*eps)",
             lambda x: tension_tikhonov(*p.pencil(float(x)), c))], frozen


def verdict(res, truth, prec, variable):
    if res.declined:
        return "declined", res.diagnostics.decline_reason, 0
    brs = bsc.to_lam_brackets([(b.lo, b.hi) for b in res.brackets], variable=variable)
    t = np.sort(np.asarray(truth, dtype=float))
    hit = np.zeros(t.size, dtype=int); impure = 0
    for lo, hi in brs:
        ix = np.nonzero((t >= lo) & (t <= hi))[0]
        hit[ix] += 1
        if ix.size > 1 and t[ix[-1]] - t[ix[0]] > prec * max(abs(t[ix[-1]]), 1.0):
            impure += 1
    spur = sum(1 for lo, hi in brs if not np.any((t >= lo) & (t <= hi)))
    ok = bool((hit > 0).all()) and impure == 0
    why = (f"spurious {spur}" if ok
           else f"covered {int((hit>0).sum())}/{t.size}, impure {impure}, spurious {spur}")
    return ("right" if ok else "wrong"), why, len(brs)


def run(precs, n_fbs, a=5.0, b=None):
    b = float(LAM[-1]) if b is None else b
    truth = LAM[(LAM >= a) & (LAM <= b)]
    for n_fb in n_fbs:
        case = hr.make_case(f"L_shape/n{n_fb}", hr.L_shape(), a, b, truth,
                            complete=True, n_fb=n_fb)
        p = case.pencil
        rk = [p.rank_at(float(x)) for x in np.linspace(a, b, 11)]
        tag = "INERT" if min(rk) >= p.n_cols else "active"
        print(f"\n{'='*94}\n=== L-shape [{a:g}, {b:.3f}]  {truth.size} levels  "
              f"n_fb={n_fb}  {p.n_cols} cols  rank {min(rk)}..{max(rk)}  "
              f"truncation {tag}", flush=True)
        for prec in precs:
            objs, frozen = objectives(p, a, b, prec)
            print(f"  prec={prec:.0e}  (frozen rank {frozen.rank})", flush=True)
            for nm, fn in objs:
                base = bsc.objective_for(case, prec)
                obj = Objective(fn, base.lo, base.hi, prec=prec, variable=case.variable,
                                expected_count=getattr(base, "expected_count", None))
                t0 = time.time()
                try:
                    res = bracket_interval(obj, prec=prec)
                    v, why, nb = verdict(res, truth, prec, case.variable)
                except Exception as ex:
                    v, why, nb = "error", f"{type(ex).__name__}: {ex}", 0
                print(f"    {nm:>26}: {v:>8}  br={nb:<4d} {obj.n_evals:>6d} evals  "
                      f"{time.time()-t0:7.1f}s  {why[:52]}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scope"
    if mode == "scope":
        run((1e-10,), (160, 240))
    else:
        run((1e-6, 1e-8, 1e-10, 1e-12), (160, 240))
