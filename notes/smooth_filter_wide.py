"""`bracket_interval` on WIDE windows, against filtered objectives.

Section 8 item 6: the decline grids are 6% wide, and family A is wide but was driven through
the older depth-based `bracketers.py`. Wide window + the slope loop is the one configuration
where the frozen rank's floor is fatal AND the threshold's noise is what breaks the slope
estimate -- so it is the one place a filter could win.

The axis is WINDOW WIDTH, because that is the mechanism claim: section 35's width table says
freezing costs one or two ranks below ~10% and is hopeless over 60x. If the filter is ever
the right answer, it is as the width grows.

Same pencil per width for every objective; only `Objective.tensions` varies.
"""
from __future__ import annotations

import sys, time, warnings
sys.path.insert(0, "tests"); sys.path.insert(0, "notes")
warnings.simplefilter("ignore")

import numpy as np
import rect_suite as rs
import bracket_score as bsc
from chladni.mps.bracket import bracket_interval
from chladni.mps.objective import Objective
from chladni.mps import MPSPencil
from smooth_filter import tension_hybrid, tension_tikhonov

EPS = np.finfo(float).eps
PHI = rs.PHI
CENTRE = 150.0
#: relative window widths: from inside section 35's "freezing is ~free" regime out past it
WIDTHS = (0.06, 0.20, 0.60, 1.60)
WIDE = (5.0, 300.0)                      # family A's own window, 60x


def window(frac):
    return CENTRE * (1 - frac / 2), CENTRE * (1 + frac / 2)


def case_for(a, b):
    return rs.make_case(f"W:[{a:.1f},{b:.1f}]", 1.0, 1.0 / PHI, a, b)


def truth_in(a, b):
    c = case_for(a, b)
    t = rs.complete_truth(c)
    return c, t[(t >= a) & (t <= b)]


def objectives(p, a, b, prec):
    frozen = MPSPencil.from_basis(p.basis, p.bdry, p.interior, reg=p.reg, rtol=p.rtol,
                                  lam_ref=(a, b))
    c_rule = float(np.sqrt(prec * EPS))
    return [
        ("threshold (current)", lambda x: p.tensions(float(x))),
        ("frozen rank",         lambda x: frozen.tensions(float(x))),
        ("hybrid c=sqrt(prec*eps)",
         lambda x: tension_hybrid(*p.pencil(float(x)), c_rule, 2.0)),
        ("tikhonov mu=sqrt(prec*eps)",
         lambda x: tension_tikhonov(*p.pencil(float(x)), c_rule)),
    ], frozen


def verdict(res, truth, prec, variable):
    if res.declined:
        return "declined", res.diagnostics.decline_reason, len(res.brackets)
    brs = bsc.to_lam_brackets([(b.lo, b.hi) for b in res.brackets], variable=variable)
    t = np.sort(np.asarray(truth, dtype=float))
    hit = np.zeros(t.size, dtype=int)
    impure = 0
    for lo, hi in brs:
        ix = np.nonzero((t >= lo) & (t <= hi))[0]
        hit[ix] += 1
        if ix.size > 1 and t[ix[-1]] - t[ix[0]] > prec * max(abs(t[ix[-1]]), 1.0):
            impure += 1
    spur = sum(1 for lo, hi in brs
               if not np.any((t >= lo) & (t <= hi)))
    ok = bool((hit > 0).all()) and impure == 0
    why = (f"spurious {spur}" if ok else
           f"covered {int((hit>0).sum())}/{t.size}, impure {impure}, spurious {spur}")
    return ("right" if ok else "wrong"), why, len(brs)


def run(precs, widths, include_wide=True):
    wins = [window(f) for f in widths] + ([WIDE] if include_wide else [])
    labels = [f"{f:.0%}" for f in widths] + (["60x"] if include_wide else [])
    for (a, b), lab in zip(wins, labels):
        case, truth = truth_in(a, b)
        p = case.pencil
        rk = [p.rank_at(float(l)) for l in np.linspace(a, b, 15)]
        print(f"\n{'='*92}\n=== window {lab}: [{a:.2f}, {b:.2f}]  {len(truth)} levels  "
              f"{p.n_cols} cols  threshold rank {min(rk)}..{max(rk)}", flush=True)
        for prec in precs:
            objs, frozen = objectives(p, a, b, prec)
            print(f"  prec={prec:.0e}  (frozen rank {frozen.rank})", flush=True)
            for nm, fn in objs:
                base = bsc.objective_for(case, prec)
                obj = Objective(fn, base.lo, base.hi, prec=prec,
                                variable=case.variable,
                                expected_count=getattr(base, "expected_count", None))
                t0 = time.time()
                try:
                    res = bracket_interval(obj, prec=prec)
                    v, why, nb = verdict(res, truth, prec, case.variable)
                except Exception as ex:
                    v, why, nb = "error", f"{type(ex).__name__}: {ex}", 0
                print(f"    {nm:>26}: {v:>8}  br={nb:<4d} {obj.n_evals:>6d} evals  "
                      f"{time.time()-t0:6.1f}s  {why[:58]}", flush=True)



# --- the discriminating case ------------------------------------------------------------------
#
# Isolated levels on a wide window do not separate the policies: family A's minimum relative
# gap is 2.0e-03, far above every floor measured, so a bracket isolates whatever the floor.
# The frozen rank's blown floor only bites when something must be resolved BELOW it. So the
# configuration that discriminates is the one nobody has run: a WIDE window that CONTAINS a
# split degeneracy. Same rectangle as the section-34 grid, a window 30x wider.

def split_case_wide(N, prec, band, lo=5.0, hi_frac=1.03):
    """The section-34 split rectangle, over a wide window instead of its own +-3%."""
    narrow = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: None})
    if not narrow:
        return None, None
    L, H = narrow[0].dims
    lam = N * rs.PI2
    a, b = lo, lam * hi_frac
    case = rs.make_case(f"S:N={N}/b={band:g}/p={prec:.0e}/[{a:.0f},{b:.0f}]", L, H, a, b)
    t = rs.complete_truth(case)
    return case, t[(t >= a) & (t <= b)]


def run_split_wide(precs, bands, N=145, lo=5.0):
    for prec in precs:
        for band in bands:
            case, truth = split_case_wide(N, prec, band, lo=lo)
            if case is None:
                continue
            p = case.pencil
            t = np.sort(np.asarray(truth, dtype=float))
            g = np.diff(t); sep = g[g > 0].min() if g.size else float("nan")
            print(f"\n{'='*92}\n=== N={N} band={band:g} prec={prec:.0e}  "
                  f"[{case.a:.0f}, {case.b:.0f}]  {t.size} levels  {p.n_cols} cols  "
                  f"min sep {sep:.2e} ({sep/t[-1]:.1e} rel)", flush=True)
            objs, frozen = objectives(p, case.a, case.b, prec)
            print(f"    frozen rank {frozen.rank}, threshold rank "
                  f"{p.rank_at(float(case.a)):d}..{p.rank_at(float(case.b)):d}", flush=True)
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
                      f"{time.time()-t0:6.1f}s  {why[:56]}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scope"
    if mode == "scope":
        run((1e-8,), WIDTHS, include_wide=True)
    elif mode == "iso":
        run((1e-6, 1e-8, 1e-10, 1e-12, 1e-14), WIDTHS, include_wide=True)
    elif mode == "splitscope":
        run_split_wide((1e-10,), (0.5, 1.3, 6.0))
    else:
        run_split_wide((1e-6, 1e-8, 1e-10, 1e-12, 1e-14), (0.5, 1.3, 6.0))
