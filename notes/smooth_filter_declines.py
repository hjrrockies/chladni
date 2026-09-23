"""Section 34's decline measurement, re-run against filtered objectives.

Produces the tables in `notes/smooth_filter.md` section 7.3. Run from the repo root:

    .venv/bin/python notes/smooth_filter_declines.py "145*pi^2"

Roughly 2.3 hours for 5 precs x 11 bands x 5 objectives. EXPERIMENTAL; nothing in `src/`
uses it, and its verdict criterion is `bracket_score.BracketScore.ok`'s, restated here so the
runner has no dependency on the scorer's cost machinery.


Section 34: on the split degeneracy at 145*pi^2, "30 of 55 cells decline, and every one
carries `_seed`'s message, 'the slope estimate never settled'". The cause it identifies is the
threshold rank flipping between adjacent samples; the remedy it adopts is FREEZING the rank,
which recovers 22 of 23 declines on that grid at the cost of the floor (section 35).

The question here: does a FILTERED objective recover the same declines, without paying that
floor? Only `Objective.tensions` varies; pencil, collocation, bands, `prec` ladder and the
bracketer are identical throughout.
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

PRECS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
BANDS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.6, 3.4, 4.5, 6.0)   # 11, as in section 34
GRIDS = {"145*pi^2": 145, "65*pi^2": 65, "50*pi^2": 50}

EPS = np.finfo(float).eps

#: PREC-AWARE filter strengths. The first run fixed `mu` and `c` across a `prec` ladder
#: spanning eight decades, which cannot work: locating uncertainty tracks the floor, the pair
#: separation here is `band*prec*lam`, so a floor of 1e-13 is guaranteed to lose a 1e-14 pair.
#: That is what its 22 "wrong" cells were, every one of them at prec 1e-12 or 1e-14. A filter
#: has to be a function of `prec`, exactly as `rtol` already is.
OBJ = [
    ("threshold (current)", lambda p, x, pr: p.tensions(float(x))),
    ("frozen rank",         None),
    ("hybrid c=sqrt(prec*eps)",
     lambda p, x, pr: tension_hybrid(*p.pencil(float(x)), np.sqrt(pr * EPS), 2.0)),
    ("hybrid c=0.01*prec",
     lambda p, x, pr: tension_hybrid(*p.pencil(float(x)), 0.01 * pr, 2.0)),
    ("tikhonov mu=0.01*prec",
     lambda p, x, pr: tension_tikhonov(*p.pencil(float(x)), 0.01 * pr)),
]


def cases_for(N, prec):
    """The split-degeneracy cases at level N, one per band."""
    out = []
    for band in BANDS:
        cs = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: None})
        out += list(cs)
    return out


def verdict(res, truth, prec, variable):
    """right / wrong / declined, on the section-34 criterion: every true level bracketed,
    no bracket holding two levels further than `prec*lam` apart."""
    if res.declined:
        return "declined", res.diagnostics.decline_reason
    # `bracket_interval` returns in the OBJECTIVE's variable (candidates_v1.py:31), which
    # for a rect_suite case is lam -- so convert through the harness, never by squaring here.
    brs = bsc.to_lam_brackets([(b.lo, b.hi) for b in res.brackets], variable=variable)
    t = np.sort(np.asarray(truth, dtype=float))
    hit = np.zeros(t.size, dtype=int)
    impure = 0
    for lo, hi in brs:
        ix = np.nonzero((t >= lo) & (t <= hi))[0]
        hit[ix] += 1
        if ix.size > 1 and t[ix[-1]] - t[ix[0]] > prec * max(abs(t[ix[-1]]), 1.0):
            impure += 1
    return ("right", "") if (hit > 0).all() and impure == 0 else ("wrong", "")


def run_grid(N, label):
    print(f"\n{'='*86}\n=== {label}  ({len(PRECS)} precs x {len(BANDS)} bands "
          f"= {len(PRECS)*len(BANDS)} cells)\n{'='*86}")
    tally = {nm: {"right": 0, "wrong": 0, "declined": 0, "slope": 0, "n": 0}
             for nm, _ in OBJ}
    reasons = {nm: {} for nm, _ in OBJ}
    byprec = {nm: {pr: {"right": 0, "wrong": 0, "declined": 0} for pr in PRECS}
              for nm, _ in OBJ}
    for prec in PRECS:
        for case in cases_for(N, prec):
            p = case.pencil
            frozen = MPSPencil.from_basis(p.basis, p.bdry, p.interior, reg=p.reg,
                                          rtol=p.rtol, lam_ref=(case.a, case.b))
            for nm, tf in OBJ:
                fn = ((lambda x, fr=frozen: fr.tensions(float(x))) if tf is None
                      else (lambda x, tf=tf, p=p, pr=prec: tf(p, x, pr)))
                base = bsc.objective_for(case, prec)
                obj = Objective(fn, base.lo, base.hi, prec=prec,
                                variable=getattr(case, "variable", "lam"))
                try:
                    res = bracket_interval(obj, prec=prec)
                    v, why = verdict(res, case.truth, prec, case.variable)
                except Exception as ex:
                    v, why = "wrong", f"{type(ex).__name__}"
                tally[nm][v] += 1
                tally[nm]["n"] += 1
                byprec[nm][prec][v] += 1
                if v == "declined":
                    key = why.split(":")[0][:44]
                    reasons[nm][key] = reasons[nm].get(key, 0) + 1
                    if "never settled" in why:
                        tally[nm]["slope"] += 1
        print(f"  prec={prec:.0e} done", flush=True)
    print(f"\n{'objective':>24} {'right':>7} {'wrong':>7} {'declined':>9} "
          f"{'of which slope':>15}")
    for nm, _ in OBJ:
        t_ = tally[nm]
        print(f"{nm:>24} {t_['right']:>7d} {t_['wrong']:>7d} {t_['declined']:>9d} "
              f"{t_['slope']:>15d}")
    print("\nper prec   (right / wrong / declined)")
    print(f"{'objective':>24} " + " ".join(f"{('p=%.0e' % pr):>16}" for pr in PRECS))
    for nm, _ in OBJ:
        cells = " ".join(f"{byprec[nm][pr]['right']:>5d}/{byprec[nm][pr]['wrong']:>4d}/"
                         f"{byprec[nm][pr]['declined']:<5d}" for pr in PRECS)
        print(f"{nm:>24} {cells}")
    print("\ndecline reasons:")
    for nm, _ in OBJ:
        if reasons[nm]:
            print(f"  {nm:>24}: " + "; ".join(f"{k} x{v}" for k, v in
                                              sorted(reasons[nm].items(), key=lambda kv: -kv[1])))
    return tally


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "145*pi^2"
    t0 = time.time()
    run_grid(GRIDS[which], which)
    print(f"\n{time.time()-t0:.0f}s")
