"""How `tau` is chosen for the penalty regularization, and what it costs on the grids.

Reproduces `notes/smooth_filter.md` sections 10.5 and 10.6 -- the `prec`-proportional `rtol`
rule, the rule-free `tau` sweep, the clean band at `prec = 1e-12`, and the ladder at one C.
Section 10.4's figure has its own script, `notes/fig_penalty_prec_limit.py`.

The question these answer, in order:

    noise   what does `tau` cost in floor and in evaluation noise, per rule?   `noise`
    rules   does the shipped `rtol` rule clear the cluster grid? does C*prec?  `rules`
    sweep   rule-free: is there ANY `tau` that clears it at this `prec`?       `sweep`
    band    where are the band's edges, and does one C serve the ladder?       `band`

The scoring is the cluster-total grid's own: `solve` on a split degeneracy, right when the
declared total matches, declined when the search says so, wrong otherwise. Cells the
`basis_sufficiency` precondition excuses are skipped, as `tests/test_cluster_totals.py` does.

EXPERIMENTAL; nothing in `src/` uses it. Run from the repo root:

    .venv/bin/python notes/smooth_filter_tau.py sweep

Minutes, not seconds: a declining cell burns its whole budget, so the small-`tau` rows are the
slow ones (the 1e-12 sweep is about 6 minutes, `band` about 10).
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

import bracket_score as bs
import rect_suite as rs
from basis_sufficiency import sufficient
from chladni.mps import MPSPencil
from chladni.mps.objective import Objective
from chladni.mps.search import solve

EPS = np.finfo(float).eps

#: The cluster grid's two clusters: the even triple at 50*pi^2 and the uneven quadruple at
#: 65*pi^2, as `tests/test_cluster_totals.py` pairs them with their true multiplicities.
CLUSTERS = ((50, 3), (65, 4))
#: Six bands per cluster, so a row is 12 cells before the precondition excuses any.
BANDS = (0.3, 0.5, 1.0, 1.3, 2.6, 6.0)


def pencil_for(p, prec, tau):
    """The pencil a row is scored on. `tau=None` is the threshold baseline.

    `tau` is ABSOLUTE on the equilibrated pencil, and `regularize` forms it as
    `rtol * ||A||_F = rtol * sqrt(n_cols)`, so the `rtol` that delivers it divides by that.
    """
    if tau is None:
        return p
    return MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec,
                     rtol=float(tau) / np.sqrt(p.n_cols))


def row(prec, tau, bands=BANDS):
    """`(right, wrong, declined)` over the cluster row at this `prec` and `tau`."""
    r = w = d = 0
    for N, m in CLUSTERS:
        for band in bands:
            (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: m})
            if not sufficient(case, prec).multiplicity:
                continue                     # the precondition excuses it; so does the real grid
            q = pencil_for(case.pencil, prec, tau)
            base = bs.objective_for(case, prec)
            obj = Objective(lambda x, q=q: q.tensions(float(x)), base.lo, base.hi, prec=prec,
                            variable=case.variable,
                            expected_count=getattr(base, "expected_count", None))
            try:
                res = solve(obj, prec=prec)
            except Exception:
                w += 1
                continue
            if res.declined:
                d += 1
            elif sum(x.mult for x in res.levels) == m:
                r += 1
            else:
                w += 1
    return r, w, d


def _print_row(label, prec, tau, t0):
    r, w, d = row(prec, tau)
    print(f"  {label:>26} {r:3d} right {w:3d} wrong {d:3d} declined   ({time.time() - t0:.0f}s)",
          flush=True)


def noise():
    """Section 10.5's first half: floor and noise per rule, no solving."""
    import hard_roster as hr

    def probe(f, c, width=1e-9, n=11):
        xs = c * (1 + np.linspace(-width, width, n))
        ys = np.array([f(x) for x in xs])
        return np.abs(ys - np.polyval(np.polyfit(xs - c, ys, 2), xs - c)).max() / np.abs(ys).mean()

    c = rs.family_A()[0]
    e = hr.ROSTER["eq_tri"]()
    cases = [("rect", c.pencil, np.unique(rs.complete_truth(c))),
             ("eq_tri", e.pencil, np.unique(np.asarray(e.truth, dtype=float)))]
    print(f"{'domain':>7} {'prec':>7} {'rule':>16} {'tau':>9} {'worst floor':>12} "
          f"{'floor/prec':>11} {'noise':>9}")
    for name, p, truth in cases:
        lv = truth[np.linspace(0, truth.size - 1, min(5, truth.size)).astype(int)]
        g = np.diff(truth)
        mids = (truth[:-1] + g / 2)[np.argsort(g)[::-1][:3]]
        sq = np.sqrt(p.n_cols)
        for P in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
            rules = [("sqrt(prec*eps) [ship]", float(np.sqrt(P * EPS)) * sq)]
            rules += [(f"C={C:g} * prec", C * P) for C in (0.3, 0.1, 0.05)]
            for label, tau in rules:
                q = pencil_for(p, P, tau)
                fl = max(float(q.tension(float(l))) for l in lv)
                nz = max(probe(lambda x: float(q.tension(float(x))), m) for m in mids)
                print(f"{name:>7} {P:7.0e} {label:>16} {tau:9.2e} {fl:12.3e} {fl / P:11.3f} "
                      f"{nz:9.1e}", flush=True)
        print()


def rules():
    """Section 10.5: the shipped rule and `C*prec` against the threshold, three rungs."""
    for prec in (1e-14, 1e-12, 1e-10):
        print(f"\nprec = {prec:.0e}")
        n = np.sqrt(rs.split_degeneracy_cases(prec, bands=(1.0,),
                                              levels={50: 3})[0].pencil.n_cols)
        for label, tau in (("threshold", None),
                           ("penalty, shipped rtol", float(np.sqrt(prec * EPS)) * n),
                           ("penalty, tau=0.3*prec", 0.3 * prec),
                           ("penalty, tau=0.1*prec", 0.1 * prec)):
            _print_row(label, prec, tau, time.time())


def sweep():
    """Section 10.6: rule-free. Is there ANY `tau` that clears the row?"""
    for prec, taus in ((1e-12, [1e-15, 3e-15, 5e-15, 7e-15, 1e-14, 3e-14, 5e-14, 7e-14,
                                1e-13, 3e-13, 1e-12]),
                       (1e-14, [1e-15, 3e-15, 1e-14, 3e-14, 1e-13])):
        print(f"\nprec = {prec:.0e}   (floor ~ 1.4*tau; the licence is prec)")
        _print_row("threshold", prec, None, time.time())
        for tau in taus:
            _print_row(f"tau={tau:.0e} (={tau / prec:.3g}*prec)", prec, tau, time.time())


def band():
    """Section 10.6's second half: does one C serve the whole ladder? (It does not.)"""
    print("tau = 0.02 * prec, the middle of the 1e-12 band, across the ladder")
    for prec in (1e-6, 1e-8, 1e-10, 1e-12):
        print(f"\nprec = {prec:.0e}")
        _print_row("threshold", prec, None, time.time())
        _print_row("tau = 0.02*prec", prec, 0.02 * prec, time.time())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    {"noise": noise, "rules": rules, "sweep": sweep, "band": band}[which]()
