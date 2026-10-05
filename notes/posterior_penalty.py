"""Can the penalty serve every `prec` if `tau` is FIXED and the search reads POSTERIOR tensions?

The question `notes/moler_payne_vs_tension.md` section 1 left open. The order-1 penalty's two jaws
-- a floor of ~1.4 tau that must stay under the licence, and rounding noise below
~3 eps sqrt(n) -- meet at 1e-14 when `tau` scales with `prec` (`penalty_prec_limit_explained.md`).
If `tau` is instead fixed near the noise jaw, and everything the search reads is the UNPENALIZED
ratio on the penalized vectors, the floor jaw is gone and only the noise jaw is left.

The posterior tensions at `lam`: take the `k` lowest penalized generalized singular vectors `X`
(`MPSPencil.coef(lam, k)`), and solve the unpenalized `k x k` problem on their span -- the
singular values of `(B X) R^(-1)`, `I X = Q R`. That is a Rayleigh-Ritz step: it can only lower
each tension, it has no rank to flip (the penalty chose the span), and its floor is the basis's
and the arithmetic's, not `tau`'s. `k = MAX_BRANCHES` (8), the most branches the multiplicity
reader examines; the rest of the vector is `sigma_tau`'s own, which is never below them.

Rows, each the cluster-total grid's scoring (right / wrong, flagged or silent / declined) over
both clusters and all eleven bands, cells the multiplicity precondition excuses skipped:

    threshold    truncation at the derived rtol: what reg="auto" ships below 1e-8. The baseline.
    fixed tau    the penalty at tau = TAU, searched on sigma_tau as today: fixing tau ALONE.
    posterior    the same pencil, searched on the posterior tensions.

EXPERIMENTAL; nothing in `src/` uses it. Run from the repo root:

    .venv/bin/python notes/posterior_penalty.py [prec ...]
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
import scipy.linalg as la

import bracket_score as bs
import rect_suite as rs
from basis_sufficiency import sufficient
from chladni.mps import MPSPencil
from chladni.mps.multiplicity import MAX_BRANCHES
from chladni.mps.objective import Objective
from chladni.mps.search import solve

CLUSTERS = ((50, 3), (65, 4))
BANDS = (0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.0, 4.0, 6.0)
#: Fixed, absolute on the equilibrated pencil: near the noise jaw, `3 eps sqrt(n)` ~ 7.5e-15 at
#: n = 126, where `notes/moler_payne_vs_tension.md` measured a clean posterior V on one rectangle.
TAU = 2e-14


def posterior_tensions(q, lam, k=MAX_BRANCHES):
    """The unpenalized Rayleigh-Ritz tensions on the span of the `k` lowest penalized vectors,
    ascending, followed by the penalized tensions beyond `k`."""
    lam = float(lam)
    pen = q.tensions(lam)
    k = min(k, len(pen))
    B, I = q.pencil(lam)
    X = q.coef(lam, k)
    Qi, R = la.qr(I @ X, mode="economic")
    post = np.sort(la.svdvals(la.solve_triangular(R, (B @ X).T, trans="T").T))
    return np.concatenate([post, pen[k:]])


def pencil(p, prec, which):
    if which == "threshold":
        return MPSPencil(p.bdry, p.interior, reg="svd", prec=prec)
    return MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec,
                     rtol=TAU / np.sqrt(p.n_cols))


def row(prec, which):
    out = dict(right=0, flagged=0, silent=0, declined=0, evals=0, wrong_cells=[])
    for N, m in CLUSTERS:
        for band in BANDS:
            (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: m})
            if not sufficient(case, prec).multiplicity:
                continue
            q = pencil(case.pencil, prec, which)
            f = ((lambda x, q=q: posterior_tensions(q, x)) if which == "posterior"
                 else (lambda x, q=q: q.tensions(float(x))))
            base = bs.objective_for(case, prec)
            obj = Objective(f, base.lo, base.hi, prec=prec, variable=case.variable,
                            expected_count=getattr(base, "expected_count", None))
            try:
                res = solve(obj, prec=prec)
            except Exception as err:                    # noqa: BLE001 -- scored, and named
                out["silent"] += 1
                out["wrong_cells"].append((N, band, f"raised {type(err).__name__}"))
                continue
            out["evals"] += obj.n_evals
            total = sum(x.mult for x in res.levels)
            if res.declined:
                out["declined"] += 1
            elif total == m:
                out["right"] += 1
            else:
                key = "flagged" if res.diagnostics.n_uncertain else "silent"
                out[key] += 1
                out["wrong_cells"].append((N, band, f"{total} of {m}, {key}"))
    return out


def main(precs):
    for prec in precs:
        print(f"\nprec = {prec:.0e}", flush=True)
        for which in ("threshold", "fixed tau", "posterior"):
            t0 = time.time()
            r = row(prec, which)
            n = r["right"] + r["flagged"] + r["silent"] + r["declined"]
            print(f"  {which:>10}  {r['right']:2d}/{n} right  {r['flagged']:2d} flagged  "
                  f"{r['silent']:2d} silent  {r['declined']:2d} declined  "
                  f"{r['evals']:6d} evals  ({time.time() - t0:.0f}s)  {r['wrong_cells']}",
                  flush=True)


if __name__ == "__main__":
    main([float(a) for a in sys.argv[1:]] or [1e-10, 1e-12, 1e-14])
