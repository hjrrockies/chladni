"""Controls: stand-in refiners that validate the pipeline SCORER rather than any refiner.

TEST SUPPORT ONLY. The refine-stage counterpart of `bracket_controls.py`, `CHLADNI_V1.md`
section 5.2. Each factory takes truth -- controls may read it, no candidate may -- and returns a
callable with `chladni.mps.search.refine`'s signature, so it can be patched into `solve` and
the whole pipeline scored around it by `tests/prec_score.py`.

# Why these exist

The campaign opened on a refiner that stopped at 1e-11 to 1e-9 on pencils good to 1e-15 and
REPORTED it honestly, and the ladder benchmark accepted every such cell as a basis limit.
Nothing had checked that the scorer could tell "short, and the basis could have served it" from
"short, because it could not". The adversaries below are that failure and its neighbours, each
built to land on exactly one verdict.

They are FABRICATED rather than real refiners stopped early, deliberately. A real refiner
stopped after one step can land within `prec` by luck and pass, which would make the control
flaky in exactly the way a control must not be. Every adversary sits ten licences off the true
value, which at `prec = 1e-14` is 1e-13 relative and representable at every level used.
"""
from __future__ import annotations

import numpy as np

from chladni.mps.refine import RefineResult

__all__ = ["oracle_exact", "oracle_at_tolerance", "stops_short_claims_met",
           "stops_short_honestly", "dishonest_bar", "rejects_everything",
           "ORACLES", "ADVERSARIES"]


def _inside(truth, lo, hi):
    """The true value inside `[lo, hi]` nearest its centre, or `None`."""
    t = np.asarray(truth, dtype=float)
    inside = t[(t >= lo) & (t <= hi)]
    if inside.size == 0:
        return None
    return float(inside[np.argmin(np.abs(inside - 0.5 * (lo + hi)))])


def _reported(truth, place, bar):
    """A refiner that reports `place(x, prec)` with bar `bar(x, prec)` for the true value `x`."""
    def fn(obj, lo, hi, *, prec=None, lam_hint=None, **_):
        x = _inside(truth, lo, hi)
        if x is None:
            return RefineResult(lam=0.5 * (lo + hi), sigma=float("nan"), accepted=False,
                                reason="not_a_minimum", evaluations=0, bracket=(lo, hi))
        lam = float(place(x, prec))
        return RefineResult(lam=lam, sigma=float(obj.tension(obj.from_lam(lam))), accepted=True,
                            reason="converged", evaluations=1, bracket=(lo, hi),
                            delta_res=float(bar(x, prec)), steps=1, width=0.0)
    return fn


# --- oracles: correct answers, which must pass ----------------------------------------------------

def oracle_exact(truth):
    """The eigenvalue itself, with a bar of a few ulps."""
    return _reported(truth, lambda x, p: x, lambda x, p: 4.0 * np.spacing(x))


def oracle_at_tolerance(truth):
    """Half a licence off, with an honest bar of one licence. Within `prec`, so correct -- a scorer
    demanding more than was asked rejects it."""
    return _reported(truth, lambda x, p: x * (1 + 0.5 * p), lambda x, p: p * x)


# --- adversaries: each must fail by the verdict named beside it -----------------------------------

def stops_short_claims_met(truth):
    """Ten licences off, claiming a bar of a tenth of one. Never excusable."""
    return _reported(truth, lambda x, p: x * (1 + 10 * p), lambda x, p: 0.1 * p * x)


def stops_short_honestly(truth):
    """The campaign's opening defect: ten licences off, reported with a bar that contains it, on
    an objective whose basis could have served the request. Short and honest is not enough."""
    return _reported(truth, lambda x, p: x * (1 + 10 * p), lambda x, p: 20 * p * x)


def dishonest_bar(truth):
    """Ten licences off, admitting a shortfall but reporting a bar of five. Clause (3)."""
    return _reported(truth, lambda x, p: x * (1 + 10 * p), lambda x, p: 5 * p * x)


def rejects_everything(truth):
    """Refuses every bracket. The eigenvalues were bracketed and then thrown away."""
    def fn(obj, lo, hi, *, prec=None, lam_hint=None, **_):
        return RefineResult(lam=0.5 * (lo + hi), sigma=float("nan"), accepted=False,
                            reason="not_a_minimum", evaluations=0, bracket=(lo, hi))
    return fn


ORACLES = {"oracle_exact": oracle_exact, "oracle_at_tolerance": oracle_at_tolerance}

ADVERSARIES = {
    "stops_short_claims_met": (stops_short_claims_met, "shortfall-silent"),
    "stops_short_honestly": (stops_short_honestly, "b-fail"),
    "dishonest_bar": (dishonest_bar, "bar-violated"),
    "rejects_everything": (rejects_everything, "missed"),
}
