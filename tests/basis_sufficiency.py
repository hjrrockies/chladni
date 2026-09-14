"""Can a case's basis serve a `prec`? The one source of every excuse a benchmark grants.

TEST SUPPORT ONLY. `tests/test_basis_sufficiency.py` pins the criteria on exact cones.

Three tiers learned the same lesson separately -- `surrogate_cases.adequate`,
`rect_suite.assert_floor`, `hard_roster.finest_prec` -- and each scored cells past its
objective's limit before the check existed, which measures the objective and not the method.
This module states the criteria once, and `tests/prec_score.py` reads its verdict to decide
whether a shortfall, a merge or a wrong total is excused.

# The criteria, all read at the TRUE eigenvalues and never from a solve

    accuracy      3 * (eps/c) / lam  <=  prec    at every level, eps = sigma_1 there
    multiplicity  20 * (sigma_mu/c) / lam <= prec   at every level, sigma_mu at its multiplicity
    truth         digits >= -log10(prec) + 1    a table good to d digits is not a reference
                                                 for prec = 10**-d itself; limits both excuses

The accuracy margin of 3 is the BRACKETING criterion, `assert_floor`'s: a bracket of width
`prec*lam` is usable while it is wider than the flat bottom `eps/c`. The multiplicity factor of
20 is `search.md` section 2.2's contract for telling a pair apart. They differ, and importing
either into the other's question is the mistake `assert_floor`'s docstring records.

`delta_res` is deliberately NOT an input. It is what the solver reports about itself, so using
it to excuse the solver is circular -- and a refiner that stopped early reports a large one.

# What is flagged but never excused

A pencil whose `rtol` equals `sqrt(prec*eps)` for its own `prec` was built with a rank cutoff
that moves with the request, so the objective differs between rungs of the ladder and a cell
confounds the basis with the rank policy. `frozen = False` with a reason; the excuses are
unaffected, so a misconfigured case fails rather than being excused.
"""
from __future__ import annotations

import numpy as np

from prec_score import Sufficiency

__all__ = ["bracketing_prec", "multiplicity_prec", "rtol_is_frozen", "sufficient"]

BRACKET_MARGIN = 3.0
MULTIPLICITY_MARGIN = 20.0
#: Offset for the wall-slope probe, relative to `lam`. `rect_suite.assert_floor` measured it
#: stable from 5e-3 down to 2e-4; at 2e-2 the probe reaches the saturated part of the curve.
H_FRAC = 0.005
#: Closed forms report 15.6 digits (`chladni.reference`); a case without a `digits` attribute
#: is assumed to carry closed-form truth.
DEFAULT_DIGITS = 15.6


def _levels(case, h_frac=H_FRAC):
    """Per distinct true level: `(lam, mult, sigma_1, sigma_mu, c)`."""
    p = case.pencil
    lv, mult = np.unique(np.asarray(case.truth, dtype=float), return_counts=True)
    for x, m in zip(lv, mult):
        s = np.asarray(p.tensions(float(x)), dtype=float)
        smu = float(s[m - 1]) if s.size >= m else np.inf
        h = h_frac * abs(x)
        c = 0.5 * (p.tension(x + h) + p.tension(x - h)) / h
        yield float(x), int(m), float(s[0]), smu, float(c)


def bracketing_prec(case):
    """The finest `prec` the ACCURACY of this case's objective supports."""
    worst = 0.0
    for x, _, s1, _, c in _levels(case):
        if c <= 0:
            return np.inf
        worst = max(worst, BRACKET_MARGIN * (s1 / c) / abs(x))
    return float(worst)


def multiplicity_prec(case):
    """The finest `prec` at which this case's objective can tell a cluster's members apart."""
    worst = 0.0
    for x, _, _, smu, c in _levels(case):
        if c <= 0:
            return np.inf
        worst = max(worst, MULTIPLICITY_MARGIN * (smu / c) / abs(x))
    return float(worst)


def rtol_is_frozen(case):
    """False when the pencil's `rtol` is the prec-derived default for its own `prec`.

    A case may state `rtol_frozen` explicitly, which wins: an `rtol` fixed to
    `sqrt(1e-14*eps)` by hand is indistinguishable from the default of a pencil built at
    `prec = 1e-14`, and only the case knows which it meant.
    """
    stated = getattr(case, "rtol_frozen", None)
    if stated is not None:
        return bool(stated)
    p = case.pencil
    rtol, prec = getattr(p, "rtol", None), getattr(p, "prec", None)
    if rtol is None or prec is None:
        return True
    derived = float(np.sqrt(prec * np.finfo(float).eps))
    return not np.isclose(rtol, derived, rtol=1e-12, atol=0.0)


def sufficient(case, prec):
    """The `Sufficiency` of `case` at `prec`, with a reason for every excuse it grants."""
    reasons = []
    bp, mp = bracketing_prec(case), multiplicity_prec(case)
    digits = float(getattr(case, "digits", DEFAULT_DIGITS))
    need = -np.log10(prec) + 1.0

    accuracy = bp <= prec
    if not accuracy:
        reasons.append(f"basis: accuracy needs prec >= {bp:.2e} (3*eps/c/lam)")
    multiplicity = mp <= prec
    if not multiplicity:
        reasons.append(f"basis: multiplicity needs prec >= {mp:.2e} (20*sigma_mu/c/lam)")
    if digits < need:
        accuracy = multiplicity = False
        reasons.append(f"truth: {digits:.1f} digits, prec {prec:.0e} needs {need:.1f}")

    frozen = rtol_is_frozen(case)
    if not frozen:
        reasons.append("configuration: the pencil's rtol is derived from its prec, so the "
                       "objective moves with the request")
    return Sufficiency(accuracy=bool(accuracy), multiplicity=bool(multiplicity),
                       frozen=bool(frozen), reasons=tuple(reasons))
