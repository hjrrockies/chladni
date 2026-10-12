"""The black-box path: `Eigenproblem(dom).solve(n, prec)`.

`docs/design/douse_requirements.md`, "Spectrum" and "Plans". A Dirichlet polygon goes in, and the
first `n` eigenvalues come out as a `Spectrum`: sorted, the level at `n` returned WHOLE, each
level with its certified bound, its margin and its flags, plus the count audit and the plan that
produced it. Nothing about the basis has to be supplied.

**Planning** (`Eigenproblem.plan`) climbs `chladni.plan.LADDER`, cheapest rung first, solving the
prefix window at each. A rung is ACCEPTED when every level up to `n + HEADROOM` meets `prec`, the
count audit is ok (or unaudited), and no count is uncertain. With `confirm=True` there is one
more condition, the starved-basis guard: every level with margin < 20 must keep its count on the
next rung. A basis too poor to separate a pair can read a double as simple at a level that
otherwise meets `prec` (`notes/margin_rule_offline.md`). It is off by default because it costs a
trial solve.

**When no rung is accepted**, the best rung (most levels met) is returned with FLAGS saying so,
or `PrecisionShortfall` is raised under `on_shortfall="raise"`. Some shapes reach no rung at
1e-10, the sharpest thin triangles among them (`notes/hard_cases.md`). The contract is that they
are REPORTED, never returned as if they met `prec`.

**The window** starts at half the Faber-Krahn bound, so it is a prefix by construction and the
count audit applies even to elongated shapes. Its top is the Weyl estimate of level
`n + HEADROOM`, padded, and it widens until that many are found.
"""
from __future__ import annotations

import numpy as np

from ._core import PREC, check_prec
from .eigensolver import Spectrum, SpectrumLevel
from .plan import HEADROOM, LADDER, plan_for

__all__ = ["Eigenproblem"]

#: A level with margin below this has its count confirmed on the next rung (`confirm=True`):
#: `basis_sufficiency`'s multiplicity precondition (`20 eps/c <= prec lam`).
CONFIRM_MARGIN = 20.0
_TOP_PAD = 0.15
_MAX_WIDEN = 8


class Eigenproblem:
    """The Dirichlet Laplacian on `domain`, solved by planning a basis. See the module docstring."""

    def __init__(self, domain):
        self.domain = domain

    # --- planning ----------------------------------------------------------------------------

    def plan(self, n, prec=PREC, *, confirm=False):
        """The first `Plan` on the ladder that meets the request for the first `n` levels, or the
        best one if none does. Cheap to keep: it is frozen, picklable and shape-relative."""
        return self._plan(int(n), check_prec(prec), confirm)[0]

    def _plan(self, n, prec, confirm):
        """`(plan, result, window, accepted)`."""
        tried = {}

        def at(rung):
            if rung not in tried:
                p = plan_for(self.domain, n, prec, rung)
                res, window = (self._solve_with(p, n, prec, "flag") if _usable(p)
                               else (None, None))
                tried[rung] = (p, res, window)
            return tried[rung]

        best = None
        for rung in range(len(LADDER)):
            p, res, window = at(rung)
            if res is None:                     # nothing to place on this shape: skip the rung
                continue
            ok = _accepts(res, n, prec)
            score = _met_fraction(res, n, prec)
            if best is None or score > best[0]:
                best = (score, p, res, window)
            if not ok:
                continue
            if confirm and rung + 1 < len(LADDER):
                low = [L for L in _prefix(res, n) if np.isfinite(L.margin)
                       and L.margin < CONFIRM_MARGIN]
                nxt = next((at(r)[1] for r in range(rung + 1, len(LADDER))
                            if at(r)[1] is not None), None) if low else None
                if nxt is not None and not _same_counts(low, nxt):
                    continue                    # a hidden partner: the next rung sees more
            return p, res, window, True
        return best[1], best[2], best[3], False

    # --- solving -----------------------------------------------------------------------------

    def solve(self, n, prec=PREC, plan=None, on_shortfall="flag", *, confirm=False):
        """The first `n` levels (the level at `n` whole) as a `Spectrum`. See the module docstring.

        `plan=None` plans (`Eigenproblem.plan`). A given `plan` is realized on this domain. If it
        was made for a looser `prec` than requested, the shortfall is reported, not fixed by
        enlarging the basis: changing `prec` is a re-plan. `on_shortfall="raise"` raises
        `PrecisionShortfall` instead of returning a flagged result."""
        from .mps.solver import _raise_on_shortfall

        if on_shortfall not in ("flag", "raise"):
            raise ValueError(f"on_shortfall must be 'flag' or 'raise', got {on_shortfall!r}")
        n, prec = int(n), check_prec(prec)
        flags = []
        if plan is None:
            plan, res, window, accepted = self._plan(n, prec, confirm)
            if not accepted:
                flags.append(f"planner: no rung met prec={prec:.0e}; returning rung {plan.rung} "
                              f"of {len(LADDER) - 1}, the one with the most levels met")
                if on_shortfall == "raise":
                    _raise_on_shortfall(res, *window, prec)
        else:
            if plan.prec > prec * (1 + 1e-9):
                flags.append(f"plan: made for prec={plan.prec:.0e}, looser than the requested "
                             f"{prec:.0e}; re-plan to meet it")
            res, window = self._solve_with(plan, n, prec, on_shortfall)
        return _spectrum(res, n, prec, plan, flags)

    def _solve_with(self, plan, n, prec, on_shortfall):
        """`(result, (lo, hi))`: the prefix window solved on `plan`'s basis, widened until
        `n + HEADROOM` eigenvalues are found (or the search declines)."""
        from . import asymp
        from .bounds import faber_krahn

        lo = 0.5 * float(faber_krahn(self.domain))
        hi = float(asymp.weyl_est_poly(n + HEADROOM, self.domain)) * (1.0 + _TOP_PAD)
        solver = plan.solver(self.domain, prec=prec)
        for _ in range(_MAX_WIDEN):
            res = solver.search_interval(lo, hi, on_shortfall=on_shortfall)
            if res.declined or sum(L.mult for L in res.levels) >= n + HEADROOM:
                break
            hi = lo + 2.0 * (hi - lo)
        res.solver = solver                     # for `Spectrum.hadamard`
        return res, (lo, hi)


# --- helpers -------------------------------------------------------------------------------------

def _usable(plan):
    """A rung can place nothing on some shapes: singular-corner rungs on a shape whose corners are
    all reflection-closed. Such a plan is skipped."""
    return plan.n_cols > 0


def _prefix(res, n):
    """The levels holding the first `n + HEADROOM` eigenvalues (the last one whole)."""
    out, count = [], 0
    for L in sorted(res.levels, key=lambda L: L.lam):
        if count >= n + HEADROOM:
            break
        out.append(L)
        count += L.mult
    return out


def _accepts(res, n, prec):
    d = res.diagnostics
    if d.declined or d.n_uncertain:
        return False
    lv = _prefix(res, n)
    if sum(L.mult for L in lv) < n + HEADROOM:
        return False
    audit = d.count_audit
    if audit is not None and audit.verdict in ("short", "excess"):
        return False
    return all(L.met(prec) for L in lv)


def _met_fraction(res, n, prec):
    lv = _prefix(res, n)
    if not lv:
        return -1.0
    return sum(L.mult for L in lv if L.met(prec)) / max(n + HEADROOM, 1)


def _same_counts(low, res2):
    """Does a richer basis read the same count at every low-margin level?"""
    lv2 = res2.levels
    for L in low:
        r = max(L.resolution, max((M.resolution for M in lv2
                                   if abs(M.lam - L.lam) <= L.resolution), default=0.0))
        if sum(M.mult for M in lv2 if abs(M.lam - L.lam) <= r) != L.mult:
            return False
    return True


def _spectrum(res, n, prec, plan, flags):
    d = res.diagnostics
    kept, count = [], 0
    for L in sorted(res.levels, key=lambda L: L.lam):
        if count >= n:
            break
        kept.append(L)
        count += L.mult
    flags = list(flags)
    if count < n:
        flags.append(f"short: found {count} of the {n} eigenvalues asked for")
    unmet = [L for L in kept if not L.met(prec)]
    if unmet:
        flags.append(f"unmet: {len(unmet)} of {len(kept)} levels miss prec={prec:.0e}, the first "
                     f"at lam = {unmet[0].lam:.10g} (margin {unmet[0].margin:.3g}; 2 is needed)")
    audit = d.count_audit
    if audit is not None and audit.verdict != "ok":
        flags.append(f"count audit: {audit.verdict}"
                     + (f" ({audit.reason})" if audit.reason else
                        f" (mean Weyl deficit {audit.statistic:.3g})"))
    if d.n_uncertain:
        flags.append(f"uncertain: {d.n_uncertain} count(s) could not be certified")
    if d.declined:
        flags.append(f"declined: {d.decline_reason}")
    lam = np.array([L.lam for L in kept], dtype=float)
    mult = np.array([L.mult for L in kept], dtype=int)
    bound = np.array([L.bound for L in kept], dtype=float)
    levels = tuple(SpectrumLevel(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                                 margin=float(L.margin), resolution=float(L.resolution),
                                 met=bool(L.met(prec)), count_shortfall=bool(L.count_shortfall),
                                 count_uncertain=bool(L.count_uncertain or L.grouping_ambiguous))
                   for L in kept)
    return Spectrum(lam, mult, bound, prec=prec, declined=bool(d.declined),
                    reason=d.decline_reason if d.declined else "", levels=levels,
                    count_audit=audit, flags=tuple(flags), plan=plan,
                    source=getattr(res, "solver", None))
