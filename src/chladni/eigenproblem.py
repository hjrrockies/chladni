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

**Warm start** (`solve(..., near=prev)`), for a sequence of nearby shapes on one plan, as douse
runs them. The window shrinks to `[min(0.98 FK, lam_1 - h_1), lam_n + h_n]` around the previous
values, `h` a third of the local mean Weyl spacing (lappy's tracking rule): still a prefix, but no
headroom. Completeness then rests on continuity, checked every time (`_warm_verdict`), with a cold
solve on any failure and every `WARM_COLD_EVERY` warm solves. Measured at douse's N = 6, k = 2
optimum after a 1e-3 move: 84 tension evaluations against 177, 2.4x faster, the same values to
1.2e-13 (`notes/warm_start.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ._core import PREC, check_prec
from .eigensolver import Level, Spectrum
from .plan import HEADROOM, LADDER, plan_for

__all__ = ["Eigenproblem", "SolveRecord"]

#: A level with margin below this has its count confirmed on the next rung (`confirm=True`):
#: `basis_sufficiency`'s multiplicity precondition (`20 eps/c <= prec lam`).
CONFIRM_MARGIN = 20.0
_TOP_PAD = 0.15
_MAX_WIDEN = 8
#: A warm solve's window half-width at each previous value, as a fraction of the local mean Weyl
#: spacing: lappy's `track`/`track_set` rule (`lappy/evp.py:472`).
WARM_SPACING_FRACTION = 1.0 / 3.0
#: A cold solve after this many consecutive warm ones: a backstop against slow drift past the
#: continuity checks (decided with the user, 2026-10-13).
WARM_COLD_EVERY = 10


@dataclass(frozen=True)
class SolveRecord:
    """What `Eigenproblem.solve` returns: the `Spectrum` (the value) and what produced it.

    `plan` is the frozen `chladni.plan.Plan` (realize it on a nearby shape, pass the record as
    `near=` for a warm solve); `solver` the `MPSEigensolver` the answer came from, which
    `hadamard` needs; `diagnostics` the search's `SolveDiagnostics` (evaluation counts by
    stage); `warm`, `warm_depth`, `warm_note` say whether this was a warm solve and, when a
    requested warm solve went cold, why. The spectrum's own fields are forwarded (`lam`,
    `unfolded`, `flags`, ...) so a record reads like the `Spectrum` it carries; prefer
    `.spectrum` in new code (2026-10-14)."""
    spectrum: Spectrum
    plan: object = None
    solver: object = field(default=None, repr=False, compare=False)
    diagnostics: object = field(default=None, repr=False, compare=False)
    warm: bool = False
    warm_depth: int = 0
    warm_note: object = None

    # --- what only the record can do -------------------------------------------------------

    def hadamard(self, i, V):
        """The Hadamard shape derivative of level `i` -- or of levels `i = [i0, i1, ...]` as one
        block, cross terms included -- `(n_fields, m, m)`: see `MPSEigensolver.hadamard`."""
        if self.solver is None:
            raise ValueError("this record carries no solver; solve with "
                             "chladni.eigenproblem.Eigenproblem to get Hadamard derivatives")
        idx = [int(i)] if np.ndim(i) == 0 else [int(j) for j in i]
        levels = [self.spectrum.levels[j] for j in idx]
        return self.solver.hadamard(levels[0] if np.ndim(i) == 0 else levels, V,
                                    prec=self.spectrum.prec)

    @property
    def domain(self):
        return self.solver.domain

    @property
    def basis(self):
        return self.solver.basis

    @property
    def source(self):
        """DEPRECATED name of `solver`."""
        return self.solver

    # --- the spectrum, forwarded -----------------------------------------------------------

    def __bool__(self):
        return bool(self.spectrum)

    lam = property(lambda self: self.spectrum.lam)
    mult = property(lambda self: self.spectrum.mult)
    bound = property(lambda self: self.spectrum.bound)
    levels = property(lambda self: self.spectrum.levels)
    unfolded = property(lambda self: self.spectrum.unfolded)
    count = property(lambda self: self.spectrum.count)
    unmet = property(lambda self: self.spectrum.unmet)
    achieved = property(lambda self: self.spectrum.achieved)
    prec = property(lambda self: self.spectrum.prec)
    flags = property(lambda self: self.spectrum.flags)
    declined = property(lambda self: self.spectrum.declined)
    reason = property(lambda self: self.spectrum.reason)
    count_audit = property(lambda self: self.spectrum.count_audit)

    def level_of(self, lam, rtol=1e-6):
        return self.spectrum.level_of(lam, rtol)

    def level_index(self, k):
        return self.spectrum.level_index(k)

    def neighbourhood(self, i, eps_abs):
        return self.spectrum.neighbourhood(i, eps_abs)

    def cluster(self, lams=None, **kw):
        return self.spectrum.cluster(lams, **kw)


class Eigenproblem:
    """The Dirichlet Laplacian on `domain`, solved by planning a basis. See the module docstring."""

    def __init__(self, domain):
        self.domain = domain

    # --- planning ----------------------------------------------------------------------------

    def plan(self, n, prec=None, *, confirm=False):
        """The first `Plan` on the ladder that meets the request for the first `n` levels, or the
        best one if none does. Cheap to keep: it is frozen, picklable and shape-relative.

        `prec=None` means the domain's own `prec` (`Domain.prec`), as `Eigensolver` reads it;
        the black-box path used to fall back to the global default instead, so a domain built
        with `prec=1e-12` was silently planned at 1e-8 (2026-10-14).

        Planning SOLVES the accepted rung; `plan_and_solve` returns that answer too, so a caller
        who wants both does not solve the same shape on the same basis twice."""
        return self.plan_and_solve(n, prec, confirm=confirm)[0]

    def plan_and_solve(self, n, prec=None, *, confirm=False):
        """`(plan, record)`: the plan `plan` would return, and the `SolveRecord` of the solve
        that selected it -- flagged as `solve` would flag it when no rung met the request.
        douse re-plans a drifted shape and then solved it again on the new plan; this is that
        second solve, for free (2026-10-14)."""
        n, prec = int(n), self._prec(prec)
        declined = self._declined(n, prec)
        if declined is not None:
            return declined.plan, declined
        plan, res, window, accepted = self._plan(n, prec, confirm)
        flags = []
        if not accepted:
            flags.append(f"planner: no rung met prec={prec:.0e}; returning rung {plan.rung} "
                          f"of {len(LADDER) - 1}, the one with the most levels met")
        return plan, _record(res, n, prec, plan, flags)

    def _declined(self, n, prec):
        """A DECLINED record when the geometry alone says no plan can meet `prec`
        (`plan.shape_refusal`), else None. A cheap "no" for a shape optimizer's wild trial: a
        value with `declined=True` and a `declined:` flag, as a search that gave up reports, not
        an exception, and before any solve (2026-10-14)."""
        from .plan import shape_refusal
        why = shape_refusal(self.domain, prec)
        if why is None:
            return None
        spectrum = replace(Spectrum.declining(why, prec), flags=(f"declined: {why}",))
        return SolveRecord(spectrum, plan=plan_for(self.domain, n, prec, 0))

    def _prec(self, prec):
        return check_prec(self.domain.prec if prec is None else prec)

    def _plan(self, n, prec, confirm):
        """`(plan, result, window, accepted)`.

        THE CLIMB ABORTS A RUNG EARLY. Each rung is solved with `on_shortfall="raise"`, so the
        search stops at the first refined level that misses `prec` (`mps.search._abort_if_unmet`:
        a level with margin < 2 was never certified met afterwards on 8062 benchmark levels, and
        stopping there saves a median 57% of a failing solve, `notes/early_shortfall.md`). Until
        2026-10-14 the climb solved every rung to the end with "flag". An aborted rung is ranked
        by what it had met before stopping; only when NO rung is accepted is the best-ranked rung
        solved in full, so the returned result is always a whole one."""
        from ._core import PrecisionShortfall
        tried = {}                              # rung -> (plan, res, window, aborted, score)

        def at(rung, full=False):
            if rung in tried and not (full and tried[rung][3]):
                return tried[rung]
            p = plan_for(self.domain, n, prec, rung)
            if not _usable(p):
                tried[rung] = (p, None, None, False, -1.0)
            elif full:
                res, window = self._solve_with(p, n, prec, "flag")
                tried[rung] = (p, res, window, False, _met_fraction(res, n, prec))
            else:
                try:
                    res, window = self._solve_with(p, n, prec, "raise")
                except PrecisionShortfall as e:
                    if e.result is not None:    # a late raise: the whole result is in hand
                        tried[rung] = (p, e.result, None, False,
                                       _met_fraction(e.result, n, prec))
                    else:                       # an early abort: rank by what was met so far
                        tried[rung] = (p, None, None, True, _aborted_fraction(e, n, prec))
                else:
                    tried[rung] = (p, res, window, False, _met_fraction(res, n, prec))
            return tried[rung]

        best = None
        for rung in range(len(LADDER)):
            p, res, window, aborted, score = at(rung)
            if res is None and not aborted:     # nothing to place on this shape: skip the rung
                continue
            if best is None or score > best[0]:
                best = (score, rung)
            if res is not None and res.diagnostics.declined:
                break                           # the search gave up here; a richer basis will not help
            if aborted or not _accepts(res, n, prec):
                continue
            if confirm and rung + 1 < len(LADDER):
                low = [L for L in _prefix(res, n) if np.isfinite(L.margin)
                       and L.margin < CONFIRM_MARGIN]
                nxt = next((at(r, full=True)[1] for r in range(rung + 1, len(LADDER))
                            if at(r, full=True)[1] is not None), None) if low else None
                if nxt is not None and not _same_counts(low, nxt):
                    continue                    # a hidden partner: the next rung sees more
            return p, res, window, True
        p, res, window, _, _ = at(best[1], full=True)
        return p, res, window, False

    # --- solving -----------------------------------------------------------------------------

    def solve(self, n, prec=None, plan=None, on_shortfall="flag", *, confirm=False, near=None,
              windows=None):
        """The first `n` levels (the level at `n` whole) as a `SolveRecord`: the `Spectrum` plus
        the plan, the solver and the search diagnostics. See the module docstring.

        `plan=None` plans (`Eigenproblem.plan`). A given `plan` is realized on this domain. If it
        was made for a looser `prec` than requested, the shortfall is reported, not fixed by
        enlarging the basis: changing `prec` is a re-plan. `on_shortfall="raise"` raises
        `PrecisionShortfall` instead of returning a flagged result.

        `near`, the `Spectrum` of a nearby shape solved on the same plan, asks for a WARM solve:
        the narrow window around its values, accepted only if every continuity check holds
        (`_warm_verdict`), otherwise a cold solve. `spec.warm`, `warm_depth` and `warm_note` say
        which happened and why. With `plan=None`, `near.plan` is used. A warm answer is
        accepted only unflagged, so `on_shortfall="raise"` has nothing to raise on it.

        `windows`, with `near`: per-value half-widths `h_i` for the first `n` values of `near`,
        from a caller that knows how far each can have moved (a shape optimizer, from the
        level's shape derivative and its step). The warm solve then re-locates each level in
        `[lam_i - h_i, lam_i + h_i]` (`MPSEigensolver.track_levels`) instead of searching the
        prefix: the solve is cheaper by the range it no longer covers, and COMPLETENESS IS THE
        CALLER'S -- nothing between the windows is examined, so the caller's step bound and a
        periodic cold solve (`WARM_COLD_EVERY`) are the guarantee (2026-10-14)."""
        if on_shortfall not in ("flag", "raise"):
            raise ValueError(f"on_shortfall must be 'flag' or 'raise', got {on_shortfall!r}")
        n, prec = int(n), self._prec(prec)
        # ONE SOLVER PER SOLVE. The basis is realized and the pencil built once here and shared
        # by the warm attempt and the cold fallback; a refused warm solve used to pay for a
        # second realize + pencil on the same domain (2026-10-14).
        solver = None
        if near is not None:
            if plan is None:
                plan = near.plan
            if plan is not None:
                solver = plan.solver(self.domain, prec=prec)
            spec, note = self._solve_warm(near, plan, n, prec, solver, windows)
            if spec is not None:
                return spec
        else:
            note = None
        spec = self._solve_cold(n, prec, plan, on_shortfall, confirm, solver)
        if note is not None:
            spec = replace(spec, warm_note=note)
        return spec

    def _solve_cold(self, n, prec, plan, on_shortfall, confirm, solver=None):
        from .mps.solver import _raise_on_shortfall

        flags = []
        if plan is None:
            declined = self._declined(n, prec)
            if declined is not None:
                if on_shortfall == "raise":
                    raise PrecisionShortfall(declined.reason)
                return declined
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
            res, window = self._solve_with(plan, n, prec, on_shortfall, solver)
        return _record(res, n, prec, plan, flags)

    def _solve_with(self, plan, n, prec, on_shortfall, solver=None):
        """`(result, (lo, hi))`: the prefix window solved on `plan`'s basis, widened until
        `n + HEADROOM` eigenvalues are found (or the search declines). `solver` is the realized
        solver when the caller already built one."""
        from . import asymp
        from .bounds import faber_krahn

        lo = 0.5 * float(faber_krahn(self.domain))
        hi = float(asymp.weyl_est_poly(n + HEADROOM, self.domain)) * (1.0 + _TOP_PAD)
        if solver is None:
            solver = plan.solver(self.domain, prec=prec)
        for _ in range(_MAX_WIDEN):
            res = solver.search_interval(lo, hi, on_shortfall=on_shortfall)
            if res.declined or sum(L.mult for L in res.levels) >= n + HEADROOM:
                break
            hi = lo + 2.0 * (hi - lo)
        res.solver = solver                     # for `Spectrum.hadamard`
        return res, (lo, hi)

    def _solve_warm(self, near, plan, n, prec, solver=None, windows=None):
        """`(spectrum, None)` if the warm solve is accepted, else `(None, why)`."""
        if plan is None or near.plan is not plan:
            return None, "plan: `near` was solved on a different plan"
        if near.warm_depth >= WARM_COLD_EVERY:
            return None, f"cadence: a cold solve every {WARM_COLD_EVERY} warm ones"
        if plan.prec > prec * (1 + 1e-9):
            return None, "plan: made for a looser prec than requested"
        prev = np.asarray(near.unfolded, dtype=float)[:n]
        if prev.size < n:
            return None, f"found: `near` holds {prev.size} values, fewer than {n}"
        if solver is None:
            solver = plan.solver(self.domain, prec=prec)
        if windows is not None:
            return self._solve_tracked(near, plan, n, prec, solver, prev, windows)
        a, b, h = _warm_window(self.domain, prev)
        res = solver.search_interval(a, b, on_shortfall="flag")
        res.solver = solver
        d = res.diagnostics
        found = np.sort(np.asarray(res.unfolded, dtype=float))
        why = _warm_verdict(found=found, prev=prev, n=n, b=b, h=h,
                            top=max((L.lam for L in res.levels), default=-np.inf),
                            declined=bool(d.declined), unmet=len(d.unmet),
                            uncertain=int(d.n_uncertain),
                            audit=d.count_audit.verdict if d.count_audit is not None else None)
        if why is not None:
            return None, why
        rec = _record(res, n, prec, plan, [])
        # A narrow window can leave the audit with nothing to say; continuity decided instead.
        spectrum = replace(rec.spectrum, flags=tuple(f for f in rec.spectrum.flags
                                                     if not f.startswith("count audit: unaudited")))
        rec = replace(rec, spectrum=spectrum, warm=True, warm_depth=near.warm_depth + 1)
        if rec.flags:                           # anything else means the checks missed a case
            return None, "flags: " + "; ".join(rec.flags)
        return rec, None


    def _solve_tracked(self, near, plan, n, prec, solver, prev, windows):
        """The warm solve on caller-supplied windows: `track_levels` on the previous LEVELS (the
        first `n` values of `near` grouped by its levels, each level's half-width the largest
        of its copies'), then the continuity verdicts that still apply."""
        h = np.broadcast_to(np.asarray(windows, dtype=float), prev.shape)
        lams, mults, hs, count = [], [], [], 0
        for L in near.levels:
            if count >= n:
                break
            take = min(int(L.mult), n - count)
            lams.append(float(L.lam))
            mults.append(int(L.mult))
            hs.append(float(np.max(h[count:count + take])))
            count += take
        res, why = solver.track_levels(np.array(lams), np.array(hs), mults=np.array(mults),
                                       prec=prec)
        if why is not None:
            return None, "track: " + why
        res.solver = solver
        d = res.diagnostics
        found = np.sort(np.asarray(res.unfolded, dtype=float))
        if len(d.unmet):
            return None, f"unmet: {len(d.unmet)} level(s) in the tracked windows miss prec"
        if found.size < n:
            return None, f"found: {found.size} values in the tracked windows, fewer than {n}"
        rec = _record(res, n, prec, plan, [])
        spectrum = replace(rec.spectrum, flags=tuple(f for f in rec.spectrum.flags
                                                     if not f.startswith("count audit: unaudited")))
        rec = replace(rec, spectrum=spectrum, warm=True, warm_depth=near.warm_depth + 1)
        if rec.flags:
            return None, "flags: " + "; ".join(rec.flags)
        return rec, None


# --- helpers -------------------------------------------------------------------------------------

def _mean_spacing(lam, domain):
    """The local mean Weyl spacing at `lam`: the gap between consecutive Weyl estimates."""
    from . import asymp
    k = max(float(asymp.weyl_count_poly(lam, domain)), 0.5)
    return float(asymp.weyl_est_poly(k + 1, domain) - asymp.weyl_est_poly(k, domain))


def _warm_window(domain, prev):
    """`(a, b, h)`: the warm window around the previous values `prev` (ascending, length n) and
    each value's allowed move `h`. `a` sits below Faber-Krahn, so the window is a PREFIX: nothing
    can be missed beneath it, and sorted index is eigenvalue index."""
    from .bounds import faber_krahn
    h = np.array([WARM_SPACING_FRACTION * _mean_spacing(v, domain) for v in prev])
    a = min(0.98 * float(faber_krahn(domain)), float(prev[0] - h[0]))
    a = max(a, 1e-3 * float(prev[0]))
    b = float(prev[-1] + h[-1])
    return a, b, h


def _warm_verdict(*, found, prev, n, b, h, top, declined, unmet, uncertain, audit):
    """None if a warm solve may be trusted, else the first continuity check it fails.

    `found` the values in the window, ascending and unfolded; `prev` the previous first `n`;
    `h` each one's allowed move; `top` the highest level found; the rest from the search."""
    if declined:
        return "declined: the search declined the warm window"
    if unmet:
        return f"unmet: {unmet} level(s) in the warm window miss prec"
    if uncertain:
        return f"uncertain: {uncertain} count(s) in the warm window could not be certified"
    if found.size < n:
        return f"found: {found.size} values in the warm window, fewer than {n}"
    moved = np.abs(found[:n] - prev[:n]) > h[:n]
    if np.any(moved):
        i = int(np.argmax(moved))
        return (f"moved: value {i + 1} moved {abs(found[i] - prev[i]):.3g}, more than its "
                f"allowed {h[i]:.3g} (an index shift, or too large a step)")
    if top > b - h[n - 1] / 3.0:
        return (f"edge: a level at {top:.10g} sits within {h[n - 1] / 3.0:.3g} of the window's "
                f"top {b:.10g}, where a cluster can be cut")
    if audit in ("short", "excess"):
        return f"audit: the count audit reads {audit}"
    return None


def _aborted_fraction(e, n, prec):
    """How much of the request an early-aborted rung had met before it stopped: the located
    values that met `prec` (each counted once; multiplicities are not read before the abort),
    over `n + HEADROOM`. A ranking for the best-rung fallback, not a measurement."""
    met = 0
    for item in e.located:
        r = item[0] if isinstance(item, tuple) else item
        lam, dres = getattr(r, "lam", None), getattr(r, "delta_res", None)
        if lam is not None and dres is not None and np.isfinite(dres) and dres <= prec * abs(lam):
            met += 1
    return met / max(n + HEADROOM, 1)


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


def _record(res, n, prec, plan, flags):
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
    spectrum = Spectrum(tuple(Level.from_search(L, prec) for L in kept), prec=prec,
                        declined=bool(d.declined),
                        reason=d.decline_reason if d.declined else "",
                        count_audit=audit, flags=tuple(flags))
    return SolveRecord(spectrum, plan=plan, solver=getattr(res, "solver", None), diagnostics=d)
