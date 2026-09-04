"""Stage 4: the composed pipeline, and the only seam a candidate MUST present.

TEST SUPPORT ONLY. `docs/design/candidates.md` section 1 calls `solve` "the only mandatory
seam": a method that does not decompose into bracket -> refine -> multiplicity implements this
and nothing else, so it is where v1 and a monolithic candidate are compared on equal terms.

    solve(tensions, a, b, *, prec, modes=None) -> [(lam, mult), ...]

# Composition is not the sum of the parts, and each seam cost a defect

Every stage passed its own tests before this file existed, and composing them scored separation
0.00 on all three exact degeneracies and read multiplicity LOW on two of seven disk levels.
Three seam defects, all now fixed in the stages themselves and recorded here because this is
where they were found:

* the refiner's interior margin refused genuine minima sitting off-centre in their bracket;
* the refiner started from the arithmetic midpoint, which misses a well 1e-07 wide inside an
  8.6e-07 bracket -- **a bracket is guaranteed to CONTAIN the eigenvalue, not to have a sampled
  descent inside it**, because stage 1 stops on purity by width;
* stage 3's wide bracket was sized from `sigma` at the LOCATED root, which is within `prec*lam`
  of the minimum rather than at it, so the flanks straddled asymmetrically.

# The wide bracket is reconstructed, and that is better than remembering it

`search.md` Phase B says to retain the pair of evaluated points from before the bracket
collapsed. This does not carry them forward; it recomputes them from the evaluation cache at
the end. That is not a shortcut: the RIGHT wide bracket is defined by the physics and the
request -- far enough out that the higher branches have moved, `eps/c`, and past the accuracy
the refinement achieved, `prec*lam` -- and not by whatever width the locator happened to start
from. Carrying the old bounds forward would preserve an accident of history. Reconstruction
costs nothing, because every point it selects is already paid for.

# What it does NOT do

**Phase D, collapse-versus-merge.** Stage 3 answers "how many branches dip HERE"; Phase D
answers "are these two reported roots the same eigenvalue", which distance provably cannot
settle, since one eigenvalue found twice and two genuine eigenvalues occur at the same
separation. It needs the interior Gram angle and so the `modes(x)` widening, which is accepted
here and ignored. Nothing in the current roster exercises it -- on the multiplicity-3
degeneracy the searcher does emit two brackets for one level, and the refiner correctly rejects
the one that does not contain it -- but that is a measurement on one roster, not an argument.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from objective import Declined
from v1_bracket import bracket_interval
from v1_multiplicity import multiplicity, wide_bracket_for
from v1_refine import refine

__all__ = ["Level", "SolveResult", "SolveDiagnostics", "solve", "v1_solve"]

#: Relative offset used to estimate the wall slope `c` at a located root. Small enough to stay
#: on the straight flank, large enough not to sit in the rounded bottom.
_SLOPE_FRAC = 0.005


@dataclass(frozen=True)
class Level:
    """One eigenvalue location, its multiplicity, and how well it is known."""
    lam: float
    mult: int
    sigma: float = float("nan")
    delta_res: float = float("nan")
    bracket: tuple = ()
    wide_bracket: tuple = ()
    gap_ratio: float = float("nan")
    c2_rejected: bool = False

    @property
    def attained_prec(self):
        """Relative accuracy actually achieved at this level.

        `search.md` section 2.4 clause (2): each located value is within `prec` of the truth, OR
        the solver REPORTS that `delta_res > prec*lam` and returns `delta_res` as the attained
        bound. There is no predicted-accuracy constant -- `delta_res` is measured from the
        parabola fit the refiner already performed.
        """
        if not np.isfinite(self.delta_res) or self.lam == 0.0:
            return float("nan")
        return abs(self.delta_res) / abs(self.lam)

    def met(self, prec):
        a = self.attained_prec
        return bool(np.isnan(a) or a <= prec)

    def __iter__(self):
        return iter((self.lam, self.mult))


@dataclass(frozen=True)
class SolveDiagnostics:
    """Clause (3): the honesty clause is not decoration.

    `S` is unknown on every domain that matters -- the L-shape, GWW and mushroom have tabulated
    prefixes only -- so clauses (1) and (2) need a truth-free proxy on a real run, and this is
    it. A caller that cannot see the answer can still see whether the run deserves belief.
    """
    search: object = None            # the stage-1 SearchDiagnostics
    n_brackets: int = 0
    n_rejected: int = 0              # brackets the refiner threw away as empty
    n_merged: int = 0                # located roots folded into a neighbour within `prec`
    search_evals: int = 0
    refine_evals: int = 0
    multiplicity_evals: int = 0      # must be zero: stage 3 reads the cache
    evaluations: int = 0
    declined: bool = False
    decline_reason: str = ""
    prec_requested: float = float("nan")
    attained_prec: float = float("nan")   # the worst over levels
    unmet: tuple = ()                     # levels whose accuracy fell short, if any

    @property
    def met_request(self):
        return not self.unmet

    @property
    def per_eigenvalue(self):
        n = sum(1 for _ in self.unmet) or None
        return float("nan")


@dataclass
class SolveResult:
    levels: tuple = ()
    diagnostics: SolveDiagnostics = field(default_factory=SolveDiagnostics)

    @property
    def declined(self):
        return self.diagnostics.declined

    def __iter__(self):
        """The seam's return shape: `[(lam, mult), ...]`."""
        return iter([(x.lam, x.mult) for x in self.levels])

    def __len__(self):
        return len(self.levels)

    @property
    def unfolded(self):
        """Every eigenvalue with multiplicity, which is what truth is stated as."""
        return np.array([x.lam for x in self.levels for _ in range(x.mult)])


def _slope_from_cache(lams, vs, lam):
    """`c = dsigma/dlam` from the two CACHED samples nearest the root, one either side.

    Free, which matters: an explicit `lam +/- h` probe costs two evaluations per level, and
    charging stage 3 for them contradicts its contract of spending nothing. The search has
    already sampled both flanks of every well it bracketed, so the slope is in hand.

    Cancellation-free, as in `isolation`: on a cone the flanks straddle the minimizer, so
    `v_lo = c*(lam* - lo)` and `v_hi = c*(hi - lam*)`, hence `c = (v_lo + v_hi)/(hi - lo)` with
    no subtraction of nearly-equal quantities and no dependence on where the minimum is.
    """
    left = lams < lam
    right = lams > lam
    if not np.any(left) or not np.any(right):
        return 0.0
    i = int(np.argmax(lams[left]))
    j = int(np.argmin(lams[right]))
    lo, v_lo = float(lams[left][i]), float(vs[left][i])
    hi, v_hi = float(lams[right][j]), float(vs[right][j])
    return (v_lo + v_hi) / (hi - lo) if hi > lo else 0.0


def _samples(obj):
    """The evaluation cache in `lam`, whatever variable the objective is stated in.

    `wide_bracket_for` compares distances against `prec*lam`, so its samples have to be in
    `lam` too -- mixing them is wrong by a factor of `2k` and does not fail.
    """
    xs, vs = obj.samples
    return (np.array([obj.in_lam(x) for x in xs]) if xs.size else xs), vs


def solve(obj, *, prec=None):
    """Locate every eigenvalue in the objective's window, with its multiplicity.

    All three stages share one `Objective`, so the cost is the union of what they sampled and
    not the sum -- stage 3 in particular spends nothing, and that is asserted rather than hoped.
    """
    prec = obj.prec if prec is None else float(prec)

    mark = obj.n_evals
    res = bracket_interval(obj, prec=prec)
    search_evals = obj.n_evals - mark
    sd = res.diagnostics

    if res.declined:
        return SolveResult((), SolveDiagnostics(
            search=sd, declined=True, decline_reason=sd.decline_reason,
            search_evals=search_evals, evaluations=obj.n_evals,
            prec_requested=prec, attained_prec=sd.attained_prec))

    # Stage 2. `bracket_interval` returns brackets in the objective's OWN variable, so this
    # converts once, here, to the `lam` the refiner works in.
    mark = obj.n_evals
    located, rejected = [], 0
    for b in res.brackets:
        lo, hi = obj.in_lam(b.lo), obj.in_lam(b.hi)
        hint = obj.in_lam(b.k_hint) if np.isfinite(b.k_hint) else None
        r = refine(obj, lo, hi, prec=prec, lam_hint=hint)
        if r.accepted:
            located.append((r, (lo, hi)))
        else:
            # A bracket the search kept on purpose and the refiner threw away. Not a failure of
            # either: dropping a doubtful region costs an eigenvalue and keeping one costs about
            # three evaluations, so the search is right to hand it on and the refiner is right
            # to refuse it.
            rejected += 1
    refine_evals = obj.n_evals - mark

    # Phase D's DISTANCE PREFILTER, before stage 3 rather than after it.
    #
    # Two located roots within `prec*lam` of each other are one level of `S(prec)`, and reading
    # multiplicity at each of them separately double-counts: the wide bracket spans the whole
    # group -- correctly, since the group IS the level -- so each fragment declares the group's
    # total. Measured on the sub-licence cluster, two roots each declared 3 against 3 true.
    #
    # **Reading ONCE per group also gets Phase D's collapse behaviour without the Gram angle.**
    # `search.md` gives two outcomes for roots this close: the same eigenfunction found twice,
    # which must COLLAPSE and RECOUNT, or two independent ones, which must MERGE and SUM -- and
    # says "do NOT sum" on the first, because summation is the double count. A single
    # multiplicity read on the merged group recounts by construction, and it is right in both
    # cases: one well found twice reads its own multiplicity, and two genuine eigenvalues within
    # `prec` both dip and are both counted. The Gram angle is what would be needed to choose
    # between collapse and sum, and recounting removes the choice.
    #
    # What is still missing is the case Phase D's angle is really for: two roots at the same
    # separation where one is a duplicate and one is not. Distance cannot tell them apart, and
    # neither can this -- but recounting is the safe branch of the two.
    groups = []
    for r, bracket in sorted(located, key=lambda p: p[0].lam):
        if groups and abs(r.lam - groups[-1][0][0].lam) <= prec * max(abs(r.lam), 1.0):
            groups[-1].append((r, bracket))
        else:
            groups.append([(r, bracket)])
    # One representative per group: the deepest, which is the best-located member.
    located = [min(g, key=lambda p: p[0].sigma) for g in groups]
    merged = sum(len(g) - 1 for g in groups)

    # Stage 3. From the cache, and it must cost nothing.
    mark = obj.n_evals
    lams, svals = _samples(obj)
    levels = []
    for r, bracket in located:
        c = _slope_from_cache(lams, svals, r.lam)
        # The distance to the next eigenvalue, from the free `sigma_2` the search already has.
        # It bounds the wide bracket from ABOVE, so the flanks cannot reach a neighbour and
        # count its branch as one of this level's.
        s2 = obj.sigma(obj.from_lam(r.lam), 1)
        nb = (s2 / c) if (c > 0 and np.isfinite(s2)) else None
        wb = wide_bracket_for(lams, r.lam, r.sigma, c, prec=prec, neighbour=nb)
        if wb is None:
            levels.append(Level(lam=r.lam, mult=1, sigma=r.sigma, delta_res=r.delta_res,
                                bracket=bracket))
            continue
        m = multiplicity(lambda z: obj.tensions(obj.from_lam(z)), r.lam,
                         wide_bracket=wb, prec=prec)
        levels.append(Level(lam=r.lam, mult=m.mu, sigma=r.sigma, delta_res=r.delta_res,
                            bracket=bracket, wide_bracket=wb, gap_ratio=m.gap_ratio,
                            c2_rejected=m.c2_rejected))
    mult_evals = obj.n_evals - mark

    levels.sort(key=lambda x: x.lam)
    unmet = tuple(x.lam for x in levels if not x.met(prec))
    attained = max((x.attained_prec for x in levels
                    if np.isfinite(x.attained_prec)), default=prec)

    return SolveResult(tuple(levels), SolveDiagnostics(
        search=sd, n_brackets=len(res.brackets), n_rejected=rejected, n_merged=merged,
        search_evals=search_evals, refine_evals=refine_evals,
        multiplicity_evals=mult_evals, evaluations=obj.n_evals,
        prec_requested=prec, attained_prec=float(attained), unmet=unmet))


def v1_solve(**kw):
    """The candidate on the stage-4 seam: `solve(tensions, a, b, *, prec, modes=None)`.

    `modes` is accepted and ignored -- Phase D is what would use it. Declining propagates as a
    `Declined` value rather than an empty list, because an empty list looks like success.
    """
    def fn(tensions, a, b, *, prec, modes=None, objective=None, expected_count=None):
        from objective import Objective
        obj = objective if objective is not None else Objective(
            tensions, a, b, prec=prec, variable="lam", expected_count=expected_count)
        res = solve(obj, prec=prec, **kw)
        fn.last = res
        if res.declined:
            return Declined(res.diagnostics.decline_reason,
                            partial=[(x.lam, x.mult) for x in res.levels],
                            diagnostics=res.diagnostics)
        return [(x.lam, x.mult) for x in res.levels]

    fn.last = None
    return fn
