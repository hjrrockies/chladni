"""Stage 4: the composed pipeline, and the only seam a candidate MUST present.

Promoted from `tests/v1_solve.py`, whose remaining `v1_solve` factory now scores THIS module on
the seam. `docs/design/candidates.md` section 1 calls `solve` "the only mandatory seam": a
method that does not decompose into bracket -> refine -> multiplicity implements this and
nothing else, so it is where v1 and a monolithic candidate are compared on equal terms.

`MPSEigensolver.solve_interval` is the intended caller and does not exist yet; it is the one
piece of this layer still to be built.

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

from .bracket import bracket_interval
from .multiplicity import WIDE_FACTOR, multiplicity, wide_bracket_for
from .refine import refine

__all__ = ["Level", "SolveResult", "SolveDiagnostics", "solve"]

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
    #: True when this level is the COLLAPSE of a neighbourhood whose internal grouping was
    #: arbitrary. The count is trustworthy; the sub-structure is coarser than `prec` asked for,
    #: so the level is deliberately wider than the merge licence. See `_collapse_ambiguous`.
    grouping_ambiguous: bool = False
    #: The spacings inside that neighbourhood, so a caller can choose a `prec` that resolves or
    #: merges it rather than re-solving blind.
    spacings: tuple = ()

    @property
    def width(self):
        """The interval this level stands for, 0 for a single location.

        For a collapsed level it is the WIDE BRACKET that was counted over, not the span
        between the roots that were folded together. The roots understate it -- two roots were
        folded where three eigenvalues lie, so their span was 0.9 licence units against a true
        1.6 -- and a level must not declare less than it covers, or a scorer holding it to its
        declaration would mark it impure for eigenvalues it correctly counted.

        It over-claims in the other direction: the counted branches lie somewhere inside the
        bracket and not necessarily out to its edges. That is the honest direction for a level
        whose whole point is that its sub-structure is unresolved, and it is not exploitable --
        a method that declared a huge span to buy purity would show up immediately in
        `separation` and `in_lattice`, which count levels rather than measure them.
        """
        if not self.grouping_ambiguous or len(self.wide_bracket) != 2:
            return 0.0
        return float(self.wide_bracket[1] - self.wide_bracket[0])

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
        """Clause (2) is an OR -- within `prec`, or the bound reached is REPORTED. A level with
        no finite bound has done neither, so it is unmet and lands in `diagnostics.unmet`."""
        a = self.attained_prec
        return bool(np.isfinite(a) and a <= prec)

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
    n_ambiguous: int = 0             # levels collapsed because their grouping was arbitrary
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


def _spanning_bracket(lams, roots, reach):
    """Cached samples at least `reach` outside the whole span of `roots`, or `None`."""
    lo_t, hi_t = min(roots) - reach, max(roots) + reach
    left, right = lams[lams <= lo_t], lams[lams >= hi_t]
    if left.size == 0 or right.size == 0:
        return None
    return float(left.max()), float(right.min())


def _collapse_ambiguous(obj, lams, levels, prec, reach_factor):
    """Cross-check each neighbourhood's total, and collapse the ones whose grouping is arbitrary.

    # The failure this exists for

    Where several eigenvalues sit within about `prec*lam` of each other, the way they group into
    LEVELS can be genuinely undetermined -- and then so are the individual counts, while the
    TOTAL is not. Three eigenvalues evenly spaced at 0, 0.8 and 1.6 in units of the merge
    licence cannot all be one level (the span is 1.6 > 1), and either pairing is legal, so the
    split is arbitrary. Worse, a count taken at the middle one must look 0.8 to see its partner
    and therefore also sees the stranger at 0.8 on the other side: **no radius includes one and
    excludes the other**, measured exactly equidistant.

    So the counts came out as 3 and 1, totalling **4 where 3 exist** -- and at a span of exactly
    one licence, 3 and 3, totalling 6. For a shape optimization that is the worst kind of error:
    a wrong total shifts the INDEXING, so `lam_3` becomes a different eigenvalue, which is how
    a solver comes to report an accurate but non-consecutive spectrum.

    # The test is a consistency check, not a threshold

    Chain the located roots into neighbourhoods, consecutive members within the counting reach.
    For each, compare the SUM of the individual counts against ONE count taken over a bracket
    spanning the whole neighbourhood:

        they agree      the fine grouping is self-consistent -- keep it
        they disagree   the grouping is arbitrary -- collapse to one level and use the
                        spanning count, which is the quantity that was never in doubt

    Measured, in units of the licence, on the multiplicity-3 cluster at `50*pi^2`:

        span   individual counts   spanning count   action     total
        1.0         3 + 3                3          collapse   3  (was 6)
        1.6         3 + 1                3          collapse   3  (was 4)
        2.0         3 + 1                3          collapse   3  (was 4)
        2.6         1 + 1 + 1            3          keep       3
        6.0         1 + 1 + 1            3          keep       3

    The last two rows are why this is a cross-check and not a merge radius: at spans of 2.6 and
    6.0 the roots are still within reach of one another, so a radius rule would collapse them
    and throw away a correct fine answer. Their counts agree with the total, so they stand.

    # What it deliberately gives up

    A collapsed level is WIDER than `prec*lam`, so it fails purity on purpose. That is the
    trade: a level that is too wide but correctly counted beats two levels correctly narrow and
    wrongly counted, because only the second corrupts an index. It is flagged rather than
    silent, and it carries the spacings, so a caller can escape the regime in either direction
    -- more precision splits the neighbourhood, less merges it, and both give a correct total.
    """
    if len(levels) < 2:
        return levels, 0

    reach = [reach_factor * max(prec * abs(x.lam), 4.0 * np.spacing(abs(x.lam)))
             for x in levels]
    groups, cur = [], [0]
    for i in range(1, len(levels)):
        if levels[i].lam - levels[i - 1].lam <= max(reach[i], reach[i - 1]):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    out, collapsed = [], 0
    for g in groups:
        if len(g) == 1:
            out.append(levels[g[0]])
            continue
        members = [levels[i] for i in g]
        deepest = min(members, key=lambda x: x.sigma)
        wb = _spanning_bracket(lams, [x.lam for x in members],
                               max(reach[i] for i in g))
        if wb is None:
            out.extend(members)
            continue
        total = multiplicity(lambda z: obj.tensions(obj.from_lam(z)), deepest.lam,
                             wide_bracket=wb, prec=prec).mu
        if total == sum(x.mult for x in members):
            out.extend(members)             # self-consistent: the fine grouping stands
            continue
        gaps = tuple(float(b.lam - a.lam) for a, b in zip(members[:-1], members[1:]))
        out.append(Level(lam=deepest.lam, mult=total, sigma=deepest.sigma,
                         delta_res=deepest.delta_res, bracket=deepest.bracket,
                         wide_bracket=wb, gap_ratio=deepest.gap_ratio,
                         grouping_ambiguous=True, spacings=gaps))
        collapsed += 1
    return out, collapsed


def _samples(obj):
    """The evaluation cache in `lam`, whatever variable the objective is stated in.

    `wide_bracket_for` compares distances against `prec*lam`, so its samples have to be in
    `lam` too -- mixing them is wrong by a factor of `2k` and does not fail.
    """
    xs, vs = obj.samples
    return (np.array([obj.in_lam(x) for x in xs]) if xs.size else xs), vs


def solve(obj, *, prec=None, collapse=True):
    """Locate every eigenvalue in the objective's window, with its multiplicity.

    All three stages share one `Objective`, so the cost is the union of what they sampled and
    not the sum -- stage 3 in particular spends nothing, and that is asserted rather than hoped.

    `collapse=False` disables the grouping cross-check, which is an ABLATION rather than an
    option: without it the total is wrong wherever a neighbourhood's grouping is arbitrary --
    4 or 6 declared where 3 exist -- and a wrong total moves an index. It exists so the guard
    can be shown to be load-bearing, and so a figure can state the before as well as the after.
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
        lo, hi = obj.in_lam_bracket(b.lo, b.hi)
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

    levels.sort(key=lambda x: x.lam)

    # The grouping cross-check. Reads only cached points, so it is charged to stage 3's window
    # and must leave it at zero.
    if collapse:
        levels, n_amb = _collapse_ambiguous(obj, lams, levels, prec, WIDE_FACTOR)
    else:
        n_amb = 0
    mult_evals = obj.n_evals - mark

    unmet = tuple(x.lam for x in levels if not x.met(prec))
    attained = max((x.attained_prec for x in levels
                    if np.isfinite(x.attained_prec)), default=prec)

    return SolveResult(tuple(levels), SolveDiagnostics(
        search=sd, n_brackets=len(res.brackets), n_rejected=rejected, n_merged=merged,
        search_evals=search_evals, refine_evals=refine_evals,
        multiplicity_evals=mult_evals, evaluations=obj.n_evals, n_ambiguous=n_amb,
        prec_requested=prec, attained_prec=float(attained), unmet=unmet))
