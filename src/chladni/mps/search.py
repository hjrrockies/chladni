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

from dataclasses import dataclass, field, replace

import numpy as np

from .bracket import bracket_interval
from .multiplicity import (BIAS_HI, BIAS_HI_CACHE, BIAS_LO, BIAS_LO_CACHE, COUNT_BELOW_CACHE,
                           MAX_BRANCHES, REJECT_ABOVE_CACHE, SIDE_RATIO, SLOPE_STANDOFF,
                           WIDE_FACTOR, multiplicity, wide_bracket_for)
from .refine import refine

__all__ = ["Level", "SolveResult", "SolveDiagnostics", "solve"]

#: Relative offset used to estimate the wall slope `c` at a located root. Small enough to stay
#: on the straight flank, large enough not to sit in the rounded bottom.
_SLOPE_FRAC = 0.005

#: A located root is a HUMP -- the local maximum between two members' wells -- when sigma_1, probed
#: three quarters of the way toward the nearest root on EACH side, reads below this fraction of its
#: own sigma both ways. On a hump the probe sits a quarter of the distance up the neighbour's wall
#: while the hump sits about the whole distance up it, so both read near 1/4; a genuine root, even
#: one located coarsely, climbs its own wall toward at least one neighbour.
_HUMP_RATIO = 0.5

#: Refine calls the side search may spend on one root, across both sides together.
#:
#: A bound on COST, never a decision. The read examines at most `MAX_BRANCHES` branches, so at
#: most `MAX_BRANCHES - 1` partners can exist for it to find, and a cap at that number cannot
#: change an answer -- it can only stop a pathology from spinning. The previous structure had an
#: implicit cap of one PER SIDE, which is what left a member unfound wherever a side held two
#: (170*pi^2, band 0.3, prec 1e-12: 7 of 8).
_SIDE_SEARCH_MAX = MAX_BRANCHES

#: Fraction of the remaining sub-bracket the side search stands off from a find before searching
#: on past it.
#:
#: **`(sigma/c)/bias_lo` alone is the wrong radius for this job**, though it is the right one for
#: the dedup it was borrowed from. At a CONVERGED find `sigma` is the tension FLOOR, not a
#: distance to anything, so the radius collapses to `eps/c` and the residual begins on the find's
#: own wall -- where the wall is the lowest point in the interval, so the refiner returns that
#: endpoint and rejects it `not_a_minimum`. Measured at 50*pi^2, band 0.3, prec 1e-12: the find at
#: 493.480220054314 reads sigma 1.13e-14, the residual starts 1e-15 later at 493.480220054315, and
#: the member at 493.480220054462 -- 1.5e-10 further in, sigma 3.7e-15 -- is never located.
#:
#: Swept on that residual, as a fraction of the span from the find to the sub-bracket's outer end:
#:
#:     fraction   0.00   0.02   0.05   0.10   0.20   0.30   0.50
#:     found        no     no     no    yes    yes    yes    yes
#:
#: A fraction of the remaining span rather than an absolute distance, for the reason `STRADDLE`
#: is one: it is dimensionless, and the scale that matters is the interval still to be searched.
#: The floor stays, so a find that is NOT converged still excludes its own resolution.
_SIDE_STANDOFF = 0.1


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
    #: Accepted, refined minima folded into this level. More than `mult` is the solver
    #: contradicting its own locator, and `_flag_uncertain_counts` says so.
    n_roots: int = 1
    #: True when two readings of this level's multiplicity disagree. The count is NOT certified:
    #: `count_alt` is the other reading and `uncertain_reason` says which two disagreed.
    count_uncertain: bool = False
    count_alt: int = 0
    uncertain_reason: str = ""
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
    n_uncertain: int = 0             # levels whose multiplicity is not certified
    uncertain: tuple = ()            # (lam, mult, count_alt, reason) per uncertain level
    search_evals: int = 0
    refine_evals: int = 0
    #: The stage-3 READ, which must be zero: it answers from the cache and nothing else. It was
    #: previously the whole stage-3 window, so it silently included the two side probes and the
    #: side search below -- and read zero only on cells where the count never declined, i.e.
    #: exactly the cells that do none of that work. Split out so the contract is asserted where
    #: it could fail rather than where it cannot.
    multiplicity_evals: int = 0
    #: The reach rule's SIDE PROBES: two evaluations per branch that lands in the undecided band,
    #: and none at all on a simple bracket.
    probe_evals: int = 0
    #: The side search's refinements, when a count declines and the partner is LOCATED instead.
    side_evals: int = 0
    evaluations: int = 0
    declined: bool = False
    decline_reason: str = ""
    prec_requested: float = float("nan")
    attained_prec: float = float("nan")   # the worst over levels
    unmet: tuple = ()                     # levels whose accuracy fell short, if any

    @property
    def met_request(self):
        """Did every level reach the requested PRECISION? Says nothing about the counts --
        `certified_counts` is that question, and both were True in every wrong cell of
        `tests/test_cluster_totals.py` because only this one was being asked."""
        return not self.unmet

    @property
    def certified_counts(self):
        """Is every level's MULTIPLICITY certified? The other half of the honesty clause."""
        return self.n_uncertain == 0

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


def _slope_outside(lams, vs, lam, min_dist):
    """`c` from the nearest cached samples at least `min_dist` away on each side, or `None`.

    The wall slope is a property of the OUTER cone and has to be measured there. Inside a bracket
    holding more than one eigenvalue the other members depress `sigma_1` at the flanks, so `c`
    reads too small and every distance `sigma_j/c` too large -- an eigenvalue inside the bracket
    then reads as outside and the count goes under. Measured on the wip branch on a pair 0.3
    licences apart: `d/reach` inflated from a true 0.6 to 1.156.

    Cancellation-free as `_slope_from_cache` is. `None` when the cache has nothing far enough
    out, which a caller must read as "cannot estimate", never as zero.
    """
    left = lams < lam - min_dist
    right = lams > lam + min_dist
    if not np.any(left) or not np.any(right):
        return None
    i, j = int(np.argmax(lams[left])), int(np.argmin(lams[right]))
    lo, hi = float(lams[left][i]), float(lams[right][j])
    v_lo, v_hi = float(vs[left][i]), float(vs[right][j])
    return (v_lo + v_hi) / (hi - lo) if hi > lo else None


def _claim(branches, lam_root, claimants, bias_lo, bias_hi):
    """Which branches the read counted are ALREADY ACCOUNTED FOR by a located level.

    Returns `(n_claimed, claimed)`, `claimed` the indices claimed, one per claimant at most.

    # The defect this exists for

    `mu` says how many eigenvalues the reach rule believes are in the bracket. It does not say
    WHICH, so when the side search then locates one of them as a level of its own, nothing
    compares the two and the member is reported twice -- `170*pi^2`, band 0.3, prec 1e-8: 9 of 8,
    and CERTIFIED, because a successful find also cleared the doubt. The mirror case is 7 of 8 at
    1e-12, where a side held two members and only one was ever looked for.

    `TODO.md` recorded two measured dead ends, and both are avoided here by construction rather
    than by tuning. Searching on after a find double-counted members the read had counted inside
    (4 of 3 at `50*pi^2`, 1e-12, band 0.3) -- safe now, because a second find CLAIMS a branch
    instead of adding to the total. Capping the root's count at what the search could not reach
    lost a member the search then failed to find (2 of 3 at 1e-6) -- not done here, because only
    a branch the read actually counted can be claimed, so nothing the read did not put in can be
    taken out.

    # The predicate, and why it needs no new constant

    `d_est = beta * d_true` with `beta` in `[bias_lo, bias_hi]`, so the true distance to a
    branch's eigenvalue lies in `[d_est/bias_hi, d_est/bias_lo]`. A claimant whose offset from
    the root falls in that interval, on a side the branch does not contradict, IS that
    eigenvalue. This is the interval `reach_multiplicity` itself decides with, and the one the
    side search already uses one-sided as `reach_find`; attribution uses the instrument the count
    used, so there is nothing new to calibrate. **The bias pair must be the one passed to the
    read that produced these branches** -- `solve` selects between two calibrated sets -- which is
    why it is an argument rather than an import.

    Branch 0 is never claimable: it is the level's own eigenvalue, the reason the level exists.
    Without that exclusion a claimant a few ulps out could claim the root's own existence. With
    it, `mu - n_claimed >= 1` holds by construction, since `mu` is `1 + |counted j>=1|`.

    # Matching, deliberately greedy

    With the cache bias pair the admissible band spans 3.6x, so one claimant can admit two
    branches and one branch two claimants. Claimants are taken nearest-first, each keeping the
    unclaimed admissible branch whose estimate best matches its offset in the LOG -- scale-free,
    so a 0.45x and a 1.64x match are compared fairly -- ties by lowest index. An optimal
    assignment is over-engineering: the read examines at most `MAX_BRANCHES` branches and no
    bracket on any measured grid held more than four members.
    """
    usable = [b for b in branches if b.verdict == "counted" and b.index >= 1 and b.d_est > 0]
    if not usable:
        return 0, ()
    claimed, taken = [], set()
    for lam_q in sorted(claimants, key=lambda q: abs(q - lam_root)):
        off = abs(lam_q - lam_root)
        if not off > 0:
            continue
        side = 1 if lam_q > lam_root else -1
        best, best_key = None, None
        for b in usable:
            if b.index in taken:
                continue
            if b.side != 0 and b.side != side:
                continue
            if not (b.d_est / bias_hi <= off <= b.d_est / bias_lo):
                continue
            key = (abs(np.log(b.d_est / off)), b.index)
            if best_key is None or key < best_key:
                best, best_key = b, key
        if best is not None:
            taken.add(best.index)
            claimed.append(best.index)
    return len(claimed), tuple(claimed)


def _spanning_bracket(lams, roots, reach):
    """Cached samples at least `reach` outside the whole span of `roots`, or `None`."""
    lo_t, hi_t = min(roots) - reach, max(roots) + reach
    left, right = lams[lams <= lo_t], lams[lams >= hi_t]
    if left.size == 0 or right.size == 0:
        return None
    return float(left.max()), float(right.min())


def _collapse_ambiguous(obj, lams, levels, prec, reach_factor, rule="gap"):
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
        # The spanning read keeps the rule this component was designed with when the caller asks
        # for `reach`: `reach` compares a distance against the bracket that was EMITTED, and this
        # reads over one the cross-check synthesised -- a different interval, so a different
        # question. The wip seam twice failed to thread `rule` here at all; it is threaded, and
        # this is the one deliberate exception.
        total = multiplicity(lambda z: obj.tensions(obj.from_lam(z)), deepest.lam,
                             wide_bracket=wb, prec=prec,
                             rule="gap" if rule == "reach" else rule).mu
        if total == sum(x.mult for x in members):
            out.extend(members)             # self-consistent: the fine grouping stands
            continue
        gaps = tuple(float(b.lam - a.lam) for a, b in zip(members[:-1], members[1:]))
        fine = sum(x.mult for x in members)
        # The disagreement is REPORTED, and the value does not move. The spanning count is not
        # the safe reading -- on the `65*pi^2` cluster at bands 1.3 to 2.0 the fine grouping is
        # the truth and the spanning read folds four levels into two -- but `fine` is not safe
        # either: 6 and 4 against 3 on the evenly spaced triple. No choice between them is right
        # on both families, so the level is flagged and carries the other reading.
        out.append(Level(lam=deepest.lam, mult=total, sigma=deepest.sigma,
                         delta_res=deepest.delta_res, bracket=deepest.bracket,
                         wide_bracket=wb, gap_ratio=deepest.gap_ratio,
                         n_roots=sum(x.n_roots for x in members),
                         count_uncertain=True, count_alt=min(total, fine),
                         uncertain_reason=(
                             f"a neighbourhood of {len(members)} located levels: the individual "
                             f"counts sum to {fine} and one count over the whole neighbourhood "
                             f"reads {total}. The grouping is arbitrary at this prec and the two "
                             f"readings disagree, so neither is certified"),
                         grouping_ambiguous=True, spacings=gaps))
        collapsed += 1
    return out, collapsed


def _flag_uncertain_counts(levels, folds_are_duplicates=False):
    """Mark every level whose multiplicity two independent readings disagree about.

    **It decides nothing; it refuses to.** Recounting at one representative under-reads two
    genuine eigenvalues inside the merge licence, summing over-counts one eigenvalue found twice,
    and distance cannot tell those apart -- that is Phase D's Gram angle, not a rule. Until it
    exists the honest output is a level that says its count is in doubt.

        mult < n_roots   the level declares fewer eigenvalues than the ACCEPTED, refined,
                         certificate-carrying minima folded into it

    Every such root passed `refine._accept`, so each is positive evidence of a minimum. It is not
    proof of error -- two roots can be one eigenvalue found twice -- which is why the level is
    flagged and not corrected: raising the count would trade a silent under-count for a silent
    over-count.

    Levels `_collapse_ambiguous` already flagged pass through and are COUNTED. Skipping them made
    the branch that detects a disagreement by construction the one whose levels never reached
    `n_uncertain` -- 14% recall on the wip branch, against 93% once they were counted.

    `folds_are_duplicates` is `solve`'s attained merge under the reach rule, and there the check
    has nothing left to catch. Attained merge folds only roots that agree to within the accuracy
    they reached, so a fold IS the same point found twice; and whether a second eigenvalue sits
    that close is the reach rule's own question -- a partner within attained accuracy is at the
    floor with `sigma_2` and counted directly. Measured at 50*pi^2, prec 1e-14, band 0.8: two
    brackets refined to one member 4 ulps apart, the count was right, and the flag made
    `solve_interval` raise on it.
    """
    out, flagged = [], []
    for x in levels:
        if x.count_uncertain:
            out.append(x)
            flagged.append(x)
            continue
        if folds_are_duplicates or x.mult >= x.n_roots:
            out.append(x)
            continue
        y = replace(x, count_uncertain=True, count_alt=x.n_roots,
                    uncertain_reason=(
                        f"{x.n_roots} accepted minima were folded into this level and the "
                        f"multiplicity read at the deepest of them is {x.mult}: either "
                        f"{x.n_roots} eigenvalues within prec*lam of each other, or one located "
                        f"{x.n_roots} times, and distance cannot tell those apart"))
        out.append(y)
        flagged.append(y)
    return out, flagged


def _samples(obj):
    """The evaluation cache in `lam`, whatever variable the objective is stated in.

    `wide_bracket_for` compares distances against `prec*lam`, so its samples have to be in
    `lam` too -- mixing them is wrong by a factor of `2k` and does not fail.
    """
    xs, vs = obj.samples
    return (np.array([obj.in_lam(x) for x in xs]) if xs.size else xs), vs


def solve(obj, *, prec=None, collapse=False, merge="attained", rule="reach",
          reach_slope="cache"):
    """Locate every eigenvalue in the objective's window, with its multiplicity.

    All three stages share one `Objective`, so the cost is the union of what they sampled and
    not the sum -- stage 3 in particular spends nothing, and that is asserted rather than hoped.

    **The defaults are attained merge, the reach rule on the cache slope, and no cross-check.**
    Promoted on held-out cluster grids against the previous defaults (licence merge, the gap
    rule, the cross-check on):

        grid                        these defaults                 previous defaults
        145*pi^2 (held out, 45)     24 right, 0 wrong, 21 declined  11 right, 13 wrong, 21 declined
        125*pi^2 (45)               45 right (1 flagged), 0 wrong   26 right, 19 wrong (1 silent)
        50/65*pi^2 (110)            103 right, 2 basis-excused      72 right, 33 wrong (1 silent)
        precision ladder            every cell acceptable           2 wrong totals at 1e-4

    The previous configuration stays reachable, as the ABLATION that shows what this replaced.

    `merge` decides which located roots are folded into one level: "attained" folds only roots
    that agree to within the accuracy the refinement actually reached, capped at the licence --
    the search never merges deliberately, only what it could not resolve; "licence" folds every
    pair within `prec*lam`. `rule` picks the multiplicity count: "reach" the number of eigenvalues
    INSIDE the emitted bracket, "gap" (C1)+(C2) over a wide bracket, "floor" coincidence with
    `sigma_1`. Attained merge must be paired with "reach": under "gap" each root of a resolved
    pair reads the pair from its own wide bracket and the total double-counts.

    `collapse=True` restores the grouping cross-check, `_collapse_ambiguous`, which recounts over
    a synthesised bracket of radius `WIDE_FACTOR`. Its radius is what placed the failing band of
    wrong totals at 1-8 merge licences, so it is off by default.

    `reach_slope` is where "reach" reads the wall slope `c` in `sigma_j/c`: "cache", the nearest
    cached samples either side of the root, with bounds recalibrated on a held-out set; or
    "outside", three bracket widths out, what the rule was first calibrated on. On a cluster wider
    than the standoff the outside samples sit in other members' wells and `c` reads low --
    measured 1.86x on the distance where "cache" read 1.02.
    """
    if merge not in ("licence", "attained"):
        raise ValueError(f"merge must be 'licence' or 'attained', got {merge!r}")
    if rule not in ("gap", "floor", "reach"):
        raise ValueError(f"rule must be 'gap', 'floor' or 'reach', got {rule!r}")
    if reach_slope not in ("outside", "cache"):
        raise ValueError(f"reach_slope must be 'outside' or 'cache', got {reach_slope!r}")
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
        r = refine(obj, lo, hi, prec=prec, lam_hint=hint,
                   interior=(obj.in_lam(b.lo), obj.in_lam(b.hi)))
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
    # neither can this.
    #
    # **"Recounting is the safe branch of the two" was written here, and it is false.** It is
    # right for a duplicate and wrong for two genuine eigenvalues inside the licence: the recount
    # reads at ONE representative, the partner's branch is minimal at the partner, so (C2)
    # refuses it and the level reads 1 where it is 2. So the number of roots folded is carried
    # on the level, and `_flag_uncertain_counts` flags the contradiction.
    # Compared against the group's FIRST member, never its predecessor: `S(prec)` groups by
    # diameter, and chaining is a mistake `bracket_search/SUCCESS.md` records as made twice.
    def _same_point(first, r):
        licence = prec * max(abs(r.lam), 1.0)
        if merge == "licence":
            return abs(r.lam - first.lam) <= licence
        tol = max(first.delta_res if np.isfinite(first.delta_res) else 0.0,
                  r.delta_res if np.isfinite(r.delta_res) else 0.0,
                  4.0 * np.spacing(abs(r.lam)))
        return abs(r.lam - first.lam) <= min(tol, licence)

    groups = []
    for r, bracket in sorted(located, key=lambda p: p[0].lam):
        if groups and _same_point(groups[-1][0][0], r):
            groups[-1].append((r, bracket))
        else:
            groups.append([(r, bracket)])
    # One representative per group: the deepest, which is the best-located member.
    located = [(min(g, key=lambda p: p[0].sigma), len(g)) for g in groups]
    merged = sum(len(g) - 1 for g in groups)

    # Stage 3. From the cache, and it must cost nothing.
    mark = obj.n_evals
    lams, svals = _samples(obj)
    levels = []
    probe_evals = side_evals = 0
    for (r, bracket), n_roots in located:
        c = _slope_from_cache(lams, svals, r.lam)
        # The distance to the next eigenvalue, from the free `sigma_2` the search already has.
        # It bounds the wide bracket from ABOVE, so the flanks cannot reach a neighbour and
        # count its branch as one of this level's.
        s2 = obj.sigma(obj.from_lam(r.lam), 1)
        nb = (s2 / c) if (c > 0 and np.isfinite(s2)) else None
        wb = wide_bracket_for(lams, r.lam, r.sigma, c, prec=prec, neighbour=nb)
        tens = (lambda z: obj.tensions(obj.from_lam(z)))
        extra, uncertain, drop = [], None, False
        if rule == "reach":
            # `mu*(B)` needs the EMITTED bracket and a slope from outside it; the wide bracket plays
            # no part in the count and is carried only as the record of what "gap" would have seen.
            width = abs(bracket[1] - bracket[0])
            c_out = (_slope_outside(lams, svals, r.lam, SLOPE_STANDOFF * width)
                     if reach_slope == "outside" else c)
            if c_out is None or not (c_out > 0):
                levels.append(Level(
                    lam=r.lam, mult=1, sigma=r.sigma, delta_res=r.delta_res, bracket=bracket,
                    wide_bracket=wb or (), n_roots=n_roots, count_uncertain=True,
                    uncertain_reason=f"the reach rule could not read a wall slope "
                                     f"({reach_slope}) from the cache at this root"))
                continue
            # Each slope carries the bounds measured for IT; the outside-slope set is the wip
            # branch's, the cache-slope set is recalibrated on a held-out set -- see
            # `multiplicity.COUNT_BELOW_CACHE`.
            bounds = (None if reach_slope == "outside" else
                      (COUNT_BELOW_CACHE, REJECT_ABOVE_CACHE, BIAS_LO_CACHE, BIAS_HI_CACHE))
            # The bias pair the READ was given, so attribution below inverts the same estimate the
            # count made. Hardcoding the cache pair would claim with the wrong band whenever
            # `reach_slope="outside"` -- the ablation path, whose bounds are the wip branch's.
            bias_lo_eff, bias_hi_eff = ((BIAS_LO, BIAS_HI) if bounds is None
                                        else (bounds[2], bounds[3]))
            m = multiplicity(tens, r.lam, bracket=bracket, c=c_out, prec=prec, rule="reach",
                             reach_bounds=bounds, counter=obj)
            probe_evals += int(m.evaluations)
            if m.ambiguous:
                # The rule could not place a branch inside the bracket or out. Every wrong total it
                # left on the cluster grid was this: one bracket holding two DISTINCT eigenvalues,
                # the partner at the bracket edge. Merging is permitted and never required, so the
                # partner need not be counted at all -- LOCATE it. Refine the bracket either side
                # of the root, outside half the branch's estimated distance so the sub-bracket's
                # inner end is off the root's own wall. A root found is its own level.
                #
                # A side is RESOLVED when it is searched and firmly rejected, or when it is too
                # short to hold the branch at all: the calibrated bias puts the eigenvalue at least
                # `BIAS_LO_CACHE * d` away, and a side shorter than `d/2` is shorter than that.
                # Requiring both sides SEARCHED kept a correct count flagged -- the uneven cluster
                # at 1e-8, band 0.3, where the declined branch belonged to a member 0.96 licences
                # out, located as its own level, with this bracket reaching 0.27 on that side. Only
                # an inconclusive search, or no distance to search at, leaves the level flagged.
                side_mark = obj.n_evals
                t_here = np.atleast_1d(np.asarray(tens(r.lam), dtype=float))
                j = min(int(m.mu), t_here.size - 1)
                half = 0.5 * float(t_here[j]) / c_out if j > 0 else float("inf")
                # Not `r` itself: each sub-bracket excludes `r +- half` by construction, so a root
                # found there is not `r`, and `r`'s own bar can be as wide as the partner's
                # distance -- 125*pi^2, prec 1e-10, band 0.5, bar 5.6e-8, partner 5.5e-8 away,
                # found and then discarded as "already located", leaving the total one short and
                # unflagged. A find that is really `r`'s own wall is a phantom at the inner end,
                # which the refiner's end probe rejects.
                taken = [p[0][0] for p in located if p[0][0] is not r]
                found, inconclusive = 0, not np.isfinite(half)
                # A WORK LIST per side, not one interval each. Refining each side exactly once
                # left a member unfound wherever a side held two: 170*pi^2, band 0.3, prec 1e-12,
                # 7 of 8, certified. Searching on after a find was measured and REVERTED once --
                # 4 of 3 at 50*pi^2, 1e-12, band 0.3 -- because a second find was added to a total
                # that had already counted it inside. Attribution is what makes it safe: below,
                # every find CLAIMS a branch rather than adding to `mu`.
                #
                # OUTWARD only. The residual on the inner side of a find is `r`'s own wall, and
                # refining there fits a parabola to rounding -- the phantom class the reach dedup
                # below records at 65*pi^2, prec 1e-10, band 1.0.
                sides = [(bracket[0], r.lam - half, -1), (r.lam + half, bracket[1], +1)]
                budget = _SIDE_SEARCH_MAX
                while sides and budget > 0:
                    a, b2, side = sides.pop(0)
                    if not (np.isfinite(half) and a < b2):
                        continue                    # too short to hold the branch: resolved
                    budget -= 1
                    r2 = refine(obj, a, b2, prec=prec, interior=(a, b2))
                    if not r2.accepted:
                        # A side is resolved by a firm rejection, by a root ALREADY located there --
                        # the branch is that root -- or by a curve that never dips toward a well.
                        # Only an unexplained dip leaves it undetermined: 50*pi^2, prec 1e-8, band
                        # 0.8, a partner 0.3e-9 relative inside the sub-bracket's end gave
                        # `no_parabola` at sigma 2.9e-9 against sigma_j 6.9e-8, and reading that as
                        # resolved left the total one short and certified. Flagging every
                        # `no_parabola` instead flagged 15 correct counts on the 50/65 grid, most of
                        # them sub-brackets a few ulps wide beside a partner already located.
                        # Explained means a root already located lies within the distance the
                        # side's own reading implies: `sigma/c`, at the calibrated bound on how far
                        # the true distance can exceed that estimate. A window of one sub-bracket
                        # width missed exactly that at 125*pi^2, prec 1e-8, band 0.8: the side read
                        # 5.1e-8 on the wall of a member located 4.5e-6 away, outside the bracket,
                        # and a correct count came back flagged -- ten such on the 50/65 grid.
                        # Plus two sub-bracket widths, for the ulp scale: at 1e-14 the side's best
                        # reading is rounding (50*pi^2, band 0.8: 7x the floor) and its implied
                        # reach fell a hair short of a member located 12 ulps away.
                        reach_side = (r2.sigma / c_out) / BIAS_LO_CACHE + 2.0 * (b2 - a)
                        explained = any(abs(q.lam - r2.lam) <= reach_side
                                        for q in taken if q is not r)
                        inconclusive |= (r2.reason != "not_a_minimum" and not explained
                                         and r2.sigma < SIDE_RATIO * float(t_here[j]))
                        continue
                    if any(abs(r2.lam - q.lam) <= max(
                            r2.delta_res if np.isfinite(r2.delta_res) else 0.0,
                            q.delta_res if np.isfinite(q.delta_res) else 0.0,
                            4.0 * np.spacing(abs(r2.lam))) for q in taken):
                        continue                    # already a located root, from its own bracket
                    # Found AGAIN, by distance. A find's own reading says how far the nearest
                    # eigenvalue can be -- `sigma/c`, at the calibrated bound -- and a root already
                    # located within that reach IS that eigenvalue. The bars above are not enough:
                    # at 65*pi^2, prec 1e-10, band 1.0 a side sub-bracket 2.2e-9 wide sat on the
                    # hump top between two located members, the refiner's fallback fitted a parabola
                    # to rounding there and reported a bar of 7.2e-12, and the point (sigma 4.8e-10,
                    # 3.2e-8 from both members) became a fifth level of four.
                    reach_find = (r2.sigma / c_out) / BIAS_LO_CACHE
                    if any(abs(r2.lam - q.lam) <= reach_find for q in taken):
                        continue
                    taken.append(r2)
                    found += 1
                    extra.append(Level(lam=r2.lam, mult=1, sigma=r2.sigma,
                                       delta_res=r2.delta_res, bracket=(a, b2), n_roots=1))
                    # Search ON, outward, past an exclusion of the width this find's own reading
                    # implies -- the same "any root within this reach IS r2" quantity the dedup
                    # above uses, reused as a radius rather than re-derived.
                    span = (r2.lam - a) if side < 0 else (b2 - r2.lam)
                    excl = max((r2.sigma / c_out) / bias_lo_eff, _SIDE_STANDOFF * span,
                               4.0 * np.spacing(abs(r2.lam)))
                    nxt = (a, r2.lam - excl) if side < 0 else (r2.lam + excl, b2)
                    if nxt[1] - nxt[0] > 4.0 * np.spacing(abs(r2.lam)):
                        sides.append((nxt[0], nxt[1], side))
                uncertain = inconclusive and found == 0
                # Is `r` itself a minimum, or the HUMP between two members? 65*pi^2, prec 1e-8,
                # band 0.3: a bracket narrower than the licence held two members, the refiner
                # accepted the hump between them (sigma 1.35e-8 against 1e-12 at the members), the
                # side search found both -- and the hump stayed as a third level, 5 of 4, certified.
                # A hump needs a well on BOTH sides, so it is only asked where roots lie both ways
                # inside the bracket; and sigma_1 falls toward both. Refining `r`'s own cell instead
                # dropped a GENUINE coarse member at 50*pi^2, prec 1e-6, band 0.3, whose cell held a
                # third member never located, and turned 3 of 3 into 1 of 3.
                if found:
                    near = [q.lam for q in taken
                            if q is not r and bracket[0] <= q.lam <= bracket[1]]
                    below = [q for q in near if q < r.lam]
                    above = [q for q in near if q > r.lam]
                    if below and above:
                        towards = (r.lam + 0.75 * (max(below) - r.lam),
                                   r.lam + 0.75 * (min(above) - r.lam))
                        reads = [float(np.atleast_1d(tens(z))[0]) for z in towards]
                        drop = max(reads) < _HUMP_RATIO * r.sigma
                side_evals += obj.n_evals - side_mark
            # ATTRIBUTION. A branch the read counted INSIDE the bracket and a level located inside
            # the same bracket can be the same eigenvalue, and until now nothing compared them --
            # so the member was reported twice (170*pi^2, band 0.3, prec 1e-8: 9 of 8, certified).
            # Each such branch is owned ONCE: by a located level near where `sigma_j/c` puts it,
            # or else by the root. Only counted branches are claimable, so this can never take out
            # what the read did not put in -- the failure capping produced at 2 of 3, prec 1e-6.
            #
            # ONLY levels the SIDE SEARCH just located may claim, not levels already located from
            # their own bracket. Both were measured, and the wider source REGRESSES the committed
            # grid: at 50*pi^2, band 0.3, prec 1e-12 and 1e-14 it takes 3 of 3 to 2 of 3, silently.
            #
            # And the cell says something worth keeping, because the 3 was never right. The read
            # at the first member counts the second inside (branch 1, d_est 1.59e-10, side +1) and
            # the second is ALSO reported from its own bracket -- a double count -- while the
            # third member at 493.480220054462 is never located at all. Two errors cancelling.
            # Letting the wider source claim removes the double count and leaves the miss visible,
            # which is the honest state and a WRONG TOTAL either way. The miss is the same refiner
            # defect the 170*pi^2 xfail records: the side search's residual starts one exclusion
            # radius past its find, that radius is `(sigma/c)/bias_lo` on a CONVERGED find where
            # `sigma` is the floor rather than a neighbour distance, so the residual begins on the
            # find's own wall and the refiner returns that endpoint (`not_a_minimum` at
            # 493.480220054315, with the member 1.5e-10 further in). Fix that first; the wider
            # claimant source is then a strict improvement rather than a trade.
            claimants = [q.lam for q in extra if bracket[0] <= q.lam <= bracket[1]]
            n_claimed, _ = _claim(m.branches, r.lam, claimants, bias_lo_eff, bias_hi_eff)
            mu = int(m.mu) - n_claimed
            assert mu >= 1, (f"attribution took the root's own branch: mu={m.mu}, "
                             f"claimed={n_claimed}")
            m = replace(m, mu=mu)
        elif wb is None:
            levels.append(Level(lam=r.lam, mult=1, sigma=r.sigma, delta_res=r.delta_res,
                                bracket=bracket, n_roots=n_roots))
            continue
        else:
            m = multiplicity(tens, r.lam, wide_bracket=wb, prec=prec, rule=rule)
        if not drop:
            levels.append(Level(lam=r.lam, mult=m.mu, sigma=r.sigma, delta_res=r.delta_res,
                                bracket=bracket, wide_bracket=wb or (), gap_ratio=m.gap_ratio,
                                c2_rejected=m.c2_rejected, n_roots=n_roots,
                                # The estimator's own DECLINE, carried through. Dropping it made every
                                # wrong total under the reach rule silent on the wip branch -- 0 of 15
                                # flagged -- which was a wiring omission, not a property of the rule.
                                count_uncertain=(bool(m.ambiguous) if uncertain is None
                                                 else uncertain),
                                uncertain_reason=(
                                    "the multiplicity estimator declined: a branch's estimated "
                                    "distance landed where the inside and outside populations "
                                    "overlap, so the evidence does not decide whether that "
                                    "eigenvalue is in this bracket"
                                    if (m.ambiguous if uncertain is None else uncertain) else "")))
        levels.extend(extra)

    levels.sort(key=lambda x: x.lam)

    # The grouping cross-check. Reads only cached points, so it is charged to stage 3's window
    # and must leave it at zero.
    if collapse:
        levels, n_amb = _collapse_ambiguous(obj, lams, levels, prec, WIDE_FACTOR, rule)
    else:
        n_amb = 0

    # The count consistency check. Free -- it compares numbers the pipeline already has -- and
    # last, so it sees the levels a caller will actually receive.
    levels, uncertain = _flag_uncertain_counts(
        levels, folds_are_duplicates=(merge == "attained" and rule == "reach"))
    # The READ's own spend: the whole stage-3 window less the probes and the side search, both of
    # which are evaluations by design and are reported as themselves.
    mult_evals = (obj.n_evals - mark) - probe_evals - side_evals

    unmet = tuple(x.lam for x in levels if not x.met(prec))
    attained = max((x.attained_prec for x in levels
                    if np.isfinite(x.attained_prec)), default=prec)

    return SolveResult(tuple(levels), SolveDiagnostics(
        search=sd, n_brackets=len(res.brackets), n_rejected=rejected, n_merged=merged,
        search_evals=search_evals, refine_evals=refine_evals,
        multiplicity_evals=mult_evals, probe_evals=probe_evals, side_evals=side_evals,
        evaluations=obj.n_evals, n_ambiguous=n_amb,
        n_uncertain=len(uncertain),
        uncertain=tuple((x.lam, x.mult, x.count_alt, x.uncertain_reason) for x in uncertain),
        prec_requested=prec, attained_prec=float(attained), unmet=unmet))
