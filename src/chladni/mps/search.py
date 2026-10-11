"""Stage 4: the composed pipeline, and the only seam a candidate MUST present.

Promoted from `tests/v1_solve.py`, whose remaining `v1_solve` factory now scores THIS module on
the seam. `docs/design/candidates.md` section 1 calls `solve` "the only mandatory seam": a
method that does not decompose into bracket -> refine -> multiplicity implements this and
nothing else, so it is where v1 and a monolithic candidate are compared on equal terms.

`MPSEigensolver.search_interval` is the caller, and `solve_interval` narrows what comes back to
the method-agnostic `Spectrum` above it.

    solve(obj, *, prec, collapse, merge, rule, reach_slope, phase_d) -> SolveResult

The scored SEAM is narrower than that signature and is what a candidate must present:

    search(tensions, a, b, *, prec, modes=None) -> [(lam, mult), ...]

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

# Phase D is BUILT, behind `phase_d=`, and OFF

Stage 3 answers "how many branches dip HERE"; Phase D answers "are these two reported roots the
same eigenvalue", which distance provably cannot settle, since one eigenvalue found twice and
two genuine eigenvalues occur at the same separation. `_gram_group` decides it on the interior
Gram angle, and the instrument it rests on is measured across the whole roster
(`notes/landscape_findings.md` section 36): worst distinct 0.2544 < `TAU_GRAM` 0.5 < weakest
duplicate 1.0000.

**It is not the default, because it changes nothing measurable yet.** Measured over five cluster
grids at five precisions and eleven bands, plus the roster:

    grid        phase_d="distance"                phase_d="angle"          gram pairs
    50*pi^2     53 right, 0 wrong                 53 right, 0 wrong            23
    65*pi^2     47 right, 0 wrong, 6 declined     identical                    29
    125*pi^2    52 right, 0 wrong, 1 declined     identical                    26
    145*pi^2    22 right, 0 wrong, 23 declined    identical                     8
    170*pi^2    41 right, 1 wrong, 5 declined     identical                    29
    roster                                        identical                     0

The angle is CONSULTED -- 115 pairs, so the rule is exercised rather than inert, which was the
expectation going in and was wrong -- and it agrees with the distance rule on every one. It
regresses nothing and improves nothing, so there is no case for promoting it and the seam keeps
both reachable. The one wrong cell left at `170*pi^2` is the refiner's residual-abutting-an-
exclusion defect and not a grouping question, which is why the angle does not touch it.

The roster asks the angle NOTHING at `prec = 1e-8`: its levels are further apart than a licence,
so no pair reaches the prefilter's survivors. The grouping regime lives in the split-degeneracy
grids and nowhere else that is currently measured.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

from .bracket import bracket_interval
from .multiplicity import (BIAS_HI, BIAS_HI_CACHE, BIAS_LO, BIAS_LO_CACHE, COUNT_BELOW_CACHE,
                           MAX_BRANCHES, PROMINENCE, REJECT_ABOVE_CACHE, SIDE_RATIO,
                           SLOPE_STANDOFF, TAU_GRAM, WIDE_FACTOR, multiplicity, wide_bracket_for)
from .refine import DELTA_SAFETY, refine

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
#:
#: **0.1 was not enough, and the whole grid says so.** One cell was swept for the value above;
#: over five cluster grids at five precisions and eleven bands, unexercisable cells excluded:
#:
#:     standoff   right   wrong   declined
#:     0.1          215       1         35      <- 170*pi^2, prec 1e-12, band 0.3: 7 of 8
#:     0.3          216       0         35
#:     0.5          216       0         35
#:
#: 0.3 closes the last wrong cell on every grid and moves nothing else -- the declines do not
#: budge, and 0.3 and 0.5 are identical, so this is a plateau and not a fitted value. It is the
#: smaller of the two that work, which leaves more of the residual still searchable.
#:
#: That wrong cell was the LAST source of a lost eigenvalue anywhere in the grids: section 37
#: labelled all 179 refiner rejections by whether the bracket held a true eigenvalue, and the
#: only two that did were this defect, at `170*pi^2` and `125*pi^2`.
_SIDE_STANDOFF = 0.3

#: Interior samples taken in a sub-bracket that starts at a find and was refused at that end,
#: before it is refined once more. Evenly spaced: the second member's well is somewhere inside and
#: nothing says where; four put a sample within a tenth of the span of any point. Spent only on a
#: refusal of that one shape, so it costs nothing where it does not fire.
_RESAMPLE = 4

#: How far above a well's floor a cached sample must sit to carry slope information for
#: `_slope_from_cache`. Inside the flat bottom `sigma` saturates at the floor, and the cone
#: identity the estimator rests on fails there; at twice the floor the cone part
#: `sqrt(v^2 - floor^2)` is already 87% of `v`, so the correction is well conditioned.
_SLOPE_FLOOR_MARGIN = 2.0

#: How far the blind cache slope may exceed the floor-aware one before it is judged to be
#: measuring the floor rather than the wall. Wherever the cone model even roughly holds the two
#: agree within 4x -- measured 1.0 to 3.9 over every root of two cells, a hump top included --
#: while the defect it guards read 4.6e8 (a secant across 2 ulps of a 9.6e-7 floor). 100 leaves
#: 25x above the first and 4.6e6 below the second.
_SLOPE_FLOOR_GUARD = 100.0


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
    #: How this level's multiplicity was settled, where something had to settle it. `"gram"` is
    #: Phase D's interior angle; empty means no grouping question arose. It is the difference
    #: between a count that survived a question and one nobody asked.
    count_decided_by: str = ""
    #: The Moler-Payne bound (`chladni.integrals.certify`), ABSOLUTE: some true eigenvalue lies
    #: within it of `lam`. NaN where none was computed; infinite where it was refused, with
    #: `bound_reason` saying why. `mp_eps` is its relative core, `cancellation` the floor the
    #: candidate's own rounding puts under it. Attached by `MPSEigensolver.search_interval`,
    #: which holds the pencil and the domain; `solve` sees only an objective and never sets them.
    mp_bound: float = float("nan")
    mp_eps: float = float("nan")
    cancellation: float = float("nan")
    bound_reason: str = ""
    #: False when `delta_res` missed the request it was refined for (`delta_res > prec*lam`)
    #: and a Moler-Payne bound is in hand, so `bound` reports that bound alone. Set by
    #: `MPSEigensolver.certify_levels`, which knows `prec`; see `bound` for why.
    delta_res_admitted: bool = True
    #: The request this level was solved for. Stamped by `solve`; NaN on a Level built
    #: elsewhere, which then claims no `margin`, no `resolution` and no shortfall.
    prec: float = float("nan")

    @property
    def margin(self):
        """`prec * lam / (eps/c)`, the plan's R: how many local floor scales the request spans.

        `delta_res = DELTA_SAFETY * eps / c` (`mps/refine.py`), so this is
        `DELTA_SAFETY * prec * lam / delta_res`. `met` held exactly when `margin >= DELTA_SAFETY`
        on all 8062 levels of the douse benchmark (`notes/margin_rule_offline.md`). It is what a
        re-plan needs to know: how far short a level fell, not only that it did."""
        if not (np.isfinite(self.prec) and np.isfinite(self.delta_res) and self.delta_res > 0):
            return float("nan")
        return float(DELTA_SAFETY * self.prec * abs(self.lam) / self.delta_res)

    @property
    def resolution(self):
        """The neighbourhood this level's COUNT stands for: `mult` eigenvalues within
        `resolution` of `lam`, whatever their internal splitting.

        `max(prec * lam, bound, width)`. On a met level it is the merge licence, as before. On
        an unmet level it widens to the value's own bar: a level whose value is known to 1e-4
        cannot claim its count at 1e-10, and on the douse benchmark every value that fell
        outside the 1e-6 match window was inside its bar. `width` (a collapsed group's wide
        bracket) is the special case this generalizes (`docs/design/douse_requirements.md`,
        "Phase 3")."""
        if not np.isfinite(self.prec):
            return float("nan")
        vals = [self.prec * abs(self.lam), self.width]
        if np.isfinite(self.bound):
            vals.append(self.bound)
        return float(max(vals))

    @property
    def count_shortfall(self):
        """True when the count was read where the floor blurs it: the level is unmet.

        A FLAG, never an exception -- unlike `count_uncertain`, it does not reach
        `certified_counts`, which `solve_interval` raises on. About 40% of levels at 1e-10 are
        unmet even on the douse references' basis. On the benchmark every count defect beside a
        returned level sat on an unmet one: a missed partner, or a simple level read double
        inside its floor. A level that MEETS `prec` with a wrong count was seen only on starved
        bases, which is the planner's to refuse (`douse_requirements.md`, "Phase 3")."""
        return bool(np.isfinite(self.prec) and not self.met(self.prec))

    @property
    def bound(self):
        """The error bar reported for this level.

        `mp_bound` (Moler-Payne, from the candidate itself) is an upper bound on the distance to
        a true eigenvalue. `delta_res` (the refiner's, `DELTA_SAFETY * eps / c` from the
        tension's floor) is a resolution limit, and is an upper bound only while the floor is
        the well's own: on a basis too poor for the request the floor is the BASIS's, the well
        can sit away from the eigenvalue, and `delta_res` read up to ~1e5x under the true error
        while Moler-Payne covered all 715 levels of `notes/bound_coverage.md`. Every such level
        had `delta_res > prec*lam`, so a `delta_res` that missed the request is not admitted
        (`delta_res_admitted`) and the level reports `mp_bound`. An admitted `delta_res` is
        reported when it is the tighter -- on the threshold path near 1e-13 it can be 100x
        tighter than Moler-Payne, and it covered the error there to within 2.6x. `delta_res`
        alone where no bound was computed (`certify=False`, or a boundary that is not
        Dirichlet), and then nothing guards it."""
        if not self.delta_res_admitted and np.isfinite(self.mp_bound):
            return float(self.mp_bound)
        vals = [v for v in (self.delta_res, self.mp_bound) if np.isfinite(v)]
        return float(min(vals)) if vals else float("nan")

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
        parabola fit the refiner already performed -- or, where `MPSEigensolver` attached one,
        from the Moler-Payne bound, whichever is tighter (`bound`).
        """
        b = self.bound
        if not np.isfinite(b) or self.lam == 0.0:
            return float("nan")
        return abs(b) / abs(self.lam)

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
    #: The rejections that were NOT plausibly empty: `(lam, sigma, reason)` per bracket the
    #: refiner refused while reaching a `sigma_1` no larger than the deepest ACCEPTED level's.
    #: An empty region's `sigma` sits on the barrier between wells, orders above any root; a
    #: bracket that dipped as far as a located eigenvalue and was still refused is a candidate
    #: LOST EIGENVALUE.
    #:
    #: **It does NOT catch the Linux wrong total it was prompted by.** That refusal was a rank
    #: flip making a bracket END look deep, and the refiner refused at that flipped flank point,
    #: `sigma = 5.18e-07` -- shallower than the worst accepted level, `4.08e-07` -- so this reads
    #: `()` there. What it does catch is the starved-disk drop in
    #: `test_a_starved_basis_does_not_DROP_levels_silently[0.45]`. `notes/session_2026-09-19.md`.
    rejected_deep: tuple = ()
    n_merged: int = 0                # located roots folded into a neighbour within `prec`
    n_ambiguous: int = 0             # levels collapsed because their grouping was arbitrary
    n_gram_pairs: int = 0            # root pairs Phase D's angle was asked about
    n_gram_collapsed: int = 0        # groups it read as ONE eigenfunction found more than once
    n_gram_summed: int = 0           # groups it read as independent eigenvalues
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
    def n_count_shortfall(self):
        """Levels whose count was read where the floor blurs it (`Level.count_shortfall`). They
        are the unmet levels: a value short of `prec` means its count is too. Reported, never
        raised -- it does not enter `certified_counts`."""
        return len(self.unmet)

    @property
    def certified_counts(self):
        """Is every level's MULTIPLICITY certified? The other half of the honesty clause.

        A DEEP REJECTION also answers no, through `n_uncertain`. The clause is "nothing is
        silently wrong about a count that is certified", and a count short by a bracket the
        refiner refused is wrong in exactly that way. Not every such refusal is deep, though:
        the Linux wrong total on the N65 cluster was refused at a rank-flipped flank point
        shallower than every located level, and this still answers True there.
        """
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


def _c2_holds(tens, x, wb, j):
    """(C2) for branch `j` at `x`: does `sigma_j` have a local minimum here, to `PROMINENCE`,
    against the wide bracket `wb`? `None` when it cannot be read (no wide bracket, or the branch
    is past what the tension vector carries).

    `multiplicity.py` calls (C2) "the whole definition" of a level's multiplicity. The reach rule
    reads `sigma_j/c` as a distance instead, which is right on a cone and wrong in two places
    measured off the rectangle grids (`tests/test_floor_multiplicity.py`): at an exact double
    `sigma_2` at the root is its FLOOR, not a distance, and on a noisy floor the cache slope `c`
    can read 14000x steep, so O(1) branches read as inside the bracket. On the wide bracket --
    cached points past `eps/c`, where the cone has risen above the floor -- a branch belonging to
    this level has risen (18x on the L-shape's `10 pi^2`) and an O(1) branch has not (1.0x on the
    spike triangle). Reads the cache: free on a shared objective.
    """
    if not wb or len(wb) != 2:
        return None
    t = np.atleast_1d(np.asarray(tens(x), dtype=float))
    tl = np.atleast_1d(np.asarray(tens(wb[0]), dtype=float))
    tr = np.atleast_1d(np.asarray(tens(wb[1]), dtype=float))
    if j >= min(t.size, tl.size, tr.size) or not (t[j] > 0):
        return None
    return bool(min(tl[j], tr[j]) >= PROMINENCE * t[j])


def _slope_from_cache(lams, vs, lam, floor=0.0):
    """`c = dsigma/dlam` from the two CACHED samples nearest the root, one either side.

    Free, which matters: an explicit `lam +/- h` probe costs two evaluations per level, and
    charging stage 3 for them contradicts its contract of spending nothing. The search has
    already sampled both flanks of every well it bracketed, so the slope is in hand.

    Cancellation-free, as in `isolation`: on a cone the flanks straddle the minimizer, so
    `v_lo = c*(lam* - lo)` and `v_hi = c*(hi - lam*)`, hence `c = (v_lo + v_hi)/(hi - lo)` with
    no subtraction of nearly-equal quantities and no dependence on where the minimum is.

    **Floor-aware, because the cone identity assumes `sigma` reaches zero.** A well that bottoms
    out at `floor` reads `sqrt(floor^2 + c^2 d^2)` on its flanks, so the sum carries about
    `2*floor` extra -- and when the nearest samples sit close together, as they do once the
    refiner has been there, the estimate becomes `2*floor/(hi - lo)` and says nothing about the
    wall. Measured at 50*pi^2, prec 1e-6, band 1.3, on a penalized objective with a 9.6e-7 floor:
    8.45e6 against a true 0.02-0.08. `solve` sizes the side search's exclusion radius from this
    slope, so the radius collapsed to 1e-13, the side search refined against the root itself, and
    found it again 1e-12 away as a new level -- one eigenvalue reported twice, 5 of 3.

    **So the floor-aware estimate is a GUARD, not a replacement.** It skips samples within
    `_SLOPE_FLOOR_MARGIN * floor` as flat bottom and reduces each flank value to its cone part
    `sqrt(v^2 - floor^2)` -- and it is used only when the blind estimate exceeds it by more than
    `_SLOPE_FLOOR_GUARD`. Everywhere else the blind estimate is returned unchanged, because the
    reach rule's bounds were calibrated on it: applied unconditionally, the floor-aware estimate
    moved a HUMP root at 65*pi^2, prec 1e-8, band 0.3 -- `sigma` 1.35e-8 there is the hump top, not
    a floor -- by 2.3x, and that flipped four calibrated tests. With `floor = 0`, or wherever the
    guard does not fire, this is the estimator it replaced, bit for bit.
    """
    blind = _cone_slope(lams, vs, lam, np.ones(np.shape(vs), bool), 0.0)
    floor = max(float(floor), 0.0)
    if floor <= 0.0:
        return blind
    aware = _cone_slope(lams, vs, lam, vs > _SLOPE_FLOOR_MARGIN * floor, floor)
    return aware if (aware > 0.0 and blind > _SLOPE_FLOOR_GUARD * aware) else blind


def _cone_slope(lams, vs, lam, usable, floor):
    """The cone identity over the nearest `usable` sample either side, floor removed. 0 if none."""
    left = (lams < lam) & usable
    right = (lams > lam) & usable
    if not np.any(left) or not np.any(right):
        return 0.0
    i = int(np.argmax(lams[left]))
    j = int(np.argmin(lams[right]))
    lo, v_lo = float(lams[left][i]), float(vs[left][i])
    hi, v_hi = float(lams[right][j]), float(vs[right][j])
    if floor > 0.0:
        v_lo, v_hi = np.sqrt(v_lo * v_lo - floor * floor), np.sqrt(v_hi * v_hi - floor * floor)
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


def _claim(branches, lam_root, claimants, bias_lo, bias_hi, *, with_ambiguity=False):
    """Which branches the read counted are ALREADY ACCOUNTED FOR by a located level.

    Returns `(n_claimed, claimed)`, `claimed` the COUNTED indices claimed, one per claimant at
    most; with `with_ambiguity=True`, `(n_claimed, claimed, ambiguous)`.

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
    a branch the read actually counted can be SUBTRACTED, so nothing the read did not put in can
    be taken out.

    # Undecided branches are targets, not subtractions

    The side search runs only where the read left a branch undecided, and is aimed at it. So an
    undecided branch can OWN a find -- which subtracts nothing, since `mu` never held it -- and
    that stops the find taking a counted member out. Counted-only targets did exactly that:
    50*pi^2, band 0.3, prec 1e-10, `tau = 0.02 prec`, a find of the undecided member claimed the
    counted one's branch and the total came back 2 of 3, SILENT (`notes/smooth_filter.md` 10.7).

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

    # Matching: every admissible assignment, and ORDER

    With the cache bias pair the admissible band spans 3.6x, so one claimant can admit two
    branches and one branch two claimants, and a best-match rule then guesses. Two facts decide
    more than a match can. Each branch is owned once. And on one side of the root, a nearer
    eigenvalue has the smaller `sigma_j`, so claimants on one side, taken outward, own branches
    in increasing index. Every assignment consistent with both is enumerated -- at most
    `MAX_BRANCHES` branches and a handful of finds, so a few hundred -- and those matching the
    most claimants kept. Two finds against a counted and an undecided branch then have one
    reading (50*pi^2, band 0.3, prec 1e-6: the counted one's member nearer, the undecided one's
    further), where greedy preference for either kind got 4 of 3 or 2 of 3.

    Where the kept assignments disagree on how many COUNTED branches they take, the evidence does
    not decide the total -- one find, admissible to a counted and an undecided branch, is either
    -- and `ambiguous` says so. The count returned is then the largest, the pre-existing reading,
    so a total that was right stays right and one that was wrong is now flagged rather than
    silent.
    """
    usable = [b for b in branches if b.verdict in ("counted", "undecided")
              and b.index >= 1 and b.d_est > 0]
    qs = sorted((q for q in claimants if abs(q - lam_root) > 0),
                key=lambda q: abs(q - lam_root))
    options = []
    for q in qs:
        off, side = abs(q - lam_root), (1 if q > lam_root else -1)
        options.append([None] + [b for b in usable if b.side in (0, side)
                                 and b.d_est / bias_hi <= off <= b.d_est / bias_lo])

    def consistent(combo):
        owned = [b.index for b in combo if b is not None]
        if len(set(owned)) < len(owned):
            return False
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):       # qs is sorted by distance, so i is nearer
                bi, bj = combo[i], combo[j]
                if bi is None or bj is None or (qs[i] > lam_root) != (qs[j] > lam_root):
                    continue
                if bi.index > bj.index:
                    return False
        return True

    kept, most = [], -1
    for combo in itertools.product(*options):
        if not consistent(combo):
            continue
        n = sum(b is not None for b in combo)
        if n > most:
            kept, most = [combo], n
        elif n == most:
            kept.append(combo)
    readings = {tuple(sorted(b.index for b in combo if b is not None and b.verdict == "counted"))
                for combo in kept}
    claimed = max(readings, key=lambda t: (len(t), [-i for i in t])) if readings else ()
    ambiguous = len({len(t) for t in readings}) > 1
    if with_ambiguity:
        return len(claimed), claimed, ambiguous
    return len(claimed), claimed


def _gram(obj, a, b):
    """`|<u(a), u(b)>|` between two reported roots. Never within one `lam`.

    Both vectors are sampled on the pencil's own interior points, which are fixed at
    construction, so this is a plain dot product -- no common refinement, no quadrature, no
    lifting to `L2(Omega)`.
    """
    u, v = obj.modes(obj.from_lam(a)), obj.modes(obj.from_lam(b))
    return abs(float(u @ v))


def _gram_group(obj, levels, prec, tau=TAU_GRAM):
    """PHASE D. Decide collapse-versus-merge among reported roots by the INTERIOR GRAM ANGLE.

    Returns `(levels, n_pairs, n_collapsed, n_summed)`.

    # The question, and why nothing else can answer it

    Two roots reported within `prec*lam` of each other are ONE level of `S(prec)` either way.
    What is not settled is its multiplicity: one eigenvalue located twice must COLLAPSE and be
    RECOUNTED, while two genuine eigenvalues must MERGE and be SUMMED -- and the two occur at
    the SAME separation, so no rule on `|lam_i - lam_j|` can separate them.
    `_flag_uncertain_counts` states this exactly and refuses to decide it; this is what decides
    it.

    # The verdicts

        |cos| >= tau    the same eigenfunction: COLLAPSE, and recount rather than sum.
                        Summation IS the double count, which is why the recount is not optional.
        |cos| <  tau    independent: MERGE and SUM, at the sigma-weighted centroid.

    The recount costs nothing here. Each member's own read already answered "how many eigenvalues
    are at this location", and members of one collapse class are at the same location by the
    angle's verdict, so their reads answer the same question -- the recount is taking one of them
    rather than adding them up. Members that DISAGREE are the interesting case and are flagged
    rather than silently resolved.

    # Both arms produce ONE level, and that is not a loss

    Leaving a distinct pair as two levels carries more information, and is wrong for this scorer:
    `S(prec)` defines two eigenvalues inside `tol_merge` to be one level, so reporting two would
    score as a spurious level against every committed grid. The distinction the angle buys is in
    the MULTIPLICITY, not in the count of levels.

    # Cost

    One `separate`-mode GSVD per level in a group, and groups are rare -- only roots the
    prefilter left within a licence of each other reach here. `search.md` section 2.6 budgets one
    such call per reported root, at the end, which is where this runs.
    """
    if not levels:
        return levels, 0, 0, 0
    # Grouped against the group's FIRST member, never its predecessor: `S(prec)` groups by
    # DIAMETER, and chaining is the mistake `_same_point` already records as made twice.
    groups = []
    for x in levels:
        if groups and abs(x.lam - groups[-1][0].lam) <= prec * max(abs(x.lam), 1.0):
            groups[-1].append(x)
        else:
            groups.append([x])

    out, n_pairs, n_coll, n_sum = [], 0, 0, 0
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        n_pairs += len(g) - 1
        # Classes of "the same eigenfunction". Greedy against each class's own representative,
        # which is enough at these sizes -- a group is two or three roots inside one licence.
        classes = []
        for x in g:
            for c in classes:
                if _gram(obj, c[0].lam, x.lam) >= tau:
                    c.append(x)
                    break
            else:
                classes.append([x])

        mults, reps, disagreed = [], [], False   # noqa: E501 -- see the SUM guard below
        for c in classes:
            rep = min(c, key=lambda y: y.sigma)      # deepest: the best-located member
            reps.append(rep)
            seen = {y.mult for y in c}
            disagreed |= len(seen) > 1
            mults.append(int(rep.mult))
        total = int(sum(mults))
        deepest = min(g, key=lambda y: y.sigma)
        n_roots = sum(y.n_roots for y in g)
        gaps = tuple(float(b.lam - a.lam) for a, b in zip(g[:-1], g[1:]))

        # THE SUM ARM IS ONLY SOUND WHEN EACH READ COUNTS ONE EIGENVALUE, and `search.md` does
        # not say so. Its claim that the ambiguous eigenspace is benign -- "parallel collapses and
        # recounts to 2, orthogonal sums 1 + 1 to 2" -- assumes every member read 1. Under the
        # reach rule a read answers "how many eigenvalues are inside MY bracket", so a degenerate
        # level located twice has both members reading the full multiplicity and the sum doubles
        # it. Measured on `_gram_group` directly, two roots inside one licence:
        #
        #     each member reads   parallel (collapse)   orthogonal (sum)   truth
        #     1                          1                    2            both arms right
        #     2                          2                    4            sum is WRONG
        #
        # Attribution is what makes the first row hold: it already removes the eigenvalues one
        # root's read counted that another root owns. It cannot help here, because both members
        # own the same degenerate eigenspace and neither is the other's partner.
        #
        # No guess is available, so the level is FLAGGED and carries the other reading. Capping by
        # a recount over the union bracket was considered and not done: `_collapse_ambiguous`
        # records that a spanning read folds four correct levels into two on the `65*pi^2`
        # cluster, so it is not the safe direction either.
        suspect_sum = len(classes) > 1 and max(mults) > 1

        if len(classes) == 1:
            n_coll += 1
            lam, delta = deepest.lam, deepest.delta_res
        else:
            n_sum += 1
            # The sigma-weighted centroid: deeper roots are better located, so they pull harder.
            w = np.array([1.0 / max(r.sigma, np.finfo(float).tiny) for r in reps])
            lam = float(np.array([r.lam for r in reps]) @ w / w.sum())
            # The level now stands for eigenvalues either side of `lam`, so its bar must reach
            # them or clause (2)'s honesty fails. Bounded by half the licence, so this can never
            # by itself put the level in `unmet`.
            half = 0.5 * (max(y.lam for y in g) - min(y.lam for y in g))
            delta = max(half, *(y.delta_res if np.isfinite(y.delta_res) else 0.0 for y in g))

        out.append(replace(deepest, lam=lam, mult=total, delta_res=delta, n_roots=n_roots,
                           spacings=gaps, count_decided_by="gram",
                           # DECIDED, so not ambiguous. Saying otherwise would be a lie about
                           # the one thing this function exists to settle.
                           grouping_ambiguous=False,
                           count_uncertain=(disagreed or suspect_sum),
                           count_alt=(max(mults) if suspect_sum
                                      else min(y.mult for y in g) if disagreed else 0),
                           uncertain_reason=(
                               f"the Gram angle read {len(classes)} independent eigenfunctions "
                               f"here and summed their counts to {total}, but a member reads "
                               f"multiplicity {max(mults)} -- so each read may already count the "
                               f"other's eigenvalues, which is what a degenerate eigenspace "
                               f"located twice looks like. The sum is not certified"
                               if suspect_sum else
                               "the Gram angle put these roots at one eigenfunction, but their "
                               "individual multiplicity reads disagree, so which count the "
                               "collapse should carry is not decided" if disagreed else "")))
    return out, n_pairs, n_coll, n_sum


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

    A level carrying `count_decided_by == "gram"` also passes: Phase D's interior angle answered
    the very question the `mult < n_roots` test exists to raise, so re-raising it would report a
    settled question -- and `solve_interval` raises on a flag rather than warning.

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
        if x.count_decided_by == "gram":
            # The one doubt this function reports is "N eigenvalues within prec*lam, or one
            # located N times?", and it says in its own docstring that distance cannot tell
            # those apart. Phase D's angle just did, on evidence. Flagging it anyway would
            # report a question that has an answer -- and `solve_interval` RAISES on a flag.
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


def _deep_rejections(refused, levels):
    """The refused brackets that were NOT plausibly empty, as `(lam, sigma, reason)`.

    **A rejection is suspicious when the `sigma_1` the refiner reached is no larger than the
    WORST accepted level's.** `search.md`'s honesty clause is that nothing is silently wrong
    about a count that is certified, and a total short by a refused bracket is wrong in exactly
    that way -- so the question is only whether the refusal plausibly dropped an empty region or
    an eigenvalue, and the refiner already measured what decides it.

    The comparison needs no new constant, which is the point. A bracket holding no root is
    refused somewhere on the barrier between two wells, and on the cell this was built for that
    barrier is 1.2e-05 against located roots at 2.6e-11 to 4.1e-07 -- four orders of daylight. The
    accepted levels supply the scale, so the rule travels to a domain whose floor sits elsewhere
    instead of encoding this pencil's.

    With NOTHING accepted, every refusal is suspicious: a solve that located no eigenvalue and
    refused a bracket has not earned a certified count either.
    """
    if not refused:
        return ()
    worst = max((x.sigma for x in levels if np.isfinite(x.sigma)), default=float("inf"))
    return tuple((float(r.lam), float(r.sigma), r.reason) for r in refused
                 if not (np.isfinite(r.sigma) and r.sigma > worst))


def _precision_summary(levels, prec):
    """`(unmet, attained)`: the levels short of `prec`, and the worst attained precision. One
    place, because `MPSEigensolver` recomputes it after attaching Moler-Payne bounds."""
    unmet = tuple(x.lam for x in levels if not x.met(prec))
    attained = max((x.attained_prec for x in levels
                    if np.isfinite(x.attained_prec)), default=prec)
    return unmet, float(attained)


def solve(obj, *, prec=None, collapse=False, merge="attained", rule="reach",
          reach_slope="cache", phase_d="distance"):
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
    if phase_d not in ("distance", "angle"):
        raise ValueError(f"phase_d must be 'distance' or 'angle', got {phase_d!r}")
    if phase_d == "angle" and not getattr(obj, "has_modes", False):
        # Refused rather than silently no-op: a caller that asked for the angle and got the
        # distance rule would be told nothing, and the whole point of the seam is that the two
        # are measured one at a time.
        raise ValueError(
            'phase_d="angle" needs candidate eigenfunctions, and this objective carries none. '
            "Build it with `Objective(..., modes=...)` or `Objective.from_pencil`, which wires "
            "`MPSPencil.interior_image` in; a bare tension function cannot answer Phase D.")
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
    located, refused, ends = [], [], []
    for b in res.brackets:
        lo, hi = obj.in_lam_bracket(b.lo, b.hi)
        hint = obj.in_lam(b.k_hint) if np.isfinite(b.k_hint) else None
        r = refine(obj, lo, hi, prec=prec, lam_hint=hint,
                   interior=(obj.in_lam(b.lo), obj.in_lam(b.hi)))
        if r.accepted:
            located.append((r, (lo, hi)))
        else:
            # A bracket the search kept on purpose and the refiner threw away. USUALLY not a
            # failure of either: dropping a doubtful region costs an eigenvalue and keeping one
            # costs about three evaluations, so the search is right to hand it on and the refiner
            # is right to refuse it.
            #
            # But "usually" was doing unpaid work here. The refiner's `sigma` says which case this
            # is, at no cost, and `_deep_rejections` below reads it: an empty region is refused at
            # a `sigma` sitting on the barrier between wells, ORDERS above any root, while a
            # bracket that dipped as far as a located eigenvalue and was still refused is a
            # candidate lost eigenvalue.
            #
            # A refusal can also be SHALLOW and still lose an eigenvalue, and this does not see
            # it: the Linux wrong total on the N65 cluster was a rank flip making the bracket's
            # upper END look deepest, so the refiner walked away from the root and refused at the
            # flipped end, above every located level. The depth test cannot tell that from an
            # empty bracket; only an objective continuous in `lam` removes it.
            refused.append(r)
            ends.append((r, (lo, hi)))
    # A bracket refused AT ITS OWN END, with the descent leading out of it, can hold the wall of
    # a member stage 1 placed just outside. 65*pi^2, band 1.3, prec 1e-12, `tau = 0.1 prec`: the
    # fourth bracket began 0.1 licences above member 3, the refiner walked to its lower end and
    # refused it -- correctly, the minimum was outside -- and nothing looked there. 3 of 4, and
    # silent, since the refusal read 1.0e-12 against located levels down to 2.6e-13.
    #
    # So step out once, one bracket width past that end, clipped to the interval and to every
    # other stage-1 bracket so the step cannot re-find a root located from its own. An empty
    # region stays refused at the cost of one more refinement; a find is located like any other.
    spans = [obj.in_lam_bracket(b.lo, b.hi) for b in res.brackets]
    edge_lo, edge_hi = obj.in_lam_bracket(obj.lo, obj.hi)
    for r, (lo, hi) in ends:
        w = hi - lo
        if not (r.reason in ("not_a_minimum", "endpoint") and w > 0):
            continue
        others = [s for s in spans if s != (lo, hi)]
        if abs(r.lam - lo) <= 0.01 * w:
            a = max([lo - w, edge_lo] + [s_hi for s_lo, s_hi in others if s_hi <= lo])
            b = lo
        elif abs(r.lam - hi) <= 0.01 * w:
            a = hi
            b = min([hi + w, edge_hi] + [s_lo for s_lo, s_hi in others if s_lo >= hi])
        else:
            continue
        if not b - a > 4.0 * np.spacing(abs(r.lam)):
            continue
        r_out = refine(obj, a, b, prec=prec, interior=(a, b))
        if r_out.accepted:
            located.append((r_out, (a, b)))
            refused.remove(r)               # explained: the end was this find's wall
    rejected = len(refused)
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
        d = abs(r.lam - first.lam)
        if d > licence:
            return False
        bars = [x.delta_res for x in (first, r) if np.isfinite(x.delta_res)]
        if d <= max(min(bars, default=0.0), 4.0 * np.spacing(abs(r.lam))):
            return True
        if d > max(bars, default=0.0):
            return False
        # Within the WIDER bar only. A refiner bar can be wider than the gap to a genuine
        # neighbour: 65*pi^2, band 0.3, 1e-12, posterior tensions, a root landed BETWEEN the two
        # members of its bracket with a bar of ~0.5 licences, merged into a root 0.33 licences
        # away, and the group read its count in the OTHER root's bracket -- 2 of 4, silent, twice
        # (`notes/posterior_penalty.md`, "The 1e-12 singles"). So the wide bar merges only a
        # SHALLOWER root that sits on the deeper one's wall: its own reading, at the calibrated
        # bound, must reach the deeper root -- the side search's "found again" test. Where no slope
        # can be read the old reading stands.
        deep, shallow = (first, r) if first.sigma <= r.sigma else (r, first)
        lams_, svals_ = _samples(obj)
        c_deep = _slope_from_cache(lams_, svals_, deep.lam, floor=deep.sigma)
        if c_deep is None or not c_deep > 0:
            return True
        return d <= (shallow.sigma / c_deep) / BIAS_LO_CACHE

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
    queue = [(r, bracket, n_roots, False) for (r, bracket), n_roots in located]
    while queue:
        r, bracket, n_roots, recentred = queue.pop(0)
        c = _slope_from_cache(lams, svals, r.lam, floor=r.sigma)
        # The distance to the next eigenvalue, from the free `sigma_2` the search already has.
        # It bounds the wide bracket from ABOVE, so the flanks cannot reach a neighbour and
        # count its branch as one of this level's.
        s2 = obj.sigma(obj.from_lam(r.lam), 1)
        nb = (s2 / c) if (c > 0 and np.isfinite(s2)) else None
        wb = wide_bracket_for(lams, r.lam, r.sigma, c, prec=prec, neighbour=nb)
        tens = (lambda z: obj.tensions(obj.from_lam(z)))
        extra, uncertain, drop, claim_reason = [], None, False, ""
        count_decided_by = ""
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
            # (C2) BACKSTOP on every branch the reach rule COUNTED. A counted branch whose sigma
            # does not dip at this root is not this level's: the spike triangle's 807.0076 came
            # back x8 because a slope read inside a noisy floor made `sigma_2..8 = 0.22..0.83`
            # read as distances 0.005..0.017 (`_c2_holds`). Only a branch that FAILS is
            # uncounted; one that cannot be read keeps the rule's verdict.
            c2_cut = [b.index for b in m.branches
                      if b.verdict == "counted" and b.index > 0
                      and _c2_holds(tens, r.lam, wb, b.index) is False]
            if c2_cut:
                m = replace(m, mu=int(m.mu) - len(c2_cut), branches=tuple(
                    replace(b, verdict="c2_rejected") if b.index in c2_cut else b
                    for b in m.branches))
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
                finds = []
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
                budget, narrowed, post_find, resampled = _SIDE_SEARCH_MAX, {}, set(), set()
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
                        # A refusal at the INNER end is not a rejection when it reads too LOW for
                        # `r`'s own wall. The end sits `half = 0.5 sigma_j/c` from `r`, where that
                        # wall reads `0.5 sigma_j` times the slope's bias -- at least BIAS_LO_CACHE.
                        # Lower than that, the descent is into a well INSIDE the exclusion: 50*pi^2,
                        # band 0.3, prec 1e-12, a cache slope half the typical one put `half` at
                        # 0.32 licences, member 0 at 0.27 was inside it, and the inner end read
                        # 0.18 sigma_j. Counted as resolved, the total was 2 of 3, silent. Search
                        # the side once more with the exclusion `r`'s own reading implies -- the
                        # radius a find gets below -- and if that too stops at its inner end
                        # reading low, the side is undetermined.
                        inner = b2 if side < 0 else a
                        off_r = abs(inner - r.lam) <= half + 4.0 * np.spacing(abs(r.lam))
                        low_inner = (r2.reason == "not_a_minimum" and not explained and off_r
                                     and abs(r2.lam - inner) <= 0.01 * (b2 - a)
                                     and r2.sigma < 0.5 * BIAS_LO_CACHE * float(t_here[j]))
                        tries = narrowed.get(side, 0)
                        if low_inner and tries < 2:
                            narrowed[side] = tries + 1
                            # Two retries, tight first. The FIRST excludes only `r`'s own reach,
                            # which at a converged root is the floor: the retry runs up to `r`, and
                            # where `r`'s wall is all there is it converges ON `r` and the dedup
                            # resolves the side -- the reading every shipped path relied on
                            # (default pencil, 50*pi^2, 1e-6, band 1.0, where an over-read cache
                            # slope makes the inner end read falsely low). The SECOND, only if the
                            # first also stops low at its inner end, uses the standoff rule finds
                            # use: 50*pi^2, band 0.3, 1e-12, posterior tensions, the tight retry
                            # refused on `r`'s own wall and flagged a level whose member lay inside
                            # (`notes/posterior_penalty.md`, "The 1e-12 singles") -- the trap
                            # `_SIDE_STANDOFF` documents. Then the side is undetermined.
                            outer = a if side < 0 else b2
                            near = max((r.sigma / c_out) / bias_lo_eff,
                                       4.0 * np.spacing(abs(r.lam)))
                            if tries == 1:
                                near = max(near, _SIDE_STANDOFF * abs(r.lam - outer))
                            if near < half:
                                sides.insert(0, (a, r.lam - near, side) if side < 0
                                             else (r.lam + near, b2, side))
                                continue
                        # A sub-bracket that STARTS AT A FIND, refused at that inner end: the
                        # resume point can land on the hump between the find and the next member,
                        # whose well is inside and unsampled, and the end then reads exactly the
                        # find's wall, so nothing above can tell it from an empty side. 50*pi^2,
                        # band 0.3, 1e-12, posterior tensions: member 2 0.15 licences inside, 2 of
                        # 3, silent (`notes/posterior_penalty.md`). Sample the interior once --
                        # `refine` starts from the deepest cached sample -- and refine it again.
                        key = (a, b2)
                        if (key in post_find and key not in resampled
                                and r2.reason == "not_a_minimum"
                                and abs(r2.lam - inner) <= 0.01 * (b2 - a)):
                            resampled.add(key)
                            for z in a + (b2 - a) * np.arange(1, _RESAMPLE + 1) / (_RESAMPLE + 1):
                                tens(z)
                            sides.insert(0, (a, b2, side))
                            continue
                        inconclusive |= low_inner or (
                            r2.reason != "not_a_minimum" and not explained
                            and r2.sigma < SIDE_RATIO * float(t_here[j]))
                        continue
                    # Already a located root, from its own bracket -- within the NARROWER of the two
                    # bars, never the wider. One wide bar used to swallow a genuine neighbour: 50*pi^2,
                    # band 0.3, prec 1e-10, on posterior tensions (`notes/posterior_penalty.md`), the
                    # side search found all three members and discarded the third, 0.22 licences from
                    # the second, because the second's refiner bar was 0.24 licences. 2 of 3, silent.
                    # The same shape `r`'s own bar has, which is why `r` is not in `taken`. A true
                    # duplicate does not need the wide bar: re-found on the same well, its reading puts
                    # that well within its reach, and the check below discards it.
                    if any(abs(r2.lam - q.lam) <= max(
                            min(r2.delta_res if np.isfinite(r2.delta_res) else 0.0,
                                q.delta_res if np.isfinite(q.delta_res) else 0.0),
                            4.0 * np.spacing(abs(r2.lam))) for q in taken):
                        continue
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
                    finds.append(r2)
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
                        post_find.add((nxt[0], nxt[1]))
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
                # RE-CENTRE when `r` is a find's WALL. The refiner may accept a point off the
                # bottom whenever its bar covers the error, and the read taken there is a read of
                # the wrong point: the disk's double at 57.58, prec 1e-8, `rtol = 1.25
                # SUITE_RTOL`, accepted `r` 0.0245 licences up the wall at sigma 1.2e-9, counted
                # one, left the partner branch undecided -- and the side search found the true
                # root at sigma 1.5e-15, which that branch then owned. One exact double came back
                # as two simple levels, total right, the solver contract broken (Linux CI, the
                # same shape). `r`'s own reading puts the nearest eigenvalue within
                # `(sigma/c)/bias_lo`, so a DEEPER find inside that reach is what `r` was reading:
                # the find replaces `r`, and the whole read is redone there. Once per entry.
                reach_r = (r.sigma / c_out) / bias_lo_eff
                walls = [q for q in finds if q.sigma < r.sigma and abs(q.lam - r.lam) <= reach_r]
                if walls and not recentred:
                    q = min(walls, key=lambda q: q.sigma)
                    located = [(p, k) if p[0] is not r else ((q, bracket), k)
                               for p, k in located]
                    lams, svals = _samples(obj)
                    queue.insert(0, (q, bracket, n_roots, True))
                    continue
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
            n_claimed, _, claim_amb = _claim(m.branches, r.lam, claimants, bias_lo_eff,
                                             bias_hi_eff, with_ambiguity=True)
            if claim_amb:
                # A find both a counted and an undecided branch admit: the total is one of two
                # and nothing here decides which. Say so rather than guess (section 10.7).
                uncertain, claim_reason = True, (
                    "a side-search find is admissible to both a counted and an undecided branch, "
                    "so whether it adds to this level's count or duplicates it is not decided")
            mu = int(m.mu) - n_claimed
            assert mu >= 1, (f"attribution took the root's own branch: mu={m.mu}, "
                             f"claimed={n_claimed}")
            # A DECLINED branch the side search left UNACCOUNTED -- nothing found, and no located
            # root anywhere within its reach -- is either a copy AT this root or an eigenvalue
            # nobody located. The side search can only find the second kind; an exact double at
            # the floor is the first (the L-shape's `10 pi^2`: `sigma_2` at the root is its own
            # floor, its `d_est` lands at the bracket edge, both sides come back empty, and the
            # level was certified simple). (C2) tells them apart: a copy at the root makes
            # `sigma_j` dip HERE. Count it, and any branch above it that also dips; a branch that
            # does not dip, or cannot be read, leaves the count flagged rather than guessed.
            undecided = [b for b in m.branches if b.verdict == "undecided"]
            if m.ambiguous and undecided and not uncertain and found == 0:
                j0 = undecided[0].index
                reach_j = undecided[0].d_est / bias_lo_eff + 2.0 * width
                accounted = any(abs(q.lam - r.lam) <= reach_j for q in taken if q is not r)
                if not accounted:
                    j, added = j0, 0
                    t_x = np.atleast_1d(np.asarray(tens(r.lam), dtype=float))
                    while j < min(t_x.size, MAX_BRANCHES) and _c2_holds(tens, r.lam, wb, j):
                        added += 1
                        j += 1
                    if added:
                        mu += added
                        count_decided_by = "c2"
                    else:
                        uncertain, claim_reason = True, (
                            "a declined branch is unaccounted -- no root located within its reach "
                            "-- and does not dip at this root, so it is either a dropped "
                            "eigenvalue or the estimate was noise; not decided")
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
                                count_decided_by=count_decided_by,
                                # The estimator's own DECLINE, carried through. Dropping it made every
                                # wrong total under the reach rule silent on the wip branch -- 0 of 15
                                # flagged -- which was a wiring omission, not a property of the rule.
                                count_uncertain=(bool(m.ambiguous) if uncertain is None
                                                 else uncertain),
                                uncertain_reason=(
                                    claim_reason or
                                    "the multiplicity estimator declined: a branch's estimated "
                                    "distance landed where the inside and outside populations "
                                    "overlap, so the evidence does not decide whether that "
                                    "eigenvalue is in this bracket"
                                    if (m.ambiguous if uncertain is None else uncertain) else "")))
        levels.extend(extra)

    levels.sort(key=lambda x: x.lam)

    # PHASE D, before both checks below. The angle needs REPORTED ROOTS, the side search has
    # just created some, and `_flag_uncertain_counts` is the consumer whose doubt it removes --
    # so it has to run first or that doubt is reported after it was settled.
    if phase_d == "angle":
        gram_mark = obj.n_evals
        levels, n_gram_pairs, n_gram_coll, n_gram_sum = _gram_group(obj, levels, prec)
        mode_evals = obj.n_evals - gram_mark
    else:
        n_gram_pairs = n_gram_coll = n_gram_sum = mode_evals = 0

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
    mult_evals = (obj.n_evals - mark) - probe_evals - side_evals - mode_evals

    # A refused bracket that dipped as deep as a located root is a candidate LOST eigenvalue, and
    # it reaches `n_uncertain` so that `certified_counts` is False and `solve_interval` raises.
    # LAST, so the comparison sees the levels a caller actually receives rather than an
    # intermediate set that Phase D or the cross-check may still have folded.
    deep = _deep_rejections(refused, levels)

    # The request, on every level, so `margin`, `resolution` and `count_shortfall` read it.
    levels = [replace(x, prec=prec) for x in levels]
    unmet, attained = _precision_summary(levels, prec)

    return SolveResult(tuple(levels), SolveDiagnostics(
        search=sd, n_brackets=len(res.brackets), n_rejected=rejected, n_merged=merged,
        search_evals=search_evals, refine_evals=refine_evals,
        multiplicity_evals=mult_evals, probe_evals=probe_evals, side_evals=side_evals,
        evaluations=obj.n_evals, n_ambiguous=n_amb, n_gram_pairs=n_gram_pairs,
        n_gram_collapsed=n_gram_coll, n_gram_summed=n_gram_sum,
        rejected_deep=deep,
        n_uncertain=len(uncertain) + len(deep),
        uncertain=(tuple((x.lam, x.mult, x.count_alt, x.uncertain_reason) for x in uncertain)
                   + tuple((lam, 0, None,
                            f"the refiner REFUSED a bracket that reached sigma_1 = {sig:.3e}, no "
                            f"larger than the worst located level's, so the total may be short "
                            f"by this eigenvalue (refusal reason: {why})")
                           for lam, sig, why in deep)),
        prec_requested=prec, attained_prec=attained, unmet=unmet))
