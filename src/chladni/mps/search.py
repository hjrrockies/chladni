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
from typing import Optional

import numpy as np

from .bracket import bracket_interval
from .multiplicity import wide_reach
from .multiplicity import (BIAS_HI_CACHE, BIAS_LO_CACHE, COUNT_BELOW_CACHE, MAX_BRANCHES,
                           PROMINENCE, REJECT_ABOVE_CACHE, SIDE_RATIO, multiplicity,
                           wide_bracket_for)
from .objective import PolicyView
from .policy import CELL_RHO, PAIR_ETA, TAU_REL, Policy, stable_rank_of
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
    #: The rank the level was refined at, when a stage froze one (`chladni.mps.policy`); None
    #: for the pencil's own policy. Certification reads the coefficients at this rank.
    rank: Optional[int] = None

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
    #: `solve(stable_rank=True)` only: evaluations spent re-refining roots at their stable rank
    #: (the stability pair where the search had none, and the refine in each root's cell), and
    #: how many roots were re-refined / kept at the threshold because the re-refine refused.
    rerank_evals: int = 0
    n_reranked: int = 0
    n_rerank_refused: int = 0
    evaluations: int = 0
    declined: bool = False
    decline_reason: str = ""
    prec_requested: float = float("nan")
    attained_prec: float = float("nan")   # the worst over levels
    unmet: tuple = ()                     # levels whose accuracy fell short, if any
    #: The returned count against Weyl's law (`asymp.CountAudit`), attached by
    #: `MPSEigensolver.search_interval`, which holds the domain; `solve` sees only an objective
    #: and leaves it None. A flag, never an exception: it does not enter `certified_counts`.
    count_audit: object = None

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


def _abort_if_unmet(r, bracket, located, prec):
    """`abort_unmet`: raise `PrecisionShortfall` the moment a refined level misses `prec`.

    "As soon as possible" is after the first deficient level is refined
    (`docs/design/douse_requirements.md`): a level's floor is known only once it is located.
    Measured on the douse benchmark at 1e-10 (`notes/early_shortfall.md`), stopping here saves a
    median 57% of a failing solve's wall time -- the remaining refines, stage 3, and the
    Moler-Payne certification. The test is the refiner's own `delta_res > prec*lam` (margin < 2),
    before any Moler-Payne bound: on the 8062 benchmark levels a level with margin < 2 was never
    certified met afterwards, so stopping on it lost nothing measured."""
    if not (np.isfinite(r.delta_res) and r.delta_res > prec * abs(r.lam)):
        return
    from .._core import PrecisionShortfall

    L = Level(lam=float(r.lam), mult=1, sigma=float(r.sigma), delta_res=float(r.delta_res),
              bracket=tuple(bracket), prec=prec)
    raise PrecisionShortfall(
        f"stopped early: the level refined at lam = {r.lam:.10g} is unmet -- its resolution "
        f"{r.delta_res:.2e} exceeds prec*lam = {prec * abs(r.lam):.2e} (margin {L.margin:.3g}; "
        f"2 is needed). {len(located)} value(s) located so far. The retry is a richer basis at "
        f"the same prec.",
        level=L, margin=L.margin, located=[float(x.lam) for x, _ in located])


def _precision_summary(levels, prec):
    """`(unmet, attained)`: the levels short of `prec`, and the worst attained precision. One
    place, because `MPSEigensolver` recomputes it after attaching Moler-Payne bounds."""
    unmet = tuple(x.lam for x in levels if not x.met(prec))
    attained = max((x.attained_prec for x in levels
                    if np.isfinite(x.attained_prec)), default=prec)
    return unmet, float(attained)



# --- the solve, in named steps ----------------------------------------------------------------
#
# `solve` ran as one ~600-line function whose stage-3 loop held four retry mechanisms inline.
# It is now a driver over the steps below (2026-10-14); `tests/test_search_golden.py` replays a
# recorded trace of the rectangle suite, so behaviour is byte-identical to the loop it replaces.
# The measurements behind each step live in `docs/design/search.md` and `notes/`.


@dataclass(frozen=True)
class _Root:
    """One queue item of stage 3: a located root, the bracket it came from, how many roots
    were folded into it, and whether it has already been re-centred on a deeper wall find."""
    r: object
    bracket: tuple
    n_roots: int
    recentred: bool = False


@dataclass(frozen=True)
class _Side:
    """What a side search around one root found."""
    finds: tuple = ()            # RefineResults located beside the root
    extra: tuple = ()            # the Levels for them
    found: int = 0
    inconclusive: bool = False   # a refusal that neither found nor excluded a neighbour
    drop: bool = False           # the root sits on a hump between two finds: not a level
    evals: int = 0


@dataclass(frozen=True)
class _Counted:
    """Stage 3's answer for one queue item."""
    levels: tuple = ()           # Levels to append, in order
    recentre: object = None      # a `_Root` to put at the queue's head instead
    located: object = None       # the located list after a re-centre
    probe_evals: int = 0
    side_evals: int = 0


def _refine_stage(obj, res, prec, abort_unmet):
    """Stage 2: refine every bracket the search emitted, then re-examine the ends of the
    brackets the refiner refused as endpoints. `(located, refused)`: located is a list of
    `(RefineResult, (lo, hi))`.

    A bracket the search kept and the refiner threw away is usually not a failure of either:
    dropping a doubtful region costs an eigenvalue and keeping one costs about three
    evaluations. But a refusal AT AN END says the well is just outside: the end was this
    find's wall, so the adjacent gap (bounded by the neighbouring brackets and the window) is
    refined once more, and a find there explains the refusal."""
    located, refused, ends = [], [], []
    for b in res.brackets:
        lo, hi = obj.in_lam_bracket(b.lo, b.hi)
        hint = obj.in_lam(b.k_hint) if np.isfinite(b.k_hint) else None
        r = refine(obj, lo, hi, prec=prec, lam_hint=hint,
                   interior=(obj.in_lam(b.lo), obj.in_lam(b.hi)))
        if r.accepted:
            located.append((r, (lo, hi)))
            if abort_unmet:
                _abort_if_unmet(r, (lo, hi), located, prec)
        else:
            refused.append(r)
            ends.append((r, (lo, hi)))
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
            if abort_unmet:
                _abort_if_unmet(r_out, (a, b), located, prec)
            refused.remove(r)               # explained: the end was this find's wall
    return located, refused


def _same_point(obj, first, r, prec, views=None):
    """Are two located roots one level? Only a pair the refinement could not tell apart --
    within the attained bars, or, when the bars disagree, within the shallower root's reach
    down the deeper root's wall -- and never further apart than the licence `prec*lam`. The
    search never merges deliberately, only what it could not resolve. (The "licence" merge,
    every pair within the licence, is in the ablations.)"""
    licence = prec * max(abs(r.lam), 1.0)
    d = abs(r.lam - first.lam)
    if d > licence:
        return False
    bars = [x.delta_res for x in (first, r) if np.isfinite(x.delta_res)]
    if d <= max(min(bars, default=0.0), 4.0 * np.spacing(abs(r.lam))):
        return True
    if d > max(bars, default=0.0):
        return False
    deep, shallow = (first, r) if first.sigma <= r.sigma else (r, first)
    # The deep root's own samples: a root refined at a frozen rank has its wall in its view only.
    lams_, svals_ = _samples((views or {}).get(id(deep), (obj, None))[0])
    c_deep = _slope_from_cache(lams_, svals_, deep.lam, floor=deep.sigma)
    if c_deep is None or not c_deep > 0:
        return True
    return d <= (shallow.sigma / c_deep) / BIAS_LO_CACHE


def _group_roots(obj, located, prec, views=None):
    """Phase D's distance prefilter: fold roots that are one level into the deepest of them.
    `(located, merged)`, located now `((RefineResult, bracket), n_roots)`. Two located roots
    within `prec*lam` are one level of `S(prec)`, and reading multiplicity at each separately
    double-counts (measured on the sub-licence cluster: two roots each declared 3 against 3)."""
    groups = []
    for r, bracket in sorted(located, key=lambda p: p[0].lam):
        if groups and _same_point(obj, groups[-1][0][0], r, prec, views):
            groups[-1].append((r, bracket))
        else:
            groups.append([(r, bracket)])
    located = [(min(g, key=lambda p: p[0].sigma), len(g)) for g in groups]
    return located, sum(len(g) - 1 for g in groups)


def _side_search(obj, r, bracket, c_out, m, t_here, taken, prec, bias_lo_eff):
    """Look for the branches the multiplicity estimate could not place, either side of the
    root inside its bracket, each side a refine on the stretch beyond `half` (half the
    undecided branch's tension over the wall slope). Three retries live here, each bounded:

    * NARROWING (`narrowed`, at most twice per side): a refusal "not a minimum" at the inner
      end, unexplained by another root and too low to be the wall, says the branch sits
      closer than `half`; the side is re-posed from the root's own reach, then from a standoff.
    * RESAMPLING (`resampled`, once per stretch): a stretch posed after a find that refuses at
      its inner end is sampled at `_RESAMPLE` interior points and posed again, so a well the
      refiner's three starting points straddled is seen.
    * EXCLUSION after a find (`post_find`): the stretch beyond the find, past its reach and a
      standoff, is queued, so several branches on one side are found in turn.

    `taken` (the other roots, then every find) is appended to in place, so attribution sees
    the finds. A find that lands within another root's attained bar or reach is that root."""
    tens = (lambda z: obj.tensions(obj.from_lam(z)))
    side_mark = obj.n_evals
    j = min(int(m.mu), t_here.size - 1)
    half = 0.5 * float(t_here[j]) / c_out if j > 0 else float("inf")
    finds, extra = [], []
    found, inconclusive = 0, not np.isfinite(half)
    sides = [(bracket[0], r.lam - half, -1), (r.lam + half, bracket[1], +1)]
    budget, narrowed, post_find, resampled = _SIDE_SEARCH_MAX, {}, set(), set()
    while sides and budget > 0:
        a, b2, side = sides.pop(0)
        if not (np.isfinite(half) and a < b2):
            continue                    # too short to hold the branch: resolved
        budget -= 1
        r2 = refine(obj, a, b2, prec=prec, interior=(a, b2))
        if not r2.accepted:
            reach_side = (r2.sigma / c_out) / BIAS_LO_CACHE + 2.0 * (b2 - a)
            explained = any(abs(q.lam - r2.lam) <= reach_side for q in taken if q is not r)
            inner = b2 if side < 0 else a
            off_r = abs(inner - r.lam) <= half + 4.0 * np.spacing(abs(r.lam))
            low_inner = (r2.reason == "not_a_minimum" and not explained and off_r
                         and abs(r2.lam - inner) <= 0.01 * (b2 - a)
                         and r2.sigma < 0.5 * BIAS_LO_CACHE * float(t_here[j]))
            tries = narrowed.get(side, 0)
            if low_inner and tries < 2:
                narrowed[side] = tries + 1
                outer = a if side < 0 else b2
                near = max((r.sigma / c_out) / bias_lo_eff, 4.0 * np.spacing(abs(r.lam)))
                if tries == 1:
                    near = max(near, _SIDE_STANDOFF * abs(r.lam - outer))
                if near < half:
                    sides.insert(0, (a, r.lam - near, side) if side < 0
                                 else (r.lam + near, b2, side))
                    continue
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
        if any(abs(r2.lam - q.lam) <= max(
                min(r2.delta_res if np.isfinite(r2.delta_res) else 0.0,
                    q.delta_res if np.isfinite(q.delta_res) else 0.0),
                4.0 * np.spacing(abs(r2.lam))) for q in taken):
            continue
        reach_find = (r2.sigma / c_out) / BIAS_LO_CACHE
        if any(abs(r2.lam - q.lam) <= reach_find for q in taken):
            continue
        taken.append(r2)
        finds.append(r2)
        found += 1
        extra.append(Level(lam=r2.lam, mult=1, sigma=r2.sigma,
                           delta_res=r2.delta_res, bracket=(a, b2), n_roots=1))
        span = (r2.lam - a) if side < 0 else (b2 - r2.lam)
        excl = max((r2.sigma / c_out) / bias_lo_eff, _SIDE_STANDOFF * span,
                   4.0 * np.spacing(abs(r2.lam)))
        nxt = (a, r2.lam - excl) if side < 0 else (r2.lam + excl, b2)
        if nxt[1] - nxt[0] > 4.0 * np.spacing(abs(r2.lam)):
            sides.append((nxt[0], nxt[1], side))
            post_find.add((nxt[0], nxt[1]))
    drop = False
    if found:
        # Two finds either side of the root: is the root a well of its own or a HUMP between
        # them? Read the tension three quarters of the way towards each; a root higher than
        # both by `_HUMP_RATIO` is the barrier, not a level.
        near = [q.lam for q in taken if q is not r and bracket[0] <= q.lam <= bracket[1]]
        below = [q for q in near if q < r.lam]
        above = [q for q in near if q > r.lam]
        if below and above:
            towards = (r.lam + 0.75 * (max(below) - r.lam), r.lam + 0.75 * (min(above) - r.lam))
            reads = [float(np.atleast_1d(tens(z))[0]) for z in towards]
            drop = max(reads) < _HUMP_RATIO * r.sigma
    return _Side(finds=tuple(finds), extra=tuple(extra), found=found,
                 inconclusive=inconclusive, drop=drop, evals=obj.n_evals - side_mark)


def _attribute(obj, m, r, bracket, width, extra, taken, found, uncertain, bias_lo_eff,
               bias_hi_eff, wb):
    """Attribution: which of the estimate's counted branches are the side finds, so a find is
    not counted twice; then the (C2) backstop for a branch the estimate declined that no find
    accounts for -- it is counted only if the tension at the root dips as a branch there
    would. `(mu, uncertain, claim_reason, count_decided_by)`."""
    tens = (lambda z: obj.tensions(obj.from_lam(z)))
    claim_reason, count_decided_by = "", ""
    claimants = [q.lam for q in extra if bracket[0] <= q.lam <= bracket[1]]
    n_claimed, _, claim_amb = _claim(m.branches, r.lam, claimants, bias_lo_eff,
                                     bias_hi_eff, with_ambiguity=True)
    if claim_amb:
        uncertain, claim_reason = True, (
            "a side-search find is admissible to both a counted and an undecided branch, "
            "so whether it adds to this level's count or duplicates it is not decided")
    mu = int(m.mu) - n_claimed
    assert mu >= 1, (f"attribution took the root's own branch: mu={m.mu}, "
                     f"claimed={n_claimed}")
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
    return mu, uncertain, claim_reason, count_decided_by


def _count_root(obj, item, lams, svals, located, prec):
    """Stage 3 for one queue item: read the multiplicity at the root. Under the reach rule the
    estimate may leave branches undecided; a side search then looks for them, a deeper find
    within the root's own reach RE-CENTRES the root on it (once), and attribution settles the
    count. Spends from the cache where it can; `probe_evals` and `side_evals` say what it
    paid beyond that."""
    r, bracket, n_roots, recentred = item.r, item.bracket, item.n_roots, item.recentred
    c = _slope_from_cache(lams, svals, r.lam, floor=r.sigma)
    s2 = obj.sigma(obj.from_lam(r.lam), 1)
    nb = (s2 / c) if (c > 0 and np.isfinite(s2)) else None
    wb = wide_bracket_for(lams, r.lam, r.sigma, c, prec=prec, neighbour=nb)
    tens = (lambda z: obj.tensions(obj.from_lam(z)))
    width = abs(bracket[1] - bracket[0])
    c_out = c
    if c_out is None or not (c_out > 0):
        return _Counted(levels=(Level(
            lam=r.lam, mult=1, sigma=r.sigma, delta_res=r.delta_res, bracket=bracket,
            wide_bracket=wb or (), n_roots=n_roots, count_uncertain=True,
            uncertain_reason="the reach rule could not read a wall slope (cache) from the "
                             "cache at this root"),))
    bounds = (COUNT_BELOW_CACHE, REJECT_ABOVE_CACHE, BIAS_LO_CACHE, BIAS_HI_CACHE)
    bias_lo_eff, bias_hi_eff = BIAS_LO_CACHE, BIAS_HI_CACHE
    m = multiplicity(tens, r.lam, bracket=bracket, c=c_out, prec=prec, rule="reach",
                     reach_bounds=bounds, counter=obj,
                     window=obj.in_lam_bracket(obj.lo, obj.hi))
    probe_evals = int(m.evaluations)
    c2_cut = [b.index for b in m.branches
              if b.verdict == "counted" and b.index > 0
              and _c2_holds(tens, r.lam, wb, b.index) is False]
    if c2_cut:
        m = replace(m, mu=int(m.mu) - len(c2_cut), branches=tuple(
            replace(b, verdict="c2_rejected") if b.index in c2_cut else b
            for b in m.branches))
    extra, uncertain, drop, found, side_evals = (), None, False, 0, 0
    taken = [p[0][0] for p in located if p[0][0] is not r]
    if m.ambiguous:
        t_here = np.atleast_1d(np.asarray(tens(r.lam), dtype=float))
        side = _side_search(obj, r, bracket, c_out, m, t_here, taken, prec, bias_lo_eff)
        extra, found, drop, side_evals = list(side.extra), side.found, side.drop, side.evals
        uncertain = side.inconclusive and found == 0
        reach_r = (r.sigma / c_out) / bias_lo_eff
        walls = [q for q in side.finds if q.sigma < r.sigma and abs(q.lam - r.lam) <= reach_r]
        if walls and not recentred:
            # A DEEPER WELL WITHIN THE ROOT'S OWN REACH is the level; the root was a shoulder
            # of it. Re-centre once: the located list carries the find in the root's place, the
            # cache is re-read, and the item goes back to the head of the queue.
            q = min(walls, key=lambda q: q.sigma)
            new_located = [(p, k) if p[0] is not r else ((q, bracket), k) for p, k in located]
            return _Counted(recentre=_Root(q, bracket, n_roots, True), located=new_located,
                            probe_evals=probe_evals, side_evals=side_evals)
    mu, uncertain, claim_reason, count_decided_by = _attribute(
        obj, m, r, bracket, width, extra, taken, found, uncertain, bias_lo_eff, bias_hi_eff, wb)
    m = replace(m, mu=mu)
    levels = []
    if not drop:
        levels.append(_level_for(r, bracket, wb, m, n_roots, uncertain, claim_reason,
                                 count_decided_by))
    levels.extend(extra)
    return _Counted(levels=tuple(levels), probe_evals=probe_evals, side_evals=side_evals)


def _level_for(r, bracket, wb, m, n_roots, uncertain, claim_reason, count_decided_by):
    amb = bool(m.ambiguous) if uncertain is None else uncertain
    return Level(lam=r.lam, mult=m.mu, sigma=r.sigma, delta_res=r.delta_res,
                 bracket=bracket, wide_bracket=wb or (), gap_ratio=m.gap_ratio,
                 c2_rejected=m.c2_rejected, n_roots=n_roots,
                 count_decided_by=count_decided_by,
                 count_uncertain=amb,
                 uncertain_reason=(
                     claim_reason or
                     "the multiplicity estimator declined: a branch's estimated "
                     "distance landed where the inside and outside populations "
                     "overlap, so the evidence does not decide whether that "
                     "eigenvalue is in this bracket"
                     if amb else ""))


def _stable_rank_at(obj, lam):
    """The stable rank at a located root, or None (see `_stable_rank_near`)."""
    if not hasattr(obj, "spectra_near"):
        return None
    return _stable_rank_near(obj, lam)[0]


def _place_flanks(view, r, cell, prec):
    """Make sure the frozen view has a sample on each side of the root at the (C2) reach.

    Stage 3 reads the multiplicity at the root's frozen rank, from samples in its cell only. The
    re-refine leaves samples right at the root, inside the flat bottom, where (C2) would read the
    floor against itself; `wide_bracket_for` then falls back to them and under-reads. So where a
    side has no sample at `wide_reach` (and inside the cell), ONE evaluation is placed there: at
    most two per re-refined root, charged to `rerank_evals`."""
    lams, vs = _samples(view)
    c = _slope_from_cache(lams, vs, r.lam, floor=r.sigma)
    if not c > 0:
        return
    s2 = view.sigma(view.from_lam(r.lam), 1)
    reach = wide_reach(r.lam, r.sigma, c, prec=prec,
                       neighbour=(s2 / c) if np.isfinite(s2) else None)
    for side in (-1.0, 1.0):
        have = lams[(side * (lams - r.lam)) >= reach] if lams.size else lams
        if have.size:
            continue
        z = r.lam + side * 1.25 * reach
        z = min(max(z, cell[0]), cell[1])
        if side * (z - r.lam) > 0:
            view.tensions(view.from_lam(z))


#: Bracketing on the coarse penalty is asked for `max(prec, COARSE_PREC_FACTOR * tau_rel)`: its
#: floor sits near `tau` (relative), so below about ten times that the floor, not `prec`, binds and
#: the cover would only stall (`bracket.STALL_LIMIT`). Brackets narrower than the coarse floor can
#: resolve hold a cluster for stages 2 and 3, which read it at the fine rank.
COARSE_PREC_FACTOR = 10.0


def _threshold_rank_of(obj, spectrum):
    """The threshold rank a spectrum implies, at the pencil's own cutoff; None without a pencil."""
    pencil = getattr(obj, "pencil", None)
    rtol = getattr(pencil, "rtol", None)
    if rtol is None or spectrum is None or not spectrum.size or not spectrum[0] > 0:
        return None
    return int(np.sum(spectrum >= rtol * spectrum[0]))


def _close_spectra(obj, lam):
    """The recorded spectra within `PAIR_ETA` (relative) of `lam`, ordered by `lam`. Where fewer
    than two are recorded, ONE evaluation at `lam (1 + 1e-8)` adds one, charged to the caller's
    stage (the spectrum is the stack's: the policy it is read under does not change it)."""
    x = obj.from_lam(lam)
    radius = abs(obj.from_lam(lam * (1.0 + PAIR_ETA)) - x)
    near = obj.spectra_near(x, radius)
    if len(near) < 2:
        PolicyView(obj, Policy.penalty()).tensions(obj.from_lam(lam * (1.0 + 1e-8)))
        near = obj.spectra_near(x, radius)
    return sorted(near, key=lambda p: p[0])


def _stable_rank_near(obj, lam):
    """`(rank, cap)`: the stable rank near `lam` -- the MEDIAN movement per index over every close
    pair (`policy.stable_rank_of`) -- at most `cap`, the threshold rank there; rank None where it is
    the threshold rank (no tail: nothing to freeze) or cannot be read."""
    near = _close_spectra(obj, lam)
    if len(near) < 2:
        return None, None
    caps = [c for c in (_threshold_rank_of(obj, sp) for _, sp in near) if c]
    cap = min(caps) if caps else None
    r = stable_rank_of([sp for _, sp in near], cap)
    return (r if (cap is None or r < cap) and r > 0 else None), cap


def _refine_coarse(obj, res, prec):
    """Stage 2 after COARSE bracketing (`solve(coarse_bracket=True)`): each bracket is refined at
    the stable rank read at its coarse minimum, frozen over the bracket (a bracket holding one
    level is wide on purpose, but the stable rank drifts by a few over a factor of three in lam,
    so freezing it across ~1e-2 costs nothing), through a view confined to the bracket. Where there
    is no tail, or the frozen refine refuses, the shipped threshold refine runs on the bracket.
    `(located, views, refused)`, `views[id(root)] = (PolicyView, rank)`."""
    located, refused, views = [], [], {}
    for b in res.brackets:
        lo, hi = obj.in_lam_bracket(b.lo, b.hi)
        hint = obj.in_lam(b.k_hint) if np.isfinite(b.k_hint) else 0.5 * (lo + hi)
        rank, _ = _stable_rank_near(obj, hint)
        if rank is not None:
            view = PolicyView(obj, Policy.frozen(rank), cell=(lo, hi))
            r = refine(view, lo, hi, prec=prec, lam_hint=hint, interior=(lo, hi))
            if r.accepted:
                _place_flanks(view, r, (lo, hi), prec)
                views[id(r)] = (view, rank)
                located.append((r, (lo, hi)))
                continue
        r = refine(obj, lo, hi, prec=prec, lam_hint=hint, interior=(lo, hi))
        if r.accepted:
            located.append((r, (lo, hi)))
        else:
            refused.append(r)
    return located, views, refused


def _rerank(obj, located, prec):
    """Stage 2b, `solve(stable_rank=True)`: re-refine each located root whose cell has a noise
    tail at the STABLE rank, frozen in that cell. `(located, views, refused)`: the located list
    with re-refined roots in place, `views[id(root)] = (PolicyView, rank)` for each re-refined
    root, and how many re-refines refused (those roots keep their threshold result)."""
    views, out, refused = {}, [], 0
    for (r, bracket), n_roots in located:
        rank = _stable_rank_at(obj, r.lam)
        if rank is None:
            out.append(((r, bracket), n_roots))
            continue
        cell = (r.lam * (1.0 - CELL_RHO), r.lam * (1.0 + CELL_RHO))
        view = PolicyView(obj, Policy.frozen(rank), cell=cell)
        a, b = max(bracket[0], cell[0]), min(bracket[1], cell[1])
        r2 = refine(view, a, b, prec=prec, lam_hint=r.lam, interior=(a, b))
        if r2.accepted:
            _place_flanks(view, r2, cell, prec)
            views[id(r2)] = (view, rank)
            out.append(((r2, bracket), n_roots))
        else:
            refused += 1
            out.append(((r, bracket), n_roots))
    return out, views, refused


def solve(obj, *, prec=None, abort_unmet=False, stable_rank=False, coarse_bracket=False):
    """Locate every eigenvalue in the objective's window, with its multiplicity.

    All three stages share one `Objective`, so the cost is the union of what they sampled and
    not the sum -- stage 3 in particular spends nothing, and that is asserted rather than hoped.

    ONE configuration: attained merge (roots are folded into one level only when the
    refinement could not tell them apart, capped at the licence `prec*lam`), the reach rule on
    the cache slope (the multiplicity is the number of eigenvalues INSIDE the emitted bracket,
    the wall slope read from the nearest cached samples), and Phase D's distance prefilter.
    Promoted on held-out cluster grids against the previous defaults (licence merge, the gap
    rule, the grouping cross-check):

        grid                        these defaults                 previous defaults
        145*pi^2 (held out, 45)     24 right, 0 wrong, 21 declined  11 right, 13 wrong, 21 declined
        125*pi^2 (45)               45 right (1 flagged), 0 wrong   26 right, 19 wrong (1 silent)
        50/65*pi^2 (110)            103 right, 2 basis-excused      72 right, 33 wrong (1 silent)
        precision ladder            every cell acceptable           2 wrong totals at 1e-4

    The alternatives (`collapse`, `merge="licence"`, `rule="gap"`/`"floor"`,
    `reach_slope="outside"`, `phase_d="angle"`) are EVIDENCE, not options: they live in
    `notes/evidence_tests/search_ablations.py` as `solve_with_options`, runnable against the
    current stages, so the table above can be regenerated (moved 2026-10-14).

    `abort_unmet` raises `PrecisionShortfall` at the first refined level that misses `prec`
    (`_abort_if_unmet`), which the planner's ladder climb uses.

    `coarse_bracket=True` (stage-policy step 5, OFF by default, 2026-10-27): stage 1 brackets on
    the coarse PENALTY (`Policy.penalty()`, no rank decision, so continuous in lam), asked for
    `max(prec, COARSE_PREC_FACTOR * tau_rel)`; stage 2 refines each bracket at the stable rank read
    at its coarse minimum, frozen over the bracket (`_refine_coarse`); stage 3 reads each root at
    its rank. Roots without a tail take the shipped threshold path. Implies the step-4 behaviour.

    `stable_rank=True` (stage-policy step 4, OFF by default, 2026-10-27): after the refine stage,
    each root whose cell has a noise tail -- the threshold rank above the stable rank read from
    two recorded spectra -- is re-refined in its cell `lam (1 +/- CELL_RHO)` with the rank FROZEN
    at the stable rank, and its multiplicity is read at that rank only; its levels carry
    `Level.rank`. Roots without a tail are untouched. With it on, `abort_unmet` is applied after
    the re-refine, since the threshold's verdict is the one being replaced.
    """
    prec = obj.prec if prec is None else float(prec)

    mark = obj.n_evals
    if coarse_bracket:
        coarse = PolicyView(obj, Policy.penalty())
        res = bracket_interval(coarse, prec=max(prec, COARSE_PREC_FACTOR * TAU_REL))
    else:
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
    views = {}
    if coarse_bracket:
        located, views, refused = _refine_coarse(obj, res, prec)
    else:
        located, refused = _refine_stage(obj, res, prec, abort_unmet and not stable_rank)
    rejected = len(refused)
    refine_evals = obj.n_evals - mark
    located, merged = _group_roots(obj, located, prec, views)

    rerank_evals, n_reranked, n_rerank_refused = 0, len(views), 0
    if coarse_bracket and abort_unmet:
        done = []
        for (r, bracket), _ in located:
            done.append((r, bracket))
            _abort_if_unmet(r, bracket, done, prec)
    if stable_rank and not coarse_bracket:
        mark = obj.n_evals
        located, views, n_rerank_refused = _rerank(obj, located, prec)
        rerank_evals, n_reranked = obj.n_evals - mark, len(views)
        if abort_unmet:
            done = []
            for (r, bracket), _ in located:
                done.append((r, bracket))
                _abort_if_unmet(r, bracket, done, prec)

    mark = obj.n_evals
    lams, svals = _samples(obj)
    levels, probe_evals, side_evals = [], 0, 0
    queue = [_Root(r, bracket, n_roots) for (r, bracket), n_roots in located]
    while queue:
        item = queue.pop(0)
        view, rank = views.get(id(item.r), (None, None))
        if view is None:
            out = _count_root(obj, item, lams, svals, located, prec)
        else:
            out = _count_root(view, item, *_samples(view), located, prec)
        probe_evals += out.probe_evals
        side_evals += out.side_evals
        if out.recentre is not None:
            located = out.located
            lams, svals = _samples(obj)
            if view is not None:
                views[id(out.recentre.r)] = (view, rank)
            queue.insert(0, out.recentre)
            continue
        levels.extend(out.levels if rank is None
                      else (replace(x, rank=rank) for x in out.levels))
    levels.sort(key=lambda x: x.lam)

    levels, uncertain = _flag_uncertain_counts(levels, folds_are_duplicates=True)
    mult_evals = (obj.n_evals - mark) - probe_evals - side_evals
    deep = _deep_rejections(refused, levels)
    levels = [replace(x, prec=prec) for x in levels]
    unmet, attained = _precision_summary(levels, prec)
    return SolveResult(tuple(levels), SolveDiagnostics(
        search=sd, n_brackets=len(res.brackets), n_rejected=rejected, n_merged=merged,
        search_evals=search_evals, refine_evals=refine_evals,
        multiplicity_evals=mult_evals, probe_evals=probe_evals, side_evals=side_evals,
        rerank_evals=rerank_evals, n_reranked=n_reranked, n_rerank_refused=n_rerank_refused,
        evaluations=obj.n_evals, rejected_deep=deep,
        n_uncertain=len(uncertain) + len(deep),
        uncertain=(tuple((x.lam, x.mult, x.count_alt, x.uncertain_reason) for x in uncertain)
                   + tuple((lam, 0, None,
                            f"the refiner REFUSED a bracket that reached sigma_1 = {sig:.3e}, no "
                            f"larger than the worst located level's, so the total may be short "
                            f"by this eigenvalue (refusal reason: {why})")
                           for lam, sig, why in deep)),
        prec_requested=prec, attained_prec=attained, unmet=unmet))
