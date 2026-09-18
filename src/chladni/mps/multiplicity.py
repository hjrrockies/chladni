"""Stage 3: how many eigenvalues sit at this level, read from evidence already paid for.

Promoted from `tests/v1_multiplicity.py`. `docs/design/candidates.md` section 1 is the seam,
`search.md` section 5 (Phase C) is the design and section 2.3 is the definition.

    multiplicity(tensions, x, *, wide_bracket, prec) -> int

# (C2) is the definition, (C1) is the test, and they are not interchangeable

`search.md` section 2.3 defines the multiplicity of a level as the largest `mu` such that
`sigma_j` has a local minimum there for every `j <= mu`. That is (C2), and it is the whole
definition. But exact local-minimality is not decidable from samples, so an implementation also
needs (C1), the ratio-gap test -- and the two must not be confused when the definition is being
reasoned about.

**(C1) alone double-counts, and the regime it fails in is entered by having a mediocre basis.**
Near a simple eigenvalue whose neighbour is close, the second branch is the OTHER cone's
descending wall: small, and therefore passing any gap test, but monotone rather than minimal.
Measured by starving the basis to raise `eps`:

    directions   d          eps       d/(eps/c)   s2/s1 at k1   (C1) reads   (C2) rejects
        10       1.01e-04   2.23e-04     0.45         1.07        2  WRONG      yes -> 1
        10       3.37e-05   2.23e-04     0.15         1.01        2  WRONG      yes -> 1
        12       1.01e-04   4.37e-06    23.17        19.76        1  correct    yes

The transition sits exactly where predicted, at `d ~ eps/c`. Since `eps` is a property of the
basis, **basis quality and multiplicity correctness are the same problem** -- a black-box solver
whose planner produces a merely adequate basis is in the dangerous regime by default. The
`gap > 20 * eps/c` contract of section 2.2 is what keeps a solver out of it.

# Why a ratio gap and not a threshold

Because no absolute threshold on a tension can serve: the floor varies FIVE ORDERS within a
single scan on a corner domain. The gap is enormous where it matters -- measured on the disk and
equilateral triangle, within a multiplet `sigma_{j+1}/sigma_j` runs 1.0 to 1.7 while across the
gap it runs 1e11 to 1e15, so `G` is not a delicate parameter and anything from about 3 to 1e10
would have done. Nine levels, nine correct readings, covering both mechanisms: the disk's
degeneracies are geometric (sin/cos pairs), the triangle's arithmetic (`m^2+mn+n^2` collisions).

# The bracket has to be WIDE, and that is not a detail

Phase C needs points where the higher branches have actually MOVED. The bracket a locator
returns is far too tight for that -- it is `prec*lam` wide by construction -- and using it is
why an earlier attempt's multiplicity test was reading noise. So the pair of evaluated points
either side of the root must sit at a distance of order `eps/c`, from before the bracket
collapsed, and `wide_bracket_for` picks them out of the evaluation cache.

# It spends no evaluations

Every point it reads was paid for by the search. Given a shared `Objective` the three reads are
cache hits, and `n_evals` does not move -- which is asserted rather than assumed, because a
multiplicity stage that quietly costs a GSVD per level would change the cost model it is
scored under.

# A real degeneracy is not bit-identical

`sigma_1` and `sigma_2` at a genuine degeneracy are numerically coincident but distinct: 6.1e-16
against 1.0e-15 at the disk's `j_1,1`. The surrogate's `mult` produces exactly EQUAL values, so
it must not be used to validate an equality test that real data would fail -- and more
generally, tier 1 cannot decide anything here, since it puts every non-well curve at the cap and
so has no `sigma_3` structure at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["multiplicity", "reach_multiplicity", "floor_multiplicity", "wide_bracket_for",
           "Multiplicity", "Branch", "GAP_RATIO", "PROMINENCE", "WIDE_FACTOR", "MAX_BRANCHES",
           "FLOOR_RATIO", "FLOOR_BAND", "COUNT_BELOW", "REJECT_ABOVE", "SLOPE_STANDOFF",
           "BIAS_LO", "BIAS_HI", "COUNT_BELOW_CACHE", "REJECT_ABOVE_CACHE", "BIAS_LO_CACHE",
           "BIAS_HI_CACHE", "TAU_GRAM"]

#: (C1). The ratio `sigma_{j+1}/sigma_j` that separates "still inside the multiplet" from
#: "across the gap". Measured 1.0-1.7 within a multiplet against 1e11-1e15 across it, so the
#: populations are separated by ten orders and this is not a tuned value -- anything from about
#: 3 to 1e10 reproduces the same nine readings. Dimensionless, and its margin GROWS as `prec`
#: tightens, since the across-gap ratio is `O(1)/eps`: it satisfies `search.md` section 2.1's
#: rule for a constant that may be fixed at all.
GAP_RATIO = 10.0

#: (C2). How far below its flanking samples a branch must sit to count as having a local
#: minimum here. Prominence, not a bare `<`: roughness minima reach `1 + 2.4e-10` while a true
#: well measured across its width reaches `plateau/eps`, worst case 72 -- ten orders apart, so
#: `>` alone would count noise as a branch. `search.md` section 2.3 gives `P = 1 + m*eta` with a
#: floor of 2; the floor is what is used here, since `eta` is per-solve and the margin is vast.
PROMINENCE = 2.0

#: Multiples of `eps/c` at which the flanking samples must sit. `search.md` Phase B says "of
#: order `eps/c`", and 2 is where two opposing constraints meet.
#:
#: **From below**, a genuine branch has to clear the prominence bar. On a cone, `sigma` at
#: distance `f*eps/c` from the minimum is `eps*sqrt(1 + f**2)`, so the prominence a true branch
#: shows is `sqrt(1 + f**2)` and passing `P = 2` needs `f > sqrt(3) = 1.73`. Below that the rule
#: would reject the multiplets it exists to count.
#:
#: **From above**, the flanks have to stay close enough for `sigma_2`'s asymmetry to show in the
#: regime where (C1) double-counts. Measured on a deliberately starved 10-direction basis, the
#: multiplicity read at a SIMPLE eigenvalue with an unresolved neighbour:
#:
#:     d/(eps/c)   0.15   0.42   0.86   2.39   18.1      truth is 1 throughout
#:     f = 1          1      1      1      1      1
#:     f = 2          2      1      1      1      1
#:     f = 3          2      2      2      1      1
#:     f = 5          2      2      2      1      1
#:
#: So smaller is safer here and `f = 1` would be safer still -- but it puts a genuine branch at
#: prominence 1.41 against a bar of 2. The band where 1 and 2 differ is `d/(eps/c) < 0.42`,
#: which is **fifty times inside** the `gap > 20*eps/c` adequacy contract of `search.md` section
#: 2.2. That contract is what keeps a solver out of this regime, which is section 9.4's own
#: conclusion: basis quality and multiplicity correctness are the same problem.
#:
#: On an adequate pencil the factor is inert. `eps/c` there is ~1e-15 and no search samples that
#: finely, so `wide_bracket_for` always falls back to the nearest sampled points and every
#: factor from 1 to 10 gives identical readings on the disk, the equilateral triangle and the
#: rectangle degeneracies -- 14 for 14.
WIDE_FACTOR = 2.0

#: The FLOOR-COINCIDENCE ratio: branch `j` is part of this level when `sigma_j <= R*sigma_1`.
#:
#: `bracket_search/search/isolate.py`'s question, which is not the question (C1) asks. (C1) takes
#: the LARGEST ratio gap in the low spectrum, and the largest gap is the one up to the PLATEAU --
#: so given `sigma = [1e-15, 1e-10, 0.4]` it puts the boundary after the second entry and counts a
#: NEIGHBOUR's descending wall as part of the multiplet. "At the floor with `sigma_1`" cannot make
#: that mistake, and it needs no flanking samples, so the wide bracket whose reach lands on the
#: next level is not required at all.
#:
#: **The value below is NOT honestly calibrated, and the number should not be trusted.** It was
#: fixed from two populations, one of which was mislabelled: the "simple level with a close
#: neighbour" set was drawn from crossover pairs at bands 0.5 and 0.8 as well as 1.3 and 2.0, and
#: the first two are INSIDE the merge licence -- one level of multiplicity 2, whose second branch
#: must be COUNTED rather than rejected. Split properly the two populations overlap almost
#: entirely, [12.9, 2.0e+15] against [29.5, 1.8e+15], so no ratio on `sigma_2/sigma_1` separates
#: them at all.
#:
#: What that overlap actually shows is a scale error, and it is the useful part: the ratio
#: measures a distance in units of the FLOOR `eps`, while the merge licence is in units of
#: `prec*lam`. They are not comparable, so a ratio test cannot answer "is the next eigenvalue
#: inside the licence" however the constant is chosen. The dimensionally right test compares a
#: DISTANCE to a distance -- `sigma_j/c` against `prec*lam`, using the identity
#: `sigma_2/c ~ dist to the next minimizer` that `bracket_search/search/isolate.py` measured
#: accurate to 0.98-1.08 exactly where the neighbour is close. An attempt to verify that here
#: also measured the wrong population -- it took the two SMALLEST eigenvalues of the window
#: rather than the dialled pair, which `rect_suite.tightest_pair` warns about by name -- so it
#: is a proposal, not a result.
#:
#: The populations that ARE sound, both taken at true eigenvalues on domains with closed-form
#: spectra, say the mechanism separates COINCIDENCE cleanly:
#:
#:     sigma_m/sigma_1 at a genuine multiplet, m = 2, 3, 4      1.04 to 1.40   (n = 10)
#:     the next branch up, which must be rejected               >= 2.3e+10
#:
#: and coincidence is what a floor test answers. It is NOT what a level's multiplicity asks,
#: which is "how many eigenvalues does this level stand for" -- and that is why turning this rule
#: on fixes every band at or above 0.8 licences and under-counts below it.
FLOOR_RATIO = 4.0

#: Half-width of the band around `FLOOR_RATIO`, in factors, where the measured populations put
#: nothing. A branch landing inside it is off the manifold this constant was calibrated on, and
#: the count is reported as uncertain rather than guessed. `[2, 8]` against IN <= 1.40 and
#: OUT >= 12.9.
FLOOR_BAND = 2.0

#: (REACH RULE) `d_est/reach` at or below which a branch's eigenvalue is certainly INSIDE the
#: bracket, and at or above which it is certainly outside. Between them the estimator DECLINES.
#:
#: `sigma_j/c` estimates the distance from `lam*` to the `j`-th nearest eigenvalue -- the identity
#: `bracket_search/search/isolate.py` relies on. Measured here in units of `reach`, over 143
#: brackets spanning `prec` 1e-6 to 1e-14 and multiplicities 1 to 4:
#:
#:     eigenvalues INSIDE the bracket    n=118   max 1.555   p95 1.206
#:     the nearest one OUTSIDE it        n=143   min 1.515   p05 1.811
#:
#: The estimate is biased HIGH -- a true distance reads about 19% larger than it is, up to 63% --
#: so the crossing sits near 1.5 rather than at 1.0, and the two populations OVERLAP in
#: `[1.515, 1.555]`. That overlap is why there are two constants and a decline between them
#: rather than one threshold: inside that band the evidence does not determine the answer, and a
#: single cut would have to be wrong on one population or the other.
#:
#: Both values are the measured bounds, not a fit: 1.5 sits below every observed OUTSIDE distance
#: and 1.6 above every observed INSIDE one.
COUNT_BELOW = 1.5
REJECT_ABOVE = 1.6

#: Bounds on the bias of `sigma_j/c` as a distance estimate: `d_est = beta * d_true` with `beta`
#: in this range. Measured over 88 (branch, eigenvalue) pairs inside brackets spanning `prec`
#: 1e-6 to 1e-14: min 0.888, median 1.186, max 1.625. It is biased HIGH, which is the direction
#: that excludes eigenvalues genuinely inside a bracket.
#:
#: These turn a probe's verdict into a decidable one. Knowing the SIDE, the true distance lies in
#: `[d/BIAS_HI, d/BIAS_LO]`, so an eigenvalue is certainly inside when even the far end of that
#: interval fits, certainly outside when even the near end does not, and undetermined between --
#: which is a decline rather than a guess.
BIAS_LO = 0.89
BIAS_HI = 1.63
#: A side probe at `x -+ d` HOLDS branch `j` when it reads below this fraction of `sigma_j`. A
#: sigma_1 probe reads the NEAREST eigenvalue on its side: a side holding nothing reads the
#: root's own wall, about `sigma_j` (measured 0.993), and a side holding the branch reads the
#: gap between estimate and truth (measured 0.096 to 0.121). The calibrated bias bounds put the
#: second at most `|1 - 1/beta|`, 0.64 at their extremes. A slope error that carries either
#: reading across the bar lands in "both sides" or "neither", which declines -- the safe way.
SIDE_RATIO = 0.5

#: Multiples of the bracket width at which the wall slope is estimated, and it must be OUTSIDE.
#:
#: Estimating `c` from points inside `B` is wrong whenever `B` holds more than one eigenvalue:
#: the other members depress `sigma_1` at the flanks, `c` comes out too small, and every distance
#: `sigma_j/c` too large. Measured on a pair 0.3 licences apart, that inflated `d/reach` from a
#: true 0.6 to 1.156 -- reading as "outside the bracket" and under-counting. Taking the flanks
#: outside took the rule from 82.5% to 96.5% on the ladder, which is the largest single effect
#: measured in its construction.
SLOPE_STANDOFF = 3.0

#: (REACH RULE, cache slope) The same four bounds, measured for `c` read from the NEAREST cached
#: samples either side of the root rather than three bracket widths out. Measured 2026-09-14 on a
#: CALIBRATION set held out from `tests/test_cluster_totals.py` -- square levels 25, 85 and 130, not
#: 50 or 65 -- dialled over bands 0.1 to 3.0 and `prec` 1e-6 to 1e-14 through the real search:
#:
#:     beta = (sigma_j/c) / true distance, 962 (branch, eigenvalue) pairs
#:     cache slope     min 0.642  p05 0.839  median 1.000  p95 1.175  max 2.098
#:     outside slope   min 0.835  p05 0.911  median 1.526  p95 5.554  max 14.853
#:
#: The outside slope is off by up to 15x once a cluster is wider than the standoff, because its
#: samples then sit in other members' wells. Each bound is the measured extreme widened by 5%:
#: `COUNT_BELOW` under every observed beta, so a count never takes an eigenvalue outside, and
#: `REJECT_ABOVE` over every one, so a rejection never drops one inside. Replayed on the same
#: pairs: 893 correct, 0 wrong, 69 declined -- against 3 wrong with the outside-slope bounds.
COUNT_BELOW_CACHE = 0.61
REJECT_ABOVE_CACHE = 2.21
BIAS_LO_CACHE = 0.61
BIAS_HI_CACHE = 2.21

#: Branches read. The low spectrum only -- a multiplet of interest is small, and reading deep
#: into the vector costs nothing but invites a spurious gap far above anything physical.
MAX_BRANCHES = 8

#: (PHASE D) `|<u_i, u_j>|` at or above which two reported roots are the SAME eigenfunction.
#:
#: Distance provably cannot make this call: one eigenvalue found twice and two genuine
#: eigenvalues occur at the SAME separation. Only the eigenfunctions tell them apart, which is
#: the whole argument for `modes(x)` riding on the `Objective`.
#:
#: Measured through `MPSPencil.interior_image` on the whole roster
#: (`notes/landscape_findings.md` section 36, `notes/gram_discrimination.py`):
#:
#:     domain     worst DISTINCT   sampling floor 1/sqrt(n_int)
#:     rect             0.1402           0.146   (n_int = 47)
#:     L_shape          0.0605           0.112   (n_int = 80)
#:     eq_tri           0.1678           0.105   (n_int = 90)
#:     disk             0.2544           0.109   (n_int = 84)
#:
#:     worst distinct 0.2544  <  0.5  <  weakest duplicate 1.0000
#:
#: A factor of four on the worst domain measured, so this is not a delicate value; anything from
#: about 0.3 to 0.9 reproduces every reading. The corner domain is the BEST of the four, which
#: retires the transfer risk section 15 named.
#:
#: **The distinct side is a FLOOR THAT MOVES, not a constant.** `search.md` quotes "distinct
#: <= 0.10"; that is the Monte-Carlo noise of a discrete inner product over `n_int` interior
#: points, so it scales as `1/sqrt(n_int)` and the rectangle's own 0.1402 already exceeded the
#: quoted number. A collocation change moves it, and this constant's margin with it.
#:
#: **WITHIN ONE `lam` THE ANGLE IS VACUOUS AND MUST NOT BE CONSULTED.** The GSVD gives
#: `A_I x_j = s_j v_j` with `V` orthogonal, so the interior images of distinct generalized
#: singular vectors are orthogonal BY CONSTRUCTION -- measured 4.6e-16. It is informative only
#: ACROSS reported roots, where the vectors come from different pencils.
#:
#: Two things not yet measured. Every reading above is at EXACT eigenvalues rather than at
#: located roots carrying their own `delta_res`. And the disk's 0.2544 is 2.3x its own sampling
#: floor, on a pair of DEGENERATE levels where which vector the GSVD returns inside the
#: eigenspace is decided by roundoff -- the one number to chase before anyone tightens this.
TAU_GRAM = 0.5


@dataclass(frozen=True)
class Branch:
    """One branch the reach rule EXAMINED, and what it concluded about it.

    The count alone cannot be reconciled with anything. `mu` says how many eigenvalues the rule
    believes are in the bracket; it does not say WHICH, so when the side search then locates one
    of them as a level of its own there is nothing to compare against and the member is counted
    twice. That was `170*pi^2`, band 0.3, prec 1e-8: 9 of 8, certified. This record is the
    missing half -- a branch carries where the rule thinks its eigenvalue is, so a located level
    can OWN it and the root's count drop by one.

    `d_est = sigma_j/c` is an ESTIMATE, biased: `d_est = beta * d_true` with `beta` in
    `[bias_lo, bias_hi]`, so the true distance lies in `[d_est/bias_hi, d_est/bias_lo]`. That is
    the same interval the rule itself decides with, and ownership is tested against it rather
    than against a new constant.

    `side` is `-1`/`+1` only where the two-evaluation probe actually resolved it. It is 0 in the
    two cheap paths, where the verdict holds whichever side the eigenvalue lies, and where the
    probe tied -- so a reader can never mistake "not measured" for "measured as centred".

    **A branch the loop never reached is ABSENT, not marked.** After a `break` or a decline there
    is no evidence about the branches above, and an `"unexamined"` entry would invite a caller to
    reason from nothing. The absence is what makes a declined read legible: `mu` is a partial
    count, and the record says exactly which branches it is partial over.
    """
    index: int                          # j, the branch's position in the tension vector at x
    sigma: float                        # sigma_j at x, so d_est can be re-derived under another c
    d_est: float                        # sigma_j/c: the ESTIMATED distance from x to it
    side: int                           # -1 below x, +1 above, 0 undetermined or not probed
    verdict: str                        # "counted" | "rejected" | "undecided"


@dataclass(frozen=True)
class Multiplicity:
    """What Phase C concluded, and the evidence it concluded it from."""
    mu: int
    gap_ratio: float = float("nan")     # sigma_{mu+1}/sigma_mu, the (C1) margin actually seen
    prominence: tuple = ()              # per counted branch, the (C2) margin
    bracket: tuple = ()                 # the WIDE bracket used
    c1_read: int = 0                    # what the gap test alone would have said
    evaluations: int = 0                # must be zero on a shared cache
    #: A branch landed in the band around `FLOOR_RATIO` where the calibration populations put
    #: nothing. The count is reported anyway and flagged: this is off the measured manifold.
    ambiguous: bool = False
    #: Per-branch evidence, under `rule="reach"` only. Empty under "gap" and "floor", which have
    #: no per-branch distance estimate to record -- they answer a different question, and an
    #: empty tuple says so rather than implying the rule declined to fill it in.
    branches: tuple = ()

    @property
    def c2_rejected(self):
        """Did (C2) cut the gap test's answer down? The load-bearing case."""
        return self.c1_read > self.mu


def wide_bracket_for(samples_lam, lam, sigma_1, c, *, factor=WIDE_FACTOR, prec=None,
                     neighbour=None):
    """The nearest EVALUATED points either side of `lam` at a distance where branches have moved.

    **Everything here is in `lam`**, including `samples_lam`, `c` and the returned pair. The
    name says so because the mix is silent otherwise: `prec` is relative to `lam` by definition
    (`docs/conventions.md`), the search works in `k`, and an `Objective` parameterized by `k`
    reports its `samples` in `k` -- so handing those straight in would compare a `k` distance
    against a `prec*lam` reach and be wrong by a factor of `2k` without failing.
    Use `[obj.in_lam(x) for x in obj.samples[0]]`.

    `eps/c` is the scale on which the cone flattens into its quadratic bottom, so inside it the
    higher branches carry no information and (C2) would be reading the floor against itself.
    This walks outward through the samples the search already paid for and takes the first on
    each side past the reach.

    **The reach must also clear the located root's OWN uncertainty, and that is not the same
    thing.** `sigma_1` here is read at the REPORTED root, which sits within `prec*lam` of the
    true minimum -- so on a well-resolved pencil it is not the floor at all. Measured on the
    disk at `prec = 1e-8`: the reported root sits ~3e-09 from the minimum, `sigma` there is
    6.2e-10 against a floor of 7.2e-16, and `2*sigma/c` gives a reach of 6.4e-09. The flanks
    then straddle asymmetrically -- one side moves TOWARD the true minimum and one away -- so
    the prominence comes out at 1.45 against `P = 2`, (C2) rejects `sigma_1` itself, and a
    genuine multiplicity-2 level reads as 1. Two of seven disk levels went that way.

    Passing `prec` fixes it: the reach then also exceeds `factor * xatol`, so both flanks are
    outside the region the refinement could not resolve. Without `prec` the behaviour is the
    old one, which is correct only when `sigma_1` really is the floor -- as it is when the
    caller reads at a SAMPLED point rather than at a located root.

    **And it must not reach past a DISTINCT neighbour**, which is the same constraint from the
    other side. Pass `neighbour` -- the estimated distance to the next eigenvalue, `sigma_2/c`,
    the quantity the isolation gate already uses -- and the reach is capped at half of it.
    Measured on the wide rectangle window at `prec = 1e-8`: the (4,5)/(5,4) pair sits 5.3e-06
    apart, the `prec*lam` term drove the reach to 8.1e-06, so the flanks straddled BOTH members
    and (C2) correctly saw two branches dipping inside a bracket that spanned the pair. One
    level over-read, and the reach now comes out at 2.98e-06, inside the pair.
    
    The cap never collapses a genuine degeneracy, because there `sigma_2` is at the floor and
    `neighbour` is itself about `eps/c` -- so the cap is floored at `factor*eps/c`, which is the
    term that lets (C2) confirm a real multiplet in the first place. The three terms together
    say: far enough for the branches to have moved, past what the refinement could resolve, and
    not as far as the next eigenvalue.

    Falls back to the outermost sample on a side that has nothing far enough: a bracket that is
    too narrow makes (C2) conservative -- it fails to confirm a branch and the multiplicity is
    under-read -- which is the safe direction, since over-reading is the double count.
    """
    xs = np.asarray(samples_lam, dtype=float)
    if xs.size == 0:
        return None
    # Three constraints, and they have to be applied in this order or two of them cancel.
    #
    #   moved  -- far enough that the higher branches have left the flat bottom, `eps/c`
    #   unc    -- far enough to be outside the located root's own uncertainty, `prec*lam`
    #   cap    -- closer than the nearest DISTINCT neighbour, `sigma_2/c`
    #
    # `cap` applies only when the neighbour is a SEPARATE LEVEL at this precision, i.e. further
    # away than `prec*lam`. That is the definition of `S(prec)` and not a tuned threshold: a
    # neighbour inside the merge licence belongs to THIS level, so the flanks should span it and
    # count its branch; a neighbour outside the licence is a different level, so they must not.
    #
    # Applying the cap on any other test makes one of the three cases fail:
    #
    #   an EXACT degeneracy has `sigma_2` at the floor, so `neighbour ~ eps/c` -- not a
    #   neighbour to avoid but the SAME level. Capping there collapsed the reach back below the
    #   location uncertainty and the degeneracies read 1 instead of 2.
    #
    #   a resolvable PAIR has `neighbour` outside the licence, and without the cap the `unc`
    #   term drove the reach past it: on the wide window the (4,5)/(5,4) pair sits 5.3e-06
    #   apart against a reach of 8.1e-06, so the flanks spanned both members and it over-read.
    #
    #   a SUB-LICENCE pair -- separation below `prec*lam`, which `S(prec)` says is one level of
    #   multiplicity 2 -- has `neighbour` small but far above the flat bottom. Capping on
    #   "further than `eps/c`" therefore pulled the flanks INSIDE the pair, (C2) saw one branch,
    #   and the level under-read: 3 declared against 4 true on the `b=0.5` crossover case.
    moved = factor * (float(sigma_1) / c) if c > 0 else 0.0
    reach = moved
    if prec is not None:
        xatol = max(float(prec) * abs(float(lam)), 4.0 * np.spacing(abs(float(lam))))
        reach = max(reach, factor * xatol)
    licence = (max(float(prec) * abs(float(lam)), 4.0 * np.spacing(abs(float(lam))))
               if prec is not None else 0.0)
    if (neighbour is not None and np.isfinite(neighbour) and neighbour > licence
            and neighbour > 2.0 * moved):
        reach = max(min(reach, 0.5 * float(neighbour)), moved)
    left = xs[xs < lam]
    right = xs[xs > lam]
    if left.size == 0 or right.size == 0:
        return None
    far_l = left[lam - left >= reach]
    far_r = right[right - lam >= reach]
    lo = float(far_l.max()) if far_l.size else float(left.min())
    hi = float(far_r.min()) if far_r.size else float(right.max())
    return lo, hi


def _local_slope(v_lo, v_hi, lo, hi):
    """`c` from the flanks, cancellation-free.

    On a cone the flanks straddle the minimizer, so `v_lo = c*(lam* - lo)` and
    `v_hi = c*(hi - lam*)`, hence `c = (v_lo + v_hi)/(hi - lo)` with no subtraction of
    nearly-equal quantities and no dependence on where the minimum is.
    """
    return (v_lo + v_hi) / (hi - lo) if hi > lo else 0.0


def floor_multiplicity(t, ratio=FLOOR_RATIO, band=FLOOR_BAND, max_branches=MAX_BRANCHES):
    """How many branches sit AT THE FLOOR with `sigma_1`. `(mu, ambiguous)`.

    One point, one comparison, no flanking samples. That is the whole of it, and the absence of
    a wide bracket is the point: the bracket is what reached into neighbouring levels.

    `ambiguous` is True when some branch lands in `[ratio/band, ratio*band]`, where the two
    calibration populations put nothing. It does not change `mu` -- it says the caller is off the
    manifold `FLOOR_RATIO` was measured on and the count should not be trusted silently.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))[:max_branches]
    if t.size == 0 or not np.isfinite(t[0]) or t[0] <= 0.0:
        return 1, False
    r = t / t[0]
    mu = int(np.count_nonzero(r <= float(ratio)))
    ambiguous = bool(np.any((r > float(ratio) / band) & (r < float(ratio) * band)))
    return max(mu, 1), ambiguous


def reach_multiplicity(tensions, x, lo, hi, c, *, count_below=COUNT_BELOW,
                       reject_above=REJECT_ABOVE, bias_lo=BIAS_LO, bias_hi=BIAS_HI,
                       side_ratio=SIDE_RATIO,
                       max_branches=MAX_BRANCHES, counter=None):
    """How many eigenvalues lie inside `[lo, hi]`. `(mu, declined, spent, branches)`.

    `branches` is one `Branch` per branch EXAMINED, in order, carrying the estimated distance and
    the verdict this function reached. It is evidence, not a second answer: `mu` counts exactly
    the entries with `verdict == "counted"`. It exists because a caller that later LOCATES an
    eigenvalue needs to know whether that eigenvalue is one this count already includes -- see
    `Branch`, and `search._claim`, which is the only consumer.

    **The question is `mu*(B)`** and this answers it by comparing A DISTANCE TO A DISTANCE, which
    is the comparison it actually calls for. `sigma_j/c` estimates the distance from `x` to the
    `j`-th nearest eigenvalue; the bracket says how far there is to go. Neither a ratio on `sigma`
    nor a prominence against flanking samples is dimensionally capable of it: a ratio measures
    distance in units of the floor `eps`, and the bracket is in units of `prec*lam`.

    # `sigma_j/c` is UNSIGNED, and that is the whole subtlety

    It gives how far the `j`-th eigenvalue is, not which side. A located root is rarely centred --
    the refiner puts it at the minimum, not the middle -- so the bracket reaches further one way
    than the other, and the two obvious tests fail in opposite directions:

        d <= min(x - lo, hi - x)    rejects eigenvalues inside on the LONGER side  -> under
        d <= max(x - lo, hi - x)    accepts eigenvalues outside on the SHORTER side -> over

    Both were measured. `min` produced every one of 15 pipeline shortfalls; `max` produced
    over-counts on the first asymmetric bracket it saw. Symmetric brackets make the two identical,
    which is why a ladder built from them scored this rule 50/50 on the clustered regime while the
    pipeline was under-counting that same regime.

    # So the side is RESOLVED rather than assumed

    Inside `near` every branch is in the bracket whichever way it lies; beyond `far` none is. Only
    between them does the side matter, and there it is settled with evidence instead of a guess:
    `sigma_1` is evaluated at `x - d` and `x + d`, and the eigenvalue is on whichever side reads
    lower. That costs two evaluations, and only for a branch that actually lands in the band --
    nothing on a simple bracket, where the nearest neighbour sits at a median `d/reach` of 12.4.

    Declines only when the probe cannot separate the two sides, which means the evidence does not
    determine the answer. A decline is the third outcome; a silent wrong count is the only
    unacceptable one.
    """
    before = counter.n_evals if counter is not None else 0
    t = np.atleast_1d(np.asarray(tensions(x), dtype=float))[:max_branches]
    if t.size == 0 or not (c > 0):
        return 1, False, 0, ()
    x, lo, hi = float(x), float(lo), float(hi)
    near, far = min(x - lo, hi - x), max(x - lo, hi - x)
    if not (far > 0):
        return 1, False, 0, ()

    # Every verdict below also EMITS a `Branch`, so the count can be reconciled with the levels a
    # later stage locates. The verdicts themselves are untouched -- this records what the rule
    # already decided and decides nothing new.
    seen = []

    def note(j, d, side, verdict):
        seen.append(Branch(index=j, sigma=float(t[j]), d_est=float(d), side=int(side),
                           verdict=verdict))

    # Branch 0 is the level's OWN eigenvalue -- the level exists because it was located -- so it
    # is counted and never tested. Testing it lost a partner at 170*pi^2, prec 1e-12, band 0.3: the
    # root sat at its bracket's edge with its eigenvalue 1.7e-14 relative beyond, branch 0 landed in
    # the band, the probe read it outside on the short side, and the loop stopped before branch 1.
    mu, declined = 1, False
    note(0, t[0] / c, 0, "counted")
    for j, d in enumerate(t[1:] / c, start=1):
        if d <= near * count_below:
            mu += 1                                   # inside whichever side it lies
            note(j, d, 0, "counted")
            continue
        if d >= far * reject_above:
            note(j, d, 0, "rejected")
            break                                     # outside both ways; branches ascend
        # In the band: the answer depends on the side, so measure the side. Each side is judged on
        # its own: does it HOLD an eigenvalue near `d`? (`d*c` is `sigma_j`.)
        v_lo = float(np.atleast_1d(tensions(x - d))[0])
        v_hi = float(np.atleast_1d(tensions(x + d))[0])
        bar = side_ratio * d * c
        has_lo, has_hi = v_lo < bar, v_hi < bar
        if has_lo != has_hi:
            # How far the bracket runs on the one side that holds the branch.
            side = -1 if has_lo else +1
            r_side = (x - lo) if has_lo else (hi - x)
            # The probe fixes the SIDE; the bias bounds fix how far. `d_est = beta * d_true`, so
            # the true distance is in `[d/BIAS_HI, d/BIAS_LO]` and the three cases are decidable:
            if d / bias_lo <= r_side:
                mu += 1                               # inside even at its furthest possible
                note(j, d, side, "counted")
                continue
            if d / bias_hi > r_side:
                # Outside on THIS side only, so this branch is not counted -- and the loop goes on,
                # since the next branch may lie on the other side, which reaches further. Breaking
                # here lost a member at 170*pi^2, prec 1e-12, band 0.3: branch 1 was outside the
                # short side, branch 2 inside the long one. Only "outside both ways" ends the loop.
                note(j, d, side, "rejected")
                continue
        else:
            # BOTH sides hold an eigenvalue near `d`, or neither reads as holding one: the side is
            # not determined, and taking the lower probe was noise. 50*pi^2, prec 1e-8, band 0.8:
            # partners 8.0e-9 away either side read 0.121 and 0.109 of sigma_j, the "lower" was
            # the SHORT side, the branch was judged outside and the loop stopped, and the partner
            # inside the long side went uncounted -- 2 of 3, not declined. Only a verdict that
            # holds on EITHER side stands.
            side = 0
            if d / bias_lo <= near:
                mu += 1                               # inside whichever side it lies
                note(j, d, side, "counted")
                continue
            if d / bias_hi > far:
                note(j, d, side, "rejected")
                break                                 # outside whichever side it lies
        # Undetermined: the eigenvalue may or may not be inside, and nothing cheap decides it.
        # This is the bracket-edge case -- measured at `65*pi^2`, `prec = 1e-10`, where the
        # partner sits exactly `near` away and `d_est` overshoots it by 55%.
        note(j, d, side, "undecided")
        declined = True
        break
    spent = (counter.n_evals - before) if counter is not None else 0
    return max(mu, 1), declined, spent, tuple(seen)


def multiplicity(tensions, x, *, wide_bracket=None, prec=None, G=GAP_RATIO, P=PROMINENCE,
                 max_branches=MAX_BRANCHES, counter=None, rule="gap", bracket=None,
                 c=None, reach_bounds=None):
    """How many eigenvalues sit at `x`. Reads the cache; spends nothing.

    `rule="reach"` counts branches whose estimated distance `sigma_j/c` falls inside the
    bracket. It needs `bracket` and `c`, and it is the only rule that answers `mu*(B)`; the other
    two answer different questions and are kept for comparison on one instrument.

    `rule="floor"` counts branches at the floor WITH `sigma_1` -- see `floor_multiplicity`. It
    needs no `wide_bracket`. NOT the default: the mechanism is right in kind and its constant is
    not yet honestly calibrated, for the reason recorded on `FLOOR_RATIO`.

    `rule="gap"` is the previous mechanism, (C1)'s largest ratio gap filtered by (C2)'s
    prominence against a wide bracket, kept so the two can be compared on one instrument. It is
    not the default because (C1) asks the wrong question -- the largest gap is the gap to the
    PLATEAU, not the gap to the neighbours -- and (C2)'s bracket has to reach far enough to land
    on the next level. `docs/design/bracket_search_divergence.md` section 3 is the argument.

    Returns a `Multiplicity`. `mu` is the count; under `rule="gap"`, `c1_read` is what the ratio
    gap alone would have said.
    """
    if rule == "reach":
        if bracket is None or c is None:
            raise ValueError('rule="reach" needs `bracket` and `c`')
        lo, hi = (float(v) for v in bracket)
        kw = {} if reach_bounds is None else dict(
            count_below=reach_bounds[0], reject_above=reach_bounds[1],
            bias_lo=reach_bounds[2], bias_hi=reach_bounds[3])
        mu, declined, spent, seen = reach_multiplicity(
            tensions, x, lo, hi, float(c), max_branches=max_branches, counter=counter, **kw)
        t = np.atleast_1d(np.asarray(tensions(x), dtype=float))
        n = min(t.size, max_branches)
        gap = float(t[mu] / t[mu - 1]) if mu < n and t[mu - 1] > 0 else float("inf")
        return Multiplicity(mu=mu, gap_ratio=gap, bracket=(lo, hi), c1_read=mu,
                            evaluations=spent, ambiguous=declined, branches=seen)
    if rule == "floor":
        before = counter.n_evals if counter is not None else 0
        t = np.atleast_1d(np.asarray(tensions(x), dtype=float))
        spent = (counter.n_evals - before) if counter is not None else 0
        mu, amb = floor_multiplicity(t, max_branches=max_branches)
        n = min(t.size, max_branches)
        gap = float(t[mu] / t[mu - 1]) if mu < n and t[mu - 1] > 0 else float("inf")
        return Multiplicity(mu=mu, gap_ratio=gap, bracket=tuple(wide_bracket or ()),
                            c1_read=mu, evaluations=spent, ambiguous=amb)
    if wide_bracket is None:
        raise ValueError('rule="gap" needs a wide_bracket')
    lo, hi = (float(v) for v in wide_bracket)
    before = counter.n_evals if counter is not None else 0

    t = np.atleast_1d(np.asarray(tensions(x), dtype=float))
    tl = np.atleast_1d(np.asarray(tensions(lo), dtype=float))
    tr = np.atleast_1d(np.asarray(tensions(hi), dtype=float))
    spent = (counter.n_evals - before) if counter is not None else 0

    n = min(t.size, tl.size, tr.size, max_branches)
    if n < 1:
        return Multiplicity(mu=1, bracket=(lo, hi), evaluations=spent)
    t, tl, tr = t[:n], tl[:n], tr[:n]
    if n < 2:
        return Multiplicity(mu=1, bracket=(lo, hi), c1_read=1, evaluations=spent)

    # (C1) -- the largest ratio gap in the low spectrum. A gap, never a threshold: the floor
    # varies five orders within one scan, so no absolute level on `sigma` can serve.
    ratios = t[1:] / np.maximum(t[:-1], np.finfo(float).tiny)
    j = int(np.argmax(ratios))
    kappa = j + 1
    gap = float(ratios[j])
    if gap < G:
        kappa = 1
    c1_read = kappa

    # (C2) -- every counted branch must be a local minimum HERE, to prominence `P`, against the
    # wide bracket. The multiplicity is a contiguous block `1..mu` by definition, so this takes
    # the longest prefix that holds rather than checking the top index and walking down: the
    # property is not monotone in `j`, and a walk-down would accept a block with a hole in it.
    prom = []
    mu = 0
    for i in range(kappa):
        flank = min(tl[i], tr[i])
        p = flank / t[i] if t[i] > 0 else float("inf")
        if p < P:
            break
        prom.append(float(p))
        mu = i + 1

    return Multiplicity(mu=max(mu, 1), gap_ratio=gap, prominence=tuple(prom),
                        bracket=(lo, hi), c1_read=c1_read, evaluations=spent)
