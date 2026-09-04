"""Stage 3: how many eigenvalues sit at this level, read from evidence already paid for.

TEST SUPPORT ONLY -- a scored candidate. `docs/design/candidates.md` section 1 is the seam,
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

__all__ = ["multiplicity", "wide_bracket_for", "Multiplicity", "GAP_RATIO", "PROMINENCE",
           "WIDE_FACTOR", "MAX_BRANCHES"]

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

#: Branches read. The low spectrum only -- a multiplet of interest is small, and reading deep
#: into the vector costs nothing but invites a spurious gap far above anything physical.
MAX_BRANCHES = 8


@dataclass(frozen=True)
class Multiplicity:
    """What Phase C concluded, and the evidence it concluded it from."""
    mu: int
    gap_ratio: float = float("nan")     # sigma_{mu+1}/sigma_mu, the (C1) margin actually seen
    prominence: tuple = ()              # per counted branch, the (C2) margin
    bracket: tuple = ()                 # the WIDE bracket used
    c1_read: int = 0                    # what the gap test alone would have said
    evaluations: int = 0                # must be zero on a shared cache

    @property
    def c2_rejected(self):
        """Did (C2) cut the gap test's answer down? The load-bearing case."""
        return self.c1_read > self.mu


def wide_bracket_for(samples_lam, lam, sigma_1, c, *, factor=WIDE_FACTOR, prec=None):
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

    Falls back to the outermost sample on a side that has nothing far enough: a bracket that is
    too narrow makes (C2) conservative -- it fails to confirm a branch and the multiplicity is
    under-read -- which is the safe direction, since over-reading is the double count.
    """
    xs = np.asarray(samples_lam, dtype=float)
    if xs.size == 0:
        return None
    reach = factor * (float(sigma_1) / c) if c > 0 else 0.0
    if prec is not None:
        xatol = max(float(prec) * abs(float(lam)), 4.0 * np.spacing(abs(float(lam))))
        reach = max(reach, factor * xatol)
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


def multiplicity(tensions, x, *, wide_bracket, prec=None, G=GAP_RATIO, P=PROMINENCE,
                 max_branches=MAX_BRANCHES, counter=None):
    """How many branches have a local minimum at `x`. Reads the cache; spends nothing.

    `wide_bracket` is the `(lo, hi)` pair of EVALUATED points either side of the root, far
    enough out that the higher branches have moved -- see `wide_bracket_for`.

    Returns a `Multiplicity`. `mu` is the count; `c1_read` is what the ratio gap alone would
    have said, so the caller can see when (C2) did the work.
    """
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
