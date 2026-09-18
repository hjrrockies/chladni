# Tension landscape: first measurements

Run log, not tracked. Supersedes an earlier version of this file whose roughness numbers
were wrong; see section 6 for what went wrong and why it matters methodologically.

## 1. The cone model holds where the basis is adequate

`sigma_1 ~ sqrt(eps^2 + c^2 delta^2)` fitted over the far field (2-6% away in k): median
relative residual **1-3%** on rect 2x1, eq_tri, L-shape, and on the disk's well-resolved
dips. It fails only where the dip floor is near the plateau, i.e. where the basis cannot
represent the eigenfunction at all.

## 2. `c` is stable within a domain, not across domains

    rect 2x1   0.98 - 1.17   (k in [3.3, 7.3])      L-shape  1.10
    disk       1.56 - 1.70                          eq_tri   0.45 - 0.49

12x across domains. Per-solve calibration is necessary and sufficient, which is what
`lipschitz-cone-mps.md` section 3.3 already prescribes.

## 3. The numerical rank of the pencil GROWS WITH k

Disk, 54 columns, at the default rtol = 1.5e-12:

    k        2.30  2.35  2.40  3.00  3.95  5.00  5.70
    rank     31    31    33    35    39    43    47

Expected in hindsight -- more oscillation supports more independent columns -- but it is
the fact the whole of the rest of this file turns on, and nothing in the code knows it.

## 4. sigma_1 is SMOOTH, provided the retained rank is safe

Residual after removing a cubic trend, window 2e-3 in k:

    disk     rank 12-27  (s_r/s_0 down to 5e-8)    ~1.5e-10 absolute, 1.4e-10 relative
             rank 38     (s_r/s_0 = 3.5e-14)        4.8e-05
             rank 43+    (s_r/s_0 ~ 1e-16)          3.8e-02
    L-shape  rank 12-57  (s_r/s_0 down to 4e-8)    5e-14 to 1e-11
             rank 73+    (s_r/s_0 < 1e-15)          1.6e-03

Roughness sets in only as `s_r/s_0` approaches machine epsilon, and then rises roughly
like `eps_mach / (s_r/s_0)`, saturating near the plateau level.

Spurious local minima, in a window containing exactly ONE eigenvalue, 400 samples:

    disk     s_r/s_0 >= 1e-13 :   1 minimum,  0% wrong-way steps
             s_r/s_0 =  1e-16 :  90 minima,  39% wrong-way steps
    L-shape  s_r/s_0 >= 7e-12 :   1 minimum,  0%
             s_r/s_0 =  1e-16 : 128 minima,  42%

The transition is sharp and sits at machine epsilon. The default rtol is safely clear of
it -- **when the rank is chosen at the right k.**

## 5. The predictor

    r = min over the scan range of choose_rank(k, rtol)

A rank frozen at a single `lam_ref` is safe there and unsafe at the LOW end of the range,
because of section 3. Measured: rank frozen at lam_ref = 15.6 gives r = 39, whose `s_r/s_0`
at k = 2.35 is **2.4e-16** -- machine epsilon -- and that pencil shows 85 spurious minima
in a one-eigenvalue window. Taking the minimum over the range instead gives r = 31 and
**exactly 1 minimum**.

Cost: a handful of `svdvals` over the range, which the pilot pass of any scan performs
anyway.

## 6. Roughness hurts DETECTION, not REFINEMENT

Locating an eigenvalue by Brent on `sigma_1^2` is machine-exact (relative error 0.0) on
disk, rect and eq_tri at every rtol tried from 1e-7 to 1e-16, rough or not: the parabolic
fit averages over the high-frequency structure. The L-shape even prefers a HIGH, rough
rank -- 111, with roughness ~1e-2, locating `lam_1` to 3.2e-12 against 1.4e-10 at the
default rank 61 -- because what limits it there is the floor `eps`, not the roughness.

So the two phases want different ranks, and one SVD serves both: truncate low for the
scan (smooth, cheap, wide dips) and high for the refinement (accurate floor). This is
`lipschitz-cone-mps.md` section 8's two-level idea, but with rank rather than basis size
as the knob -- no second basis, no second assembly, and the two share their factorization.

## 7. Two methodological corrections, recorded deliberately

The first version of this file reported "sigma_1 is rough, eta is 1e-6 to 1e-3, ten orders
above machine epsilon" and concluded the Lipschitz doc's precision arithmetic was broken.
Both halves were artifacts:

* **The metric measured curvature.** Roughness was taken as the residual from a straight
  line over a window of width 2e-3. With sigma ~ 0.2 varying on an O(1) scale, curvature
  alone contributes `0.2 * (2e-3)^2 ~ 8e-7` -- which is exactly the "floor" that appeared,
  rank-independently, in every column. A cubic detrend drops it by four orders.
* **The rank was frozen at the wrong k**, per section 5, so the pencil under test was
  genuinely degenerate at the sampled k -- a real effect, but caused by the harness rather
  than intrinsic to the objective.

The corrected statement is close to the opposite of the original: the objective is smooth
to 1e-10 relative or better, and `eta` is within about four orders of machine epsilon
rather than ten. `lipschitz-cone-mps.md`'s precision arithmetic survives.

Worth keeping because the failure mode is general: a detrending metric that leaves signal
in the residual manufactures a floor, and a floor that does not move when you vary the
thing that supposedly causes it is the tell. It appeared in three separate columns before
it was noticed.

---

# 8. Frozen vs threshold rank: the comparison, run properly

Recall test: a window holding exactly ONE eigenvalue, 400 samples, count local minima.

    domain    frozen chosen at   frozen r   thr r range   minima frozen   minima thr
    disk      point k=3.95             39         31-33              85            1
    disk      range (2.2, 5.7)         31         31-33               1            1
    L-shape   point k=3.5              61         61-61               1            1
    L-shape   range (2.9, 3.6)         61         61-61               1            1

**Threshold is correct in every case, including the one where the frozen rank fails.** It
cannot pick an unsafe rank, because it re-derives from the actual spectrum at each k. The
frozen rank is correct only if the range it was chosen over is right.

Margin on the frozen side is thin. A rank of 37 (chosen for k in [3.5, 5.7]) still behaves
at k = 2.35; 39 does not. Two columns separate correct from 85 spurious minima.

Continuity, the original worry, is a non-issue: over k in [2.3, 5.7] the threshold rank
varied 31 to 47 and the tension's largest step was 5.1x the median. Mild, and it does not
affect Brent.

Cost, same pencils, assembly excluded:

    disk 54 cols     frozen 1.51 ms/k    threshold 2.43 ms/k    1.61x
    L-shape 132 cols frozen 11.3  ms/k   threshold 13.4  ms/k   1.19x

No recall failures for either method on the windows checked; the apparent shortfalls in
[3.6,4.1] and [5.0,5.6] on the disk are degeneracies (j_{1,1} and j_{2,1} are each double),
where one minimum for two eigenvalues is the correct answer.

## Scorecard

    axis                          frozen (range-chosen)      threshold
    recall                        correct if range is right  correct always
    robustness to a wrong range   thin margin (37 ok, 39 no) immune by construction
    cost                          1.0x                       1.2 - 1.6x
    reproducibility               rank is a solver property  rank varies with k
    continuity                    exact                      5.1x median step
    eigenvalue accuracy           tie                        tie

Under "recall above all", threshold is the safer default: it is self-correcting, and the
thing the frozen rank needs -- a correctly declared range -- is exactly what a black-box
solver iterating over generated domains cannot be trusted to supply.

---

# 9. Clusters, degeneracies, and multiplicity

## 9.1 The tunable rectangle: a resolution curve

`rect(1+delta, 1)`, modes (1,2) and (2,1), exactly degenerate at delta = 0 and splitting
like `6*pi^2*delta`. Dips located by refinement, not off the grid.

    separation d     sigma_1 dips   err in k    floor
    4.2e-02               2          8.0e-10    4.3e-10
    4.2e-03               2          7.5e-12    3.6e-12
    4.2e-04               2          1.8e-13    3.2e-13
    4.2e-05               2          9.9e-08    8.4e-08
    1.3e-06               2          1.3e-09    1.2e-09
    0 (exact)             1          5.9e-09    4.8e-09

Two dips are resolved down to d = 1.3e-06, far better than expected; only an exact
degeneracy collapses to one. Location accuracy degrades around d ~ 4e-05, with the floor
rising in lockstep -- the delta_res relationship, visible directly.

**Watch the grid.** A first version of this table reported the floor jumping by seven orders
at small delta. That was an artifact: the window was `max(3*d, 2e-3)`, so whenever it was
NOT clamped the eigenvalues landed exactly on grid points (index 250 of 601) and the sampled
minimum was the true floor, and whenever it was clamped they did not. Nothing physical.

## 9.2 sigma_2's minimum is at the crossing point

Confirmed, and it is the first structural claim from `lipschitz-cone-mps.md` verified on real
pencils. Across every separation above, sigma_2 has exactly ONE minimum and it sits at the
midpoint to within about 1% of d/2. Since sigma_1's minima are at `mid +/- d/2`, the offset
from sigma_2's minimum to the nearest sigma_1 minimum measures the separation -- readable
BEFORE the pair is resolved, which is what section 3.4 promises.

## 9.3 Multiplicity reads correctly off the spectrum

Symmetry-protected degeneracies, derived with multiplicity rather than tabulated. `read` is
the multiplicity inferred from the largest ratio gap in the low spectrum.

    disk       k        true   sigma_1    sigma_2    s2/s1     s3/s2    read
             2.40483      1    5.8e-16    1.6e+00    2.7e+15   1.0e+00     1
             3.83171      2    6.1e-16    1.0e-15    1.7e+00   1.5e+15     2
             5.13562      2    1.0e-15    1.2e-15    1.2e+00   5.0e+14     2
             5.52008      1    9.2e-16    6.3e-01    6.8e+14   1.0e+00     1
             6.38016      2    1.4e-15    1.8e-15    1.3e+00   4.9e+14     2

    eq_tri   7.25520      1    1.6e-15    1.3e+00    7.8e+14   1.1e+00     1
            11.08250      2    2.5e-14    2.9e-14    1.2e+00   4.1e+13     2
            14.51039      1    5.3e-14    2.6e-01    4.8e+12   1.0e+00     1
            15.10290      2    3.7e-13    3.8e-13    1.0e+00   7.2e+11     2

Nine for nine. Within a multiplet the ratio is 1.0 to 1.7; across the gap it is 1e11 to
1e15. The doc suggests `G` between 10 and 100; anything from about 3 to 1e10 would do here,
so that parameter is not delicate. Two mechanisms are covered: the disk's are geometric
(sin/cos pairs), the triangle's arithmetic (`m^2+mn+n^2` collisions).

Note that a real degeneracy gives sigma_1 and sigma_2 numerically coincident but NOT
bit-identical -- 6.1e-16 against 1.0e-15 at the disk's j_1,1. The surrogate's `mult` produces
exactly equal values, so it must not be used to validate an equality test that real data
would fail.

Avoided crossings: at every degeneracy above, the sigma_1/sigma_2 separation is at the noise
floor, so no gap is measurable. Consistent with section 9's claim that the coupling is
exponentially small, though this bounds the gap rather than confirming the mechanism.

## 9.4 (C2) is load-bearing, not belt-and-braces

The doc argues that a smallness-or-gap test alone double-counts when
`delta_res < d <~ eps/c`, because near lambda_1 the second branch is the OTHER cone's
descending wall: small, but monotone rather than minimal. That regime was constructed
deliberately, by shrinking the basis to raise `eps`:

    directions   d          eps       d/(eps/c)   s2/s1 at k1   (C1) reads   (C2) rejects
        10       1.01e-04   2.23e-04     0.45         1.07        2  WRONG      yes -> 1
        10       3.37e-05   2.23e-04     0.15         1.01        2  WRONG      yes -> 1
        12       1.01e-04   4.37e-06    23.17        19.76        1  correct    yes

The transition sits exactly where predicted. With `G = 10` the gap rule reports multiplicity
2 at a SIMPLE eigenvalue, and would do so again at its neighbour -- the double count. (C2)
rejects it in every case, because sigma_2 there is strictly decreasing
(2.564e-04 -> 2.390e-04 -> 2.274e-04).

The consequence for design is larger than the multiplicity rule. **The dangerous regime is
entered by having a mediocre basis**, since it is `eps` that sets `eps/c`. A black-box solver
whose planner produces a merely adequate basis is in it by default. Basis quality and
multiplicity correctness are the same problem, which is another entry for the planner
contract.

---

# 10. Ties and stagnation at the bottom of a well

Asked whether a grid fine enough for roundoff to produce identical consecutive tension values
would blind a discrete local-minimum detector. Measured on real pencils, in both regimes:

    well                       floor eps    flat half-width   ties at representable spacing
    disk j_0,1, good basis      5.5e-16          1.1e-23              0, down to h = 5e-15
    near-square, 10 waves       2.2e-04          4.7e-12              0, down to h = 7.8e-14

Zero ties in either case, and the reasons differ. A well whose floor is at machine level
never flattens -- it stays conical all the way down, and consecutive relative differences
hold at about 5%. A well with a large floor does flatten, and its flat bottom (half-width
`eps*sqrt(2*eps_mach)/c`) is comfortably representable at 4.7e-12 against a float spacing of
8.9e-16 at k = 7 -- but the curve's own roughness there, about 1e-10 relative, exceeds the
quadratic variation across the flat region and breaks ties before they can form.

So the tie-breaking in the local-minimum detector is **cheap insurance, not load-bearing**.
An earlier note in this file said the opposite; it was written from the mechanism rather than
from the measurement, before the measurement existed.

**The real finding is next to it.** In the flat bottom the strict detector reports no local
minimum at all -- not because of ties, but because there is no clean minimum there to find:
the curve is roughness, not signal. Refining below the flat-bottom scale samples noise, which
is exactly the stagnation that `lipschitz-cone-mps.md` section 4 terminates on ("the last
three values of f agree to within eta"). That termination rule is necessary, and the scale it
must trigger at is `eps*sqrt(2*eps_mach)/c` -- computable from the parabola fit the
refinement already performs.

Where the surrogate DOES manufacture ties, it is because its curves are exact parabolas in
the square and therefore exactly symmetric about a well: 85 of the 184 grid sizes from 16 to
199 blind the strict rule. Real curves are not symmetric (0 of 144 on the disk, smallest
neighbour gap 6e-4). The surrogate is harder than reality here, which is the safe direction
for a test double, and `tests/test_harness.py` records it as such.

---

# 11. lappy's baseline, scored -- and why its cluster rescue barely works

`tests/searches.py` reimplements `solve_interval` behind the harness interface: a uniform
grid, discrete local minima of sigma_1, sigma_2-guided recursion, parabolic refinement of
sigma_1 squared, prefix multiplicity against an absolute `ttol`, merge on `ltol`.

It reproduces the known failure immediately. On surrogate pairs at separation `d`, with the
grid spacing `h` set explicitly so the algorithm rather than my spacing heuristic is what is
being measured:

    n_pts    h        d=1e-5  d=1e-4  d=1e-3  d=1e-2      <- recall
      200    1.0e-02    0.50    0.50    0.50    0.50
      500    4.0e-03    0.50    0.50    0.50    1.00
     1000    2.0e-03    0.50    0.50    0.50    1.00
     2000    1.0e-03    1.00    1.00    0.50    1.00
     5000    4.0e-04    1.00    1.00    1.00    1.00

Isolated eigenvalues, doubles, triples and eight crowded simples all score 1.000. Pairs are
the whole of the problem.

## The sigma_2 recursion is gated on the wrong quantity

Its stated purpose is to split a pair too close for sigma_1 to have separated. It fires when
the sampled sigma_2 falls below `ttol`. But sigma_2's minimum VALUE is only `c*d/2` exactly
at the midpoint, and the midpoint is essentially never a grid point. What a grid of spacing
`h` actually samples is

    min over the grid of sigma_2  ~  c*h/2

Measured, for a pair at `d = 1e-5` whose true sigma_2 floor is 5.00e-06:

    h = 1e-02   sampled min sigma_2 = 5.03e-03    (c*h/2 = 5.0e-03)
    h = 1e-03   sampled min sigma_2 = 5.05e-04    (c*h/2 = 5.0e-04)

So the gate tests a property of the GRID, not of the pair. It engages when `h <= 2*ttol/c`
and not otherwise, whatever the separation -- which is exactly the transition in the table,
between h = 2e-3 and h = 1e-3 at `ttol = 1e-3`. The mechanism advertised as the cluster
rescue is really "refine again once the grid is already fine", and by then sigma_1 has
usually separated the pair on its own.

## Which is precisely the case for section 3.4

`lipschitz-cone-mps.md` proposes a different sigma_2 test: not its VALUE but the OFFSET
between its minimum and the nearest sigma_1 minimum, which estimates `d/2`. That quantity
survives coarse sampling in a way the value does not -- the location of a minimum is
resolved to about `h`, while its depth is wrong by a factor of `h/d`. Section 9.2 of this
file confirmed the offset relation holds on real pencils to about 1% of `d/2`.

That is now the strongest single argument for the alternative design, and it came from
scoring the baseline rather than from reading either document.

## Caveat on the numbers

The recall table uses an explicit `n_pts`. With the harness's crude spacing estimator both
searches get a 20-point grid on these cases, because a coarse pass sees the pair as one dip
and concludes the interval holds one eigenvalue. That estimator is mine, not lappy's -- the
original uses a Weyl estimate, which the harness interface deliberately withholds. Any
comparison that turns on the initial spacing is measuring my heuristic; the table above
avoids that by fixing `h`.

---

# 12. Fitting the surrogate to real curves

Until now the surrogate was validated against measured PROPERTIES -- cone slope, floor,
two-dip resolution, the sigma_2 signature. It had never been fitted to a specific real curve
and overlaid on it. Doing that found a calibration bug and a limit.

Wells fitted at known eigenvalues (`eps` from refinement, `c` from the far field), then the
whole curve compared point by point:

    median relative difference        rect(2,1)     L-shape
      sigma_1, in the dips             4.3e-03      5.3e-03
      sigma_1, elsewhere               2.3e-02      2.2e-02
      sigma_2                          9.2e-02      7.3e-02
      sigma_3                          1.7e-01      8.9e-02
      max secant slope inflation        1.02x        1.01x

The L-shape used a corner-enriched basis and `lappy.reference`'s table (14 digits,
independently cross-checked); the rectangle used plane waves and a closed form. Fidelity is
essentially the same on both, so the surrogate is not an artifact of the smooth case.

## The plateau is not a plateau

`Surrogate` took `plateau` as the level of the non-well curves, and it was being set to the
MEDIAN of a real curve. That is wrong twice over. Between two eigenvalues `sigma_1` is not
flat: it is where the neighbouring cones meet, at `c*d/2`. On rect(2,1) it peaks at 0.645
exactly midway between k = 4.443 and 5.664, and `1.18 * 0.55 = 0.65` accounts for it. Setting
the level to the median then CAPS the cones and flattens the structure a search navigates --
2.0e-01 median error off the dips against 2.3e-02 once the cones are allowed to set it.

`plateau` is now documented as a far-field CAP and defaults to `None`, computed as 1.5x the
widest cone crossing.

## The higher branches are not modelled

Every non-well curve sits at the cap, so `sigma_3` and above have no structure, and the fit
degrades accordingly (1.7e-01 on the rectangle). The (C1) gap test reads `sigma_{kappa+1}`,
so multiplicity conclusions drawn on the surrogate must be confirmed on a real pencil. Now in
the class docstring as a known limit.

## Two things found on the way

**An independent check on the pipeline.** Refining at the five L-shape reference values
located them to relative error 0.0 to 1.9e-10 -- chladni agreeing with an independent MPS
implementation on a domain with no closed form.

**`eps` varies by FIVE orders within one scan**, not the fifty-fold recorded earlier:
3.1e-13 at k = 4.443 against 5.5e-08 at k = 5.649 on the L-shape. And `4.443**2 = 19.739 =
2*pi**2`, a mode whose eigenfunction vanishes at the reentrant corner and is therefore
analytic there, which the basis represents almost exactly. The spread is structural, not
incidental: on a domain with a singular corner, the modes that happen to vanish at it are
resolved far better than their neighbours. Any absolute `tau_accept` has to survive that.

---

# 13. The cheap hypothesis wins

Repairing lappy's two measured defects, and changing nothing else:

* **The recursion trigger.** It refined only where the sampled `sigma_2` fell below `ttol` --
  a quantity that is about `c*h/2`, a property of the grid. Replaced by: recurse into EVERY
  discrete minimum of `sigma_1`, unconditionally, bounded by depth. There is no cleverer
  trigger available, because the offset detector fires at the same coarseness at which
  `sigma_1` first shows two minima. The only question is how cheaply you reach that
  resolution, and the answer is "refine only where it is dark".
* **Multiplicity.** The prefix rule against an absolute `ttol`, replaced by (C1) the largest
  ratio gap plus (C2) the local-minimum condition.

Scored on twelve surrogate cases:

                        perfect cases    total evaluations
    uniform grid             6/12                1 422
    lappy original          10/12               54 443
    lappy REPAIRED          10/12                5 573
    lipschitz B&B            7/12                2 093

On the pairs, where the original needed 2012 / 6021 / 8015 evaluations at separations
4e-03 / 4e-05 / 4e-07, the repaired version needs 306 / 292 / 270 -- and unlike the original
it also holds 1.000 precision throughout.

So the `O(log(1/d))` advantage is NOT a property of Piyavskii-Shubert. It is a property of
adaptive subdivision, which lappy's `bracket_mins` already had; its trigger was the only
thing broken. What remains unique to the branch and bound is the attic (reversible pruning)
and the `L` margin diagnostic.

## What no method handles

Two cases fail for everything tested:

* **A pair at 4e-09.** Every search, original lappy included.
* **A narrow dip, `c = 50`.** Repaired lappy 0.00, branch and bound 0.00, uniform grid 0.00.
  Only the ORIGINAL scores 1.00, and only because its fixed 2000-point grid happens to sample
  a 6e-03-wide dip. That is brute force, not intelligence.

Both are failures of INITIAL RESOLUTION, not of the refinement strategy: no amount of clever
subdivision helps if the first pass never sees the dip. Section 9 of `lipschitz-cone-mps.md`
predicted exactly this and it is now measured against four searches. It is also the strongest
argument for keeping the branch and bound's diagnostics even if its scan is not adopted --
the `L` margin is the only signal any of these searches has that its first pass was too
coarse.

## Revised recommendation

Ship the repaired bisection. Treat the branch and bound as a later refinement to be justified
by its diagnostics rather than by its cost, and note that the comparison above understates it,
because the implementation scored here does not work on pairs.

---

# 14. The narrow dip does not exist

Section 13 reported that all four searches fail on a "narrow dip" with `c = 50`. Asked which
domain and basis produce that, the answer is **none**. It was invented as a stress case in
the surrogate, and it is the same error as the planted null direction of section 8: a
construction that demonstrates a mechanism which may not occur.

Measured `c` at 19 real eigenvalues across four domains: **0.445 to 1.725**, median 1.175.
`c = 50` is 29x larger than anything observed.

## The invariant

The quantity a scan actually has to resolve is the dip width relative to the eigenvalue
SPACING, and that turns out to be nearly constant:

    domain     diam   c range      S      dip half-width   mean gap   width/gap
    rect2x1    2.24   1.03-1.18   0.223       0.205         0.868       0.473
    disk       2.00   1.56-1.73   0.398       0.239         1.038       0.460
    eq_tri     1.00   0.45-0.47   0.313       0.674         2.616       0.515
    L-shape    2.83   1.19-1.39   0.199       0.157         0.636       0.495

Dips occupy about HALF the space between eigenvalues, on every domain, despite diameters
spanning 1.0 to 2.83, `c` spanning 0.45 to 1.73 and `S` spanning 0.20 to 0.40. The dip width
is not an independent parameter; it tracks the spacing. So about six samples per eigenvalue
gap place at least three points inside every dip -- and Weyl's law is exactly the tool for
providing that.

## What this changes

The narrow-dip failure mode should be demoted, in `search.md` section 5 and in section 13
above. It is not that the searches handle it -- none of them does -- it is that nothing in
the measured behaviour of MPS produces it.

The failure mode that IS real is the mirror image: a CLUSTER, where the gap shrinks toward
zero while the width tracks it. That is what sections 9 and 13 measured, and it is what the
search has to be designed against.

## Caveat

Four domains, all fairly benign, at low k. A domain with a genuinely clustered spectrum --
GWW, a high aspect ratio, a near-symmetry -- has small local gaps by construction, and the
invariant above then says the dips overlap. That is the cluster problem again rather than a
new one, but the ratio should be re-measured there before it is relied on.

---

# 15. The interior Gram separates "two eigenfunctions" from "one, found twice"

Section 9 measured multiplicity off the tension SPECTRUM -- (C1)'s ratio gap and (C2)'s local
minimum. This section measures it off the EIGENFUNCTIONS instead, and it answers a question
neither condition can reach.

`search.md`'s Phase D has to classify two roots reported at nearby `lam` as either ONE well
found twice (collapse, then RECOUNT the multiplicity there) or TWO independently resolved
eigenvalues (merge, then SUM). It decides that by DISTANCE: diameter against `delta_res`, then
against `tol_merge*x`. **Distance cannot separate those two cases, because they occur at the
same separation.** One simple eigenvalue found twice from two brackets and two genuine simple
eigenvalues can both present as a pair of roots `d` apart, and the tension values at them are
both small. That is the double-count the harness scores as one match plus one spurious.

The angle between the candidate eigenfunctions does separate them, and it is nearly free.

## It needs no lifting to L2(Omega)

The `Sampling` is fixed at construction and only `k` changes inside `assemble`, so two roots
at `lam_1 != lam_2` produce vectors on the SAME interior point set. The comparison is a plain
dot product: no common refinement, no quadrature, no L2 inner product materialized.

    u_i = I(lam_i) @ coef(lam_i, 1)          I from pencil(lam), already equilibrated

`coef` lives in that same scaled column space, so nothing has to be unscaled -- which matters,
because at a sharp corner the unscaled coefficient vector does not exist in float64.

Cost is one GSVD in `separate` mode per reported root; the scan itself only ever calls
`gsvdvals`. Negligible against thousands of scan evaluations.

## Within one lam the test is VACUOUS, and that is not a small point

The GSVD gives `U' A_B X = D1` and `V' A_I X = D2` with `U, V` orthogonal, so for the
minimizing columns `A_I x_j = s_j * v_j`: the interior images of distinct generalized singular
vectors are **orthogonal by construction**, for any pencil, at an eigenvalue or not. Measured
at the exact degeneracy of `rect(1,1)`, `coef(lam, 2)`:

    |cos| = 4.6e-16

So an independence check applied to `coef(lam, mult)` always passes and proves nothing. The
check is only informative ACROSS roots, where the vectors come from different pencils. This
survives regularization: `regularize` reduces in column space and `back` is that map, so
`A_I @ back(x) == I_reduced @ x` exactly.

## The two arms

`rect(1+delta, 1)`, modes (2,1) and (1,2), degenerate at `delta = 0` and splitting like
`6*pi**2*delta`. 40 plane waves, 800 boundary, 400 interior points. `notes/gram_angle.py`.

    PAIR   u(lam_a) vs u(lam_b)          two distinct eigenfunctions, d apart
    DUP    u(lam_c) vs u(lam_c + d)      ONE isolated well (mode (1,1)), sampled twice

    delta     d(lam)     eps_a     eps_b   PAIR cos    DUP cos
  3.0e-01  1.209e+01  7.99e-10  9.80e-09  5.758e-02  1.038e-01
  1.0e-01  5.139e+00  1.03e-09  1.63e-09  5.758e-02  9.751e-01
  3.0e-02  1.700e+00  5.15e-10  2.99e-10  5.758e-02  9.976e-01
  1.0e-02  5.834e-01  2.96e-10  2.60e-11  5.758e-02  9.997e-01
  3.0e-03  1.769e-01  2.08e-10  1.12e-10  5.758e-02  1.000e+00
  1.0e-03  5.913e-02  1.82e-10  1.39e-10  5.758e-02  1.000e+00
  3.0e-04  1.776e-02  1.73e-10  1.48e-10  5.758e-02  1.000e+00
  1.0e-04  5.921e-03  1.70e-10  1.51e-10  5.758e-02  1.000e+00
  3.0e-05  1.776e-03  1.69e-10  1.52e-10  5.758e-02  1.000e+00
  1.0e-05  5.922e-04  1.69e-10  1.52e-10  5.758e-02  1.000e+00

The `delta = 3e-1` row is DUP being meaningless rather than DUP failing: at `d = 12.1` the
offset has left the well entirely and landed in a neighbour's basin. The mean gap here is
`4*pi/area = 12.4`. Nobody would put a collapse decision there.

## PAIR's floor is 1/sqrt(n_int), not anything physical

`5.758e-02` to four digits down five decades of `d` is the tell. It is not a property of the
eigenvalues: the two modes are exactly L2-orthogonal on a rectangle, so the true `cos` is
zero, and what is measured is the MONTE-CARLO overlap of two orthogonal functions at 400 fixed
random interior points. Same seed, same points, essentially the same two modes as `delta`
varies -- hence the same number every row.

Varying the count and the seed at `delta = 1e-2`:

    n_int  seed    PAIR cos  x sqrt(n_int)
      100     0   7.789e-02          0.779
      100     2   7.803e-03          0.078
      200     0   1.020e-01          1.442
      400     2   1.015e-01          2.031
      800     0   1.172e-02          0.331
     1600     0   4.068e-02          1.627
     3200     2   1.539e-03          0.087

`cos * sqrt(n_int)` is O(1) with no trend across a 32x range in `n_int`. So the discriminating
floor is bought with INTERIOR POINTS, which are cheap and do not touch the pencil's
conditioning. Worst case observed anywhere in the sweep: 0.10.

## DUP departs from 1 like d**2

    offset d     DUP cos     1 - cos       /d**2
    8.00e+00    0.948156   5.184e-02   8.101e-04
    4.00e+00    0.987126   1.287e-02   8.046e-04
    1.00e+00    0.999214   7.856e-04   7.856e-04
    1.00e-01    0.999992   7.778e-06   7.778e-04
    1.00e-02    1.000000   7.770e-08   7.770e-04
    1.00e-04    1.000000   7.769e-12   7.769e-04

Flat in the last column over six decades: the minimizing direction rotates linearly in the
offset, so `1 - cos ~ theta**2/2`. The coefficient is well-specific and this is one well, but
the scale it sets is the useful part.

## The margin, and the threshold

                            PAIR (distinct)   DUP (duplicate)
    d ~ 1    (0.08 gaps)         <= 0.10           0.9992
    d ~ 8    (0.65 gaps)         <= 0.10           0.948

Any threshold from about **0.3 to 0.9** works. Extrapolating `1 - cos = 7.8e-04 * d**2`, DUP
would only fall to PAIR's level at `d ~ 34`, about three eigenvalue gaps -- far outside any
regime where collapse-versus-merge is a question at all. `tau_gram = 0.5` sits in the middle
with an order of magnitude of margin either side.

**Not delicate, on the same pattern as (C1)'s `G`** (section 9.3: "anything from about 3 to
1e10 would do here"). Two independent multiplicity parameters have now come out non-delicate,
which is worth noting against the general expectation that this layer is full of tuning.

Stated properly the threshold is relative to `1/sqrt(n_int)` rather than a bare constant, but
with `n_int >= 100` giving PAIR <= 0.1 a fixed 0.5 is safe in practice.

## Why the ambiguous case is benign

Inside a genuinely degenerate eigenspace, WHICH basis vector the GSVD returns is arbitrary --
at a true double it is decided by whichever direction has infinitesimally smaller tension,
i.e. by roundoff and basis asymmetry. So a true double found twice can come back nearly
parallel or nearly orthogonal, essentially at random, and the test gives either verdict.

That does not matter, because of the asymmetry already in Phase D:

    parallel   -> collapse -> RECOUNT at the survivor -> 2      correct
    orthogonal -> merge    -> SUM 1 + 1                -> 2      correct

Both verdicts give the right total. The test is decisive exactly where distance is helpless
-- one simple eigenvalue found twice (parallel, collapse, recount to 1) against two simple
eigenvalues (orthogonal, sum to 2) -- and degrades gracefully on the case it cannot resolve.

## What was not checked

* **The sweep used the EXACT eigenvalues.** A real search only has `lam_hat` within
  `delta_res`. `1 - cos ~ d**2` says that perturbation should cost almost nothing, but that
  is an extrapolation, not a measurement.
* **One domain, one pair of modes, low lam.** The rectangle's modes are exactly L2-orthogonal,
  which is what makes PAIR's floor pure sampling noise. On a domain whose eigenfunctions are
  merely orthogonal-in-truth-and-approximate-in-basis, the floor may sit above `1/sqrt(n_int)`.
  The L-shape at a corner-adapted basis is the case to check.
* **Clustered spectra.** Section 14's caveat applies here too: everything above is at gaps of
  order `4*pi/area` on a benign domain.

## Consequence for the spec

`search.md` Phase D's collapse-versus-merge should be decided by the Gram angle, with distance
demoted to a prefilter that decides which pairs are worth testing at all. That is the first
place in this design where a decision currently made by a threshold on a derived length scale
can be replaced by one made on the actual object in question.

---

# 16. P and kappa: one is not delicate, the other is

Section 15 ended a run of parameters that turned out not to matter -- `G` works from 3 to 1e10,
`tau_gram` from 0.3 to 0.9. The operational definition needs two more, and the pattern breaks:
**P behaves like the others, kappa does not.** `notes/prom_coal.py`.

Both make a predicate decidable. `S` is defined as the local minima of `sigma_1`, with `mu_i`
the largest `mu` such that `sigma_j` has a local minimum at `lam_i` for all `j <= mu`. Neither
"has a local minimum" nor "at `lam_i`" survives contact with floating point unmodified.

* **P**, the PROMINENCE: the factor by which `sigma_1` must rise on both sides for a minimum
  to count. Without it, section 10's flat bottom -- half-width `eps*sqrt(2*eps_mach)/c` --
  supplies either nothing or a crowd.
* **kappa**, the COALESCENCE: how close `argmin sigma_1` and `argmin sigma_2` must be to count
  as "the same `lam_i`". At a real degeneracy the branches are numerically coincident but not
  bit-identical.

## Two traps in measuring P, both hit first

**Prominence measured over a window of half-width W is W/(eps/c) by construction.** For a cone
with a rounded bottom, `sigma_1(W)/sigma_1(0) = sqrt(1 + (cW/eps)**2)`. A first run used
windows of `10*eps/c` and `1*eps/c` and got 10.0 and 1.41 in every row of the table, on every
basis -- it was measuring the window, not the well. The intrinsic quantity is the whole well:
PLATEAU OVER FLOOR.

**A good basis has no flat bottom to measure.** 40 plane waves on a rectangle give
`eps = 7e-15` and a flat half-width of `1.8e-21`, twenty orders below the float spacing at
`lam ~ 19.5`. The well is conical all the way down. That is section 10's finding -- "a well
whose floor is at machine level never flattens" -- reappearing as an obstacle rather than a
result, and it means the roughness regime can only be reached by DELIBERATELY STARVING THE
BASIS.

Starved plane waves do it badly: `eps` jumps 1.7e-03 -> 9.9e-07 -> 1.9e-10 -> 1.0e-14 over
`nw` = 6, 8, 12, 16, so almost nothing lands in the narrow band where the flat bottom is both
reached and representable. **A Fourier-Bessel family at ONE corner is the better ladder** --
structurally deficient rather than merely small, and it steps smoothly:

    fb4   7.53e-03      fb6   1.69e-05      fb8   1.18e-08      fb14  1.81e-15
    fb5   2.65e-05      fb7   3.53e-08      fb10  3.25e-12

## P: twelve orders of margin

Thirteen bases, `eps` from 7.5e-03 to 7.0e-15. Window = 0.4 of the mean eigenvalue gap
(`4*pi/area = 12.4`), so the well is measured whole.

   basis        eps   plateau     eps/c  n_min  P (deepest)  plateau/eps
     fb4   7.53e-03  3.42e-01  6.44e-02      1     7.18e+01     4.53e+01
     pw6   1.68e-03  2.46e-01  2.03e-02      1     2.39e+02     1.47e+02
     fb6   1.69e-05  3.41e-01  1.45e-04      1     3.20e+04     2.02e+04
     pw8   9.87e-07  2.46e-01  1.20e-05      1     4.06e+05     2.49e+05
     fb8   1.18e-08  3.41e-01  1.02e-07      1     4.57e+07     2.88e+07
     pw12  1.94e-10  2.46e-01  2.35e-09      1     2.07e+09     1.27e+09
     fb10  3.25e-12  3.41e-01  2.79e-11      1     1.66e+11     1.05e+11
     fb14  1.81e-15  3.41e-01  1.56e-14      1     2.99e+14     1.89e+14
     pw40  7.04e-15  2.46e-01  8.55e-14      1     5.69e+13     3.49e+13

**Exactly one minimum in every row**, and `P` tracks `plateau/eps` within a factor of 1.6 --
as it should, since that is what the ratio means. The WORST case is the worst basis: `fb4`,
`P = 72`.

Now the roughness, window = 100 times the flat half-width:

     basis     flat w     h/ulp  n_min   max P - 1   med P - 1
       pw8   2.53e-13      35.5     72    7.62e-11    1.33e-11
      pw10   1.11e-13      15.6     84    2.38e-10    4.21e-11
       fb5   4.80e-12     675.5      0    no minima at all
       fb6   3.06e-12     431.0      0    no minima at all
       pw6   4.29e-10   60344.1      0    no minima at all
       fb7   6.39e-15       0.9     --    below float spacing

So the flat bottom gives EITHER a crowd of minima with `P - 1 <= 2.4e-10`, OR nothing at all,
depending on how many ulps wide the sample spacing is. Both are the same statement -- there is
no signal down there -- and section 10 reported both halves separately without connecting
them.

    true wells      P >= 72        (and >= 4.5e+04 for any basis worth using)
    roughness       P <= 1 + 2.4e-10

**Ten orders of margin, and `P - 1` at the roughness level is `eta` itself** -- section 4's
measured 1e-10 relative, arrived at by a completely different route. So the principled form is
`P = 1 + m*eta` with `eta` measurable per solve, and any fixed P from about 1.001 to 40 works.
**P = 2 -- a doubling -- sits 4e+09 times above the roughness and 36x below the worst true
well.** Not delicate, on the pattern of `G` and `tau_gram`.

## kappa: no margin by construction

`|argmin sigma_1 - argmin sigma_2|`, both located by bounded Brent on the branch squared,
against separation. `rect(1+delta, 1)`, 40 plane waves.

    delta     d(lam)        d/2    |Delta|  |Delta|/(d/2)     eps/c  |D|/(eps/c)
  1.0e-01  5.139e+00  2.569e+00  2.715e+00          1.057  1.74e-08     1.6e+08
  1.0e-02  5.834e-01  2.917e-01  2.960e-01          1.015  1.10e-08     2.7e+07
  1.0e-03  5.913e-02  2.956e-02  2.989e-02          1.011  3.78e-09     7.9e+06
  1.0e-04  5.921e-03  2.960e-03  2.991e-03          1.010  3.44e-09     8.7e+05
  1.0e-05  5.922e-04  2.961e-04  2.995e-04          1.012  3.41e-09     8.8e+04
  1.0e-06  5.922e-05  2.961e-05  2.993e-05          1.011  3.41e-09     8.8e+03
  0.0e+00  0.000e+00  0.000e+00  2.834e-08            nan  2.83e-09       10.01

**`|Delta|/(d/2) = 1.01` across six decades.** That is section 9.2 confirmed on an independent
instrument -- refinement of each branch rather than a grid argmin -- and tightened from "within
about 1%" to 1.010-1.057 with a clear trend toward 1 as `d` shrinks.

The last row is the one that matters. At an EXACT degeneracy the residual is not zero and not
at machine level: `2.8e-08`, about `10*eps/c`. And it is structural, not optimizer noise --
identical to four digits as `xatol` goes 1e-12 -> 1e-14 -> 1e-16:

     xatol  seed     |Delta|      eps/c  |D|/(eps/c)
     1e-12     0   2.834e-08   2.83e-09        10.01
     1e-14     0   2.834e-08   2.83e-09        10.01
     1e-16     0   2.834e-08   2.83e-09        10.01
     1e-14     1   3.486e-07   2.99e-09       116.44
     1e-14     2   8.176e-09   3.05e-09         2.68

**But it moves by 43x with the interior-point seed alone**, from `2.7*eps/c` to `116*eps/c`,
while `eps/c` itself moves by 8%. So the residual is a property of the particular collocation,
not a constant of the problem, and it cannot be predicted -- only bounded, loosely.

## Why kappa is different in kind

`kappa` has to satisfy, simultaneously,

    kappa > |Delta| at a true degeneracy      up to ~120 * eps/c observed
    kappa < d/2 for the tightest real pair    the resolution limit

There is no wide empty band between those the way there is between roughness and a true well.
It is a CONTINUUM: `|Delta|` runs from `~10*eps/c` at `d = 0` smoothly up to `d/2`, and `kappa`
is a cut across it. Every other parameter in this design separates two populations with orders
of empty space between them; this one separates a population from itself.

**And it sharpens the basis contract.** Feasibility requires `d_min/2 > 120*eps/c`, i.e.

    d_min  >  240 * (eps/c)

not `d_min > eps/c`. The operational definition's assumption was written as "`eps/c` smaller
than the smallest true gap"; the constant is **two to three orders, not one**. That is
consistent with section 9.4, where (C1) failed at `d/(eps/c)` = 0.45 and 0.15 and was correct
at 23 -- the boundary there was tens, and this pushes it to hundreds.

At the numbers measured here the margin is comfortable -- at `d = 5.9e-05`, `d/2 = 3.0e-05`
against `120*eps/c = 4.1e-07`, a factor of 70 -- but `eps` is 1.7e-10 there, a very good basis.
`fb4`'s `eps/c = 6.4e-02` would need `d_min > 15`, larger than the whole mean gap.

## What this changes

* `P = 2`, or `1 + m*eta` with `eta` measured per solve. Settled.
* `kappa` should NOT be a constant. It has to be `kappa = m * eps/c` with `eps` and `c`
  estimated per eigenvalue, `m` of order 100, AND the solver must report when
  `d_min/2` and `kappa` come within a decade of each other -- because that is the signal that
  the basis, not the search, is the binding constraint.
* The basis contract is `gap > 240 * eps/c`, and it is the one number the planner has to hit.

## What was not checked

One domain, one mode pair, low `lam`, and the `delta = 0` residual on three seeds only. The
43x seed spread is measured; whether 120 is a bound or just the largest of three samples is
not. That number wants ~30 seeds and a second domain before `m` is chosen.

---

# 17. The interior fill: one problem it solves, one it does not

Sections 15 and 16 both landed on the interior point set. The PAIR floor is `1/sqrt(n_int)`,
and the exact-degeneracy residual moved 43x with the seed. `interior_pts` fills by REJECTION
SAMPLING from the bounding box, so the discrete interior inner product is a MONTE-CARLO
estimate of `||u||_{L2(Omega)}` with a random realization. Setting a seed does not help: for a
domain nobody has solved before there is no way to know which seed is a good one.

`notes/interior_pts.py`. `Domain.interior_pts` already accepts `method="grid"`; `sobol` is a
Sobol sequence in the bbox, rejected to the domain, and is not in `src/`.

## The measurement that needs no search

On `rect(1+delta, 1)` the modes (2,1) and (1,2) are EXACTLY L2-orthogonal. So

    |cos| = |<u_a, u_b>| / (||u_a|| ||u_b||)

over the interior points is the quadrature error of the discrete inner product on a
known-zero integral -- a direct read of the point set's quality, with no reference to any
search, any threshold, or any eigenvalue.

  n_int | random, 12 seeds: med        min        max   spread |       grid      sobol
    100 |              4.99e-02   4.15e-03   2.41e-01       58x |   3.48e-03   7.84e-03
    200 |              3.16e-02   5.82e-04   1.52e-01      261x |   1.40e-03   6.79e-03
    400 |              5.93e-02   3.10e-03   1.15e-01       37x |   2.71e-03   5.16e-03
    800 |              2.58e-02   5.15e-03   7.63e-02       15x |   1.29e-04   2.12e-04
   1600 |              2.23e-02   2.11e-03   5.18e-02       25x |   6.27e-04   3.18e-04
   3200 |              6.73e-03   1.40e-03   2.62e-02       19x |   9.14e-05   5.21e-04
   6400 |              8.96e-03   2.20e-04   1.72e-02       78x |   1.77e-04   9.78e-07

**30x to 100x better at every count, and DETERMINISTIC** -- the spread column simply does not
exist for the other two. A first version of this table fitted a rate to ONE random realization
per `n` and got `r = 1.28` against a theoretical 0.5; with 12 seeds the medians give 0.74.
That is the same trap as sections 7 and 12 and it was walked into again.

**The sobol RATE is not established.** Its values run 2.12e-04, 3.18e-04, 5.21e-04, 9.78e-07
over n = 800 to 6400 -- not monotone, and the fitted `r = 2.34` is carried entirely by the last
point, which may be a power-of-two equidistribution effect since the generator is asked for
`2**ceil(log2(n*factor))` points. What is solid is the LEVEL and the determinism, not the
exponent.

## The alignment worry was wrong

A grid on an axis-aligned rectangle is a tensor rule against separable integrands, which
should be unfairly favourable, and a fill whose quality depends on the domain lining up with
its bounding box is useless in a black box. Rotating the domain tests it. `n_int = 1600`:

   theta  bbox/area |  random (med of 12)        grid       sobol
      0d      1.000 |            2.23e-02    6.27e-04    3.18e-04
      5d      1.174 |            1.86e-02    2.14e-04    6.76e-04
     15d      1.500 |            1.96e-02    5.34e-04    1.13e-03
     30d      1.866 |            2.67e-02    2.22e-04    5.29e-04
     45d      2.000 |            1.59e-02    1.45e-03    6.72e-04

No systematic degradation for either deterministic fill, out to `bbox/area = 2.0` where half
of every rejection draw is thrown away. The hypothesis was reasonable and it is refuted.

## What it buys: the located eigenvalue

The number `douse` actually consumes. Isolated mode (1,1) of rect(1.01, 1), `lam` known in
closed form, located by Brent on `sigma_1**2`:

    random, 12 seeds   rel err  med 2.91e-10   min 1.79e-11   max 1.97e-09   spread 110x
    grid                        1.38e-12
    sobol                       6.69e-12

**Two hundred times more accurate than the median random fill, and the 110x run-to-run spread
goes to zero.** `eps` is 5.9e-15 to 9.3e-15 in every case, so the fill is not changing what the
basis can represent -- it is changing how faithfully the tension ratio reflects the L2
normalisation, and therefore where the minimum sits.

There is a second argument that does not depend on accuracy at all. **An unseeded random fill
makes the eigenvalue a non-deterministic function of the domain.** Re-solving a perturbed shape
then returns an objective with 1e-9-relative noise on it, and `mps_symmetry.md` section 4 is
already about the optimizer losing near optimizers. A deterministic fill removes a noise source
from the shape objective for free.

## What it does NOT buy: the coalescence residual

The payoff test for section 16, and it fails. `rect(1, 1)`, modes (2,1) and (1,2) exactly
degenerate, `|argmin sigma_1 - argmin sigma_2|` normalised by `eps/c`. `random` varied over
seeds; the deterministic fills have no seed, so they are varied over ROTATION.

    random, 12 seeds       med    2.3   min   0.1   max  116.4   spread   1338x
    grid,   5 rotations    med   84.7   min   6.3   max  170.8   spread     27x
    sobol,  5 rotations    med    1.3   min   0.0   max  110.3   spread   11712x
    random, 5 rotations    med    0.5   min   0.1   max   10.0   spread     115x

**Improving the inner product by 30-100x changed nothing here.** The two quantities are
decoupled, and in hindsight they should be: at an exact degeneracy the residual is the
splitting of a two-dimensional eigenspace by the asymmetry of the WHOLE discretization --
boundary quadrature and basis included -- and it lives at the scale of the tension floor
`eps`, not at the scale of the interior quadrature error. Nothing about the interior fill
removes that asymmetry.

Note also that `grid` has the WORST median (84.7) while having the smallest spread, so it is
not even monotone in the direction one would guess.

## Conclusions

* **Switch the default interior fill to a deterministic low-discrepancy set.** `grid` is
  already implemented and measured; `sobol` is comparable at moderate `n` and better at large
  `n`. The wins are a 30-100x better inner product, a 200x better located eigenvalue, zero
  run-to-run spread, and a shape objective that stops being noisy. `method="random"` should
  stay available, because a deterministic fill is exactly the thing that can be pathologically
  aligned with some future domain, and having the alternative is how that would be diagnosed.
* **The section 15 Gram angle gets a directly better floor**, which widens `tau_gram`'s
  already-wide margin. `1/sqrt(n_int)` is the MC floor and it is not the floor of a
  deterministic fill.
* **`kappa` is not fixable this way and remains delicate.** The bound on the residual widens
  slightly with the larger sample -- max 171 across all fills and perturbations, against 116
  from three seeds -- which strengthens rather than weakens section 16's conclusion that
  `kappa = m * eps/c` needs `m` of order 100 and that the solver must REPORT when `d_min/2`
  approaches it.

## What was not checked

One domain. Rejection from the bounding box is what degrades a low-discrepancy set, and the
rectangle is the best possible case at `theta = 0` and still benign at 45 degrees. An L-shape
(area/bbox = 0.75), and worse a domain with a thin protrusion, is where a Sobol net after
rejection could stop being low-discrepancy at all. That is the test to run before this changes
a default.

## Scope: this is a determinism change, NOT an interior-cubature project

Recorded because everything above reads as "the interior inner product got 100x better, get
more of that", and that is the wrong lesson.

**Accurate interior inner products for eigenfunctions are known to be very hard, and chasing
them by interior cubature is a dysfunctional path.** The MPS interior block does not need an
accurate `||u||_{L2}`: its job is to bound `||u||` away from zero so the trivial and
near-trivial solutions cannot win the ratio. A crude lower bound is what the formulation asks
for, and random interior points are the standard choice precisely because that is all that is
needed. Nothing above changes that.

So the recommendation is **swap the FILL, at unchanged `n_int`**, and the argument is
reproducibility:

* the same domain returns the same eigenvalue twice;
* the shape objective stops carrying 1e-9-relative noise, which is `mps_symmetry.md`
  section 4's problem;
* the section 15 Gram angle stops having a random floor.

The 30-100x and the 200x are how that change happens to land at fixed cost. They are NOT a
reason to raise `n_int`, and they are not a step toward computing L2 norms on interior points.

**The accurate path for eigenfunction norms and orthonormalization is the RELLICH IDENTITY**,
which converts `int_Omega u**2` into boundary data -- `docs/design/boundary_integrals.md`
section 1, unbuilt. That is a BOUNDARY quadrature problem, where the adaptive corner-aware
machinery already exists and the singularity structure is understood, and it is the right home
for any quantity that has to be accurate rather than merely non-degenerate.

Section 15's Gram angle should therefore be read as an INTERIM instrument. It works on the
interior points because it only needs to tell `|cos| ~ 0` from `|cos| ~ 1`, a discrimination
with ten orders of margin that a crude inner product settles easily. When the Rellich layer
lands, the same test should move onto it and stop depending on the fill at all.

Section 16's negative result is evidence for the same caution, from the other direction:
improving the interior inner product by two orders bought **nothing** for the coalescence
residual, the one quantity that actually needed help.

## The L-shape number, which decides it -- and it reverses the sobol/grid ranking

The rectangle's mode-pair test does not generalize, but the instrument does: **eigenfunctions
at DISTINCT eigenvalues of the same Dirichlet Laplacian are exactly L2-orthogonal**, whatever
the domain. So any two entries of a `reference` table give the same known-zero integral, on any
domain with a table and no closed form at all.

Validated first by reproducing the numbers it was derived from -- rect(1.01, 1), n_int = 1600,
three distinct-eigenvalue pairs instead of the degenerate mode pair:

    method   |cos|(lam1,lam2)   |cos|(lam1,lam3)   |cos|(lam2,lam4)
    random           2.66e-02           2.09e-02           1.95e-02
      grid           9.65e-05           8.53e-05           5.43e-05
     sobol           6.65e-04           7.36e-04           3.76e-04

Same orders as the mode-pair table. The instrument is sound.

**L-shape**, `bbox/area = 1.333`, pure Fourier-Bessel at the reentrant corner with n = 80 --
`reference`'s own recipe, tension 2.3e-15 to 1.7e-14 at the six tabulated eigenvalues, so the
basis is not what is being measured. Median over five eigenvalue pairs:

  n_int | random, 12 seeds: med        min        max   spread |       grid      sobol
    200 |              4.66e-02   3.08e-02   9.60e-02        3x |   5.93e-04   9.00e-03
    400 |              2.94e-02   5.04e-03   9.50e-02       19x |   1.97e-05   2.82e-03
    800 |              2.85e-02   8.24e-03   5.20e-02        6x |   4.05e-04   3.16e-03
   1600 |              2.11e-02   1.08e-02   4.19e-02        4x |   2.17e-06   5.09e-04
   3200 |              1.60e-02   7.18e-03   4.01e-02        6x |   5.59e-05   1.51e-04

`grid` is 1.5 to 4 orders better than random and holds up through the rejection. But it is
badly NON-MONOTONE in `n_int` -- 5.9e-04, 2.0e-05, 4.1e-04, 2.2e-06, 5.6e-05 -- so the right
question is not its median but its WORST case. Swept over 28 values of `n_int` from 300 to
3000:

    grid    med 1.33e-04   best 2.17e-06   WORST 8.67e-04   swing 399x
    sobol   med 6.20e-04   best 7.47e-05   WORST 5.02e-03   swing  67x
    random  med ~2e-02

**`grid` at its worst is still 23x better than random at its median**, and never comes near it.
The erraticism is real and it is bounded well below the level that motivated the change.

**And the ranking reverses from the rectangle.** `sobol` won there at large `n` (9.8e-07 at
n = 6400); here its worst case is 5.0e-03, only 4x better than the random median, which is not
a margin worth having. Rejection from the bounding box is what degrades a low-discrepancy net,
and at `bbox/area = 1.333` it already does. The rectangle at `theta = 0` is the one case where
rejection does nothing at all, so sobol's advantage there was an artifact of the easiest
possible domain -- exactly the transferability check that section 8's methodological note asks
for, and it fails.

**So: `grid`, not `sobol`.** The concrete change is `MPSEigensolver.__init__`'s convenience
path, `src/chladni/mps/solver.py:61`, from the default `method="random"` to `method="grid"`.
`Domain.interior_pts`'s own default stays, because `plotting` wants random scatter and thirty
test call sites depend on it. `sobol` should not be added to `src/` on this evidence.

Still two domains, both polygons, both at `bbox/area <= 1.34`. A thin sliver or a spiral, where
rejection throws away most of every draw, is untested and is where `grid` would be expected to
fail next.

---

# 18. The grid recommendation was wrong: it annihilates a mode exactly

Section 17 recommended switching the interior fill to `method="grid"`. **That recommendation is
withdrawn.** It was made on a metric that could not see the failure mode that matters.

The interior block's PRIMARY job is not accuracy. The tension is `||A_B c|| / ||A_I c||`, and
the interior points exist to keep the DENOMINATOR away from zero. If they lie on the nodal set
of the target eigenfunction, `||A_I u||` collapses for a function that is not small, the
tension is inflated, and the eigenvalue is missed outright. Random points make that
probabilistically impossible. **A grid can be commensurate with a nodal pattern, and this one
is.**

Section 17 measured the ACCURACY of the discrete inner product on two well-behaved functions.
That is a different quantity from its WORST-CASE degeneracy over all modes, and `grid` is
excellent at the first while failing completely at the second. Another entry for section 8: the
metric was chosen before the failure mode was, and it flattered the alternative.

## The mechanism is exact

`Domain.interior_pts(method="grid")` sets `side = ceil(sqrt(4*n_int))` and puts the x-values at
`k/(side+1)`, `k = 1..side`. The mode `sin(m*pi*x)` has nodal lines at `x = l/m`. At
`m = side+1`,

    sin((side+1) * pi * k/(side+1)) = sin(k*pi) = 0     for EVERY grid point

so the mode `(side+1, side+1)` is annihilated to machine zero on the whole point set.
Subsampling to `n_int` does not help, because every one of the candidate points is on a nodal
line. `rect(1, 1)`, `R = mean(u**2 over the points) / (1/4)`, where `R = 1` is a fair sample:

  n_int |  grid: worst R        at |   random: worst R (12 seeds) |      sobol
    200 |       1.43e-56  (30, 30) | med    0.759  min    0.702 |      0.698
    400 |       2.17e-56  (41, 41) | med    0.821  min    0.786 |      0.723

**`1e-56` against `0.70`.** Random never comes near degeneracy at any count or seed, which is
exactly the property it is there for. Sobol is safe too, at 0.70 to 0.81.

## The apparent escapes are an artifact of how high I looked

A first version of this table swept `m, n = 1..50` and showed `grid` at 0.85-0.90 for
`n_int >= 800`, which reads as "safe above some size". It is not. Extending to `m, n = 1..100`:

    n_int  side+1   worst R, mmax=50     mmax=100         at
      400      41           2.17e-56     2.17e-56   (41, 41)
      800      58           8.54e-01     1.39e-56   (58, 58)
     1000      65           9.74e-01     2.87e-56   (65, 65)
     1225      71           9.15e-01     1.76e-56   (71, 71)
     1600      81           8.96e-01     1.67e-54   (81, 81)
     3200     115           8.93e-01     8.93e-01    (8, 87)

Every escape was the search range failing to reach `side+1`. The `n_int = 3200` row escapes
only because `115 > 100`. **There is always an exactly-annihilated mode; whether you see it
depends on arithmetic nobody chose deliberately.**

In the realistic low range, `m, n <= 15`, `grid` is fine -- 0.85 to 1.02, against random's 0.81
to 0.94 -- so the damage is confined to `m ~ side+1`, at `lam = 2*pi**2*(side+1)**2`, about
3.3e+04 for `n_int = 400`. High, but reachable, and a black-box solver cannot be asked to check
whether its scan range collides with the factorization of `ceil(sqrt(4*n_int))+1`.

## What to do instead

The reproducibility problem section 17 measured is real -- 110x run-to-run spread in the
located eigenvalue, and a shape objective carrying that as noise. But `grid` was the wrong fix,
because determinism and non-alignment are not actually in conflict:

**Change `MPSEigensolver`'s `rng=None` default to a fixed constant.** That is reproducible
(same domain, same answer), still pseudorandom (no commensurability with any nodal pattern),
and deforms about as continuously as rejection sampling allows -- the same underlying uniform
draws, with the accept/reject set changing only for points near the boundary.

The objection to a fixed seed was that we cannot know which seed is good for an unseen domain.
The measurement answers it: **no seed is bad.** Worst-case `R` runs 0.70 to 0.91 over 12 seeds
and every count, and the located eigenvalue runs 1.8e-11 to 2.0e-09 relative -- a 110x spread,
but every member of it clears the 1e-08 target. The seed choice affects accuracy within a band
that is entirely acceptable; it cannot produce a catastrophe. `grid` can.

Do NOT seed from a hash of the domain's geometry, which is the obvious alternative: a tiny
shape perturbation would change the seed and hence every point, making the objective
discontinuous in shape -- the same defect `mps_symmetry.md` section 2 rejects the auto-swap
for.

## The record

Section 17's measurements stand as measurements. Its conclusion does not. The generalized
orthogonality instrument (distinct eigenvalues of any domain are L2-orthogonal) is still the
right tool for what it measures, and `grid` really is 30x to 1e4x more accurate -- it is simply
that accuracy was never the binding constraint on this component, and a recommendation was
made before asking what the component is for.

---

# 19. kappa is not m*eps/c, and the 240 contract is withdrawn

Section 16 wrote the coalescence tolerance as `kappa = m * eps/c` and took `m ~ 120` from the
max of THREE seeds, then derived a basis contract `gap > 240 * eps/c`. Both are wrong, and the
error is in the functional form rather than in the constant.

Extending to 30 seeds on three domains with protected degeneracies gave `|Delta|/(eps/c)` of
1.8e+04 on rect(1,1), 9.5e+06 on the disk and 1.7e+04 on eq_tri -- against section 16's 10.
Three orders apart on the same quantity is not sampling noise; it means the normalization is
wrong. The absolute splitting was about the same in both, and it was `eps` that had moved,
because the two runs used different `lam_ref` and therefore different rank truncation.

## The splitting does not depend on eps at all

`rect(1, 1)`, exact degeneracy, one basis ladder per family, 10 seeds each:

   basis        eps       c      eps/c |    |D| med    |D| max |  |D|/(eps/c)    |D|/lam
    pw16   1.65e-09   0.049   3.40e-08 |   9.10e-08   3.13e-07 |          2.7   1.84e-09
    pw24   2.53e-13   0.049   5.21e-12 |   9.14e-08   3.13e-07 |      17538.1   1.85e-09
    pw32   2.56e-13   0.049   5.27e-12 |   9.20e-08   3.14e-07 |      17462.3   1.86e-09
    pw40   2.55e-13   0.049   5.26e-12 |   9.09e-08   3.12e-07 |      17296.2   1.84e-09
     fb6   4.56e-04   0.069   6.65e-03 |   3.64e-04   3.65e-04 |          0.1   7.37e-06
    fb10   2.05e-09   0.069   2.99e-08 |   1.94e-07   3.83e-07 |          6.5   3.93e-09
    fb16   8.48e-15   0.069   1.24e-13 |   1.94e-07   3.81e-07 |    1567136.8   3.92e-09

**`eps` moves four orders across pw16-pw40 and six across fb10-fb16; `|D|` is constant to 1%.**
It is a property of the basis family and the boundary discretization, not of the tension floor.
`|D|/(eps/c)` therefore spans 0.1 to 1.6e+06 and means nothing -- section 16's `m ~ 10-171` was
one point on that range, read as a bound.

The stable normalization is RELATIVE: `|D|/lam` is 1.84e-09 for plane waves and 3.9e-09 for
Fourier-Bessel, constant within each family across every basis size, and within 2x of each
other across families. Only `fb6` breaks it, and `fb6` has `eps = 4.6e-04` -- a basis that
cannot represent the mode at all.

## Consequences, all in the direction of a tighter definition

* **`kappa` needs no new constant.** A relative tolerance is what Phase D's second rule already
  is -- "diameter <= `tol_merge * x`". A discretization-split degeneracy is a diameter of about
  `4e-09 * lam`; setting `tol_merge` two decades above that at `1e-07` merges it. `kappa`
  collapses into `tol_merge` and stops being a separate parameter.
* **The `gap > 240 * eps/c` contract is withdrawn.** It was derived through the wrong form. The
  basis contract that survives is the one measured DIRECTLY in section 9.4: (C1) is correct at
  `d/(eps/c) = 23` and wrong at 0.45 and 0.15, so **`gap > ~20 * eps/c`**. An order of
  magnitude, not two-and-a-half, and it comes from a measurement of the thing itself rather
  than from a chain.
* **Clearance is real but not generous.** `tol_merge = 1e-07` against pairs measured resolvable
  down to about `3e-06` relative leaves a factor of 30. That is a number to report per solve,
  not to assume.

## The methodological point, which is the same one as sections 7, 12, 14 and 18

`m` was taken as the max of three samples of a ratio whose denominator was not the right scale.
Every symptom of that was visible at the time -- the 43x seed spread, the fact that `|Delta|`
sat 100x outside the well it was supposedly a tolerance within -- and it was written into a
contract anyway. **A constant derived through another quantity should be checked against a
direct measurement of what it is supposed to bound before it is used.** Section 9.4 already had
that direct measurement and it was not consulted.

---

# 20. Tension curves on real corner domains

Everything before this measured the surrogate, a rectangle with plane waves, or the L-shape.
The solver has to work on domains with SEVERAL reentrant corners, where Fourier-Bessel at those
corners is what makes it work at all. `notes/curve_lab.py` builds the bases,
`notes/curve_study.py` runs the studies.

**A corner is eigenfunction-singular when `pi/omega` is not an integer.** `Corner.is_singular`
means "geometric corner", a much larger and different set -- all four corners of a rectangle
are `is_singular` and none of them needs a Fourier-Bessel family. The basis builder keys on the
fractional exponent instead.

The roster, with plane waves sized by `n_useful(k*R)` plus 40 FB terms per non-analytic corner,
and its tension at the tabulated eigenvalues -- so the bases are known adequate before anything
is concluded from the curves:

    L_shape   134 cols  1 reentrant                  1e-08 .. 2e-11
    mushroom  168 cols  2 reentrant                  7e-10 .. 2e-10
    GWW1      272 cols  2 reentrant + two 3pi/4      3e-09 .. 1e-09
    H_shape   258 cols  4 reentrant                  5e-08 .. 2e-08
    plus      248 cols  4 reentrant, D4              (no table)
    chevron   252 cols  an 18-degree spike           2e-07 .. 5e-08
    rect81    176 cols  no singular corner, control  (closed form)

`cut_square` was dropped: `reference`'s recipe is an ORIGIN-CENTRED Fourier-Bessel expansion of
320 terms, which the corner-adapted builder does not offer, and plane waves leave the tension
at 1e-02. A basis-planning gap, not a curve finding.

## 20.1 The plateau is remarkably stable; the floor is not

    domain     plateau   floor min   floor max   SPREAD
    L_shape     0.1981     3.6e-11     9.1e-07   2.5e+04
    mushroom    0.1829     2.2e-10     1.4e-08   6.4e+01
    GWW1        0.1228     1.0e-10     2.6e-08   2.5e+02
    H_shape     0.2251     2.7e-12     1.2e-07   4.4e+04
    plus        0.2131     1.1e-11     3.7e-08   3.4e+03
    chevron     0.3555     4.9e-08     2.3e-07   4.7e+00
    rect81      0.1541     5.3e-07     4.9e-02   9.2e+04

**The plateau spans 0.12 to 0.36 across seven domains** -- a factor of 2.9, across areas from
1.0 to 14.0 and from zero to four reentrant corners. That is the strongest support yet for
`tau_scan` as a FRACTION OF THE PLATEAU, and against any absolute tension threshold.

**The floor spreads four to five orders WITHIN a single scan on three of seven domains.** The
L-shape's five orders was not special. This is the direct justification for clause (2) of the
operational definition being per-eigenvalue: one `prec` target cannot be met uniformly when the
attainable floor moves 1e+05 across the interval being scanned.

`rect81` is the informative failure: floor max 4.9e-02, which is no dip at all. A high aspect
ratio with plane waves is a genuinely inadequate basis, and it is the only roster member whose
audit flags for that reason rather than for multiplicity.

## 20.2 Six samples per gap is NOT enough

`search.md` section 6 says "roughly six samples per eigenvalue gap always place three inside
every dip, and Weyl's law sizes that directly." That was inferred from the width/gap invariant
on four benign domains. Measured END TO END -- subsample the fine scan at stride `h`, count how
many true dips survive as discrete minima:

    domain     recall=1.000 down to    at ~6 per gap    at ~4 per gap
    L_shape         4.8 per gap            1.000            0.889
    mushroom        5.0                    1.000            0.917
    GWW1            8.0                    0.933            0.867
    H_shape         9.1                    0.875            1.000
    plus            4.4                    1.000            1.000
    chevron        10.1                    0.667            0.667
    rect81         17.7                    0.929            0.929

**Full recall on every domain needs about 18 samples per median gap, not six.** At six, four of
seven domains lose eigenvalues.

The mechanism is that the invariant is about the MEAN gap and recall is bounded by the MINIMUM
LOCAL gap. Measured `min observed gap / Weyl mean gap`:

    mushroom 0.127   H_shape 0.210   rect81 0.295   chevron 0.336
    L_shape  0.571   plus    0.740   GWW1    0.792

**The local gap runs 1.3x to 8x tighter than Weyl's mean.** A stride computed from Weyl is
therefore an estimate of the wrong quantity, and no a-priori stride can be safe -- which is the
case for correcting it IN THE LOOP rather than choosing it once.

Note also that the half-width/gap ratio itself is not the near-constant of section 14. Measured
consistently as the half-width at HALF the plateau, it runs 0.13 to 0.39 across the roster --
a 3x range, with `H_shape` and `chevron` at the top. Section 14's 0.46-0.52 used a different
definition (full width at the plateau) on four benign domains; converted, the roster runs about
0.5 to 1.6. The invariant is much looser than one domain family suggested.

## 20.3 The Weyl audit is a POST-MULTIPLICITY check, and reads as broken before it

Study 2 ran the audit at each stride. On four of seven domains it flagged `MISS` at recall
1.000, which looks like a false positive and is not: **`sigma_1` shows one dip per LEVEL and
Weyl counts WITH MULTIPLICITY**, so the deficit was reporting exactly the multiplicity that had
not been estimated yet.

Closing the loop -- (C1) with `G = 10` plus (C2) at prominence `P = 2` on a wide bracket, then
auditing the UNFOLDED list:

    domain     dips   sum mult   deficit on dips   deficit after multiplicity
    L_shape       9         10              0.41                         0.41
    mushroom     12         12              0.59                         0.59
    GWW1         15         15              0.33                         0.33
    H_shape       8          8              1.18                         1.18
    plus          8         10              2.79                         0.79
    chevron       3          3              0.12                         0.12
    rect81       14         14              2.55                         2.55

`plus` is the case that proves it: D4 symmetry, two doubles found, and the deficit falls from
2.79 to 0.79. **(C1)+(C2) works on a domain with four reentrant corners and a corner-adapted
basis** -- section 9.3's 9-for-9 was on the disk and the equilateral triangle, both smooth.

The residual flags are then real and each names a different fault:

* `rect81` at 2.55 -- an inadequate BASIS, floor 4.9e-02 at some levels.
* `H_shape` at 1.18 -- a genuine miss; its minimum observed gap is 0.378 against a Weyl mean
  of 1.795, so there is a tight pair the stride merged.
* `plus` at 0.79 -- one more level or multiplicity not yet accounted.

And `chevron` reads 0.12 at recall 0.667. **The audit is blind here**, because three distinct
levels give too few cuts and every cut lies below the top level. That limitation is documented
in `asymp.weyl_deficit` and this is it firing on a real domain.

## 20.4 What this changes

* `tau_scan` as a fraction of the plateau is measured across seven domains, not assumed.
* The per-eigenvalue accuracy clause is necessary, not defensive: 1e+05 floor spread in one
  scan.
* **The "six samples per gap" rule in `search.md` section 6 is wrong and should be replaced.**
  About 18 per MEDIAN gap reaches full recall on this roster, and even that is a fitted number
  rather than a bound, because it is the minimum local gap that governs.
* **The Weyl audit must run on the UNFOLDED list, after multiplicity.** Run on dips it reads
  as a false alarm on every symmetric domain, which is exactly the class the solver will meet
  at the optimizers of a shape optimization.
* The audit needs enough distinct levels to have cuts. Below about five it says nothing, and
  `chevron` shows it saying nothing while a third of the spectrum is missing.

---

# 21. An approach tested: the Weyl deficit as a LOCALIZER, not a verdict

Section 20.2 says no a-priori stride is safe, because recall is bounded by the minimum local
gap and that runs 0.13 to 0.79 of Weyl's mean. So the stride has to be corrected in the loop.
`notes/adaptive_scan.py` tests one way to do it.

**The idea.** The Weyl deficit is currently a scalar reported at the end -- "worst over the
gaps". Make it a PROFILE over cuts, so it says not only THAT a level is missing but WHERE, and
re-scan only that interval. Cost then goes as the number of tight spots rather than as the
whole range.

Scored against the section 20 fine scans as truth, `s0 = 8` (one sample per Weyl gap / 8),
prominence `P = 2`, `G = 10`, `tau_scan = 0.5 * plateau`:

    domain    truth  found  recall  spurious  deficit  evals   fine-scan n
    L_shape       9      9   1.000         0     0.13    205           317
    mushroom     12     11   0.917         0     0.47    238           348
    GWW1         15     15   0.867         2    -0.09    326           483
    H_shape       8      8   0.875         1     0.81   1556           376
    plus          8      8   0.875         1     0.43    206           355
    chevron       3      3   0.667         1    -0.78     84           154
    rect81       14     17   1.000         0     0.94   4094           582

## The bug it found first was mine, and it is worth recording

The first version anchored the profile by subtracting `weyl_count(a)`, on the reasoning that
the count is only meaningful relative to the bottom of the scanned range. **The three-term
count is an ASYMPTOTIC and goes NEGATIVE at the low end** -- measured `weyl_count(a)` of
-0.230 on L_shape, -0.687 on H_shape, -0.752 on rect81, -1.041 on chevron. Subtracting it ADDS
that much to every deficit, so the loop chased a phantom:

    plus      1794 evaluations -> 206      8.7x, and the deficit reads 0.43 instead of 0.89
    L_shape   deficit 0.36 -> 0.13
    GWW1      deficit -0.01 -> -0.09

The count is absolute, so the correct form subtracts nothing and instead REQUIRES the found set
to be a complete prefix from the bottom of the spectrum -- the same precondition
`asymp.weyl_deficit` already carries, and it should be asserted rather than papered over with
an anchor that looks principled and is not.

## Two structural flaws, both real

**(1) A cluster miss is AT a level, not BETWEEN levels.** The localizer re-scans the interval
between the two found levels straddling a hot cut. But when the missing eigenvalue is the
second member of an unresolved pair, it sits at the SAME location as a found level, and no
amount of re-scanning between levels will ever reach it. That is `mushroom` (recall 0.917,
deficit 0.47, minimum observed gap 0.352 against a Weyl mean of 2.771) and `H_shape` (0.875,
deficit 0.81, minimum gap 0.378 against 1.795). H_shape got MORE expensive after the fix,
958 -> 1556, because the corrected deficit is real and the loop keeps trying to fix it by the
one mechanism that cannot.

So the deficit response has to be TWO-PRONGED:

    between levels   re-scan the interval        a genuinely unsampled dip
    AT a level       re-examine multiplicity     an unresolved cluster

and only the first is implemented.

**(2) No basis-adequacy termination guard.** `rect81` spends 4094 evaluations -- 7x the fine
scan -- and still reports a deficit of 0.94, because its floor reaches 4.9e-02 at some levels
and no stride can fix a basis. This is the unbounded-refinement stall, and the predicate that
stops it already exists: `gap > 20 * eps/c`. Measured on H_shape at every level, `gap/(eps/c)`
runs **2.2e+06 to 1.4e+11** -- massively adequate, so the guard correctly does NOT fire there
and correctly separates "refine harder" from "the basis cannot do this". That discrimination is
the thing the loop most needs and it is one predicate away.

## Two cautions on the scoring itself

**`chevron` reads deficit -0.78 at recall 0.667.** The audit says clean while a third of the
spectrum is missing, because three distinct levels give too few cuts and every cut lies below
the top level. Section 20.3 saw this and here it is inside a control loop, where it is worse:
the loop TERMINATES on it.

**The truth is basis-dependent.** "Truth" here is the fine scan's dips below `0.5 * plateau`.
On `rect81` the floor reaches `4.9e-02`, about a third of the plateau, so real dips fall below
that cutoff and the adaptive scan finding 17 against a "truth" of 14 is the truth
undercounting, not the search over-reporting. Any recall number on a domain whose basis is
marginal is measuring the cutoff as much as the search.

## Verdict

The localizer is worth keeping -- it is what turned `plus` into 206 evaluations, and on the
three domains where the miss is genuinely a resolution problem it reaches the fine scan's
recall at 0.6x its cost. It is not sufficient on its own, and the two additions it needs are
both specified above and neither is speculative: the at-level prong, and the `20*eps/c` guard.

What it should NOT be is a verdict. On `chevron` the deficit is a termination condition that
fires while the answer is wrong.

---

# 22. Six approaches tested; two survive, three die, one changes its meaning

`notes/approaches.md` enumerates 27 candidates. Six were picked for testing because each one,
if it failed, would remove a whole family. `notes/test_approaches.py`.

The reframing that generated the list is worth stating first: **this is not global
optimization.** We need `{x : x is a local minimum of sigma_1 with prominence >= P}` with a
multiplicity at each and a certificate that none was missed. That is a LEVEL-SET problem with a
local-structure problem inside each component, and the hard half is the CERTIFICATE. Methods
that converge fast to one minimum are mostly irrelevant; methods that COUNT or ENCLOSE are
relevant even when they locate nothing.

## A2 -- Weyl-stratified sampling: FALSIFIED

Sample equispaced in `N(lam)` rather than in `lam`, so points concentrate where the density is
high. Measured against uniform at equal point count, on all seven domains at n = 40, 60, 90,
130: **28 comparisons, and stratification is worse or equal in 22 of them.** At n = 130 it is
worse on four of seven.

The reason is structural and kills the family: `N(lam)` is nearly LINEAR in `lam`, so
stratifying by it is nearly uniform sampling, and the deviations it does introduce are not
aligned with where the local gaps are tight. **Local clustering is a fluctuation phenomenon --
level repulsion, accidental near-degeneracy -- not a density phenomenon, and Weyl's law
describes only the density.** No a-priori schedule computed from Weyl can concentrate samples
where they are actually needed.

## A6 -- Lipschitz calibration: UNSOUND at small pilots

`L = gamma * max|ds/dlam|` over `n_pilot` points, `gamma = 2`, against the true maximum slope
from the fine scan. Ratios above 1 mean the certificate holds:

    domain     true max    n=20    n=40    n=80   n=160
    L_shape      0.2013   1.894   1.989   1.996   2.000
    mushroom     0.3589   1.524   1.945   1.974   1.995
    GWW1         0.7612   0.672   1.974   1.988   1.999
    H_shape      0.2438   1.886   1.941   1.985   1.991
    plus         0.2738   1.939   1.986   1.996   1.999
    chevron      0.0472   1.993   2.000   2.000   2.000
    rect81       0.2264   1.917   1.945   1.951   1.961

**`GWW1` at `n = 20` reads 0.672: the pilot underestimates the true slope even with
`gamma = 2`, so the certificate is FALSE.** It is the domain with the densest spectrum, and
the pilot simply never lands on a steep wall. Everything converges to 2.000 by `n = 40`, so
the failure is not asymptotic -- it is that "a fine enough pilot" is domain-dependent and not
knowable in advance, which is section 20.2's stride problem wearing a different hat.

## A7 -- DIRECT, Lipschitz-free: STRONG

Keep every interval that is potentially optimal for SOME Lipschitz constant -- the lower-right
convex hull of (width, value) -- plus every interval already below `tau`. No constant to
calibrate, so A6's failure mode cannot occur. At about 230 evaluations:

    domain    DIRECT recall   uniform recall at the same n
    L_shape           1.000                          0.778
    mushroom          1.000                          0.917
    GWW1              0.800                          0.867
    H_shape           0.875                          0.625
    plus              1.000                          0.875
    chevron           1.000                          0.333
    rect81            1.000                          0.857

**Six of seven at or above uniform, mean recall 0.95 against 0.75**, and it beats section 21's
audit-guided loop on `mushroom`, `chevron` and `plus` while costing 7x less than it on
`H_shape` and 18x less on `rect81`. Caveat: this simulation interpolates from the cached fine
scan, so it does not pay real evaluation cost and the interpolation smooths the curve. The
structural result -- no constant needed, and it beats uniform -- does not depend on that.

## A11 -- the argument principle on `det A_B`: FALSIFIED

With a square collocation, `det A_B(lam)` is analytic and its sign changes count the
odd-multiplicity roots. The determinant method is known numerically bad, but that objection is
about the MAGNITUDE, and counting needs only the SIGN -- which comes from LU pivot parity and
is a different question.

It is not a different question. On 900 samples:

    L_shape    log|det| spans 214 orders   464 sign flips for 9 eigenvalues   455 spurious
    mushroom   log|det| spans 163 orders   456 sign flips for 12 eigenvalues  444 spurious

Every true eigenvalue does have a flip near it -- 9/9 and 12/12 -- but at 2% precision that is
worthless. **The conditioning destroys the determinant's topology, not merely its magnitude.**
That is a stronger statement than the usual objection to the determinant method, and it kills
A11 and most of A12 (Beyn on a square MPS collocation) with it, because they rest on the same
matrix.

## A14 -- the exact `k`-derivative: CONFIRMED

Every particular solution here is `Psi(k*x)` about its own centre, so Euler's identity gives
`d(psi)/dk = (x . grad psi)/k`. Measured against a central difference:

    PlaneWave    3.12e-10       (FD noise floor 2.23e-10)
    FB @ corner  2.35e+01   ->  4.65e-09 after the correction below

The raw identity FAILS for Fourier-Bessel and the reason is the k-DEPENDENT COLUMN SCALE. With
`psi_s = exp(-L(k)) psi` and `L` spatially constant,

    d(psi_s)/dk  =  (x . grad psi_s)/k  -  L'(k) * psi_s

and `|dL/dk|` reaches 1.74 for the corner family against exactly 0 for plane waves, which is
why one passed and one did not. With the correction both sit at the finite-difference floor.

**So `dsigma/dlam` is available for one gradient evaluation plus a scalar per column**, rather
than two extra tension evaluations. And `x . grad psi` is the RELLICH MULTIPLIER, which the
boundary-integral layer is already planned around. This opens A10 (a Hermite envelope, far
tighter than a secant bound), A15 (sign-change bracketing with no extra evaluations) and A16
(curvature for `delta_res` directly).

## A19 -- the eigenvector swing: FALSIFIED as proposed, STRONG as reinterpreted

The proposal was that the angle between `c(lam)` and `c(lam+h)` peaks at eigenvalues and so
samples more coarsely than the dip. **It does not: recall 0.333 at 4 samples per gap, falling
to 0.000 at 8 and finer.** The swing peaks where the minimising vector CHANGES IDENTITY, which
is at a branch CROSSING -- and crossings lie BETWEEN eigenvalues.

Which is far more useful than the original hypothesis. A peak in every gap partitions `[a, b]`
into cells holding exactly one eigenvalue each. Measured, "gaps hit" out of the gaps between
known dips:

    domain     3/gap    4/gap    6/gap    8/gap    extra peaks at 4/gap    ms per sample
    L_shape    8/8      8/8      8/8      9 (1 extra)              0            13.7
    mushroom   10/11    11/11    11/11    11/11                    0            32.7
    GWW1       14/14    14/14    14/14    14/14                    0           111.3
    H_shape    7/7      7/7      7/7      7/7                      0            81.7
    plus       7/7      7/7      7/7      7/7                      0            78.9
    chevron    2/2      2/2      2/2      2/2                      0           106.9
    rect81     12/13    12/13    13/13    13/13                    0            10.7

**At four samples per Weyl gap, six of seven domains give a PERFECT cell decomposition**, and
`rect81` needs six. Against the 18 per median gap that `sigma_1` dip detection needs for full
recall (section 20.2), that is a 3x to 4.5x reduction in samples.

**And it is nearly free.** `coef` costs 13.7 ms against `tensions`' 13.6 on L_shape, 111.3
against 104.8 on GWW1, 81.7 against 76.6 on H_shape -- **5 to 7 percent**, because both need
the same regularization and factorization and the `separate`-mode GSVD is comparable to
`gsvdvals`.

Extra peaks are 0 or 1-2 and are the safe direction: an extra cell is an empty cell, which is
cheap to detect and discard, whereas a missing cell is a missing eigenvalue.

Three caveats, all real. The denominator is gaps between DETECTED dips, so this measures
"does it separate the levels we already know about", not "does it find a hidden pair" -- the
pair-resolution question is untouched. The `5 x median` peak threshold is a parameter and has
not had the margin study that `G` and `tau_gram` got. And it BRACKETS rather than locates:
refinement still runs inside each cell, though refinement inside a bracket known to hold
exactly one minimum is the easy and robust case, which is the entire point.

## Shortlist

    KEEP, tested       A7 (DIRECT), A19b (swing cell decomposition), A14 (exact derivative)
    KEEP, untested     A3, A5, A8, A15, A24, A25, A26, A27
    DEAD               A2, A11, A12-square, A9, A13, A22, A23
    WOUNDED            A6 -- unsound at small pilots; usable only with A7 or A19b sizing it

The two survivors attack the same defect from opposite sides. **A19b gives the partition; A7
gives the search inside it, with no constant to calibrate.** They compose: use the swing at
4 per gap to cut `[a, b]` into one-eigenvalue cells, then run DIRECT inside each cell against
`tau`, then refine. That combination has a completeness argument that neither the Lipschitz
certificate (A6, measured unsound) nor the Weyl audit (section 20.3, blind at the top and
needing five levels) can provide, and it is the first candidate in this project that does.

---

# 23. The swing sees a pair ~290x tighter than the dip does

Section 22 left A19b's central question open: its cells were measured against gaps between
DETECTED dips, so it had never been shown to split a pair `sigma_1` presents as one dip. That
is the failure mode sections 9.4 and 19 call real, and it decides whether the composed method
is a capability win or only a cost win. `notes/swing_pairs.py`.

**It is a capability win.** `rect(1+delta, 1)`, modes (2,1) and (1,2), one window holding the
pair and no neighbour, stride `h` as a fraction of the mean gap:

    h = 1.005 (about 12 samples per gap)
    d(lam)      sigma_1 dips   swing peaks   verdict
    5.139e+00        2              1        sigma_1 already splits
    1.700e+00        2              1        sigma_1 already splits
    5.834e-01        1              1        SWING SEES IT, dip does not
    1.769e-01        1              1        SWING SEES IT
    5.913e-02        1              1        SWING SEES IT
    1.776e-02        1              1        SWING SEES IT
    5.921e-03        1              1        SWING SEES IT
    1.776e-03        1              0        neither
    0 (exact)        1              0        degenerate control

At that stride `sigma_1` stops splitting the pair below `d ~ 1.7`, and the swing keeps
splitting it down to `d ~ 5.9e-03`. **A factor of about 290 in separation, at the same
sampling cost.** In stride units the swing fires at `d/h = 5.9e-03` -- it resolves a pair
about 170x FINER than the sample spacing.

**This contradicts a recorded conclusion.** `search.md` section 3 says "there is no detector
here that sees a pair before `sigma_1` does", and section 11 measured the section 3.4 offset
detector firing "at exactly the same coarseness at which `sigma_1` first shows two minima".
Both are correct about the offset, and both were generalized into a claim about detectors in
general. The swing is a counterexample, and the reason it works is that it keys on something
neither of those does: **the IDENTITY of the minimising eigenfunction, not the shape of the
tension curve.** Section 15 measured the two eigenfunctions staying near-orthogonal at
`|cos| ~ 1/sqrt(n_int)` far past where the tension merges them, and that orthogonality is what
survives.

## The degenerate control is the other half of the result

At `delta = 0` the swing reads ZERO peaks. That is correct and it is not a failure: a true
double has no crossing, because there is no other branch for the minimiser to switch to. So

    swing peak inside a dip   ->  two DISTINCT eigenvalues
    no swing peak             ->  a genuine degeneracy, or a pair below the swing's own limit

The first arrow is a multiplicity discriminator of a kind (C1)+(C2) does not have. Section 9.4's
double-count -- the gap rule reporting multiplicity 2 at a SIMPLE eigenvalue once
`d <~ eps/c` -- is exactly the case where the swing says "distinct", and (C2) currently has to
reject it by a monotonicity argument on `sigma_2`.

The second arrow is still ambiguous, so this does not remove the resolution limit; it moves it.

## Where the limit actually sits, and it is not the pencil

    h = 3.770   fires to d = 5.9e-02, not 1.8e-02
    h = 1.885   fires to d = 5.9e-02, not 1.8e-02
    h = 1.005   fires to d = 5.9e-03, not 1.8e-03

`d_min` tracks `h`, roughly `h/30` to `h/170`. It is NOT `eps/c`, which is about 3.5e-09 here
-- six orders below where the swing stops. **So the swing's limit is set by the STRIDE and the
peak threshold, not by the basis**, and there is a great deal of headroom left: a finer stride
buys proportionally finer pair detection, which is not true of any dip-based detector.

## Caveats

One domain, one mode pair, one basis, `thresh = 5 * median`. The threshold has had no margin
study -- `G` and `tau_gram` both turned out non-delicate and this one has not been checked at
all. And `d_min ~ h/30` to `h/170` is a 5x spread across three strides, so the scaling is
indicative rather than established.

---

# 24. The composed candidate, scored

`tests/searches.py::swing_direct_search`. Swing cells at 6 samples per estimated gap, DIRECT
inside each cell against `tau = 0.5 * plateau`, Brent in the resulting one-root bracket,
(C1)+(C2) for multiplicity. Scored through the harness on five real domains with `reference`
table truth, windows cut inside a tabulated gap so the truth is COMPLETE:

    L_shape   [4.82, 49.35]   9 eigenvalues      GWW1     [1.27, 11.94]   8
    mushroom  [2.75, 33.39]   8                  H_shape  [3.87, 25.58]   8
    chevron   [19.83, 192.25] 8

    method            mean recall   total evals   spurious   mult errors
    uniform_grid            0.881          1010          0             0
    lappy_repaired          0.319          1946          0             0
    swing + DIRECT          0.975          2221          0             0

Per case:

    case       recall   prec  miss  spur  multX   max err   evals   weyl
    L_shape     1.000  1.000     0     0      0   2.3e-08     473   0.41
    mushroom    1.000  1.000     0     0      0   4.1e-09     442   0.02
    GWW1        1.000  1.000     0     0      0   8.8e-09     454   0.33
    H_shape     0.875  1.000     1     0      0   8.0e-09     385   0.76 !
    chevron     1.000  1.000     0     0      0   9.2e-09     467   0.12

**Four of five perfect, no spurious roots and no multiplicity errors anywhere**, and the one
miss is FLAGGED -- H_shape reads 0.76 against the 0.5 threshold. That is clause (3) of the
operational definition doing its job: the failure is reported rather than silent, which is the
difference between recoverable and fatal inside a shape-optimization loop.

## The honest ledger

**It costs 2.2x the uniform grid** for +0.094 recall. Recall is the headline metric by design
-- a missed eigenvalue cannot be repaired downstream -- so this is the right trade under the
stated principle, but it is a trade and not a free win.

**`lappy_repaired`'s 0.319 is not a fair comparison.** It is starved by the harness's crude
`_spacing` estimator, exactly the caveat section 11 records: a coarse pass sees a pair as one
dip, concludes the interval holds one eigenvalue, and sizes a 20-point grid. The number
measures my estimator, not lappy's architecture. Its own recall table, with `n_pts` set
explicitly, is 10/12.

**The accuracy column is F11 working, not a defect.** `swing+DIRECT` reports max errors of
4e-09 to 2e-08 against a requested `prec = 1e-8`; the uniform grid reports 8e-10 to 6e-09
because it refines to `xatol = 1e-13` regardless of what was asked. The composed search stops
at `max(prec*lam, delta_res)` and the first term binds. Doing better than asked is allowed;
doing it at 2x the refinement cost on every root is what section 2.6 rules out.

## Two efficiency findings

The first version re-scanned the CELL grid to estimate the plateau, having already scanned a
64-point pilot for the spacing. The plateau is a median over an O(1)-varying curve and needs no
cell-stride resolution; folding both into one 48-point pilot cut 2535 evaluations to 2221 with
identical recall.

Dropping to 4 samples per gap and a DIRECT budget of 10 cuts cost further to 1583 -- 1.6x --
but recall falls to 0.925 (H_shape 0.875 -> 0.750, chevron 1.000 -> 0.875). Section 22
measured 4 per gap as sufficient for the CELLS, and it is; what fails at 4 is the DIRECT search
inside them. The two budgets are separable and should be tuned separately.

## What is still untested

Five domains, all with table truth, all at low `lam`. No pair below the swing's own limit
appears in any of these windows, so section 23's 290x is not exercised here -- the composed
search has not yet been scored on a case where it should beat a dip-based method outright.
`rect(1+delta, 1)` is the instrument and it is not wired into the harness.

---

# 25. rect(1+delta, 1) in the harness: detection is not separation

Section 24 noted that no pair below the swing's limit appeared in any scored window, so
section 23's advantage was never exercised. `rect(1+delta, 1)` wired in as harness cases fixes
that -- window `[10, 88]`, holding `(1,1)`, the `(2,1)/(1,2)` pair and `(2,2)`, truth from the
closed form and COMPLETE, `delta` swept so the pair separation runs 5.8e-01 down to 0.

**The first run failed, at every separation including the widest.** All three methods scored
recall 0.750 -- three of four -- and the degenerate `delta = 0` case scored 1.000. A window
whose SPLIT cases fail while its DEGENERATE case passes says the pair is being merged, not
that a level is missing.

## The defect, and it is a correction to section 23

Instrumenting the `delta = 1e-2` case, `d = 5.83e-01`:

    pilot sees 3 wells in a 4-eigenvalue window   ->   estimated gap 26.0
    cell grid n = 19 over [10, 88]                ->   stride h = 4.33
    crossing truly at 48.86
    swing peak REPORTED at 46.83
    cells [10, 29.5, 46.83, 64.17, 88]            ->   BOTH pair members in one cell

The swing detected the crossing. It localized it only to `+/- h/2`, and with `d < h` the cell
boundary landed outside the pair instead of between its members.

**Section 23 measured DETECTION, not SEPARATION**, and I generalized one into the other. Its
in-pair test counted a peak as in-pair if it lay within `[lam_a - h, lam_b + h]` -- a tolerance
of a full stride on each side. Every number in section 23 stands as a statement about
detection; none of them was ever a statement about splitting. **A pair detector and a pair
separator are different things, and the 290x is the first, not the second.**

## The repair: a peak is a feature to BISECT TOWARD

The crossing lies in whichever half of an interval carries the larger angle, the halves are
equal width so the comparison is fair, and each level costs one mode evaluation. Cost is
`log2(h/d)` -- the same logarithmic behaviour adaptive subdivision has on the tension, obtained
here from the eigenvector instead, and it converges to the crossing regardless of how coarse
the initial stride was. That last property matters, because the stride here was set by a pilot
that merged the pair and therefore could not have been chosen correctly in advance.

    swing+DIRECT on rect(1+delta, 1), before and after localization

    d(lam)      recall before   recall after   uniform_grid   lappy_repaired
    5.834e-01           0.750          1.000          0.750            0.250
    1.769e-01           0.750          1.000          0.750            0.750
    5.913e-02           0.750          1.000          0.750            0.250
    1.776e-02           0.750          1.000          0.750            0.250
    5.921e-03           0.750          0.750          0.750            0.750
    1.776e-03           0.750          0.750          0.750            0.500
    5.922e-04           0.750          0.750          0.750            0.500
    0 (exact double)    1.000          1.000          1.000            0.250
    mean                0.781          0.906          0.781            0.438

**The uniform grid never splits this pair at any separation tested, including the widest**,
because its stride comes from the same merged-well estimate. The composed search splits it down
to `d = 1.78e-02`, so its advantage over the control is at least 32x in separation and the true
factor is larger, since the control's own limit is above 5.8e-01.

The roster of section 24 is unchanged at mean recall 0.975, at 2596 evaluations against 2221 --
localization costs about 17% and buys the pair regime.

## The remaining frontier

At `d <= 5.9e-03` the composed search still scores 0.750 with one spurious and one
multiplicity error: the cell boundary IS found (section 23 says the swing detects to
5.9e-03) but the two refinements in adjacent cells converge to the same minimum. The tension
floor at this pair is 9.1e-08 to 1.1e-07, so `eps/c` is about 2e-06 and `d = 5.9e-03` is
3000x above it -- the pair is resolvable in principle and this is a search failure, not a
basis limit. Two candidates, neither tested: DIRECT's fixed budget is too small for a cell
whose dip sits near its edge, and Phase D's Gram-angle collapse is NOT IMPLEMENTED in
`swing_direct_search`, so a duplicate from two adjacent cells is reported rather than
collapsed and recounted.

---

# 26. The pair frontier is the STRIDE, and it is reachable

Section 25 left the `d <= 5.9e-03` failure with two named candidates -- DIRECT's fixed budget,
and Phase D's Gram-angle collapse being unimplemented. **Both guesses were wrong, and so was a
third.** Recording the sequence because it is the same failure mode this file keeps catching.

**Guess 1: the collapse rule.** Wrong on its own terms. If two cells converge to the same
minimum, collapsing the duplicate leaves one root where truth has two: precision improves,
the missed eigenvalue stays missed. Collapse cannot recover recall.

**Guess 2: the localization tolerance.** `tol = max(prec*|b|, 1e-12, 1e-3*gap)` has a floor of
`2.6e-02` in this window, wider than the pair, so the boundary could again land outside it --
section 25's defect at a smaller scale. Plausible, and the pass/fail transition straddled it.
Untying the tolerance from `gap` changed recall not at all (0.906 before and after) and cost
1917 -> 2276 evaluations.

**What it actually is** was already in section 23's own table, unread:

    h = 3.770   the swing fires down to d = 5.9e-02, not 1.8e-02
    h = 1.005   the swing fires down to d = 5.9e-03, not 1.8e-03

The working stride in these cases is `h = 4.33`. At that stride the swing's detection limit is
about `5.9e-02`, so the tighter pairs were never DETECTED, and nothing downstream -- localizing
a crossing, collapsing a duplicate -- can recover a crossing that was never found. Sweeping
`per_gap` confirms it end to end:

        d(lam)     per_gap=6      per_gap=12     per_gap=24     per_gap=48
      5.92e-03   0.750 /  267   1.000 /  388   1.000 /  420   1.000 /  488
      1.78e-03   0.750 /  258   0.750 /  323   1.000 /  434   1.000 /  502
      5.92e-04   0.750 /  234   0.750 /  314   0.750 /  347   1.000 /  496

**Every pair in the tested range is reachable.** Each halving of `d` needs about a doubling of
`per_gap` -- section 23's `d_min ~ h` scaling, now confirmed through the whole search rather
than through the detector alone -- and the cost is nearly flat: 267 to 496 evaluations for a
10x gain in resolvable separation, because the swing grid is one term among several and
localization, DIRECT and refinement do not scale with it.

## The design consequence

A global `per_gap = 48` pays everywhere to buy resolution in one cell, which is the uniform-grid
mistake in a new place. The stride must refine LOCALLY, and the trigger is already
characterized: section 9.4's signature of two simple eigenvalues too close together is **(C1)
reporting multiplicity >= 2 while (C2) rejects it.** A cell in that state is exactly a cell
whose swing was sampled too coarsely, and it should have its swing resampled rather than its
tension.

That makes three adaptive mechanisms in the same design, all triggered by different evidence:
DIRECT subdivides on the tension, `_locate` bisects on the swing angle at a known crossing, and
this would resample the swing where multiplicity is ambiguous. They are not redundant -- each
one is blind to what the others see.

## The methodological entry

Two guesses, both plausible, both mechanism-first, both wrong, and the answer was in a table
written three sections earlier in this same file. The cost was two rebuild-and-rescore cycles.
**Read the measurement you already have before proposing the mechanism** -- and section 19
recorded the identical lesson about section 9.4, which was also already measured and also not
consulted.

---

# 27. Two blocking corrections found while building the rectangle suite

The suite's own verification step (assert the basis is adequate at every true eigenvalue before
scoring anything on the case) and an adversarial review of the design both fired before a
single candidate was run. Both invalidate recorded claims.

## 27.1 A frozen rank over a wide window starves the high end -- and `rect81` was never a bad basis

`curve_lab.make` and the first `rect_suite.pencil_for` both passed `lam_ref=(lam_lo, lam_hi)`,
freezing the rank. Section 5 says a frozen rank takes the MINIMUM over the range, and the
minimum is set by the LOW end because the numerical rank GROWS with `lam` (section 3). Over a
WIDE window that starves everything above.

Measured on `rect(1, 1/phi)` over `[5, 300]`, a 60x window:

    pad   n_cols   rank(lo, mid, hi)   max tension at a true eigenvalue
    1.3      102   23, 23, 23          3.46e-02
    2.0      156   23, 23, 23          3.46e-02
    3.0      236   23, 23, 23          3.66e-02
    4.0      314   23, 23, 23          4.09e-02

**Tripling the columns does not move the rank and makes the tension slightly WORSE.** Threshold
rank on the identical basis gives rank 35, 49, 53 across the window and a worst tension of
**2.12e-12** -- ten orders, at 6.8 ms/eval against the frozen 7.9.

Re-running the section 20 roster both ways:

    domain     window        ratio   frozen      threshold   improvement
    L_shape    [ 5.0, 60.0]   12.0   9.13e-07    1.63e-08        5.6e+01
    mushroom   [ 5.0, 45.0]    9.0   8.18e-10    6.57e-10        1.2e+00
    GWW1       [ 2.0, 20.0]   10.0   1.07e-08    2.80e-09        3.8e+00
    H_shape    [ 2.0, 30.0]   15.0   1.16e-07    5.26e-08        2.2e+00
    rect81     [ 2.0, 40.0]   20.0   5.44e-02    1.63e-12        3.3e+10

**So `rect81` was never an inadequate basis.** Section 20.1 called it "the informative failure
-- a high aspect ratio with plane waves is a genuinely inadequate basis", and section 21 said
"no stride can fix a basis" of its 4094-evaluation stall. Both are WRONG. It was a frozen rank
over a 20x window, i.e. my pencil construction, and threshold rank fixes it by ten orders.

What this does and does not touch:

* **Voided.** `rect81`'s 9.2e+04 floor spread (section 20.1), its 2.55 Weyl deficit and
  4094-evaluation stall (section 21), and every conclusion drawn from "an inadequate basis
  looks like this". The intended family F of the rectangle suite went with them -- with
  threshold rank its worst tension is 1.14e-12 and there is no spread left to test.
* **Weakened but standing.** The corner roster moved 1.2x to 56x, so the section 20 tables
  are mildly pessimistic but their conclusions (plateau stability, floor spread WITHIN a scan,
  the stride result) do not turn on a factor of 56. They should be re-run.
* **Untouched.** Everything on `rect(1+delta, 1)` in sections 15, 16, 19, 23, 25 and 26, which
  used narrow windows where frozen and threshold rank agree.

The general lesson is section 8's, again: the frozen rank was chosen once, early, and then
carried into every later measurement without being re-checked when the windows widened.

## 27.2 The scorer is blind AND INVERTED in exactly the regime the suite exists to probe

`harness.score` matches greedily at `rtol = 1e-6` RELATIVE, each truth entry claimable once,
and `unfold` expands `(x, mult)` into `mult` copies of `x`. So two copies of one location can
match two DISTINCT truth entries whenever they lie within `rtol`. Measured, at `lam = 49.35`,
against a true pair separated by `d`:

         d      d/lam   |  reports ONE root, claims mult 2  |  reports ONE root, mult 1
    5.9e-04   1.2e-05   |  recall 0.000  spur 2             |  recall 0.000  spur 1
    5.9e-05   1.2e-06   |  recall 1.000  spur 0  PERFECT    |  recall 0.500  spur 0
    5.9e-06   1.2e-07   |  recall 1.000  spur 0  PERFECT    |  recall 0.500  multX 1
    5.9e-07   1.2e-08   |  recall 1.000  spur 0  PERFECT    |  recall 0.500  multX 1

**A method that never splits the pair and inflates the multiplicity scores perfectly, and the
honest answer is penalised.** The metric does not merely go blind, it rewards the wrong
behaviour, and it does so below `rtol * lam = 4.9e-05` -- which is essentially the adequacy
contract boundary `20 * eps/c = 4e-05` at this pencil. The scorer stops working at exactly the
line between "the basis can resolve this, so the solver must" and "it cannot".

`levels_agree` shares the defect: same `rtol`, `collapse` applied to both sides.

Consequences for the suite as planned: families B, C and D cannot be scored on recall alone,
because in the interesting half of their range recall is not measuring separation. Needed
before any run:

* a matching tolerance tied to the CASE, e.g. `rtol_case = max(prec, 0.1 * d_true / lam)`, so
  a match must land within a tenth of the true separation;
* a first-class SEPARATION metric distinct from recall -- distinct levels found against
  distinct levels true -- since recall-with-multiplicity provably cannot tell "split the pair"
  from "guessed 2";
* an adversarial control in the candidate list, `always_report_mult_2`. If it scores well, the
  metric is broken rather than the control. This is the cheapest possible check that the fix
  worked, and it should have existed from the start.

---

# 28. The generic basis is 15x to 7e+05 too weak, and the floor spread was mostly its fault

`curve_lab.basis_for` -- plane waves sized by `n_useful` plus 40 Fourier-Bessel terms per
non-analytic corner -- was used for every corner-domain measurement in sections 20 to 26. It is
far weaker than the specs `chladni.reference.recipe()` records, which are what lappy needed to
produce the tables in the first place. Against the tabulated eigenvalues, same collocation,
threshold rank both sides:

    domain    generic max   recipe max      gain   spread generic   spread recipe
    L_shape       1.6e-08      2.4e-14   6.9e+05          8.5e+06         1.0e+01
    mushroom      6.6e-10      4.3e-11   1.5e+01          1.5e+01         1.5e+00
    GWW1          2.8e-09      3.2e-11   8.7e+01          5.2e+00         4.3e+00
    H_shape       5.3e-08      6.5e-10   8.1e+01          7.2e+05         8.8e+02
    chevron       7.1e-08      1.9e-09   3.8e+01          4.4e+00         2.5e+00

## The cause is the Fourier-Bessel ORDER, not the plane waves

That correction is worth recording because the obvious reading is wrong and it was nearly
written down. The recipe for the L-shape says "PURE Fourier-Bessel ... no mixed basis", which
reads as an indictment of mixing. Isolating the two variables:

    basis                n_cols    eps min    eps max    spread   margin @ prec=1e-8
    pure FB n=40             40   3.00e-15   4.94e-08   1.6e+07              9.0e-01
    pure FB n=80             80   2.08e-15   2.09e-14   1.0e+01              2.1e+06
    PW54 + FB n=40          148   1.38e-15   1.38e-08   1.0e+07              1.9e+00
    PW54 + FB n=80          188   2.28e-15   2.01e-14   8.8e+00              1.7e+06

**40 -> 80 Fourier-Bessel terms moves `eps_max` by six orders whether or not plane waves are
present. Adding 54 plane waves buys about 2x for 108 extra columns.** So the plane waves are
wasteful rather than harmful, and the recipe's own text said the number outright -- "1e-08 at
40 terms, 4e-13 to 2e-12 at 60, and 2e-15 to 1e-14 at 80" -- which matches the generic basis's
1.6e-08 exactly. The recipe told us the answer and it was not read.

## What this does to section 20.1

Section 20.1 reported the floor spreading "four to five orders WITHIN a single scan" and
explained it structurally: on a domain with a singular corner, modes that happen to vanish
there are resolved far better than their neighbours. **The spread is mostly a marginal-basis
artifact.** With recipe-grade bases it collapses to 1.0e+01 (L_shape, from 8.5e+06), 1.5e+00
(mushroom), 4.3e+00 (GWW1) and 2.5e+00 (chevron).

The structural mechanism is real -- `lam_3 = 2*pi**2` has no singular part at the reentrant
corner and reaches 3e-15 at 40 terms alone -- but it only shows up as a five-order SPREAD when
the basis is too short to resolve everything else. Given a basis that can, every mode lands
within an order of the same floor.

`H_shape` is the exception that stays interesting: 8.8e+02 even at recipe grade, the hardest of
the non-excluded set with four reentrant corners. Whatever survives of the per-eigenvalue
accuracy argument lives there, not on the L-shape.

## The rule going forward

**Use `reference.recipe()`'s spec whenever precision is what is being measured**, at least
until something beats it. `curve_lab.RECIPES` now encodes them and `curve_lab.make(...,
label=...)` selects them; `basis_for` remains for domains with no table (rectangles, where
plane waves are adequate -- `rect_suite` measures 1e-15 to 2e-12 with margins of 6e+04 at
`prec = 1e-8`).

Two parameters of the mixed specs are NOT in the recipe text and were fitted here against the
tables: the Fourier-Bessel share of the total, and the fundamental-solution offset distance as
a fraction of the diameter. They are recorded in `RECIPES` as fitted, not as given.

## What has to be re-measured

Sections 20 through 26 all used the generic basis. Most of their conclusions are about the
SEARCH and should survive, but three are basis-sensitive and should not be trusted until
re-run: section 20.1's floor spread (already corrected above), section 20.2's recall-vs-stride
table (a deeper floor changes the dips a stride has to catch), and section 24/25's end-to-end
scores, where `H_shape`'s persistent miss may simply have been a basis that could not resolve
its 3.97e-03 pair.

---

# 29. The bracketing filter: results

Full write-up in `notes/bracket_report.pdf`; this records the load-bearing numbers and the two
places the first reading was wrong.

## The instrument

25 rectangle cases across five families, plane-wave bases, `rtol = 1e-16`, threshold rank, the
pencil FIXED across the whole `prec` sweep so only the algorithm's `prec` varies. Hard
precondition passed on every case before scoring: `sigma_mu` at or below 4.5e-15 counting
multiplicity, headroom 2.8x to 7.0x at `prec = 1e-14` by the bracketing criterion
(`prec*lam > 3*eps/c`). Note that is the BRACKETING criterion; section 9.4's `20*eps/c` is the
MULTIPLICITY one and importing it here would apply a bound from a phase not under test.

## The headline

The pair ladder, oracle cost at the tightest `prec` still requiring a split:

    d          d/lam     uniform  lappy  lappy_rep  lipschitz  swing_cells  swing_direct
    5.8e-01  1.2e-02        446    302       1316        293          131           152
    1.8e-01  3.6e-03         --   1016       1103       1298          528           688
    5.9e-02  1.2e-03         --     --        922       1308          184           225
    1.8e-02  3.6e-04         --     --        830         --           --            --
    5.9e-03  1.2e-04         --     --       1426         --          658           844
    1.8e-03  3.6e-05         --     --         --         --           --            --

**Every method's splitting limit is between `d/lam = 1.2e-04` and `3.6e-05` at 1500
evaluations** -- four to five orders coarser than the pencil's own `eps/c ~ 1e-15` relative.
**Ten orders separate what the objective supports from what the searches extract.**

## Two corrections to the first reading

**Reading family B alone gives the wrong verdict.** A first draft cut `lipschitz` on the ladder
above. The per-family breakdown at `prec = 1e-8`, oracle/autonomous, says otherwise:

    method            A(1)   B(12)   C(8)   D(3)   E(1)
    uniform_grid      1/0     3/2    3/2    3/3    0/0
    lappy             0/0     4/3    4/2    3/3    0/0
    lappy_repaired    1/0     6/0    6/2    3/3    0/0
    lipschitz         1/0     5/5    5/4    3/3    0/0
    swing_cells       0/0     6/5    3/3    3/3    0/0
    swing_direct      1/1     6/5    3/3    3/3    0/0

* **The swing is much weaker on CLUSTERS than on pairs** -- 3 of 8 on family C against 6 of 12
  on B, where `lappy_repaired` reaches 6/8. Cutting cells at eigenvector crossings suits two
  eigenvalues and suits a tight group of three or four with unequal gaps much less.
* **`lappy_repaired` has the best oracle record and the worst autonomous one**, 6/12 against
  **0/12** on family B. Almost all of its capability is per-case tuning. That is the largest
  oracle-autonomous gap in the study and precisely what the two-column protocol exists to
  expose: the method that looks strongest tuned would ship as the weakest.
* **`lipschitz` is the most consistent autonomous performer**, its two columns nearly
  coinciding. Its unsound-`L` risk is real but did not bite on rectangles.
* **Nobody passes family E**, the 29-eigenvalue mixed window, at any `prec` or setting.

**The `prec` ladder was degenerate as first built.** With the original B deltas the results were
IDENTICAL at 1e-8, 1e-11 and 1e-14, because the smallest true gap exceeded `prec*lam` at all
three -- so `prec` changed cost but never the requirement. Fixed by extending the ladder to
`delta = 1e-8, 3e-10, 1e-11`, chosen to STRADDLE each `prec`. And a hard limit fell out:
`delta = 1e-13` is the only separation distinguishing `prec = 1e-11` from `1e-14`, and its
`d = 5.9e-12` sits 1.2x above `snap`'s 4.9e-12 floor. **No case on this instrument can exercise
`prec = 1e-14` as a requirement**; anything measured there is cost, not capability.

## Cost scales with `prec` even when the requirement does not

`swing_cells` on `d = 5.8e-01` needs 131, 170, 200, 230 evaluations at the four precisions --
1.8x for ten orders of `prec`, buying nothing, because every method bisects to `prec*lam`
regardless of the local gap. Refining a cell edge to 1e-13 when the nearest other eigenvalue is
0.58 away is waste. Section 2.6's "not more work than asked" is violated by the whole field.

## The controls did their job

`single_bracket` is rejected by PURITY on 18 of 21 cases; the three it passes are the family-D
exact degeneracies, which hold ONE level, where one bracket is the correct answer. `fine_grid`
is rejected by COST on every case. Each degenerate strategy is rejected by the criterion meant
for it, so the metrics are validated rather than assumed.

## Verdict

    CUT   uniform_grid (keep as the control), lappy (dominated by lappy_repaired everywhere)
    KEEP  lappy_repaired -- fix the self-configuration, not the search
          lipschitz      -- most consistent autonomously; test the certificate on a corner domain
          swing_cells    -- cheapest where it works; weak on clusters
          swing_direct   -- orthogonal to cells, not better: cells handle pairs, DIRECT handles
                            isolated wells

## 29.1 Correction: `prec = 1e-14` IS exercisable, and the 1e-14 failures are cost

Section 29 claimed no case could exercise `prec = 1e-14` as a REQUIREMENT, because the only
separation distinguishing it from `1e-11` -- a pair at `d/lam ~ 1e-13` -- sits 1.2x above
`snap`'s merge floor. **That reasoning was wrong: family B does not use snapped truth.** It
takes the pair values from `rect_eig` directly, precisely so a controlled separation is not at
the mercy of a merge policy. Measured: `rect_eig` puts the two 833 ulps apart at `delta = 1e-13`
and 250 apart at `3e-14`, and the pencil has `d/(eps/c) = 138` of headroom at the first.

Checked across all 25 original cases: **no case's truth is corrupted by snap.** The tightest,
`B:d=5.9e-10`, clears the threshold by 75x; everything else by orders.

Two cases were built and run. Both pass the floor precondition (headroom 4.1 and 3.3):

    B:d=5.9e-12 (d/lam=1.2e-13)   prec=1e-4/1e-8/1e-11: pair MERGES, most methods pass
                                  prec=1e-14:           pair must SPLIT, EVERY method fails
    B:d=1.8e-12 (d/lam=3.6e-14)   same pattern

So `prec = 1e-14` is now exercised as a genuine requirement, and the result is consistent with
the ladder -- nothing splits at `1.2e-13` when nothing splits at `3.6e-05`.

**And the cost hypothesis is confirmed directly.** On `d = 5.9e-03` at `prec = 1e-14`, where the
requirement is identical to `1e-11`, `swing_cells` fails at the 1500 cap and **succeeds at 1549
evaluations** once the cap is raised, against 1279 at `1e-11`. A 21% cost increase, and the only
thing between "fails" and "succeeds" is the budget line. Cost, not capability, as suspected --
now measured rather than inferred.

The methodological entry: the first claim was reasoned from a property of the reference layer
(`snap`) without checking whether the code under test actually used that path. It did not.

## 29.2 Correction: family E's universal failure was the flat budget

Section 29 reported that NO method passes `E:mixed` -- 29 eigenvalues in [150, 560] -- at any
`prec` or setting. That is true at the 1500-evaluation cap and misleading as stated. Raising the
cap, at `prec = 1e-4`:

    cap    per eig   uniform_grid  lappy_repaired  lipschitz  swing_cells  swing_direct
    1500        52           FAIL            FAIL       FAIL         FAIL          FAIL
    4000       138           FAIL            FAIL       1538         3906          FAIL

**`lipschitz` handles it at 1538 evaluations -- 38 over the cap.** So the "universal failure"
was a razor-thin budget miss, not a capability limit.

The cause is that the cap was FLAT at 1500 across every case regardless of how many eigenvalues
the window holds. Family B has 4 (375 evaluations each at the cap); family E has 29 (52 each).
Successful family-B costs run 33 to 366 evaluations per eigenvalue, so E at 52 was inside the
band where success is marginal. **The budget has to scale with the eigenvalue count**, and a
flat cap silently penalises exactly the dense windows a black-box solver will meet.

This also strengthens `lipschitz`: it is the only method that handles the dense mixed window at
all, and it does so at 53 evaluations per eigenvalue -- the cheapest per-eigenvalue figure in
the study. Combined with its already-noted autonomous consistency, the case for keeping it is
stronger than section 29 implied.

The methodological entry is the same one twice over: a threshold chosen for convenience
(`COST_CAP = 1500`, picked to keep the sweep finite) became a scientific claim ("nobody handles
family E") without being varied. Both this and the `snap` error of 29.1 were caught only because
the conclusion was challenged.

---

# 30. Why the methods fail in the band -- three different mechanisms, none of them the objective

Section 29's band is `prec < d/lam < ~1e-4`: separations that must be split and are not. Panel B
of `notes/solvability_band.png` already rules out the obvious explanation -- the tension curve
is the SAME shape at every separation, two clean minima with the bump between them, and the
pencil resolves all of them. So the failure is in the searches. Diagnosed on
`B:d=1.8e-03` (`d/lam = 3.6e-05`) at `prec = 1e-8`, inside the band.

## 30.1 uniform_grid, lappy: pure counting

To see two minima you need about three samples inside an interval of width `d`, so a uniform
pass needs `~3W/d` samples. At `d = 1.8e-03` over a window of 78 that is 1.3e+05, and at
`d/lam = 1e-6` it is 4.8e+06. Not a bug -- the design cannot get there at any budget worth
paying. Their failure IS fundamental to their design.

## 30.2 lappy_repaired: the recursion reaches the scale, and the cost explodes

Adaptive subdivision needs only `O(log(W/d))` levels -- about 7 eight-fold narrowings here --
which should cost a couple of hundred evaluations. Measured with the cap raised to **20000**:

    max_depth   7      10     13     16
    result    >cap    >cap   >cap   >cap

**Every depth exceeds 20000.** The reason is in the repair itself: section 13 removed the
`sigma_2` gate and made the recursion UNCONDITIONAL -- it descends into every discrete minimum
at every level. With `n_sub = 9` that branches exponentially in depth, so the levels needed to
reach `d` cost far more than the `log` suggests. The subdivision is right; the lack of a gate is
what makes it unaffordable. Section 11 showed the ORIGINAL gate was broken (it tested the grid,
not the pair), so the fix is a gate that works, not the old one back.

## 30.3 The swing: refining the stride DESTROYS the detector

This is the surprising one. The swing completes within budget at `per_gap = 2560` and still
fails. At that stride:

    per_gap    h        h/d    median angle   max angle   peaks > 5*median
        160  1.62e-01  91.5      5.20e-02      9.98e-01          49
        640  4.06e-02  22.9      3.98e-02      9.98e-01         213
       2560  1.02e-02   5.7      4.08e-02      9.98e-01         907

**907 candidate crossings for 4 eigenvalues.** A genuine crossing gives an angle of 0.998 --
the minimising eigenvector flips identity -- while in-well rotation gives about 0.04. The two
populations are cleanly separated. But the rule is `5 x median`, and **the median rises with the
stride**, so the threshold tracks the noise instead of discriminating against it. The detector
defeats itself as it is refined.

Worse, and measured: **more samples make the decomposition WORSE.** Cuts landing in a genuine
gap go 3/3 at `per_gap = 160` to 2/3 at 2560. "Sample more finely" is not the fix.

An absolute threshold is better conditioned but not sufficient on its own:

    rule            per_gap=160              per_gap=2560
    5 x median      49 peaks, 3/3 gaps       907 peaks, 2/3 gaps
    abs > 0.5       32 peaks, 2/3 gaps       501 peaks, 2/3 gaps
    abs > 0.9        5 peaks, 2/3 gaps       111 peaks, 2/3 gaps

It cuts spurious peaks by an order (49 -> 5) but loses a genuine crossing. So the criterion
needs rethinking, not retuning.

**And a structural point that was not noticed until now: the swing's cell-cutting is a UNIFORM
scan.** `_cells` samples `linspace(a, b, n)`; only `_locate` is adaptive, and it runs only after
a peak has been found. So the cost of reaching stride `h` is `W/h` -- exactly a uniform grid.
The method inherits 30.1's counting problem for its detection stage and only escapes it once a
crossing is already bracketed.

## 30.4 Is it fixable?

The band is not a property of the objective, so nothing here is a hard limit. Three specific,
untested repairs fall out of the diagnosis:

* **An adaptive stride for the swing.** Refine the sampling only where a cell looks suspicious,
  rather than uniformly. This is section 26's proposal, and 30.3 is now a second and independent
  argument for it: uniform refinement does not merely cost too much, it actively degrades the
  detector.
* **A scale-free crossing criterion.** Genuine crossings sit at 0.998 and noise at 0.04, so the
  populations are separable -- the current rule simply picks the wrong statistic. An absolute
  threshold alone loses crossings, so the criterion probably has to be local (a peak against its
  own neighbourhood) rather than global.
* **A working recursion gate for `lappy_repaired`**, to stop the exponential branching. The
  eigenvector signal is the natural candidate, since it is the one quantity measured to
  distinguish two eigenvalues before `sigma_1` does.

All three are hypotheses supported by measurement, not verified fixes. Diagnosed on one case at
one `prec`.

---

# 31. Fitting the surrogate to rect(1+delta, 1): usable for some prototyping, not for the swing

`notes/surrogate_fit.py`. Wells fitted from truth -- location exact, `eps` and `c` measured off
the real pencil -- across the whole family B ladder, window [10, 88].

    case          med |ds1|  max |ds1|  med |ds2|  near pair  bump real  bump surr  bump err
    B:d=5.8e-01    1.17e-01   6.36e-01   2.65e-01   8.57e-02   4.37e-03   4.64e-03  6.3e-02
    B:d=5.9e-03    1.24e-01   8.01e-01   3.66e-01   1.45e-01   5.20e-05   5.08e-05  2.4e-02
    B:d=5.9e-07    1.03e-01   5.34e-01   3.10e-01   2.06e-01   5.69e-09   4.84e-09  1.5e-01
    B:d=5.9e-12    1.31e-01   8.93e-01   3.66e-01   1.74e-01   4.81e-14   4.57e-14  4.8e-02

**Speed: 3,177x** -- 11.39 ms against 3.6 us.

## What survives

`sigma_1` sits at 10-24% median error, 50-89% worst. That is far above the 1.7e-02 section 12
reported on rect(2,1), and it is NOT the `c` estimate: sweeping `h_frac` from 0.005 to 0.20
leaves the median between 1.2e-01 and 2.6e-01. It is the single `plateau`: this window spans
`lam` 19.7 to 79.0, over which the real inter-well maxima and the cone slope both vary by 2.7x,
and the surrogate has one number for the whole range.

**The bump between a close pair -- the feature a splitting method must detect -- is right to
about 8% median, and scales correctly over TWELVE decades**, 4.4e-03 down to 1.5e-14 with the
surrogate tracking at 4.6e-03 down to 1.4e-14. So the pair-splitting geometry is faithfully
reproduced even where the absolute curve is 20% off.

`sigma_2` is 26-45% out, confirming the documented limit on this family. Anything reading the
spectrum above the multiplet -- lappy's gate, (C1) -- has to be on a real pencil.

## What does NOT survive, and it is the interesting one

The surrogate's `matrix(k)` is Hermitian, so a minimising eigenvector exists and the swing angle
can be computed. It is not usable, for a reason that is easy to miss:

    per_gap    source      median ang    max ang   peaks>5med
        6      real          5.00e-02   8.84e-01        2
               surrogate     1.67e-16   1.00e+00        2
       96      real          3.75e-02   9.96e-01       32
               surrogate     0.00e+00   1.00e+00       58

**The surrogate's median swing angle is ZERO.** Its wells are independent diagonal entries under
a fixed rotation, so between crossings the minimising eigenvector is a constant coordinate
direction and consecutive samples are identical. The real one rotates continuously as the basis
coefficients change with `lam`, giving a median of 3.7e-02 to 5.0e-02.

That in-well rotation is **exactly the mechanism section 30.3 identified as what breaks the real
swing in the band**: the median angle rises with the stride, the `5 x median` threshold tracks
the noise instead of discriminating against it, and 907 spurious peaks appear. The surrogate
omits precisely the failure we would be prototyping a fix for, so it would report success where
the real pencil fails. It cannot even test threshold RULES: with a median of zero, `5 x median`
is zero and every local maximum passes.

## Verdict

    USE FOR      sigma_1-only bracketing: subdivision schedules, DIRECT, recursion gates
                 keyed on sigma_1, cost and scaling studies, pair-splitting geometry
    DO NOT USE   the swing or any eigenvector method; anything reading sigma_2 or above;
                 any threshold calibration

**A cheap fix if the swing is to be prototyped:** make the surrogate's rotation `lam`-dependent
-- let `Q` vary slowly with `lam` instead of being fixed -- so that in-well rotation exists and
the angle has a realistic noise floor. That is a few lines, and it would restore the one
mechanism that matters. Until then the surrogate is a `sigma_1` instrument only.

---

# 32. Correction: the L-shape's five-order floor spread is a BASIS artifact

Section 20.1 measured the L-shape's tension floor at **3.1e-13 at `k = 4.443` and 5.5e-08 at
`k = 5.649`** -- five orders within one scan -- and explained it structurally: whichever modes
VANISH at the reentrant corner are analytic there and resolved almost exactly, while their
neighbours carry the `r**(2/3)` singularity. Section 20.4 then made that the direct
justification for the per-eigenvalue accuracy clause, and `search.md` section 1 repeats it as
"the floor varies FIVE ORDERS within a single scan, structurally".

Measured 2026-09-03 with the recipe-grade basis -- 80 pure Fourier-Bessel terms at the
reentrant corner, `tests/hard_roster.py` -- over `[5, 40]`:

    lam      9.640   15.197   19.739   29.521   31.913
    k        3.105    3.898    4.443    5.649    5.649
    eps    1.9e-15  5.4e-15  2.8e-15  8.2e-15  5.3e-15      spread 4.2x

**5.3e-15 at `k = 5.649`, against the 5.5e-08 recorded there: seven orders.** So the spread is
a property of the basis that was used, not of the domain, and it collapses from 1e+05 to 4.2
once the basis is adequate.

This is section 27.1 a second time. There, `rect81`'s 9.2e+04 floor spread turned out to be a
frozen rank over a wide window, and section 20.1's "informative failure" reading was withdrawn.
Here the same section's other headline number goes the same way, for the neighbouring reason: a
basis that was not the one the domain needs. Both were measured with `curve_lab.basis_for`,
which section 28 puts at 15x to 7e+05 worse than the recipe.

**What this touches.**

* **Voided.** "The L-shape is the domain where the floor varies five orders." It is not, with a
  basis that resolves it. Anything that cites section 20.1's L-shape numbers as evidence about
  the DOMAIN should cite them as evidence about `basis_for` instead.
* **Standing.** The per-eigenvalue accuracy clause. The floor still varies -- 4.2x over these
  five levels -- and a domain whose basis cannot be made adequate everywhere will still show a
  large spread. What changes is that the spread is a statement about the basis, so the right
  response is a per-case adequacy measurement rather than a design assumption. That is what
  `hard_roster.finest_prec`, `rect_suite.assert_floor` and `surrogate_cases.adequate` now do on
  the three tiers.
* **Standing, and strengthened.** Section 20.4's "no absolute tension threshold, anywhere". A
  floor that is a property of the basis rather than the domain is if anything a stronger
  argument for thresholds as fractions of the measured plateau.

The general lesson is section 8's and section 27.1's, for the third time: **a number measured
through a construction that was chosen early and never re-checked is a number about the
construction.** The floor spread, the frozen rank and the oversized collocation were all of
this shape, and all three were found by re-measuring rather than by inspection.

---

> **ROLLED BACK 2026-09-08.** The implementation these two sections describe was reverted: the
> v1 solver is back in `tests/` as scored candidates and `MPSEigensolver.solve_interval` raises
> again. The code is on the branch `wip/v1-promotion-and-multiplicity`, and the file references
> below are STALE -- `tests/test_uneven_cluster.py` was later renamed `test_cluster_totals.py` and
> now exists only on that branch, as does the `TODO.md` item cited at the end of section 33.
>
> The MEASUREMENTS stand and are why these sections are kept: the multiplicity total really is
> wrong in the band described, on real pencils, with the preconditions holding. Two caveats before
> using them:
>
> * **Section 33's framing is corrected by section 33.1.** It is not about uneven spacing. The
>   evenly spaced triple fails too away from `prec = 1e-8`, which is the single clean row every
>   earlier measurement happened to use.
> * **Section 33 judged its cells with the BRACKETING precondition, not the multiplicity one.**
>   `assert_floor` reports both and they differ by 20x. Re-checked against `mult_prec = 20*eps/c`
>   the conclusion survives -- the pencils admit 4.7e-15 to 7.9e-15, so the whole sweep clears it
>   by at least 100x -- but it survived by luck rather than by design.
>
> What the later work added, and what a mathematical treatment should probably start from: the
> failing band's width tracks `WIDE_FACTOR` proportionally (0.3-1.0 / 0.8-2.0 / 2.0-4.0 / 4.0-6.0
> licences at 1 / 2 / 4 / 8), which is causation shown by intervention rather than correlation; and
> stage 4 was found to DISCARD located roots, a merged group collapsing to a single representative
> while the old ratio-gap rule over-read on a wide bracket and hid the loss. Those two are the
> mechanism; the tables below are the symptom.


# 33. The uneven cluster: a wrong TOTAL that five precisions and both preconditions agree on

Measured while taking the one measurement `TODO.md` said had to come before the multiplicity
contract could be written -- whether an unevenly spaced cluster has a non-degenerate window. It
does, but the interesting part is what sits below it.

## The spacing was never a choice

`crossover_cases(window="cluster")` sits at `50*pi^2`, modes (5,5), (7,1), (1,7). To first order a
`(1+d) x 1` perturbation moves each level by `-2*d*pi^2*m^2`, and `m^2` runs over {25, 49, 1}, so
both gaps are proportional to `49-25 = 25-1 = 24`. **The evenly spaced triple is an arithmetic
accident of that level**, not a property of the dial, and every statement about the ambiguous band
inherited it.

`65*pi^2` has modes (1,8), (8,1), (4,7), (7,4). `m^2` runs over {1, 64, 16, 49}, the gaps go
`64-49 : 49-16 : 16-1 = 15 : 33 : 15`, and that is the 1 : 2.2 : 1 that `surrogate_cases.cluster`
mimics -- on a real pencil, where `sigma_2` and `sigma_3` exist. `split_degeneracy_cases` has
always dialled this level and nothing had ever swept it.

## The table

Declared total against a truth of 4. `D` is a decline, which is an honest third outcome and is
NOT counted as a wrong answer; `*` is a wrong total with `assert_floor` AND `assert_smooth` both
holding at that cell:

    prec     0.3   0.5   0.8   1.0   1.3   1.6   2.0   2.6   3.0   4.0   6.0
    1e-06     D     D    2*    2*    2*    2*    2*     D     4     4     4
    1e-08     4     4    2*    2*    2*    2*    2*     4     4     4     4
    1e-10     4    2*    2*    2*     D    2*    3*     4     4     4     4
    1e-12    2*    2*    2*     D    2*    2*     4     4     4     4     4

The correct-total window opens around band 2.6, a diameter of about **11 merge licences**.
`rect_suite.multiplicity_undecided` encodes the even measurement's `1 < diameter/licence < 2` and
reads "decided" for every starred cell -- so the constant the contract was about to be written
from is wrong by a factor of five on the case the contract is for. That is section 7's error in
its original form, one commit away from being made.

## The mechanism, and why no constant fixes it

At `prec = 1e-8`, band 0.8, members at 0, 0.80, 2.56, 3.36 licences. The outer pairs are
sub-licence, so `S(prec)` says two levels of multiplicity 2. All four roots are located and the
prefilter merges them correctly into two. Then:

    wide bracket    [-2.65, +3.00] licences -- WIDER THAN THE WHOLE CLUSTER
    (C1) ratios     55.0, 2.37, 1.33, 1.6e+06    -> largest gap at j=4, kappa = 4
    (C2) prominence 24.05, 0.58, 1.16, 1.23      -> branch 2 fails the bar of 2, mu = 1

Both halves behave exactly as specified. `sigma_2` is minimal at the PARTNER, 0.8 licences away,
so (C2) is right that branch 2 is not minimal at the representative -- and the right flank sits
beside the next level's member at +3.36, which is why its prominence is below 1.

The proximate cause is that `wide_bracket_for` skips its `neighbour` cap when the neighbour is
inside the merge licence, which is correct and is what lets an exact degeneracy be confirmed at
all -- but the exemption is unbounded, so the reach falls back to `factor*xatol = 2` licences and
the next level sits at `2.2*g`, inside it for every `g < 0.91`. Evenly spaced, the next level is
at `2*g` and the two constraints nearly coincide.

**But capping is not the fix.** Sweeping the reach from 0.5 to 6 licences never reads the right
total at any reach, at any band in 1.0 to 2.0. (C2) asks whether branch `j` is minimal AT THE
POINT being read, and the higher branches of a level with internal structure are minimal at their
own members. No bracket around one point can see them. Working the algebra at band 0.8 gives an
admissible flank window of `[1.6, 1.76]` licences -- 10% wide -- which is a knife edge and not a
constant.

## The cross-check is net-negative here, which inverts its own ablation

    band   gaps / licence        collapse ON         collapse OFF
    0.3    0.30 : 0.66 : 0.30    4/4  [4]            6/4  [2, 4]
    0.5    0.50 : 1.10 : 0.50    4/4  [4]            3/4  [2, 1]
    0.8    0.80 : 1.76 : 0.80    2/4  [1, 1]         2/4  [1, 1]
    1.0    1.00 : 2.20 : 1.00    2/4  [1, 1]         3/4  [1, 1, 1]
    1.3    1.30 : 2.86 : 1.30    2/4  [1, 1]         4/4  [1, 1, 1, 1]
    1.6    1.60 : 3.52 : 1.60    2/4  [1, 1]         4/4  [1, 1, 1, 1]
    2.0    2.00 : 4.40 : 2.00    2/4  [1, 1]         4/4  [1, 1, 1, 1]
    2.6    2.60 : 5.72 : 2.60    4/4                 4/4

At 1.3 to 2.0 the guard turns a correct answer into a wrong one -- the mirror of section 32's
neighbour, where ablation gave 6/3 and 4/3 against 3/3. It chains roots within `WIDE_FACTOR`
licences, catching genuinely distinct levels, and then prefers the SPANNING count over the sum of
the individual ones because "the total was never in doubt". Here the spanning count is the member
that is wrong -- same (C2) defect, read once over a wider interval. Below the licence it is still
load-bearing, so it is not a guard to remove; it is a guard sitting on something broken.

## The general lesson, and it is section 7's again

Every previous measurement of the ambiguous band was on evenly spaced triples because that is what
one window happened to produce. The band, the reach, `WIDE_FACTOR`, and
`multiplicity_undecided`'s predicate were all fitted to that geometry and all describe it
correctly. **A constant measured on a family that a construction happened to generate is a
constant about the construction** -- section 32's lesson, arriving for the fourth time, and this
time it was about to be written into a contract.

Recorded in tracked form at `tests/test_uneven_cluster.py` (five strict xfails, bodies running)
and `TODO.md` section 1, which now carries it as the `[blocking]` item.

## 33.1 Correction: it is not about the spacing, and section 33 said one precision

Section 33 above called the evenly spaced cluster a clean control and concluded that spacing is
the variable. **That was `prec = 1e-8` and nothing else.** Swept across the ladder, the even
triple at `50*pi^2` fails too:

    N = 50, evenly spaced, truth 3
    prec    0.3  0.5  0.8  1.0  1.3  1.6  2.0  2.6  3.0  4.0  6.0
    1e-06    3    3   1*   2*   1*   2*   2*    3    3    3    3
    1e-08    3    3    3    3    3    3    3    3    3    3    3
    1e-10    3    3    3    3    3   2*    3    3    3    3    3
    1e-12    3    3    3   2*    3    3   2*    3    3    3    3
    1e-14    3   2*   2*   1*   2*   2*   1*    3    3    3    3

1e-8 is the single clean row, and it is the row section 33's "control" was measured on -- and,
for that matter, the row most of this document's cluster work has used. So the statement is not
about evenness. **The total is unreliable wherever a cluster's diameter falls between roughly one
and eight merge licences**, on both spacings. Unevenness widens the band and makes it hit at more
precisions; it does not create it.

Checked against the right precondition this time. `assert_floor` reports `brack_prec` and
`mult_prec` and they differ by 20x; section 33 judged the cells with the BRACKETING criterion,
which is the mistake `assert_floor`'s own docstring warns against -- importing a bound from a
phase that is not under test. On the multiplicity criterion, `20*eps/c`, these pencils admit
4.7e-15 to 7.9e-15, so every cell in both tables clears it by at least a factor of 100. The
conclusion survives the stricter test; it was luck that it did.

**And the thing that matters most is not in section 33 at all: the wrong total is SILENT.** In
every failing cell `declined` is False, `unmet` is empty, `met_request` is True, and no warning is
raised. `c2_rejected` and `n_ambiguous` fire on correct cells at least as often as on wrong ones,
so neither is usable as an alarm. A Weyl deficit tracked 9 of 10 uneven failures and no correct
cell but missed the even one, and over a `0.97*lam` to `1.03*lam` window Weyl is a
plus-or-minus-one estimator -- a heuristic, not a guarantee.

The general lesson is section 32's yet again, one level up: **a claim checked at one point on a
ladder is a claim about that point.** Section 33 caught a constant fitted to one geometry by
varying the geometry, and then made the same error in the other coordinate by fixing `prec`.

---

# 34. The `145*pi^2` declines are the THRESHOLD RANK, not the domain and not the seed

2026-09-17. `TODO.md` carried "declines at `145*pi^2`" as a search-quality item, pointing
upstream at the bracketer. It is upstream of the bracketer too: the objective the bracketer is
handed is genuinely not Lipschitz, because the threshold rank flips between adjacent samples and
each flip moves `sigma_1` by about a factor of two.

## Every decline is the same one

Swept 5 precisions x 11 bands on the split degeneracy at `145*pi^2`: **30 of 55 cells decline,
and every one carries `_seed`'s message**, "the slope estimate never settled". Not one is the
`MAX_GROWTHS` path, not one is stage 2 or stage 3. The 25 cells that do not decline are all
correct: 25 right, 0 wrong. A typical history, the largest observed secant per doubling:

    0.7101  0.8090  2.201  3.920  7.765  16.24

Rising monotonically, no plateau in sight, so the doubling loop exhausts `MAX_DOUBLINGS`.

## What the rising secants ARE

At every seed density the largest secant sits **exactly on a rank flip**, and `sigma_1` roughly
halves across it. `prec = 1e-14`, band 1.3:

    n      max secant   at pair   rank         sigma across it        flips in grid
    64        2.0485      48      107 -> 108   6.956e-02 -> 3.266e-02       7
    127       3.3788      97      107 -> 108   6.309e-02 -> 3.266e-02      23
    253       6.0387     195      107 -> 108   5.985e-02 -> 3.266e-02      31
    505      19.8187     370      107 -> 108   9.157e-02 -> 4.694e-02      61

The jump in `sigma` is a fixed ratio while `dk` halves, so the secant grows like `1/dk` and
**cannot** settle. Refining the seed does not approach a slope; it approaches a discontinuity.
Rank ranges 106-109 over the window, so this is not one bad point but a boundary the grid keeps
re-crossing.

## Freezing the rank removes it, and does not hide the wells

Same case, same basis, same collocation, same `rtol` -- only `lam_ref=(a, b)` added, so the rank
is chosen once at 107 instead of re-derived per `lam`:

    n       threshold: max secant / flips     frozen: max secant / flips    min sigma thr / frz
    64          2.0485 /  7                      0.7602 / 0                4.88e-04 / 4.88e-04
    127         3.3788 / 23                      0.7616 / 0                2.03e-05 / 2.07e-05
    253         6.0387 / 31                      1.0718 / 0                2.03e-05 / 2.07e-05
    505        19.8187 / 61                      1.5155 / 0                2.03e-05 / 2.07e-05

An order of magnitude off the secant, every flip gone, and **the minimum sampled `sigma` moves by
2%** -- so the discontinuity was not the wells and removing it does not remove them.

## And it recovers the declines, introducing nothing

Whole grids, unexercisable cells excluded, the ONLY change being `lam_ref=(a, b)` on the pencil:

    grid        threshold rank                    frozen rank
    145*pi^2    22 right, 0 wrong, 23 declined    44 right, 0 wrong,  1 declined
    65*pi^2     47 right, 0 wrong,  6 declined    47 right, 0 wrong,  6 declined
    50*pi^2     53 right, 0 wrong,  0 declined    53 right, 0 wrong,  0 declined

**22 of 23 declines recovered, no wrong answer introduced, and nothing else moves.** Compare the
hand-over above, which bought its 27 with 9 wrong answers: the difference is that this removes
the discontinuity while that one asks the search to bound it.

Two things this does NOT say. `65*pi^2`'s six declines are unaffected, so not every decline is a
rank flip and this is one cause rather than the cause. And it is measured on the split-degeneracy
rectangles only -- the roster pencils are also threshold-rank, and whether they carry the same
artifact is untested.

## Two remedies measured NOT to work

**Raising `MAX_DOUBLINGS`.** The history above says it cannot help, and it does not: at 12
doublings the seed is ~260k points and still rising. This is the cost the table predicts, paid
for nothing.

**Handing a non-settled seed to the cover phase**, on the argument that `MAX_GROWTHS = 16` is
already the falsification test. Measured across five cluster grids, unexercisable cells excluded:

    configuration       right   wrong   declined   median evals at 145
    decline (shipped)     215      1        35            961
    hand over to cover    242      9         0           3249

It converts 35 honest declines into 27 more right answers and **9 wrong ones**, three of them
absurd -- 176 of 6, 184 of 6, 72 of 8 -- and triples the cost. With `L` set from a max secant
that is a rank-flip artifact, the exclusion radii are far too small, nothing is excluded, and
every live region becomes a bracket. **The decline is doing real work and must not be relaxed at
the seam that reports it.**

## What this means

The bracketer is CORRECT here. It is being handed an objective with genuine discontinuities and
is refusing to bound a slope that has no bound -- which is exactly what `_seed`'s docstring says
the bootstrap is for. The remedy is not in `bracket.py` at all.

It is also not a property of these test cases. The threshold rank is the SHIPPED default --
`MPSEigensolver` takes no `lam_ref` and `pencil.rank_at` re-derives per `lam`, on the argument
that it "is the only rule that cannot pick an unsafe rank". That argument is about picking a rank
that is too high; it says nothing about the objective being piecewise in `lam`, which is what
this section measures.

So this joins the open `_UNSAFE_RATIO` / `SUITE_RTOL` question rather than standing alone: the
frozen path is unsafe at cuts the threshold path tolerates, and the threshold path is
discontinuous where the frozen path is smooth. Neither has been measured against the other on
the question that matters, which is what a SEARCH can work on. A rank policy chosen for the
pencil's conditioning is being inherited by an optimizer, and section 17's lesson applies
unchanged: a solver whose objective moves for reasons the caller cannot see is not one a shape
optimization can descend on.

Note this is the same mechanism as section 6's narrow plateau spikes, where the threshold rank
"flips 85 <-> 86 on a one-ulp change in `lam` and each flip moves `sigma_1` by 2.3x" -- recorded
there as an oddity affecting one gate cell. Here it is the dominant effect and it costs 30 of 55
cells outright.
