# Proposal: bracketing for `chladni` v1

A design for the eigenvalue-bracketing stage of a Dirichlet-Laplacian solver on planar domains,
together with the refinement stage it feeds and the validation apparatus that keeps both honest.

Everything here was measured on one family -- the `(1+delta) x 1` rectangle, solved by the
method of particular solutions with a plane-wave basis -- over roughly 500 tests and seven
tagged versions. Section 8 says which parts should transfer and which are the rectangle's
own. **Read section 7 before trusting any number in sections 1-4**: the single most reliable
finding here is about how measurements mislead, not about the algorithm.

---

## 1. The problem, precisely

Given a pencil `A(lam), B(lam)` whose smallest generalized singular value `sigma_1(lam)` -- the
*tension* -- drops to a floor at each Dirichlet eigenvalue, produce **a set of non-overlapping
intervals, each believed to contain an eigenvalue**, over a search window `[a, b]`, at a
requested relative precision `prec`.

Two metrics are hard:

* **Coverage.** Every eigenvalue lies in some bracket. A lost one cannot be recovered downstream.
* **Purity.** No bracket contains eigenvalues spanning more than `prec * lam`.

Everything else -- bracket count, spurious brackets, cost -- is reported and traded.

Three facts about the objective shape the whole design:

1. Near an eigenvalue, `sigma_1(lam) ~ sqrt(eps^2 + c^2 (lam - lam*)^2)`: a cone with a rounded
   bottom, floor `eps ~ 1e-15`, wall slope `c`.
2. One evaluation returns the **whole ascending vector** `sigma_1 <= sigma_2 <= ...`. Reading
   `sigma_2` is free, and it carries the information that decides clustering.
3. Evaluations are expensive (a QR, an SVD and a GSVD; 5-70 ms depending on basis size), so cost
   is counted in evaluations and nothing else.

---

## 2. The bracketing algorithm

### 2.1 Work in the wavenumber `k = sqrt(lam)`

Not a numerical trick -- it is the Helmholtz wavenumber, the natural variable of the MPS
formulation. The wall slope satisfies `c(lam) = C / sqrt(lam)`, so under `u = lam**p`

```
dsigma/du = (C/p) * lam**(1/2 - p)
```

which is constant for `p = 1/2` and no other power. Measured slope spread over 15 levels:

| transform | `lam` | `lam^(3/4)` | `lam^(1/4)` | `log lam` | **`sqrt(lam)`** |
|---|---|---|---|---|---|
| spread | 57% | 27% | 17% | 32% | **5%** |

**This was worth 1.8x of a 2.6x total improvement -- more than every algorithmic change
combined.** A single global slope bound is conservative by the ratio of the largest local slope
to the smallest, and paying that everywhere is the largest avoidable cost in the method.

Brackets are mapped back and returned in `lam`. The two stages want different variables: the
slope is constant in `k`, and the cone is symmetric in `lam`, which is what the refiner wants.

**Widen the returned bracket by ~4 ulps.** The round trip is not exact: a bracket correctly
fencing `k*` can fail to fence `lam* = k*^2` after squaring, because the endpoint rounds the
wrong way. Observed as a one-ulp coverage loss whose occurrence depended only on seed density.

### 2.2 Seed: size it from Weyl, then double until the slope settles

Weyl with the perimeter correction, `N(lam) ~ A*lam/(4*pi) - P*sqrt(lam)/(4*pi)`, gives the
expected eigenvalue count to 1-11% on this family. It says nothing about **where** anything is
and must not be used as if it did. It sets the starting density: **1.5 samples per expected
gap.**

Then double -- as `n -> 2n-1`, so each grid *contains* the previous one exactly -- until the
largest observed secant stops growing by more than 10%.

* `n -> 2n` does **not** nest. New points land within an ulp of old ones, and a secant across
  such a pair is rounding rather than slope: an unguarded estimate reached **7.3e10** against a
  true 0.6, and a slope model that never shrinks cannot recover from believing it.
* The starting density barely matters: 0.5, 1.0 and 1.5 samples per gap all cost the same,
  because the doubling adapts. Choose the larger for margin.
* The bootstrap **is** the falsification test. An estimate that never settles is the evidence
  that nothing bounds the local slope -- decline (section 2.8).

### 2.3 Slope model: honest about what it is

`L = safety * (largest secant observed)`, with `safety = 1.5`, monotone non-decreasing, ignoring
pairs closer than `1e-9` of the sampled span.

**No finite sample set bounds a Lipschitz constant from above.** The largest observed secant is
a *lower* bound; inflating it produces a number with no formal status. Do not call it a
certificate. Two measurements set the policy:

* The bound is nearly tight: using the 99th-percentile slope instead of the max wrongly excluded
  13 of 600 samples. A 1.4% underestimate loses coverage.
* Margin is cheap in the seed: safety 1.2 -> 2.0 cost +40% evaluations for identical coverage.

Report growth past 10% *after* something has relied on the value; enough of those means the
premise is failing. Recompute exclusions from the current `L` every pass, so an underestimate
becomes extra cost rather than a silent miss.

### 2.4 Cover: exclusion, and its structural cost

Between adjacent samples `i, i+1`, the Piyavskii lower bound is
`env = (v_i + v_j)/2 - L*(x_j - x_i)/2`. Where `env >= 0`, no eigenvalue can lie between them.
Merge contiguous live pairs into **regions**; sample the point attaining the lowest bound in the
region with the deepest unexplained dip.

**Know the cost law before trying to optimize this.** Exclusion of a wall pair at distances
`u_a > u_b` needs `u_b/u_a >= (safety-1)/(safety+1)`, independent of `c`. So a wall is covered by
a **geometric ladder** of ratio `(safety+1)/(safety-1)`, costing `log(range)/log(ratio)` per side.
Predicted against measured, exactly:

| safety | 1.2 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|
| predicted ratio | 11.00 | 5.00 | 3.00 | 2.00 | 1.50 |
| measured | 11.00 | 5.00 | 3.00 | 2.00 | 1.50 |

Two consequences, and they retire a family of ideas:

* **A better prediction of where the eigenvalue is cannot help the cover phase.** A sample placed
  *exactly on* an eigenvalue leaves the live region unchanged; so do two straddling it to 1e-9.
  What is live is the *wall*, and the wall yields to the ladder or to nothing. Six attempts to
  make cover cheaper by sampling more cleverly all failed for this reason.
* **A larger safety factor is a finer ladder, not just margin.** It costs `1/log(ratio)` per side.

### 2.5 Isolation gate: `sigma_2` decides when the fast path opens

A locator that converges to a single minimizer will happily converge to one member of a close
pair and leave the other outside the bracket -- this is exactly how the `grid + bisect` baseline
falls from coverage 1.0 to 0.577. So the fast path is gated.

At the deepest sample of a region, `sigma_2 / c` estimates the distance to the next eigenvalue.
Certify the region as holding one eigenvalue when `neighbour > margin * region_width`, with
**`margin = 1.5`**.

Four details, each of which cost a coverage or purity failure to learn:

1. **Judge the REGION, not the certificate triple inside it.** The region is what gets handed to
   the locator and can be several times wider. A pair 8.4e-4 apart was certified inside a region
   1.24e-3 wide, then collapsed onto one member.
2. **`c` must be an independent UPPER bound**, taken as `max(local estimate, L)`. Where a region
   holds two eigenvalues its flanks span both cones, the local estimate comes out too small, the
   neighbour looks further than it is, and the gate certifies precisely what it must not. Cost
   3 of 26 eigenvalues.
3. **Silence is not permission.** No descent certificate means no decision; the region stays on
   the slow path.
4. **Do NOT read the level's multiplicity off the sigma vector.** Counting branches at the floor
   and reading the first one that is not is *correct in principle* -- at a multiplicity-2 level
   `sigma_2` is that level's own second branch -- and saves 40% on exactly degenerate cases. It
   is also ambiguous at range by construction: approaching an unresolved pair from distance `u`,
   `sigma_2/sigma_1 = 1 + d/u`, which is 1.0006 at the start of a cover pass. Swept over two
   windows the failure count is **not monotone** in the threshold:

   | floor_ratio | 1.0 | 1.05 | 1.2 | 1.5 | 2.0 |
   |---|---|---|---|---|---|
   | cost/target | 29.4 | 29.6 | 30.1 | 29.0 | 27.2 |
   | failures | 0 | 1 | 0 | 1 | 2 |

   A parameter whose safety alternates as it is tightened has no safe setting, only settings that
   pass the cases tried. Left off; the 40% is given back. **If `chladni` wants it, it needs a
   floor estimate the objective supplies, not one inferred from samples so far.**

### 2.6 Locate: predict with a parabola, then straddle the prediction

`sigma_1**2` is exactly quadratic near an eigenvalue, so the vertex through three points is the
answer rather than an approximation.

But prediction alone does not tighten a bracket. **The bracket's width is set by whichever side
of the prediction has no sample near it**, and the prediction error has a systematic sign -- the
two branches meeting at the bottom have slightly different slopes -- so every scheme that samples
*at* the prediction converges from one side and the far flank never moves.

So place the sample a fixed fraction of the bracket to one side of the prediction, on whichever
side is not already fenced. **`straddle = 0.02`**, which is a sharp optimum:

| straddle | 0.01 | **0.02** | 0.04 | 0.08 | 0.15 |
|---|---|---|---|---|---|
| relative cost | +1% | **0** | +89% | +425% | +823% |

Four designs were tried before this one worked, and each is a plausible thing to try again:
plain V-interpolation stalled with the bracket stuck; bisecting the lopsided side cured the stall
at no better than exclusion; mirroring the near side across the prediction oscillated.

### 2.7 Stop at isolation, and emit only what is fenced

**Purity does not constrain the width of a bracket holding one eigenvalue** -- its span of
contained eigenvalues is zero whatever the interval. Refining past isolation buys nothing the
metric asks for, and the refiner reaches machine precision from a wide bracket in ~7 evaluations.
Hand it over once the neighbour is **4 bracket-widths clear**.

What that gives up: a bracket narrower than `prec*lam` cannot be impure *however wrong the gate
was*. That guarantee goes; purity then rests on the gate. Keep the other one --

**Every emitted bracket endpoint must be a SAMPLED point carrying a three-point descent
certificate** (`f(p) > f(q) < f(r)`, rigorous under continuity alone). Never an extrapolated
root: two-point secant roots from opposite walls of a close pair were measured to *cross*,
yielding an inverted bracket or a sliver containing nothing.

Depth is what distinguishes a cone from a wiggle, and it must be normalized **by the further
flank**. By the nearer one, a cone bottoming out mid-cell has almost equidistant flanks, scores
~1, and is rejected: 19 of 26 eigenvalues found instead of 26, at *every* grid density.

Before emitting an unconfirmed region, spend up to 3 evaluations at the Piyavskii point to find a
descent. Keep anything that still *looks* like a dip: dropping a region costs an eigenvalue,
keeping one costs ~7 evaluations that the refiner spends discarding an empty bracket.

### 2.8 Declining

A method may assume the pencil is well behaved at the scales its precision makes relevant, detect
that it is not, and stop. Declining is a **third outcome**, neither pass nor failure. Two triggers:

* **Slope instability** -- the seed's estimate never settles over its doublings.
* **No progress** -- N samples bought no reduction in the unresolved regions.

Declining is only honest if reported. A method that declines whenever it is about to be wrong has
a perfect record over what it accepted and has demonstrated nothing. **Quote the decline rate
beside the pass rate, over the same case set**, always.

---

## 3. Refinement

### 3.1 Use iterated parabolic interpolation on `sigma_1**2`

`lappy.opt.parabolic_iter_min` is the right algorithm and beats the alternatives on **both** axes.
Measured over 26 eigenvalues at `prec = 1e-14`, brackets of width 1e-6 to 1e-1:

| refiner | median relative error | evaluations/bracket |
|---|---|---|
| **iterated parabolic on `sigma_1**2`** | **1.6e-16** | **5.5 - 7.9** |
| Brent on `sigma_1**2` | 5.4e-10 to 5.6e-5 | 5.0 - 11.7 |
| straddled interpolation on `sigma_1` | 0 (exact) | 48 - 63 |

**Brent does not exploit the parabolic structure.** Its parabolic step is guarded by a
golden-section fallback and it terminates long before the curvature information is used up. Do
not reach for `scipy.optimize.minimize_scalar` here.

Note also: **squaring the objective does not cost half the significant digits.** The
`sqrt(eps_machine)` limit belongs to methods that locate a minimum by comparing function *values*.
Solving for the vertex from the curvature does not suffer it.

### 3.2 Four guards the audit says to add

1. **Refuse when the parabola never closed** -- no successful step at all. `C <= 0` means the
   three points do not describe a single minimum: an empty bracket, or one holding two
   well-separated eigenvalues, where the fit is across a W. In the second case the lowest sampled
   point can be a bracket *endpoint*, 1e-3 from either eigenvalue, returned as success.
2. **But only when there was no progress.** `C <= 0` *also* fires at the end of a good
   refinement, once the three points have collapsed onto the floor and the curvature is gone.
   Treating that as failure discards correct answers.
3. **Track the lowest point seen independently.** The triple is re-formed by position rather than
   value, so a bad vertex can discard the sub-interval holding the minimum.
4. **Guard `dx == 0`** in the vertex formula, and clamp the duplicate-point nudge inside the
   interval.

Accept by comparing against the cone *outside* the bracket, where it must be rising -- not
against the bracket's own ends, which are all near the floor once the bracket is narrow.

### 3.3 Set the tolerance from the target precision

`xatol = max(prec * |lam|, 4 * ulp(lam))`. A tolerance derived from `lam` alone can *exceed* the
bracket it is refining -- a 2.7e-13 bracket handed a tolerance of 7.9e-13 -- so the refiner
terminates before doing anything and returns a point that is not the minimum.

### 3.4 Charge refinement by what it costs

A flat per-bracket charge is wrong in both directions: 3 evaluations for a bracket of width
1e-12, 12 for width 20, and 2-7 to *reject* an empty one. Measured against a flat 10, the model
overcharged tight brackets threefold and undercharged spurious ones by the same factor -- and it
reversed the answer to "should the search refine or hand off?" **twice**.

Clusters closer than `prec` refine cleanly: 7 evaluations, accepted, landing between the two
members, with the degenerate-points guard never firing across any close pair of any test case.

---

## 4. Cost, for calibration

Per eigenvalue, including the refinement each bracket hands on, over the crossover sweep and
exact degeneracy at `prec` in {1e-3, 1e-9, 1e-14}:

| window | eigenvalues | search | refine | **total per eigenvalue** |
|---|---|---|---|---|
| [10, 120] | 6-8 | 24.1 | 7.1 | **31.1** |
| [10, 400] | 26-27 | 26.3 | 6.1 | **32.4** |
| [10, 500] | 33 | 25.8 | 6.0 | **31.8** |

Roughly **30 evaluations per eigenvalue**, near-flat in window size and in `prec` from 1e-3 to
1e-14. For comparison, a uniform grid dense enough to detect everything costs ~43 per eigenvalue
and **fails purity**, and grid-then-bisect costs 74-95 and **loses a third of the eigenvalues**.

---

## 5. Validation apparatus

This is not optional infrastructure. It found every one of the ~15 defects in this repo, and
inspection found none of them.

### 5.1 Build the scorer before the searcher

Coverage and purity hard; spurious, fineness and cost reported. Score against **distinct
locations** with multiplicity, never against modes: at an exact degeneracy several modes sit at
the identical `lam`, and no interval can separate a level from itself.

Purity tolerance keyed to the group's **largest** member -- forced, not chosen: with
`h(t) = (t - g0)/max(t,1)` increasing in `t`, the largest member is binding and the predicate is
exactly equivalent to "every prefix would have been admitted by the merge rule".

Brackets are **closed**; one containment predicate shared by coverage and purity.

### 5.2 Validate the metric with controls, before believing any result

Six adversaries that must fail, and -- equally important -- **five oracles that must pass**. The
oracles catch a metric too strict to accept a correct answer, which is the failure that looks
like rigour.

| control | must |
|---|---|
| whole window | fail purity |
| grid partition, returned unevaluated | fail on cost |
| evaluated fine grid | fail on cost |
| oracle minus one target | fail coverage |
| oracle with two merged | fail purity |
| brackets in the gaps | fail coverage |
| declines everything | report **declined**, never pass |
| oracle at the fine end / coarse end / mixed / split cluster / endpoint-exact | **pass** |

A partition of the window returned *without evaluating anything* satisfies coverage, purity and a
search-only budget. Only a cost model that charges for the refinement handed on rejects it.

### 5.3 Baselines, separate from controls

Honest but dumb methods that may not read truth: `uniform_grid`, `grid_then_bisect`,
`random_probe`. They fail in the two ways the metric is built to separate -- the grid holds
coverage and fails purity, grid-then-bisect holds purity and fails coverage -- which is the check
that the scoring is not merely agreeing with itself.

### 5.4 Sweep the merge/split crossover, on three windows

Place the window's **tightest** pair at known multiples of `prec` -- 0.5x, 0.8x, 1.3x, 3x, 30x,
300x -- so the method is exercised on both sides of the licence. A method that always merges and
one that always splits both look correct on a sweep that stays on one side.

Two traps:

* **The tightest pair is not the one you are dialling.** Relative separation scales as
  `|m^2-n^2|/(m^2+n^2)`: 3/5 for the (2,1)/(1,2) pair but 7/25 for (3,4)/(4,3). A case placed by
  the former leaves the latter twice as tight as intended.
* **Near the float64 floor the dial is quantized** (~2.7% at 1e-14). Classify each case by the
  separation it **achieved**, never the one requested. Requesting an unrepresentable separation
  rounds `delta` to zero and silently yields an *exact degeneracy* -- the opposite case wearing
  the right label.

Also sweep **exact degeneracy as a class**, not as one case: the crossover sweep varies a pair's
separation and never reaches zero, while at `delta = 0` every level may be genuinely multiple.
That is the regime a shape optimization crosses whenever eigenvalues meet.

### 5.5 Three test tiers, and a standing gate

| tier | contents | time |
|---|---|---|
| default | everything with no objective evaluations: scorer, components, properties | ~3 s |
| `-m slow` | floor preconditions, baselines, searcher behaviour | ~80 s |
| `-m gate` | **the standing acceptance gate** | ~5 min |

**Run the gate before tagging anything.** It covers the crossover sweep and exact degeneracy on
**three** windows at `prec` in {1e-3, 1e-9, 1e-14}. The widest exists specifically to hold a
level of **multiplicity three** (in the rectangle, `50*pi^2`, where (1,7), (7,1) and (5,5)
coincide, splitting into a genuine three-member cluster at `delta > 0`). Every other cluster is a
pair.

**Have the gate assert its own coverage** -- a test that the multiplicity-3 level is present,
does split, and is bracketed -- so it cannot quietly stop covering the case it was widened for.

### 5.6 The floor is a precondition, not a diagnostic

Assert that `sigma_m` reaches the tension floor at every known eigenvalue of multiplicity `m`,
before believing any measurement. Three separate ways of getting the discretization wrong each
degraded the objective by orders while looking healthy.

---

## 6. Objective hygiene

Findings about the MPS pencil itself, all of which cost real accuracy or speed:

* **The rank cutoff must actually truncate.** At `rtol = 1e-16` the numerical rank came back
  equal to the full column count: the truncation was inert and every noise-floor direction was
  retained. It did **not** move the floor -- which is why a floor check never noticed -- but it
  made `sigma_1` rough *between* eigenvalues, where a search lives. Moving to `1e-15` cut the
  plateau's relative second difference from 8.9e-2 to 3.9e-3 and spurious local minima from 7 to
  **0**, while the barrier between a close pair *rose*. Cost: ~5e-16 of floor.
* **Size the collocation to the pencil**, not to a fixed floor: ~3x boundary and ~2x interior
  points per column. The flat 1200/600 defaults were ~4x oversized for no accuracy.
* **The basis pad can be smaller than folklore.** `pad = 1.2` on `n_useful(kR)` held the floor
  with ~1.4x margin in columns; breakdown was at ~0.75.
* **The floor fails from the TOP of the window down**, where basis demand is highest. A floor
  checked only at the low end proves nothing.
* **Attach `expected_count(lo, hi)` to the objective**, from Weyl. It comes from the PDE, which is
  honestly where it comes from, and a search may read it for sizing only.

---

## 7. How to not fool yourself

The most transferable finding in this work. **Every claim measured on the case that motivated it
was overstated, without exception**, until a case set fixed in advance became the gate.

| claim as first made | measured on the fixed sweep |
|---|---|
| "2.2-2.7x cheaper" | 1.77-1.79x |
| "1.4-1.6x cheaper" | 1.08-1.28x |
| "14-34% cheaper" | 10-22% |
| "spurious brackets down to zero" | 1-2 remain |
| "coverage 1.0 at every precision" | lost eigenvalues on 3 of 7 cases |

Worse, twice the narrowness cost **coverage** rather than only credibility: a version passed
every gate that existed and lost three eigenvalues on a window nobody had run.

Practices that actually worked:

1. **Fix the case set before measuring**, and widen it only on its own merits -- never because
   widening makes a change look good.
2. **Tag versions, and audit the tags later.** Correct a tag message in place when a wider sweep
   falsifies it, and say what it originally claimed. The pattern is more useful than the number.
3. **Write the ablation that must fail.** If disabling a guard does *not* break anything, the
   guard is not doing what you think. Two `sigma_2` guards here each looked load-bearing; the
   ablation had to remove **both** to show either mattered.
4. **Prefer a measured cost model.** The flat per-bracket charge reversed an architectural
   conclusion twice.
5. **Record failed ideas in the source, with numbers.** Six cover-phase optimizations failed for
   one shared reason; without that written down, a seventh is inevitable.

---

## 8. What should transfer, and what should not

**Expect to transfer** -- these rest on the MPS formulation or on generic structure:

* `k = sqrt(lam)` as the search variable. It is the Helmholtz wavenumber; the constant-slope
  property follows from the basis being built at wavenumber `k`, not from the rectangle.
* `sigma_1**2` exactly parabolic near an eigenvalue; the cone model generally.
* `sigma_2` as a free, conservatively-biased distance to the next eigenvalue.
* The exclusion ladder law -- derived from the envelope test alone, independent of `c`.
* Weyl sizing from area and perimeter, which are known for any domain under optimization.

**Do not assume** -- these are measured only here:

* The numerical values: `c*sqrt(lam) ~ 0.27`, floor `~1.5e-15`, `straddle = 0.02`,
  `margin = 1.5`, `safety = 1.5`. Re-measure all of them.
* That the slope spread in `k` stays at 5%. A domain with corners or a re-entrant angle may
  behave differently, and if the spread is large the case for a **local** slope model returns.
* That no window holds a cluster of multiplicity above 3.
* **That stopping at isolation is safe.** It trades a guarantee that holds unconditionally for
  one resting on the `sigma_2` gate, which has produced coverage failures twice. Of everything
  here, this is the change most in need of confirmation elsewhere.

---

## 9. Recommended build order

1. **Scorer, controls, baselines.** No searcher. The controls must be green before any result
   means anything.
2. **Truth, and the crossover sweep** including exact degeneracy and the highest multiplicity the
   domain family offers. Fix the case set here.
3. **Objective hygiene**: floor assertion across the window, rank truncation that truncates,
   collocation sized to the pencil.
4. **Refiner** -- iterated parabolic with the four guards. It is small, it is needed to price
   brackets, and it settles where the stage boundary should fall.
5. **Searcher**: seed -> cover -> gate -> locate -> confirm, in `k`.
6. **The standing gate**, then tag.

Expect the searcher itself to be the smallest part. In this repo it is ~400 lines against ~2500
of scoring, controls, baselines and tests -- and that ratio is the reason the numbers can be
believed.

---

## 10. Future work, in the order I would do it

1. **A second domain.** Every number here is one rectangle. A domain with a known spectrum -- a
   disc, an equilateral triangle, an L-shape with a re-entrant corner -- would test the
   transferable claims of section 8 directly. **This is worth more than any remaining speedup**,
   and it is the only item here that could invalidate the design rather than tune it.
2. **Asymmetric cone prediction in the locator.** The straddle is pinned at 0.02 by a prediction
   error dominated by the two branches having different slopes (~1.7%). Fitting `c_left` and
   `c_right` separately should shrink it. Estimated 10-20% of the locate phase.
3. **A `sigma`-vector completeness audit.** The full vector at one point gives distances to the
   nearest several eigenvalues to 5-50%. Comparing that against what has been located is a
   *falsifiable* check on coverage, where the Lipschitz argument can only be silently wrong. It
   would also let the exclusion ladder be coarser -- but it converts a certificate into an
   audited estimate, so do it deliberately, paired with the decline mechanism.
4. **Revisit the multiplicity reading** *only* with a tension-floor estimate supplied by the
   objective. It is worth 40% on degenerate cases, which is exactly the regime shape
   optimization lives in.
5. **Batched / vectorized evaluation**, if the pencil allows several `lam` at once. The cover
   phase is inherently sequential, but the seed is embarrassingly parallel and is ~20% of cost.

**What not to try**, with the reason: anything that makes the cover phase sample more cleverly.
The ladder law says a wall yields to a geometric ladder set by the safety factor and to nothing
else. Six attempts have failed; a seventh will too.

---

## 11. Honest summary

A method that reaches **coverage 1.0 with pure brackets at ~30 evaluations per eigenvalue**,
across separations from 0.5x to 300x the requested precision, precisions from 1e-3 to 1e-14,
exact degeneracies, and a multiplicity-3 cluster -- on three windows, with a standing gate that
has caught two coverage regressions that every narrower gate passed.

It is 2.6x cheaper than the first working version, and **1.8x of that came from changing the
variable**, not from improving the search.

The largest uncertainty is not in any of the numbers. It is that all of them come from one
rectangle.

---

# As built in `chladni`: deltas from this spec

What the port found. As in `geometry.md`, `particular_solutions.md` and `search.md`, the body
above records what was intended and this section records what the implementation measured; the
gap is the useful part. Every number here is from `chladni`'s own tiers, not from the
originating repo.

## What is built

| | |
|---|---|
| `docs/design/candidates.md` | the stage seams every candidate plugs into, and how a method that does not decompose is compared fairly |
| `tests/objective.py` | the shared seam: one cache, one evaluation counter, `Declined` as a value |
| `tests/bracket_score.py` | coverage / purity / spurious / fineness / **measured** cost, the `prec` ladder, separation classification |
| `tests/bracket_controls.py` | six adversaries and **five oracles** |
| `tests/baselines.py` | `uniform_grid`, `grid_then_bisect`, `random_probe` -- may not read truth |
| `tests/surrogate_cases.py` | tier 1, with `adequate()` |
| `tests/rect_suite.py` | tier 2, with `assert_floor` and the new `assert_smooth` |
| `tests/hard_roster.py` | tier 3: disk, eq_tri, L-shape, each with the `prec` its basis supports |
| `tests/v1_refine.py`, `tests/v1_bracket.py`, `tests/v1_multiplicity.py` | the prototype: stages 1, 2 and 3, as scored candidates |
| `tests/level_score.py`, `tests/level_controls.py` | the multiplicity scorer, and the adversary that broke the one it replaces |
| `tests/test_gate.py` | the standing gate, `-m gate` |
| `src/chladni/asymp.expected_count`, `src/chladni/mps/calibrate.py`, `MPSPencil.with_rtol`, `MPSEigensolver.findability` | the parts that are features rather than candidates |

## Corrections to this document

**Section 2.4's exclusion test is right and my first reading of it was not.** `env >= 0`
assumes `sigma_1` reaches zero at an eigenvalue, and it bottoms at `eps > 0`, so a finely
sampled well can exclude itself. I added a `tau` threshold, which broke every case (0 of 66).
The actual resolution is that the cover loop stops on **purity by width** before that regime is
reached, so the zero threshold never bites. Recorded because the objection is reasonable and the
answer is structural rather than a tuned constant.

**Section 3.4's refinement-cost slope does not transfer.** Fitted to this document's two
anchors -- 3 evaluations at width 1e-12, 12 at width 20 -- the model reads 1.46 evaluations off
on `chladni`, over-predicting wide brackets by half again. Re-fitted on 126 real refinements:
`evals = 3.73 + 0.274*log10(width/xatol)`, median residual 0.84 over a measured range of 4 to
10. Rejection measured flat at **3** evaluations, inside this document's quoted 2 to 7.

**Section 5.4's separation formula is the two-mode case of a more general rule.**
`2*|n^2-m^2|/(m^2+n^2)` is correct for a level holding two modes and wrong for a cluster: at
`50*pi^2` the modes are (1,7), (5,5), (7,1), and (5,5) sits exactly halfway, so the minimum
ADJACENT gap grows at half the outer-pair rate. Every cluster case dialled with the pair formula
landed at exactly 0.5x its intended band. The general rule is
`2*min(diff(sorted(m^2)))/(m^2+n^2)`, which reduces to the pair formula when a level holds two
modes.

**"Tightest" must mean relatively tightest.** On `[10, 520]` the (4,5)/(5,4) level is
relatively 2.7x tighter than (2,1)/(1,2) while being absolutely 3x WIDER. Purity is keyed to
`span > prec*lam`, so the relative gap binds, and an absolute `argmin` names the wrong level.

**Section 6's `rtol = 1e-15` is not the right value here, and the reasoning differs.**
`chladni` thresholds RELATIVE to the largest singular value where `lappy` was absolute. Measured
on `rect(1, 1/phi)` over `[5, 120]`, 92 columns, worst tension at a true eigenvalue and count of
sampled local minima away from one:

    rtol       rank(lo, hi)   floor      floor ok   spurious minima
    1e-16        42, 92       1.1e-15      yes            264
    1e-15        28, 51       2.1e-15      yes              3
    1.49e-15     28, 51       2.1e-15      yes              0
    1.5e-12      23, 45       4.2e-13      NO               0

`1e-16` is the inert regime by derivation, being below machine epsilon. `1e-15` leaves 3
spurious minima where this document measured 0. And the choice is set by the FLOOR, not the
spurious count: `1.5e-12` also reaches zero spurious but costs two orders of floor and makes
the bottom of the `prec` ladder unexercisable. The suite uses `sqrt(FINEST_PREC * eps)`,
`chladni`'s own derived rule at the tightest precision the sweep must serve.

**Section 6's collocation guidance is right and its own numbers were still generous.** At
`SUITE_RTOL`, 92 columns: the worst floor is 1.4e-15 at 8x boundary / 3x interior and 6.1e-16 at
1.2x/1x, i.e. **it does not move at all**, while an evaluation costs 3.2 ms against 2.0. The
suite uses 3x boundary and 1x interior; the margin over 1.2x is there so a test that has to RUN
does not fail on a symmetry that makes the boundary sampling degenerate, and it costs 5%.

**Oversized collocation was also making the objective ROUGHER, not just slower.** The gate's
first run reported six of its cells UNEXERCISABLE because the tension curve admitted no slope
bound -- the largest observed secant climbing 0.64 -> 61 -> 261 across doublings against a wall
slope of 0.64 on the same curve, from narrow spikes in the plateau far from any eigenvalue.
After sizing the collocation to the pencil that count fell to **one**, and the gate went from
10:16 to 4:48. So part of what looked like a property of those windows was the 8x/3x point sets.
The one cell that remains is genuinely rough and is reported as such.

**`rtol` is safe to raise up to `prec`.** Above the working precision the truncation discards
directions the answer depends on; below it, it discards directions whose contribution is
already negligible. So the calibration searches `[sqrt(prec*eps), prec]` and needs no ceiling
constant of its own.

## `kappa` ports; the note that predicted otherwise was being careful

`notes/narrow_wells_and_rtol.md` warned that `kappa = 2.38` is specific to `lappy`'s
normalization of `sigma` and that `chladni` should expect a different constant. Re-measured on
six domains whose plane-wave basis reaches a floor below 1e-6, taking `C_k` on the straight
flank: **2.591 +- 0.226 against the study's 2.384 +- 0.231** -- within one standard deviation.

And the `|Omega|/|dOmega|` scaling is confirmed exactly and independently. The same 3:1
rectangle at three sizes, which is the test that separates `A/P` from area:

    scale        0.5      1.0      2.0
    C_k       0.4613   0.9225   1.8448      exactly proportional to the length scale
    C_k/(A/P) 2.4600   2.4600   2.4598      identical to four significant figures

`C_k/area` over those three reads 2.46, 1.23, 0.61. The residual spread is systematic in the
same direction the study found -- fat domains high (disk 2.82, square 2.81), elongated low
(rect 5x1 2.22).

## The rank-truncation lever is INERT in `chladni`

The study's first remedy was to raise `sigma_base` by loosening the truncation, having measured
it move by a factor of 48. On a `chladni` rectangle pencil it does not move at all: **0.3329 to
0.3346, a range of 0.5%, while the retained rank goes 47 -> 19.** Two candidate reasons, neither
settled: a relative threshold removes columns proportionally so the plateau, a ratio of norms,
stays put where an absolute cutoff bites at a fixed level; and `chladni`'s plateau is already
healthy across domains (0.12 to 0.36 over seven) where the study's varied two orders.

So the shipped default is the **check**, not the fix -- which is what the note itself argues
("the check is arguably more valuable than the fix"). `calibrate_rtol` is implemented and
exported, with the inertness pinned by a test so it is not rediscovered. The check discriminates
where it should: a sharp elongated triangle needs 8x the scan density of a square, and reports
it as `UNFINDABLE (5.3x short) ... needs dk <= 9.43e-02`.

## The L-shape's five-order floor spread is a BASIS artifact

`notes/landscape_findings.md` section 20.1 measured this domain's floor at 3.1e-13 at
`k = 4.443` and 5.5e-08 at `k = 5.649`, and explained it structurally. With the recipe-grade
basis -- 80 pure Fourier-Bessel terms at the reentrant corner -- the same levels over `[5, 40]`
read 1.9e-15, 5.4e-15, 2.8e-15, 8.2e-15, 5.3e-15: **a spread of 4.2x, and 5.3e-15 at
`k = 5.649` against the 5.5e-08 recorded there.** Seven orders.

This is section 27.1's correction a second time -- there a 9.2e+04 floor spread on `rect81`
turned out to be a frozen rank over a wide window. It does not retire the per-eigenvalue
accuracy clause, since the floor still varies by 4x and a domain whose basis cannot be made
adequate will still show the large spread. It retires "the L-shape is the domain where the floor
varies five orders", which was a statement about a basis.

## The precondition lesson, three times over

The same mistake arrived on each tier independently, and it is the most transferable thing in
this port: **a case that violates a precondition measures the case, not the method.**

* Tier 1 scored 13 of 66 cells as coverage failures with a surrogate floor of `1e-13`, which
  cannot support `prec = 1e-14` at all -- the adequacy contract is `prec*lam >= margin*eps/c`
  and the request was a third of what the objective could deliver. At `1e-15`: 66 of 66.
* Tier 2's floor check `assert_floor` existed, implemented correctly, and was **never called
  from a test**.
* The gate's first run reported nine failures, most of them windows whose tension curve admits
  no slope bound at any sampled scale -- where the searcher declines, correctly.

So every tier now checks its preconditions before scoring, and a cell that fails one is reported
UNEXERCISABLE with the reason rather than counted as a failure or silently dropped.

## Being asked for more than the objective can give is not a failure

The searcher's stall and budget paths raised, which turned "I achieved 3e-12 where you asked for
4e-13" into "DECLINED, nothing found" -- losing brackets that were correct and hiding the one
number a caller needs. They now emit what is held, flag it `floor_limited`, and report
`attained_prec` beside `prec_requested`. This is `search.md` clause (2) in the bracketing stage:
meet `prec`, or report the attained bound and say which bound was active.

## Defects the tiers found in the prototype

Each was found by running, not by reading, and each is pinned by a test:

* **Budget defaulted to one eigenvalue** where the objective offered no Weyl count, so the
  locator exhausted it and returned partial results *silently*.
* **Regions retired by overlap**, so bracketing one member of a pair retired the whole region
  and lost the other -- coverage 0.75 on every pair, which is this document's own
  "converges to one member and leaves the other outside" arriving through the bookkeeping.
* **The fast and slow paths were collapsed.** The straddle places its sample at a predicted
  vertex, so on a region holding two wells it piles samples onto one and never lands between
  them: coverage 1.00 with purity failing at every precision. A non-isolated region must go to
  the exclusion ladder.
* **`STRADDLE` used as the fallback when the parabola does not close** shrinks the bracket by 2%
  per evaluation: measured creeping 6.2e-08 -> 3.4e-08 over twelve evaluations and stopping.
  Bisect the wider side instead.
* **"Silence is not permission" applies on the way OUT of the locator too.** With the post-check
  unable to certify, keeping the narrowed bracket discarded whatever else the region held.
* **A region lying entirely inside a well reads as FLAT** -- every sample on the floor, internal
  ratio ~1 -- and was dropped by the confirmation step. Flatness at the floor is the opposite of
  flatness on the plateau, so the level test is a fraction of the measured plateau.
* **The slope-growth limit of 4 threw away solvable cases.** Growth is COST, not error, because
  exclusions are recomputed from the current `L` every pass. On the multiplicity-3 cluster the
  seed settles at 0.31 against a true wall slope of 0.683 and needs 7 growths to reach
  `L = 1.082`; at 4 it declined having found nothing, at 8 it passes at 80 evaluations.
* **The loop budget counted cache MISSES**, so a refiner re-reading known points spun forever at
  zero cost.

## The standing gate, as it stands

    pytest              1380 passed          9.8 s     no pencil evaluations
    pytest -m slow        52 passed, 1 skip   51 s     floors, calibration, baselines, tier 3
    pytest -m gate       111 passed, 1 skip  4:48      the six-precision sweep on three windows

The single gate skip is a cell whose objective admits no slope bound at any sampled scale
(secant 0.72 -> 15.3 across doublings), reported UNEXERCISABLE with that reason rather than
counted as a failure.

## Stage 3 confirms this document's one deferred reading, on its own terms

Section 2.5.4 declines to read multiplicity off the sigma vector INSIDE the bracketer, and that
stands -- the reading is ambiguous at range by construction, since approaching an unresolved
pair from distance `u` gives `sigma_2/sigma_1 = 1 + d/u`, which is 1.0006 at the start of a
cover pass. Section 10 item 4 says to revisit it "only with a tension-floor estimate the
objective supplies".

Read AFTER location, against a wide bracket, it is unambiguous. Measured on the disk, the
equilateral triangle and the square's exact degeneracies: **14 levels, 14 correct readings, at
zero evaluations** -- every point a cache hit. Within a multiplet `sigma_{j+1}/sigma_j` runs
1.0 to 1.7 and across the gap 1e11 to 1e15, so `G = 10` is not a delicate parameter.

Two corrections to the rule as it stood in `search.md`:

* **(C2) is a prominence test, not a bare comparison.** Both existing implementations used
  `sigma_j` lower than its two flanks, i.e. prominence > 1. Roughness minima reach `1 + 2.4e-10`
  and true wells reach 72, so `>` counts noise as a branch; the spec's floor of `P = 2` does not.
* **(C2) takes the longest PREFIX.** The multiplicity is a contiguous block `1..mu` by
  definition, and checking only the top index and decrementing accepts a block with a hole in
  it, because the property is not monotone in `j`.

The wide bracket's distance is set by two constraints meeting. A genuine branch at `f*eps/c`
shows prominence `sqrt(1+f**2)`, so clearing `P = 2` needs `f > sqrt(3)`; and the flanks must
stay close enough for `sigma_2`'s asymmetry to show where (C1) double-counts, which measured
`f = 1` rejecting throughout the ambiguous band and `f = 3` accepting into it. `f = 2` is where
they meet. On an adequate pencil the factor is inert -- `eps/c` is ~1e-15 and no search samples
that finely -- so every factor from 1 to 10 gives the same 14 readings.

## Cost, measured on real pencils

Including the refinement each bracket hands on, at `prec = 1e-8`:

    window                eigenvalues   evaluations per eigenvalue
    pair    [10, 88]            4                  27
    cluster 50*pi^2 +-3%        3                  27
    wide    [10, 520]          35                  26

Against this document's ~30. The 35-eigenvalue window is the one worth noting: it is the case a
flat evaluation cap penalises, and it comes in at 26.
