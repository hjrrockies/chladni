# The eigenvalue search

Given a tension function `sigma(x)` returning the ascending vector
`sigma_1 <= sigma_2 <= ...`, find every approximate eigenvalue in `[a, b]` with its
multiplicity, using as few evaluations as possible.

This is the layer above `mps`. It sees only the tension function and the interval: no domain,
no basis, no pencil. That restriction is deliberate and is what makes two candidate searches
comparable, and what will let a later FEM or boundary-integral method reuse the same search.

The originating proposal is `lipschitz-cone-mps.md`. This document is the spec: it records
which of that proposal's claims have been measured, corrects the ones that were wrong, and
specifies the parts it left open -- which are exactly the parts that a first implementation
attempt got stuck on.

Section 2 is the OPERATIONAL DEFINITION of a successful run, and it is the part to read if you
read only one. The proposal has no such section, and every parameter argument in the sections
below is settled by section 2.1's rule about which constants may be fixed at all.

---

# 1. What is measured

Everything in this section is a number from `notes/`, on real pencils unless stated. The
design rests on these rather than on the local model being assumed.

**The cone model holds.** `sigma_1 ~ sqrt(eps^2 + c^2*(x - x*)^2)` fits the far field to a
median relative residual of 1-3% on rect, disk, eq_tri and L-shape, failing only where the
floor is close to the plateau, i.e. where the basis cannot represent the eigenfunction.

The two parameters are independent and it matters which is which. **`eps` is the FLOOR** --
`sigma_1` at the eigenvalue, dimensionless, the residual of the best eigenfunction the finite
basis can represent, and therefore a property of THE BASIS. **`c` is the WALL SLOPE** --
`d(sigma_1)/dk` away from the eigenvalue, units of length, a property of THE DOMAIN. Measured
on the L-shape at 20, 40 and 60 Fourier-Bessel terms, `eps` falls 4.0e-05 -> 1.6e-08 ->
1.1e-12 while `c` moves only 1.602 -> 1.595.

**`eps/c` is the resolution limit**: the scale in `k` where the cone flattens into its
quadratic bottom, so two eigenvalues closer than that have overlapping bottoms and cannot be
separated. Since `c` is fixed by the geometry, `eps` alone sets it -- which is the concrete
form of "the search is bounded by the basis". See `notes/cone_model.py` for the figure.

**`c` is stable within a domain, not across domains.** 0.98-1.17 on rect over k in [3.3, 7.3];
1.56-1.70 on the disk; 0.45-0.49 on eq_tri. Per-solve calibration is necessary and sufficient.

**The spacing is uniform in `lam`, not `k`.** `N(lam)` is linear in `lam`, so `lam_n` is linear
in `n` and `k_n ~ sqrt(n)`: the gap is `4*pi/area` in `lam` and shrinks like `1/k` in `k`. The
proposal's section 2.2 says the opposite; see the As-built section.

**The floor varies by FIVE ORDERS within a single scan, structurally.** On the L-shape with a
corner-enriched basis, `eps` is 3.1e-13 at k = 4.443 and 5.5e-08 at k = 5.649. That is not
noise: `4.443**2 = 19.739 = 2*pi**2`, a mode whose eigenfunction VANISHES at the reentrant
corner and is therefore analytic there, so the basis represents it almost exactly while its
neighbours carry the `r**(2/3)` singularity. On a domain with a singular corner, which modes
happen to vanish at it decides how well they are resolved.

So no absolute threshold can serve as a multiplicity test, and `tau_accept` is a far more
dangerous parameter than a single scan of a smooth domain suggests. (An earlier version of
this section put the spread at fifty-fold, measured on a rectangle; the five-order figure is
the one to design against.)

**sigma_2's minimum sits at the crossing point.** On rect(1+delta, 1) at every separation
tested, sigma_2 has exactly one minimum, at the midpoint to within 1% of `d/2`. So the offset
from it to the nearest sigma_1 minimum measures the separation.

**Multiplicity reads off the spectrum cleanly.** Nine for nine on the disk and equilateral
triangle, covering both geometric and arithmetic degeneracies. Within a multiplet the ratio
`sigma_{j+1}/sigma_j` is 1.0-1.7; across the gap it is 1e11-1e15. The gap parameter `G` is
not delicate: anything from about 3 to 1e10 works.

**(C2) is load-bearing.** Once `d <~ eps/c`, the gap rule alone reports multiplicity 2 at a
SIMPLE eigenvalue and double-counts. Measured with a ten-direction basis: at `d/(eps/c) = 0.45`
the ratio `sigma_2/sigma_1` is 1.07 and `G = 10` reads multiplicity 2. The local-minimum
condition rejects it every time, because sigma_2 there is strictly decreasing.

**sigma_1 is smooth when the rank is safe.** Roughness after a cubic detrend is 1e-10 relative
or better while `s_r/s_0` stays clear of machine epsilon, rising to 1e-2 when it does not.
Pairs are resolvable down to `d = 1.3e-06`; only an exact degeneracy collapses to one dip.

**Refining below the flat bottom samples noise.** The flat half-width of a well is
`eps*sqrt(2*eps_mach)/c`. Below that the curve is roughness, not signal.

**Prominence separates a well from roughness by ten orders.** Over thirteen bases with `eps`
from 7.5e-03 to 7.0e-15, a true well measured across its whole width has prominence
`plateau/eps`, worst case 72; roughness minima in the flat bottom have prominence
`1 + 2.4e-10` or less, and `P - 1` at that level IS the measured `eta`. A good basis has no
flat bottom to measure at all -- 40 plane waves on a rectangle give a flat half-width of
1.8e-21, twenty orders below float spacing -- so the roughness regime is reachable only by
starving the basis deliberately.

**The interior Gram angle separates a duplicate from a distinct pair.** Two roots reported
close together give `|cos|` of at most 0.10 when they are distinct eigenfunctions -- a floor set
by `1/sqrt(n_int)`, since the discrete interior inner product is a Monte-Carlo estimate -- and
at least 0.948 when they are the same well found twice, even at two-thirds of an eigenvalue
gap. `1 - cos = 7.8e-04 * d**2` inside a well, so the two arms would only meet at about three
gaps' separation.

Within ONE `lam` the test is vacuous and must not be used: the GSVD gives `A_I x_j = s_j * v_j`
with `V` orthogonal, so the interior images of distinct generalized singular vectors are
orthogonal BY CONSTRUCTION, measured at 4.6e-16. It is informative only ACROSS roots, where the
vectors come from different pencils.

**The discretization splits an exact degeneracy, and the splitting does not scale with `eps`.**
Measured `|argmin sigma_1 - argmin sigma_2|` at protected degeneracies: constant to 1% while
`eps` moves four orders across a plane-wave ladder and six across a Fourier-Bessel one. The
stable normalisation is RELATIVE -- `1.8e-09 * lam` for plane waves, `3.9e-09 * lam` for
Fourier-Bessel -- so it is a property of the basis family and the boundary discretization, not
of the tension floor. Any tolerance written as a multiple of `eps/c` is therefore the wrong
shape; see section 19 of the notes for the contract that was withdrawn over this.

**The interior points are a normalization, not a quadrature.** Their job is to keep
`||A_I c||` away from zero, and a STRUCTURED fill fails at exactly that: `method="grid"` places
x at `k/(side+1)`, so the mode `side+1` sits on a nodal line at every point and is annihilated
to 1e-56. Random points measure 0.70 to 0.91 on the same scale at every count and seed. The
fill should stay pseudorandom, with a FIXED default seed for reproducibility -- no seed is bad,
whereas a grid is catastrophic at one predictable mode.

---

# 2. What success means

The layer above sets `prec`. The contract is that a user gets eigenvalues to relative accuracy
`prec`, or an honest statement that MPS could not reach it here. Doing better than asked is
always allowed; doing materially more work than `prec` requires is not.

## 2.1 The rule that decides which constants may be fixed

A constant may be **fixed** only if it is DIMENSIONLESS and separates two populations whose
separation does not shrink as `prec` tightens. Anything carrying units of `lam`, and anything
expressing HOW ACCURATE, must derive from `prec`.

That rule is what removed the two loose constants an earlier draft of this section had. It is
stated first because it, rather than any individual number, is the thing to apply when a new
parameter appears.

## 2.2 Assumptions

The pencil -- basis, collocation points, regularization AND rank policy -- is given and
adequate. The solver may DETECT inadequacy; it is not responsible for repairing it. Rank policy
belongs in that list because the truth depends on it: a rank frozen at the wrong `lam` gives
85 to 128 spurious minima in a window holding one eigenvalue.

Adequacy is quantitative:

    gap  >  20 * eps/c        on [a, b]

measured directly in `notes/landscape_findings.md` section 9.4 -- (C1) is correct at
`d/(eps/c) = 23` and wrong at 0.45 and 0.15. This is the basis planner's contract, and it gets
EASIER as `prec` tightens, because `eps` falls.

## 2.3 Ground truth, `S(prec)`

`sigma_1` has local minima `lam_1 .. lam_n` with multiplicity `mu_i`, the largest `mu` such
that `sigma_j` has a local minimum at `lam_i` for every `j <= mu`. `S` is the ordered multiset
with each `lam_i` repeated `mu_i` times, and each member corresponds to a linearly independent
approximate eigenfunction.

Neither predicate survives floating point unmodified, so each gets a scale:

* **"has a local minimum"** -- prominence at least `P = 1 + m*eta`, with `eta` measured per
  solve and a floor of `P >= 2`.
* **"at `lam_i`"** -- two branch minima are the same level when they lie within
  `tol_merge = prec`, relative. There is no separate coalescence constant; an earlier draft had
  one and section 19 of the notes shows the form it assumed does not hold.

**`S` is indexed by `prec`, deliberately.** Two levels 4e-09 apart are ONE level at
`prec = 1e-08` and TWO at `prec = 1e-12`. That matches `chladni.reference`, which returns an
unfolded multiset and merges nothing, because collapsing is a comparison-time choice rather
than a property of the spectrum.

`S` is the Laplacian spectrum restricted to modes separated by more than about `20*eps/c`.
Hitting `S` is the SEARCH's job; making `S` the whole spectrum is the PLANNER's.

## 2.4 The three clauses

A solver is **successful at `prec`** on `[a, b]` when all three hold.

**(1) Completeness and multiplicity.** `S_hat` matches `S(prec)` as a multiset. Among roots
reported close together, collapse-versus-merge is decided by the INTERIOR GRAM ANGLE, not by
distance:

    u_i = I(lam_hat_i) @ coef(lam_hat_i, 1)          # equilibrated; no unscaling

    |<u_1, u_2>| >= tau_gram   ->  the same eigenfunction: collapse and RECOUNT
    |<u_1, u_2>|  < tau_gram   ->  independent: merge and SUM

Distance survives only as a prefilter deciding which pairs are worth testing. The angle is what
distance cannot do: one simple eigenvalue found twice and two genuine simple eigenvalues can
sit at the SAME separation, and only the eigenfunctions tell them apart.

**(2) Accuracy.** Each `|lam_hat_i - lam_i| / lam_i <= prec`, OR the solver reports that
`delta_res(lam_i) > prec * lam_i` and returns `delta_res` as the attained bound. There is no
predicted-accuracy constant: `delta_res` is MEASURED from the parabola fit Phase B already
performs, and compared against `prec`.

**(3) Honesty.** The reported per-eigenvalue bar contains the true error, and the completeness
diagnostics fire whenever (1) fails.

Clause (3) is not decoration. `S` is unknown on every domain that matters -- `L_shape`, `GWW1`
and `mushroom` have tabulated prefixes only -- so (1) and (2) need a truth-free proxy on a real
run, which is section 6's diagnostics. And the recursion-floor bug was found by the WEYL AUDIT
rather than by prefix truth, on a case the surrogate does not reproduce at all.

## 2.4a The same contract, per stage

Section 2.4 states the contract for a whole run. This section states it for each stage of
`candidates.md` section 1's decomposition, because a stage and a pipeline fail differently and
a clause that only exists at the pipeline level cannot be tested on the stage that violates it.
Every stage postcondition below is derivable from 2.4; nothing new is asserted. What is new is
WHERE each clause is owned.

`prec` plays a different role in each stage, and conflating the three is the recurring error:

| stage | product | role of `prec` |
|---|---|---|
| bracket | intervals + a conditional completeness claim | a WIDTH -- when to stop splitting |
| refine | a location + an attained bar | an ACCURACY -- when to stop stepping |
| multiplicity | a grouping + an integer per group | a GROUPING TOLERANCE -- what counts as one level |

The same number, three jobs. They are consistent because all three come from `S(prec)` in
section 2.3, and NOT because any one of them implies the others: a bracket set can satisfy the
width rule on an interval where refinement cannot attain `prec*lam`, and both can succeed while
the grouping is arbitrary -- which is section 2.7's inversion.

Each stage is stated in four parts: what it PRODUCES, what it GUARANTEES given section 2.2's
assumption, what it SAYS INSTEAD when it cannot, and what it HANDS FORWARD.

### Stage 1 -- bracket

**Produces.** A finite set of disjoint intervals, each COMMITTED to rather than a subdivision of
`[a, b]` (`candidates.md` section 1), together with the evaluation cache and the slope and
plateau model built along the way.

**Guarantees.**

* *Coverage, conditional on `L`.* Every `lam*` in `S(prec)` on `[a, b]` lies in some bracket.
  Conditional is not a hedge: `L` is a largest observed secant inflated by `safety`, a LOWER
  bound with no formal status, so the claim is about the sampling and is reported with the
  margin that justifies it (section 6).
* *Purity at `prec`.* The eigenvalues inside one bracket are ONE LEVEL of `S(prec)`, i.e. they
  span at most `prec*lam`. This is the right phrasing rather than "at most one eigenvalue": a
  multiplet spans zero and is legal, and a cluster tighter than `prec*lam` is one level BY
  DEFINITION at this `prec`.

**Not guaranteed, and it cost a seam defect to learn.** A bracket guarantees CONTAINMENT, not
that it holds a sampled descent -- stage 1 stops on purity by width, so the deepest sample may
sit on an endpoint or nowhere near the well. The refiner must be able to start anywhere inside.
Nor is `#brackets == #levels`: over-bracketing is legal and is stage 3's problem;
under-bracketing is a coverage failure.

**Says instead.** `Declined`, carrying the brackets found so far (`candidates.md` section 5).
Per bracket it also records whether the `sigma_2` gate certified isolation, or whether the
interval is a live region handed on unresolved.

**Hands forward.** The cache -- stage 3 cannot work without it and spends nothing of its own --
plus `L` and its growth history, the plateau, and the measured roughness `eta`.

### Stage 2 -- refine

**Produces.** Per bracket, one of THREE outcomes, and the third is the one that leaks when it is
missing: a location, a rejection, or an admission that the bracket holds structure it cannot
separate.

**Guarantees.** For a located root, `|lam_hat - lam*| / lam* <= max(prec, delta/lam_hat)`, with
`delta = delta_res = DELTA_SAFETY * eps/c`, both quantities read off the parabola fit the
refiner already performed: `eps` is `sigma_1` where it stopped and `c` is the wall slope from
the fitted curvature. That is the RESOLUTION LIMIT of section 1 -- a minimum cannot be located
finer than the region over which the curve is flat -- and it is measured against truth at a
ratio of 0.99 to 1.27 over 12 levels, three domains and two basis families, which is why it
carries a margin rather than being reported bare.

It is a bound rather than a fit because it is SELF-CALIBRATING: `eps` is `sigma_1` where the
refiner stopped, not the floor of the well, so on the wall -- where `sigma ~ c*delta` --
`eps/c` IS the remaining distance, read off the value already in hand. Inside the flat bottom
`sigma` saturates and the estimate becomes conservative, which is the direction a bound may err
in. Which stop rule bound is reported. A rejection means the bracket holds no interior
minimum to the check of section 5 Phase B; smallness alone is not enough, and before that check
existed it passed 8 phantom roots against 20 real ones.

**Says instead.** A rejection is an OUTPUT, not a failure -- stage 2 is the only stage that can
falsify a bracket. So is "this bracket holds structure I cannot separate", which is a stage 1
PURITY failure that only stage 2 can see; flattening it into a plausible midpoint is the shape
of the error.

**Hands forward.** The by-products of the fit, which cost nothing extra and which stage 3 and
the basis planner both need: `delta_res`, the local wall slope `c`, and the floor `eps` at the
root.

**Not its job.** Splitting a bracket that holds two levels, or deciding multiplicity. Detecting
both is.

### Stage 3 -- multiplicity

Multiplicity is undefined without a grouping, so the stage owns both. "How many branches dip
here" is only half the count; "are these two reported roots the same level" is the other half,
and section 5's Phase D is where it is decided -- distance as a PREFILTER only, the interior
Gram angle as the decision, collapse-and-RECOUNT rather than sum.

**Produces.** A partition of the reported roots into levels, and an integer `mu` per level.

**Guarantees**, in two nested clauses, weaker than "the multiplicity of every level" and true:

1. *Total, on a window.* Over any sub-window whose endpoints are clear of the nearest level by
   more than that level's spread, `sum(mu)` is the true count in `S(prec)`. This holds EVEN WHEN
   the grouping inside the window is arbitrary.
2. *Per level.* `mu_i` is correct when that level is clear of its neighbours by more than its
   own spread. Outside that condition, clause 1 is all there is.

The slogan, because it is what makes the weaker contract useful: **the count is a property of a
WINDOW, the grouping is a property of a SEPARATION.** A caller whose grouping is not guaranteed
knows exactly what to do -- widen the window and take the total.

Both clauses are conditional on section 2.2's `gap > 20*eps/c`. Below it (C1) alone
double-counts a simple eigenvalue and (C2) is what rejects it.

**Says instead.** Two admissions that must not be conflated. COLLAPSED: the grouping was
arbitrary at this `prec`, here is the total over a level coarser than was asked for. DECLINED:
the basis violates 2.2 and `mu` is not readable here at all. Section 2.7's ceiling is a third
report of its own -- below `prec ~ 4e-09` the discretization's own splitting of a degenerate
pair exceeds `tol_merge`, so a true double is reported as two simples. That is a real limit on
attainable `prec` at degenerate levels and it is stated, never silently hit.

**Costs nothing.** Every point read was paid for by stage 1. The one exception is the Gram
angle, one GSVD in `separate` mode per reported root, at the end.

## 2.5 The constants

Fixed, because dimensionless and because the margin GROWS as `prec` tightens:

    P            1 + m*eta, floor 2   roughness <= 1 + 2.4e-10, true wells >= 72
    G            10                   within a multiplet 1.0-1.7, across the gap 1e11-1e15
    tau_gram     0.5                  distinct <= 0.10, duplicate >= 0.948
    contract     20 * eps/c           correct at 23, wrong at 0.45 and 0.15
    tau_scan     a fraction of the plateau S, NEVER an absolute tension
    weyl tol     0.5                  a correct set reaches +0.32, one missing reads +1.16

Each of those improves with `prec`: a smaller `eps` deepens the wells (P), raises the
across-gap ratio, which is `O(1)/eps` (G), and loosens the contract; a larger `n_int` lowers
the `1/sqrt(n_int)` floor that sets `tau_gram`'s margin. `tau_scan` is relative to the plateau
because `eps` varies FIVE ORDERS within a single scan, so no absolute tension threshold can
serve.

Derived from `prec`, because they carry units of `lam` or express accuracy:

    tol_merge          prec
    accuracy target    prec * lam
    refinement stop    max(prec * lam, delta_res)
    planner target     the `eps` that makes the MEASURED delta_res <= prec * lam

The planner target now has a closed form, because `delta_res` does: `delta_res <= prec*lam`
means `eps <= prec*lam*c/DELTA_SAFETY`, so a planner is asked for a floor rather than for a
number nobody could compute. An earlier version of this line said no closed form was
available, on the strength of `sqrt(2*eps*eta)/c` failing to predict observed error -- that
form was wrong, not the idea of a formula. See the As-built note.

## 2.6 Not more work than asked

* Refinement stops at `max(prec * lam, delta_res)`. At loose `prec` the first term binds and
  Brent stops early even though the floor would permit more.
* The scan stride comes from Weyl -- about six samples per eigenvalue gap, since dips occupy
  roughly half the spacing on every domain measured. **Completeness is not a precision
  question** and the stride does not tighten with `prec`.
* Phase C spends no new evaluations.
* The Gram check costs one GSVD in `separate` mode per REPORTED ROOT, at the end. The scan
  itself only ever calls `gsvdvals`.

## 2.7 The limit to watch

`tol_merge = prec` against pairs measured resolvable to about `3e-06` relative leaves a factor
of 30 at `prec = 1e-08`. Below `prec ~ 4e-09` it INVERTS: the discretization's own splitting of
a degenerate pair -- `1.8e-09 * lam` for plane waves and `3.9e-09 * lam` for Fourier-Bessel,
and independent of basis size over six orders of `eps` -- exceeds `tol_merge`, so a true double
is reported as two simple eigenvalues. That is a real ceiling on attainable `prec` at degenerate
levels. It is measurable per solve, and clause (3) is what makes it surface instead of silently
corrupting the count.

# 3. What the proposal got right, and what it got wrong

## Right

**Squaring removes the kink.** Minimise `sigma_1^2`, never `sigma_1`. Confirmed: refinement is
machine-exact on disk, rect and eq_tri at every regularization tolerance tried.

**Section 3.4's offset RELATION.** The offset between sigma_2's minimum and the nearest
sigma_1 minimum does equal `d/2`, confirmed on real pencils to about 1%. What that buys is
less than section 3.4 claims -- see below.

**Section 5's (C2).** Confirmed, and shown to be necessary rather than defensive.

**Section 3.2's own self-criticism.** The honest comparison is a well-chosen uniform grid, and
the gain over one is a constant. The harness scores that grid as a control for exactly this
reason.

## Wrong

**Section 2.1's `eta`.** Taken as `u*||.||`, machine epsilon. Measured roughness is 1e-10
relative at a safe rank, four orders above that, and 1e-2 at an unsafe one. Every quantity
derived from `eta` -- `delta_res` above all -- must be recomputed against the measured value,
and `delta_res` should be estimated per eigenvalue from the parabola fit rather than assumed.

**And `delta_res` in particular was wrong for exactly that reason, for as long as this
paragraph stood above it unapplied.** `sqrt(eps*eps_mach/curvature)` assumes the tension
carries absolute noise of order machine epsilon; it does not, and the correct scale is the
resolution limit `eps/c` that section 1 already names. The two differ by
`sqrt(eps_mach/eps)`, so the shortfall GROWS as the floor rises -- measured against
closed-form truth, the old form under-reported by 5x at `eps ~ 1e-14` and 2100x at
`eps ~ 1e-9`, and contained the true error at ZERO of six disk levels where the corrected
form contains all six. It is fixed in `chladni.mps.refine`; this entry stays because the
gap between a recorded correction and an applied one is the useful part.

**Section 3.4's claim that a cluster can be "measured before it is resolved."** It cannot.
Measured: the offset becomes nonzero at exactly the same sampling coarseness at which
sigma_1 first shows two discrete minima -- both fire at `h/d = 0.35` and neither at 0.5. The
offset is a way to MEASURE `d` once the pair is resolved, not a way to detect it earlier. An
earlier version of this document built its entire section 3 on the opposite, and that section
has been withdrawn.

**Section 8's two-level basis.** Not wrong, but it is the wrong knob. Rank, not basis size, is
the cheap lever: one SVD serves both a low-rank smooth scan and a high-rank accurate
refinement, with no second basis and no second assembly. It is also out of scope here, because
it needs the pencil and this layer only has the tension function.

---

# 4. Why adaptive subdivision -- the argument that survives

An earlier version of this section argued that the design's advantage was section 3.4's
offset detector, on the grounds that lappy's sigma_2 gate tests a VALUE that a coarse grid
gets wrong. Half of that is right and the conclusion does not follow, so both halves are
recorded here.

**Right: lappy's gate measures the grid.** Its cluster rescue fires when the sampled sigma_2
falls below `ttol`. But sigma_2's minimum value is `c*d/2` only at the crossing point, which
is essentially never a sample, so a grid of spacing `h` sees about `c*h/2` -- measured 5.03e-03
at `h = 1e-02` and 5.05e-04 at `h = 1e-03`, for a pair whose true floor is 5.00e-06. The gate
therefore engages when `h <= 2*ttol/c` whatever the separation.

**Wrong: the offset does not fix that.** It fires at the same coarseness as simply looking for
two minima in sigma_1. There is no detector here that sees a pair before sigma_1 does.

**What actually distinguishes the design is the COST of getting to that resolution.** A pair
at separation `d` is resolved once the sampling near it is finer than about `d/3`. A uniform
grid must be that fine everywhere, at `n ~ 3*(b-a)/d`. Adaptive subdivision only refines where
`sigma_1` is small, at `O(log(1/d))`. Measured, resolving a pair in an interval of width 2:

    d        uniform grid n     branch and bound     ratio
    3e-03          2048                47              43x
    1e-04        ~60 000                57            1053x
    1e-06         ~6e+06                71          84 507x
    1e-10         ~6e+10                97           6e+08

That is not section 3.2's "modest constant" -- the proposal understates its own advantage. And
it matters because `d` is not known in advance: a black-box solver cannot tune its grid to the
tightest pair it will meet, so the uniform-grid column is a cost it would have to pay
everywhere, forever, against a pair it may or may not have.

**But adaptive subdivision is not unique to Piyavskii-Shubert.** lappy already refines
recursively in `bracket_mins`; its recursion is simply triggered by the broken gate above. The
open question this spec does not settle is how much the envelope, the attic and the `L`
calibration buy OVER a repaired recursive bisection. That comparison is cheap and should be
run before the full machinery is built.

# 5. The algorithm

Four phases. Where the proposal says "use this to decide how far to subdivide" without saying
how, this section says how; those gaps are where the first implementation attempt failed.

## Phase A -- scan by Lipschitz branch and bound

Work in `k`, not `lam`: eigen-wavenumbers are asymptotically equispaced and `c` is far more
nearly constant in `k`, so a single `L` is meaningful over a wide range.

**Pilot.** Sample `n_pilot` points uniformly. Set `S` to the median `sigma_1` (the plateau)
and `L = gamma * max|ds/dk|` over the pilot secants, `gamma ~ 2`.

**Envelope.** For an interval `[x0, x1]` of width `h` with endpoint values `v0, v1`, no
interior point can fall below

    s_hat = (v0 + v1 - L*h) / 2      attained at   x_hat = (x0 + x1)/2 + (v0 - v1)/(2L)

Keep intervals on a heap keyed by `s_hat`, smallest first.

**Prune, but reversibly.** If `s_hat > tau_scan` the interval is certified free; move it to an
ATTIC tagged with the `s_hat` that justified pruning, never discard it. Whenever `L` grows,
re-test the attic and revive anything no longer certified. Pruning is the only irreversible
act in the algorithm and the attic is what makes it reversible.

**Subdivision depth -- specified.** An interval becomes a candidate when

    h < max( beta * d_est , h_floor )      and      h < alpha * S / L

where `d_est` is the separation estimate of Phase A.1 (infinite when no pair is signalled, so
that the second condition alone governs an isolated root) and `h_floor` is the resolution
floor of Phase B. Both bounds are needed, and the first is the one the earlier attempt was
missing: `alpha*S/L` measured 3.75e-02 on cases whose pairs were 4e-03 apart, so the interval
was declared isolated before the pair could separate, and one root was returned for two.

### A.1 The separation estimate

Within a candidate interval, sample `n_probe` points and take
`c(x) = #{ j : sigma_j(x) <= tau_scan }` at the best of them.

* `c < 2` -- one branch is down. No pair; `d_est = inf`.
* `c >= 2` -- either a multiplet or a cluster. Estimate
  `d_est = 2 * |argmin sigma_2 - argmin sigma_1|` over the probe points.
  `d_est = 0` means the two branches are coincident, i.e. a genuine degeneracy, and the
  subdivision must NOT chase it -- that is what `h_floor` is for.

## Phase B -- refine

**Refine from the samples, not from the brackets.** The envelope's minimiser for a cone is its
APEX, so the scan lands exactly on a root and then splits there, leaving it on the shared
endpoint of two intervals and in the interior of neither. Refining those separately stops on
the wall -- measured at 7.9e-08 against a 1e-10 floor. Three attempts at repairing this after
the fact each fixed some cases and broke others.

So: sort the evaluation cache, take discrete local minima OF THE CACHE, and bracket each with
its two neighbouring samples. This uses every evaluation the scan paid for and needs no
bracket bookkeeping. Ties are broken to the left, because a well sampled symmetrically has two
exactly equal flanking values and a strict comparison on both sides then finds nothing.

Minimise `sigma_1^2` on each bracket by Brent. Reject a result unless it is a genuine interior
minimum -- `sigma_1(x +/- h_check) > sigma_1(x)`. Smallness alone is not enough: bounded
minimisation returns an endpoint when the true minimum lies outside the bracket, and next to a
deep root that endpoint still passes any threshold test. Measured before this check existed: 8
phantom roots against 20 real ones, each 2e-05 to 4e-04 away from a genuine root with
`sigma_1` between 1.6e-04 and 9.8e-04.

**Termination.** Stop at `max(prec * x, delta_res)` with `delta_res` from the current
parabola fit, or on stagnation -- the last three values agreeing to within `eta`. `prec * x` is
the term that stops a loose request from paying for a tight one, and it binds first whenever
the basis is better than the request. The flat half-width `eps*sqrt(2*eps_mach)/c` is the
scale below which the curve is roughness rather than signal; `h_floor` in Phase A is set from
the same quantity. Report the attained precision and which bound was active.

**Retain a wide bracket.** Keep the pair of evaluated points either side of the root at
distance of order `eps/c`, from before the bracket collapsed. Phase C needs points where the
higher branches have actually moved; the final bracket is far too tight for that, and using it
is why an earlier attempt's multiplicity test was reading noise.

## Phase C -- multiplicity

From the cache; no new evaluations. Index `j` is active when both hold:

* **(C1)** the largest ratio gap in the low spectrum falls after `j`, i.e.
  `sigma_{j+1}/sigma_j >= G`, `G = 10`. A gap test, not an absolute threshold, because the
  floor varies FIVE ORDERS within one scan.
* **(C2)** `sigma_j` is a local minimum at the root, to prominence `P`, against the WIDE
  bracket of Phase B.

The multiplicity is the number of active indices, necessarily a contiguous block `1..kappa`.

Section 2.3 defines multiplicity by (C2) ALONE, and that is deliberate: (C2) is the
DEFINITION, (C1) is the TEST. Exact local-minimality is not decidable from samples, so the
implementation needs the ratio gap as well; the two must not be confused when the definition
is being reasoned about. (C1) alone double-counts once `d <~ eps/c`, which is what the
`20*eps/c` contract of section 2.2 keeps the solver out of.

## Phase D -- collapse, then merge

Bottom-up, and the order matters.

**Distance is a PREFILTER, not the decision.** Roots whose diameter exceeds `tol_merge * x`
with `tol_merge = prec` are separate levels and are not tested further. Everything closer than
that goes to the Gram angle, because at those separations distance cannot discriminate: one
simple eigenvalue found twice and two genuine simple eigenvalues occur at the SAME diameter.

    |<u_1, u_2>| >= tau_gram   the same eigenfunction. Collapse to the smallest `sigma_1` and
                               RECOUNT the multiplicity there. Do NOT sum -- summation is the
                               double-count.
    |<u_1, u_2>|  < tau_gram   independent. SUM the multiplicities, place at the
                               `sigma_1`-weighted centroid, report the diameter as the
                               uncertainty.

The ambiguous case is benign. Inside a genuinely degenerate eigenspace which vector the GSVD
returns is arbitrary, decided by roundoff, so a true double found twice may come back either
nearly parallel or nearly orthogonal. Both verdicts are correct, because collapse RECOUNTS
rather than discarding: parallel gives 2 by the recount, orthogonal gives `1 + 1 = 2` by the
sum.

An earlier version of this section used `delta_res` for the first rule and a fixed
`tol_merge` for the second. Both are wrong. `delta_res` is the wrong scale -- the quantity
that actually has to be merged is the discretization's splitting of a degeneracy, which does
not scale with `eps` at all -- and a fixed `tol_merge` breaks the precision contract of
section 2.1 by fixing a value that expresses resolution.

---

# 6. Diagnostics, and the shape of the claim

This section is the machinery behind section 2.4's clause (3). Everything here is REQUIRED
output, not optional instrumentation, because `S` is unknown on every domain that matters and
these are the only signals that clause (1) has failed.

The completeness claim is conditional and must be reported as such:

> No `x` in `[a, b]` with `sigma_1(x) <= tau_scan` exists outside the returned set, GIVEN the
> Lipschitz constant `L`.

Every run reports `L`, the largest observed secant slope, the ratio between them, the number
of attic revivals, and the number of evaluations. A large margin means the scan was
conservative; a small one means the constant was being pushed and the run deserves a rerun
with a larger `gamma`.

This matters because the failure is silent: a dip that is never sampled is invisible to any
finite-evaluation method, and the branch and bound in particular returns nothing after its
pilot with no error at all.

But the case that demonstrates it is synthetic and should not drive the design. A well with
`c = 50` defeats every search tried; no domain produces one. Measured `c` at 19 real
eigenvalues on four domains runs 0.445 to 1.725, and more to the point the DIP WIDTH RELATIVE
TO THE EIGENVALUE SPACING is nearly invariant -- 0.46, 0.47, 0.50, 0.52 on disk, rect,
L-shape and eq_tri, across diameters from 1.0 to 2.83. Dips occupy about half the space
between eigenvalues, so roughly six samples per gap always place three inside every dip, and
Weyl's law sizes that directly.

The real failure mode is the mirror image: a CLUSTER, where the gap shrinks while the width
tracks it. Design against that. The diagnostics still earn their place -- they are the only
signal a search has that its first pass was too coarse -- but the scenario motivating them is
a cluster too tight to resolve, not a dip too narrow to see.

The Weyl cross-check belongs to the harness rather than to any search, because it audits all
of them.

---

# 7. Tests

Written before the implementation, against the surrogate and against real pencils.

| file | what it defends |
|---|---|
| `tests/test_search_scan.py` | Phase A: the envelope never prunes an interval containing a root when `L` is a true bound; the attic revives on a raised `L`; the subdivision depth reaches `beta*d_est` on a pair and stops at `h_floor` on a degeneracy |
| `tests/test_search_refine.py` | Phase B: the apex case (root on a shared endpoint) is recovered to the floor; a wall point adjacent to a deep root is rejected; symmetric sampling still finds the well; stagnation terminates |
| `tests/test_search_multiplicity.py` | Phase C: 9 for 9 on disk and eq_tri; (C2) rejects the `d <~ eps/c` double-count; a roughness minimum fails prominence `P` and a true well passes it by orders; the wide bracket is wide enough |
| `tests/test_search_collapse.py` | Phase D: the Gram angle reads `<= 0.10` on a resolved pair of `rect(1+delta, 1)` and `>= 0.948` on one well sampled twice; within one `lam` it is 0 to machine precision and must not be consulted; a true double found twice gives multiplicity 2 under BOTH verdicts |
| `tests/test_search_precision.py` | section 2's contract: the same domain solved at `prec` = 1e-6, 1e-8, 1e-10 returns levels agreeing to the looser `prec` and costs monotonically more; refinement stops on `prec * lam` when the basis is better than the request; `delta_res > prec * lam` is REPORTED rather than silently missed |
| `tests/test_search_score.py` | end to end through the harness, against the uniform-grid control and the lappy baseline, on the surrogate and on real domains |

The harness scores recall first, counted with multiplicity, and a root already claimed cannot
be claimed twice -- so a cluster reported from two brackets scores as one match and one
spurious rather than as two matches.

`test_search_precision.py` is the one that defends the contract rather than the algorithm, and
it is the one to write first. Every other row can pass while the solver quietly ignores `prec`,
either by always working to machine precision or by always working to 1e-8.

---

# 8. Deferred

The two-level rank of section 3, which needs the pencil. Automatic `gamma` escalation on a
small margin. Parallel evaluation, which changes the objective from evaluation count to
wall-clock rounds and may favour a graded uniform grid instead -- the branch and bound's
advantage is adaptivity, and adaptivity is worth less when points are nearly free.

---

# Superseded in part: read `CHLADNI_V1.md` and `candidates.md` alongside this

Two later records overlap this document and neither existed when it was written. The split:

**`docs/design/CHLADNI_V1.md` supersedes section 5's Phase A and the metric.** Its bracketing
design -- seed from Weyl and double until the slope settles, cover by Piyavskii exclusion, gate
isolation on `sigma_2`, locate by predict-then-straddle -- replaces the Lipschitz branch and
bound with the attic, and its coverage/purity scoring on BRACKETS replaces the
recall-with-multiplicity metric that section 27.2 of the notes shows is blind and inverted in
exactly the regime it was built to probe. `CHLADNI_V1.md` has its own As-built section
recording what did not transfer into `chladni`.

**`docs/design/candidates.md` owns the seams.** Sections 1-3 there define the four stages, the
variable convention, and how a method that does not decompose is compared fairly -- questions
this document raises implicitly by comparing three searches and never settles.

**What still lives only here**, and is not superseded: section 2's `prec` contract and its rule
about which constants may be fixed at all; `S(prec)` as ground truth; Phase C's multiplicity
conditions (C1) and (C2); Phase D's collapse-versus-merge on the interior Gram angle; the Weyl
audit's placement in the harness; and the measurement that refinement belongs in `lam` while
the scan belongs in `k`.

# As built: deltas from this spec

What exists at HEAD, what it scores, and what is still wrong. As in `geometry.md` and
`particular_solutions.md`, the body above records what was intended and this section records
what the implementation found; the gap between them is the useful part.

## What is built

The older generation, which this document specified:

| | |
|---|---|
| `tests/harness.py` | the scorer, the Weyl audit, `unfold`, `levels_agree`, and the uniform-grid control |
| `tests/searches.py` | `lappy_search`, `lappy_repaired_search`, `lipschitz_search` |
| `tests/surrogate.py` | a 2 us stand-in for the tension curve, fitted and validated against real pencils |

And the current instrument, built against `candidates.md` and `CHLADNI_V1.md`:

| | |
|---|---|
| `tests/objective.py` | the shared seam: one cache, one counter, `Declined` as a value |
| `tests/bracket_score.py` | coverage / purity / spurious / fineness / measured cost, the `prec` ladder |
| `tests/bracket_controls.py`, `tests/baselines.py` | six adversaries, five oracles, three baselines |
| `tests/surrogate_cases.py`, `tests/rect_suite.py`, `tests/hard_roster.py` | the three tiers |
| `tests/v1_bracket.py`, `tests/v1_refine.py` | the winning candidate, stages 1 and 2 |
| `tests/test_gate.py` | the standing acceptance gate, `pytest -m gate` |

Nothing is in `src/chladni/` yet -- except the parts that are features rather than candidates:
`asymp.expected_count`, `mps/calibrate.py`, `MPSPencil.with_rtol` and
`MPSEigensolver.findability`. When a search wins it moves there, rewritten against a spec that
this section will have corrected. The v1 candidate has now won: it passes the gate at 26-27
evaluations per eigenvalue on rectangles and 22-24 on the disk, equilateral triangle and
L-shape, where the three searches below stop splitting at `d/lam ~ 1e-4`.

## The repaired baseline is the thing to beat

Two changes to `lappy_search`, and the win is larger than the branch and bound's:

* recurse into EVERY discrete minimum of `sigma_1`, rather than gating on `sigma_2`'s value;
* multiplicity by the gap plus local-minimum conditions, not a prefix against an absolute
  `ttol`.

Twelve surrogate cases: 10/12 perfect at 5573 evaluations against the original's 10/12 at
54443. On real rectangle pencils with exact truth it also LOCATES better -- 1e-15 to 3e-13
against 6e-09 -- because the deeper recursion hands Brent a much tighter bracket.

So the `O(log(1/d))` advantage of section 4 is a property of adaptive subdivision, which
lappy's architecture already had. Its trigger was the only thing broken.

## `lipschitz_search` does not work

Committed anyway, clearly labelled, because the spec above is worth more with a failed
attempt beside it than without. It scores 7/12. Five attempts at the candidate-to-root path
each fixed some cases and broke others:

    interior-minimum check       fixed 20-roots, broke 3-simple and triple
    union touching brackets      fixed those, broke every resolved pair
    union only on same-point     fixed pairs, broke rough
    refine from the samples      cleanest structure, still broke pairs
    section 3.4 offset depth     no change: the signal never fired

The last one is diagnosed and is section 5's `d_est` rule not being reached, not the rule
being wrong. The isolation width `alpha*S/L` measured 3.75e-02 against pairs 4e-03 apart, so
intervals were declared isolated before the pair could separate.

**Do not resume by patching.** The oscillation above is what a missing spec looks like; section
4 was written afterwards precisely so a rebuild has something to work against.

## The open bug, and how it was found

The repaired search **misses a different eigenvalue depending on its recursion floor.** On the
L-shape over `k` in [3, 5] -- that is `lam` in [9, 25], holding `lam` = 9.640, 15.197,
19.739:

    floor_frac 1e-06    loses lam = 15.197
    floor_frac 1e-09    loses lam = 9.640
    floor_frac 1e-12    finds all three, in 53 seconds

The basis is demonstrably adequate there: tension at all three is 1.5e-06 or better against
`tau_accept = 1e-3`. So this is the search, not the basis, and `floor_frac` is load-bearing in
a way nothing yet explains.

Every surrogate benchmark used 1e-12, which is why the repaired search looked solid. It was
the WEYL AUDIT that surfaced it -- reading 0.79 and flagging, independently of the prefix
truth that showed the same miss. That is the argument for the audit in one example: on a
domain with no closed form it is the only completeness signal there is.

## The open bug is explained, and it was never in the search

Rewritten from "load-bearing in a way nothing yet explains". It is an ACCEPTANCE failure
downstream of a correct search, which is why looking at the recursion never explained it.

Over `k` in `[3, 5]`, every one of the three levels is LOCATED at every setting -- to at worst
1.2e-08 -- and at the DEFAULT `floor_frac = 1e-3` all three are then thrown away:

    floor_frac   located, error in k          error/h              kept
    1e-03        6.4e-09, 5.9e-09, 3.5e-09    2.06, 1.51, 0.78     0/3
    1e-06        4.4e-15, 1.2e-08, 3.6e-15    0.00, 2.97, 0.00     2/3
    1e-09        3.1e-10, 1.4e-10, 1.3e-10    0.10, 0.04, 0.03     3/3
    1e-12        5.7e-14, 2.8e-13, 9.0e-14    0.00, 0.00, 0.00     3/3

What throws them away is the interiority probe, which reads `T(x +/- h)` at a fixed
`h = 1e-9 * k` and demands both exceed `T(x)`:

    h = max(1e-9 * max(abs(r.x), 1.0), 1e-12)
    if not (T(r.x - h)[0] > f and T(r.x + h)[0] > f): continue

`h` is a FIXED relative step, so the probe demonstrates interiority only when the located root is
nearer the true minimum than `h` is. Otherwise one step moves TOWARD the minimum and reads lower,
and the genuine root is refused. `floor_frac` enters only because it sets how tight a bracket the
polisher is handed, and therefore how accurate the root is -- and that the MIDDLE level is the one
lost at 1e-6 is the L-shape's five-order floor spread showing through Brent's attained accuracy,
which section 3.1 measured varying from 5.4e-10 to 5.6e-5.

Stated as a SEPARATION rather than a threshold. The kept and dropped populations split by a factor
of eight in `error/h` -- 0.000 to 0.098 against 0.782 to 2.974 -- but not at 1: the probe compares
tension values on flanks of unequal slope, and one row is dropped at 0.78. A first version of the
test asserted `error > h` exactly and was falsified by that row; what survives is the mechanism and
the separation, not a constant.

**`v1_refine._accept` is immune, and that is not a coincidence.** Strictly interior, and below the
tension at both SAMPLED endpoints, it keeps 3 of 3 at every `floor_frac` including the default
where the fixed-step probe keeps none. Its docstring already made this argument about a different
wrong version of the same probe: a probe whose reach is set by the bracket width cannot know what
it is stepping across. One whose reach is set by a CONSTANT cannot know that, and cannot know how
accurate the point it steps from is either. `tests/test_searches.py` is the tracked measurement;
`tests/searches.py` had no test file at all before it.

## Refine in `lam`, not `k`

Section 5 says Phase B minimises `sigma_1**2`, following the proposal's section 2.2 in working
in `k`. That is measured to be the wrong variable for the REFINEMENT, though 2.2's reasons --
Weyl spacing, and `c` being flatter -- are about the SCAN and are untouched.

Fitting a quadratic to `sigma_1**2` over identical physical sample points, the residual
relative to the data range is smaller in `lam` in **17 of 17 cases** across rect 1x1, 2x1 and
5x1, the disk, eq_tri and the L-shape: median 5.1x, range 2.5x to 15.4x, with rectangles among
the strongest. The located vertex is correspondingly 5.1x more accurate on median.

So the local model of the proposal's section 2 is stated in the wrong variable. Writing
`sigma_1 ~ sqrt(eps**2 + c**2 (k-k*)**2)` makes `sigma_1**2` exactly quadratic in `k`; the data
say it is closer to quadratic in `lam`, and fitting the model form directly leaves a residual
of 1.5e-02 against 8.7e-04 for a plain quadratic in `lam` on the same points.

Phase A and Phase B are separate choices and this changes only the second. `notes/k_vs_lam.py`
regenerates the figure. Not yet checked: whether the advantage survives at `delta_res`-scale
brackets, and whether it grows at large `lam` where `dlam/dk = 2k`.

**And one of 2.2's two reasons for `k` is backwards.** It says Weyl's law makes eigen-wavenumbers
"asymptotically equispaced with density |Omega|k/2pi" -- but a density that depends on `k` is
not equispacing. `N(lam)` is LINEAR in `lam`, so `lam_n ~ 4*pi*n/|Omega|` and `k_n ~ sqrt(n)`:
the spacing is uniform in `lam` and shrinks like `1/k` in `k`. Measured on rect 2x1 over
n = 20 to 390, `d(lam)` runs 7.09, 6.83, 6.83, 6.33 against the predicted `4*pi/area = 6.28`,
while `d(k)` falls 0.208 to 0.064 and only `d(k)*k` is flat.

That leaves "`c` is flatter in `k`" as the only surviving reason, and it is about the scan
alone. The width/gap invariant of section 1 was measured at low `k` and does survive --
0.515, 0.495, 0.587 over `k` bands [3.5,5], [7,8.5], [12,13.5] on rect 2x1 -- because the
PLATEAU falls with `k` (0.285 to 0.089) while `c` barely moves, so the dip width shrinks in
step with the gap. The mild upward drift should be re-measured above `k ~ 15`.

## What settling the operational definition changed

Section 2 was written last, against measurements in `notes/landscape_findings.md` sections
15-19, and writing it invalidated four things this document had previously asserted.

**Two constants were eliminated rather than tightened.** A coalescence tolerance `kappa` and a
predicted-accuracy constant `C` both failed section 2.1's rule. `kappa` was written as
`m * eps/c` with `m ~ 120` taken from the max of THREE seeds; extending to 30 seeds on three
domains moved the same ratio by three orders, because the splitting it was bounding does not
scale with `eps` at all -- it is constant to 1% while `eps` moves six orders. The right form is
relative, so `kappa` collapses into `tol_merge = prec` and stops existing. `C` was replaced by
measuring `delta_res` and comparing it against `prec`.

**A basis contract was withdrawn.** `gap > 240 * eps/c` was derived THROUGH `kappa` and
inherited its wrong form. The contract that survives, `gap > 20 * eps/c`, was already measured
directly in section 9.4 of the notes and simply had not been consulted. An order of magnitude,
not two and a half, and from a measurement of the thing itself rather than from a chain.

**Phase D's collapse-versus-merge moved off distance.** Two roots at the same separation can be
one eigenvalue found twice or two genuine eigenvalues, so no length threshold can decide it.
The interior Gram angle can, with distinct pairs at `|cos| <= 0.10` against duplicates at
`>= 0.948`.

**An interior-fill change was proposed and then withdrawn.** `method="grid"` measures 30x to
1e4x more accurate as a discrete inner product, and annihilates the mode `side+1` exactly,
because its points sit on that mode's nodal lines. The metric had been chosen before the
failure mode was, and it flattered the alternative -- the same error as the planted null
direction and the `c = 50` narrow dip. The interior block wants NON-DEGENERACY, not accuracy.
What should change is only `MPSEigensolver`'s `rng=None` default, to a fixed constant: no seed
is bad, and a fixed one removes the 110x run-to-run spread in the located eigenvalue and the
noise it puts into a shape objective.

The pattern in all four is one thing: a constant derived through another quantity was used
before being checked against a direct measurement of what it was supposed to bound.

## What the scoring apparatus learned

**Ground truth must not pre-judge multiplicity.** `chladni.reference`'s `*_spectrum` returns an
unfolded multiset with true degeneracies made bit-identical, and merges nothing. An earlier
version collapsed at `rtol = 1e-9`, which would have scored a search that correctly resolved a
4e-09 pair as reporting a spurious root.

**Unfolding separates failure modes.** Matching multisets tells "you missed an eigenvalue"
apart from "you found the level and undercounted it" -- different bugs, previously one number.

**The surrogate is for speed, not fidelity.** 1350x faster than a real pencil, matches
`sigma_1` in the dips to 4e-03 relative, and does NOT model `sigma_3` and above at all. The
(C1) gap test reads exactly that branch, so any multiplicity conclusion drawn on it is
provisional until confirmed on a real pencil.
