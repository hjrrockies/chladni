# The eigenvalue search

Given a tension function `sigma(x)` returning the ascending vector
`sigma_1 <= sigma_2 <= ...`, find every approximate eigenvalue in `[a, b]` with its
multiplicity, using as few evaluations as possible.

This is the layer above `mps`. It sees only the tension function and the interval: no domain,
no basis, no pencil. That restriction is deliberate and is what makes two candidate searches
comparable, and what will let a later FEM or boundary-integral method reuse the same search.

The originating proposal is `lipschitz-cone-mps.md`. This document is the spec: it records
which of that proposal's claims have been measured, corrects the two that were wrong, and
specifies the parts it left open -- which are exactly the parts that a first implementation
attempt got stuck on.

---

# 1. What is measured

Everything in this section is a number from `notes/`, on real pencils unless stated. The
design rests on these rather than on the local model being assumed.

**The cone model holds.** `sigma_1 ~ sqrt(eps^2 + c^2*(x - x*)^2)` fits the far field to a
median relative residual of 1-3% on rect, disk, eq_tri and L-shape, failing only where the
floor is close to the plateau, i.e. where the basis cannot represent the eigenfunction.

**`c` is stable within a domain, not across domains.** 0.98-1.17 on rect over k in [3.3, 7.3];
1.56-1.70 on the disk; 0.45-0.49 on eq_tri. Per-solve calibration is necessary and sufficient.

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

---

# 2. What the proposal got right, and what it got wrong

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

# 3. Why adaptive subdivision -- the argument that survives

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

# 4. The algorithm

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

**Termination.** Stop at `max(tol_rel * x, delta_res)` with
`delta_res = sqrt(2*eps*eta)/c` from the current parabola fit, or on stagnation -- the last
three values agreeing to within `eta`. The flat half-width `eps*sqrt(2*eps_mach)/c` is the
scale below which the curve is roughness rather than signal; `h_floor` in Phase A is set from
the same quantity. Report the attained precision and which bound was active.

**Retain a wide bracket.** Keep the pair of evaluated points either side of the root at
distance of order `eps/c`, from before the bracket collapsed. Phase C needs points where the
higher branches have actually moved; the final bracket is far too tight for that, and using it
is why an earlier attempt's multiplicity test was reading noise.

## Phase C -- multiplicity

From the cache; no new evaluations. Index `j` is active when both hold:

* **(C1)** the largest ratio gap in the low spectrum falls after `j`, i.e.
  `sigma_{j+1}/sigma_j >= G`. A gap test, not an absolute threshold, because the floor varies
  fifty-fold within one scan.
* **(C2)** `sigma_j` is a local minimum at the root, against the WIDE bracket of Phase B.

The multiplicity is the number of active indices, necessarily a contiguous block `1..kappa`.

## Phase D -- collapse, then merge

Bottom-up, and the order matters.

* **Diameter <= `delta_res`**: the members are the same well found twice. Collapse to the one
  with the smallest `sigma_1` and RECOUNT the multiplicity there. Do not sum -- summation is
  the double-count.
* **Diameter <= `tol_merge * x`**: separately resolved and independently verified, so total
  multiplicity is the SUM. Place at the `sigma_1`-weighted centroid and report the diameter as
  the uncertainty.

---

# 5. Diagnostics, and the shape of the claim

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

# 6. Tests

Written before the implementation, against the surrogate and against real pencils.

| file | what it defends |
|---|---|
| `tests/test_search_scan.py` | Phase A: the envelope never prunes an interval containing a root when `L` is a true bound; the attic revives on a raised `L`; the subdivision depth reaches `beta*d_est` on a pair and stops at `h_floor` on a degeneracy |
| `tests/test_search_refine.py` | Phase B: the apex case (root on a shared endpoint) is recovered to the floor; a wall point adjacent to a deep root is rejected; symmetric sampling still finds the well; stagnation terminates |
| `tests/test_search_multiplicity.py` | Phase C: 9 for 9 on disk and eq_tri; (C2) rejects the `d <~ eps/c` double-count; the wide bracket is wide enough |
| `tests/test_search_score.py` | end to end through the harness, against the uniform-grid control and the lappy baseline, on the surrogate and on real domains |

The harness scores recall first, counted with multiplicity, and a root already claimed cannot
be claimed twice -- so a cluster reported from two brackets scores as one match and one
spurious rather than as two matches.

---

# 7. Deferred

The two-level rank of section 2, which needs the pencil. Automatic `gamma` escalation on a
small margin. Parallel evaluation, which changes the objective from evaluation count to
wall-clock rounds and may favour a graded uniform grid instead -- the branch and bound's
advantage is adaptivity, and adaptivity is worth less when points are nearly free.

---

# As built: deltas from this spec

What exists at HEAD, what it scores, and what is still wrong. As in `geometry.md` and
`particular_solutions.md`, the body above records what was intended and this section records
what the implementation found; the gap between them is the useful part.

## What is built

| | |
|---|---|
| `tests/harness.py` | the scorer, the Weyl audit, `unfold`, `levels_agree`, and the uniform-grid control |
| `tests/searches.py` | `lappy_search`, `lappy_repaired_search`, `lipschitz_search` |
| `tests/surrogate.py` | a 2 us stand-in for the tension curve, fitted and validated against real pencils |

Nothing is in `src/chladni/` yet. When a search wins it moves there, rewritten against a spec
that this section will have corrected.

## The repaired baseline is the thing to beat

Two changes to `lappy_search`, and the win is larger than the branch and bound's:

* recurse into EVERY discrete minimum of `sigma_1`, rather than gating on `sigma_2`'s value;
* multiplicity by the gap plus local-minimum conditions, not a prefix against an absolute
  `ttol`.

Twelve surrogate cases: 10/12 perfect at 5573 evaluations against the original's 10/12 at
54443. On real rectangle pencils with exact truth it also LOCATES better -- 1e-15 to 3e-13
against 6e-09 -- because the deeper recursion hands Brent a much tighter bracket.

So the `O(log(1/d))` advantage of section 3 is a property of adaptive subdivision, which
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

The last one is diagnosed and is section 4's `d_est` rule not being reached, not the rule
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
