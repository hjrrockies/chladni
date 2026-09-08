# Handoff: picking up the minimization routine

Working note, not tracked. Written at `5e1f0aa`, after the session that built the reference
layer and the recall harness. Read `docs/design/search.md`'s "As built" section first -- this
adds what a fresh context window needs that is not in the tracked docs.

---

# 1. Where to start

The next concrete task is the **recursion-floor bug** in `lappy_repaired_search`, recorded in
`search.md`. It is the only known correctness failure in the search that currently works, it
was found by the Weyl audit rather than by the surrogate, and it is small enough to be a good
first move in a new context.

After that, in rough order:

1. Rebuild `lipschitz_search` against `search.md` section 4, or decide the repaired bisection
   is enough and delete it. Do not patch the existing one.
2. Move the winner into `src/chladni/mps/` with its own spec, and implement
   `MPSEigensolver.solve_interval`, which currently raises.
3. Basis planning, which everything above is quietly waiting on -- see section 4.

---

# 2. Methodological lessons, in order of how much they cost

These are the ones worth carrying. Every single one was learned by being wrong first.

## Measure before you argue, and especially before you design

Essentially every confident claim made from a mechanism rather than a measurement turned out
false this session:

    the rank-truncation discontinuity story      refuted: the minimizer is never near the cut
    "eta is 1e-3, ten orders above eps"          artifact of a line-fit metric measuring curvature
    "roundoff ties blind the detector"           0 ties on real curves at any usable spacing
    "the offset detects clusters earlier"        fires at exactly the same coarseness as sigma_1
    "true degeneracies are bit-identical"        false for rect(sqrt(2), 1)
    "narrow dips are a failure mode"             no domain produces one; c is 0.45 to 1.73

The `lipschitz-cone-mps.md` proposal is written in the same style -- plausible mechanisms,
carefully reasoned. Two of its claims held under measurement, two did not. Treat it as a
source of hypotheses, not conclusions.

## Profile the baseline, not only the alternative

Three separate speed comparisons were run against a baseline that was itself doing redundant
work (`choose_rank` computing an SVD that `regularize` then recomputed). All three erred in
the same direction: flattering the alternative. `qrp` measured 1.28-1.48x and is actually
1.01-1.04x. Fixing the baseline was worth more than either alternative.

## A synthetic stress case must be checked for transferability

Two constructions "proved" mechanisms that cannot occur in MPS:

* a planted null direction showed threshold truncation deleting an eigenfunction -- but it made
  `||A_I v||` small too, which is a redundant basis column, not an eigenfunction. The interior
  block guarantees a real minimizer has `||A_I c||` of order one.
* a `c = 50` narrow dip defeated every search -- and no domain produces one.

Both were written into a design document as justification before being checked.

## When patching oscillates, the spec is missing

Five attempts at `lipschitz_search`'s bracket handling each fixed some cases and broke others.
That pattern means the structure is wrong, not the details. Stop, write the spec, rebuild.
`search.md` section 4 exists because of this.

## Test the equality case, not the inequality

`bounds.szego_weinberger` was wrong by a factor of 4.3 -- it used the first zero of `J_1` where
Szego-Weinberger needs the first zero of `J_1'`. An inequality four times too weak passes every
test that only asks whether it is violated. It was caught because the disk attains the bound.

## Ground truth must not encode policy

`*_spectrum` collapsed near-equal eigenvalues at `rtol = 1e-9` by default. That put "these two
count as one" INTO the truth, so a search correctly resolving a 4e-09 pair would have scored a
spurious root. Truth says how many eigenvalues there are; merging is the search's declaration
and the harness's to score.

---

# 3. Traps hit in this session, mechanically

* **`cat > existing_file`** silently destroyed 16 tests in `tests/test_library.py`. Caught only
  because `git status` showed the file as modified rather than new. Check before overwriting.
* **`git stash --keep-index`** swept the entire working tree including untracked files. Use
  explicit copies when reconstructing commit states.
* **`k` versus `lam` in prose.** The harness works in `k`; the reference and every user-facing
  number are in `lam`. Reporting `k` values as "eigenvalues" caused real confusion.
  `conventions.md` asks that a parameter's NAME state its variable -- extend that to prose.
* **API CHANGE, later in the same session: the whole `mps` layer now speaks `lam`.**
  `MPSPencil.tension/tensions/coef/rank_at/regularized/matrices/pencil/log_scale/
  rank_diagnostics` all take `lam`, and `k_ref` is now `lam_ref` (a scalar or a `(lo, hi)`
  pair, in `lam`). The seam is `mps`/`helmholtz` and nothing finer: the last function taking
  `k` is the free `assemble(bdry, interior, k)`, because it calls into `helmholtz`, where a
  basis function IS a function of `k*r`. Sections 3b and 3c below argued the refinement should
  happen in `lam`; this makes the API stop fighting that. **Anything in this file written as
  `p.tension(k)` should now read `p.tension(lam)`** -- the notes scripts (`cone_model.py`,
  `k_vs_lam.py`, `landscape_lab.py`) were updated and re-run, and reproduce their previous
  numbers exactly. The harness and searches still scan in `k`, which is a separate and still
  open choice; see 3b.
* **numpy 2:** `arr.ptp()` is gone, use `np.ptp(arr)`.

---

# 3a. What `eps` and `c` actually are

Everything downstream is stated in these two, so get them straight before reading section 4.
Near an eigenvalue `k*` the tension is a cone with a rounded bottom:

    sigma_1(k)  ~  sqrt( eps**2 + c**2 * (k - k*)**2 )

`notes/cone_model.png` is the picture, regenerated by `notes/cone_model.py`. It is the L-shape's
first eigenvalue with a pure Fourier-Bessel basis at the reentrant corner, at three basis sizes.

**`eps` -- the FLOOR.** `sigma_1` AT the eigenvalue. Dimensionless, since sigma is the ratio
`||A_B c|| / ||A_I c||`. It is the boundary residual of the best eigenfunction the finite basis
can represent, so it is a property of THE BASIS, not the domain. It is never zero and only
accidentally near machine epsilon:

    20 FB terms    eps = 4.0e-05
    40 FB terms    eps = 1.6e-08
    60 FB terms    eps = 1.1e-12

**`c` -- the WALL SLOPE.** `d(sigma_1)/dk` away from the eigenvalue. Units of length, since
sigma is dimensionless and `k - k*` is an inverse length. It is a property of THE DOMAIN, not
the basis -- across that same table it moves only 1.602 -> 1.595 while `eps` falls seven
orders. Measured 0.45 to 1.73 over every domain tried, and roughly half the diameter.

The left panel shows the separation directly: all three basis sizes lie on ONE wall for
`|delta| > 1e-4`, then peel off onto their own floors.

**`eps/c` -- the KNEE, and the resolution limit.** The scale in `k` at which the cone stops
being a cone and flattens into its quadratic bottom. The stars in the left panel walk left by
four decades as the basis improves. The right panel is the bottom rescaled by `eps` and
`eps/c`, so it is basis-independent: the pure cone `c|delta|` is fine beyond about two
knee-widths and badly wrong inside one, saying zero at `delta = 0` where the truth is `eps`.

Two eigenvalues closer than about `eps/c` have overlapping rounded bottoms. Neither the dip
structure nor the second branch can then separate them, which is exactly the regime where the
(C1) gap rule reports multiplicity 2 at a SIMPLE eigenvalue -- measured failing at
`d/(eps/c)` = 0.45 and 0.15, and correct at 23.

**Which is the concrete form of "the search is bounded by the basis."** `c` is fixed by the
geometry, so `eps` ALONE decides how close two eigenvalues can be before the solver stops
being able to tell them apart. On the L-shape, 20 -> 60 terms moves the resolution limit from
2.5e-05 to 6.7e-13, a factor of 4e+07. Whatever the basis planner hands you sets that number.

One caveat on the left panel: the 60-term points below `delta ~ 1e-13` sit where `k*` itself is
only known to about 14 digits, so they are at the edge of what the reference can resolve rather
than a clean measurement of the floor.

---

# 3b. Refine in `lam`, not in `k` -- measured, and it contradicts the proposal

`lipschitz-cone-mps.md` section 2.2 says to work in `k`, and gives two reasons: Weyl spacing is
asymptotically uniform in `k`, and the wall slope `c` is more nearly constant there. Both are
about THE SCAN. Neither is about the REFINEMENT, where Brent fits a parabola to `sigma_1**2` --
and there the answer goes the other way.

`notes/k_vs_lam.png`, from `notes/k_vs_lam.py`. Fitting a quadratic to `sigma_1**2` over
IDENTICAL physical sample points, and measuring the residual relative to the data range:

    domain      lam*        vs k       vs lam     lam better
    rect 2x1    12.33701    3.6e-03    4.0e-04       8.9x
    rect 1x1    19.73921    3.6e-03    3.9e-04       9.1x
    rect 5x1    13.42266    3.5e-03    4.0e-04       8.9x
    disk         5.78319    2.8e-03    3.5e-04       8.1x
    eq_tri      52.63789    2.9e-03    3.1e-04       9.3x
    L-shape      9.63972    3.4e-03    2.2e-04      15.4x

**17 of 17 across four domains and three aspect ratios, median 5.1x, range 2.5x to 15.4x.**
Not an L-shape artifact -- rectangles are among the strongest cases. The left panel shows why:
the residual against `k` is a systematic arc, the residual against `lam` is flat noise.

The vertex accuracy follows: fitting in `lam` locates the eigenvalue 5.1x better on median
(14/14 cases), from the same samples.

**And one of section 2.2's two reasons for working in `k` is simply backwards** -- see 3c
below. So the local model in section 2 is written in the wrong variable too. It states

    sigma_1(k) ~ sqrt(eps**2 + c**2 (k - k*)**2)

which makes `sigma_1**2` exactly quadratic in `k`; measurement says it is closer to quadratic
in `lam`. Fitting `eps**2 + c**2 delta**2` directly to the data leaves a residual of 1.5e-02 at
h = 0.1, against 8.7e-04 for a plain quadratic in `lam` on the same points.

The practical consequence is narrow and concrete: **Phase B should minimise over `lam`, and
Phase A can stay in `k`** if its own reasons hold. They are separate choices and the proposal
conflates them under one section.

**Not yet checked** -- worth doing before acting on this:

* whether the advantage survives once the bracket is at `delta_res` scale, where the rounded
  bottom dominates and both parametrisations are near-quadratic anyway;
* whether it holds at large `lam`, since `dlam/dk = 2k` grows and the two variables become
  more different, not less;
* whether it changes the SCAN at all -- nothing here bears on the stride or the envelope.

On symmetry, since it is the natural question: neither variable is exactly symmetric about the
minimiser, because the cone model is local and the neighbouring eigenvalues are not symmetric
about this one. At matched physical offsets on the L-shape, `sigma(k*-h)/sigma(k*+h)` reaches
0.900 at h = 0.3 while the `lam`-domain ratio reaches 0.987 -- so `lam` is MORE symmetric here,
not less, which is the opposite of what the `sqrt` map alone would suggest and is another
reason not to reason about this from the mapping instead of measuring it.

---

# 3c. The spacing is uniform in `lam`, not in `k`

`lipschitz-cone-mps.md` section 2.2 says "Weyl's law makes eigen-wavenumbers asymptotically
equispaced with density |Omega|k/2pi". That sentence contradicts itself -- a density that
depends on `k` is not equispacing -- and the conclusion is backwards.

Weyl's law is LINEAR IN LAM:

    N(lam) ~ |Omega|*lam/(4*pi) - |dOmega|*sqrt(lam)/(4*pi)

so `lam_n ~ 4*pi*n/|Omega|`, linear in `n`, and `k_n ~ sqrt(n)`. Eigenvalues are asymptotically
equispaced in `lam`, with constant gap `4*pi/area`; in `k` the gap SHRINKS like `1/k`.

Measured, rect 2x1 (predicted `lam` gap `4*pi/2 = 6.283`) and the unit disk (predicted 4.0):

    n range      mean d(lam)   mean d(k)   d(k)*k
    (20, 60)        7.086        0.208      3.502
    (100, 140)      6.833        0.118      3.402
    (200, 240)      6.833        0.089      3.408
    (350, 390)      6.327        0.064      3.158

`d(lam)` is flat and converging on the predicted value; `d(k)` falls by a factor of three over
the same range, and it is `d(k)*k` that is constant.

This matters twice. It removes one of the two reasons section 2.2 gives for working in `k`,
leaving only "`c` is flatter in `k`" -- which is measured and true, but is about the SCAN
alone. And combined with 3b it means `lam` is the better variable on both counts that have
actually been checked.

## The width/gap invariant survives, with a caveat

Section 5 records "dip half-width / eigenvalue spacing is 0.46 to 0.52 on four domains", which
was measured entirely at LOW `k`. If the gap shrinks like `1/k` while the dip width stayed
fixed, the ratio would grow and dips would eventually overlap. Measured on rect 2x1:

    k band        plateau S     c      dip 2S/c   mean gap   width/gap
    [3.5, 5.0]      0.2854    1.192     0.479      0.931      0.515
    [7.0, 8.5]      0.1934    1.090     0.355      0.717      0.495
    [12.0, 13.5]    0.0886    1.047     0.169      0.288      0.587

It holds, and the mechanism is that the PLATEAU FALLS with `k` -- 0.285 to 0.089 -- while `c`
barely moves, so the dip width shrinks in step with the gap. There is a mild upward drift
(0.515 to 0.587 over that range) which should be re-measured before anyone relies on the
invariant far above `k ~ 15`.

---

# 4. The thing everything is waiting on

**Basis planning is deferred and it bounds all of it.** `MPSEigensolver` requires an explicit
basis. Two independent findings point at this:

* `eps` varies by FIVE orders within a single L-shape scan (3.1e-13 at k = 4.443 against
  5.5e-08 at k = 5.649), structurally: modes with no singular amplitude at the reentrant
  corner are resolved far better than their neighbours;
* multiplicity correctness depends on `eps/c` -- once a pair is closer than that, the gap rule
  alone reports multiplicity 2 at a SIMPLE eigenvalue.

So search difficulty is downstream of basis quality, and benchmarking a search against
hand-built bases may not predict black-box behaviour. Whatever the planner ends up being, the
lab's output should include a CONTRACT: the `eps` and dip width a search can tolerate.

`chladni.reference`'s `recipe(fn)` now records what basis each table needed, which is the
beginning of that contract -- the L-shape wants pure Fourier-Bessel at one corner, 80 terms
for 14 digits, and nothing else.

---

# 5. What is measured and where

`notes/landscape_findings.md`, sections 1-14, is the technical record: the cone model, the
roughness-versus-rank relation, cluster resolution, the (C2) necessity measurement, the
lappy-versus-repaired scoring, and the width/gap invariant. Do not re-derive these.

The two most reusable numbers:

    dip half-width / eigenvalue spacing    0.46 to 0.52 on four domains
    cone slope c                           0.45 to 1.73 across all domains measured

The first says about six samples per eigenvalue gap always place three inside every dip, which
is what Weyl's law sizes directly. The second says a per-solve Lipschitz calibration is both
necessary and sufficient.

---

# 6. State of the tree

    5e1f0aa   harness, three searches, Weyl audit          (lipschitz_search KNOWN BROKEN)
    1bee8b4   reference recipes, the 2*pi**2 correction
    bbaa93b   bounds, asymp, weyl_window refactor          (fixed a 4.3x lappy bug)
    cc23e62   the reference package
    8dbbfe0   ten domain factories                         (fixed a lappy stadium bug)

1017 tests with `slow`, 1009 without. `pytest -m ""` is what CI runs.

`MPSEigensolver.solve_interval` raises `NotImplementedError` naming the deferred layer. That
is the seam the minimization work fills.
