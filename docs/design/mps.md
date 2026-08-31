# Overview

The core feature of `chladni` is a robust implementation of the (regularized) GSVD
formulation of the Method of Particular Solutions (MPS). MPS is built around finding linear
combinations of Helmholtz-exact "particular solutions" which satisfy (or nearly satisfy) the
boundary condition. These combinations are found with the Generalized Singular Value
Decomposition. Eigenvalues are located by scanning across `lam`, identifying near-zero minima
of a quantity called the *tension*.

This document specifies the layer up to and including the tension. The eigenvalue search and
multiplicity estimation are deferred; so is Rellich-identity orthonormalization, which belongs
to `boundary_integrals.md`.

---

# Mathematical formulation

Let `psi_1, ..., psi_n` be Helmholtz-exact particular solutions at wavenumber `k`, and let
`x^B_1, ..., x^B_{m_B}` and `x^I_1, ..., x^I_{m_I}` be boundary and interior collocation
points. Write `A_B(k)` for the `m_B x n` matrix of boundary *traces* of the particular
solutions and `A_I(k)` for the `m_I x n` matrix of their interior values.

The trace is the boundary condition applied to the solution, node by node:

    (A_B)_ij = (1 - bc_i) * psi_j(x^B_i) + bc_i * dpsi_j/dN (x^B_i)

with `bc = 0` Dirichlet, `bc = 1` Neumann, and `bc` in `(0, 1)` Robin. The `bc_i` are
per-node, so a boundary carrying different conditions on different segments needs no special
case.

Both matrices depend on `k`: the particular solutions themselves vary with `k`, remaining
Helmholtz-exact at every value.

The **tension** is the smallest generalized singular value of the pair:

    sigma_1(k) = min_{c != 0} ||A_B(k) c|| / ||A_I(k) c||

It measures how well a candidate eigenfunction can satisfy the boundary condition without
being trivially small in the interior. The denominator is what stops the minimizer from
returning the zero function; it is why this is a *generalized* singular value problem and not
an ordinary one. `lam = k**2` is an eigenvalue exactly where `sigma_1` has a near-zero
minimum, and the minimizing `c` is the coefficient vector of the eigenfunction.

The full ascending vector `sigma_1 <= sigma_2 <= ...` carries more: an eigenvalue of
multiplicity `m` drives `m` of them to zero together, which is what the deferred multiplicity
estimator will consume.

---

# Design principles

- A base eigensolver class, method-agnostic, keeping the door open for FEM and
  boundary-integral methods.
- A new version of `lappy`'s `MPSEigensolver`, responsible for assembling the collocation
  matrices, dispatching to `gsvd4py`, and solving for eigenvalues and eigenfunctions.
- Focus first on matrix assembly, regularization, GSVD calls, and GSVD post-processing. The
  optimization routine comes after.
- Use `lappy` for inspiration, not as the definitive reference. We want to do better.

---

# The design questions, answered

## 1. Can `gsvd4py`'s tolerances regularize the pencil for us?

**Yes -- and so can a threshold. The rank rule turns out not to be a correctness question at
all.** This answer was written the other way round and the measurement overturned it; the "As
built" section below carries the numbers, and this text is the correction rather than the
original claim softened.

`gsvd4py` exposes `tola`/`tolb`, which feed LAPACK's `?ggsvp3` and set a numerical rank by
thresholding. lappy's SVD regularization does the same thing one level up, cutting at
`(s >= rtol).sum()` and re-deriving the rank at every parameter.

The expected failure was that a re-derived rank makes the retained rank a step function of
`k`, so a singular value crossing the cutoff jumps the tension -- possibly deleting the
eigenfunction outright, since near an eigenvalue the minimizing combination is the one `A_B`
annihilates and is therefore small. **The second half of that is wrong, and the interior block
is why.** The tension is `||A_B c|| / ||A_I c||`, so a minimizer has `||A_I c||` of order one
by construction -- that is exactly what the denominator is for -- and the cutoff acts on the
STACKED pencil, where the minimizer therefore has healthy norm. Measured on the L-shape at
`lam_1`, where the tension reaches 1.7e-12 and the danger should be worst, the minimizer
carries *all* of its energy in the retained subspace and the threshold rank never moves.

A direction small in both blocks does exist and does get truncated, but that is a redundant
basis column rather than an eigenfunction, and dropping it is correct.

So `chladni` still freezes the rank, on two narrower grounds than the spec first claimed:

- **Reproducibility.** The rank becomes a property of the solver rather than of `k`, so
  `tension` is a function of `k` alone: two visits to the same parameter agree, and a
  re-solve does not move for reasons unrelated to what changed.
- **Cost.** It skips one `svdvals` of the stack per evaluation -- 11.3 ms against 13.4 ms per
  `k` on a 132-column L-shape pencil, about 19%.

`rank=None` selects `r` once, at a reference parameter, by a *relative* cutoff
`s >= rtol * s[0]` -- relative unlike lappy, whose `rtol` was absolute despite the name, so
its retained rank moved when the pencil was scaled. `rtol` derives from `prec`.
`rank_diagnostics` returns the singular values so the frozen choice can be audited, and warns
when `s[r]/s[r-1]` approaches 1, where a frozen index straddles a near-degenerate pair and the
retained subspace is genuinely ill-defined.

SVD-based and column-pivoted-QR regularization are both implemented, both index-truncated. The
`gsvd4py` tolerance path is `reg="implicit"`, labelled diagnostic-only for reproducibility
rather than for accuracy.

## 2. How do we expose a bare-bones domain-free API?

**As a second object, not as a hole in the first one.**

`lappy` tried to sever one `MPSEigensolver` class from the domain, reasoning that a user may
want a bespoke solver from a hand-built basis and collocation points. It failed twice: on
boundary conditions, where `make_bdry_vander` had to branch on `bc_param` and could not
express a mixed boundary except as a degenerate Robin blend; and on Rellich orthonormalization,
which needs boundary quadrature weights the severed object did not carry.

Neither failure is available here. `Sampling.trace(k)` already assembles
`(1-bc)*u + bc*du_dn` from the `Quadrature`'s own per-node `normals` and `bc`, so the pencil
never branches on a boundary condition and never needs a `Domain` to ask what the condition
is. And a `Quadrature` carries its weights, so the deferred Rellich work has what it needs
from the same object.

Two classes:

- **`MPSPencil`** -- domain-free, built purely from a basis and collocation points, speaks
  `k`. Owns equilibration, regularization, the GSVD call, tension and coefficients. This *is*
  the bespoke path; it exists by construction rather than by pretending.
- **`MPSEigensolver`** -- explicitly domain-dependent, speaks `lam`, wraps an `MPSPencil`.
  Owns the domain, and eventually the basis planner and the point budget. Either may be fixed
  explicitly by the caller.

The dependency that survives is on a *quadrature*, not on a domain, which is the thing
actually needed.

## 3. How do we plan ahead for the deferred minimization?

By fixing what the search will consume and nothing else.

`tensions` returns the full ascending vector, is callable on an array of parameters,
is deterministic across repeated calls at the same parameter, and is safe to share across
threads. `tests/test_mps_seam.py` pins those four properties, the way
`test_helmholtz_seam.py` pinned this layer's dependencies on the one below.

The base class declares `solve_interval(a, b)` abstract and implements `solve(n)` over it
with a Weyl window. `MPSEigensolver.solve_interval` raises `NotImplementedError` naming the
deferred layer -- one honest stub, against `lappy`'s three `pass`-bodied `default_*` methods,
one of which is missing `cls` and would fail if ever called.

One cost is noted and not solved: `gsvdvals` computes every generalized singular value, and
`lappy`'s search uses two. There is no way to ask LAPACK for fewer.

## 4. What else can we do about conditioning?

Three positions, and one open measurement.

**Equilibration is ours, accuracy is not.** `conventions.md` splits these: `helmholtz` owns
entry-wise relative accuracy and hands back columns pre-scaled by `exp(-sigma)`; `mps` owns
pencil conditioning. So the unit-norm rescale is a step inside `MPSPencil`, not `lappy`'s
`NormalizedBasis` wrapper around the basis object. The log scales add, so a coefficient vector
still means the same function on every point set.

**Unscaled coefficients are never materialized.** At a `pi/8` corner with `k*R = 10` the
`sigma` range spans about 1149 nats. `coef` returns the vector in the scaled space and
`log_scale` hands back the recovery factor.

**Evaluate first, sandwich never.** `lappy`'s rule for eigenfunction quantities: form
`A @ c` once and work with the resulting vector, rather than forming an `n x n` Gram in basis
space and sandwiching it. Recorded here so the eigenfunction work inherits it.

**Open:** whether a coefficient vector can be reused across point sets to build Cauchy data,
or whether that needs a GSVD recompute on the new pencil. This is a measurement, and it
belongs with the eigenfunction work rather than here.

---

# Structure

`Eigensolver` is method-agnostic and lives at `chladni/eigensolver.py`, above `geometry` and
below every method package -- the same position `helmholtz` occupies. Not in `chladni/mps/`,
because a later `chladni/fem/` must not import its base class out of the MPS package; and not
in `chladni/_core.py`, which is deliberately dependency-free.

    chladni/eigensolver.py     Eigensolver, lam_to_k, k_to_lam, weyl_window
    chladni/mps/pencil.py      MPSPencil; assemble, equilibrate
    chladni/mps/_regularize.py regularize, choose_rank; RegularizedPencil
    chladni/mps/_gsvd.py       tensions, coef
    chladni/mps/solver.py      MPSEigensolver

**Every step is a free function; the classes only thread parameters.** As in `lappy`, where
`regularize_pencil`, `tensions` and `nullspace_coef` are module-level and callable on raw
arrays. The pipeline is ill-conditioned by nature, so every stage has to be inspectable on its
own inputs without standing up a solver. The free function holds the algorithm; the method
holds the defaults, and nothing else.

The names collide deliberately: `mps.tensions(A_B, A_I, ...)` is the free function,
`MPSPencil.tensions(k)` assembles and calls it, `MPSEigensolver.tensions(lam)` converts and
calls that. Below the class level everything is arrays.

---

# The pipeline

    assemble      (bdry, interior, k)  ->  A_B, A_I
    equilibrate   A_B, A_I             ->  B, I, log_col_norms
    regularize    B, I                 ->  RegularizedPencil(B, I, back, rank, s)
    gsvd          B, I                 ->  tensions  or  coefficients

**Assemble.** `A_B = bdry.trace(k)`, `A_I = interior.values(k)`. That is the whole of it; the
boundary-condition logic lives one layer down and this layer never sees it.

**Equilibrate.** Each column of the stacked `[A_B; A_I]` is scaled to unit 2-norm, and the log
of that norm adds to `family.log_scale(k)` to give the pencil's total per-column log scale.
Columns that underflow to exact zeros keep scale 1 and are left for the regularization to
drop, rather than dividing by zero -- `lappy` pruned them by hand.

**Regularize.** One function, four modes, and `back` is the map from reduced coordinates to
basis coordinates, produced by whichever mode ran. That is the fix for `lappy`'s
`nullspace_coef`, which branches on `reg_type` *again* at the recovery site: two places that
must agree about a transformation only one of them computed.

- `"svd"` (default) -- economic QR of the stack, SVD of `R`, truncate to `r`.
- `"qrp"` -- column-pivoted QR, truncate to `r`.
- `"implicit"` -- no pre-truncation; hand `tola`/`tolb` to `gsvd4py`. Diagnostic only.
- `None` -- raw pencil.

`lappy`'s `gesdd -> gesvd` fallback is kept, with its warning. Its docstring records one shape
in roughly 1500 solves raising, taking down an hour-long sweep.

**GSVD.** `gsvd4py.gsvdvals` returns `c` descending and `s` ascending, so `c/s` is descending
and the tension vector is its reverse; division by zero gives `+inf` rather than raising. For
coefficients, `mode="separate"` gives `Q` and `R` without forming and inverting `X`; the
`mult` smallest ratios are found by `lexsort((s, -c))`, since separate mode is unsorted, and
`Xr = Q[:, -q:] @ solve_triangular(R, Er)` is mapped back by `back`.

**Caching.** The regularized pencil at a given `k` is wanted by both `tensions` and `coef`, so
it is cached in a bounded dict created at construction. No lock, no eviction policy, no
weakref ledger: the object is immutable, which is the whole reason `lappy`'s 370-line
`instance_cache` is unnecessary here.

---

# Ground truth

Closed form only, per `conventions.md`. `lappy`'s tabulated tables are not ported: its own
`todo.md` records `ellipse_eigs(a=2)[0]` being wrong in the 9th digit.

- **Dirichlet circular sector of angle `omega`.** A *single* Fourier-Bessel column with
  `mu = l*nu` is an exact eigenfunction. The sharpest test available, it needs no basis
  planner, and it pins the tension and the coefficients at once.
- **Rectangle, disk, equilateral triangle.** Closed forms, with ambient families.
- **L-shape `lam_1 = 9.6397238440219410527`**, `slow`. Should reach 1e-12 with corner
  enrichment against about 1e-5 without; that gap is itself the regression test that the
  corner enrichment is wired up.

---

# Deferred

The eigenvalue search, bracketing, and multiplicity estimation. Rellich orthonormalization and
the boundary-integral Gram. Basis planning and the point budget -- the `basis=`, `n_bdry=` and
`n_int=` arguments are where it lands. And whether `k` or `lam` is the better variable for the
minimization itself, which `conventions.md` is deliberately loose about so it can be settled
by measurement.

---

# As built: deltas from this spec

Everything above still describes the intent correctly. This records where the code went
further, or differently, and why -- as `geometry.md` and `particular_solutions.md` do. The
live record is the code, the test suite, and the numbers each carries.

## The discontinuity argument was wrong, and the interior block is why

The spec argued for an index-based rank from the shape of the tension curve: a threshold makes
the rank a step function of `k`, so the curve should acquire jumps, and near an eigenvalue
truncation might delete the eigenfunction outright. That was written before it was measured.
It is wrong, and this section replaces it rather than softening it.

**The curve does not jump.** On a real L-shape pencil where the threshold rank genuinely moves
-- 41 to 43 across `k` in [3.0, 3.3] at the default `rtol` -- the tension barely notices:
largest step over median step is 2.8 for the threshold rule against 2.1 for the frozen one.
Both are noise.

**The eigenfunction is never at risk.** The deletion worry assumed a minimizer is small, hence
near the cutoff. It is small in `A_B` only. `||A_I c||` is of order one by construction, and
the cutoff acts on the stacked pencil. Measured on the L-shape at `lam_1` with 22 degrees per
corner, where the tension reaches 1.7e-12:

    ||A_B c||                                    1.2e-12
    ||A_I c||                                    7.4e-01
    cutoff (rtol * s[0])                         7.0e-12
    minimizer's energy in the retained subspace  1.000000

All of it. Scanning across `lam_1` in that configuration, the threshold rank never moves and
the two rules produce bit-identical curves.

A synthetic pencil with a direction small in BOTH blocks does show the deletion -- the tension
jumps from 1.9e-09 to 6.8e-01 as it crosses the cutoff -- but such a direction is a redundant
basis column, not an eigenfunction. That construction was briefly in the test suite as this
design's justification and has been removed: a mechanism that cannot arise in MPS is not
evidence about MPS.

**What the frozen rank is still for** is reproducibility and 19% of the per-evaluation cost.
Both are pinned in `tests/test_regularization.py`, alongside the measurement that the two
rules agree -- because that is what someone should find when they ask this question again.

## `rtol` derives from `prec`, and lands on lappy's measured constant

The spec says `rtol` derives from `prec` without saying how. It is `sqrt(prec * eps)`, the
geometric mean of what was asked for and what the arithmetic can do. At the default
`prec = 1e-8` that is **1.49e-12** -- which is where lappy's `rtol` landed after measuring
twelve domains and calling the result "a compromise, not an optimum". Derived here rather
than fitted, and it moves in the right direction: a tighter `prec` lowers the cutoff and
retains more columns, the same direction `PlaneWave.n_useful` documents one layer down and
the same one that is easy to invert.

## Two modules are private, because their names collide with their contents

`chladni/mps/_regularize.py` and `chladni/mps/_gsvd.py`, not `regularize.py` and `gsvd.py`.
`import chladni.mps.regularize` would be ambiguous between the module and the function of the
same name -- and since Python 3.7 resolves `import a.b as c` through the attribute first, the
package `__init__` re-export silently wins, which breaks any caller reaching for the module.
The function is the name callers want, so the module yielded.

## The solver may guess a reference parameter; the pencil may not

`MPSPencil` raises when asked for `rank=None` with no `k_ref`, on the rule
`Fundamental.radius` already set one layer down: a hand-built object has to state its
standoff, because guessing is a silent scale error. `MPSEigensolver` has a domain, so it can
do better than guess -- it takes the geometric midpoint of `weyl_window(domain, 10)` -- and
does.

## `weyl_window` and `solve(n)` came with the base class

Not in the spec, which only named `solve_interval` abstract. `solve(n)` needs an interval and
Weyl's law is the only method-agnostic way to get one, so both live on `Eigensolver` and both
are tested now, through a stub subclass that returns closed-form rectangle eigenvalues. That
keeps the one concrete method on the base class inside the spec-tests-implementation rule
rather than shipping untested against a search that does not exist.

The lower endpoint is Faber-Krahn padded by `1e-3`. The padding is load-bearing and has its
own test: Faber-Krahn is an *equality* for the disk, so an unpadded endpoint lands exactly on
`lam_1`, and lappy records what follows -- "the unit disk returned modes 2..k+1: every value
correct to ~14 digits, every certificate valid, and nothing in the pipeline able to detect the
omission."

## Measured accuracy

The L-shape at its first eigenvalue, `lam_1 = 9.6397238440219410527`, with the corner family
at every vertex and 800 boundary / 300 interior collocation points:

    degrees per corner    tension at lam_1
          14                  3.9e-09
          18                  1.6e-10
          22                  1.7e-12

against **3.6e-01** for plane waves alone. The spec predicted "1e-12 to 1e-13 with FB at the
reentrant corner against about 1e-5 without"; the enriched figure is confirmed and the
unenriched one was optimistic by four orders, measured with a plane-wave basis sized by
`PlaneWave.n_useful` rather than oversized.

Oversizing matters and is now a rule in the test file: past `n_useful` the extra plane-wave
columns arrive at the noise floor, and the `helmholtz` layer's own warning says what they do
there -- "they lower the smallest singular value and corrupt the tension". An oversized basis
makes the tension look better while meaning less.

## Still deferred

Unchanged: the eigenvalue search and multiplicity estimation, Rellich orthonormalization,
basis planning and the point budget, and whether `k` or `lam` is the better variable for the
minimization.

## The rank must be chosen over a RANGE, and the default is now threshold

Two corrections to the section above, both found by measuring the objective before designing
a search for it (`notes/`, not tracked).

**The pencil's numerical rank grows with `k`.** Measured on a 54-column disk at the default
`rtol`: 31 at `k = 2.3` rising to 47 at `k = 5.7`. Obvious in hindsight -- more oscillation
supports more independent columns -- and nothing in the first implementation knew it.

**So a rank frozen at one `k` is unsafe at the low end of a scan.** `MPSEigensolver` defaulted
`lam_ref` to the MIDPOINT of the Weyl window, which on that disk gives rank 39, whose relative
singular value at `k = 2.35` is 2.4e-16 -- machine epsilon. The retained directions are then
undetermined and the tension goes rough: **85 spurious local minima in a window containing
exactly one eigenvalue**, with 39% of the steps on the descending wall going the wrong way.
Choosing the rank over the range instead gives 31, and one minimum.

`k_ref` therefore accepts `(lo, hi)` and takes the minimum over it; `MPSEigensolver` passes
its whole Weyl window; and `rank_diagnostics` warns when the frozen rank has reached the noise
floor at an evaluated `k`.

**The default rank rule is now threshold**, re-derived at every `k`. It cannot make the above
mistake, because it reads the spectrum actually in front of it. The comparison, on the recall
metric that matters:

    domain    frozen chosen at   frozen r   thr r   minima frozen   minima thr
    disk      point k=3.95             39   31-33              85            1
    disk      range (2.2, 5.7)         31   31-33               1            1
    L-shape   either                   61      61               1            1

Threshold is correct in every case, including the one that breaks the frozen rank, and the
margin on the frozen side is thin: rank 37 still behaves at `k = 2.35`, rank 39 does not. It
costs 1.2 to 1.6x, and under "recall above all" that is the right trade -- what the frozen
rank needs in exchange, a correctly declared `k` range, is exactly what a black-box solver
iterating over generated domains cannot be trusted to supply. Freezing remains available, is
now safe when given a range, and keeps its reproducibility and cost advantages for callers who
know their range.

## `reg="implicit"` was given a fair trial and is not the default

The idea: hand `tola`/`tolb` to `gsvd4py` and skip our QR and SVD entirely. It works, and it
is supported. It is not the default, for two measured reasons.

**It applies a different criterion, not a different scale.** LAPACK's `?ggsvp3` thresholds `A`
and `B` separately in 1-norms; `choose_rank` thresholds the stacked pencil in 2-norms. At
equal `rtol` implicit truncates two to four columns harder, and on the disk **no single
`rtol` reproduces the svd rank at every `k`** -- 1e-15 gives [31, 34, 39, 43, 45] against
[31, 35, 39, 43, 47]. Calibrated on the L-shape (`rtol = 1e-15`, rank 61 throughout) the two
agree exactly on the eigenvalue: relative error 1.41e-10 and floor 4.14e-09, both identical.

**At matched rank the saving is small.** 1.10x on the L-shape. The profile says why: the
QR and SVD really are a large share of the route, but implicit does not remove that work, it
moves it inside LAPACK, whose `?ggsvp3` preprocesses all `n` columns rather than the reduced
`r`. Uncalibrated it looked like 1.6 to 2.0x, which was implicit doing LESS work, not the
same work faster.

One tempting shortcut is closed off, and worth recording because it is not obvious: `tola` is
purely a rank threshold in `gsvd4py`, never a convergence knob. `gsvd4py/_gsvd.py` pins
`?tgsja`'s Jacobi tolerances to the LAPACK defaults on purpose, "so that convergence quality
is independent of the user's rank choice", so loosening `tola` cannot buy speed by converging
less hard.

One behavioural difference for consumers: implicit can return `+inf` tensions, where the
`"svd"` route structurally cannot. The reduced stack `Q Z_1` has orthonormal columns, so
`s > 0` is guaranteed there; implicit hands LAPACK the raw pencil, and a direction the
interior block annihilates comes back as an infinite generalized singular value. It is the
honest answer and it sorts to the end, but nothing downstream may assume finiteness.

## The rank is derived inside each mode's own factorization

Found while giving `"qrp"` a fair trial, and it turned out to matter more than the trial did.

Threshold mode called `choose_rank` -- an `svdvals` of the stacked pencil -- and then handed
the resulting integer to `regularize`, whose `"svd"` branch computes the SVD of `R`. Since
`Q` has orthonormal columns those two decompositions have the SAME singular values, verified
to 4e-15. **The default path was computing the same SVD twice.** On a 134-column L-shape
pencil the redundant `svdvals` was 2.13 ms of an 8.19 ms evaluation.

Each mode now derives its own rank from its own factorization: `"svd"` from the singular
values it already has, `"qrp"` from `|diag R|`, `"implicit"` from LAPACK. `choose_rank`
remains a free function, for diagnostics and for picking a rank to FREEZE, where the number
is genuinely wanted before the reduction.

Measured gain on the default path, regularization and GSVD only:

    domain    cols    before    after    gain
    disk        54    2.61 ms   1.99     1.31x
    eq_tri     106    6.19      4.56     1.36x
    rect       126    7.04      4.83     1.46x
    L-shape    134    8.07      5.97     1.35x

## `"qrp"` is not faster, and that finding is entirely an artifact of the above

QRP was measured at 1.28 to 1.48x against the double-SVD baseline. Against the corrected one
it is **1.01 to 1.04x** -- no gain, and slower than `"svd"` on the largest pencil in one run.
Pivoted QR costs about what an unpivoted QR plus an SVD of `R` costs at these shapes: pivoting
inhibits blocking, and it loses on the QR what it saves on the SVD.

So the trial answers itself. What `"qrp"` costs in robustness is small but real -- it
overestimated the rank by up to ten columns on the rectangle, and its floor `eps` is one to
two digits worse (1.4e-13 against 1.6e-15 there) -- and what it buys is nothing. Its rank rule
is also only a WEAK rank revealer: on a Kahan matrix at n = 90 it overestimates the smallest
singular value by 3.4e4. Across 240 evaluations on four domains that never bit, the cut never
falling below 2.2e-13, three decades clear of where the tension goes rough. Real failure mode,
not observed here, no compensating benefit.

`"qrp"` stays available and stays honest about all three facts. **QR -> SVD -> GSVD is the
default because it is now also the fastest robust option**, which is a more comfortable place
to be than defending it on robustness alone.

A note on method, since this is the third time in this file: every speed comparison in the two
sections above was first measured against a baseline that was itself doing redundant work, and
every one of those numbers was wrong in the same direction -- flattering the alternative. The
fix was found by profiling the baseline rather than the alternatives.
