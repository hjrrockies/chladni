*Status: design, approved 2026-10-05. M0 (seams), M1 (rules), M2 (plan, Cauchy data, the
identity), M3 (the Moler-Payne bound, wired into the solver) and M4 (orthonormal cluster
eigenfunctions) are in; M5 (Hadamard) is next. Package: `chladni.integrals`.*

# Overview
`chladni` is going to rely on accurate boundary integration for two key parts of the MPS solve:
1. Eigenfunction orthonormalization using Rellich identities
2. Eigenvalue derivatives w.r.t. shape perturbation using Hadamard formulas

In both cases, we need to integrate the Cauchy data (values, normal and tangent derivatives) against a weight function. The Cauchy data of an eigenfunction is generically singular in the presence of corners. This means that `chladni` needs highly robust adaptive boundary quadrature, tailored to the actual singularities and the desired precision for the calculations. This all needs to be served up as a convenient container for consumption by an MPS eigensolver.

# Why now: a third use, and the evidence for it

3. **A posteriori error bounds.** The Moler-Payne inclusion bound needs `||u||_{L2(Omega)}` for a
   Helmholtz-exact `u`, and the Rellich identity below gives it from boundary data alone.

`notes/moler_payne_vs_tension.md` measured what that would buy, on `rect(1, 1/phi)` where an exact
interior rule is available as a reference. The shipped default (the penalty at `prec = 1e-8`)
reports a bar of 1-2e-10 on errors that are really 6e-14 to 2.5e-12: its bar is built from the
penalty's floor, and the floor is the penalty term, not the eigenvalue's error. The Moler-Payne
bound held on all 15 rows, is about 10x tighter than that bar, and is within 2-6x of the true error
on a penalty with `tau` near rounding. Three consequences, each a reason to build this layer before
more regularization work:

* **Certification no longer depends on the regularizer.** The bound reads only `u`, so `tau` can be
  chosen for search stability and the reported precision comes from the bound. That removes the
  squeeze that leaves an order-1 penalty no feasible `tau` at 1e-14
  (`notes/penalty_prec_limit_explained.md`) as a problem for REPORTING.
* **The planner gets its stopping signal.** "Work harder until the answer is reliable" needs a
  per-eigenvalue attained precision to decide when to enrich a basis.
* **Clusters get a Gram matrix.** One function's bound does not settle a cluster; the bilinear form
  of the same identities does.

# The identities

For `u` with `Delta u + lam u = 0` in `Omega` -- every MPS candidate, at every `lam`, with NO
condition on `dOmega` -- and `x` measured from any origin `x0`, in two dimensions

    div[ (x . grad u) grad u  -  1/2 x |grad u|^2  +  1/2 lam x u^2 ]  =  lam u^2,

so the interior norm is a boundary integral of the Cauchy data:

    lam ||u||^2_{L2(Omega)} = oint [ (x . grad u) d_n u  -  1/2 (x . n) |grad u|^2
                                     +  1/2 lam (x . n) u^2 ] ds.

Derived here 2026-10-05, and CHECKED numerically on `rect(1, 1/phi)` at `lam = 50`, away from
any eigenvalue, for a Helmholtz-exact `u` far from zero on the boundary (six random plane waves):
the boundary integral matches an exact interior tensor rule to 3e-15, independently of the origin
`x0` (three origins, identical to all printed digits). Written in normal and tangential
derivatives (`x . grad u = r_N u_N + r_T u_T`, `|grad u|^2 = u_N^2 + u_T^2`) it is exactly lappy's
"master identity" (`lappy/docs/rellich.md:27`):

    ||u||^2 = (1/2 lam) oint r_N [u_N^2 - u_T^2 + lam u^2] ds  +  (1/2 lam) oint r_T [2 u_T u_N] ds,

with `r = x - x0`, `r_N = r . N`, `r_T = r . T`. lappy then DROPPED the `r_T` term and the
boundary-data terms, which vanish only for an exact eigenfunction; `chladni` keeps the full form,
so the norm is exact for every candidate and the identity needs no boundary condition at all.
Not yet checked at a corner. Where `u = 0` on
`dOmega` this is the familiar `||u||^2 = (1 / 2 lam) oint (x . n) (d_n u)^2 ds`, but the general form
is the one needed: a candidate is not zero on the boundary, and the bound is ABOUT how far from
zero it is. It is exact for Helmholtz-exact `u`, so it inherits no error from the approximation
quality of `u` -- only from the quadrature.

The Gram matrix of candidates at one `lam` is the symmetric bilinear form obtained by polarizing
the right-hand side. For two candidates at DIFFERENT `lam_i != lam_j`, Green's identity gives
`(lam_i - lam_j) int u_i u_j = oint (u_i d_n u_j - u_j d_n u_i) ds`, which divides by a small
number inside a cluster -- so it is never used: a cluster's members are re-solved at one `lam`
first (Wiring, below).

**Integrability at corners.** Near a corner where `u ~ r^nu`, `|grad u|^2 ~ r^(2 nu - 2)`, which is
integrable for `nu > 0` and singular for `nu < 1` -- exactly the re-entrant corners, where
`nu = pi / alpha` is in `(1/2, 1)` for Dirichlet. That is the `r^(2 nu - 2)` weight `Quadrature`'s
docstring already names, and the reason `weight_family` exists.

# The design

The plan of record is lappy's measured design (`lappy/eigfun_integrals.py`, `lappy/quad.py`,
`lappy/benchmarks/reference/certify.py`; docs `corner_quadrature.tex`,
`corner_integrand_dirichlet.tex`), keeping what lappy measured to work, dropping what it measured
not to, and closing two gaps it left (the full identity above, and the cancellation check below).

## What lappy measured, and this layer adopts

1. **Evaluate first, sandwich never.** Contract the basis with the coefficient vectors at the
   nodes, THEN integrate an `m x m` block (`m` the cluster size). A basis-sized Gram was off by
   ~1e-4 on a basis of condition 1e16, and corner rules lose 2-4 orders on columns centred at OTHER
   corners, so no node set serves a basis-sized Gram.
2. **Corner rules are substitution nodes with fitted weights -- NOT geometric grading.** Nodes
   from Gauss-Jacobi in `t` with `r = t^(1/nu)` (lappy `cornerjacgauss`); weights refit by least
   squares on the actual exponent set, `{gamma + j nu + 2q}` on a straight edge and
   `{gamma + j nu + m}` on a curved one, with `n_exp = order // 2` (the square system reaches
   condition 1e19). Kress sigmoid grading was retired: its tau floor truncates the integral (a
   2.2e-7 plateau from 565 to 6790 nodes). `chladni`'s own `grading="corner"` puts about two nodes
   on its 1/256 corner panel, and is not a substitute either.
3. **Singular is not re-entrant.** A corner needs the corner rule when `gamma = 2 nu - 2` is not
   an integer AND plain Gauss misses the target on `r^gamma`. A 135-degree corner (`nu = 4/3`,
   `r^(2/3)`) needs it; giving it to every non-integer `nu` made some corners worse. `nu` must be
   EXACT: 0.667 for 2/3 costs about ten digits.
4. **Size from closed-form model integrands; certify by refinement.** Per panel, the order is the
   largest of an oscillation requirement (`k = 2 sqrt(lam_max) L`, since a product of
   eigenfunctions oscillates at twice the wavenumber), a corner model (scanned, taking the
   ARGMIN -- accuracy is not monotone in order), and a geometry self-check
   (`oint r . N ds = 2 |Omega|`). A corner panel is at most 0.9x the corner's clearance; dropping
   that cap cost up to 12 orders. The sizing is a model bound. The certificate is `verify`: shrink
   the corner panels by `2^depth`, difference the Gram; it tracked the true error within 6.3x.
5. **Two corner families, chosen by the WEIGHT's parity.** `'even'` (substitution exponent `nu`)
   for the Rellich and Gram integrals; `'integer'` (exponent 1/2) for a Hadamard weight
   `V . n = O(r)` at a moving corner. Not interchangeable in either direction (L-shape: 4.0e-15
   against 1.4e-5 on `r^0`; 2.5e-9 against 1.5e-13 on `r^1`).
6. **One primitive.** `weighted_integral(cd, kernel, w)`, kernels `uv`, `NN`, `TT` and the cross
   term; every identity is a fixed combination of calls.
7. **The origin.** Default `x0` at the singular corner of smallest `nu`, which zeroes `r_N` on both
   of its straight edges. The spread over several `x0` is a DETECTOR, not an error bar (it carries
   the eigenvalue's own error). Never put `x0` on the diagonal of a 3pi/2 corner, where the corner
   contribution cancels exactly; report the corner/edge split.
8. **Löwdin, not Cholesky,** with a deficiency guard. Only the cluster's spectrum is promised,
   never a particular orthonormal basis of it.
9. **Moler-Payne,** `eps = sqrt|Omega| max |u|_dOmega / ||u||`, inclusion
   `|lam - lam_k| <= lam eps / (1 - eps)`. The maximum from graded dense sampling, re-done at
   double density and inflated by the drift; the norm as a CONSERVATIVE under-estimate (the
   minimum over several `x0`, less their spread).

Limits inherited and NOT solved here: Zaremba (Dirichlet/Neumann) corners with `nu < 1`, where
`(d_N u)^2 ~ r^(nu - 2)` and the identity diverges (refused); pure-Neumann singular corners, whose
`uv` and `TT` kernels want different node sets (smooth fallback, reported); `nu -> 1/2`, where the
interpolatory rule tops out near 1e-10; eccentric curved boundaries.

## Modules

`chladni.integrals` depends on `geometry` and `helmholtz`; `mps` depends on it, never the reverse,
so `douse` can take the Hadamard half without the solver.

    rules.py       Gauss-Jacobi; the corner substitution and interpolatory rules; corner
                   exponent sets; closed-form model errors; order selection (corner, smooth,
                   geometry); the order cap (innermost node * scale > 1e-9)
    plan.py        corner classification from `Domain.corners` (omega, bc, clearance, segments):
                   singular / integer-gamma / demoted / refused; panels through
                   `Domain.bdry_quad(panels=...)`; `IntegralQuad` = the rule plus family, x0,
                   sizing precision, shortfalls, corners
    cauchy.py      `CauchyData(quad, lam, u, u_n, u_t)`, evaluate-first through
                   `Sampling.values_and_grad` on FAMILY coefficients (`MPSPencil.family_coef`),
                   and the cancellation ratio from the same data
    identities.py  `weighted_integral`; `gram` (the full identity); norms; `lowdin`
    certify.py     the boundary maximum; `moler_payne`; `verify`
    hadamard.py    (M5) normal velocity, vertex velocity, shape derivative

## Precision

The quadrature is sized to `integral_prec = max(0.1 * prec, 1e-14)`, overridable. The bound
reuses it and is never the binding constraint (below). A corner that cannot reach the target is
reported in `shortfalls`, with its achievable precision, never silently.

## Wiring

`MPSEigensolver.search_interval` post-processes the levels -- it holds the pencil and the domain,
and `search.solve` sees only an `Objective`. Per level: `coef(lam, mult)`, Cauchy data, Gram,
Löwdin, then a bound per column. A level gains `mp_eps`, `mp_bound` (absolute), the norm's
quadrature error, the cancellation ratio and a `bound_reason`; `Spectrum.bound` becomes
`min(delta_res, mp_bound)` -- both are valid upper bounds, so the tighter one is honest -- and
stays `delta_res` where the bound is refused. Members of a merged level are re-solved at the
level's one `lam` before their Gram is formed, so no Gram ever divides by `lam_i - lam_j`.
`MPSEigensolver.eigenfunctions(level)` returns the orthonormal family coefficients and the rule.

## Milestones

    M0  seams and docs: plan tiling check, Quadrature family semantics, family_coef, reference
        gradients, this document                                                  [done]
    M1  rules.py, with lappy's test_quad contracts                                [done]
    M2  plan, Cauchy data, the identity: non-eigen functions against exact interior rules;
        exact norms of sector (singular), rect, polyomino, disk eigenfunctions;
        a real L-shape candidate against a graded oracle                          [done]
    M3  Moler-Payne on every level, wired into the solver                         [done]
    M4  clusters: Gram, Löwdin, orthonormal eigenfunctions                        [done]
    M5  Hadamard (after M4)
    M6  later: Neumann and mixed, curved boundaries, a guaranteed maximum, planner integration

## What M2 measured (2026-10-05)

* **The full identity holds to rounding on functions that are NOT eigenfunctions**: plane waves on
  a rectangle at `lam = 50` (Gram, off-diagonal included, to 1e-13 at three origins), and a random
  combination of the re-entrant corner's Fourier-Bessel columns on the L-shape (1e-12 against a
  graded interior oracle, three origins).
* **Exact sector norms meet lappy's leg-1 bars** at every apex angle tried (1.25pi to 1.9pi), and
  the rectangle, L-polyomino and disk norms meet 1e-13.
* **The clearance cap is load-bearing** on the 1 x 16 strip: machine precision with it, worse
  than 1e-6 without.
* **`SMOOTH_SAFETY = 2`, not lappy's 1.** On the first L-shape eigenfunction from an 80-column
  corner basis, the smooth panels sized at 1x missed a 1e-13 target by 180x (1.8e-11) while the
  corner panel was exact; 2x gives 3.3e-15 for 19% more nodes (`plan.SMOOTH_SAFETY`).
* **Cancellation must be measured against the INTERIOR size of `u`.** Against its boundary rms
  it read 1e15 at an exact eigenvalue -- a good Dirichlet candidate is near zero on the boundary
  by design. Against `||u|| / sqrt|Omega|` the same candidate reads about 2.
* **Not yet measured: a candidate with smooth columns beside the corner ones.** The L-shape recipe
  basis is pure corner Fourier-Bessel, so its candidate is inside the corner family and the
  cross-term question of `plan.py`'s docstring did not arise. Off the eigenvalue the norm error
  tracked the tension (1e-9 at tension 2e-9, 1e-6 at 2e-6), which is the boundary-data size, not
  the quadrature: a Moler-Payne bound there is large anyway.

## What M3 measured (2026-10-05)

* **The bound covers the true error on every level tried, and tightly.** On the L-shape (80 corner
  Fourier-Bessel columns), candidates taken at KNOWN offsets of 1e-11 to 1e-5 from the first and
  third eigenvalues: the bound is 3.2x the error at the singular first, 3.5x at the smooth third,
  at every offset. At the tabulated eigenvalue it bottoms out near 1e-14 relative -- the table's
  own 14 digits, and above the `eps * cancellation` floor it must respect. On `rect(1, 1/phi)` it
  reproduces the notes' table through the Rellich norm: valid on all 15 rows.
* **Through the solver**, every level of a five-level L-shape solve carries a finite bound that
  covers its error against the table, and `Spectrum.bound` is `min(delta_res, mp_bound)`.
  `Level.attained_prec` and the diagnostics' `unmet` follow the tighter bar.
* **Cancellation enters the bound as a floor, not a refusal**: the sampled maximum is raised by
  `4 eps` times the largest term behind it, so a bound below float64's own reach cannot be
  reported. Measured cancellation on real candidates: 2 to 3.
* **Refused, with a reason**: any non-Dirichlet boundary (the inclusion needs `u - v` to vanish on
  all of it), and `eps >= 1`.

## What M4 measured (2026-10-05)

* **Clusters come back orthonormal against INTERIOR oracles**, not the identity they were formed
  with: the unit square's exact double at `5 pi^2` (tensor Gauss, `|G - I| < 1e-10`) and the
  disk's `m = 1` pair (polar rule, `< 1e-9`). `MPSEigensolver.eigenfunctions(level)` forms them at
  the level's one `lam` and reports `gram_residual` at a second Rellich origin; a deficient Gram
  comes back normalized column by column and FLAGGED.
* **"Sandwich never" did not reproduce, and the reason is the scaling.** lappy measured a
  basis-sized Gram `c^T G c` off by ~1e-4 on a basis of condition 1e16 with raw coefficients. Here,
  on the L-shape corner basis (Gram condition 1e8) and on plane-wave bases from 20 to 120 columns
  under both regularizations, the sandwich and evaluate-first agree to 1e-14: both regularizers
  keep the minimizing coefficients bounded (cancellation 2-4 throughout), so there is nothing for
  the sandwich to amplify. Evaluate-first stays -- it forms an `m x m` block instead of `n x n`, and
  lappy's OTHER reason, corner rules losing 2-4 orders on columns centred at other corners, is not
  yet measured here (a multi-corner basis) -- but no test claims the sandwich fails.
* Seen in passing, not this layer's: with 16 plane-wave directions the unit square's search
  returns NO level in [45, 55] and does not decline (24 directions find the double). That is the
  starved-basis loss `TODO.md` section 4 already records.

# Accuracy: the bound needs few digits, orthonormalization needs many

**The bound is never the binding constraint.** It is an inequality whose smallness lives in
`max |u|`, not in the norm: an error of 1% in `||u||` moves a 1e-12 bound to 1.01e-12. So the bound
needs only a few RELIABLE digits of the norm, which is why the conservative under-estimate (item 9)
is affordable.

**Cancellation is a precondition, and it is checkable.** `u` is evaluated as `sum_j c_j psi_j`, so
its rounding error is about `eps * sum_j |c_j| |psi_j|`, and a bound below that is not a bound. At
the eigenvalues measured the candidates are benign -- `max sum |c_j||psi_j| / rms(u)` is 3 to 4 on
the default pencil, the near-rounding penalty and the threshold alike -- but at `lam = 50`, off any
eigenvalue, the minimizing vector cancels by 1.8e13, and there `u` is rounding: the interior rule
moved 1e-4 under refinement and the Rellich value moved with the origin. So the layer computes the
cancellation ratio from the same Cauchy data and REFUSES a bound below `eps` times it.

Orthonormal eigenfunctions are different: Hadamard derivatives and a cluster's Gram matrix need the
integrals to near `prec`, which is what the corner rules and the sizing of items 2-4 deliver.

# Validation oracles

Interior cubature is a test ORACLE here and never a production path -- `notes/solver_features.md`
A6 argues the interior block wants non-degeneracy, not accuracy, and that anything that must be
accurate belongs on these boundary integrals.

* **Rectangle:** tensor Gauss-Legendre, exact to rounding for plane-wave candidates. The reference
  table in `notes/moler_payne_vs_tension.md` is what the boundary version must reproduce.
* **Disk:** polar tensor rule; smooth boundary, a check on the curved-boundary path even before it
  is in scope.
* **L-shape:** three unit squares, tensor rules geometrically graded toward the re-entrant corner.
  An oracle with a singular corner (`nu = 2/3`) for NON-eigen Helmholtz functions.
* **Exact eigenfunctions,** where `||u||` is known in closed form (`chladni.reference`):
  `sector_eigfun` -- the sharpest, one corner-adapted column, singular at a re-entrant apex; then
  `rect_eigfun`, `polyomino_eigfun` (smooth at its re-entrant corners: a control, not a corner
  test) and `disk_eigfun`, each with an analytic gradient. There is NO equilateral-triangle
  eigenfunction in `reference` -- an earlier draft of this note claimed one -- and none for the
  L-shape, whose eigenvalues are a 14-digit table.
* **References are closed-form or Beta-function integrals only.** lappy found `mpmath.quad` wrong
  by 4e-2 at 40 digits near the slit limit, producing fake plateaus.

The acceptance test is the bound against the known error, on the rectangle, disk, sector and
L-shape (against `L_shape_eigs`): valid on every level, and within a stated factor of the error.

# Open questions

* **A guaranteed maximum.** lappy's sampled maximum with drift inflation is an estimate, good to
  0.16 digits against the full bound over 150 cells, but not a guarantee. A guarantee needs either
  a bound between nodes (per panel, from the tangential derivative or a Chebyshev-tail estimate) or
  the `L2(dOmega)` form of the inclusion bound. Which form, and its constant, is to be sourced, not
  derived. M6.
* **The subspace form of the bound** for a cluster: lappy certified each column separately and
  reported the worst. Whether a sharper multi-function bound is worth having is open.
* **Cost.** `coef` needs a separate-mode GSVD per level, and the Cauchy data one evaluation of the
  basis on the quadrature nodes -- per reported level, not per search step. Unmeasured.
