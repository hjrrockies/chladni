*Status: scope note, with a DRAFT scope for the first slice (2026-10-05). The layer is not
written yet.*

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
`x0` (three origins, identical to all printed digits). Not yet checked against a published form,
nor at a corner. Where `u = 0` on
`dOmega` this is the familiar `||u||^2 = (1 / 2 lam) oint (x . n) (d_n u)^2 ds`, but the general form
is the one needed: a candidate is not zero on the boundary, and the bound is ABOUT how far from
zero it is. It is exact for Helmholtz-exact `u`, so it inherits no error from the approximation
quality of `u` -- only from the quadrature.

The Gram matrix of candidates at one `lam` is the symmetric bilinear form obtained by polarizing
the right-hand side. For two candidates at DIFFERENT `lam_i != lam_j`, Green's identity gives
`(lam_i - lam_j) int u_i u_j = oint (u_i d_n u_j - u_j d_n u_i) ds`, which divides by a small
number inside a cluster -- see the open questions.

**Integrability at corners.** Near a corner where `u ~ r^nu`, `|grad u|^2 ~ r^(2 nu - 2)`, which is
integrable for `nu > 0` and singular for `nu < 1` -- exactly the re-entrant corners, where
`nu = pi / alpha` is in `(1/2, 1)` for Dirichlet. That is the `r^(2 nu - 2)` weight `Quadrature`'s
docstring already names, and the reason `weight_family` exists.

# The first slice (DRAFT)

**Goal: a Moler-Payne bound on every reported Dirichlet level on polygons, validated against an
exact interior rule.** In order:

1. **Cauchy data at quadrature nodes.** For a coefficient vector `x` from `MPSPencil.coef` at
   `lam`: `u`, `d_n u` and `d_t u` at the nodes of a `Quadrature`, through the families'
   `values_and_grad`, with the pencil's equilibration scale divided out -- the same composition
   `notes/moler_payne_vs_tension.py` uses (`u_at`). A container for it, since the norm, the Gram
   matrix and Hadamard all consume the same data.

2. **Composite Gauss-Legendre on polygon edges, graded toward corners.** Geometric panels toward
   each vertex, with the grading driven by the basis's own `singular_exponents()`
   (`helmholtz._core.SingularStructure`) and delivered through `Domain.bdry_quad(panels=...)`, the
   seam built for it. This replaces `grading="corner"`, which `docs/design/geometry.md:449` calls
   "a placeholder, not a design". Gauss-Jacobi panels at singular corners are the refinement,
   not the first step: see "Accuracy" below.

3. **The Rellich norm** of one candidate, from 1 and 2.

4. **The bound**, `sqrt(A) max |u| / ||u||`, attached to each `Level` beside `delta_res`, as an
   attained precision the caller and the planner can read. Reported, not acted on: whether the
   search should stop on it is the planner's decision.

5. **Then the Gram matrix and orthonormalization** for clusters, and the subspace form of the
   bound.

**Not in this slice:** Neumann and mixed conditions (the inclusion bound differs), curved
boundaries (`Quadrature` supports them; the grading rule is polygon-specific), and Hadamard
derivatives.

# Accuracy: the bound needs few digits, orthonormalization needs many

**This is what makes the first slice cheap.** The bound is an inequality whose smallness lives in
`max |u|`, not in the norm: an error of 1% in `||u||` moves a 1e-12 bound to 1.01e-12. So for
step 4 the norm needs only a few RELIABLE digits, plus a safety factor -- robust convergence
matters, high order does not. Composite Gauss-Legendre with geometric grading converges on
`r^(2 nu - 2)` without knowing `nu` exactly.

**Cancellation is a precondition, and it is checkable.** `u` is evaluated as `sum_j c_j psi_j`, so
its rounding error is about `eps * sum_j |c_j| |psi_j|`, and a bound below that is not a bound. At
the eigenvalues measured the candidates are benign -- `max sum |c_j||psi_j| / rms(u)` is 3 to 4 on
the default pencil, the near-rounding penalty and the threshold alike -- but at `lam = 50`, off any
eigenvalue, the minimizing vector cancels by 1.8e13, and there `u` is rounding: the interior rule
moved 1e-4 under refinement and the Rellich value moved with the origin. So the layer computes the
cancellation ratio from the same Cauchy data and REFUSES a bound below `eps` times it.

Orthonormal eigenfunctions are different: Hadamard derivatives and a cluster's Gram matrix need the
integrals to near `prec`. That is where Gauss-Jacobi panels matched to each corner's exponent earn
their keep (the `'even'` and `'integer'` weight families lappy found "not interchangeable in either
direction"), and it belongs to step 5, not step 4. Tolerances are `prec`-aware either way, per
`CLAUDE.md`.

# Validation oracles

Interior cubature is a test ORACLE here and never a production path -- `notes/solver_features.md`
A6 argues the interior block wants non-degeneracy, not accuracy, and that anything that must be
accurate belongs on these boundary integrals.

* **Rectangle:** tensor Gauss-Legendre, exact to rounding for plane-wave candidates. The reference
  table in `notes/moler_payne_vs_tension.md` is what the boundary version must reproduce.
* **Disk:** polar tensor rule; smooth boundary, a check on the curved-boundary path even before it
  is in scope.
* **L-shape:** three unit squares, tensor rules geometrically graded toward the re-entrant corner.
  The first oracle with a singular corner (`nu = 2/3`), and the one that tests step 2's grading.
* **Exact eigenfunctions** of the rectangle, disk and equilateral triangle, where `||u||` is known
  in closed form.

The acceptance test is the bound against the known error, on the rectangle, disk, equilateral
triangle and L-shape: valid on every level, and within a stated factor of the error.

# Open questions

* **A guaranteed maximum.** `max |u|` over 8000 sampled points served for measurement. A bound
  needs either a guarantee between nodes (per panel, from the tangential derivative or a
  Chebyshev-tail estimate) or the `L2(dOmega)` form of the inclusion bound, which replaces the
  maximum with a boundary integral the same quadrature computes. Which form, and its constant, is
  to be sourced, not derived here.
* **Clusters.** Which subspace form of the inclusion bound to use, and how to form the Gram matrix
  of candidates located at slightly different `lam` without dividing by `lam_i - lam_j`. One
  option: re-solve all members at a common `lam` and use the Rellich bilinear form there.
* **Where the layer lives.** It needs geometry (panels), helmholtz (Cauchy data and exponents) and
  mps (coefficient vectors and equilibration scales). `chladni.mps` is the natural owner unless
  the Hadamard work, which is `douse`'s consumer, argues for its own package.
* **Cost.** `coef` needs a separate-mode GSVD per level, and the Cauchy data one evaluation of the
  basis on the quadrature nodes. Per reported level, not per search step, so likely small --
  unmeasured.
