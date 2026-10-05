# The eigenvalue is better than the tension says, and a Moler-Payne bound shows by how much

2026-10-05. Raised by two questions about the penalty regularization: is its floor at
`prec < 1e-8` an artifact of the penalty, and is the eigenvalue already more accurate than the
penalized tension predicts? Recorded because the answers argue for building the boundary-integral
layer (`docs/design/boundary_integrals.md`) ahead of further regularization work.

Reproduce with `notes/moler_payne_vs_tension.py posterior` and `... bound`. Every number here is
`rect(1, 1/phi)`, Dirichlet, 126 plane-wave columns (`rect_suite.family_A()[0]`): one domain, one
basis family, smooth eigenfunctions, and the family the penalty's reach constants were fitted on.

## 1. The penalty floor is the penalty's own term

`tikhonov_qr` minimizes `sigma_tau^2 = (||A_B x||^2 + tau^2 ||x||^2) / ||A_I x||^2` -- the penalty
is in the NUMERATOR -- so on the minimizing `x` the unpenalized ratio
`t(x) = ||A_B x|| / ||A_I x||` satisfies `t^2 = sigma_tau^2 - tau^2 ||x||^2 / ||A_I x||^2`, and can
only be lower. Recomposing `sigma_tau` from `MPSPencil.coef` reproduces the reported tension
exactly. At the first eigenvalue, `tau = 0.02 prec`:

    prec    tau      sigma_tau   penalty term   t(x)
    1e-8    2e-10    1.94e-10    1.91e-10       3.7e-11
    1e-10   2e-12    2.13e-12    2.11e-12       2.7e-13
    1e-12   2e-14    2.26e-14    2.26e-14       1.2e-15
    1e-14   2e-16    5.0e-16     3.8e-16        1.1e-15     <- below the reach: noise

At the eigenvalue 98-100% of the floor is the penalty term. But `t(x)` is not a free win while
`tau` is large: the penalized minimizer trades boundary residual for coefficient norm, so it is not
the best vector for the unpenalized ratio, and `t(x)` has a floor of its own, about `tau/5` to
`tau/8`, until it reaches rounding. At `tau = 2e-14` it does: `t(x)` is 1.2e-15 at the eigenvalue
and 1.72e-14 at an offset of 1e-14 relative, where the threshold reads 1.70e-14 -- the threshold's
V, from a pencil with no rank to flip. At `tau = 2e-16` both are noise: the wall slopes are wrong
already at an offset of 1e-6 (6.4e-7 and 8.5e-7 against a true 1.75e-6).

So `tau` has two jobs only if the penalized tension is also the reported quantity. Set `tau` from
the arithmetic (near `_REACH_ROUNDING * eps * sqrt(n)`, 7.5e-15 here) for noise control, search on
`sigma_tau`, and read `t(x)` for localization: the squeeze that leaves no feasible `tau` at 1e-14
(`notes/penalty_prec_limit_explained.md`) does not apply to that split. NOT measured on clusters,
where the single minimizing vector jumps between members and the posterior has to be a small
Rayleigh-Ritz on the lowest few penalized vectors.

## 2. Accuracy against the bar, and against a Moler-Payne bound

Five eigenvalues, each refined with `refine` on three pencils. The Moler-Payne bound used is the
simplest: for `u` Helmholtz-exact at `lam`, some true eigenvalue satisfies

    |lam_k - lam| / lam_k <= sqrt(A) max_{dOmega} |u| / ||u||_{L2(Omega)}.

`max|u|` is taken over 8000 boundary points and `||u||` from an 80 x 60 tensor Gauss-Legendre rule,
which is exact to rounding for these functions on a rectangle. Ranges over the five eigenvalues:

    pencil                         true err/lam      bar/lam          sigma           MP bound         MP / err
    default (penalty, prec 1e-8)   6e-14 - 2.5e-12   1.1 - 2.2e-10    ~2e-10          1.3 - 3.6e-11    14 - 250
    penalty, tau = 2e-14           7e-16 - 3e-14     2 - 6e-14        2.4e-14-1.2e-13 1.4e-14-1.6e-13  2 - 6
    threshold, SUITE_RTOL          <= 2e-16          ~1e-15           ~1e-15          2.5 - 4.9e-15    rounding

**The shipped default is two to three orders more accurate than it reports.** Its bar,
`delta_res`, is built from the floor and the wall slope, and the floor is the penalty term; the
minimum's LOCATION is barely biased by it.

**The bound held in all 15 rows**, is about 10x tighter than the bar at the default -- it certifies
about 1e-11 where the bar claims 1e-10 -- and is within 2-6x of the true error at `tau = 2e-14`. On
the threshold it bottoms out near 3e-15, which is the rounding in evaluating `u`.

**`t(x)` tracks the bound** (`MP / t` between 1 and 1.7) but is not rigorous: it uses the
collocation points for the maximum and a random interior sample for the norm. The bound is the
rigorous version of the same quantity.

## 3. What this argues for

* **Certification decoupled from regularization.** The bound depends only on `u`, not on how `x`
  was found, so `tau` can stay conservative for search stability and the reported precision comes
  from the bound rather than the floor.
* **The planner's stopping signal.** "Work harder until the answer is reliable" needs a
  per-eigenvalue ATTAINED precision to decide when to enrich a basis. The bound is that number.
* **Clusters.** One function's bound does not settle a cluster; that needs the Gram matrix of the
  approximate eigenfunctions, which the same boundary integrals deliver (Rellich-type identities).

What it needs, and the rectangle did not exercise, is the subject of
`docs/design/boundary_integrals.md`'s scope section: an accurate `||u||_{L2(Omega)}` on a general
domain, where the boundary Cauchy data is singular at re-entrant corners, and a maximum that is
guaranteed rather than sampled.

## 4. Two checks made while drafting the scope

* **The general Rellich identity** (`docs/design/boundary_integrals.md`, derived there) was checked
  at `lam = 50` on six random plane waves -- Helmholtz-exact, far from zero on the boundary: it
  matches the exact interior rule to 3e-15 and does not depend on the origin.
* **Cancellation in `u`.** `max sum_j |c_j||psi_j| / rms(u)` is 3 to 4 for every eigenvalue
  candidate above, on all three pencils, so their `u` is clean. The minimizing vector at
  `lam = 50`, off any eigenvalue, cancels by 1.8e13: its `u` is rounding, and both the interior
  rule and the Rellich integral are unstable there. A bound must check this ratio before it is
  believed.

## Not measured

Corners with singular Cauchy data, Fourier-Bessel or fundamental-solution bases, Neumann or mixed
conditions, clusters, and the cost of `coef` (a separate-mode GSVD) on the solve path. The
rectangle is the reference the general layer must reproduce, not evidence that it will.
