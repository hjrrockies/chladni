# Why the Tikhonov penalty can't reach `prec = 1e-14`

*A self-contained explanation for readers who don't work on the method of particular solutions (MPS).
The detailed measurements are in `notes/smooth_filter.md`, sections 9 and 10, and the derivations are
in `notes/tikhonov_derivations.pdf`. Written 2026-09-23.*

**Short version.** The penalty parameter `tau` has two jobs, and they pull in opposite directions.
If `tau` is too large, the solver can't tell eigenvalues apart to the requested precision. If `tau`
is too small, rounding error takes over the objective function. The upper limit on `tau` shrinks
as the requested precision `prec` gets tighter. The lower limit is set by machine epsilon, so it
doesn't move. At `prec = 1e-12` there is still a narrow range of `tau` that works. At `1e-14` the
range is empty.

---

## 1. The problem being solved

We want the eigenvalues `lam` of the Laplacian on a planar domain, for example with zero
(Dirichlet) boundary conditions:

    -Laplacian(u) = lam * u   inside,       u = 0   on the boundary.

MPS starts with a set of functions `phi_1, ..., phi_n`, each of which solves `-Laplacian(phi) = lam * phi`
exactly for the current `lam`. Examples are Bessel functions and plane waves. Every linear
combination `u = sum c_j phi_j` also solves the PDE, so the only question is whether some
combination also vanishes on the boundary. That can only happen when `lam` is an eigenvalue.

To test this numerically, evaluate the basis at points on the boundary and at points inside the
domain. This gives two matrices, `A_B` (boundary) and `A_I` (interior), with one column per basis
function. Then compute the **tension**:

    sigma(lam) = min over c  of   ||A_B c|| / ||A_I c||

The numerator measures how far `u` is from zero on the boundary. The denominator stops the
minimiser from choosing `u = 0`. `sigma` is the smallest generalized singular value of the pair
`(A_B, A_I)`. Near an eigenvalue it falls to (nearly) zero, and elsewhere it stays away from zero.
Finding eigenvalues therefore means finding the zeros, or minima, of a one-variable function:

    sigma
      |\                 /\                /
      | \               /  \              /
      |  \             /    \            /
      |   \           /      \          /
      |    \         /        \        /
      +-----V-------------------V-------------> lam
          lam_1               lam_2

Each eigenvalue sits at the bottom of a V-shaped "well". The search algorithm uses the slopes of
the walls to locate the bottoms.

## 2. Why regularization is needed at all

Good bases are **over-complete**: many columns are nearly linear combinations of the others. The
singular values of the stacked matrix `A = [A_B; A_I]` decrease smoothly until they reach machine
epsilon (about 1e-16, relative), and after that they are pure rounding noise.

The directions at that rounding floor carry no information. The minimiser can still use them,
though, because it can pick a coefficient vector `c` whose boundary values are small only because
of cancellation in rounding. That drives `sigma` spuriously close to zero at values of `lam` that
are not eigenvalues. Those directions have to be kept out.

**The standard fix is truncation.** Take the SVD of `A`, discard directions whose singular values
fall below a cutoff (`rtol * s_max`), and solve the problem in what remains. This works, but the
number of directions kept is an integer computed at every `lam`. Singular values sit near the cutoff,
so rounding decides whether each one is kept. As a result, the rank flips back and forth between
neighbouring values of `lam`, and each flip makes `sigma` jump. We measured this "flicker" as an
irreducible relative noise of about 0.4% in `sigma`, at every scale down to `1e-10` in `lam`.
It also produced a wrong answer on one BLAS library and a correct one on another.

## 3. The penalty (Tikhonov) alternative

Instead of removing directions, **penalize** them. Add a term to the numerator:

    sigma_tau(lam) = min over c  of   sqrt( ||A_B c||^2 + tau^2 ||c||^2 ) / ||A_I c||

The columns are normalized to unit length first, so `||c||` has a meaning. In the SVD basis of
`A`, the penalty acts on each direction `j` with a smooth filter factor

    f_j = s_j^2 / (s_j^2 + tau^2)

This factor is close to 1 when `s_j` is much larger than `tau` (the direction is left alone) and
close to 0 when `s_j` is much smaller than `tau` (the direction is effectively removed). The switch
between the two happens over about one decade, centered at `s_j = tau`. Nothing is an integer, so
nothing can flip, and `sigma_tau` is a smooth function of `lam`.

(The code option is called `tikhonov_qr` because of how it is computed. It takes a QR factorization
of `[A_B; tau*I; A_I]` and passes the orthonormal factor to the GSVD. In exact arithmetic this is the
same problem. In floating point it stops LAPACK from making its own internal rank decision, which
would bring the flicker back.)

The penalty has two costs, and both are controlled by `tau`.

## 4. Cost 1: the wells no longer reach zero (this caps `tau` from above)

The penalty charges **every** direction, including the ones that represent the true eigenfunction.
At a true eigenvalue with eigenfunction coefficients `c*`, `||A_B c*||` is essentially zero, but
the penalty term `tau ||c*||` is not. This gives a proven bound:

    sigma_tau(lam_true)  <=  sqrt( eps_basis^2 + tau^2 * rho^2 ),      rho = ||c*|| / ||A_I c*||

Here `eps_basis` is how well the basis can represent the eigenfunction. In measurements, the bottom
of each well (the **floor**) is about `1.4 * tau`:

    sigma
      |\          /
      | \        /
      |  \      /
      |   \____/   <- floor ~ 1.4 * tau, instead of ~0
      +-------------> lam

A flat bottom makes the eigenvalue hard to locate. The solver can locate the minimum only to within
roughly the floor. When two eigenvalues are closer together than that, their wells blur into one, or
one well hides the other, and an eigenvalue is lost. For the solver to deliver relative precision
`prec`, and in particular to separate eigenvalues that are only about `prec` apart, the floor has to
be a small fraction of `prec`:

    tau  <~  C_hi * prec

On our test problems at `prec = 1e-12`, `C_hi` came out at about **0.05**. Larger `tau` made the
solver start dropping eigenvalues from close clusters.

**Truncation doesn't pay this cost.** It gives zero penalty to the directions it keeps, and those
include the eigenfunction. Its floor is set by the basis alone, about 1.6e-15, and doesn't depend on
the cutoff.

## 5. Cost 2: rounding noise comes back as `tau` shrinks (this caps `tau` from below)

The only thing keeping the minimiser away from the garbage directions is the penalty, which has size
`tau`. The garbage in those directions has size about `eps * ||A||`, because each computation is
exact for some perturbation of that size. With 126 unit-norm columns, `eps * ||A||` is about
`2.2e-16 * sqrt(126)`, or **2.5e-15**. As `tau` falls toward that value, the minimiser leans more
and more on directions whose content is rounding error, and `sigma_tau` becomes noisy.

The relative noise in `sigma_tau`, measured on a test rectangle:

| `tau`   | floor (well bottom) | relative noise in `sigma` |
|---------|---------------------|---------------------------|
| 1.7e-11 | 2.3e-11             | 5e-8                      |
| 1.7e-13 | 2.4e-13             | 7e-6                      |
| 1.7e-14 | 2.4e-14             | 6e-4                      |
| 3.4e-15 | 5.2e-15             | 1e-2                      |
| (truncation) | 1.6e-15        | 1e-3 (flicker)            |

Two details matter here:

* **The noise is relative, not absolute.** It is the same fraction of `sigma` all the way down a
  well's wall. Moving closer to the eigenvalue doesn't reduce it.
* **The search needs clean slopes.** It estimates wall slopes from differences between nearby
  samples. With 1e-2 relative noise those estimates never settle, so the solver stops and reports
  that it can't certify an answer (a *decline*). That behavior is correct but gives no result.

That gives a lower limit:

    tau  >~  tau_min,       with tau_min ~ a few * eps * ||A||    (independent of prec!)

At `prec = 1e-12` the measured value was `tau_min` ≈ **7e-15**, about 3x `eps * ||A||`. Smaller
`tau` produced declines.

This is a heuristic, not a theorem. It isn't proved, but the measurements fit it consistently.

## 6. The squeeze

Putting the two limits together:

    tau_min  (≈ 7e-15, fixed by machine epsilon)   <~   tau   <~   0.05 * prec   (shrinks with prec)

A usable `tau` exists only while `0.05 * prec > 7e-15`, which means `prec` greater than roughly
**1e-13**. The table below compares that prediction with measurements. Each "cell" is a test case
with a cluster of closely spaced eigenvalues, and a cell counts as right only if every eigenvalue in
the cluster is found:

| `prec` | allowed `tau` range (predicted) | measured                                                                      |
|--------|---------------------------------|-------------------------------------------------------------------------------|
| 1e-12  | [7e-15, 5e-14], about 7x wide   | `tau` in [7e-15, 5e-14]: **12 of 12 right**, the same as truncation           |
| 1e-14  | [7e-15, 5e-16]: **empty**       | `tau` ≤ 3e-15: mostly declines. `tau` ≥ 1e-14: **every cell wrong**. No gap between them. |

At 1e-14 the two limits have crossed. Any `tau` small enough to resolve the eigenvalues is too small
to stay out of the rounding noise.

**The underlying reason is that one parameter controls two separate things.** An order-1 penalty
uses the same `tau` to shield the garbage directions and to charge the eigenfunction. Truncation
treats those two groups differently: it applies an infinite penalty to garbage and zero penalty to
signal. That's why truncation keeps a basis-limited floor at any cutoff, although it pays for this
with the flicker.

## 7. A separate, avoidable problem: the current `tau` rule

The shipped code doesn't choose `tau` inside the allowed range. It uses the rule
`rtol = sqrt(prec * eps)`, which was designed for truncation, where the floor doesn't depend on the
cutoff. For a penalty that rule gives `tau ~ sqrt(n * prec * eps)`, so `floor / prec` grows like
`1 / sqrt(prec)`:

| `prec` | floor / `prec` under the shipped rule |
|--------|---------------------------------------|
| 1e-8   | ~0.002                                |
| 1e-10  | ~0.02                                 |
| 1e-12  | 0.1 – 0.24                            |
| 1e-14  | 1.0 – 2.4                             |

At `prec = 1e-12` this puts `tau` above the allowed range, and 4 of 12 cells are wrong. At `1e-14`
**every cell is wrong, and the solver gives no warning**, which is the worst possible outcome. A rule
of the form `tau = C * prec` would fix the 1e-12 case. Nothing fixes 1e-14, for the reasons in
section 6.

## 8. Ways out (not settled)

* **A second-order penalty.** Replace the per-direction weight `tau / s_j` with `(tau / s_j)^2`.
  This is built from QR factors, with no SVD and no rank decision. It is almost free on
  well-conditioned directions, so the floor can sit well below `tau`, and it suppresses garbage
  directions much more strongly. This separates the two jobs. First measurements at `prec = 1e-14`
  are promising: on one test set it found all 11 of 11 where truncation and the order-1 penalty
  declined every case. It is currently blocked by bookkeeping problems in the search, not by the
  penalty.
* **Smaller bases.** The garbage directions exist because the bases are over-complete. A basis
  sized to what the problem actually needs might make truncation stable enough that no penalty is
  required.

## Caveats

* The noise mechanism in section 5 is a heuristic that fits the data, not a proof. The floor bound
  in section 4 is proved.
* The constants 0.05 and 7e-15 come from one family of test domains (rectangles with clustered
  eigenvalues). The lower limit should depend on `n` through `eps * ||A||`, which hasn't been
  checked across basis sizes.
* A separate problem is still unexplained. The solver occasionally loses one eigenvalue from a
  cluster even when the floor is 4–12x smaller than the gap between eigenvalues. This has happened
  at several settings, including one wrong cell at `prec = 1e-10` with `tau = 0.02 * prec`. It looks
  like a bug in the search, not in the regularization. Until it's understood, the boundaries of the
  allowed range in section 6 could be slightly misplaced, although the fact that the range closes
  at 1e-14 doesn't depend on it.
