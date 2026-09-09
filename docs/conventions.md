# Conventions

`CLAUDE.md` asks for a file that records naming and notation decisions as they are made. This is
it. It is a ledger, not a spec: each entry is a choice that is already load-bearing somewhere in
the code, with the reason it went that way.

Layer-specific conventions stay in that layer's design doc when they only concern it. What lands
here is anything a second layer has to agree with.

---

# Arithmetic and array shapes

**Points are `complex128`**, public and internal. `.real`/`.imag` are free views, rotation is
multiplication by `exp(1j*theta)`, and a corner angle is one call to `np.angle`. `chladni` is
planar-only, so the usual objection to complex does not apply. Conversion to the `(N, 2)` form
happens only at the `shapely`/`matplotlib` seam, through `as_xy` and `from_xy`.

Vector algebra in that encoding, used everywhere:

    dot(a, b)    ==  a.real*b.real + a.imag*b.imag
    cross(a, b)  ==  a.real*b.imag - a.imag*b.real
    rotate(z, t) ==  z * np.exp(1j*t)
    outward N    == -1j * T

**Matrices of function values are `(n_pts, n_cols)`**, float64 real. Rows are evaluation points,
columns are basis functions. This is `lappy`'s convention and there is no reason to break it.

**Gradients are complex**: one `(n_pts, n_cols)` array holding `du_dx + 1j*du_dy`, not a third
axis. A directional derivative is then `dot(vec, grad)` in the form above, with `vec` the complex
unit direction -- which is why `Quadrature` can carry `normals` as complex and the derivative seam
needs no reshaping.

**Pairwise work broadcasts as (query, source)**: `flat[:, None]` against `source[None, :]`.

---

# Immutability

**Objects are immutable.** Every transform returns a new object; there are no setters. This is what
makes `prec` a fixed attribute rather than a cache key: an object's approximations can never be
stale, so `functools.cached_property` is correct and thread-safe, and `lappy`'s `self._x = None`
sentinel pattern is unnecessary. It also matters for `douse`, which generates domains in a loop and
runs solves across threads.

Two implementations of the same idea, chosen by role: `@dataclass(frozen=True)` for pure data
records, and the hand-rolled `Frozen` base with `_set` for objects with behaviour and
`cached_property`. Both live in `geometry/_core.py`.

The transform protocol is `translated` / `rotated` / `scaled` / `reversed` / `with_bc` /
`with_prec`, defined once on a base class over a private `_rebuilt` hook, with `+` and `*` as
sugar.

---

# Precision

**`prec` replaces `tol`.** It is the working *relative* precision, default `1e-8` (`PREC` in
`geometry/_core.py`), valid down to about `1e-15`. It is set at construction; a composite object
takes `min(prec)` over its parts; `.with_prec(p)` returns a new object. Every derived tolerance is
a documented function of `prec` -- there is no central tolerance helper, because the scale that
converts relative to absolute is different at every site and belongs in that site's comment.

**A reported precision is never a claim contradicted elsewhere in the same object.** Where a bound
on the achievable precision is cheap, compute it and warn when the request exceeds it, rather than
silently under-delivering: `Domain.achievable_prec` is the pattern.

`PrecisionWarning` has its own category so a `douse` grid can `simplefilter("ignore",
PrecisionWarning)` without a blanket filter that would hide everything else.

---

# The spectral parameter

**`k`, the wavenumber, is the parameter of the `helmholtz` layer.** The equation is

    laplacian(u) + k**2 * u = 0

and `k` is real and strictly positive. Every particular solution is literally a function of `k*r`,
validation is one rule instead of a `sqrt` of a possibly-negative line-search iterate, and the
Bessel accuracy regimes that layer has to reason about are stated in `(nu, k*r)`.

**`lam = k**2` is the parameter of the `mps` layer and of everything a user talks about.** Weyl's
law, `Domain.normalized`, and the word "eigenvalue" are all in `lam`.

**A parameter's NAME states its variable, and the conversion seam is `mps`/`helmholtz`.** An
earlier version of this entry said the conversion "happens at exactly one seam, `Eigenproblem`",
which was too strong; a later one said there was no seam at all, only a naming discipline, which
was too weak. The naming discipline is the rule and still holds: any function may work in
whichever variable suits it, provided its argument is *named* for that variable -- `k` or `lam`,
never an argument whose meaning has to be inferred.

What the discipline alone did not settle is where the conversion should FALL, and the measurement
that settles it is this: **`mps` and above is `lam`, `helmholtz` and below is `k`.** The seam is
where the meaning changes rather than where a class boundary happens to be. Below it, `k` is not
a parametrisation choice -- a Fourier-Bessel column IS a function of `k*r`, and its accuracy
bound `Sampling.accuracy(k)` is a statement about the Bessel kernel's own scale. Above it, `k`
buys nothing: a tension is not a function of `k*r`, Weyl's law and the user's vocabulary are in
`lam`, and eigenvalue SPACING is asymptotically uniform in `lam` and not in `k`, which is the
variable a minimizer's step control should see.

The arrangement this replaced had `MPSPencil` in `k` and `MPSEigensolver` in `lam`. That put the
conversion inside a single layer, between two classes a user walks between -- `MPSEigensolver`
HOLDS an `MPSPencil` and exposes it as the `pencil` property -- so the variable changed across a
hop that changed nothing about the numbers. Both now speak `lam`; `MPSPencil.matrices(lam)` calls
`assemble(bdry, interior, k)`, and that call is the seam.

**One clause above is now measured wrong, and the seam it was arguing for is not.** "Eigenvalue
SPACING is asymptotically uniform in `lam` and not in `k`, which is the variable a minimizer's
step control should see" -- true of the REFINEMENT and false of the COVERING, and the covering is
where the cost is. What a cover phase's step is controlled by is not the spacing but the WALL
SLOPE, and `c(lam) = C/sqrt(lam)`, so under `u = lam**p` the slope is constant for `p = 1/2` and
no other power. Measured spread of the wall slope across levels:

    transform   lam    lam^(3/4)   lam^(1/4)   log lam   sqrt(lam)
    spread      57%       27%         17%        32%        5%

A single global slope bound is conservative by the ratio of the largest local slope to the
smallest, so that 5% was worth 1.8x of a 2.6x total cost improvement -- more than every
algorithmic change in the search combined. In the other direction, fitting `sigma_1**2` to a
quadratic over identical physical points, the residual is smaller in `lam` in 17 of 17 cases
across six domains, median 5.1x, so the refinement is in `lam`.

**The seam does not move, and the two answers do not fight.** The candidate searcher is driven
through an objective stated in `lam` (`chladni.mps.objective`), and it transforms to `k` internally
through a view that shares the caller's evaluation cache (`chladni.mps.bracket`'s `_KView`),
returning brackets in the variable it was handed. An algorithm owning its own abscissa is not a
variable convention; it is an implementation detail that the naming discipline above already
covers, since every argument that crosses the boundary is named for what it carries.

The search is a scored candidate in `tests/` and not part of the package, so this entry describes
where the conversion falls whenever it is promoted, not a seam that exists in `src/` today.

Under `Domain.normalized(by=s)`: `lam -> lam * s**2`, equivalently `k -> k * s`.

---

# Boundary conditions

`'dir' -> 0.0`, `'neu' -> 1.0`, and any float in `[0, 1]` is the Robin parameter of

    (1 - bc)*u + bc*du_dN = 0

Kept from `lappy`, but validated once at construction rather than through a mutable property
setter. `bc_kind(bc)` gives back `'dir'`, `'neu'` or `'rob'`.

---

# Orientation

Outer boundaries are CCW, holes are CW. The outward unit normal is then `N = -1j*T` uniformly on
every boundary of every domain, with no special-casing for holes, and signed areas add so that
`sum(signed_area(b) for b in boundaries)` is the true area. This one convention is why holes cost
so little to support -- and why the corner angle convention below needs no hole branch either.

---

# Corner-local notation

Set at the `helmholtz` layer, but stated here because `geometry` produces the inputs and `mps`
consumes the outputs.

**`theta` is always an angular VARIABLE; a named particular angle gets its own letter.** So
`theta` is the polar angle, the argument of `rotated`, and `ArcSegment`'s `theta0`/`theta1`,
while a corner's interior angle is `omega` --- Grisvard's symbol, and the one the
corner-singularity literature uses. The reverse assignment was tried first and was wrong in
three ways at once: it left `Corner.theta` as the single site in the package where `theta`
did not mean a variable, it disagreed with `lappy` (which calls the wedge angle `phi`), and
it disagreed with the thesis this work belongs to.

| symbol | meaning |
|---|---|
| `omega` | interior angle at a corner, in `(0, 2*pi)`, measured inside Omega. Produced by `geometry`. |
| `nu` | base corner exponent, `pi/omega`. `lappy` called this `alpha`. |
| `l` | integer degree index of a corner-adapted family. |
| `mu` | the Bessel order of one column: `l*nu`, or `(l + 1/2)*nu` at a mixed-BC corner. |
| `m` | integer multipole order of `Y_m`. Reserved for that; never a degree index. |
| `r` | radius with respect to a named origin. Always. |
| `theta` | angular variable: the polar angle, a rotation, or an arc's parameter. Corner-locally, `0` on the **outgoing** ray, increasing CCW to `omega` on the incoming ray. |
| `beta` | the branch-cut direction in the same relative measure, in `(omega, 2*pi)`. |
| `sigma` | per-column log scale. A returned column is `psi * exp(-sigma)`. |

**`geometry` stops at `omega` and the BC pair. It may classify the ANGLE; it never treats
`pi/omega` as an exponent.** `Corner.is_reflection_closed` asks whether `omega = pi/m`, which is a
fact about the wedge -- the reflections in its two edges generate a finite group that tiles a
neighbourhood of the vertex -- and geometry needs it to place graded panels. It builds no
expansion, reads no offset and imports no basis. That line is load-bearing: `lappy` crossed it with
`geometry.pts_per_seg(domain, fb_basis)`, a layer-1 signature taking a layer-2 object, and the whole
`Corner` design exists so that function has nowhere to come back.

**"Singular", unqualified, means the eigenfunction is not smooth there. Only `helmholtz` may say
it.** The word carries three technical senses in this project and they were already colliding inside
single docstrings, so each of the other two is spelled out in full, every time:

| phrase | meaning | where |
|---|---|---|
| singular (bare), singular corner, `singular_exponents()` | the eigenfunction is non-smooth there | `helmholtz`, `mps` |
| **generalized** singular value / vector | the GSVD of the pencil `(A_B, A_I)` | `mps` |
| singular **weight** | a weight function `w` folded into a quadrature rule's `wts` | `geometry.quadrature` |

The corner sense is settled by `CornerModes.is_singular`: the orders `(l + offset)*nu` are not all
integers. It needs the offset, which comes from the BC pair through Grisvard's Robin rule, so it is
not available -- and not nameable -- one layer down. `Corner` therefore has no `is_singular` and no
`is_zaremba`; it offers `is_geometric`, `is_bc_junction` and `is_reflection_closed`, and a caller who
wants the singular set asks `CornerModes.from_corner(c)`. What geometry guarantees instead is a
bound: every singular corner satisfies `not is_reflection_closed or is_bc_junction`, which is why
`grading="corner"` can be built on geometry alone without ever under-resolving.

**The angle lift is anchored to `beta`, never to `0` or `-pi`.** The obvious
`np.angle((z - c.z)/c.t_out) % (2*np.pi)` is wrong: a point that should sit at `theta = 0` comes out
of `np.angle` as `-1e-17` and wraps to `6.283`, so the Dirichlet family evaluates to `0.11` on the
ray where it must vanish identically. Written correctly, that ray gives *structurally* exact `0.0`.

---

# Scaling

**A column's scale depends on `(family, column, k)` and never on the evaluation points.** A
returned column is `psi_j * exp(-sigma_j)`, and `sigma_j` is analytic. That is what lets a
downstream unit-norm rescale compose without double-counting: the log scales simply add, and a
coefficient vector means the same function on every point set.

The two concerns stay separate and are owned by different layers. `helmholtz` owns **entry-wise
relative accuracy** -- a column representable to full relative accuracy is a different object from
a column of underflowed zeros. `mps` owns **pencil conditioning** -- rescaling to exact unit norm
if the GSVD wants it. Neither impersonates the other.

**Unscaled coefficients are never materialized.** At a `pi/8` corner with `k*R = 10` the `sigma`
range spans about 1149 nats. An eigenfunction is always evaluated as `scaled_matrix @ c`, and the
recovery factor is handed back in the log.

---

# Documentation and testing

**Every non-obvious choice carries the number that forced it, at the code.** A comment that says a
threshold matters without saying what it was measured against is not finished.

**Order of work is spec, then tests, then implementation.** The spec then grows an
"As built: deltas from this spec" section rather than being rewritten. The geometry layer was built
that way and several real bugs were found by writing the test before the code.

**Ground truth is never regenerated from `chladni` output.** Closed forms, `scipy`, and `mpmath` at
high precision. Once the solver is trustworthy that becomes an option for features with no closed
form, but not before.

**Values from an INDEPENDENT solver are admissible, with their accuracy recorded as data.**
The rule above was written against self-reference and does not cover this case. `chladni.reference`
ships nine tabulated spectra computed by `lappy`, for domains with no closed form; they are not
`chladni` output, so they can be checked against. What makes them usable is that each carries
`digits`, `source` and per-entry `notes` rather than a prose caveat, because a docstring cannot be
read by the test that has to choose a tolerance. They range from 14 digits to under 3, and one
entry is known to be wrong and shipped flagged rather than silently corrected -- see
`docs/design/reference.md`.

This is the same rule as `Domain.achievable_prec` and `Sampling.accuracy`, applied to values
instead of to computations: a reported precision is never a claim contradicted elsewhere.

Tests are named `tests/test_<concept>.py`, one file per contract rather than one per module, each
with a module docstring saying what it defends. `EXACT = 1e-13` is for closed-form quantities that
must land at machine level; `PREC = 1e-8` is the library default for converged approximations.
Anything costing seconds is marked `slow` and deselected by default.
