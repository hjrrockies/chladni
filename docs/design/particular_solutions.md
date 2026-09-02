*Status: implemented. Written before the code; see "As built: deltas from this spec" at the end for where the implementation went past it, and why.*

# Overview
The core eigensolver in `chladni` is the Method of Particular Solutions (MPS), which is built around evaluating exact Helmholtz solutions at a candidate eigenvalue. MPS then searches for a linear combination which minimizes the boundary residual, which translates to a bound on the distance to the nearest true eigenvalue of the domain. As $\lambda$ varies, the distance to eigenvalues can be tracked and minimized, giving approximate eigenvalues.

`chladni` needs to have a rich library of "particular solutions" which are Helmholtz-exact by construction. These include:

- Corner-adapted "Fourier-Bessel" functions $J_{\ell \nu}(\sqrt{\lambda}r)\sin(\ell\omega), J_{\ell \nu}(\sqrt{\lambda}r)\cos(\ell\omega)$, where $\ell=1,\dots,L$ are the *degrees* of the family, to a maximum degree $L$.
- Real-valued fundamental solutions $Y_0(\sqrt{\lambda}r)$
- Multipole fundamental solutions $Y_\ell(\sqrt{\lambda}r)\sin(\ell\omega),Y_\ell(\sqrt{\lambda}r)\cos(\ell\omega)$, where $\ell$'s are the degrees.
- Plane waves
- More?

A collection of Helmholtz-exact particular solutions is called a particular solution *basis*, and chosing a good basis is the single most-important aspect of computing eigenvalues in MPS. For now, we will defer the algorithms for choosing good bases, and focus on making sure the objects themselves are fully-featured, highly-performant, and easy-to-use.

# Design Principles
- Built *from* domains, but not *attached* to domains: Particular solution families are generally created based on geometry, but they should also be able to be created stand-alone from the essential data that defines them. The __init__ methods should accept the minimally-sufficent stand-alone parameters. This ensures that a bespoke basis doesn't have to be routed through "ghost" geometry. That being said, the functionality for building "default" bases in a geometry-adapted way should be viewed as the standard usage.
- Performant above all else: Particular solutions get evaluated over and over again within MPS. The first job of any object is to give speedy, vectorized evaluation of the particular solutions. Since particular solutions often have $\lambda$-agnostic parts (such as the angular terms in Fourier-Bessel functions), smart caching should be used to eliminate duplicate calculations.
- Derivative evaluations: the gradients and directional-derivatives of the particular solutions should be just as accessible as the function values themselves: Neumann boundary and general Cauchy-data boundary integrals require this.
- Consistent scaling: Basis functions with Bessel terms exhibit poor scaling for combinations of order and argument. To make all other calculations consistent, bases should be normalized (or normalizeable) with respect to a fix set of evaluation points, so that $f(x,\lambda)$ is being rescaled by the same constant regardless of what $x$'s are in the evaluation set. Of course, the scaling constant will change with $\lambda$.
- Use `lappy` as inspiration, not as definitive reference. We want `chladni` to do better than `lappy`, down to the bones.

# Design Questions
- Should we default to working in the $\lambda$ eigenvalue domain or the $k = \sqrt{\lambda}$ wavenumber domain? Does it matter, numerically?
- Should our atomic basis objects be simple (one source point with multiple degrees, when relevant) or compound (multiple source points and corresponding degrees inside one object)?
- Are there any tricks we can use to correct poor scaling in a way that maximizes our digits of precision?
- Are there any flags/warnings we can raise when evaluations of particular solutions are losing precision? Can we detect that behavior cheaply?
- What should be our naming/notation conventions: for example, do we stick with $\nu$ to always mean "the (base) order of a corner singularity", which differs from `lappy` (which used $\alpha$ originally)? Is $r$ always "radius w.r.t. some base point"? Important to consider.
- Base class/inheritance structure: We are *not* going to worry about getting particular solutions for other operators (e.g. biharmonic), but it does seem prudent to name our base class something like `HelmholtzFamily` or similar, since it costs us nothing to be specific? Overall, what should we name our classes and how should they relate?
- What familes, beyond those listed above, are worth adding?
- How much effort do we put in up-front to provide convenience constructors for bases-from-geometry? The corner-adapted bases are essential, and boundary-offset fundamental solutions are too. I would also include Trefethen-Gopal-style root-exponentially-clustered fundamental solution "Lightning charges". Since sources can ultimately be placed *anywhere* (with caveats about singularities), there is no end the list of convienence builders we may discover that we want/need.

---

*Everything below was added as a handoff at the close of the geometry work. It is context,
not decisions -- the Design Questions above are still open, and the API spec for this layer
has deliberately not been written, so it can be done spec-first like the geometry layer was.*

# Where things stand

`chladni.geometry` and `chladni.plotting` are implemented, 355 tests, CI green on 3.11-3.13.
Nothing solves an eigenvalue problem yet. `examples/geometry_tour.ipynb` is the fastest way
to see what the geometry layer can do; `docs/design/geometry.md` has an "As built" section
recording where the code went past its own spec, and why.

The project convention is spec, then tests, then implementation. The geometry layer was built
that way and it paid: several real bugs were found by writing the test before the code.

# What the geometry layer hands this one

Geometry was designed around this layer's needs, so the seams already exist. The names below
are public and stable.

**Per corner** -- `Domain.corners` is a tuple of frozen `Corner`:

    z                            vertex location
    omega                        interior angle in (0, 2*pi)
    bc_in, bc_out                the boundary condition either side (0.0 dir, 1.0 neu, else Robin)
    is_singular                  is_geometric or is_bc_junction
    clearance                    radius bounding where a local corner expansion stays valid
    branch_angle                 direction of the final ray of a branch cut, NaN if none
    branch_vertices              intermediate vertices when the cut had to wall-follow; () for a ray
    branch_path(length)          the whole cut as a polyline, for drawing or testing
    admits_exterior_branch_cut   False on a hole: no cut avoids Omega there, topologically
    on_hole, bdry_idx            which boundary it belongs to
    seg_out, seg_in              indices -- the join key into a Quadrature
    segment_out, segment_in      the segment objects themselves, for e.g. clearance/length

**Per boundary node** -- `Domain.bdry_quad(n)` returns a frozen `Quadrature`:

    pts, normals, tangents       normals are outward on every boundary, holes included
    wts                          arclength weights; see weight_family
    tau, seg_idx, bdry_idx       provenance: where each node came from
    bc                           the boundary condition at each node
    integrate(f), radial(x0)     radial is (x - x0).n, the Rellich integrand's shape
    weight_family                'arclength' for everything geometry builds

**To supply your own rule** -- `Panel`, `legendre_nodes`, and `panels=` on `Segment.quad`,
`Boundary.quad` and `Domain.bdry_quad`. A `Panel` names its boundary, segment, span in `tau`
and a `nodes(tau0, tau1)` callable returning `(tau, w)` with **weights in tau**; geometry
multiplies by the segment length to reach `ds` and does nothing else to them. This is the hook
for corner-adapted Gauss-Jacobi rules: pass `weight_family='even'` (or whatever names the
exponent class) so the object never claims to be a plain arclength rule. Rescaling a singular
weight onto a shorter panel picks up a power of the panel width set by that weight's own
homogeneity, which is why that arithmetic stayed with the caller.

**Also available**: `Domain.interior_pts(n, rng=...)` for the GSVD regularization points,
`Domain.normalized(by=...)` (eigenvalues transform as `lam -> lam * s**2`),
`Domain.achievable_prec`, and `chladni.plotting.field_grid(domain, n) -> (X, Y, mask)`, which
is the primitive an eigenfunction colormap or surface will use.

# The layer boundary, stated once

**Geometry reports `omega` and the boundary-condition pair. It never computes `nu = pi/omega`,
and it must not start.** Deriving the singularity exponent -- and deciding which corners get an
enriched basis -- is this layer's first job.

That line is load-bearing. `lappy` crossed it with `geometry.pts_per_seg(domain, fb_basis)`,
which put a basis object into a layer-1 signature; the whole `Corner` design here exists so
that function has nowhere to come back.

# What lappy already knows, and where

Worth reading before designing, and worth citing rather than re-deriving -- lappy's docstrings
carry the measurements that settled each choice.

- `lappy/bases.py` -- `ParticularBasis` (ABC), `FourierBesselBasis`, `FundamentalBasis`,
  `MultiBasis` (composition), `NormalizedBasis`, and `ExPrecFBBasis` /
  `NormalizedExPrecFBBasis` / `InfNormalizedExPrecFBBasis`. That last cluster is three
  successive attempts at the Bessel scaling problem in Design Question 3 -- start there.
  Also `fb_corner_orders` / `fs_corner_orders` for how budget was split across corners.
- `lappy/basis_plan.py` -- `PlanConfig`, `CornerBlock`, `ArcBlock`, `BasisPlan`, plus
  `local_thickness` and `exterior_clearance`. This is the "choosing a good basis" machinery
  the overview above defers; the geometric inputs it needs are already on `Corner`.
- `lappy/eigfun_integrals.py::corner_specs` -- computes `nu = pi/alpha` and records three
  documented limits, all of which this layer will meet:
  - `nu <= 1/2 + margin` (near-slit): "the Rellich integrand is barely integrable and no rule
    recovers it; place x0 on this corner".
  - **mixed boundary conditions at a corner are not handled**: "the Zaremba corner integrand
    ~ r^(nu-2) is not integrable for nu<1, so the identity itself fails here". Note that
    `chladni` detects these corners at the geometry layer (`is_bc_junction`), which lappy did
    not -- so this layer can decide what to do about them deliberately rather than discovering
    them late.
  - only Dirichlet corners are wired: "a Neumann corner's 'uv' and 'TT' kernels need
    different exponents".

# Geometry-side facts bearing on the open Design Questions

Not answers -- inputs.

1. **lambda or k.** Geometry is agnostic; nothing in it carries either. The one place it
   touches the choice is `Domain.normalized()`, documented as `lam -> lam * s**2`, so a
   `k`-domain convention would want that restated as `k -> k * s`.
2. **Simple or compound atomic objects.** Geometry hands out corners one at a time
   (`Corner`), but boundary data as one object for the whole boundary (`Quadrature`, carrying
   `seg_idx`/`bdry_idx` so a subset is a mask, not a separate object). lappy went both ways:
   `FourierBesselBasis` spans corners internally while `MultiBasis` composes families.
3. **Bessel scaling.** See the `ExPrec*` cluster above. Note also that `Quadrature` is built
   once and reused across every `lambda`, so any `lambda`-agnostic precomputation has a
   natural home keyed to a node set.
4. **Detecting precision loss cheaply.** `Domain.achievable_prec` and `PrecisionWarning` are
   the same pattern one layer down: compute a bound in microseconds, warn only when the
   request exceeds it, and give the warning its own category so a shape-optimization grid can
   silence exactly it. lappy's `BoundaryQuad.shortfalls` is the same idea again, and its
   comment says why -- so that a reported precision "is never a claim contradicted elsewhere
   in the same object".


---

*Everything below is the API spec, written after the handoff above and before the
implementation, in the style of `docs/design/geometry.md`. It settles the Design Questions
at the top of this file. Notation and cross-layer conventions live in
`docs/conventions.md`.*

# The module is `chladni.helmholtz`

It names what the objects are -- exact solutions of the Helmholtz equation -- parallel to
`geometry` naming its own domain of discourse. `helmholtz.FourierBessel`,
`helmholtz.Fundamental`, `helmholtz.PlaneWave`, `helmholtz.Basis`.

Answering Design Question 7 directly: the base class is `HelmholtzFamily`. Being specific
costs nothing, and it leaves the door open for a `Biharmonic*` sibling without a rename.

# The spectral parameter is k

`laplacian(u) + k**2 * u = 0`, with `k` real and positive. Every particular solution is
literally a function of `k*r`; validation is one rule rather than a `sqrt` of a possibly
negative line-search iterate; and the Bessel accuracy regimes this layer must reason about
are stated in `(nu, k*r)`. `lam = k**2` stays the user-facing parameter and the `mps`
parameter, converted at one seam. See `conventions.md`.

# The three things this layer must get right

**1. `J_nu(k*r)` at `nu >> k*r`.** At a sharp corner the order reaches 100-350 while `k*r`
stays small. `scipy.special.jv` holds 13-14 digits and then falls off a cliff to exactly
`0.0` -- there is no accuracy-loss band, only a hard underflow edge at `|J| ~ 1e-291`:

    nu                      100    200    300    400    500    700
    x where jv hits 0      0.096   5.32  23.98  51.74  92.06  194.1

lappy lived with this: its FB columns at a sharp corner are literally zero arrays, and its
own benchmark notebook concluded those columns "carry almost no independent information
about the domain". Half of that is true -- near-dependence at a sharp corner is intrinsic --
and half is an artifact we can remove. A column representable at full *relative* accuracy is
not the same object as a column of zeros, and only one of the two is worth truncating.

**2. The scale must be point-set-free.** The design principle above says a column is
rescaled by the same constant regardless of which points it is evaluated at. lappy's
`NormalizedBasis` normalizes by a norm over a stored reference point set, which costs a full
extra evaluation per `k`: measured at 2.04-2.27x on the evaluation half of every sigma
(`lappy/bases.py:251`). An analytic scale is free and satisfies the principle exactly.

**3. The angle lift must be anchored to the branch cut.**
`np.angle((z - c.z)/c.t_out) % (2*np.pi)` is the obvious one-liner and it is wrong -- a
boundary point at `theta = 0` comes out as `-1e-17` and wraps to `6.283`. Measured
consequence: the Dirichlet family evaluates to `0.11, 0.17, 0.22` on the ray where it must
be identically zero.

# Module layout

    src/chladni/helmholtz/
        __init__.py         re-exports; k, the (m,n) shape, the scale invariant
        _core.py            HelmholtzFamily (ABC), Sampling, check_k
        _bessel.py          the scaled Bessel kernel -- the numerical heart
        corners.py          CornerModes: nu, the angular kind, the exponent family, the lift
        fourier_bessel.py   FourierBessel
        sources.py          Fundamental (Y_m multipoles) + placement builders
        waves.py            PlaneWave
        basis.py            Basis (the composite)

# The evaluation contract

    class HelmholtzFamily(Frozen, ABC)

      n_cols: int                   # also __len__
      prec: float                   # PREC = 1e-8

      at(pts) -> Sampling           # k-INDEPENDENT precompute; pts is complex[] or a Quadrature
      log_scale(k, cols=None) -> float[n]           # sigma; analytic, point-set-free
      singular_exponents() -> (complex[n], float[n])  # per column: origin, leading r exponent

      values(k, pts, cols=None)     # convenience; builds a throwaway Sampling
      grad(k, pts, cols=None)
      deriv(k, pts, vecs, cols=None)

      __add__(other) -> Basis       # flattens; a Basis is itself a HelmholtzFamily
      translated / rotated / scaled / with_prec

    class Sampling(Frozen)

      family: HelmholtzFamily
      pts: complex[m]
      normals: complex[m] | None    # present when built from a Quadrature
      bc: float[m] | None           #   "

      values(k, cols=None)          -> (m, n) float
      grad(k, cols=None)            -> (m, n) complex     # dx + 1j*dy
      deriv(k, vecs, cols=None)     -> (m, n) float
      values_and_grad(k, cols=None) -> two (m, n)         # ONE transcendental pass
      trace(k, cols=None)           -> (m, n) float       # (1-bc)*u + bc*du_dn
      log_scale(k, cols=None)       -> float[n]
      accuracy(k, cols=None)        -> float[n]           # per-column rel-error bound, free

Four points about this shape, each fixing something measured in lappy.

**`at(pts)` is explicit, not a hidden cache.** lappy hid the k-independent geometry behind a
byte-budgeted `instance_cache` keyed on a value-hashed `PointSet`: 370 lines with a lock, a
global byte ledger, weakref finalizers and an eviction policy. Here the object *is* the
cache. The solver builds one `Sampling` per point set at construction and calls
`sampling.values(k)` in the loop. Immutable, thread-safe by construction, nothing to evict,
no `clear_instance_caches`. `family.values(k, pts)` remains as sugar, documented as sugar.

**`trace(k)` is the seam the solver actually wants.** `Quadrature` already carries `pts`,
`normals` and `bc`, so a `Sampling` built from one can assemble the boundary matrix itself.
This deletes lappy's `make_bdry_vander` boundary-condition branch
(`lappy/mps.py:257-289`) and, more to the point, lets a Robin or mixed-BC boundary share one
Bessel pass between the value and the normal derivative instead of paying for two.

**`values_and_grad` exists because lappy paid twice.** Its `_eval_pointset` and
`_grad_pointset` each call `_bessel` at the same parameter with no memo between them. The
series recurrence here produces `J` and `J'` from one loop and one extra accumulator.

**`cols=` is designed in, not retrofitted.** One index space -- the global column index -- at
every level. lappy has two inside `FourierBesselBasis` (column versus radial mode) with the
translation duplicated in two methods, and `MultiBasis._dispatch_cols` silently reorders
output for an unsorted `cols` while its docstring promises "in the given order"
(`lappy/bases.py:154`). Here `Basis` dispatches and then scatters back through the inverse
permutation, and the test passes deliberately shuffled indices.

`singular_exponents()` has exactly one intended consumer: corner-graded Gauss-Jacobi panels,
which geometry's `Panel` seam exists for. lappy's analogue `corner_terms` carries an
elaborate contract and has *no production consumer*; this one ships with its consumer's test
or it does not ship.

**Deliberately absent**: any analogue of `pts_per_seg(domain, fb_basis)`. Geometry's spec
already lists it under "Dropped: which cannot exist here". Its honest replacement is a
polymorphic collocation hint on the family, which belongs with basis *planning* -- deferred.

# The scaled Bessel kernel

    log_j(nu, x) -> Scaled     #  J_nu(x) = S*exp(sigma),  J_nu'(x) = Sp*exp(sigma)
    log_y(m, x)  -> Scaled     #  Y_m(x)  = S*exp(sigma),  Y_m'(x)  = Sp*exp(sigma)

    class Scaled(NamedTuple)
      S, Sp: float[]      # scaled value and scaled derivative
      sigma: float[]      # the log scale
      rel_err: float[]    # bound on the relative error of S and Sp -- free, see below

`sigma` is the SAME canonical quantity in every branch, `nu*log(x/2) - gammaln(nu+1)`, so
that the branch a given entry took is invisible to the caller. Watson's bound
`|J_nu(x)| <= (x/2)**nu / Gamma(nu+1)` for `nu >= -1/2` then says **`|S| <= 1` always**,
which is a structural invariant a test can assert directly.

Both branch on a predicted largest series term `M = exp(x**2/(4*nu))`, about ten flops. The
mask is taken over the whole `(m, n)` block, so each branch is one vectorized call:

* **A**, `M <= 1e3` (equivalently `x <= 5.25*sqrt(nu)`): the convergent ascending series,
  with `sigma = nu*log(x/2) - gammaln(nu+1)`. Recurrence `t *= q/(j*(nu+j))` with
  `q = -x**2/4`, accumulating `S += t` and `Sp += (nu + 2*j)*t`. Three terms at `x = 0.01`,
  24 at `(400, 50)`, 45 at `(400, 100)`. Measured against mpmath at 50 dps: **5e-14 typical,
  2.8e-13 worst** over `nu <= 400`.
* **B**, `M > 1e3`: `scipy.special.jv` / `jvp`, rescaled by `sigma`. Guaranteed nonzero here.
* **C**, `nu > 575` and `M > 1e3`: four-term Olver/Debye. A safety net; measured `<= 2e-12`
  at the corner where it is reachable at all.

The branches **overlap widely** rather than abutting. Scanning `nu` in `[0.2, 600]` against
`x` in `[0.5, 100]` for cells where `jv` underflows *and* the series is inaccurate finds
**zero**; the gap first opens near `nu = 515`. That overlap is the strongest available test
of the hybrid, and it is what fails first if a threshold is edited.

Two things worth stating before the code exists. First, `log|J|` is genuinely singular at the
zeros in the oscillatory regime `nu <~ x`, which is why `sigma` is defined as a
**z-independent analytic number** and never as a log of the value: the scaled value keeps the
oscillation, the sign changes and the zeros. Second, the log path is not a slow fallback.
Measured on a 600 x 100 sharp-corner block: **5.7 ms for values and derivatives**, against
4.8 ms for `scipy.special.jv` alone -- which returned **43% exact zeros** -- and 14.2 ms for
`jv` plus `jvp`, which is the honest comparison since that is what the caller needs. About
20% over a call that is wrong on half the block, and 2.5x faster than the scipy pair.

`Y_m` is the same problem with the opposite sign: it overflows rather than underflows. The
last finite order, bisected against scipy, is 65 at `x = 1e-3`, 149 at `x = 1` and 294 at
`x = 20`, with the value there around 1e302 -- a clean cliff at the float64 ceiling, and the
same empty gap. The truncated finite sum with
`sigma = gammaln(m) - m*log(x/2) - log(pi)` measures 1e-13 across the region where `yv`
overflows, and is poor only for `m <~ 5` with `x >~ m` -- exactly where `yv` is O(1).

**Answering Design Question 4**, a free precision diagnostic: `max_j|term_j| / |S|` is
already computed in the loop and bounds the entry's relative error at `eps * M`. Reduce to a
per-column max, expose it as `Sampling.accuracy(k)`, and warn when it exceeds `prec`. Same
pattern as `Domain.achievable_prec` one layer down -- a bound computed in microseconds, with
its own warning category so a shape-optimization grid can silence exactly it.

# The column scale

`sigma` depends on `(family, column, k)` and never on the evaluation points. That is what
makes it compose with a downstream unit-norm rescale without double-counting: with
`A[:,j] = psi_j * exp(-sigma_j)` and the solver's `A_hat = A @ diag(1/d)`, the coefficient of
`psi_j` is `c_j/(d_j*exp(sigma_j))`, and the two log scales simply add.

**Answering Design Question 3.** For Fourier-Bessel, `sigma_j` is the Debye log-envelope at
the corner's own `clearance` radius `R`, smoothly capped by the global maximum of `J_mu`:

    z = k*R/mu ;  q = 1 - z**2 ;  w = sqrt(max(q, 0))
    eta    = w + log(z/(1 + w))   for q > 0, else 0        # C1 across the crossover
    floor  = 0.34952 * (mu + 0.3)**(2/3) / mu              # the Airy transition width
    q_plus = (q + sqrt(q**2 + floor**4)) / 2               # a SMOOTH positive part
    a = mu*eta - 0.5*log(2*pi*mu*(q_plus**2 + floor**4)**0.25)
    b = log(0.6748 * (mu + 0.3)**(-1/3))
    sigma = -logaddexp(-4*a, -4*b) / 4                     # a soft minimum, not min

Three of those lines exist because a simpler version was measured and failed, and the
failures are worth recording.

**The amplitude has to be floored.** `-0.5*log(2*pi*mu*w)` diverges as `w -> 0` at the
crossover, and its derivative diverges faster: the second difference of `sigma` in `k`
reaches **8.4e5** at `mu = 20`, so the first version of this formula was smoother than `min`
without being smooth. Flooring `w` at the Airy transition width -- where the asymptotic stops
being valid anyway -- brings that to **3.1**. The floor constant is SOLVED, not tuned: it is
the value that makes the floored amplitude equal the global maximum exactly at the crossover,
so the envelope and the cap meet rather than cross.

**The positive part has to be smooth.** Past the crossover `q` goes negative and `q**2`
starts growing again, which drives the amplitude down and the column maximum up -- measured
at **27.7**. `max(q, 0)` fixes the size but leaves a jump in the second derivative; the
smooth form has the same two limits and is C-infinity.

**The blend is a soft minimum, not `min`.** `min` puts a kink in `k`, and `sigma(k)` sits
inside the objective the solver parabola-fits. Sharpness 4 costs a factor `2**(1/4) = 1.19`
in the column maximum where the two branches meet, which is the price of that.

Measured over `mu` from 0.05 to 320 and `k*R` from 0.5 to 300: column maxima land in
**[0.802, 1.185]**, a spread of 1.48x, against raw columns spanning hundreds of orders of
magnitude over the same grid. `|d2 sigma / dk2|` is at most 20.8 at `mu = 2`, 3.1 at
`mu = 20` and 0.5 at `mu = 200`.


The obvious alternatives were checked and are worse. Normalizing by `log J_mu(k*R)` puts a
**pole in `sigma(k)`** at every zero of `J_mu`. The leading small-argument term alone leaves
a residual spread of 4.7x at `k*R = 10`, growing to **3.0e8x at `k*R = 100`**. Any norm over
the sampling points violates point-set independence outright.

Because `sigma` depends on `k` and `R` only through `k*R`, it is **exactly invariant** under
`Domain.normalized(by=s)` with `k -> k*s`. That is an equality test, not a tolerance test.

**Coefficient recovery, stated once.** At a `pi/8` corner with `k*R = 10` the `sigma` range
spans 1149 nats, so `exp(sigma)` spans 1e-499. Unscaled coefficients must never be
materialized: an eigenfunction is always `scaled_matrix @ c`, and the recovery factor is
handed back in the log.

# Corner modes

The layer boundary, stated at `boundary.py:43`: geometry stops at `omega` and the BC pair and
never computes `nu`. This is where it happens.

    class CornerModes(Frozen)
      nu: float          # pi/omega
      kind: str          # 'sin' | 'cos'   -- fixed by the BC on the theta = 0 ray
      offset: float      # 0.0 or 0.5      -- fixed by the BC on the theta = omega ray
      beta: float        # branch-cut angle relative to t_out, in (omega, 2*pi)

      orders(degrees) -> float[]           # mu = (l + offset)*nu

    from_corner(corner) -> CornerModes

The quantization decouples, which is the piece lappy never had:

    bc_out (theta=0)   bc_in (theta=omega)   angular        mu                    leading
    dir              dir                 sin(mu*theta)    l*nu,  l = 1,2,...    nu
    neu              neu                 cos(mu*theta)    l*nu,  l = 0,1,...    0
    dir              neu                 sin(mu*theta)    (l + 1/2)*nu          nu/2
    neu              dir                 cos(mu*theta)    (l + 1/2)*nu          nu/2

`sin(mu*0) = 0` and `cos'(mu*0) = 0` hold automatically, so the `theta = 0` ray only chooses
sin against cos; the `theta = omega` ray then forces `sin(mu*omega) = 0` or
`cos(mu*omega) = 0`. **`theta = 0` stays on `t_out` in all four cases.** Flipping to "measure
from the Dirichlet ray" is the error-prone formulation; this way the mixed and unmixed cases
are one code path with two booleans. Verified numerically at `omega = 1.9*pi`, `k = 7.3`:
all four rows satisfy their own boundary conditions, `0.0` exactly on the `theta = 0` ray and
`8e-17` on the other.

At a **flat** BC junction -- `omega = pi`, which `Corner.is_bc_junction` catches and
`is_geometric` does not -- the leading exponent is `mu = 1/2`, the classic Zaremba `sqrt(r)`.
That is a stronger singularity than any convex Dirichlet corner, and it is exactly the case
lappy's `eigfun_integrals.corner_specs` declares unhandled. `chladni` detects it one layer
down, so it gets enriched here deliberately rather than discovered late.

**Robin, plainly: there is no closed-form separable local solution.** On a ray the Robin
condition gives `f'(0)/r = h*f(0)`, an `r`-dependent condition that no single `(mu, f)`
satisfies; the `1/r` is structural, not removable. But as `r -> 0` it dominates `h*u`, so
Robin degenerates to Neumann in the leading local asymptotics (Grisvard). Use the Neumann
family at a Robin corner, and the neu/dir family at a Robin/Dirichlet junction. The docstring
must say **singularity-adapted, not BC-adapted**: the functions are Helmholtz-exact, which is
all MPS requires, and they carry the correct leading exponent, but they do not satisfy the
Robin condition termwise.

**Hole corners** are rejected unless the exponent family is integral. `admits_exterior_branch_cut`
is False there for a topological reason, and a non-integer `mu` would be multivalued on Omega.
Raise with the reason rather than silently producing a wrong function.

# Fourier-Bessel

    class FourierBessel(HelmholtzFamily)
      origin: complex
      t_out: complex                # unit tangent: the theta = 0 ray
      modes: CornerModes
      degrees: int[]
      clearance: float              # sets sigma
      branch_vertices: tuple = ()   # polyline cut, when a straight ray does not clear Omega

      at_corner(corner, degrees) -> FourierBessel

**Answering Design Question 2**: one object is **one corner**. Atoms are simple; batching
across corners happens in `Basis`.

Two reasons, against lappy's compound `FourierBesselBasis` that spans every corner behind a
flat `alphak_vec` and an `_src_idx`. First, that form silently drops `orders == 0` sources in
the constructor and then has to re-match sources back to `domain.corners` by `np.isclose` on
complex positions to recover what it threw away (`lappy/bases.py:421, 551-576`). Second, the
performance argument for batching does not apply here: the corner count is small, at most
about twenty, and one `jv` call per corner over an `(m, M_i)` block does the same work as one
call over `(m, n)`. Fundamental solutions are genuinely different -- 200 sources at order 0
must not become 200 calls -- so `Fundamental` holds an **array** of origins sharing one order
structure. The rule, stated once: *an atom is a set of origins sharing the same structural
parameters.*

The angle lift, which is the load-bearing line of the module:

    beta = (branch_angle - angle(t_out)) % (2*pi)                     # in (omega, 2*pi)
    theta  = ((angle((z - origin)/t_out) - beta) % (2*pi)) + beta - 2*pi

`theta = 0` is the outgoing ray, `theta` increases CCW to `omega` on the incoming ray, and the
`2*pi` discontinuity sits exactly on the cut in the exterior wedge. Anchoring the lift to
`beta` -- never to `0` or `-pi` -- is what makes the `theta = 0` ray evaluate to *structurally*
exactly `0.0`.

Checked against the geometry source. `Boundary.corners` (`boundary.py:319`) computes
`omega = angle(-t_in/t_out) % (2*pi)` with `t_out = segments[i].T0`, so
`arg((z - c.z)/t_out)` in `(0, omega)` is the interior wedge by construction; `_branch_angles`
(`boundary.py:365`) sets the bisector at `angle(t_out) + omega/2 + pi`, i.e.
`beta = omega + (2*pi - omega)/2` in this same relative measure. Verified on all six L-shape
corners and all four corners of a square hole.

**Holes need no sign flip.** `Domain` reverses them to CW, which puts Omega on the left of
travel on every boundary, so `-1j*T` is outward everywhere and one convention covers both.
Measured: a square hole corner reports `omega = 3*pi/2`, the wedge `(0, omega)` is entirely
inside Omega, and `beta = 5.4978` exactly.

One asymmetry worth a docstring line: the boundary condition on the `theta = 0` ray holds to
**exactly 0.0** (structural), the one on `theta = omega` only to about 1e-16 relative, from
roundoff in `omega`. A test asserting exact zero must target `bc_out`'s ray.

Polyline branch cuts, when geometry had to wall-follow, carry a sheet correction inside the
cut polygon as in lappy: precomputed once per `Sampling`, with zero per-evaluation cost when
the cut is a plain ray, which is the common case.

# Fundamental solutions

    class Fundamental(HelmholtzFamily)
      origins: complex[s]
      order: int = 1                # multipole degrees m = 0..order-1; 2*order-1 cols per origin

      offset(quad, distance) -> Fundamental        # along the outward normal
      lightning(corner, n, sigma=4.0, L=None) -> Fundamental

`Y_0` and `Y_0' = -Y_1` get a fast path. lappy measured `y0` at **42x** `yv(0, .)` and `y1`
at **85x** `yvp(0, .)`, worth 1.98 s of a 5.88 s L-shape solve (`lappy/bases.py:1096`).
Dispatch on the order *values*, not on the `order` attribute, so a hand-assembled family
still takes it.

**Sources must be exterior, and that is an exception rather than a filter.** lappy's
docstring at `bases.py:889` is worth citing rather than re-deriving: an interior pole breaks
the MPS premise, voids the Moler-Payne certificate, and makes the tension *better* rather
than worse. Measured there, 24 of 240 sources inside took the tension background from 5e-2 to
3e-7, which made the acceptance test swallow everything, produced a spurious eigenvalue and
hid a real one. Silence is the failure mode; raise.

**Lightning clustering** (Gopal-Trefethen): `s_j = L*exp(-sigma*(sqrt(n) - sqrt(j)))` for
`j = 1..n`, placed along `branch_angle`. lappy's formula matches the published form exactly
(`lappy/bases.py:1001`) but its **defaults do not**: `sigma=1` against the published
`sigma ~ 4`. Measured at `n = 25`, `L = 1`: the innermost pole sits at **1.1e-07** with
`sigma = 4` and at **1.8e-02** with `sigma = 1` -- five orders of magnitude of reach toward
the corner, which is the entire point of clustering. Default `sigma = 4.0` here. lappy also has **no taper**: the
outer end of the cluster should thin into the ambient source spacing so the two families do
not double-count, and when the same corner also carries Fourier-Bessel terms the cluster
should stop around `0.05*R_c` with 8-12 poles retained
(`lappy/docs/mps_heuristics.tex:232`) -- a conditioning bridge, not an approximation tool.

Multipole order is capped by `m <= ln(1/prec)/ln(1 + D/rho)`, since `Y_m(k*r) ~ (2/(k*r))**m`
has dynamic range `(r_max/r_min)**m` across Omega. That cap is also what keeps the `Y`
overflow branch of the kernel rare.

# Plane waves

    class PlaneWave(HelmholtzFamily)
      directions: complex[d]        # unit
      uniform(n) -> PlaneWave       # alpha_j = pi*j/n

`u_c = cos(k*dot(z, d))`, `u_s = sin(k*dot(z, d))`, with gradients `-k*d*u_s` and `k*d*u_c`.
`sigma = 0`: these columns are already O(1) by construction, which is the cleanest available
illustration that the scale is a per-family property and not a global postprocessing step.

**Answering Design Question 8** in part: lappy has no plane waves at all, and they are the
standard MPS family for a smooth or convex domain. Directions sweep `[0, pi)`, not
`[0, 2*pi)` -- `d` and `-d` give the same `{cos, sin}` pair up to sign, so a full sweep
duplicates every column exactly.

Plane-wave bases are exponentially ill-conditioned past a threshold that moves out linearly
with `k*R`. Measured on the unit disk with pre-normalized columns, `cond2` of the `2N`-column
matrix:

    k*R \ N       5      10      20      30      40      60
      5         8.2   1.9e2  1.0e10  1.2e16  1.4e16  1.7e16
     10         1.9   2.0e1   2.1e4  1.6e11  6.9e15  9.6e15
     20         1.8     2.2   2.6e1   1.4e3   1.8e8  2.9e15
     40         1.3     2.0   1.7e1   2.0e2   2.1e2   1.0e6

Numerical rank saturates at the Jacobi-Anger cutoff: measured ranks 33/45/65/95/149 at
`k*R = 2/5/10/20/40` fit

    n_useful ~ k*R + 11*(k*R)**(1/3)

with `R` the radius of the smallest enclosing disk. Beyond that, columns arrive at machine
noise -- they do not enrich the space, they lower the smallest singular value and corrupt the
tension. `PlaneWave.uniform` warns past the bound; the bound costs a cube root.

# The composite

`Basis(families)`, with `__add__` flattening, `at(pts)` returning a composite `Sampling` that
hstacks, and ownership of the `cols` dispatch and its inverse-permutation scatter.
`basis.families` is the tuple; `basis.column_family` is the int array; `log_scale` and
`singular_exponents` concatenate.

No `_basis_plan` monkey-patch. lappy's `basis_plan.realize` writes an attribute onto the
basis object and `plan_of` then walks up to eight `.basis` levels hunting for it, described
in its own docstring as "introspection, not part of the basis contract". Provenance is a real
field or it does not exist.

# Ground truth for the test suite

Tolerances below are measured, not aspirational.

* **mpmath at 50 dps** for the kernel, value and derivative, over a `(nu, x)` grid that
  **includes the underflow region**: 1e-13 relative over `nu <= 400`, `x <= 50`. Do not
  assert tighter than 5e-13 at `nu ~ 400`; the kernel is a 5e-13 object there, not a 1e-15
  one.
* **Branch A against branch B across their overlap**: 1e-12. This is the single strongest
  test of the hybrid.
* **Dirichlet vanishing on the `theta = 0` ray**: assert `== 0.0`, not `< tol`. This is the
  test that catches the `% 2*pi` lift bug. On the `theta = omega` ray, 1e-15 relative.
* **Analytic Helmholtz residual** in polar form against `-k**2*u`: 1e-14. Preferred for unit
  tests.
* **Fourth-order FD residual** at `h = 1e-3`: 1e-9, and the floor is the differencing, not
  the code. Measured h-sweep: 1.7e-3 at `h=1e-1`, 1.8e-7 at 1e-2, 1.9e-11 at 1e-3, 1.2e-9 at
  3e-4 where roundoff takes over. State the tolerance so nobody tightens it.
* **Gradient against fourth-order central FD** at `h = 1e-4`: 4.3e-13. Second-order FD only
  reaches 6.5e-10 at its best `h`; use fourth order.
* **Complex-step gradients do not work here** -- measured relative error 8.5e3.
  `scipy.special.jv` does not deliver the imaginary part of a 1e-20 perturbation to relative
  accuracy, and `arctan2` does not complexify. Recorded so it is not tried again.
* **Circular sector of angle `omega`, Dirichlet**: a *single* FB column with `mu = l*nu` is
  an exact eigenfunction -- the sharpest completeness test available, 1e-14 on the
  eigenvalue. The dir/neu sector supplies exact Zaremba references the same way.
* **Square, disk, equilateral triangle**: 1e-13 to 1e-14 on known eigenvalues, for the
  ambient families.
* **L-shape `lam_1 = 9.6397238440219410527`**: end to end, and `slow`. Reaches 1e-12 to
  1e-13 with FB at the reentrant corner against about 1e-5 without it; that gap is itself the
  regression test that the corner enrichment is wired up. Deferred until the `mps` layer
  exists.

# Deferred

Basis *planning* -- how many terms at which corner, how sources are budgeted, refinement
against a measured residual. That is lappy's `basis_plan.py`, the overview above defers it,
and its geometric inputs are already on the geometry objects. The one thing this layer owes
it is that every family be constructible from explicit parameters without routing through
geometry, which is why `at_corner` / `offset` / `lightning` sit beside the constructors
rather than replacing them.

---

# As built: deltas from this spec

Everything above still describes the intent correctly. This records where the code went
further, or differently, and why -- as `geometry.md` does. The live record is the code, the
test suite, and the numbers each carries.

## The scale needed the EVALUATION radius, not the clearance

The spec says `sigma` is the Debye envelope at "the corner's own `clearance` radius R". That
is wrong, and the benchmark caught it: `clearance` bounds where the local expansion stays
VALID, while the columns are evaluated over the whole boundary. The error is
`(r_max/R)**mu`, and on the L-shape at `k = 7.3` with 14 degrees per corner that is
`2.83**28` -- measured column maxima of **5.7e+10** against the O(1) the whole design is for.

`FourierBessel` therefore takes `radius`, and `evaluation_radius(domain, origin)` computes it
as `max |z - origin|` over the boundary. `at_corner(corner, n, domain=...)` fills it in;
without a domain it falls back to `clearance`, which is right only for a corner-local
evaluation, and the docstring says so with the number. Both quantities are fixed geometric
constants, so point-set independence holds either way; only one of them makes the columns
O(1) over the set the solver actually evaluates on.

## The scale needed two more terms to be smooth

The spec's formula was `-logaddexp(-a, -b)` with the plain Debye `a`. Writing the test for
"sigma is smooth in k" showed it is not: `-0.5*log(2*pi*mu*w)` diverges as `w -> 0` at the
crossover and its derivative diverges faster. Second difference in `k` at `mu = 20`:

    as specified (unfloored, logaddexp)     8.4e+05
    a hard min instead                      1.9e+03
    as built (floored, soft blend p = 4)        6.5

The middle row is the surprise: unfloored, the smooth blend is worse than `min`, because
`min` clips the divergence while `logaddexp` follows it in. The full formula and both
corrections are in the section above, with their measurements.

## The kernel's error floor is `sigma`, not the series

`rel_err` was specified as `eps * largest_term / |S|`. That is not the dominant term.
`gammaln(401)` is 2000.6, held to 1e-16 relative and therefore 2e-13 ABSOLUTE, and an
absolute error in a log is a relative error in the value: measured 1.1e-13 at `(400, 50)`
against a largest-term bound of 1.6e-15. `rel_err` carries both.

That part is also, usefully, constant along a column, so a scale-invariant consumer sees only
the series half. `Sampling.accuracy` reports the honest entrywise number and the docstring
says it is conservative for the GSVD.

## `Sp is None` when the derivative was not asked for

Not in the spec. The scipy branch gets `J'` from `J_{nu-1} - J_{nu+1}`, so asking for the
derivative triples the number of `jv` calls -- and a Dirichlet boundary, the common case,
never wants it. Measured on a 600 x 144 L-shape block: **26.3 ms with, 13.2 ms without**.
`log_j` and `log_y` take `derivative=`, and `Sampling._eval` threads `want_grad` into it.

`scipy.special.jvp` and `yvp` are also gone, replaced by direct `jv`/`yv` calls on the
symmetric recurrence: `jvp` routes every call through `_bessel_diff_formula`, a generic
n-th-derivative sum with a Python loop and a `comb` per term, which profiled at 39% of the
whole per-k evaluation -- more than the Bessel functions it was calling.

## Branch C selects itself

The spec gives branch C a threshold, `nu > 575 and M > 1e3`. The code instead runs branch B
and falls through wherever scipy returned exactly 0.0. That is one fewer tuned constant and
it cannot drift out of date when scipy's own underflow edge moves.

## Polyline branch cuts use a winding number

The spec says "a sheet correction inside the cut polygon, as in lappy". lappy precomputes a
single `+-1` sign, which is correct for one wrap; a wall-following cut on a spiral can wrap
more than once. The code uses the winding number of the closed loop (cut outward, ray
inward), which counts. Verified on every walled corner of `spiral()`, at four radii: every
`2*pi` jump of `theta` lands within half an angular sample of `Corner.branch_path`.

## Measured numbers that differ from the ones the handoff carried

Recorded because both versions are now in this file's history and only one of them was
measured here.

* `Y` overflows at order 65 at `x = 1e-3` and 294 at `x = 20`, not the "m ~ 45 / ~150" the
  handoff carried.
* `y0` agrees with `yv(0, .)` to 2.7e-15 ABSOLUTE, not relatively -- near a zero of `Y_0`
  the relative gap reaches 3.7e-12. lappy's "~1 ulp" is true in one sense only. The measured
  speedup is 69x on 20000 points.
* Lightning clustering at `n = 25`, `L = 1`: the innermost pole sits at 1.1e-07 with
  `sigma = 4` and 1.8e-02 with `sigma = 1`, not "2e-9 against 7e-3".
* The log-domain kernel is NOT faster than `scipy.special.jv` alone; it is about 20% slower
  than a call that is wrong on 43% of a sharp-corner block, and 2.5x faster than the `jv`
  plus `jvp` pair that is what a caller actually needs.

## The handoff's `is_singular` line was wrong; this layer owns the word now

The table at the top of this file records `is_singular` as `is_geometric or is_bc_junction`,
which is what geometry shipped. That set is strictly too large -- every right angle of a
rectangle is in it, and none of them needs a Fourier-Bessel family. The honest question is
whether the orders `(l + offset)*nu` are all integers, which is `is_single_valued`, already
here. It now also has the name a caller actually reaches for: **`CornerModes.is_singular`**,
the documented negation of it, and the only place in the project that uses that word
unqualified. `CornerModes.is_zaremba` (`offset == 0.5`) came here too.

Geometry keeps the half of the question that is genuinely about the wedge --
`Corner.is_reflection_closed`, `omega = pi/m` -- and none of the half that is about this
equation. The offset depends on the BC pair through `_singularity_family`, whose Robin rule is
Grisvard's asymptotics, so a `Corner.is_zaremba` would have been this layer's fact stored one
level down. `docs/conventions.md` carries the reservation and the layer rule; `geometry.md`'s
As-built section carries the argument.

For a basis builder the practical consequence is one line: the corners needing a
Fourier-Bessel family are `[c for c in dom.corners if CornerModes.from_corner(c).is_singular]`,
and no geometric predicate is a substitute for it.

## Additions with no home in the spec

* `chladni/_core.py`. `Frozen`, `_set`, `check_prec`, `PREC` and `PrecisionWarning` moved up
  out of `geometry/_core.py` when this layer needed the same contract; `geometry._core`
  re-exports them, so nothing below changed. A second `Frozen` would be a second answer to
  "is `cached_property` safe here", and that has to be one answer.
* `docs/conventions.md`, which `CLAUDE.md` had been asking for.
* `CornerModes.bc_adapted`, False where a Robin condition was mapped onto the Neumann
  family, so a docstring cannot quietly overclaim.
* `CornerModes.is_single_valued` and `first_degree`. The second is why a Dirichlet corner
  starts at `l = 1`: `sin(0*theta)` is the zero function and would put a column of exact zeros
  into the pencil for the rank truncation to find.
* `Fundamental.radius` is required rather than inferred. A hand-built family has to state
  its standoff, because guessing it is a silent scale error of the same kind as the
  clearance one above.
* `PlaneWave.translated` returns `self`, since a translation is a phase and a phase mixes
  the cosine and sine of the same direction -- a rotation within the span.

## `prec` governs the claim, not the cost -- and now actually does

The spec says (under the free precision diagnostic) to "expose it as `Sampling.accuracy(k)`,
and warn when it exceeds `prec`". `accuracy(k)` shipped; the warning did not, and an audit of
the test suite found the wider gap behind it: `prec` in this layer was validated, stored,
composed by `min` and threaded through every transform, and **read by nothing**. An AST walk
for `self.prec` loads found only pass-throughs into constructors.

That is half a design position and half an omission, so both halves are now explicit.

**The position.** This layer has no adaptive loop. Its cost is `n_points x n_cols`, chosen by
the caller, and the kernel already delivers the best float64 can do -- there is no
looser-but-cheaper path to take. So `prec` here governs what the layer will CLAIM, never what
it computes, and `test_prec_does_not_change_the_numbers_in_this_layer` asserts exactly that so
the position is recorded rather than assumed.

**The omission, in three parts.**

* `Sampling._check_accuracy` warns when the entries cannot be delivered to the family's own
  `prec`. The bound is free -- it falls out of `_bessel` -- so this costs one reduction. The
  message quotes the shortfall rounded to a power of ten, because Python's default warning
  filter keys on message text and a k-dependent float would fire hundreds of times in a solve.
* `Fundamental.max_order(radius, extent, prec)` implements the cap the plan described and the
  code never had: `m <= ln(1/prec)/ln(1 + extent/standoff)`. `extent` is a new optional field,
  the far radius, supplied automatically by `offset(..., domain=...)`. Without it the check
  cannot run and does not guess, on the same rule as `PlaneWave.n_useful`.
* `PlaneWave.n_useful` takes `prec`. Its constant was reported as "about 11", fitted to
  measured ranks; it is not fitted at all -- `(ln(1/eps_mach))**(2/3) = 10.912`. Writing it
  that way makes the bound precision-aware for free, and the derived law tracks the measured
  numerical rank to within 2% over `k*R` in [5, 40] against relative cuts from 1e-13 to 1e-4.
  This is the one place in the layer where a looser precision genuinely buys a smaller basis.

Note the direction of that last one, which is easy to invert: a TIGHTER `prec` resolves MORE
independent columns, so the bound RISES as `prec` falls, and a fixed direction count becomes
wasteful as `prec` loosens.

The geometry side was audited in the same pass. `prec` there is load-bearing for accuracy and
was tested by exactly one assertion, `tight <= loose`, on a mild ellipse where both sides sit
at machine zero and the test passes vacuously. It was never tested for work at all, though the
effect is a factor of 760 between 1e-2 and 1e-8 on an eccentric ellipse. `tests/test_precision.py`
covers both halves for both layers.

## Still deferred

Basis planning, unchanged from the spec: how many degrees at which corner, how sources are
budgeted, refinement against a measured residual. Every family is constructible from
explicit parameters, which is what that layer needs from this one.

The end-to-end ground truth in the table above -- the sector eigenfunctions, the square, the
disk, the equilateral triangle, and the L-shape's `lam_1` -- is written down but not yet
tested, because there is no eigensolver to test it with. Those are the `mps` layer's first
tests, not this one's.
