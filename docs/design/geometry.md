*Status: implemented. Written before the code, as the spec the tests were built against; everything above "As built" is preserved as written, and the deltas are recorded there rather than edited in.*

# Overview

The `geometry` module of `chladni` is the first layer of the system. It gives the user access to
convenient functions for creating segments, boundaries, and domains. These geometric objects can be
queried to obtain points sampled from the boundary, points sampled from the interior of a domain,
tangent and normal vectors to the boundary, and other quantities like area, perimeter, and diameter.
Segments and boundaries come with attached boundary conditions, which defaults to `dir` for
Dirichlet. Individual segments always have a single boundary condition to be enforced, which is
`dir` for Dirichlet, `neu` for Neumann, or a float value for the convex-combination form of Robin
boundary conditions: `(1-bc_param)*u + bc_param*du_dN = 0`.

Its real consumer, however, is the solver, not a human. Everything below is arranged so that the
`particular_solutions` and `mps` layers can get what they need from a `Domain` without reaching back
into private helpers, and without geometry needing to know anything about the Helmholtz equation.

> This document was written **before** the implementation, as the API spec the tests were
> built against. It is accurate about intent and about why each decision was made, but the
> code moved past it in places. See **As built** at the end for the deltas.

# Design principles

- Positive/counter-clockwise orientation: Segments understood to be taken counter-clockwise (CCW),
  boundaries consisting of multiple segments store the segments in CCW order, and outward-pointing
  unit normal vectors are generated with that assumption.
- Adaptive algorithms: Approximations (such as reparameterization with respect to arclength, or
  computing the area of a domain) should be done adaptively.
- User-specified precision: Approximations should also adapt to working relative precision.
- Lazy evaluation: Quantities like area and perimeter should be computed once, but only lazily.
- Pythonic sugar: Pythonic design should include "syntatic sugar" to make manipulating the geometry
  easy.

# Settled conventions

**Points are `complex128`**, public and internal. `.real`/`.imag` are free views, rotation is
multiplication by `exp(1j*theta)`, and a corner angle is one call to `np.angle`. `chladni` is
planar-only, so the usual objection to complex does not apply. Conversion happens only at the
`shapely`/`matplotlib` seam, through `as_xy(z) -> (N,2)` and `from_xy(a) -> complex[]`.

**Objects are immutable.** Every transform returns a new object; there are no setters. This is what
makes `prec` a fixed attribute rather than a cache key: an object's approximations can never be
stale, so `functools.cached_property` is correct and thread-safe, and `lappy`'s `self._x = None`
sentinel pattern is unnecessary. It also matters for `douse`, which generates domains in a loop and
runs solves across threads.

**`prec` replaces `tol`.** `prec` is the working relative precision, default `1e-8`, valid down to
about `1e-15`. It is set at construction; a composite object takes `min(prec)` over its parts;
`.with_prec(p)` returns a new object. Every derived tolerance is a documented function of `prec`.

**Orientation.** Outer boundaries are CCW, holes are CW. The outward unit normal is then

    N(tau) = -1j * T(tau)

uniformly on every boundary of every domain, with no special-casing for holes. Signed areas add:
`area = sum(signed_area(b) for b in boundaries)` is the true area, because hole loops contribute
negatively. This one convention is why holes cost so little to support.

**Boundary conditions.** `'dir' -> 0.0`, `'neu' -> 1.0`, and any float in `[0,1]` is the Robin
parameter, as in the convex-combination form above. Kept from `lappy`, but validated once at
construction rather than through a mutable property setter.

# Where prec cannot be delivered

`prec` is a relative precision, and one configuration defeats it for reasons that are
arithmetic rather than implementation. A domain whose features have size `h` sitting at
coordinate `R` from the origin is resolved to roughly `eps * R / h` in relative terms: at
`R = 1e6` and `h = 1e-4` that is 8e-7, so a 1e-8 promise is not available in float64
however the code is written. Measured on the L-shape: 1.2e-11 at (1e3, 1e-3), 2.7e-7 at
(1e6, 1e-4), both inside the bound.

There is no post-hoc repair, and two plausible ones do not work. The bits go when the
vertices are ROUNDED into their far-from-origin coordinates, so translating the finished
domain back recovers nothing; and `normalized()` scales about the origin, which leaves the
offset-to-size ratio exactly as it was. The fix belongs upstream, with whoever builds the
domain: keep it about its own centre and carry the offset separately. Nothing at unit scale
near the origin -- which is every library domain -- is affected.

# Module layout

    src/chladni/geometry/
        __init__.py      public re-exports
        segments.py      Segment (ABC), LineSegment, ArcSegment, ParametricSegment, SplineSegment
        boundary.py      Boundary, Corner
        domain.py        Domain, Polygon
        quadrature.py    Quadrature, rules, panel breaking
        validate.py      contiguity, closure, simplicity, orientation, hole containment
        _adaptive.py     arclength table + Newton inverse, adaptive polyline
        library.py       rect, L_shape, disk, GWW1/2, polyomino, eq_tri, ...

# Segment

A segment carries a native parameter `t` in `[t0, t1]` supplied by whoever constructed it, and
exposes a **normalized arclength parameter `tau` in `[0,1]`** as the public one. The arclength map is
built lazily.

The reparameterization is ported from `lappy._reparameterize` in approach, not rewritten: an
adaptive table supplies bracketing nodes and a monotone cubic *initial guess only*, then `t_of_s`
solves `s(t) = s` by safeguarded Newton using `t'(s) = 1/|p'(t)|`. `lappy`'s docstring records what
the earlier interpolate-the-inverse approach cost: boundary integrals on an ellipse stalled at
~7e-5 *regardless of quadrature order*, because `p(tau)` was only C^1 in `tau`. Do not re-derive
this.

    class Segment(ABC)

      # identity
      bc: float                     # 0.0 dir, 1.0 neu, else Robin parameter
      prec: float
      length: float                 # cached

      # arclength parameterization, tau in [0,1]
      p(tau) -> complex[]           # position
      dp(tau) -> complex[]          # derivative; |dp| == length by construction
      T(tau) -> complex[]           # unit tangent
      N(tau) -> complex[]           # outward unit normal, -1j*T

      # endpoints
      p0, p1: complex               # p(0), p(1)
      T0, T1: complex               # T(0), T(1)

      # quadrature panel breaks
      break_ts  -> float[]          # derivative breaks in the NATIVE parameter (spline knots)
      break_taus -> float[]         # the same, mapped into tau

      # sampling and quadrature
      sample(n, *, grading=None) -> complex[]
      quad(n, *, rule='legendre', grading=None, panels=None) -> Quadrature

      # predicates and queries
      is_simple: bool               # lazy
      dist(pt) -> float
      intersects(other) -> bool
      intersection(other) -> complex[]

      # transforms, all returning a new Segment
      translated(dz)  rotated(theta, about=0j)  scaled(s, about=0j)  reversed()
      with_bc(bc)  with_prec(prec)

      # sugar
      __add__(dz)   -> translated      __mul__(s) -> scaled (complex s rotates and scales)
      __neg__()     -> reversed

`break_ts`/`break_taus` are kept from `lappy` for a measured reason: a degree-k B-spline is only
C^(k-1) at its knots, and a Gauss panel spanning a knot loses spectral convergence no matter how
accurate the parameterization is. `lappy` measured 8.3e-11 at 96 knot-aligned nodes against 1.3e-06
at 128 nodes with one global panel.

`reversed()` is new relative to `lappy` and is load-bearing: a hole is a CW loop, so constructing one
means reversing segments and their order.

Concrete subclasses: `LineSegment(p0, p1)` (exact everything, `length = abs(p1-p0)`),
`ArcSegment(center, radius, theta0, theta1)` (exact arclength; new, since `lappy` builds every
circular arc as a general `ParametricSegment` and pays the adaptive machinery for a closed form),
`ParametricSegment(p, dp, t0, t1)`, and `SplineSegment(spline)`.

# Corner

A frozen object per junction, replacing `lappy`'s four parallel arrays.

    @dataclass(frozen=True)
    class Corner:
        z: complex            # vertex location
        seg_out: int          # index of the segment whose p0 is this corner
        seg_in: int           # (seg_out - 1) % n_segments
        omega: float          # interior angle in (0, 2*pi)
        t_in, t_out: complex  # unit tangents, incoming and outgoing
        bc_in, bc_out: float
        clearance: float      # distance to the nearest NON-adjacent piece of boundary
        branch_angle: float   # direction of a ray from z into the exterior; NaN if none exists

        is_geometric: bool    # abs(omega - pi) > angle_atol
        is_reentrant: bool    # omega > pi
        is_bc_junction: bool  # bc_in != bc_out
        is_reflection_closed: bool  # omega = pi/m; an angle fact, not "singular"

A `Corner` exists exactly where `is_geometric or is_bc_junction`: where the boundary turns, or
where the boundary condition changes. A BC junction on a straight edge is a NON-GEOMETRIC corner
(`omega == pi`, `is_geometric` False) and is listed in `Domain.corners` like any other. A smooth
join between segments carrying the same condition is not a corner. There is no `is_singular` here:
whether a corner is singular depends on the BC offset, and `helmholtz.CornerModes.is_singular`
answers it (`docs/conventions.md`).

with `omega = np.angle(-t_in/t_out) % (2*np.pi)` and `angle_atol = max(prec, 1e-12)`, exposed on
`Boundary` rather than buried as a literal.

Three deliberate breaks from `lappy`:

1. **No parallel arrays.** `lappy` carries corner-indexed `corners`/`corner_idx`/`corner_angle0`/
   `corner_angle1` alongside segment-indexed `int_angles`, and its own `_find_corners` docstring
   records the resulting bug: on a half-disk `corners` had 2 entries while the angle arrays had 3,
   "so no `orders` length could be masked consistently." A tuple of `Corner` objects cannot desync.
2. **A boundary-condition junction is a corner.** `lappy`'s test is `abs(turning_angle) > 1e-10`,
   purely geometric. A straight edge where Dirichlet meets Neumann carries a genuine singularity and
   is invisible to that test; `lappy` only discovers it three layers up, in
   `eigfun_integrals.corner_specs`, as a bail-out reason. Here it is a corner, flagged by
   `is_bc_junction`, at the layer that holds both facts; whether it is singular is decided one
   layer up, by `CornerModes.is_singular`.
3. **`clearance` and `branch_angle` come home.** `corner_clearance` currently lives in
   `eigfun_integrals.py` and `corner_branch_cut_rays` in `geometry.py`, both hand-rolling the same
   `polyline()` + owner-mask pattern. Both are purely geometric. Computed once, on the corner.

Geometry stops here. It reports the angle and the BC pair, and never computes `nu = pi/omega` or
anything else about the Helmholtz solution. That is the boundary that keeps
`pts_per_seg(domain, fb_basis)` -- `lappy`'s layer-1-imports-layer-2 violation -- from having
anywhere to come back.

# Boundary

A contiguous chain of segments; closed when `p1` of the last meets `p0` of the first.

    class Boundary

      segments: tuple[Segment]
      length, seg_lengths
      is_contiguous, is_closed, is_simple      # lazy
      orientation: int                          # +1 CCW, -1 CW; requires closed and simple
      signed_area: float                        # (1/2) * Im(oint conj(z) dz)

      corners -> tuple[Corner]
      angle_atol: float

      polyline() -> (b0, b1, owner)             # adaptive sub-segment decomposition
      sample(n=None, *, n_per_seg=None, grading=None) -> complex[]
      quad(n=None, *, n_per_seg=None, rule='legendre', grading=None) -> Quadrature

      dist(pt)  reversed()  with_prec(prec)
      translated / rotated / scaled, and the matching operators
      from_vertices(vertices, bc='dir', close=True)   # classmethod

`n` is the **total** node count, distributed across segments proportionally to arclength with a
floor per segment; `n_per_seg` overrides it with explicit per-segment counts. `lappy` only offers the
latter, which pushes an allocation decision onto every caller.

# Domain

    class Domain

      outer: Boundary                # CCW, closed, simple
      holes: tuple[Boundary]         # each CW, closed, simple, inside outer, pairwise disjoint
      boundaries -> (outer,) + holes
      is_simply_connected -> len(holes) == 0

      corners -> tuple[Corner]       # across all boundaries
      area, perimeter, diameter, inradius, centroid, bbox
      bc_type -> 'dir' | 'neu' | 'mixed' | 'rob'

      contains(pts) -> bool[]        # inside outer and outside every hole
      bdry_quad(n=None, **kw) -> Quadrature       # concatenated over boundaries
      interior_pts(n, *, method='random', rng=None) -> complex[]

      normalized(by='diameter') -> Domain         # 'diameter' | 'area' | 'inradius'
      with_bc(bc)  with_prec(prec)
      translated / rotated / scaled, and the matching operators

    class Polygon(Domain)            # every segment a LineSegment
      # closed-form overrides for area, diameter, inradius, contains
      n_vertices, n_sides, vertices, edge_lengths, interior_angles

`area` is `sum(b.signed_area for b in boundaries)`, correct for holes with no branching, because a
hole's CW loop contributes a negative signed area.

`normalized()` returns a new `Domain` scaled so the named quantity is 1. The caller recovers the
factor `s` from the old value; eigenvalues transform as `lambda_new = lambda_old * s**2`.

`interior_pts` uses rejection sampling in the bounding box against `contains`, or a filtered
lattice. This deliberately drops `lappy`'s `pygmsh`/Dunavant meshing path: the solver needs interior
points for GSVD regularization, not an interior cubature rule, and `area` comes from the boundary by
Green's theorem. Interior *cubature* is deferred until something asks for it.

# Quadrature

The central new abstraction, and the main structural correction to `lappy`.

    @dataclass(frozen=True)
    class Quadrature:
        pts, normals, tangents: complex[]
        wts: float[]                 # see "what the weights integrate" below
        tau: float[]                 # normalized arclength within the owning segment
        seg_idx, bdry_idx: int[]     # provenance
        bc: float[]                  # BC parameter at each node
        panel_id: int[]
        panels: tuple
        weight_family: str = 'arclength'

        dir_mask, neu_mask -> bool[]
        integrate(f) -> complex      # (f(pts) * wts).sum()
        radial(x0=0j) -> float[]     # (x - x0) . n, i.e. Re(conj(pts - x0) * normals)

## What the weights integrate

`integrate(f)` is always `(f(pts) * wts).sum()`, and in every family it approximates
`oint f ds` -- the measure is always arclength. The family names the class of integrands the
rule is EXACT on:

- `weight_family == 'arclength'` -- smooth integrands, and `wts.sum() == length`. Everything
  geometry builds is in this family, `grading` included: grading relocates nodes, it does not
  change the class.
- `'even'` and `'integer'` -- corner rules built one layer up (`chladni.integrals`), exact on
  the singular integrands at a corner. Their weights carry the substitution's Jacobian with the
  Jacobi weight divided OUT, so the caller passes the raw integrand; they are not exact on
  constants, so `wts.sum()` is **not** the perimeter, by design.

*Corrected 2026-10-05.* This section used to say a non-arclength rule had a singular weight `w`
"folded in", with `integrate(f) == oint f w ds` and the caller passing `f = integrand / w`. That
is not what lappy built, nor what `chladni.integrals` builds: lappy's `cornerjacgauss` divides the
Jacobi weight out (`lappy/quad.py:29-37`, `:186-233`), so the rule integrates the raw integrand
and the family only labels its exactness. Nothing in `chladni` ever relied on the folded form.

The corner families are what the Rellich boundary integrals need: near a singular corner the
integrand carries `r^(2*nu-2)`, and the only rules that resolve it to working precision are
Gauss-Jacobi rules built around that exponent. Those rules need `nu`, so they are built one
layer up -- but they are returned as `Quadrature`, which is why `weight_family` lives on this
class rather than in the layer that sets it. lappy reached the same conclusion, and its
values (`'even'` for `sub = nu`, matched to the eigenfunction's own exponent family, and
`'integer'` for `sub = 1/2`, exact on the integer powers a corner-moving shape velocity
supplies) carry over unchanged. Its comment records that the two "are not interchangeable in
either direction."

Consequently `wts.sum() == length`, `oint n ds == 0`, and `oint (x-x0).n ds == 2*area` are
identities **of the arclength family only**. They are the strongest checks available on a
geometry-built rule, and they are meaningless applied to a singular one; a consumer that
wants them must test `weight_family` first rather than assume.

`lappy`'s `Segment.pts(n, kind='legendre')` does `tau, _wts = quadfunc(n)` and **throws the weights
away**; `MultiSegment.pts` then reassembles per-segment results with `np.sum([PointSet, ...])`,
abusing `+` as concatenation. Rellich normalization integrates `(x-x0).n * |grad u|^2` over the
boundary, so the weights have to survive. The fields here mirror `lappy`'s `BoundaryQuad` -- itself
documented as "geometry-only boundary quadrature: no domain reference, no lam dependence, no
basis," which is exactly a layer-1 object that ended up in `eigfun_integrals.py`. `seg_idx` and
`tau` are carried for the reason `lappy` gives: without them a downstream shape package has to call
private helpers and redo the affine panel map by hand.

`radial(x0)` is a method rather than a stored `x0` field, because the Rellich base point is a solver
choice while the arithmetic is pure geometry.

Scope: geometry owns **nodes, weights, and provenance**. Panels break at corners and at
`break_taus`, and `grading` is a hook for clustering nodes toward corner endpoints. Geometry does
**not** own corner-exponent-aware Gauss-Jacobi rules -- those need `nu`, and stay in the layer that
knows it.

# Validation

One `validate=True` flag, replacing `lappy`'s three (`val_simple`, `val_closed`, `val_orientation`,
whose defaults disagree between `ParametricSegment` and `MultiSegment`). Opt-out, as in `lappy`;
`douse` opts out.

- **Always checked**, being O(n) on data already present: contiguity, closure, non-degeneracy
  (zero-length segments, zero signed area).
- **Checked under `validate`**, being the O(n^2) work: simplicity, hole-inside-outer, hole
  disjointness. Use `shapely` -- already a dependency -- rather than `lappy`'s hand-rolled pairwise
  intersection loop.
- **Auto-orient rather than raise.** A CW outer boundary is silently reversed; a CW hole likewise.
  `lappy` raises `"boundary must be in CCW (positive) orientation"`, which makes every caller
  pre-sort vertices -- wrong for an optimizer trialling wild polygons.
- `.is_valid` remains available lazily for callers that passed `validate=False`.

# Ported from lappy

`cache.py` (byte-budgeted instance caches; its docstring records a five-worker `douse` grid taking a
16 GB machine to 15.4 GB of swap, and why neither `lru_cache` nor a per-instance cache works -- do
not re-derive), `adaptive_arclength_table` and `adaptive_polyline`, the arclength Newton solve,
`break_ts`/`break_taus`, the BC float encoding, and the domain library.

# Dropped

`pygmsh` and the Dunavant cubature registry; `PointSet` (immutable complex arrays hash by value via
`.tobytes()` without a wrapper class); `len` as a property name, in favour of `length`; and
`pts_per_seg`, which cannot exist here.

# Ground truth for the test suite

Written before the implementation, per the project's TDD principle.

- **Exact, closed form:** polygon area, perimeter, interior angles, normals; `LineSegment` and
  `ArcSegment` arclength; the equilateral triangle's `pi/3` corners; regular n-gon inradius.
- **Special functions:** ellipse arclength against complete/incomplete elliptic integrals
  (`scipy.special.ellipeinc`), with `mpmath` where more digits are wanted.
- **Invariants:** `|dp(tau)| == length` to `prec` on every segment type (this is the property the
  Newton reparameterization exists to guarantee, and the one the old interpolated inverse violated
  at 8.6e-3); `wts.sum() == length`; `quad.integrate(1) == length`; area by Green's theorem against
  the closed form; `contains` against a lattice for a shape with a hole.
- **Regression cases from lappy:** the half-disk, whose smooth `pi` junction desynced the corner
  arrays; the L-shape's reentrant `3*pi/2` corner; a Dirichlet/Neumann junction on a straight edge,
  which `lappy` classifies as no corner at all.


# As built: deltas from this spec

Written after the geometry layer landed (355 tests, CI green on 3.11-3.13). Everything above
still describes the intent correctly; this records where the code went further, and why. The
live record is the code, the test suite, and `examples/geometry_tour.ipynb`.

## Corner grew four fields

The spec's `Corner` predates all of these, each added for a reason found by testing:

- `bdry_idx` -- `Domain.corners` flattens across boundaries while `seg_out`/`seg_in` stay
  relative to the owning one, so on a square with a square hole, corner 0 of the outer
  boundary and corner 0 of the hole **both reported `seg_out = 0`**. `Quadrature` already
  carried `bdry_idx` for exactly this reason.
- `segment_out` / `segment_in` -- direct references *alongside* the indices, not instead of
  them. Indices are the join key into `Quadrature`'s integer arrays; the references are the
  accessor, because the first thing a caller does with `clearance` is divide it by the
  adjacent segment's length. Safe only because segments are immutable, and they point at
  leaves -- never at the owning `Boundary` or `Domain`, which would close the reference cycle
  `lappy`'s `cache.py` exists to undo. Excluded from equality and repr.
- `on_hole`, with the derived `admits_exterior_branch_cut` -- a loop in Omega encircling a
  hole winds around every point of that hole's boundary, so omega measured from a corner on
  the hole gains 2*pi around it. Killing that loop needs a cut joining hole to outer boundary,
  which must cross Omega. The obstruction is topological, not a search that needs to try
  harder. `branch_angle` stays finite there -- it is still a real direction into the adjacent
  exterior component, useful for placing a fundamental solution's source inside the hole -- but
  it is the flag, not the angle, that says whether a cut exists.
- `branch_vertices`, with `branch_path(length)` -- the spec assumed a branch cut is always a
  ray. On a spiral, half the corners are *surrounded*: their exterior is a channel that turns,
  and no ray escapes. Those get a wall-following polyline cut, ported from `lappy`'s
  `corner_branch_cut_polyline`. `branch_angle` is then the final ray's direction.

## `geometry` names the angle, `helmholtz` names the singularity

The spec defines `is_singular` as `is_geometric or is_bc_junction` (line 174), and the code
shipped that. It is the wrong set, and strictly too large: it calls every right angle of a
rectangle singular. The local orders at a corner are `mu = (l + offset)*nu` with
`nu = pi/omega` and `offset = 1/2` when the two sides sit in different families, so the corner
is regular exactly when all of them are integers. A right angle under one boundary condition
gives `nu = 2` and orders 2, 4, 6 -- analytic, and needing neither a fractional expansion nor
a graded panel. Two cases the old test got backwards:

- a flat Dirichlet/Neumann junction, `omega = pi`, is **singular** -- orders 1/2, 3/2, the
  strongest in the project, at a corner `is_geometric` cannot see. Delta 2 of the spec is
  right about this corner mattering; it just named the wrong condition;
- a **right-angled** Zaremba junction is **regular**: `nu = 2`, `offset*nu = 1`, orders
  1, 3, 5.

The first attempt was to redefine `is_singular` here, on `Corner`, with an `is_zaremba` to
supply the offset. That is the right set and the wrong layer, for two reasons. `chladni` is an
eigensolver, so "singular corner" reads as a claim about the eigenfunction, which is not a
claim this layer is in a position to make; and `is_zaremba` folds Robin into the Neumann
family, which is Grisvard's leading-asymptotics result, not a fact about the boundary.

So the predicate splits along the layer line instead:

- **`Corner.is_reflection_closed`** -- `near_integer(pi/omega)`, i.e. `omega = pi/m`. The
  reflections in the two edges generate a finite group of order 2m whose images tile a
  neighbourhood of the vertex exactly. Purely an angle fact: no BC, no equation, no solution.
  Reentrant corners are never closed, since `omega > pi` puts `nu` in `(1/2, 1)`.
- **`CornerModes.is_singular`** (`helmholtz`) -- the eigenfunction is non-smooth here. The
  named negation of `is_single_valued`, and the only place in the project that says the word
  unqualified. `docs/conventions.md` records the reservation, along with the two other live
  senses -- "generalized singular value" and "singular weight" -- that must be spelled out.

`Corner` keeps no `is_singular` and no `is_zaremba`.

### What `grading="corner"` does now, and why it is provisional

`_grade_flags` grades where `not is_reflection_closed or is_bc_junction`. That is a superset of
the singular set, and provably: a nonzero offset requires the two sides in different families,
which requires the bc values to differ. So geometry never under-grades. It over-grades exactly
one shape of corner -- a right-angled Dirichlet/Neumann vertex -- and the honest statement of
the rule is *geometry grades where the angle alone cannot rule out a singularity, plus every BC
junction, because the offset that would settle those is not this layer's fact*. Measured
against every constructor in `chladni.geometry.library`: zero difference from the exact set,
because no library domain has a BC junction. The only corner in the repo where the containment
is strict is `z = 1` of the `split_edge_bc_junction` fixture, and a test now pins it.

This is a placeholder, not a design. Node placement wants the integrand: the Rellich and
Hadamard boundary integrals are bilinear in Cauchy data against a weight, so choosing a rule
needs geometric, spectral and functional information at once, and none of the three belongs to
the other two. When a quadrature layer exists to hold all three it should own the choice and
reach in through `panels=`, which `Panel`'s docstring already describes as the seam for exactly
that. `geometry` keeps what it is actually good for -- the arclength map, the frames, the
provenance, the BC tagging.

Two stale claims the split turned up and fixed: `Domain.corners` said it returned "every
singular corner" and returns all of them, and `admits_exterior_branch_cut` said `omega = pi/m`
was "the caller's test to make, since geometry does not compute nu" -- that test now has a name
here, and the hole rule reads `admits_exterior_branch_cut or is_reflection_closed`, with the
mixed-corner parity condition staying in `helmholtz`.

## Clearance is a domain-level quantity

`Boundary.corners` can only see its own boundary, so its `clearance` is an upper bound.
`Domain.corners` tightens it against the other boundaries: on a 3x3 square with a unit hole,
the corner at the origin reported 3.0 against a true sqrt(2). Clearance sizes corner panels,
where `lappy` measured up to twelve orders of accuracy riding on that length.

## The panel seam is real API

`Segment.quad`, `Boundary.quad` and `Domain.bdry_quad` all take `panels=` and
`weight_family=`, with `Panel` and `legendre_nodes` exported. A plan's `nodes(tau0, tau1)`
returns weights **in tau**; geometry multiplies by the segment length to reach `ds` and does
nothing else to them. Rescaling a singular weight onto a shorter panel picks up a power of the
panel width set by that weight's homogeneity, and only the layer that chose the weight knows
the exponent -- geometry owning that arithmetic would be a silent factor-of-h bug waiting for
the first corner-adapted rule.

## Precision is reported, not just requested

`Domain.achievable_prec` and `PrecisionWarning` (in `chladni.geometry`). A feature of size h at
coordinate R resolves to about `eps*R/h`, and an edge much shorter than the domain to about
`eps*h/min_edge`. Construction warns when that falls short of `prec`. The check does **not**
respect `validate=False`: that flag skips the O(n^2) geometric checks, and gating on it was
actively wrong, because transforms rebuild with `validate=False` -- so `translated()`, the very
operation that moves a domain away from the origin, created the problem silently.

## Other additions

- A module the layout above does not list: `_core.py`, holding the `Frozen` base, the BC
  encoding, `prec` validation, `as_xy`/`from_xy` and `PrecisionWarning`.
- `Boundary`: `bbox`, `free_ray_from`, cached `polyline()`.
- `Domain`: `boundary_distance`, `is_valid`, `plot()`, `plot_diagnostic()`.
- `Quadrature`: `concat`, `rob_mask`.
- Library: `annulus`, `reg_ngon`, `iso_right_tri`, `plus_shape`, `spiral`.
- A whole `chladni.plotting` package, specced separately. `geometry` never imports matplotlib;
  the convenience `.plot()` methods import inside the method body, and a test asserts the rule
  in a subprocess.

## Performance decisions with no home in this spec

All measured, all with the number that motivated them recorded at the code:

- Gauss order per panel capped at 32. `leggauss` is O(order^2) -- 39 ms at order 1000 -- and a
  shape-optimization loop changes the per-segment split every iterate, so an order-keyed cache
  missed every time.
- `contains` uses exact ray-crossing for lines and arcs, never the polyline. An inscribed
  polyline makes a hole slightly too small, so points in its chord gaps read as interior.
- The polyline fallback picks between an outer product and a sort-and-sweep by problem size;
  the sweep alone made polygons 19x slower, the outer product alone wanted 13 GB on a spline.
- `Corner` construction is batched over corners. Per-corner it was O(n^2): 534 ms for a
  400-gon, which is the size range a shape optimizer generates.
- Segment `bbox` is closed-form, so a curved domain never materializes its prec-resolution
  polyline during a solve-shaped lifecycle.
- `refine()` takes a `scale` floor. A purely relative convergence test cannot converge on a
  quantity whose true value is zero; the disk's centroid drove n to 8192 and one test took 55
  of the suite's 56 seconds.
