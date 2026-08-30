"""What the layers above `helmholtz` depend on.

`mps` will be written against this file, the way `helmholtz` was written against
`test_solver_seam.py`. Everything here is a contract that, if missing, forces the solver to
reach into this layer's internals -- which is how lappy ended up with `make_bdry_vander`
branching on boundary conditions and an `isinstance` ladder that had to know every concrete
basis type.

Three things are being pinned down.

The Sampling IS the cache. lappy hid the k-independent geometry behind 370 lines of
byte-budgeted `instance_cache` with a lock, a global ledger and weakref finalizers. Here the
solver holds the object, so it must be safe to hold: immutable, reusable across hundreds of
k, and never mutated by an evaluation.

`trace(k)` assembles the boundary matrix. `Quadrature` already carries pts, normals and bc,
so the solver should not be reassembling `(1-bc)*u + bc*du_dn` itself, and a Robin or
mixed-BC boundary should not pay for two Bessel passes to get it.

`singular_exponents` ships with its consumer. Geometry's `Panel` seam exists so a
corner-graded Gauss-Jacobi rule can be built one layer up; that rule needs a leading exponent
per column. lappy's analogous `corner_terms` has a detailed contract and no production
consumer at all.
"""
import numpy as np
import pytest

from conftest import EXACT, L_SHAPE
from chladni.geometry import Domain, LineSegment, Boundary, Panel, Polygon, legendre_nodes
from chladni.helmholtz import FourierBessel, Fundamental, PlaneWave

K = 3.3


def basis_for(poly):
    fb = sum((FourierBessel.at_corner(c, n=3) for c in poly.corners[1:]),
             FourierBessel.at_corner(poly.corners[0], n=3))
    return fb + Fundamental(np.array([-1 - 1j, 3 + 3j]), radius=1.0)


# --- the boundary matrix ----------------------------------------------------------------------

def test_trace_on_a_dirichlet_boundary_is_the_value():
    poly = Polygon(L_SHAPE, bc="dir")
    q = poly.bdry_quad(200)
    b = basis_for(poly)
    assert np.array_equal(b.at(q).trace(K), b.values(K, q.pts))


def test_trace_on_a_neumann_boundary_is_the_normal_derivative():
    poly = Polygon(L_SHAPE, bc="neu")
    q = poly.bdry_quad(200)
    b = basis_for(poly)
    assert np.array_equal(b.at(q).trace(K), b.deriv(K, q.pts, q.normals))


def test_trace_on_a_robin_boundary_is_the_convex_combination():
    poly = Polygon(L_SHAPE, bc=0.3)
    q = poly.bdry_quad(200)
    b = basis_for(poly)
    v = b.values(K, q.pts)
    dn = b.deriv(K, q.pts, q.normals)
    want = 0.7 * v + 0.3 * dn
    assert np.allclose(b.at(q).trace(K), want, rtol=0, atol=EXACT * np.abs(want).max())


def test_trace_handles_a_boundary_with_different_conditions_on_different_segments():
    """The case lappy's `make_bdry_vander` cannot express: it branches once, on whether ALL
    the boundary is Dirichlet or ALL of it is Neumann, and otherwise takes the Robin path."""
    segs = [LineSegment(L_SHAPE[i], L_SHAPE[(i + 1) % len(L_SHAPE)],
                        bc="dir" if i % 2 else "neu") for i in range(len(L_SHAPE))]
    poly = Domain(Boundary(segs))
    q = poly.bdry_quad(240)
    assert q.dir_mask.any() and q.neu_mask.any()
    b = Fundamental(np.array([-1 - 1j, 3 + 3j, 3 - 1j]), order=2, radius=1.0)
    got = b.at(q).trace(K)
    v, dn = b.values(K, q.pts), b.deriv(K, q.pts, q.normals)
    assert np.allclose(got[q.dir_mask], v[q.dir_mask], rtol=0, atol=EXACT)
    assert np.allclose(got[q.neu_mask], dn[q.neu_mask], rtol=0, atol=EXACT)


def test_trace_needs_a_quadrature_and_says_so():
    poly = Polygon(L_SHAPE)
    b = basis_for(poly)
    with pytest.raises(ValueError, match="Quadrature"):
        b.at(np.array([0.5 + 0.5j])).trace(K)


def test_trace_takes_one_pass_not_two():
    """A Robin boundary needs the value and the normal derivative at the same k. Taking them
    from one `values_and_grad` rather than two `_eval`s is the whole reason that method
    exists -- lappy paid for `_bessel` twice at the same parameter because `_eval_pointset`
    and `_grad_pointset` had no memo between them."""
    poly = Polygon(L_SHAPE, bc=0.4)
    q = poly.bdry_quad(400)
    s = basis_for(poly).at(q)
    calls = []
    orig = type(s)._eval
    object.__setattr__(s, "_eval",
                       lambda k, c, g, _o=orig: (calls.append(g), _o(s, k, c, g))[1])
    s.trace(K)
    assert calls == [True], f"trace made {len(calls)} evaluation passes"


# --- the Sampling is the cache ----------------------------------------------------------------

def test_one_sampling_serves_many_k():
    poly = Polygon(L_SHAPE)
    q = poly.bdry_quad(150)
    s = basis_for(poly).at(q)
    ks = np.linspace(2.0, 12.0, 40)
    first = {k: s.trace(k).copy() for k in ks[:3]}
    for k in ks:
        s.trace(k)
    for k, want in first.items():
        assert np.array_equal(s.trace(k), want)


def test_a_sampling_is_immutable():
    poly = Polygon(L_SHAPE)
    s = basis_for(poly).at(poly.bdry_quad(60))
    with pytest.raises(AttributeError):
        s.pts = np.array([0j])


def test_values_and_grad_equals_the_two_taken_separately():
    poly = Polygon(L_SHAPE)
    q = poly.bdry_quad(120)
    s = basis_for(poly).at(q)
    v, g = s.values_and_grad(K)
    assert np.array_equal(v, s.values(K))
    assert np.array_equal(g, s.grad(K))


def test_the_sugar_and_the_bound_form_agree():
    poly = Polygon(L_SHAPE)
    pts = poly.interior_pts(50, rng=np.random.default_rng(1))
    b = basis_for(poly)
    assert np.array_equal(b.values(K, pts), b.at(pts).values(K))


def test_an_interior_point_set_is_a_sampling_too():
    """`Domain.interior_pts` is what the GSVD regularization consumes; it is not a
    Quadrature, and it must not have to be."""
    poly = Polygon(L_SHAPE)
    pts = poly.interior_pts(80, rng=np.random.default_rng(0))
    s = basis_for(poly).at(pts)
    assert s.values(K).shape == (80, len(basis_for(poly)))
    assert s.normals is None and s.bc is None


# --- the graded-panel seam -----------------------------------------------------------------------

def test_singular_exponents_give_a_graded_rule_what_it_needs():
    """A corner-adapted Gauss-Jacobi panel is graded to `r**(2*p - 2)`, so it needs `p` per
    column and the origin `p` is measured from. That is exactly this, and nothing else in the
    layer supplies it."""
    poly = Polygon(L_SHAPE)
    b = basis_for(poly)
    s = b.singular_exponents()
    singular = np.isfinite(s.origin)
    assert singular.sum() == 3 * len(poly.corners)
    assert np.all(s.exponent[singular] > 0)
    assert np.all(s.exponent[~singular] == 0)
    # every singular origin is a corner vertex
    for z in np.unique(s.origin[singular]):
        assert np.min(np.abs(np.array([c.z for c in poly.corners]) - z)) < EXACT


def test_the_leading_exponent_at_the_reentrant_corner_is_two_thirds():
    poly = Polygon(L_SHAPE)
    c = [c for c in poly.corners if c.is_reentrant][0]
    fam = FourierBessel.at_corner(c, n=4)
    assert fam.singular_exponents().exponent.min() == pytest.approx(2 / 3, abs=EXACT)


def test_a_caller_supplied_panel_plan_reaches_this_layer_intact():
    """Geometry's `Panel` seam exists so a corner-graded rule can be built up here. Whatever
    rule arrives, the families evaluate on its nodes with no special casing."""
    poly = Polygon(L_SHAPE)
    n_seg = len(poly.boundaries[0].segments)
    plan = [Panel(0, i, 0.0, 1.0, legendre_nodes(16)) for i in range(n_seg)]
    q = poly.bdry_quad(panels=plan)
    b = basis_for(poly)
    assert b.at(q).trace(K).shape == (len(q), len(b))


# --- k, and the seam to lam ------------------------------------------------------------------------

def test_k_must_be_a_positive_real():
    poly = Polygon(L_SHAPE)
    s = basis_for(poly).at(poly.bdry_quad(40))
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="k must"):
            s.values(bad)


def test_scaling_the_domain_by_s_and_k_by_one_over_s_is_the_same_problem():
    """`Domain.normalized(by=s)` sends `lam -> lam*s**2`, i.e. `k -> k*s`. The families
    transform with the geometry, and the scaled columns are unchanged -- which is the
    statement that this layer is covariant with the geometry layer."""
    poly = Polygon(L_SHAPE)
    c = poly.corners[0]
    fam = FourierBessel.at_corner(c, n=4)
    pts = poly.interior_pts(30, rng=np.random.default_rng(2))
    s = 3.0
    got = fam.scaled(s).values(K / s, pts * s)
    assert np.allclose(got, fam.values(K, pts), rtol=1e-11, atol=1e-300)


# --- the cost of the seam ------------------------------------------------------------------------

def test_a_dirichlet_trace_does_not_pay_for_the_derivative():
    """The scipy branch gets `J'` from `J_{nu-1} - J_{nu+1}`, so asking for it TRIPLES the
    number of `jv` calls -- and a Dirichlet boundary, the common case, never wants it.
    Measured on a 600 x 144 L-shape block: 26.3 ms with the derivative, 13.2 ms without.

    Pinned as a contract rather than as a timing, so it survives a slow machine: `log_j`
    must return `Sp is None` when the derivative was not asked for, and the value path must
    not ask.
    """
    from chladni.helmholtz._bessel import log_j

    assert log_j(2.0, 3.0, derivative=False).Sp is None
    assert log_j(2.0, 3.0).Sp is not None

    poly = Polygon(L_SHAPE, bc="dir")
    s = basis_for(poly).at(poly.bdry_quad(120))
    asked = []
    orig = type(s)._eval
    object.__setattr__(s, "_eval",
                       lambda k, c, g, _o=orig: (asked.append(g), _o(s, k, c, g))[1])
    s.trace(K)
    assert asked == [False]


def test_skipping_the_derivative_does_not_change_the_values():
    from chladni.helmholtz._bessel import log_j, log_y

    nu = np.array([0.5, 3.0, 40.0, 300.0])[None, :]
    x = np.array([0.2, 4.0, 30.0, 90.0])[:, None]
    assert np.array_equal(log_j(nu, x, derivative=False).S, log_j(nu, x).S)
    m = np.array([0, 1, 5, 90])[None, :]
    assert np.array_equal(log_y(m, x, derivative=False).S, log_y(m, x).S)
