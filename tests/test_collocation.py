"""Matrix assembly and pencil conditioning: everything before the GSVD.

Two claims are being defended.

`A_B` is the trace and `A_I` is the values, and this layer contains no boundary-condition
logic at all. lappy's `make_bdry_vander` branches on `np.all(bc_param == 0)` and
`np.all(bc_param == 1)`, so a boundary carrying Dirichlet on some segments and Neumann on
others falls into the ROBIN path and is expressed as a degenerate convex blend. Here
`Sampling.trace` does it one layer down, per node, and the mixed case is not a case.

Equilibration belongs to `mps` and accuracy belongs to `helmholtz`. `conventions.md` splits
them: `helmholtz` hands back columns pre-scaled by `exp(-sigma)` so they are representable at
all, and `mps` rescales for pencil conditioning. The two must COMPOSE -- the log scales add --
or a coefficient vector stops meaning the same function on different point sets, which is the
property the whole scaled-column design exists for.
"""
import numpy as np
import pytest

from chladni.geometry import Boundary, Domain, LineSegment, Polygon
from chladni.helmholtz import FourierBessel, Fundamental, PlaneWave
from chladni.mps import MPSPencil, assemble, equilibrate

from conftest import EXACT, L_SHAPE

K = 3.3
LAM = K**2
RNG = 0


def basis_for(poly):
    fb = sum((FourierBessel.at_corner(c, n=3, domain=poly) for c in poly.corners[1:]),
             FourierBessel.at_corner(poly.corners[0], n=3, domain=poly))
    return fb + Fundamental(np.array([-1 - 1j, 3 + 3j]), radius=1.0)


@pytest.fixture
def pencil():
    poly = Polygon(L_SHAPE, bc="dir")
    b = basis_for(poly)
    return MPSPencil.from_basis(b, poly.bdry_quad(300),
                                poly.interior_pts(120, rng=np.random.default_rng(RNG)),
                                lam_ref=LAM)


# --- assembly is the trace and the values -----------------------------------------------


@pytest.mark.parametrize("bc", ["dir", "neu", 0.3])
def test_A_B_is_the_trace_and_A_I_is_the_values(bc):
    poly = Polygon(L_SHAPE, bc=bc)
    b = basis_for(poly)
    q = poly.bdry_quad(200)
    pts = poly.interior_pts(50, rng=np.random.default_rng(RNG))
    A_B, A_I = assemble(b.at(q), b.at(pts), K)
    assert np.array_equal(A_B, b.at(q).trace(K))
    assert np.array_equal(A_I, b.values(K, pts))


def test_a_mixed_boundary_needs_no_special_case():
    """The case lappy's `make_bdry_vander` cannot express. Here it is not a case at all:
    `trace` reads `bc` per node, so this test only has to check that nothing upstream
    collapsed the per-node vector into a scalar decision."""
    segs = [LineSegment(L_SHAPE[i], L_SHAPE[(i + 1) % len(L_SHAPE)],
                        bc="dir" if i % 2 else "neu") for i in range(len(L_SHAPE))]
    poly = Domain(Boundary(segs))
    q = poly.bdry_quad(240)
    b = Fundamental(np.array([-1 - 1j, 3 + 3j, 3 - 1j]), order=2, radius=1.0)
    pts = poly.interior_pts(40, rng=np.random.default_rng(RNG))
    A_B, _ = assemble(b.at(q), b.at(pts), K)
    v, dn = b.values(K, q.pts), b.deriv(K, q.pts, q.normals)
    assert np.allclose(A_B[q.dir_mask], v[q.dir_mask], rtol=0, atol=EXACT)
    assert np.allclose(A_B[q.neu_mask], dn[q.neu_mask], rtol=0, atol=EXACT)


def test_shapes(pencil):
    A_B, A_I = pencil.matrices(LAM)
    assert A_B.shape == (pencil.n_bdry, pencil.n_cols)
    assert A_I.shape == (pencil.n_int, pencil.n_cols)
    assert A_B.dtype == np.float64 and A_I.dtype == np.float64


# --- construction-time validation -------------------------------------------------------


def test_the_boundary_sampling_must_come_from_a_quadrature():
    """Bare points carry no normals and no bc, so `trace` cannot assemble anything. Fail at
    construction, not three hundred evaluations later."""
    poly = Polygon(L_SHAPE)
    b = basis_for(poly)
    bare = b.at(poly.bdry_quad(100).pts)
    with pytest.raises(ValueError, match="Quadrature"):
        MPSPencil(bare, b.at(poly.interior_pts(40, rng=np.random.default_rng(RNG))), lam_ref=LAM)


def test_the_two_samplings_must_share_a_column_space():
    poly = Polygon(L_SHAPE)
    a = basis_for(poly)
    other = PlaneWave.uniform(8, radius=2.0)
    with pytest.raises(ValueError, match="same"):
        MPSPencil(a.at(poly.bdry_quad(100)),
                  other.at(poly.interior_pts(40, rng=np.random.default_rng(RNG))), lam_ref=LAM)


def test_the_default_truncation_rule_is_threshold():
    """A truncation with no `rank` and no `lam_ref`: the rank is re-derived at every `lam`.

    Of the truncating rules this is the one used whenever a truncation is (`reg="auto"` below
    `prec = 1e-8`, or `reg="svd"`), because it is the only one that cannot pick an UNSAFE rank.
    The pencil's numerical rank grows with `lam` -- 31 to 47 across a disk scan -- so any fixed
    integer is too large somewhere, and a rank sitting at machine epsilon turns the tension
    rough: 85 spurious local minima in a window holding one eigenvalue. Threshold
    re-derives from the spectrum actually in front of it and cannot make that mistake.
    """
    poly = Polygon(L_SHAPE)
    b = basis_for(poly)
    p = MPSPencil(b.at(poly.bdry_quad(200)),
                  b.at(poly.interior_pts(80, rng=np.random.default_rng(RNG))), reg="svd")
    assert p.rank is None and not p.is_frozen
    assert p.rank_at(LAM) == len(p.tensions(LAM))
    assert "threshold" in repr(p)


def test_the_default_at_the_default_prec_chooses_no_rank_at_all():
    """`reg="auto"` at `prec = 1e-8` is the penalty (`tests/test_regularization.py` has the
    range): nothing to freeze and nothing re-derived, and every column yields a tension."""
    poly = Polygon(L_SHAPE)
    b = basis_for(poly)
    p = MPSPencil(b.at(poly.bdry_quad(200)),
                  b.at(poly.interior_pts(80, rng=np.random.default_rng(RNG))))
    assert p.reg == "tikhonov_qr" and p.rank is None and not p.is_frozen
    assert p.rank_at(LAM) == len(p.tensions(LAM)) == p.n_cols


def test_a_frozen_rank_still_needs_a_reference_or_a_value():
    """Freezing is still available and still refuses to guess where to freeze."""
    poly = Polygon(L_SHAPE)
    b = basis_for(poly)
    q, pts = poly.bdry_quad(200), poly.interior_pts(80, rng=np.random.default_rng(RNG))
    frozen = MPSPencil(b.at(q), b.at(pts), lam_ref=(9.0, 16.0))
    assert frozen.is_frozen and isinstance(frozen.rank, int)
    assert MPSPencil(b.at(q), b.at(pts), rank=12).rank == 12


# --- equilibration ----------------------------------------------------------------------


def test_equilibrated_columns_have_unit_norm_over_the_STACK(pencil):
    """Not over either block separately: the GSVD sees the pair, and normalizing the blocks
    independently would change the ratio the tension measures."""
    B, I = pencil.pencil(LAM)
    norms = np.linalg.norm(np.vstack([B, I]), axis=0)
    assert np.allclose(norms, 1.0, rtol=0, atol=1e-14)


def test_equilibration_does_not_change_the_column_SPAN(pencil):
    """A diagonal rescale, and nothing else. If it were anything else the tension would not
    be the tension of the problem that was posed."""
    A_B, A_I = pencil.matrices(LAM)
    B, I, log_norms = equilibrate(A_B, A_I)
    assert np.allclose(B * np.exp(log_norms), A_B, rtol=1e-13, atol=0)
    assert np.allclose(I * np.exp(log_norms), A_I, rtol=1e-13, atol=0)


def test_a_column_that_underflowed_to_zeros_survives_equilibration():
    """Dividing by its norm would be a divide-by-zero. lappy pruned such columns by hand;
    here they keep scale 1 and are left for the rank truncation to drop."""
    A_B = np.random.default_rng(0).standard_normal((20, 5))
    A_I = np.random.default_rng(1).standard_normal((10, 5))
    A_B[:, 2] = 0.0
    A_I[:, 2] = 0.0
    B, I, log_norms = equilibrate(A_B, A_I)
    assert np.all(np.isfinite(B)) and np.all(np.isfinite(I))
    assert log_norms[2] == 0.0
    assert np.all(B[:, 2] == 0.0)


def test_the_log_scales_add(pencil):
    """`conventions.md`: "the log scales simply add, and a coefficient vector means the same
    function on every point set." The pencil's total scale is the family's plus its own."""
    A_B, A_I = pencil.matrices(LAM)
    _, _, own = equilibrate(A_B, A_I)
    assert np.allclose(pencil.log_scale(LAM), pencil.basis.log_scale(K) + own,
                       rtol=0, atol=EXACT)


# --- a coefficient vector means the same function everywhere -----------------------------


@pytest.mark.not_under_jitter  # compares assemble with itself; tests/ulp_jitter.py patches assemble
def test_a_coefficient_vector_means_the_same_function_on_a_second_point_set(pencil):
    """The property the whole scaled-column design exists for.

    A vector `c` in the equilibrated column space names the function
    `sum_j c_j * exp(-own_j) * psi_j`. `own` is a per-COLUMN constant, so that function is
    defined independently of any point set -- and evaluating it through the pencil's own
    interior block must agree with evaluating it from the basis on points chosen afterwards.
    If `own` depended on the points, these two would disagree and an eigenfunction could not
    be evaluated off the collocation set at all.
    """
    poly = Polygon(L_SHAPE)
    c = np.random.default_rng(5).standard_normal(pencil.n_cols)
    own = equilibrate(*pencil.matrices(LAM))[2]

    # through the pencil, on its own interior points
    _, I = pencil.pencil(LAM)
    here = I @ c

    # from the basis, on the same points, undoing only the pencil's own scale
    there = pencil.basis.values(K, pencil.int_pts) @ (c * np.exp(-own))
    assert np.allclose(here, there, rtol=1e-12, atol=1e-300)

    # and on points the pencil has never seen, the function is still well defined
    other = poly.interior_pts(37, rng=np.random.default_rng(99))
    assert np.all(np.isfinite(pencil.basis.values(K, other) @ (c * np.exp(-own))))


# --- covariance with the geometry layer --------------------------------------------------


def test_scaling_the_domain_scales_the_spectrum(pencil):
    """`Domain.normalized(by=s)` sends `lam -> lam*s**2`, i.e. `k -> k*s`. The tension at
    corresponding parameters must agree, or this layer is not covariant with the one two
    below it -- the same statement `test_helmholtz_seam` makes for the families."""
    poly = Polygon(L_SHAPE, bc="dir")
    s = 2.5
    pts = poly.interior_pts(90, rng=np.random.default_rng(RNG))
    b = basis_for(poly)

    small_p = MPSPencil.from_basis(b, poly.bdry_quad(240), pts, lam_ref=LAM)
    # the SAME points and the SAME basis, both carried over by the transform protocol, so
    # the only difference between the two pencils is the scale
    big_p = MPSPencil.from_basis(b.scaled(s), poly.scaled(s).bdry_quad(240), pts * s,
                                 lam_ref=LAM / s**2)
    assert small_p.tension(LAM) == pytest.approx(big_p.tension(LAM / s**2), rel=1e-8)
