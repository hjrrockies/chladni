"""Weyl asymptotics: the forward and inverted laws agree, and the corner term is right.

The two sign flips are the thing most likely to be wrong and the hardest to see. In the
forward count the perimeter term SUBTRACTS for Dirichlet -- clamping the boundary removes
modes -- and adds for Neumann. On inversion it crosses to the other side of the quadratic and
flips again, so the Dirichlet inverse carries `+P`. A round trip catches a single flip; only
comparing the two boundary conditions catches a doubled one.

The corner correction has an exact value on the unit square, `4 * (pi^2 - (pi/2)^2) /
(24*pi*(pi/2)) = 1/4`, which pins it without reference to any spectrum.
"""
import numpy as np
import pytest

from chladni import asymp
from chladni import reference as ref
from chladni.geometry import GWW1, L_shape, cut_square, disk, eq_tri, rect

SHAPES = [("rect 2x1", rect(2.0, 1.0)), ("square", rect(1.0, 1.0)),
          ("L_shape", L_shape()), ("eq_tri", eq_tri(1.0)), ("GWW1", GWW1())]


# --- the two-term law, forward and inverted ----------------------------------------------


def test_the_dirichlet_perimeter_term_subtracts():
    d = rect(2.0, 1.0)
    lam = 200.0
    one_term = d.area * lam / (4 * np.pi)
    assert asymp.weyl_count(lam, d) < one_term


def test_neumann_counts_more_than_dirichlet_at_the_same_lam():
    """The sign of the perimeter term IS the difference between the two."""
    a, p = 2.0, 6.0
    lam = 200.0
    dir_ = asymp.weyl_count(lam, area=a, perim=p, bc_type="dir")
    neu = asymp.weyl_count(lam, area=a, perim=p, bc_type="neu")
    assert neu > dir_
    assert neu - dir_ == pytest.approx(2 * p * np.sqrt(lam) / (4 * np.pi), rel=1e-12)


@pytest.mark.parametrize("bc", ["dir", "neu"])
@pytest.mark.parametrize("name,dom", SHAPES, ids=[n for n, _ in SHAPES])
def test_count_and_est_round_trip(name, dom, bc):
    """`weyl_est` inverts `weyl_count` exactly. A single sign error breaks this; a doubled
    one survives it for Dirichlet alone, which is why both BCs are swept."""
    for k in (1.0, 5.0, 20.0, 100.0):
        lam = asymp.weyl_est(k, area=dom.area, perim=dom.perimeter, bc_type=bc)
        back = asymp.weyl_count(lam, area=dom.area, perim=dom.perimeter, bc_type=bc)
        assert back == pytest.approx(k, rel=1e-10)


def test_weyl_est_tracks_the_true_spectrum():
    """Asymptotic, so not exact -- but it must be the right size. Within 20% at k = 20."""
    d = rect(2.0, 1.0)
    assert asymp.weyl_est(20, d) == pytest.approx(ref.rect_eigs(20, 2.0, 1.0)[-1], rel=0.2)


def test_weyl_est_accepts_a_non_integer_k():
    d = rect(2.0, 1.0)
    a, b = asymp.weyl_est(5, d), asymp.weyl_est(6, d)
    assert a < asymp.weyl_est(5.5, d) < b


# --- the corner correction ----------------------------------------------------------------


def test_the_unit_square_correction_is_exactly_a_quarter():
    """Four right angles: `4 * (pi^2 - (pi/2)^2) / (24*pi*(pi/2)) = 1/4`, with no spectrum
    involved."""
    assert asymp.corner_correction([np.pi / 2] * 4) == pytest.approx(0.25, rel=1e-14)


def test_the_correction_does_not_depend_on_lam():
    d = rect(1.0, 1.0)
    diffs = [asymp.weyl_count_poly(lam, d) - asymp.weyl_count(lam, d)
             for lam in (10.0, 1e3, 1e6)]
    assert np.allclose(diffs, diffs[0], rtol=0, atol=1e-12)
    assert diffs[0] == pytest.approx(0.25, rel=1e-12)


def test_a_reentrant_corner_contributes_negatively():
    """`omega > pi` flips the sign of `pi**2 - omega**2`, so the L-shape's 270-degree corner
    subtracts where its five right angles add."""
    assert asymp.corner_correction([3 * np.pi / 2]) < 0
    assert asymp.corner_correction([c.omega for c in L_shape().corners]) == pytest.approx(
        5 / 18, rel=1e-12)


def test_corner_angles_come_from_corners_not_from_an_isinstance_check():
    """lappy gates the three-term count on `isinstance(domain, Polygon)` and refuses anything
    else. `cut_square` is not a polygon and has four corners; here it works."""
    d = cut_square(0.5)
    assert len(d.corners) > 0
    got = asymp.weyl_count_poly(100.0, d) - asymp.weyl_count(100.0, d)
    assert got == pytest.approx(asymp.corner_correction([c.omega for c in d.corners]))


def test_a_smooth_domain_has_no_corner_term():
    """The disk has no corners, so the three-term law degenerates to the two-term one. There
    is no curvature term here; that makes it a weaker detector, not a wrong one."""
    d = disk(1.0)
    assert asymp.weyl_count_poly(100.0, d) == pytest.approx(asymp.weyl_count(100.0, d))


def test_est_poly_inverts_count_poly():
    d = L_shape()
    for k in (3.0, 10.0, 50.0):
        lam = asymp.weyl_est_poly(k, d)
        assert asymp.weyl_count_poly(lam, d) == pytest.approx(k, rel=1e-8)


def test_est_poly_returns_an_array_for_an_array():
    d = rect(2.0, 1.0)
    out = asymp.weyl_est_poly([3.0, 10.0], d)
    assert out.shape == (2,)
    assert out[0] < out[1]


# --- the diagnostics ----------------------------------------------------------------------


def test_the_deficit_reads_clean_on_an_exact_spectrum():
    """Calibration, from lappy: a correct consecutive set reaches at worst about +0.3."""
    d = rect(1.0, 1.0)
    assert asymp.weyl_deficit(ref.rect_eigs(10, 1.0, 1.0), d) < 0.5


def test_the_deficit_sees_a_missing_mode():
    """And near or above +1 when one is gone. That gap is the whole value of the diagnostic."""
    d = rect(1.0, 1.0)
    lam = ref.rect_eigs(10, 1.0, 1.0)
    assert asymp.weyl_deficit(np.delete(lam, 4), d) > 1.0


def test_the_deficit_sees_a_deleted_degenerate_copy_below_the_top():
    """Degeneracy alone does not hide a missing mode: the cuts above the deleted level all
    move. lappy's calibration note is about the TOP level specifically -- see below."""
    d = rect(1.0, 1.0)
    lam = ref.rect_eigs(10, 1.0, 1.0)
    dup = [i for i in range(1, len(lam)) if abs(lam[i] - lam[i - 1]) < 1e-9]
    assert dup, "this spectrum no longer has a degenerate pair to delete from"
    assert asymp.weyl_deficit(np.delete(lam, dup[0]), d) > 1.0


def test_the_deficit_is_blind_at_the_TOP_of_the_range():
    """Its real blind spot, asserted so nobody relies on what it cannot do.

    Every cut is placed BETWEEN two distinct levels, so every cut lies below the highest one.
    A mode missing at or above the top level moves no cut and is invisible. That is why a
    caller has to ask for more eigenvalues than its objective consumes -- the audit only
    covers what it can cut beneath.
    """
    d = rect(1.0, 1.0)
    lam = ref.rect_eigs(10, 1.0, 1.0)
    assert abs(lam[-1] - lam[-2]) < 1e-9, "the top level is meant to be degenerate here"
    assert asymp.weyl_deficit(lam[:-1], d) < 0.5


def test_the_deficit_is_nan_when_there_is_nowhere_to_cut():
    d = rect(1.0, 1.0)
    assert np.isnan(asymp.weyl_deficit([12.34], d))
    assert np.isnan(asymp.weyl_deficit([12.34, 12.34], d))


def test_count_check_oscillates_around_zero():
    d = rect(2.0, 1.0)
    dev = asymp.weyl_count_check(ref.rect_eigs(25, 2.0, 1.0), d)
    assert dev.size == 25
    assert dev.min() < 0 < dev.max()


# --- what is not supported ----------------------------------------------------------------


def test_a_mixed_or_robin_boundary_raises_rather_than_guessing():
    """There is no single perimeter coefficient for a boundary that is not uniformly one or
    the other, so the honest answer is to refuse."""
    for bc in ("rob", "mixed"):
        with pytest.raises(NotImplementedError, match="bc_type"):
            asymp.weyl_count(10.0, area=1.0, perim=4.0, bc_type=bc)
    with pytest.raises(NotImplementedError):
        asymp.weyl_est(3, rect(1.0, 1.0, bc=0.3))


def test_missing_arguments_say_what_is_missing():
    with pytest.raises(ValueError, match="area"):
        asymp.weyl_count(10.0)
    with pytest.raises(ValueError, match="perim"):
        asymp.weyl_count(10.0, area=1.0)
    with pytest.raises(ValueError, match="angles"):
        asymp.weyl_count_poly(10.0, area=1.0, perim=4.0)
