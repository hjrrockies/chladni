"""Immutability, transforms, operator sugar, and the prec knob.

Immutability is what lets `prec` be a plain attribute instead of a cache key: an object's
cached approximations can never be stale, so cached_property is correct under threads.
These tests exist to keep that guarantee real.
"""
import numpy as np
import pytest

from chladni.geometry import ArcSegment, Boundary, LineSegment, Polygon, as_xy, from_xy

from conftest import EXACT, L_SHAPE, PREC, SQUARE

# --- immutability -----------------------------------------------------------------------


def test_objects_reject_attribute_assignment(square):
    for obj, attr in ((square, "outer"), (square.outer, "segments"), (square.outer.segments[0], "bc")):
        with pytest.raises((AttributeError, TypeError)):
            setattr(obj, attr, None)


def test_transforms_do_not_mutate_the_original(l_shape):
    before = l_shape.area
    moved = l_shape.translated(5 + 5j).scaled(3.0).rotated(0.7)
    assert l_shape.area == pytest.approx(before, abs=EXACT)
    assert l_shape.centroid == pytest.approx(Polygon(L_SHAPE).centroid, abs=PREC)
    assert moved is not l_shape


def test_with_prec_returns_a_new_object(square):
    tighter = square.with_prec(1e-13)
    assert tighter.prec == 1e-13
    assert square.prec == pytest.approx(1e-8)
    assert tighter is not square


def test_with_bc_returns_a_new_object(square):
    neu = square.with_bc("neu")
    assert neu.bc_type == "neu"
    assert square.bc_type == "dir"


# --- transform correctness --------------------------------------------------------------


def test_translation(square):
    moved = square.translated(2 - 3j)
    assert moved.area == pytest.approx(square.area, abs=EXACT)
    assert moved.centroid == pytest.approx(square.centroid + 2 - 3j, abs=PREC)
    assert moved.perimeter == pytest.approx(square.perimeter, abs=EXACT)


def test_scaling_scales_area_quadratically(l_shape):
    s = 2.5
    big = l_shape.scaled(s)
    assert big.area == pytest.approx(l_shape.area * s**2, rel=PREC)
    assert big.perimeter == pytest.approx(l_shape.perimeter * s, rel=PREC)
    assert big.diameter == pytest.approx(l_shape.diameter * s, rel=PREC)


def test_scaling_about_a_point_fixes_that_point(l_shape):
    fixed = 1 + 1j
    out = l_shape.scaled(3.0, about=fixed)
    assert np.min(np.abs(out.outer.sample(400) - fixed)) == pytest.approx(
        3.0 * np.min(np.abs(l_shape.outer.sample(400) - fixed)), rel=1e-3
    )


def test_rotation_preserves_metrics_and_rotates_angles(l_shape):
    theta = 0.37
    rot = l_shape.rotated(theta)
    assert rot.area == pytest.approx(l_shape.area, rel=PREC)
    assert rot.perimeter == pytest.approx(l_shape.perimeter, rel=PREC)
    assert np.allclose(
        sorted(c.omega for c in rot.corners), sorted(c.omega for c in l_shape.corners), atol=PREC
    )
    assert rot.centroid == pytest.approx(l_shape.centroid * np.exp(1j * theta), abs=PREC)


def test_rotation_composes_back_to_identity(square):
    out = square
    for _ in range(6):
        out = out.rotated(np.pi / 3)
    # compare against the original CURVE, not against sorted samples: sort_complex orders
    # by real part first, so a 1e-18 residue on an axis-aligned edge reshuffles the array
    assert np.max(square.outer.dist(out.outer.sample(64))) < PREC
    assert out.area == pytest.approx(square.area, rel=PREC)
    assert out.centroid == pytest.approx(square.centroid, abs=PREC)


def test_reversed_is_an_involution():
    b = Boundary.from_vertices(L_SHAPE)
    assert b.reversed().reversed().signed_area == pytest.approx(b.signed_area, abs=EXACT)


def test_segment_reversal_swaps_endpoints_and_flips_the_frame():
    seg = LineSegment(0j, 3 + 4j)
    rev = seg.reversed()
    assert rev.p0 == pytest.approx(seg.p1, abs=EXACT)
    assert rev.p1 == pytest.approx(seg.p0, abs=EXACT)
    assert rev.T(0.5) == pytest.approx(-seg.T(0.5), abs=EXACT)
    assert rev.length == pytest.approx(seg.length, abs=EXACT)


# --- operator sugar ---------------------------------------------------------------------


def test_add_translates(square):
    assert (square + (1 + 1j)).centroid == pytest.approx(square.centroid + 1 + 1j, abs=PREC)


def test_mul_by_a_real_scales(square):
    assert (square * 2).area == pytest.approx(4 * square.area, rel=PREC)


def test_mul_by_a_complex_rotates_and_scales(square):
    z = 2 * np.exp(1j * 0.4)
    out = square * z
    assert out.area == pytest.approx(4 * square.area, rel=PREC)
    assert out.centroid == pytest.approx(square.centroid * z, abs=PREC)


def test_neg_reverses_a_segment():
    seg = LineSegment(0j, 1 + 1j)
    assert (-seg).p0 == pytest.approx(seg.p1, abs=EXACT)


def test_rmul_is_supported(square):
    assert (3 * square).area == pytest.approx(9 * square.area, rel=PREC)


# --- prec propagation -------------------------------------------------------------------


def test_prec_defaults_to_1e_8():
    assert LineSegment(0j, 1 + 0j).prec == pytest.approx(1e-8)


def test_prec_propagates_as_the_tightest_of_the_parts():
    outer = Boundary.from_vertices(SQUARE * 3)
    hole = Boundary.from_vertices(SQUARE + 1 + 1j).reversed().with_prec(1e-14)
    from chladni.geometry import Domain

    assert Domain(outer, holes=(hole,)).prec == 1e-14


def test_transforms_carry_prec_and_bc_through():
    seg = LineSegment(0j, 1 + 0j, bc="neu", prec=1e-11)
    for out in (seg.translated(1j), seg.scaled(2.0), seg.rotated(0.3), seg.reversed()):
        assert out.bc == 1.0
        assert out.prec == 1e-11


@pytest.mark.parametrize("bad", [0.0, -1e-8, 1.0, 2.0])
def test_invalid_prec_rejected(bad):
    with pytest.raises(ValueError):
        LineSegment(0j, 1 + 0j, prec=bad)


# --- the complex/xy seam ----------------------------------------------------------------


def test_as_xy_and_from_xy_roundtrip():
    z = np.array([0, 1 + 2j, -3 - 4j], dtype=complex)
    xy = as_xy(z)
    assert xy.shape == (3, 2)
    assert np.allclose(xy[:, 0], z.real)
    assert np.allclose(xy[:, 1], z.imag)
    assert np.allclose(from_xy(xy), z)
