"""Boundary: contiguity, closure, orientation, signed area, and node allocation."""
import numpy as np
import pytest

from chladni.geometry import ArcSegment, Boundary, LineSegment

from conftest import EXACT, PREC, SQUARE, ngon_vertices

# --- construction and invariants --------------------------------------------------------


def test_from_vertices_closes_the_loop():
    b = Boundary.from_vertices(SQUARE)
    assert len(b.segments) == 4
    assert b.is_closed and b.is_contiguous and b.is_simple
    assert b.segments[-1].p1 == pytest.approx(b.segments[0].p0, abs=EXACT)


def test_from_vertices_open_chain():
    b = Boundary.from_vertices(SQUARE, close=False)
    assert len(b.segments) == 3
    assert not b.is_closed


def test_non_contiguous_segments_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        Boundary([LineSegment(0j, 1 + 0j), LineSegment(5 + 5j, 6 + 5j)])


def test_length_is_the_sum_of_segment_lengths():
    b = Boundary.from_vertices(SQUARE)
    assert b.length == pytest.approx(4.0, abs=EXACT)
    assert np.allclose(b.seg_lengths, 1.0, atol=EXACT)


def test_prec_is_the_tightest_of_its_parts():
    b = Boundary(
        [
            LineSegment(0j, 1 + 0j, prec=1e-6),
            LineSegment(1 + 0j, 1 + 1j, prec=1e-12),
            LineSegment(1 + 1j, 1j, prec=1e-8),
            LineSegment(1j, 0j),
        ]
    )
    assert b.prec == 1e-12


# --- orientation and signed area --------------------------------------------------------


def test_signed_area_of_a_ccw_square():
    assert Boundary.from_vertices(SQUARE).signed_area == pytest.approx(1.0, abs=EXACT)


def test_signed_area_flips_under_reversal():
    b = Boundary.from_vertices(SQUARE)
    assert b.reversed().signed_area == pytest.approx(-1.0, abs=EXACT)
    assert b.orientation == 1
    assert b.reversed().orientation == -1


def test_reversed_preserves_geometry_and_bcs():
    b = Boundary(
        [
            LineSegment(0j, 1 + 0j, bc="neu"),
            LineSegment(1 + 0j, 1 + 1j, bc="dir"),
            LineSegment(1 + 1j, 1j, bc=0.4),
            LineSegment(1j, 0j, bc="dir"),
        ]
    )
    r = b.reversed()
    assert r.length == pytest.approx(b.length, abs=EXACT)
    assert sorted(s.bc for s in r.segments) == sorted(s.bc for s in b.segments)
    # the loop keeps its base point and runs the other way round, so the first reversed
    # segment starts where the original loop started and retraces the last original segment
    assert r.segments[0].p0 == pytest.approx(b.segments[0].p0, abs=EXACT)
    assert r.segments[0].p1 == pytest.approx(b.segments[-1].p0, abs=EXACT)
    assert r.reversed().segments[0].p0 == pytest.approx(b.segments[0].p0, abs=EXACT)


def test_signed_area_of_a_full_circle():
    b = Boundary([ArcSegment(0j, 2.0, 0.0, 2 * np.pi)])
    assert b.signed_area == pytest.approx(4 * np.pi, rel=PREC)


@pytest.mark.parametrize("n", [3, 5, 6, 12, 40])
def test_signed_area_of_a_regular_ngon(n):
    """Closed form: (n/2) sin(2 pi / n) at circumradius 1."""
    b = Boundary.from_vertices(ngon_vertices(n))
    assert b.signed_area == pytest.approx(n / 2 * np.sin(2 * np.pi / n), abs=EXACT)


# --- polyline ---------------------------------------------------------------------------


def test_polyline_of_a_polygon_is_one_piece_per_segment():
    b = Boundary.from_vertices(SQUARE)
    b0, b1, owner = b.polyline()
    assert len(b0) == len(b1) == len(owner) == 4
    assert np.array_equal(owner, np.arange(4))
    assert np.allclose(b1[:-1], b0[1:])


def test_polyline_of_a_curve_refines_adaptively():
    coarse = Boundary([ArcSegment(0j, 1.0, 0.0, 2 * np.pi)], validate=False)
    b0, b1, owner = coarse.polyline()
    assert len(b0) > 8
    chord_total = np.abs(b1 - b0).sum()
    assert chord_total == pytest.approx(coarse.length, rel=1e-3)
    assert np.all(owner == 0)


# --- node allocation --------------------------------------------------------------------


def test_total_n_is_split_by_arclength(rect_4x1):
    """`n` is a TOTAL. lappy only offers per-segment counts, which is what forced a
    basis-aware allocator into the geometry layer."""
    q = rect_4x1.outer.quad(1000)
    counts = np.bincount(q.seg_idx, minlength=4)
    assert counts.sum() == pytest.approx(1000, abs=8)
    long_edges = counts[[0, 2]].sum()
    short_edges = counts[[1, 3]].sum()
    assert long_edges / short_edges == pytest.approx(4.0, rel=0.05)


def test_n_per_seg_overrides_the_split(rect_4x1):
    q = rect_4x1.outer.quad(n_per_seg=[7, 3, 5, 11])
    assert np.array_equal(np.bincount(q.seg_idx, minlength=4), [7, 3, 5, 11])


def test_every_segment_gets_at_least_one_node(rect_4x1):
    q = rect_4x1.outer.quad(8)
    assert np.all(np.bincount(q.seg_idx, minlength=4) > 0)


def test_sample_returns_points_on_the_boundary():
    b = Boundary.from_vertices(SQUARE)
    pts = b.sample(64)
    assert pts.shape == (64,)
    assert np.iscomplexobj(pts)
    assert np.allclose([b.dist(p) for p in pts], 0.0, atol=PREC)
