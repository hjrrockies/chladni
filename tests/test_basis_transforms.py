"""Particular-solution transforms are COLUMN-EXACT.  (2026-11-03)

`docs/conventions.md`, "Particular-solution transforms are column-exact". Boundary integrals
evaluate the eigenfunction near a corner `c` as `basis.translated(-c).at(z - c)` with the SAME
coefficient vector (`notes/hadamard_accuracy.md`, addendum 2), which is only right if
translation moves every column's function, not merely the span. `PlaneWave.translated` used to
return `self` -- a translation is a phase, so the span is unchanged -- and would have silently
broken that.
"""
import numpy as np
import pytest

from chladni.geometry import L_shape, Polygon
from chladni.helmholtz import PlaneWave
from chladni.plan import LADDER, plan_for

K = 7.3
SHIFT = 3 + 2j
#: Each column is compared against its own evaluation accuracy (`Sampling.accuracy`): the
#: high-order Fourier-Bessel columns of a planned basis are only good to 1e-10..1e-12 at these
#: points, and the two frames round differently. Measured worst ratio 32 (hexagon, rung 4),
#: with a 1e-13 floor for columns whose reported accuracy is eps.
ACC_FACTOR, FLOOR = 100.0, 1e-13


def _hexagon():
    return Polygon([0, 1.1 + 0.1j, 1.6 + 0.9j, 1.2 + 1.7j, 0.3 + 1.5j, -0.4 + 0.7j])


def _points(dom, n=40):
    """Boundary points away from the corners plus interior points: near a corner, `z - c`
    formed in floating point is itself the rounding this contract exists to avoid."""
    rng = np.random.default_rng(0)
    v = np.asarray(dom.vertices)
    a, b = v, np.roll(v, -1)
    t = rng.uniform(0.2, 0.8, n)
    i = rng.integers(0, len(v), n)
    bdry = a[i] + t * (b[i] - a[i])
    c = v.mean()
    inner = c + 0.5 * (bdry - c)
    return np.concatenate([bdry, inner])


def _check(fam, pts, dz):
    v0, g0 = fam.values(K, pts), fam.grad(K, pts)
    moved = fam.translated(dz)
    v1, g1 = moved.values(K, pts + dz), moved.grad(K, pts + dz)
    bar = FLOOR + ACC_FACTOR * fam.at(pts).accuracy(K)
    sv = np.max(np.abs(v0), axis=0) + 1e-300
    sg = np.max(np.abs(g0), axis=0) + 1e-300
    assert np.all(np.max(np.abs(v1 - v0), axis=0) / sv < bar)
    assert np.all(np.max(np.abs(g1 - g0), axis=0) / sg < bar)


def test_plane_waves_translate_column_by_column():
    fam = PlaneWave.uniform(9, radius=2.0)
    pts = np.array([0.3 + 0.1j, -1.2 + 0.7j, 0.9 - 1.4j, 1.5 + 1.5j])
    _check(fam, pts, SHIFT)
    _check(fam, pts, -SHIFT)


def test_plane_waves_rotate_and_scale_column_by_column():
    fam = PlaneWave.uniform(7, radius=2.0, origin=0.4 - 0.2j)
    pts = np.array([0.3 + 0.1j, -1.2 + 0.7j, 0.9 - 1.4j])
    about, theta = 0.5 + 0.5j, 0.7
    rot = fam.rotated(theta, about)
    np.testing.assert_allclose(rot.values(K, about + (pts - about) * np.exp(1j * theta)),
                               fam.values(K, pts), atol=1e-13)
    s = 1.7 * np.exp(0.4j)              # `scaled` moves k -> k/|s|, see its docstring
    sc = fam.scaled(s, about)
    np.testing.assert_allclose(sc.values(K / abs(s), about + (pts - about) * s),
                               fam.values(K, pts), atol=1e-13)


@pytest.mark.parametrize("make", [L_shape, _hexagon], ids=["L", "hexagon"])
@pytest.mark.parametrize("rung", range(len(LADDER)))
def test_planned_bases_translate_column_by_column(make, rung):
    dom = make()
    basis, _, _ = plan_for(dom, 8, 1e-10, rung=rung).realize(dom)
    pts = _points(dom)
    _check(basis, pts, SHIFT)
    _check(basis, pts, -dom.vertices[1])
