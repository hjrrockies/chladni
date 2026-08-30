"""Plane waves: the standard MPS family for a smooth or convex domain.

lappy has none. `make_default_basis` routes a corner-free domain to boundary fundamental
solutions instead, and the phrase "plane wave" appears in that codebase only as the
worst-case integrand used to size a cubature rule. This is a real gap, and it is cheap to
fill: `cos(k*dot(z, d))` and `sin(k*dot(z, d))` are exact for any unit direction, need no
Bessel function, and need no scaling at all.

The one thing that has to be said out loud is the conditioning. A plane-wave basis is
exponentially ill-conditioned past a threshold that moves out linearly with k*R, and past it
the extra columns are not merely wasted -- they arrive at machine noise, lower the smallest
singular value and corrupt the tension the solver is minimizing.
"""
import numpy as np
import pytest

from conftest import EXACT
from chladni.helmholtz import PlaneWave, PrecisionWarning

K = 5.5


def test_a_direction_gives_a_cosine_and_a_sine():
    d = np.exp(0.7j)
    fam = PlaneWave([d])
    assert len(fam) == 2
    z = np.array([0.3 + 0.2j, -0.5 + 0.9j])
    s = z.real * d.real + z.imag * d.imag
    assert np.allclose(fam.values(K, z), np.stack([np.cos(K * s), np.sin(K * s)], axis=1),
                       rtol=0, atol=EXACT)


def test_every_column_solves_the_helmholtz_equation_analytically():
    """No differencing needed: the Laplacian of `cos(k*dot(z,d))` is `-k**2` times itself for
    any unit d, so this holds at machine level rather than at a finite-difference floor."""
    fam = PlaneWave.uniform(6)
    z = np.array([0.3 + 0.2j, -0.5 + 0.9j, 1.2 - 0.4j])
    h = 1e-3
    u = lambda w: fam.values(K, w)
    lap = np.zeros_like(u(z))
    for step in (h, 1j * h):
        lap = lap + (-u(z + 2 * step) + 16 * u(z + step) - 30 * u(z)
                     + 16 * u(z - step) - u(z - 2 * step)) / (12 * h**2)
    assert np.abs(lap + K**2 * u(z)).max() / K**2 < 5e-9


def test_the_gradient_is_analytic():
    fam = PlaneWave.uniform(5)
    z = np.array([0.3 + 0.2j, -0.5 + 0.9j])
    h = 1e-4
    g = fam.grad(K, z)
    for axis, step in ((0, h), (1, 1j * h)):
        fd = (-fam.values(K, z + 2 * step) + 8 * fam.values(K, z + step)
              - 8 * fam.values(K, z - step) + fam.values(K, z - 2 * step)) / (12 * h)
        got = g.real if axis == 0 else g.imag
        assert np.max(np.abs(got - fd)) < 1e-10 * np.abs(g).max()


def test_the_scale_is_identically_zero():
    """These columns are already O(1) by construction, which is the cleanest illustration
    that `sigma` is a per-family property and not a global postprocessing step."""
    fam = PlaneWave.uniform(7)
    assert np.array_equal(fam.log_scale(K), np.zeros(len(fam)))
    assert np.abs(fam.values(K, np.linspace(-2, 2, 50) + 0.3j)).max() <= 1.0


# --- the direction set --------------------------------------------------------------------

def test_directions_sweep_a_half_turn_not_a_full_one():
    fam = PlaneWave.uniform(8)
    assert len(fam.directions) == 8
    assert np.all(np.angle(fam.directions) < np.pi)


def test_opposite_directions_would_duplicate_every_column_exactly():
    """Why the sweep is [0, pi). `cos(k*dot(z,-d)) = cos(k*dot(z,d))` and the sine flips
    sign, so a [0, 2*pi) sweep buys nothing and doubles the column count.

    Checked as an exact identity rather than through a rank, which at these wavenumbers is
    limited by the Jacobi-Anger cutoff rather than by the duplication and so would say
    nothing about it."""
    d = np.exp(0.7j)
    z = np.linspace(-1, 1, 40) + 0.2j
    a = PlaneWave([d]).values(K, z)
    b = PlaneWave([-d]).values(K, z)
    assert np.array_equal(a[:, 0], b[:, 0])
    assert np.array_equal(a[:, 1], -b[:, 1])


def test_directions_are_normalized():
    fam = PlaneWave([3 + 4j, -2 + 0j])
    assert np.allclose(np.abs(fam.directions), 1.0, atol=EXACT)


# --- conditioning -----------------------------------------------------------------------------

@pytest.mark.slow
def test_the_useful_direction_count_matches_the_measured_numerical_rank():
    """The numerical rank saturates at the Jacobi-Anger cutoff. Measured on the unit disk
    with pre-normalized columns and a 1e-13 relative cut: ranks 33/45/65/95/149 at
    k*R = 2/5/10/20/40, which the bound `k*R + 11*(k*R)**(1/3)` tracks."""
    z = np.exp(1j * np.linspace(0, 2 * np.pi, 2000, endpoint=False))
    for kR, want in ((5.0, 45), (10.0, 65), (20.0, 95)):
        fam = PlaneWave.uniform(120)
        A = fam.values(kR, z)
        A = A / np.linalg.norm(A, axis=0)
        rank = int((np.linalg.svd(A, compute_uv=False) > 1e-13 * np.linalg.norm(A, 2)).sum())
        assert abs(rank - want) <= 6, f"k*R={kR}: rank {rank}, expected about {want}"
        assert rank <= 2 * PlaneWave.n_useful(kR) + 8


def test_n_useful_is_the_jacobi_anger_cutoff():
    for kR in (2.0, 5.0, 10.0, 20.0, 40.0):
        assert PlaneWave.n_useful(kR) == pytest.approx(kR + 11 * kR ** (1 / 3), rel=1e-12)


def test_asking_for_more_directions_than_are_useful_warns():
    """Past the cutoff the columns arrive at machine noise: they do not enrich the space,
    they lower the smallest singular value and corrupt the tension. The bound costs a cube
    root, so there is no reason not to check it."""
    fam = PlaneWave.uniform(60, radius=1.0)
    with pytest.warns(PrecisionWarning, match="directions"):
        fam.log_scale(5.0)


def test_a_family_within_the_bound_is_silent():
    import warnings

    fam = PlaneWave.uniform(20, radius=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        fam.log_scale(40.0)


def test_a_family_with_no_radius_cannot_and_does_not_warn():
    """`n_useful` needs the enclosing radius, and a bare direction set does not have one.
    Warning on a guess would be worse than not warning."""
    import warnings

    fam = PlaneWave.uniform(200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        fam.log_scale(1.0)


# --- shapes -----------------------------------------------------------------------------------

def test_the_matrix_is_points_by_columns_and_cols_is_honoured():
    fam = PlaneWave.uniform(5)
    z = np.linspace(-1, 1, 9) + 0.4j
    assert fam.values(K, z).shape == (9, 10)
    want = np.array([7, 0, 3])
    assert np.array_equal(fam.values(K, z, cols=want), fam.values(K, z)[:, want])


def test_plane_waves_are_regular_everywhere():
    s = PlaneWave.uniform(3).singular_exponents()
    assert np.all(np.isnan(s.origin))
    assert np.all(s.exponent == 0.0)
