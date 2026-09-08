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

def test_n_useful_is_the_jacobi_anger_cutoff():
    """The constant is derived, not fitted: `(ln(1/eps_mach))**(2/3) = 10.912` is the "about
    11" the original measurement reported."""
    eps = np.finfo(float).eps
    assert np.log(1 / eps) ** (2 / 3) == pytest.approx(10.912, abs=5e-4)
    for kR in (2.0, 5.0, 10.0, 20.0, 40.0):
        assert PlaneWave.n_useful(kR, eps) == pytest.approx(
            kR + 10.912 * kR ** (1 / 3), rel=1e-3)


def test_n_useful_loosens_as_prec_loosens():
    """The one place in this layer where a looser precision genuinely buys a smaller basis:
    fewer directions are independent when you are willing to accept less."""
    counts = [PlaneWave.n_useful(20.0, p) for p in (1e-13, 1e-8, 1e-4)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] / counts[2] > 1.4


@pytest.mark.slow
def test_n_useful_tracks_the_measured_numerical_rank_at_every_prec():
    """The bound is only worth having if it predicts the rank it claims to. Measured on the
    unit disk against relative singular-value cuts from 1e-13 to 1e-4."""
    z = np.exp(1j * np.linspace(0, 2 * np.pi, 3000, endpoint=False))
    for kR in (5.0, 10.0, 20.0, 40.0):
        A = PlaneWave.uniform(140).values(kR, z)
        A = A / np.linalg.norm(A, axis=0)
        sv = np.linalg.svd(A, compute_uv=False)
        for tol in (1e-13, 1e-8, 1e-4):
            rank = int((sv > tol * sv[0]).sum())
            predicted = 2 * PlaneWave.n_useful(kR, tol) + 1
            assert abs(rank - predicted) / rank < 0.06, f"k*R={kR} tol={tol:g}"


def test_n_useful_at_reports_the_bound_against_this_familys_own_radius():
    """The same 40 directions at the same k*R: ample at 1e-13, wasteful at 1e-4.

    Note the direction of that, which is easy to get backwards. A TIGHTER prec resolves MORE
    independent columns, because it is willing to keep smaller singular values -- so the
    bound rises as prec falls, and a fixed direction count becomes wasteful as prec LOOSENS.
    """
    assert PlaneWave.n_useful(20.0, 1e-13) > 40 > PlaneWave.n_useful(20.0, 1e-4)

    tight = PlaneWave.uniform(40, radius=1.0, prec=1e-13)
    loose = PlaneWave.uniform(40, radius=1.0, prec=1e-4)

    # Reported in COLUMNS, because the consumer is a RANK. `n_useful` counts directions and
    # a direction is a cosine and a sine.
    assert tight.n_useful_at(20.0) == pytest.approx(2 * PlaneWave.n_useful(20.0, 1e-13))
    assert tight.n_cols == loose.n_cols == 80
    assert tight.n_useful_at(20.0) > 80 > loose.n_useful_at(20.0)


def test_a_family_with_no_radius_reports_no_bound_rather_than_guessing():
    """`n_useful` needs the enclosing radius, and a bare direction set does not have one.
    A guess would be worse than an honest `None`."""
    assert PlaneWave.uniform(200).n_useful_at(1.0) is None


def test_an_over_enriched_family_does_NOT_warn_because_it_cannot_know():
    """Whether extra columns are harmful is a PENCIL question, and the basis cannot answer it.

    `n_useful` is a sizing helper, not a verdict. A basis deliberately enriched past the
    cutoff and then cut back by a THRESHOLD rank is not a mistake -- it is the configuration
    `rect_suite.pencil_for` measures as ten orders better than the alternative, 2.1e-12
    against 3.5e-02 at the worst true eigenvalue. A warning here fires on the recommended
    setup, at every evaluation of every scan, and it did: 316 of them in one suite run, which
    is why eighteen test modules had `filterwarnings` marks blanket-suppressing the whole
    `PrecisionWarning` class and with it the pencil-level signals that ARE load bearing.

    The judgment lives at `MPSPencil._warn_if_unsafe`, which compares the RETAINED rank's
    singular value against the noise floor and so knows the thing the basis does not.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        PlaneWave.uniform(60, radius=1.0).log_scale(5.0)      # far past the bound
        PlaneWave.uniform(20, radius=1.0).log_scale(40.0)     # comfortably within it
        PlaneWave.uniform(200).log_scale(1.0)                 # no radius at all


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
