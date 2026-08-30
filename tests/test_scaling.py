"""The column scale: what it buys, and the two properties that let it compose.

lappy's benchmark notebook concluded that Fourier-Bessel columns at a sharp corner "carry
almost no independent information about the domain". Half of that is true -- near-dependence
at a sharp corner is intrinsic to float64 -- and half is an artifact: in lappy those columns
are literally arrays of zeros, because `scipy.special.jv` underflowed. A column known to full
RELATIVE accuracy is a different object, and this file measures the difference.

The two properties everything downstream leans on:

* `sigma` depends on (family, column, k) and NEVER on the evaluation points. So a solver that
  additionally rescales columns to unit norm composes with this by ADDING log scales, and a
  coefficient vector means the same function on every point set.
* `sigma` is smooth in k. It sits inside the objective MPS parabola-fits, so a kink there is
  a kink in the thing being minimized.
"""
import numpy as np
import pytest

from conftest import EXACT, mp_log_besselj
from chladni.helmholtz import FourierBessel
from chladni.helmholtz.corners import CornerModes
from chladni.helmholtz.fourier_bessel import _log_envelope


def fam(theta, n=40, radius=1.0):
    m = CornerModes.from_wedge(theta, 0.0, 0.0)
    return FourierBessel(0j, 1 + 0j, m, m.degrees(n), radius)


#: A wedge fraction that is not a zero of any column's angular factor. `phi = theta/2` is
#: the obvious choice and is wrong: `sin(mu*theta/2) = sin(l*pi/2)`, which vanishes for every
#: even degree, so half the columns evaluate to roundoff there and any relative comparison
#: against them is meaningless. 0.37 is not a rational multiple of a half.
PHI_FRAC = 0.37


def column_maxima(f, k, n_r=2000):
    """max |column| over the wedge.

    Computed as `max_r |radial|` times `max_phi |angular|`, and the second factor is exactly
    1 for every column here (`|sin(mu*phi)|` reaches 1 somewhere in `(0, theta)` whenever
    `mu*theta >= pi/2`, which holds for every degree `l >= 1`). So a dense radial sweep is
    the whole measurement, and the full 2-D grid would be 80x the cost for the same number.
    """
    from chladni.helmholtz._bessel import log_j

    r = np.linspace(1e-8, f.radius, n_r)
    s = log_j(f.orders[None, :], k * r[:, None])
    return np.abs(s.S * np.exp(s.sigma - f.log_scale(k)[None, :])).max(axis=0)


# --- what the scaling buys ---------------------------------------------------------------

def test_the_raw_columns_span_hundreds_of_orders_of_magnitude():
    """The premise. Without a scale these columns are not representable together, and the
    sharpest ones are not representable at all."""
    f = fam(np.pi / 8, n=40)
    sig = f.log_scale(10.0)
    assert sig.max() - sig.min() > 500.0          # nats
    assert np.exp(sig.min()) == 0.0               # underflows outright in float64


def _scale_band(thetas, kRs, n=40):
    lo, hi = 9.0, 0.0
    for theta in thetas:
        f = fam(theta, n=n)
        for kR in kRs:
            # >= 80 radial samples per wavelength, or the sweep misses the maximum and
            # reports a spuriously small column -- which is how a 0.35 lower bound appeared
            # the first time this was measured.
            c = column_maxima(f, kR, n_r=max(2000, int(80 * kR)))
            lo, hi = min(lo, c.min()), max(hi, c.max())
    return lo, hi


def test_scaled_columns_land_near_one():
    """The inner-loop version: two wedge angles either side of the crossover."""
    lo, hi = _scale_band((np.pi / 8, 3 * np.pi / 2), (1.0, 40.0))
    assert 0.75 < lo and hi < 1.25, f"column maxima ran to [{lo:.3f}, {hi:.3f}]"


@pytest.mark.slow
def test_every_scaled_column_lands_near_one():
    """The full sweep: mu from 0.05 to 320, k*R from 0.5 to 300, four wedge angles.
    Measured band [0.802, 1.185], a spread of 1.48x. The band, not the exact numbers, is the
    contract -- widen it only with a measurement, and record the measurement."""
    lo, hi = _scale_band((np.pi / 8, np.pi / 2, 1.4 * np.pi, 3 * np.pi / 2),
                         (0.5, 1.0, 10.0, 40.0, 100.0, 300.0))
    assert 0.75 < lo and hi < 1.25, f"column maxima ran to [{lo:.3f}, {hi:.3f}]"


def test_a_column_scipy_cannot_represent_at_all_comes_back_accurate():
    """The sharpest corner column, checked against mpmath at 50 dps. In lappy this column
    is an array of exact zeros."""
    from scipy.special import jv

    f = fam(np.pi / 8, n=40)
    mu = f.orders[-1]
    r = np.array([0.3, 0.6, 0.9])
    k = 10.0
    assert np.all(jv(mu, k * r) == 0.0), "this column is supposed to be past scipy's cliff"
    phi = PHI_FRAC * f.modes.theta
    got = f.values(k, r * np.exp(1j * phi), cols=[len(f) - 1])[:, 0]
    ang = np.sin(mu * phi)
    sigma = f.log_scale(k, cols=[len(f) - 1])[0]
    for i, ri in enumerate(r):
        _, ref_log = mp_log_besselj(mu, k * ri)
        assert np.log(abs(got[i] / ang)) + sigma == pytest.approx(ref_log, abs=5e-13)


# --- point-set independence ------------------------------------------------------------------

def test_the_scale_does_not_depend_on_the_evaluation_points():
    """The design principle, stated as a test: `f(x, k)` is rescaled by the same constant
    whatever x's are in the evaluation set. lappy's `NormalizedBasis` cannot say this without
    a stored reference point set, and pays a full extra evaluation per k for it (measured
    2.04-2.27x on the evaluation half of every sigma)."""
    f = fam(1.4 * np.pi, n=8)
    a = f.at(np.linspace(0.1, 0.9, 5) * np.exp(0.6j))
    b = f.at(np.array([0.01 + 0.001j, 3.0 + 4.0j]))
    assert np.array_equal(a.log_scale(7.0), b.log_scale(7.0))
    assert np.array_equal(a.log_scale(7.0), f.log_scale(7.0))


def test_a_downstream_unit_norm_rescale_composes_by_addition():
    """What the invariant is FOR. The solver may divide each column by its own norm; the
    coefficient of the mathematical function is then recovered from the sum of two logs, and
    neither layer needs to know the other's number."""
    f = fam(1.4 * np.pi, n=6)
    k = 7.0
    pts = np.linspace(0.05, 0.95, 60) * np.exp(0.7j)
    A = f.values(k, pts)
    d = np.linalg.norm(A, axis=0)
    total = f.log_scale(k) + np.log(d)
    # the doubly-scaled column times exp(total) is the mathematical function, exactly.
    assert np.allclose((A / d) * np.exp(total)[None, :], A * np.exp(f.log_scale(k))[None, :],
                       rtol=1e-14, atol=0.0)


# --- scale covariance ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", [1e-3, 0.5, 2.0, 1e3])
def test_the_scale_is_exactly_invariant_under_k_to_k_times_s(s):
    """`sigma` depends on k and R only through k*R, so `Domain.normalized(by=s)` with
    `k -> k*s` leaves every column scale bit-for-bit unchanged. An equality test, not a
    tolerance test -- if it ever needs a tolerance, the formula grew a term it should not
    have."""
    m = CornerModes.from_wedge(1.4 * np.pi, 0.0, 0.0)
    mu = m.orders(m.degrees(30))
    assert np.array_equal(_log_envelope(mu, 12.0), _log_envelope(mu, (12.0 / s) * s))
    a = FourierBessel(0j, 1 + 0j, m, m.degrees(30), 1.0).log_scale(12.0)
    b = FourierBessel(0j, 1 + 0j, m, m.degrees(30), 1.0 / s).log_scale(12.0 * s)
    assert np.array_equal(a, b)


# --- smoothness in k ------------------------------------------------------------------------------

def test_the_scale_has_no_kink_where_the_two_branches_meet():
    """A soft minimum, not `min`, and a floored amplitude, not the raw Debye one.

    `sigma(k)` is inside the objective the solver parabola-fits, so a kink there is a kink in
    the minimand. Measured as a second difference in k across the crossover k*R = mu.

    The numbers this pins down, all measured on the way in:
      * raw Debye amplitude, `-0.5*log(2*pi*mu*w)`:     8.4e5 at mu = 20
      * floored at the Airy width, blended softly:        6.5 at mu = 20
    on k in [19, 21] at 4001 samples. See the next test for the full comparison.
    """
    for mu0, cap in ((2.0, 50.0), (20.0, 10.0), (200.0, 2.0)):
        mu = np.array([mu0])
        k = np.linspace(0.5 * mu0, 1.5 * mu0, 4001)
        h = k[1] - k[0]
        sigma = np.array([_log_envelope(mu, kk)[0] for kk in k])
        assert np.abs(np.diff(sigma, 2)).max() / h**2 < cap, f"mu={mu0}"


def test_the_shipped_blend_beats_both_simpler_versions():
    """The comparison that justifies paying 1.19x in the column maximum for the soft blend
    and a floored amplitude. All three measured on the same k grid across the crossover, as
    a second difference:

        shipped: floored amplitude, soft blend      6.5
        unfloored amplitude, plain logaddexp    8.4e+05     128000x worse
        unfloored amplitude, hard min           1.9e+03        296x worse

    Note the ordering: without the floor, the SMOOTH blend is worse than `min`, because
    `min` happens to clip the amplitude's divergence while `logaddexp` follows it in. The
    floor is the load-bearing fix; the soft blend is what removes the kink that remains.
    """
    mu0 = 20.0
    mu = np.array([mu0])
    k = np.linspace(19.0, 21.0, 4001)
    h = k[1] - k[0]
    b = np.log(0.6748 * (mu0 + 0.3) ** (-1 / 3))

    def unfloored(kk, hard):
        z = kk / mu0
        if z >= 1.0:
            return b
        w = np.sqrt(1 - z * z)
        a = mu0 * (w + np.log(z / (1 + w))) - 0.5 * np.log(2 * np.pi * mu0 * w)
        return min(a, b) if hard else -np.logaddexp(-a, -b)

    d2 = lambda v: np.abs(np.diff(v, 2)).max() / h**2
    shipped = d2(np.array([_log_envelope(mu, kk)[0] for kk in k]))
    assert d2(np.array([unfloored(kk, False) for kk in k])) > 1e4 * shipped
    assert d2(np.array([unfloored(kk, True) for kk in k])) > 100 * shipped


# --- the accuracy diagnostic -----------------------------------------------------------------------

def test_accuracy_bounds_the_error_it_reports_on():
    """`Sampling.accuracy(k)` is a per-column relative-error bound, free out of the series
    loop. It has to be an upper bound or it is worse than nothing. Same pattern as
    `Domain.achievable_prec` one layer down: microseconds to compute, and it never claims a
    precision the object cannot deliver."""
    f = fam(np.pi / 8, n=20)
    k = 12.0
    r = np.array([0.2, 0.5, 0.85])
    phi = PHI_FRAC * f.modes.theta
    pts = r * np.exp(1j * phi)
    s = f.at(pts)
    bound = s.accuracy(k)
    sigma = f.log_scale(k)
    ang = np.sin(f.orders * phi)
    got = f.values(k, pts)
    for j, mu in enumerate(f.orders):
        for i, ri in enumerate(r):
            _, ref = mp_log_besselj(mu, k * ri)
            actual = abs(np.log(abs(got[i, j] / ang[j])) + sigma[j] - ref)
            assert actual <= bound[j] * 20, f"col {j} mu={mu:.2f} r={ri}"


def test_accuracy_is_cheap_enough_to_ask_every_time():
    """It comes off the same loop as the values. If it ever needs its own Bessel pass, it
    stops being a diagnostic and becomes a cost."""
    import time

    f = fam(np.pi / 8, n=60)
    s = f.at(np.linspace(0.01, 0.99, 400) * np.exp(0.2j))
    t0 = time.perf_counter(); s.values(9.0); t_v = time.perf_counter() - t0
    t0 = time.perf_counter(); s.accuracy(9.0); t_a = time.perf_counter() - t0
    assert t_a < 5 * t_v + 5e-3
