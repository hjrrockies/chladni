"""The scaled Bessel kernel, against mpmath at 50 dps.

What this file defends: that `J_nu(x)` and `Y_m(x)` come back at full RELATIVE accuracy in
the regime where float64 cannot hold the value at all. `scipy.special.jv` holds 13-14 digits
and then flushes to exactly 0.0 at |J| ~ 1e-291 -- measured underflow edges: x = 0.096 at
nu = 100, 5.32 at 200, 51.74 at 400. lappy lived with that, and its Fourier-Bessel columns at
a sharp corner are literally zero arrays. A column representable to full relative accuracy is
a different object from a column of zeros, and only one of the two is worth truncating.

Tolerances here are measured, not aspirational. The kernel is a 5e-13-relative object at
nu ~ 400, not a 1e-15 one; do not tighten `KERNEL_REL` without re-measuring.
"""
import numpy as np
import pytest

from conftest import (
    mp_besselj,
    mp_besselj_prime,
    mp_bessely,
    mp_bessely_prime,
    mp_log_besselj,
    mp_log_bessely,
)

from chladni.helmholtz._bessel import log_j, log_y

#: Measured worst case against mpmath over nu <= 400, x <= 50 is 2.8e-13.
KERNEL_REL = 5e-13


def _value(scaled):
    """The mathematical value, where float64 can hold it."""
    return scaled.S * np.exp(scaled.sigma)


def _log_value(scaled):
    """(sign, log|value|) -- the comparison that still works past the underflow edge."""
    return np.sign(scaled.S), np.log(np.abs(scaled.S)) + scaled.sigma


# --- J: the regime float64 can hold -------------------------------------------------------

@pytest.mark.parametrize("nu", [0.0, 0.5, 1.0, 2.0, 2.5, 10.0, 40.0, 80.0])
@pytest.mark.parametrize("x", [0.01, 0.1, 1.0, 5.0, 20.0, 50.0])
def test_scaled_j_matches_mpmath_where_float64_can_hold_it(nu, x):
    got = float(_value(log_j(nu, x)))
    assert got == pytest.approx(mp_besselj(nu, x), rel=KERNEL_REL)


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.5, 10.0, 40.0, 80.0])
@pytest.mark.parametrize("x", [0.01, 0.1, 1.0, 5.0, 20.0, 50.0])
def test_scaled_j_prime_matches_mpmath(nu, x):
    s = log_j(nu, x)
    got = float(s.Sp * np.exp(s.sigma))
    assert got == pytest.approx(mp_besselj_prime(nu, x), rel=KERNEL_REL)


# --- J: the regime that is the reason this module exists ----------------------------------

#: (nu, x) pairs where scipy.special.jv returns exactly 0.0 while the true value is finite.
#: Verified below that scipy really does fail on every one of them, so the test cannot
#: silently degrade into a re-test of scipy.
UNDERFLOWED = [
    (150.0, 0.1), (150.0, 1.0), (250.0, 0.1), (250.0, 1.0), (250.0, 5.0),
    (350.0, 1.0), (350.0, 5.0), (350.0, 20.0), (400.0, 20.0), (500.0, 50.0),
]


@pytest.mark.parametrize("nu,x", UNDERFLOWED)
def test_scipy_really_does_underflow_on_these(nu, x):
    """The premise. If scipy ever fixes this the test below stops meaning anything."""
    from scipy.special import jv

    assert jv(nu, x) == 0.0


@pytest.mark.parametrize("nu,x", UNDERFLOWED)
def test_scaled_j_is_relatively_accurate_past_the_underflow_edge(nu, x):
    sign, logv = mp_log_besselj(nu, x)
    got_sign, got_log = _log_value(log_j(nu, x))
    assert got_sign == sign
    # An absolute error in the log IS a relative error in the value.
    assert abs(got_log - logv) < KERNEL_REL


@pytest.mark.parametrize("nu,x", UNDERFLOWED)
def test_scaled_j_prime_is_relatively_accurate_past_the_underflow_edge(nu, x):
    import mpmath as mp

    with mp.workdps(50):
        n, xx = mp.mpf(nu), mp.mpf(x)
        ref = 0.5 * (mp.besselj(n - 1, xx) - mp.besselj(n + 1, xx))
        ref_log = float(mp.log(abs(ref)))
        ref_sign = 1 if ref > 0 else -1
    s = log_j(nu, x)
    assert np.sign(s.Sp) == ref_sign
    assert abs(np.log(abs(s.Sp)) + s.sigma - ref_log) < KERNEL_REL


# --- the structural invariant -------------------------------------------------------------

def test_scaled_j_never_exceeds_one_in_magnitude():
    """Watson: |J_nu(x)| <= (x/2)**nu / Gamma(nu+1) for nu >= -1/2, and that bound IS
    exp(sigma). So |S| <= 1 identically, in every branch. This is the cheapest possible
    check that a branch is using the canonical sigma and not one of its own."""
    nu = np.array([0.0, 0.3, 1.0, 3.7, 20.0, 100.0, 400.0, 900.0])[:, None]
    x = np.array([1e-3, 0.1, 1.0, 7.0, 30.0, 120.0, 400.0, 1500.0])[None, :]
    s = log_j(nu, x)
    assert np.all(np.abs(s.S) <= 1.0 + 1e-14)


def test_the_scale_is_the_leading_small_argument_term():
    """sigma is canonical and analytic, not a log of the computed value -- which is what
    lets a caller subtract two of them without knowing which branch either one took."""
    from scipy.special import gammaln

    nu, x = 37.5, 0.25
    s = log_j(nu, x)
    assert float(s.sigma) == pytest.approx(nu * np.log(x / 2) - gammaln(nu + 1), rel=1e-15)


def test_S_tends_to_one_as_the_argument_vanishes():
    s = log_j(12.0, 1e-8)
    assert float(s.S) == pytest.approx(1.0, abs=1e-15)


# --- broadcasting and shape ----------------------------------------------------------------

def test_orders_and_arguments_broadcast_as_a_block():
    """The branch mask is taken over the whole (m, n) block, not per column -- one
    vectorized call per branch. A per-column mask is the shape of a performance bug."""
    nu = np.linspace(1.0, 300.0, 7)[None, :]
    x = np.linspace(0.5, 40.0, 11)[:, None]
    s = log_j(nu, x)
    assert s.S.shape == (11, 7)
    assert s.Sp.shape == (11, 7)
    assert s.sigma.shape == (11, 7)
    for i in range(11):
        for j in range(7):
            one = log_j(nu[0, j], x[i, 0])
            assert float(s.S[i, j]) == pytest.approx(float(one.S), rel=1e-15)
            assert float(s.sigma[i, j]) == pytest.approx(float(one.sigma), rel=1e-15)


def test_scalars_in_scalars_out():
    s = log_j(2.0, 3.0)
    assert np.ndim(s.S) == 0


# --- the free precision diagnostic ---------------------------------------------------------

def test_rel_err_bounds_the_measured_error():
    """`rel_err` is max|term| / |S| from the series loop, times eps. It has to be an upper
    bound on the real thing or it is worse than nothing."""
    cases = [(400.0, 20.0), (400.0, 50.0), (200.0, 50.0), (100.0, 50.0), (40.0, 30.0)]
    for nu, x in cases:
        s = log_j(nu, x)
        _, ref_log = mp_log_besselj(nu, x)
        actual = abs(np.log(abs(s.S)) + s.sigma - ref_log)
        assert actual <= max(float(s.rel_err), 1e-16) * 10


def test_rel_err_grows_with_the_cancellation():
    """The diagnostic must actually track the cancellation, not just be conservative
    everywhere: along fixed nu, more argument means a bigger largest term.

    Measured against the series branch itself, not the dispatcher -- past x ~ 5*sqrt(nu) the
    dispatcher has already switched to scipy, whose error is a function of the order and not
    of a cancellation the series never performed."""
    from chladni.helmholtz._bessel import _j_series

    nu = 200.0
    errs = [float(_j_series(nu, x).rel_err) for x in (5.0, 20.0, 50.0, 100.0)]
    assert errs == sorted(errs)


# --- Y: the same problem with the opposite sign ---------------------------------------------

@pytest.mark.parametrize("m", [0, 1, 2, 5, 20])
@pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 20.0, 50.0])
def test_scaled_y_matches_mpmath_where_float64_can_hold_it(m, x):
    got = float(_value(log_y(m, x)))
    assert got == pytest.approx(mp_bessely(m, x), rel=KERNEL_REL)


@pytest.mark.parametrize("m", [0, 1, 2, 5, 20])
@pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 20.0, 50.0])
def test_scaled_y_prime_matches_mpmath(m, x):
    s = log_y(m, x)
    got = float(s.Sp * np.exp(s.sigma))
    assert got == pytest.approx(mp_bessely_prime(m, x), rel=KERNEL_REL)


#: Y overflows rather than underflowing. Bisected against scipy directly: the last finite
#: order is 65 at x = 1e-3, 80 at 1e-2, 149 at x = 1, 205 at 5, 294 at 20 -- and the value
#: there is ~1e302, so this is a clean cliff at the float64 ceiling, not a decay.
OVERFLOWED = [(70, 1e-3), (90, 1e-2), (160, 1.0), (220, 5.0), (320, 20.0)]


@pytest.mark.parametrize("m,x", OVERFLOWED)
def test_scipy_really_does_overflow_on_these(m, x):
    from scipy.special import yv

    assert not np.isfinite(yv(m, x))


@pytest.mark.parametrize("m,x", OVERFLOWED)
def test_scaled_y_is_relatively_accurate_past_the_overflow_edge(m, x):
    sign, logv = mp_log_bessely(m, x)
    s = log_y(m, x)
    assert np.sign(s.S) == sign
    assert abs(np.log(abs(s.S)) + s.sigma - logv) < KERNEL_REL


def test_y0_is_not_given_a_power_law_scale():
    """Y_0 has a log singularity, not a power one, so its canonical scale is 0 -- the
    m >= 1 formula would divide by a Gamma(0) that is not there."""
    s = log_y(0, 1e-6)
    assert float(s.sigma) == 0.0
    assert float(s.S) == pytest.approx(mp_bessely(0, 1e-6), rel=KERNEL_REL)
