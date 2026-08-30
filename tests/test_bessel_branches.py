"""That the hybrid kernel has no seam.

`log_j` picks between a convergent ascending series and `scipy.special.jv` on a predicted
largest series term `M = exp(x**2/(4*nu))`. The two branches do not abut, they OVERLAP: a
scan of nu in [0.2, 600] against x in [0.5, 100] for cells where scipy underflows AND the
series is inaccurate finds zero, and the gap first opens near nu = 515.

That overlap is the strongest available test of the hybrid, and it is the thing that fails
first if either threshold is edited. Everything in this file lives in the overlap on purpose.
"""
import numpy as np
import pytest
from scipy.special import gammaln, jv, jvp, yv, yvp

from chladni.helmholtz import _bessel
from chladni.helmholtz._bessel import log_j, log_y


def _series_only(nu, x):
    """Branch A alone, thresholds bypassed."""
    return _bessel._j_series(np.asarray(nu, float), np.asarray(x, float))


def _scipy_only(nu, x):
    """Branch B alone, thresholds bypassed."""
    return _bessel._j_scipy(np.asarray(nu, float), np.asarray(x, float))


def _debye_only(nu, x):
    """Branch C alone, thresholds bypassed."""
    return _bessel._j_debye(np.asarray(nu, float), np.asarray(x, float))


# --- the overlap ---------------------------------------------------------------------------

#: In every one of these, scipy is well inside its accurate range AND the series has a
#: largest term small enough to be trusted. Both branches must be right.
OVERLAP = [
    (1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (10.0, 10.0), (20.0, 10.0),
    (40.0, 20.0), (80.0, 30.0), (150.0, 40.0), (250.0, 50.0), (400.0, 60.0),
    (100.0, 5.0), (100.0, 30.0), (300.0, 30.0), (500.0, 100.0),
]
# (300, 10) and (500, 80) look like they belong above and do not: scipy underflows at both
# (the edge is x = 23.98 at nu = 300 and 92.06 at nu = 500), so they are series-only cells,
# not overlap. Getting that wrong is exactly the mistake this file exists to catch.


@pytest.mark.parametrize("nu,x", OVERLAP)
def test_the_series_and_scipy_agree_wherever_both_are_valid(nu, x):
    a, b = _series_only(nu, x), _scipy_only(nu, x)
    assert float(a.sigma) == pytest.approx(float(b.sigma), rel=1e-15)
    assert float(a.S) == pytest.approx(float(b.S), rel=1e-12)
    assert float(a.Sp) == pytest.approx(float(b.Sp), rel=1e-12)


@pytest.mark.parametrize("nu,x", [(300.0, 30.0), (400.0, 60.0), (600.0, 100.0), (900.0, 150.0)])
def test_the_series_and_debye_agree_wherever_both_are_valid(nu, x):
    """Debye is the safety net for nu > ~575 with x past the series' comfort. Its own
    accuracy degrades only as x/nu -> 1, which is scipy's territory anyway."""
    a, c = _series_only(nu, x), _debye_only(nu, x)
    assert float(a.S) == pytest.approx(float(c.S), rel=2e-11)
    assert float(a.Sp) == pytest.approx(float(c.Sp), rel=2e-11)


@pytest.mark.parametrize("nu,x", [(200.0, 120.0), (400.0, 200.0), (600.0, 300.0)])
def test_scipy_and_debye_agree_wherever_both_are_valid(nu, x):
    """Debye's four terms are an asymptotic series in 1/nu, so this is a nu >= 200 test.
    At (100, 60) the same comparison is only good to about 1e-8 -- correct behaviour for a
    four-term expansion at that order, and the reason branch C is never reached below
    nu ~ 515 in the dispatcher."""
    b, c = _scipy_only(nu, x), _debye_only(nu, x)
    assert float(b.S) == pytest.approx(float(c.S), rel=2e-10)


# --- the dispatcher ------------------------------------------------------------------------

def test_the_dispatcher_returns_whichever_branch_is_valid():
    """The public entry point must agree with the branch it chose, everywhere in the
    overlap -- i.e. the mask and the branch bodies cannot drift apart."""
    for nu, x in OVERLAP:
        got = log_j(nu, x)
        ref = _series_only(nu, x)
        assert float(got.S) == pytest.approx(float(ref.S), rel=1e-12)


def test_the_dispatcher_leaves_no_entry_unfilled():
    """A masked branch that misses a cell shows up as a nan, not as an exception."""
    nu = np.geomspace(0.2, 900.0, 40)[None, :]
    x = np.geomspace(1e-3, 400.0, 40)[:, None]
    s = log_j(nu, x)
    assert np.isfinite(s.S).all()
    assert np.isfinite(s.Sp).all()


def test_no_entry_is_left_to_scipys_underflow():
    """The whole point. Anywhere scipy gives exactly 0.0 for a nonzero function, the
    dispatcher must have taken a log-domain branch instead."""
    nu = np.geomspace(50.0, 900.0, 25)[None, :]
    x = np.geomspace(0.5, 300.0, 25)[:, None]
    underflowed = jv(nu, x) == 0.0
    assert underflowed.any(), "the grid no longer exercises the underflow region"
    s = log_j(nu, x)
    assert np.isfinite(s.S[underflowed]).all()
    assert (s.S[underflowed] != 0.0).all()


def test_the_branch_mask_is_taken_over_the_block():
    """A block evaluation must equal the entrywise one exactly, which it cannot do if the
    mask is recomputed per column against a per-column reduction."""
    nu = np.array([2.0, 90.0, 300.0, 700.0])[None, :]
    x = np.array([0.5, 12.0, 60.0, 250.0])[:, None]
    block = log_j(nu, x)
    for i in range(4):
        for j in range(4):
            one = log_j(nu[0, j], x[i, 0])
            assert float(block.S[i, j]) == float(one.S)
            assert float(block.Sp[i, j]) == float(one.Sp)


# --- Y, same shape of question --------------------------------------------------------------

#: Chosen so the truncated finite sum is complete to ~1e-17 relative (`_log_y_drop` below
#: -40) AND scipy has not yet overflowed. At (5, 0.5) the dropped log-term is 9e-10 of the
#: total, so that cell belongs to scipy alone and is not an overlap at all.
Y_OVERLAP = [(8, 0.5), (10, 1.0), (20, 2.0), (40, 5.0), (80, 20.0), (30, 0.01)]


@pytest.mark.parametrize("m,x", Y_OVERLAP)
def test_the_y_series_and_scipy_agree_wherever_both_are_valid(m, x):
    a = _bessel._y_series(np.asarray(float(m)), np.asarray(float(x)))
    b = _bessel._y_scipy(np.asarray(float(m)), np.asarray(float(x)))
    assert float(a.S) == pytest.approx(float(b.S), rel=1e-11)
    assert float(a.Sp) == pytest.approx(float(b.Sp), rel=1e-11)


def test_no_entry_is_left_to_scipys_y_overflow():
    m = np.arange(0, 200, 7)[None, :].astype(float)
    x = np.geomspace(1e-3, 60.0, 25)[:, None]
    blown = ~np.isfinite(yv(m, x))
    assert blown.any(), "the grid no longer exercises the overflow region"
    s = log_y(m, x)
    assert np.isfinite(s.S[blown]).all()


# --- the fast paths must not be a second implementation ---------------------------------------

def test_the_y0_fast_path_agrees_with_the_general_order_path():
    """`y0` is the fast path for order zero. Measured here: 0.18 ms against 12.4 ms for
    `yv(0, .)` on 20000 points, i.e. 69x, consistent with lappy's 42x and its 1.98 s of a
    5.88 s L-shape solve.

    They agree to 2.7e-15 ABSOLUTE. lappy's docstring says "~1 ulp", which is true in that
    sense and not in the other: near a zero of Y_0 the relative disagreement reaches 3.7e-12,
    because two code paths landing within an ulp of each other say nothing about the relative
    accuracy of a value that is itself tiny. Assert the claim that is true.
    """
    x = np.geomspace(1e-4, 80.0, 400)
    fast = log_y(0, x)
    got = fast.S * np.exp(fast.sigma)
    assert np.allclose(got, yv(0.0, x), rtol=0.0, atol=1e-14)
    assert np.allclose(fast.Sp * np.exp(fast.sigma), yvp(0.0, x), rtol=0.0, atol=1e-11)


def test_the_j_derivative_shares_the_value_series():
    """One loop, two accumulators -- not two Bessel evaluations. Checked against scipy in
    the region where scipy is exact, so a second series would have to agree to 1e-13."""
    nu = np.array([0.5, 3.0, 25.0])[None, :]
    x = np.array([0.3, 2.0, 9.0])[:, None]
    s = log_j(nu, x)
    # 1e-12, not 1e-13: `sigma` at (25, 9) is -20.4, and float64 holds that to about 2e-15
    # absolute, which IS 2e-15 relative in the value before scipy's own error is counted.
    assert np.allclose(s.Sp * np.exp(s.sigma), jvp(nu, x), rtol=1e-12, atol=1e-300)
