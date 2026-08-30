"""The scaled Bessel kernel: `J_nu(x)` and `Y_m(x)` at full relative accuracy, everywhere.

`scipy.special.jv` is excellent right up to the point where it is useless. It holds 13-14
digits and then flushes to exactly 0.0 -- there is no accuracy-loss band, only a cliff at
|J| ~ 1e-291, about 17 decades above DBL_MIN. Measured edges (bisected against mpmath at
50 dps):

    nu                      100    200    300    400    500    700   1000
    x where jv hits 0      0.096   5.32  23.98  51.74  92.06  194.1  380.4

At a sharp corner a Fourier-Bessel order reaches 100-350 while k*r stays small, so this is
not an edge case for this project -- it is the working regime. lappy lived with it, and its
FB columns at a sharp corner are literally zero arrays; its own benchmark notebook then
concluded those columns "carry almost no independent information about the domain". Half of
that is true (near-dependence at a sharp corner is intrinsic) and half is an artifact. A
column representable at full RELATIVE accuracy is a different object from a column of zeros,
and only one of the two is worth truncating.

So every value comes back factored:

    J_nu(x) = S * exp(sigma),    J_nu'(x) = Sp * exp(sigma)

with `sigma = nu*log(x/2) - gammaln(nu+1)` -- the log of the leading series term. That choice
is canonical in two useful ways. It is the SAME quantity in every branch, so which branch an
entry took is invisible to the caller and two sigmas can be subtracted without asking. And
Watson's bound `|J_nu(x)| <= (x/2)**nu / Gamma(nu+1)` for `nu >= -1/2` is exactly
`|S| <= 1`, an invariant the test suite asserts directly.

`sigma` is analytic and z-independent by construction, never a log of the computed value.
That matters in the oscillatory regime `nu <~ x`, where `log|J|` is genuinely singular at
every zero: the scale stays smooth and the scaled value keeps the oscillation, the sign
changes and the zeros.

Three branches, selected by a mask over the WHOLE block so each is one vectorized call:

    A  small largest term    the convergent ascending series      13+ digits
    B  otherwise             scipy, rescaled                      13-14 digits
    C  B underflowed         four-term Olver/Debye                <= 2e-12

They overlap widely rather than abutting. A scan of nu in [0.2, 600] against x in [0.5, 100]
for cells where scipy underflows AND the series is inaccurate finds ZERO; the gap first opens
near nu = 515, which is why branch C is a safety net rather than a workhorse.

Branch A is not a slow fallback. Measured on a 600 x 100 sharp-corner block (k = 30, r out
to 0.4, orders 3.5 to 350): 5.7 ms for values AND derivatives, against 4.8 ms for
`scipy.special.jv` alone -- which returned 43% exact zeros -- and 14.2 ms for `jv` plus
`jvp`, which is the honest comparison because that is what the caller actually needs. So the
log-domain path costs about 20% over a `jv` call that is wrong on nearly half the block, and
saves 2.5x over a scipy pair that is wrong on nearly half the block. One loop, two
accumulators, no second Bessel evaluation.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import gammaln, jv, jvp, y0, y1, yv, yvp

#: Largest series term we will accept, as a log. The series' relative error is about
#: eps * (largest term), and the fitted largest term is exp(x**2/(4*nu)) -- measured to
#: predict the real thing within ~20x, always conservatively. log(1e3) keeps branch A at
#: 1e-12 or better while still covering every cell where scipy underflows below nu ~ 515.
_LOG_MAXTERM = np.log(1e3)

#: Hard stop on the series loop. The real counts are small (3 terms at x = 0.01, 24 at
#: (400, 50), 45 at (400, 100)); this only exists so a pathological call cannot hang.
_MAX_TERMS = 600

#: Below this the truncated Y_m finite sum is not the whole function. The dropped pieces are
#: relatively exp(_log_y_drop(m, x))-small, so the cut is on that ratio and not on `sigma`.
_LOG_Y_DROP = np.log(1e-16)

_EPS = np.finfo(float).eps


class Scaled(NamedTuple):
    """A Bessel value in factored form.

    `value = S * exp(sigma)` and `derivative = Sp * exp(sigma)`. `rel_err` bounds the
    relative error of `S` and `Sp` -- it falls out of the series loop for free (it is
    `eps * largest_term / |S|`) and is what `Sampling.accuracy` reports upward.
    """

    S: np.ndarray
    Sp: np.ndarray
    sigma: np.ndarray
    rel_err: np.ndarray


# --- the canonical scale -------------------------------------------------------------------

def _scale_err(a, b):
    """The relative error a large log costs, as the two pieces of `sigma` are formed.

    This is the floor nobody expects and it dominates everything else in the sharp-corner
    regime. `sigma` at (nu, x) = (400, 50) is -712.95, assembled from 1287.7 and 2000.6;
    float64 holds each to about 1e-16 RELATIVE, i.e. 2e-13 ABSOLUTE, and an absolute error
    in a log is a relative error in the value. Measured error there is 1.1e-13 against a
    largest-term bound of only 1.6e-15 -- so a bound that omitted this term would be wrong
    by two orders of magnitude, in the unsafe direction.

    It is also, and usefully, CONSTANT ALONG A COLUMN: `nu` is fixed per basis function, and
    while `x = k*r` varies, `nu*log(x/2)` varies only as `nu*log(r)`, so at fixed `k` the
    gammaln half is a pure per-column factor. A scale-invariant consumer -- which the GSVD
    is -- sees the largest-term part of the bound and not this part. `Sampling.accuracy`
    reports the honest entrywise number; the solver is entitled to know it is conservative.
    """
    return _EPS * (np.abs(a) + np.abs(b))


def _sigma_j(nu, x):
    """`nu*log(x/2) - gammaln(nu+1)`, with x = 0 handled rather than left to 0 * -inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = nu * np.log(x / 2) - gammaln(nu + 1)
    # At x = 0 the factorization degenerates: put the whole value in S and take sigma = 0,
    # which keeps |S| <= 1 and keeps every downstream `exp(sigma_a - sigma_b)` finite.
    return np.where(x > 0, s, 0.0)


def _sigma_y(m, x):
    """`gammaln(m) - m*log(x/2) - log(pi)` for m >= 1; 0 for m = 0.

    Y_0 has a log singularity rather than a power one, so the m >= 1 formula would want a
    Gamma(0) that is not there. Its scale is simply 1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        s = gammaln(np.maximum(m, 1.0)) - m * np.log(x / 2) - np.log(np.pi)
    return np.where((m >= 1) & (x > 0), s, 0.0)


# --- branch A: the ascending series ----------------------------------------------------------

def _j_series(nu, x):
    """`J_nu` and `J_nu'` from the convergent ascending series, in one loop.

    `S = sum_j t_j` with `t_0 = 1` and `t_j = t_{j-1} * q / (j*(nu+j))`, `q = -x**2/4`;
    differentiating termwise gives `x*J' = sum_j (nu+2j) t_j` off the SAME `t_j`. The
    derivative therefore costs one accumulator, not a second Bessel evaluation -- which is
    the whole reason `values_and_grad` can promise a single transcendental pass.

    Accuracy is governed entirely by the largest term: measured relative error is about
    `eps * max_j|t_j|`, i.e. 5e-14 typical and 2.8e-13 worst over nu <= 400, x <= 50, and
    still 1.8e-12 at (400, 100). `rel_err` reports that bound, computed in the loop.
    """
    nu, x = np.broadcast_arrays(np.asarray(nu, float), np.asarray(x, float))
    q = -0.25 * x * x
    t = np.ones_like(x)
    S = np.ones_like(x)
    Sx = np.array(nu, dtype=float)          # accumulates x*J'/exp(sigma)
    biggest = np.ones_like(x)
    # A direct call outside the dispatcher's mask can overflow -- that is a caller asking for
    # a branch the dispatcher would not have chosen, and the honest answer is inf, not a
    # crash and not a warning storm.
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(1, _MAX_TERMS):
            t = t * q / (j * (nu + j))
            S = S + t
            Sx = Sx + (nu + 2 * j) * t
            at = np.abs(t)
            biggest = np.maximum(biggest, at)
            if np.all(at <= _EPS * np.abs(S)):
                break
    with np.errstate(divide="ignore", invalid="ignore"):
        Sp = np.where(x > 0, Sx / np.where(x > 0, x, 1.0), _dj_at_zero(nu))
        rel = _EPS * biggest / np.abs(S) + _scale_err(nu * np.log(x / 2), gammaln(nu + 1))
    return Scaled(_at_zero_value(nu, x, S), Sp, _sigma_j(nu, x), rel)


def _at_zero_value(nu, x, S):
    """At x = 0, sigma is 0 by convention, so S must carry the value itself: 1 at nu = 0
    and 0 otherwise."""
    return np.where(x > 0, S, np.where(nu == 0, 1.0, 0.0))


def _dj_at_zero(nu):
    """`J_nu'(0)`: 0 at nu = 0 and nu > 1, 1/2 at nu = 1, and genuinely infinite for
    0 < nu < 1 -- which is a real singularity of the corner expansion, not a code defect,
    and is why a collocation node must never sit exactly on a corner vertex."""
    return np.where((nu == 0) | (nu > 1), 0.0, np.where(nu == 1, 0.5, np.inf))


def _y_series(m, x):
    """`Y_m` and `Y_m'` from the leading finite sum, for m >= 1.

    `Y_m` overflows rather than underflowing -- bisected against scipy, the last finite
    order is 65 at x = 1e-3, 80 at 1e-2, 149 at x = 1, 205 at 5, 294 at 20, with the value
    there around 1e302 -- so this is the same problem with the opposite sign. Factoring
    `Gamma(m)(2/x)**m/pi` out of the finite sum leaves `a_0 = 1` and
    `a_{j+1} = a_j*(x**2/4)/((m-j-1)*(j+1))`, and the same termwise differentiation gives
    `x*Y' = -sum_j (2j - m) a_j` off the same `a_j`.

    The dropped pieces (the `J_m log(x/2)` term and the psi-series) are relatively
    `exp(_log_y_drop)`-small, which is what the dispatcher branches on.
    """
    m, x = np.broadcast_arrays(np.asarray(m, float), np.asarray(x, float))
    quarter = 0.25 * x * x
    a = np.ones_like(x)
    S = np.ones_like(x)
    Sx = -np.array(m, dtype=float)          # accumulates x*Y'/(-exp(sigma))
    live = m >= 2                            # a_1 exists only when m >= 2
    for j in range(0, int(np.max(m, initial=1.0))):
        if not np.any(live):
            break
        denom = (m - j - 1) * (j + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(live, quarter / np.where(live, denom, 1.0), 0.0)
        a = a * step
        S = S + np.where(live, a, 0.0)
        Sx = Sx + np.where(live, (2 * (j + 1) - m) * a, 0.0)
        live = live & (m >= j + 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        Sp = np.where(x > 0, -Sx / np.where(x > 0, x, 1.0), np.inf)
        rel = 4 * _EPS + _scale_err(m * np.log(x / 2), gammaln(np.maximum(m, 1.0)))
    return Scaled(-S, Sp, _sigma_y(m, x), rel)


# --- branch B: scipy, rescaled ----------------------------------------------------------------

def _rescale(value, sigma):
    """`value * exp(-sigma)`, formed in logs so a large sigma cannot overflow on the way.

    A legitimate zero of the function survives: log(0) is -inf and exp(-inf) is 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sign(value) * np.exp(np.log(np.abs(value)) - sigma)


def _j_scipy(nu, x):
    nu, x = np.broadcast_arrays(np.asarray(nu, float), np.asarray(x, float))
    sigma = _sigma_j(nu, x)
    # scipy's own relative error grows with the order: measured 3e-17 at nu = 10 and
    # 1.9e-13 at nu = 250. Report that rather than a flat eps nobody measured.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.maximum(_EPS, 4e-16 * (1.0 + nu)) + _scale_err(nu * np.log(x / 2),
                                                                gammaln(nu + 1))
    return Scaled(_rescale(jv(nu, x), sigma), _rescale(jvp(nu, x), sigma), sigma, rel)


def _y_scipy(m, x):
    m, x = np.broadcast_arrays(np.asarray(m, float), np.asarray(x, float))
    sigma = _sigma_y(m, x)
    # lappy measured `y0` at 42x `yv(0, .)` and `y1` at 85x `yvp(0, .)` -- 1.98 s of a
    # 5.88 s L-shape solve, for 36 of 159 columns. The guard is on the ORDER VALUES rather
    # than on a stored `order` attribute, so a hand-assembled family still takes the path.
    if np.all(m == 0):
        val, der = y0(x), -y1(x)             # Y_0' = -Y_1, exactly
    else:
        val, der = yv(m, x), yvp(m, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.maximum(_EPS, 4e-16 * (1.0 + m)) + _scale_err(m * np.log(x / 2),
                                                               gammaln(np.maximum(m, 1.0)))
    return Scaled(_rescale(val, sigma), _rescale(der, sigma), sigma, rel)


# --- branch C: the Olver/Debye uniform asymptotic -----------------------------------------------

#: Olver's U_k and V_k as (power of p, coefficient) pairs, p = 1/sqrt(1 - (x/nu)**2).
#: DLMF 10.19.6 and 10.19.7. Stored as explicit powers rather than a dense coefficient
#: tuple: the polynomials are sparse in p (only every other power appears) and a positional
#: encoding invites exactly the off-by-one that a 2e-7 disagreement with the series caught.
_U = [
    [(0, 1.0)],
    [(1, 3.0 / 24), (3, -5.0 / 24)],
    [(2, 81.0 / 1152), (4, -462.0 / 1152), (6, 385.0 / 1152)],
    [(3, 30375.0 / 414720), (5, -369603.0 / 414720),
     (7, 765765.0 / 414720), (9, -425425.0 / 414720)],
]
_V = [
    [(0, 1.0)],
    [(1, -9.0 / 24), (3, 7.0 / 24)],
    [(2, -135.0 / 1152), (4, 594.0 / 1152), (6, -455.0 / 1152)],
    [(3, -42525.0 / 414720), (5, 451737.0 / 414720),
     (7, -883575.0 / 414720), (9, 475475.0 / 414720)],
]


def _olver_sum(coeffs, p, nu):
    """`sum_k C_k(p) / nu**k`, the asymptotic correction factor. O(1) by construction."""
    total = np.zeros_like(p)
    for k, terms in enumerate(coeffs):
        term = np.zeros_like(p)
        for power, c in terms:
            term = term + c * p**power
        total = total + term / nu**k
    return total


def _j_debye(nu, x):
    """The `nu > x` uniform asymptotic, converted back onto the canonical `sigma`.

    Only reachable where scipy has already underflowed and the series' largest term is too
    big to trust, which the branch scan says starts near nu = 515. Measured four-term
    accuracy at the corner where it is reachable: <= 2e-12. It degrades only as x/nu -> 1,
    the Airy transition -- which is exactly where scipy is fine and this is never called.
    """
    nu, x = np.broadcast_arrays(np.asarray(nu, float), np.asarray(x, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x / nu
        w = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        p = 1.0 / w
        eta = w + np.log(z / (1.0 + w))
        # exp(sigma_debye) = e^(nu*eta) / (sqrt(2*pi*nu) * sqrt(w))
        sigma_d = nu * eta - 0.5 * np.log(2 * np.pi * nu * w)
        sigma = _sigma_j(nu, x)
        shift = np.exp(sigma_d - sigma)
        S = _olver_sum(_U, p, nu) * shift
        Sp = (w / z) * _olver_sum(_V, p, nu) * shift
    return Scaled(S, Sp, sigma, np.full(np.shape(x), 2e-12))


# --- the dispatchers --------------------------------------------------------------------------

def _pick(shape, parts):
    """Assemble one Scaled from (mask, Scaled-of-the-masked-cells) pairs.

    Each branch is handed ONLY its own cells, as a flat array, rather than being run over
    the whole block and masked afterwards. Running the series everywhere and discarding
    would overflow on the very cells it was not selected for -- and would do the work twice.
    """
    out = tuple(np.full(shape, np.nan) for _ in range(4))
    for mask, part in parts:
        for dest, src in zip(out, part):
            dest[mask] = src
    return Scaled(*out)


def _flatten(a, b):
    """Broadcast, then flatten. The dispatcher indexes with boolean masks throughout, and a
    0-d array cannot be indexed that way -- so a scalar call is not allowed to be a separate
    code path, it is just a block of one."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return a.shape, a.ravel(), b.ravel()


def _unflatten(scaled, shape):
    if shape == ():
        return Scaled(*(a.reshape(())[()] for a in scaled))
    return Scaled(*(a.reshape(shape) for a in scaled))


def _refine(outer, inner):
    """The subset of `outer` picked out by `inner`, which indexes `outer`'s own True cells."""
    out = outer.copy()
    out[outer] = inner
    return out


def log_j(nu, x):
    """`J_nu(x)` and `J_nu'(x)` in factored form. See the module docstring.

    `nu` and `x` broadcast against each other; the branch mask is taken over the resulting
    block, so each branch is one vectorized call. Taking it per column instead is a
    performance bug the test suite checks for.
    """
    shape, nu, x = _flatten(nu, x)

    # The largest series term is ~exp(x**2/(4*nu)); the (nu+1) keeps nu = 0 finite and errs
    # toward scipy, which is exact there anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_maxterm = x * x / (4.0 * (nu + 1.0))
    use_series = log_maxterm <= _LOG_MAXTERM

    parts = []
    if np.any(use_series):
        parts.append((use_series, _j_series(nu[use_series], x[use_series])))
    rest = ~use_series
    if np.any(rest):
        b = _j_scipy(nu[rest], x[rest])
        # Self-selecting rather than threshold-driven: branch C is exactly the cells where
        # scipy gave up. That is one fewer tuned constant, and it cannot drift out of date.
        gave_up = (~np.isfinite(b.S)) | ((b.S == 0.0) & (x[rest] > 0))
        parts.append((_refine(rest, ~gave_up), Scaled(*(a[~gave_up] for a in b))))
        if np.any(gave_up):
            fell = _refine(rest, gave_up)
            parts.append((fell, _j_debye(nu[fell], x[fell])))
    return _unflatten(_pick(nu.shape, parts), shape)


def _log_y_drop(m, x):
    """Log of the size of what `_y_series` throws away, relative to what it keeps."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (2 * m * np.log(x / 2) - gammaln(m + 1) - gammaln(np.maximum(m, 1.0))
                + np.log(np.pi))


def log_y(m, x):
    """`Y_m(x)` and `Y_m'(x)` in factored form. Integer order only."""
    shape, m, x = _flatten(m, x)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        blown = ~np.isfinite(_sigma_y(m, x)) | (_sigma_y(m, x) > 700.0)
        use_series = (m >= 1) & (x > 0) & ((_log_y_drop(m, x) <= _LOG_Y_DROP) | blown)

    parts = []
    if np.any(use_series):
        parts.append((use_series, _y_series(m[use_series], x[use_series])))
    rest = ~use_series
    if np.any(rest):
        parts.append((rest, _y_scipy(m[rest], x[rest])))
    return _unflatten(_pick(m.shape, parts), shape)
