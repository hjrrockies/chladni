"""Corner quadrature rules on `[0, 1]`, anchored at the corner `tau = 0`.

Ported from lappy's `quad.py`, whose design was measured before it was kept
(`lappy/docs/corner_quadrature.tex`, `corner_integrand_dirichlet.tex`). Every rule here integrates
a RAW integrand: the weights carry the substitution's Jacobian with the Jacobi weight divided
out, so `sum(w * f(tau))` approximates `int_0^1 f(tau) dtau` for every `f` in the rule's exact
class (`docs/design/geometry.md`, "What the weights integrate").

# The integrand at a corner

At a corner of interior angle `omega`, `nu = pi / omega`, a Dirichlet eigenfunction is
`sum_l c_l J_(l nu)(k r) sin(l nu theta)`, so on an edge leaving the corner
`d_n u = r^(nu - 1) F(r)` and `(d_n u)^2 = r^gamma G(r)` with `gamma = 2 nu - 2`. Which powers `G`
carries depends on the edge:

* STRAIGHT edge: `{gamma + j nu + 2q}` -- the corner family and the Bessel factors' even powers;
  `r . N` is constant and `r` is arclength, so no odd powers appear.
* CURVED edge: `{gamma + j nu + m}`, every integer `m` -- `r . N` and `r` are series in arclength.

Scoring a straight-edge rule against the curved set made it read 1e-8 where it achieves 2.4e-15.

# The two rules, and why not a third

The SUBSTITUTION rule takes `r = t^(1/sub)` and Gauss-Jacobi in `t`. With `sub = nu` it is exact
on `tau^gamma` times a polynomial in `tau^nu`, which covers the straight family only when `2/nu`
is an integer -- among re-entrant angles, only `3pi/2`. With `sub = 1/2` it is exact on integer
powers of `r`, which is what a Hadamard weight `V . n = O(r)` at a moving corner supplies; that
is the `'integer'` family, and it is NOT interchangeable with `sub = nu` in either direction.

The INTERPOLATORY rule keeps the `sub = nu` nodes and refits the weights by least squares on the
corner's actual exponent set, `n_exp = order // 2` of them. That gets exactness wherever `nu` is,
irrational included, without moving the nodes.

The rationalizing substitution `sub = 1/q` for `nu = p/q` would be exact as an abstract rule, and
is NOT used: `tau = t^q` crushes the innermost node to 1.1e-10 at `5pi/4` and 1.6e-29 at `11pi/6`,
where a segment's parametrization rounds it onto the corner and the basis's `1/r` terms are NaN.

# Sizing

Every moment is closed-form, `int_0^1 t^e dt = 1/(e + 1)`, so a rule's error on a representative
member of its class is computable directly: `corner_model_error`. `corner_order_for_prec` SCANS it
and returns the argmin when the target is out of reach -- accuracy is not monotone in order, since
past a `nu`-dependent point rounding in the `tau^(2 nu - 2)` dynamic range wins. The sizing is a
model, not a certificate; the certificate is refinement, one layer up.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

#: Entries per rule cache. Corner rules are keyed by the corner's exponent `nu`, which moves on
#: every step of a shape optimizer, so an unbounded cache grows for the life of a douse run
#: (~0.75 MB per solve, 2026-10-13). One solve adds at most ~600 entries to the largest of them,
#: so this keeps several solves' worth of within-solve reuse at well under a few MB per cache.
_RULE_CACHE = 4096

import numpy as np
from scipy.special import roots_jacobi

from ..geometry._adaptive import gauss_legendre

__all__ = ["NU_MIN", "NODE_FLOOR", "gauss_jacobi", "corner_rule", "corner_exponents",
           "corner_interp_rule", "corner_rule_kind", "corner_model_error", "order_cap",
           "corner_order_for_prec", "smooth_power_error", "smooth_order_for_prec"]

#: The slit limit. `gamma = 2 nu - 2 > -1` needs `nu > 1/2`; at `nu = 1/2` the integral diverges.
NU_MIN = 0.5

#: The smallest `tau` a node may take on a unit panel. Below it a segment's `(1 - tau) p0 + tau p1`
#: rounds onto the corner itself, where a basis's `1/r` terms are fatal. About 1e6 x float64
#: epsilon; lappy's `_KRESS_TAU_FLOOR`, kept for the same reason.
NODE_FLOOR = 1e-9


def _frozen(*arrays):
    """Cached rules are shared, so they are returned read-only: a caller scaling `w` in place
    would otherwise corrupt every later rule of the same order."""
    for a in arrays:
        a.setflags(write=False)
    return arrays


@lru_cache(maxsize=_RULE_CACHE)
def gauss_jacobi(order, a=0.0, b=0.0):
    """`(tau, w)` on `[0, 1]` integrating a RAW integrand that behaves like `tau^a` at 0 and
    `(1 - tau)^b` at 1: Gauss-Jacobi nodes, with the Jacobi weight divided back out of the
    weights. Exact on `tau^a (1 - tau)^b` times a polynomial of degree `<= 2 order - 1`."""
    x, w = roots_jacobi(int(order), float(b), float(a))      # scipy's (alpha, beta) are (1-x, 1+x)
    w = w / ((1 + x) ** a * (1 - x) ** b)
    return _frozen((x + 1) / 2, w / 2)


@lru_cache(maxsize=_RULE_CACHE)
def corner_rule(order, nu, gamma=None, sub=None):
    """The SUBSTITUTION rule. `(tau, w)` ascending on `[0, 1]`, exact on `tau^gamma` times a
    polynomial of degree `<= 2 order - 1` in `tau^sub`.

    `gamma` defaults to `2 nu - 2`, the exponent of the Rellich and Gram kernels at a Dirichlet
    corner; `sub` defaults to `nu`. With `tau = t^(1/sub)` the factor `tau^gamma` and the
    Jacobian combine into a Jacobi exponent `beta = (gamma + 1)/sub - 1` at `t = 0`; `gamma`
    then survives only in the nodes, and the weight is the Jacobian alone.

    Two properties that must not be "fixed". `sum(w) == 1` only when constants lie in the exact
    class; renormalizing would destroy exactness on the class the rule exists for. And the
    substitution crowds nodes toward the corner as the order grows -- cap the order
    (`order_cap`), never clamp `tau`.
    """
    nu = float(nu)
    if not nu > NU_MIN:
        raise ValueError(f"nu must exceed {NU_MIN} (got {nu}): the integral diverges at the slit "
                         f"nu = 1/2, and no rule can help")
    gamma = 2.0 * nu - 2.0 if gamma is None else float(gamma)
    sub = nu if sub is None else float(sub)
    if not sub > 0:
        raise ValueError(f"the substitution exponent must be positive, got {sub}")
    beta = (gamma + 1.0) / sub - 1.0
    if not beta > -1.0:
        raise ValueError(f"gamma={gamma} with sub={sub} gives a Jacobi exponent {beta} <= -1: "
                         f"the integral does not converge")
    t, W = gauss_jacobi(int(order), beta, 0.0)
    return _frozen(t ** (1.0 / sub), (W / sub) * t ** (1.0 / sub - 1.0))


def corner_exponents(nu, gamma, n, curved=True):
    """The first `n` exponents of the integrand's family at a corner, ascending:
    `{gamma + j nu + m}` with `m` any integer on a curved edge, EVEN `m` only on a straight one.
    For irrational `nu` no two `(j, m)` coincide, so there are no log terms."""
    step = 1 if curved else 2
    E = sorted({gamma + j * nu + m for j in range(n + 2) for m in range(0, n + 2, step)})
    return np.array(E[:n])


@lru_cache(maxsize=_RULE_CACHE)
def corner_interp_rule(order, nu, gamma=None, n_exp=None, curved=True):
    """The INTERPOLATORY rule: the substitution rule's `sub = nu` nodes, with weights refit so
    that `sum_i w_i tau_i^e = 1/(e + 1)` on the first `n_exp` exponents of the corner's family.

    `n_exp` defaults to `order // 2` and must stay BELOW `order`. The square solve is exact but
    reaches condition 1e19 at order 16, with weights near -1e4 whose total variation multiplies
    every rounding error in the integrand; the rectangular one is a minimum-norm least-squares
    solve that keeps `sum|w|` near 1 and the exactness (lappy, measured at `nu = 0.65736`: order
    16 square 3.0e-13, order 16 with 10 exponents 2.0e-15).
    """
    nu = float(nu)
    gamma = 2.0 * nu - 2.0 if gamma is None else float(gamma)
    n_exp = max(2, int(order) // 2) if n_exp is None else int(n_exp)
    if n_exp >= order:
        raise ValueError(f"n_exp ({n_exp}) must be < order ({order}): the square solve is "
                         f"exact but ill-conditioned, and inflates sum|w| by orders of magnitude")
    tau, _ = corner_rule(int(order), nu, gamma)
    E = corner_exponents(nu, gamma, n_exp, curved)
    V = tau[None, :] ** E[:, None]
    w, *_ = np.linalg.lstsq(V, 1.0 / (E + 1.0), rcond=None)
    return _frozen(tau.copy(), w)


def corner_rule_kind(nu, curved=True, rtol=1e-12):
    """`(kind, sub)`: which rule a corner takes, and the node-placement exponent (always `nu`).

    `'substitution'` only on a STRAIGHT edge where `2/nu` is an integer -- order 8 there where the
    interpolatory rule needs 24. `'interpolatory'` everywhere else.
    """
    nu = float(nu)
    if not curved:
        q = 2.0 / nu
        if abs(q - round(q)) <= rtol * q:
            return "substitution", nu
    return "interpolatory", nu


def _rule(kind, order, nu, gamma, sub, curved):
    if kind == "substitution":
        return corner_rule(order, nu, gamma, sub)
    if kind == "interpolatory":
        return corner_interp_rule(order, nu, gamma, None, curved)
    raise ValueError(f"unknown corner rule {kind!r}; expected 'substitution' or 'interpolatory'")


@lru_cache(maxsize=_RULE_CACHE)
def corner_model_error(kind, order, nu, gamma=None, sub=None, curved=True, k=1.0, n_j=6, n_m=8):
    """Relative error of a corner rule on a REPRESENTATIVE member of its class, in closed form.

    The model mirrors the real integrand: `s^gamma (sum_j a_j s^(2j))^2 (sum_m b_m s^m)`, where
    `a_j = (k/2)^(2j) / (j!)^2` are the Bessel factors' coefficients and `b_m = (k/2)^m / m!` an
    analytic weight, `m` even on a straight edge. `k = sqrt(lam_max) * panel length`. The factorial
    decay is the point: an unweighted max over monomials over-weights high powers that carry
    nothing, and lands its argmin in the wrong place (order 40 where the true error keeps falling
    to 64).
    """
    nu = float(nu)
    gamma = 2.0 * nu - 2.0 if gamma is None else float(gamma)
    tau, w = _rule(kind, int(order), nu, gamma, sub, curved)
    a = np.array([(0.5 * k) ** (2 * q) / factorial(q) ** 2 for q in range(n_j)])
    a_exp = np.array([2.0 * q for q in range(n_j)])
    ms = range(0, n_m, 1 if curved else 2)
    b = np.array([(0.5 * k) ** m / factorial(m) for m in ms])
    b_exp = np.array([float(m) for m in ms])
    terms = {}
    for a1, e1 in zip(a, a_exp):
        for a2, e2 in zip(a, a_exp):
            for bm, em in zip(b, b_exp):
                e = gamma + e1 + e2 + em
                terms[e] = terms.get(e, 0.0) + a1 * a2 * bm
    E = np.array(sorted(terms))
    C = np.array([terms[e] for e in E])
    exact = float(np.sum(C / (E + 1.0)))
    approx = float(np.sum(C * (w[None, :] * tau[None, :] ** E[:, None]).sum(axis=1)))
    return abs(approx / exact - 1.0)


@lru_cache(maxsize=_RULE_CACHE)
def order_cap(nu, gamma=None, sub=None, scale=1.0, order_max=128):
    """The largest order whose innermost node stays above `NODE_FLOOR / scale`, `scale` the
    panel's tau-length. Necessary, NOT sufficient: accuracy turns over well before the cap (at
    `nu = 0.526` the best order is 16, 1.2e-11, against a cap of 75 that gives 4.2e-9)."""
    cap = 0
    for order in range(2, int(order_max) + 1):
        if corner_rule(order, float(nu), gamma, sub)[0].min() * scale <= NODE_FLOOR:
            break
        cap = order
    return cap


@lru_cache(maxsize=_RULE_CACHE)
def corner_order_for_prec(kind, nu, gamma=None, sub=None, curved=True, prec=1e-14, scale=1.0,
                          k=1.0, order_min=4, order_max=64, step=2):
    """`(order, achieved)`: the smallest order whose `corner_model_error` meets `prec`, under
    `order_cap`. When none does, the ARGMIN and what it achieves -- so the caller can report the
    shortfall instead of silently missing the target."""
    cap = order_cap(nu, gamma, sub, scale=scale, order_max=order_max)
    best = (np.inf, order_min)
    for order in range(order_min, max(order_min, min(cap, order_max)) + 1, step):
        try:
            err = corner_model_error(kind, order, nu, gamma, sub, curved, k)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if err <= prec:
            return order, err
        if err < best[0]:
            best = (err, order)
    return best[1], best[0]


@lru_cache(maxsize=_RULE_CACHE)
def smooth_power_error(gamma, order):
    """Relative error of `order`-point Gauss-Legendre on `int_0^1 tau^gamma dtau`: does a corner
    need the corner rule at all? An integer `gamma` is a polynomial; a large fractional one is
    smooth enough; a small fractional one -- a corner near 135 degrees -- is not, whether or not
    the corner is re-entrant. Infinite for `gamma < 0`, which plain Gauss cannot integrate."""
    if gamma <= 0:
        return float("inf") if gamma < 0 else 0.0
    tau, w = gauss_legendre(int(order))
    exact = 1.0 / (gamma + 1.0)
    return float(abs(np.sum(w * tau**gamma) - exact) / exact)


@lru_cache(maxsize=_RULE_CACHE)
def smooth_order_for_prec(k, prec=1e-14, order_min=4, order_max=512, step=2):
    """`(order, achieved)`: the smallest Gauss-Legendre order integrating `exp(i k tau)` on
    `[0, 1]` to `prec`, ABSOLUTE against the unit measure. `k` is the integrand's wavenumber over
    the panel -- `2 sqrt(lam_max) L` for a product of two eigenfunctions, which oscillates at twice
    either's wavenumber. Absolute because the integral is ~2/k: a relative target would demand
    roundoff-level absolute error at large `k` and scan to `order_max` chasing it."""
    if k <= 0:
        return order_min, 0.0
    exact = (np.exp(1j * k) - 1.0) / (1j * k)
    err = np.inf
    for order in range(order_min, order_max + 1, step):
        tau, w = gauss_legendre(order)
        err = float(abs(np.sum(w * np.exp(1j * k * tau)) - exact))
        if err <= prec:
            return order, err
    return order_max, err
