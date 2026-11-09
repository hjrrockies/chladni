"""Corner quadrature rules for the boundary integrals: `chladni.integrals.rules`.

Ported from lappy's `tests/test_quad.py`, whose contracts were each measured there before they
were written down (`lappy/docs/corner_quadrature.tex`). The rules integrate a RAW integrand on
`[0, 1]`, anchored at the corner `tau = 0`: their weights carry the substitution's Jacobian with
the Jacobi weight divided out (`docs/design/geometry.md`, "What the weights integrate").

Every reference below is a closed-form moment, `int_0^1 t^e dt = 1/(e + 1)`. lappy found
`mpmath.quad` wrong by 4e-2 at 40 digits near the slit limit, so nothing here uses it.
"""
import numpy as np
import pytest

from chladni.geometry._adaptive import gauss_legendre
from chladni.integrals import rules as R

#: Corner angles as multiples of pi: re-entrant, the 3pi/2 L-shape corner among them.
CORNER_ALPHAS = [1.25, 4 / 3, 1.5, 1.6, 1.75, 1.9]

#: Angles an arc-arc corner produces: fixed by the circles, so `nu` is irrational and no
#: substitution rationalizes the exponent family.
IRRATIONAL_NUS = [0.657360, 0.730455, 0.587300]


# --- the substitution rule ----------------------------------------------------------------

@pytest.mark.parametrize("alpha", CORNER_ALPHAS)
@pytest.mark.parametrize("order", [8, 16])
def test_the_substitution_rule_is_exact_on_the_corner_family(alpha, order):
    """The defining property: exact for `tau^gamma * (polynomial of degree <= 2 order - 1 in
    tau^nu)`, so every `tau^(gamma + p nu)` integrates to `1/(gamma + p nu + 1)`."""
    nu = 1.0 / alpha
    gamma = 2 * nu - 2
    tau, w = R.corner_rule(order, nu)
    for p in range(2 * order):
        e = gamma + p * nu
        assert np.sum(w * tau**e) == pytest.approx(1.0 / (e + 1.0), rel=1e-13), p


@pytest.mark.parametrize("order", [4, 12, 40])
def test_at_nu_one_the_substitution_rule_IS_gauss_legendre(order):
    """`nu = 1` is a straight angle: `gamma = 0` and no substitution, so the rule collapses onto
    plain Gauss-Legendre -- to rounding, since the two come from different scipy/numpy paths."""
    tau, w = R.corner_rule(order, 1.0)
    u, wl = gauss_legendre(order)
    assert tau == pytest.approx(u, abs=1e-14)
    assert w == pytest.approx(wl, abs=1e-14)


@pytest.mark.parametrize("alpha", CORNER_ALPHAS)
def test_nodes_are_ascending_interior_and_weights_positive(alpha):
    tau, w = R.corner_rule(16, 1.0 / alpha)
    assert np.all(np.diff(tau) > 0)
    assert 0.0 < tau[0] and tau[-1] < 1.0
    assert np.all(w > 0.0)


@pytest.mark.parametrize("alpha, exact_on_one",
                         [(1.0, True), (1.5, True), (1.25, False), (4 / 3, False),
                          (1.6, False), (1.9, False)])
def test_the_weights_sum_to_one_only_when_two_over_nu_is_an_integer(alpha, exact_on_one):
    """`sum(w) == 1` says the rule is exact on constants, which needs `t^(2/nu - 2)` to be a
    polynomial: `2/nu` an integer. Among re-entrant angles only `3pi/2` qualifies. A property to
    RESPECT -- renormalizing would destroy exactness on the class the rule exists for, and
    `sum(w) - 1` is its cheapest self-diagnostic."""
    _, w = R.corner_rule(16, 1.0 / alpha)
    dev = abs(np.sum(w) - 1.0)
    if exact_on_one:
        assert dev < 1e-13
    else:
        assert 1e-9 < dev < 1e-2


@pytest.mark.parametrize("alpha", CORNER_ALPHAS)
def test_the_weight_sum_residual_shrinks_with_order(alpha):
    nu = 1.0 / alpha
    devs = [abs(np.sum(R.corner_rule(n, nu)[1]) - 1.0) for n in (8, 16, 32)]
    if devs[0] < 1e-13:
        assert max(devs) < 1e-13
    else:
        assert devs[2] < devs[1] < devs[0]


def test_a_slit_or_worse_is_refused_loudly():
    """`nu <= 1/2` puts the Jacobi exponent at or below -1: the integral itself diverges."""
    for nu in (0.5, 0.4, 0.0, -1.0):
        with pytest.raises(ValueError, match="nu must exceed"):
            R.corner_rule(16, nu)
    R.corner_rule(8, R.NU_MIN + 0.01)


def test_the_rules_are_cached_and_read_only():
    """Cached, so they must not be mutable: a caller scaling `w` in place would corrupt every
    later rule of the same order."""
    a = R.corner_rule(16, 2 / 3)
    assert a is R.corner_rule(16, 2 / 3)
    with pytest.raises(ValueError):
        a[1][0] = 0.0


# --- the 'integer' family: substitution 1/2, for Hadamard weights ---------------------------

@pytest.mark.parametrize("alpha", [1.25, 1.5, 1.75])
def test_substitution_one_half_is_exact_on_INTEGER_powers(alpha):
    """A Hadamard weight `V . n` that moves a corner is `O(r)`: integer powers of `r` on top of
    `r^gamma`. `sub = 1/2` maps every `r^m` to `t^(2m)`, so the rule is exact on them for ANY
    `nu` -- which `sub = nu` is not (it sends `r` to `t^(1/nu)`)."""
    nu = 1.0 / alpha
    gamma = 2 * nu - 2
    tau, w = R.corner_rule(16, nu, sub=0.5)
    tn, wn = R.corner_rule(16, nu)
    e = gamma + 1.0
    assert np.sum(w * tau**e) == pytest.approx(1.0 / (e + 1.0), rel=1e-13)
    assert abs(np.sum(wn * tn**e) * (e + 1.0) - 1.0) > 1e-9, "sub = nu should miss r^1"


# --- the order cap ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", CORNER_ALPHAS + [1.99])
def test_the_order_cap_keeps_the_innermost_node_off_the_corner(alpha):
    """Past the cap the innermost node falls below `NODE_FLOOR`, and a segment's
    parametrization rounds it onto the corner, where a basis's `1/r` terms are fatal."""
    nu = 1.0 / alpha
    cap = R.order_cap(nu)
    assert cap >= 8
    assert R.corner_rule(cap, nu)[0].min() > R.NODE_FLOOR
    if cap < 128:
        assert R.corner_rule(cap + 1, nu)[0].min() <= R.NODE_FLOOR


def test_the_cap_tightens_toward_the_slit_and_on_shorter_panels():
    caps = [R.order_cap(1.0 / a) for a in (1.5, 1.75, 1.9, 1.99)]
    assert caps == sorted(caps, reverse=True) and caps[-1] < caps[0]
    assert R.order_cap(1 / 1.9, scale=0.1) < R.order_cap(1 / 1.9, scale=1.0)


def test_nu_must_be_EXACT():
    """`nu` comes from the geometry, `pi / omega`, and is never rounded or padded: a rule built
    for 0.667 instead of 2/3 -- a relative error of 5e-4 -- loses ten digits on the true 3pi/2
    integrand (measured 2.0e-15 against 7.9e-5 at order 16)."""
    nu = 2 / 3
    gamma = 2 * nu - 2

    def f(t):
        return t**gamma * (1.0 - 0.7 * t**nu + 0.35 * t**(2 * nu)) * (1.0 + t**2 / 8)

    exact = sum(c / (gamma + e + 1.0)
                for c, e in [(1.0, 0.0), (-0.7, nu), (0.35, 2 * nu),
                             (0.125, 2.0), (-0.0875, nu + 2), (0.04375, 2 * nu + 2)])
    err = {n: abs(np.sum(R.corner_rule(16, n)[1] * f(R.corner_rule(16, n)[0])) / exact - 1)
           for n in (nu, 0.667)}
    assert err[nu] < 1e-14 and err[0.667] > 1e-6, err


def test_the_substitution_rule_resolves_the_real_270_integrand():
    """The integrand both rules face in production at a 3pi/2 corner: `tau^gamma` times a series
    in `tau^nu` and `tau^2`. Machine precision at order 16."""
    nu = 2 / 3
    gamma = 2 * nu - 2

    def f(t):
        return t**gamma * (1.0 - 0.7 * t**nu + 0.35 * t**(2 * nu)) * (1.0 + t**2 / 8)

    exact = sum(c / (gamma + e + 1.0)
                for c, e in [(1.0, 0.0), (-0.7, nu), (0.35, 2 * nu),
                             (0.125, 2.0), (-0.0875, nu + 2), (0.04375, 2 * nu + 2)])
    tau, w = R.corner_rule(16, nu)
    assert abs(np.sum(w * f(tau)) / exact - 1) < 1e-13
    u, wl = gauss_legendre(64)
    assert abs(np.sum(wl * f(u)) / exact - 1) > 1e-6, "and plain Gauss cannot"


# --- the interpolatory rule, for irrational nu -------------------------------------------------

def _curved_corner_integrand(nu):
    """`r^gamma` times a series in the CURVED family `{j nu + m}`, odd integer powers included."""
    gamma = 2 * nu - 2
    terms = [(1.0, 0, 0), (-0.7, 1, 0), (0.35, 2, 0), (0.5, 0, 1),
             (-0.2, 1, 1), (0.1, 0, 2), (-0.05, 1, 2)]

    def f(t):
        return t**gamma * sum(c * t**(k * nu + m) for c, k, m in terms)

    return f, sum(c / (gamma + k * nu + m + 1.0) for c, k, m in terms)


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_the_exponent_set_is_sorted_integrable_and_starts_at_gamma(nu):
    gamma = 2 * nu - 2
    E = R.corner_exponents(nu, gamma, 12)
    assert len(E) == 12 and np.all(np.diff(E) > 0)
    assert E[0] == pytest.approx(gamma) and np.all(E > -1.0)


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
@pytest.mark.parametrize("order", [12, 16, 24])
def test_the_interpolatory_rule_is_exact_on_its_exponent_set(nu, order):
    gamma = 2 * nu - 2
    tau, w = R.corner_interp_rule(order, nu)
    for e in R.corner_exponents(nu, gamma, max(2, order // 2)):
        assert np.sum(w * tau**e) == pytest.approx(1.0 / (e + 1.0), rel=1e-11), e


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
@pytest.mark.parametrize("order", [8, 12, 16, 24, 32])
def test_the_interpolatory_weights_stay_well_conditioned(nu, order):
    """`sum|w|` multiplies every rounding error in the integrand. The square solve is exact but
    drives it to 1e3-1e4; `n_exp = order // 2` keeps it near 1."""
    assert np.abs(R.corner_interp_rule(order, nu)[1]).sum() < 1.5


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_the_interpolatory_rule_beats_the_substitution_at_irrational_nu(nu):
    f, exact = _curved_corner_integrand(nu)
    ti, wi = R.corner_interp_rule(24, nu)
    tj, wj = R.corner_rule(24, nu)
    err_i = abs(np.sum(wi * f(ti)) / exact - 1)
    err_j = abs(np.sum(wj * f(tj)) / exact - 1)
    assert err_i < 1e-11 and err_i < err_j / 1e3, (err_i, err_j)


@pytest.mark.parametrize("nu", IRRATIONAL_NUS)
def test_the_interpolatory_rule_converges_with_order(nu):
    f, exact = _curved_corner_integrand(nu)
    errs = [abs(np.sum(w * f(t)) / exact - 1)
            for t, w in (R.corner_interp_rule(o, nu) for o in (8, 16, 24))]
    assert errs[1] < errs[0] / 1e3 and errs[2] < errs[1] / 1e2, errs


def test_the_square_interpolatory_solve_is_refused():
    with pytest.raises(ValueError, match="must be <"):
        R.corner_interp_rule(16, 0.65736, n_exp=16)


def test_the_exponent_set_respects_the_edge_type():
    """A straight edge admits only EVEN integer powers (`r . N` constant, `r` is arclength); a
    curved edge admits odd ones too. Scoring a straight-edge rule against the curved set makes it
    look ~7 orders worse than it is."""
    nu, gamma = 2 / 3, 2 * (2 / 3) - 2
    Es = R.corner_exponents(nu, gamma, 10, curved=False)
    Ec = R.corner_exponents(nu, gamma, 10, curved=True)
    allowed = np.array([gamma + j * nu + 2 * q for j in range(14) for q in range(14)])
    assert all(np.any(np.isclose(allowed, e, atol=1e-12)) for e in Es)
    assert np.all(Ec <= Es + 1e-12) and np.any(Ec < Es - 1e-12)
    assert np.any(np.isclose(Ec, gamma + 1.0)) and not np.any(np.isclose(Es, gamma + 1.0))


# --- which rule, and how many nodes ---------------------------------------------------------------

def test_the_substitution_rule_is_chosen_only_where_it_is_exact():
    """The plain substitution wins only on a STRAIGHT edge with `2/nu` an integer -- among
    re-entrant angles, only `3pi/2`; everywhere else nodes from `sub = nu` and exactness from the
    interpolatory weights. Never `sub = 1/q`, which crushes the innermost node to 1e-29."""
    assert R.corner_rule_kind(2 / 3, curved=False) == ("substitution", pytest.approx(2 / 3))
    for alpha in (1.25, 1.5, 1.6, 1.75, 11 / 6):
        nu = 1.0 / alpha
        kind, sub = R.corner_rule_kind(nu, curved=True)
        assert kind == "interpolatory" and sub == pytest.approx(nu)
        assert R.corner_interp_rule(24, nu)[0].min() > R.NODE_FLOOR


def test_a_straight_270_corner_reaches_machine_precision_at_low_order():
    kind, sub = R.corner_rule_kind(2 / 3, curved=False)
    order, achieved = R.corner_order_for_prec(kind, 2 / 3, sub=sub, curved=False, prec=1e-14)
    assert order <= 12 and achieved <= 1e-14


#: Measured achievable model error per angle, straight and curved (lappy `test_quad.py`). The
#: capability degrades toward the slit, so a flat target would be a wish, not a test.
ACHIEVABLE = [(1.25, 1e-14, 1e-13), (4 / 3, 1e-14, 1e-14), (1.5, 1e-14, 1e-13),
              (1.521236, 1e-13, 1e-10), (1.6, 1e-12, 1e-10),
              (1.75, 1e-12, 1e-9), (1.9, 1e-10, 1e-8)]


@pytest.mark.parametrize("alpha, bar_straight, bar_curved", ACHIEVABLE)
@pytest.mark.parametrize("curved", [False, True])
def test_the_order_search_reaches_its_measured_capability(alpha, bar_straight, bar_curved,
                                                          curved):
    nu = 1.0 / alpha
    bar = bar_curved if curved else bar_straight
    kind, sub = R.corner_rule_kind(nu, curved=curved)
    order, achieved = R.corner_order_for_prec(kind, nu, sub=sub, curved=curved, prec=bar)
    assert achieved <= bar and order <= 64, (kind, order, achieved)


def test_a_shortfall_near_the_slit_is_RETURNED_not_hidden():
    nu = 1.0 / 1.9
    kind, sub = R.corner_rule_kind(nu, curved=True)
    order, achieved = R.corner_order_for_prec(kind, nu, sub=sub, curved=True, prec=1e-14)
    assert achieved > 1e-14 and order >= 4


@pytest.mark.parametrize("alpha", [1.25, 1.5, 1.75])
def test_the_order_grows_as_prec_tightens(alpha):
    nu = 1.0 / alpha
    kind, sub = R.corner_rule_kind(nu, curved=True)
    orders = [R.corner_order_for_prec(kind, nu, sub=sub, curved=True, prec=p)[0]
              for p in (1e-4, 1e-8, 1e-10)]
    assert orders == sorted(orders)


def test_the_order_search_takes_the_ARGMIN_not_the_last_order():
    """Accuracy is not monotone in order: past a `nu`-dependent point, rounding in the
    `tau^(2 nu - 2)` dynamic range wins. Measured: the substitution rule on a straight edge at
    `nu = 1/1.9` is best at order 32 (1.2e-14) and worse by 40x at order 64. An unreachable target
    must return the BEST order seen, not the last one scanned."""
    nu = 1.0 / 1.9
    order, achieved = R.corner_order_for_prec("substitution", nu, curved=False, prec=1e-16)
    top = min(R.order_cap(nu), 64)                       # the search's own order_max
    errs = {o: R.corner_model_error("substitution", o, nu, curved=False)
            for o in range(4, top + 1, 2)}
    assert errs[top] > 10 * min(errs.values()), "the premise: accuracy turns over before 64"
    assert achieved == min(errs.values()) and order == min(errs, key=errs.get) < top


# --- does a corner need the corner rule at all? ----------------------------------------------

@pytest.mark.parametrize("gamma, needs", [(1 / 3, True), (2 / 3, True), (1.0, False),
                                          (11.55, False)])
def test_whether_plain_gauss_suffices_is_asked_directly(gamma, needs):
    """Singular is not re-entrant. `gamma = 2 nu - 2` at a 135-degree corner (`nu = 4/3`) is 2/3,
    and Gauss-Legendre converges on `r^(2/3)` only algebraically; an integer `gamma` is a
    polynomial; a large fractional one is smooth enough already."""
    err = R.smooth_power_error(gamma, 64)
    assert (err > 1e-12) == needs, err


def test_a_negative_gamma_is_never_smooth():
    assert R.smooth_power_error(-1 / 3, 64) == float("inf")


@pytest.mark.parametrize("k", [1.0, 10.0, 50.0, 100.0])
@pytest.mark.parametrize("prec", [1e-8, 1e-12])
def test_the_smooth_order_meets_its_target(k, prec):
    order, achieved = R.smooth_order_for_prec(k, prec)
    tau, w = gauss_legendre(order)
    exact = (np.exp(1j * k) - 1.0) / (1j * k)
    assert achieved <= prec and abs(np.sum(w * np.exp(1j * k * tau)) - exact) <= prec


def test_the_smooth_order_scales_with_wavenumber_and_not_per_wavelength():
    orders = [R.smooth_order_for_prec(k, 1e-12)[0] for k in (10, 50, 100, 200)]
    assert orders == sorted(orders) and orders[-1] < 400


def test_the_rule_caches_are_bounded():
    """Corner rules are keyed by the corner's exponent `nu`, which moves on every shape-optimizer
    step, so an unbounded cache grows for the life of a douse run (2026-10-13: ~0.75 MB per
    solve). Bounded, with room for several solves' worth, so within-solve reuse is kept."""
    from chladni.integrals import rules
    cached = [f for f in vars(rules).values() if hasattr(f, "cache_info")]
    assert cached
    for f in cached:
        assert f.cache_info().maxsize is not None, f.__name__
        if f.__module__ == rules.__name__:          # `gauss_legendre` is keyed by order alone
            assert f.cache_info().maxsize >= 1024, f.__name__


# --- the graded rule (2026-11-03, `notes/hadamard_accuracy.md`, addenda) -------------------
#
# `tau = s^p`, `p = max(1, 1/(2 nu - 1))`, geometric Gauss-Legendre panels in `s`. No fit, so
# no stall: it reaches the floor on the whole CURVED lattice `{gamma + j nu + m}`, which contains
# both the Gram and the Hadamard integrands. Rationality of `nu` buys and costs nothing.

GRADED_NUS = [4 / 7, 2 / 3, 0.7, 71 / 101, 1.04, 1.39, 2.59, 11.8]


def _lattice_error(nu, d, w, rho=2.0):
    """Worst-case-weighted relative error on `sum rho^-(j nu + m) d^(gamma + j nu + m)`: the
    corner panel at half the expansion's radius, every lattice point present."""
    g = 2 * nu - 2
    E = np.array([g + j * nu + m for j in range(80) for m in range(80)
                  if rho ** -(j * nu + m) > 1e-20])
    C = rho ** -(E - g)
    exact = np.sum(C / (E + 1))
    return abs(np.sum(C * (w[None, :] * d[None, :] ** E[:, None]).sum(axis=1)) - exact) / exact


@pytest.mark.parametrize("nu", GRADED_NUS)
def test_the_graded_rule_reaches_the_floor_on_the_whole_lattice(nu):
    """The rule the sizing picks at `k = 1`, checked on the prototype's independent lattice --
    harsher than the sizing model, since it decays only geometrically in `m` as well. Measured
    worst 2.3e-14 (nu = 11.8), the summation floor of a lattice with ~1e3 terms."""
    (lv, o, n_top), _ = R.graded_order_for_prec(nu, 1e-14, 1.0)
    d, w = R.graded_corner_rule(nu, lv, o, R.GRADED_SIGMA, n_top)
    assert _lattice_error(nu, d, w) < 5e-14
    assert np.all(w > 0) and np.all(np.diff(d) > 0) and d[0] > 0 and d[-1] < 1


@pytest.mark.parametrize("rational, neighbour", [(2 / 3, 2 / 3 + 1e-3 * np.sqrt(2)),
                                                 (0.7, 71 / 101), (4 / 7, 53 / 93)])
def test_a_rational_nu_is_nothing_special(rational, neighbour):
    """The study's worry about rationalizing substitutions: `p` is continuous in `nu`, so a
    small-denominator `nu` and a large-denominator neighbour get the same accuracy."""
    for lv, o, n_top in ((4, 10, 2), (4, 10, 8)):
        a = _lattice_error(rational, *R.graded_corner_rule(rational, lv, o, R.GRADED_SIGMA, n_top))
        b = _lattice_error(neighbour, *R.graded_corner_rule(neighbour, lv, o, R.GRADED_SIGMA,
                                                            n_top))
        assert max(a, b) / max(min(a, b), 1e-16) < 10


def test_the_graded_power_flattens_the_leading_term():
    for nu in (0.55, 2 / 3, 0.9):
        p = R.graded_power(nu)
        # tau^gamma dtau = s^(p gamma) p s^(p - 1) ds = p ds
        assert p * (2 * nu - 2) + p - 1 == pytest.approx(0.0, abs=1e-14)
    assert R.graded_power(1.3) == 1.0
    with pytest.raises(ValueError):
        R.graded_power(0.5)


@pytest.mark.parametrize("nu", GRADED_NUS + [0.53])
@pytest.mark.parametrize("k", [1.0, 10.0, 40.0])
def test_the_graded_sizing_meets_1e14_everywhere_short_of_the_slit(nu, k):
    """Where the interpolatory search stalls at 5e-8..9e-6 on this lattice. Node counts as
    measured 2026-11-03: 32 (convex, k = 1) to 600 (nu = 0.53, k = 40) on macOS; Linux CI picks
    672 there, because 1e-14 is near the model's rounding floor (the 600-node rule models
    5.6e-15) and which candidate clears it is platform-dependent. The cap bounds cost; it is not
    a node count to reproduce."""
    (lv, o, n_top), ach = R.graded_order_for_prec(nu, 1e-14, k)
    # Beyond what the ladder reaches, and past the thesis box (re-entrant at most 300 degrees, nu
    # above 0.6). The sizing reports them now; the positive-coefficient model used to claim 1e-14.
    assert ach <= 1e-14 or (k, nu) in {(40.0, 0.53), (40.0, 4 / 7), (10.0, 11.8)}
    assert (lv + n_top) * o <= 800


def test_the_outermost_split_is_what_buys_high_wavenumbers():
    one = R.graded_model_error(4 / 7, 14, 20, R.GRADED_SIGMA, 10.0, 1)
    split = R.graded_model_error(4 / 7, 6, 12, R.GRADED_SIGMA, 10.0, 8)
    assert one > 1e-10 and split < 1e-14


def test_a_graded_shortfall_is_RETURNED_not_hidden():
    """At `nu = 0.5005`, `p = 1000` and the nodes underflow: the search reports a miss (here
    infinite) rather than a rule."""
    (lv, o, n_top), ach = R.graded_order_for_prec(0.5005, 1e-14, 40.0)
    assert not ach <= 1e-14
