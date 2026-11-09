"""What a corner's quadrature is sized against.  (2026-11-08)

Two ways the sizing claimed more than it delivered, found by comparing with a dense reference on
real eigenfunctions, every matrix entry rather than a trace (traces cancel on symmetric doubles):
a graded-rule model whose positive coefficients let errors cancel, and a 120-degree corner taken
for smooth because its leading exponent is an integer.
"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import jv

from chladni.eigenproblem import Eigenproblem
from chladni.geometry import Polygon
from chladni.integrals import cauchy, plan as P, rules as R
from chladni.mps.solver import INTEGRAL_CORNERS


def _true_error(nu, k, d, w):
    x, wx = leggauss(120)
    x, wx = 0.5*(x + 1), 0.5*wx
    edges = np.r_[0, np.geomspace(1e-7, 1, 40)]
    s = np.concatenate([a + (b - a)*x for a, b in zip(edges[:-1], edges[1:])])
    ws = np.concatenate([(b - a)*wx for a, b in zip(edges[:-1], edges[1:])])
    p = max(1.0, 1/(2*nu - 1))*3
    f = lambda r: (jv(nu, k*r)*nu/r)**2
    exact = np.sum(f(s**p)*p*s**(p - 1)*ws)
    return abs(np.sum(w*f(d)) - exact)/abs(exact)


@pytest.mark.parametrize("omega", [127.0, 135.0, 150.0, 240.0])
@pytest.mark.parametrize("k", [5.0, 15.0, 30.0])
def test_the_rule_the_sizing_picks_delivers_on_the_true_integrand(omega, k):
    nu = 180.0/omega
    (lv, o, n_top), ach = R.graded_order_for_prec(nu, 1e-11, k)
    d, w = R.graded_corner_rule(nu, lv, o, R.GRADED_SIGMA, n_top)
    assert ach <= 1e-11
    assert _true_error(nu, k, d, w) < 10*1e-11


def _kinds(z, order=30, prec=1e-11):
    dom = Polygon(np.asarray(z, complex))
    return [s.kind for s in P.corner_specs(dom, {(0, i): order for i in range(len(z))}, prec)]


def test_a_120_degree_corner_is_not_smooth():
    assert "smooth" not in _kinds(np.exp(2j*np.pi*np.arange(6)/6))


def test_a_right_angle_stays_smooth():
    assert set(_kinds([0, 1, 1 + 1j, 1j])) == {"smooth"}


def test_the_series_beyond_the_leading_term_is_what_decides():
    assert R.smooth_power_error(1.0, 30) < 1e-14 < R.smooth_power_error(1.0, 30, 1.5)


def test_the_panel_next_to_a_corner_is_sized_for_it():
    z = np.exp(2j*np.pi*np.arange(5)/5)
    dom = Polygon(z)
    specs = P.corner_specs(dom, {(0, i): 30 for i in range(5)}, 1e-11)
    near = P._near_singularities(specs, 0, 0, dom.outer.segments[0])
    assert np.any(np.abs(near) < 1e-12) and np.any(np.abs(near - 1) < 1e-12)


def test_a_singularity_on_the_panel_is_left_to_its_own_rule():
    size = lambda h: (20, 1e-14)
    assert P._smooth_panels(0.0, 1.0, np.array([0j]), size, 1e-11) == [(0.0, 1.0, 20, 1e-14)]


def _entrywise_error(z, n, level, prec=1e-10):
    z = np.asarray(z, complex)
    rec = Eigenproblem(Polygon(z)).solve(n, prec=prec)
    sp, sol = rec.spectrum, rec.solver
    basis, L = sol._pencil.basis, sp.levels[level]
    lam, c = float(L.lam), sol.eigenfunctions(L).coef
    iq = P.integral_quad(sol.domain, lam, prec=prec, family="integer", corners=INTEGRAL_CORNERS)
    q = iq.quad
    un = np.asarray(cauchy.cauchy_data(basis, c, lam, q).u_n).reshape(len(q.wts), -1)
    seg, tau, wts = np.asarray(q.seg_idx, int), np.asarray(q.tau), np.asarray(q.wts)
    g, gw = leggauss(40)
    e = 0.5*0.2**np.arange(26, -1, -1.0)
    e = np.r_[0.0, e[e < 0.5]]
    edges = np.unique(np.r_[e, np.linspace(0.2, 0.8, 13), 1 - e])
    worst = 0.0
    for s in range(len(z)):
        a, b = z[s], z[(s + 1) % len(z)]
        T = np.concatenate([x + (y - x)*(g + 1)/2 for x, y in zip(edges[:-1], edges[1:])])
        W = np.concatenate([(y - x)*gw/2 for x, y in zip(edges[:-1], edges[1:])])*abs(b - a)
        _, G = basis.at(a + T*(b - a)).values_and_grad(np.sqrt(lam))
        nrm = (b - a)/abs(b - a)*(-1j)
        U = (G.real*nrm.real + G.imag*nrm.imag) @ c
        m = seg == s
        for wt in (lambda t: 1 - t, lambda t: t):
            rule = (un[m].T*(wts[m]*wt(tau[m]))) @ un[m]
            dense = (U.T*(W*wt(T))) @ U
            worst = max(worst, float(np.abs(rule - dense).max())/lam)
    return worst


@pytest.mark.slow
def test_the_regular_hexagons_double_is_integrated_entry_by_entry():
    """7.5e-9 against a reported 8e-11 before: its identities cancel by symmetry, so only the
    entries show it."""
    assert _entrywise_error(np.exp(2j*np.pi*np.arange(6)/6), 8, 1) < 1e-11


@pytest.mark.slow
def test_a_127_degree_corner_meets_its_sizing_on_a_real_eigenfunction():
    a = np.deg2rad(127)/2
    assert _entrywise_error([0, 2*np.sin(a), np.sin(a) + 1j*np.cos(a)], 14, 4) < 1e-11
