"""Multiplicity at a NOISY or NEAR-FLOOR root: two defects found off the rectangle grids.

Stage 3's reach rule reads each `sigma_j / c` as a distance to the `j`-th eigenvalue. Both readings
fail when the numbers are floor noise rather than cone values, and both failures were silent:

* **An exact double at the floor** (`notes/double_10pi2.py`). The L-shape's `10 pi^2` on its
  80-term recipe: `sigma_2` at the root is ITS floor, 8.8e-13, not a distance, and its `d_est`
  lands at the bracket edge. The rule declines; the side search, built to locate a DISTINCT
  partner, finds none -- the partner is the root itself -- and the level came back simple.
* **A slope read inside a noisy floor** (`notes/spike_x8.py`). douse's spike triangle (catalogue
  T1.3), where `sigma_1` floors at 1.1e-8 with ~2x noise: the cache slope read 47.7 against a true
  3.4e-3, every branch up to `MAX_BRANCHES` read as inside the bracket, and a simple level came
  back x8.

The fix restores (C2), the definition `multiplicity.py` states -- a branch belongs to a level only
if `sigma_j` has a local minimum there -- read on the cached wide bracket, where the cone has
risen above the floor. FEM and two other bases agree that every T1.3 level below is simple.
"""
import numpy as np
import pytest

from chladni.geometry import L_shape, Polygon
from chladni.helmholtz import FourierBessel, Fundamental
from chladni.mps import MPSEigensolver

from hard_roster import recipe_basis

pytestmark = [pytest.mark.slow,
              pytest.mark.filterwarnings("ignore::chladni.helmholtz.PrecisionWarning")]

TEN_PI2 = 10 * np.pi ** 2


def _near(res, lam, rel=5e-10):
    return [L for L in res.levels if abs(L.lam - lam) <= rel * lam]


@pytest.fixture(scope="module")
def l_solver():
    dom = L_shape()
    b = recipe_basis(dom, 80)
    n = len(b)
    return MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=1e-12)


@pytest.mark.parametrize("window", [(5.0, 100.0), (5.0, 110.0), (90.0, 105.0), (95.0, 100.0)])
def test_an_exact_double_at_the_floor_is_counted_twice(l_solver, window):
    """`10 pi^2` is a true double (`L_shape_eigs` lists it twice). Measured before the fix: count
    1, 1, 2, 2 over these windows, never flagged. It may come back as one level of multiplicity
    2 or as two simple levels -- merging is permitted, not required -- but the total is 2."""
    res = l_solver.search_interval(*window)
    near = _near(res, TEN_PI2)
    assert sum(L.mult for L in near) == 2, [(L.lam - TEN_PI2, L.mult) for L in near]


# douse's spike triangle, T1.3: `start_shape(3, 4)` at perimeter 4 (benchmarks/douse/cases.json).
SPIKE = np.array([1.972069685870855 + 0j, 1.1756103887576692 + 0.23218091730644705j, 0j])
SPIKE_WINDOW = (124.32878268649651, 966.949607249548)
#: Its spectrum in the window, all simple: three chladni bases and P2 FEM agree on the count
#: (`benchmarks/douse/references.json`); the values are good to ~1e-8.
SPIKE_LEVELS = [248.6534194, 344.5901718, 430.3800057, 518.6158009, 611.3374157, 705.7496637,
                807.0076093, 887.4724418, 916.0815880]


def _spike_basis(dom, scale, sources):
    """`benchmarks/douse/make_references.basis_for`, runs "A" (1.5, no sources) and "B" (1.2,
    with): Fourier-Bessel at every corner, sized as lappy's planner sizes it."""
    k = np.sqrt(SPIKE_WINDOW[1])
    zs = np.array([c.z for c in dom.corners])
    b = None
    for c in dom.corners:
        r = float(np.max(np.abs(zs - c.z)))
        n = int(np.ceil(scale * (k * r + 2 * (k * r) ** (1 / 3) + 3) / (np.pi / c.omega)))
        if sources and c.omega < np.pi / 2:
            n //= 2
        f = FourierBessel.at_corner(c, n=max(n, 3), domain=dom)
        b = f if b is None else b + f
    if sources:
        perim = float(np.sum(np.abs(np.diff(np.append(zs, zs[:1])))))
        nsrc = int(np.ceil(scale * 3 * k * perim / (2 * np.pi)))
        b = b + Fundamental.offset(dom.bdry_quad(nsrc), 0.05 * dom.diameter, domain=dom)
    return b


@pytest.mark.parametrize("scale,sources", [(1.2, True), (1.5, False)], ids=["B", "A"])
def test_a_noisy_floor_does_not_multiply_a_simple_level(scale, sources):
    """Measured before the fix: run B returned 807.0076 x8, run A x8 at 807, 887 and 916. Every
    level here is simple. A declined search is not a pass for this test's purpose -- it says
    nothing about the count -- so it is reported as a skip, not hidden."""
    dom = Polygon(SPIKE)
    b = _spike_basis(dom, scale, sources)
    n = len(b)
    s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=1e-12)
    res = s.search_interval(*SPIKE_WINDOW)
    if res.diagnostics.declined:
        pytest.skip(f"search declined: {res.diagnostics.decline_reason[:80]}")
    multi = [(L.lam, L.mult) for L in res.levels if L.mult > 1]
    assert not multi, multi
    for lam in SPIKE_LEVELS:
        near = _near(res, lam, rel=1e-6)
        assert sum(L.mult for L in near) <= 1, (lam, [(L.lam, L.mult) for L in near])
