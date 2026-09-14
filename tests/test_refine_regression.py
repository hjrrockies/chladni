"""Stage 2 regression: the refiner declared `converged` far from the minimum.

On real plane-wave rectangle pencils whose tension at the true eigenvalue is ~1.5e-15, `refine`
without a hint returned `converged, accepted` after 3-9 steps with `sigma_1` still at 1e-11 to
1e-9, and the pipeline carried that error unchanged from `prec = 1e-8` to `1e-14`. The ladder
benchmark scored it `b` and blamed the basis.

Two defects, and neither was visible to `tests/test_refine.py`, whose accuracy tests all hand
the refiner SYMMETRIC brackets `LAM +/- w` -- so the midpoint start already IS the answer and
the tests only check that it is not lost.

1. **`_vertex` cancels.** It built the linear coefficient from `x_i**2` with the origin at 0,
   while the samples sit near `lam`. Each rounded square carries an error of order
   `ulp(lam**2)`, so the vertex is off by about `ulp(lam**2)/h` and stalls once the triple
   spacing `h` reaches `lam*sqrt(eps)`. Measured on exact parabolas: 2e-9 relative at
   `h = 1e-8*lam`, 2e-7 at `1e-10*lam`, against 0 from a formula centred on the middle point.
2. **The repeated-point exit had no width check.** Once the vertex scatters at that scale it
   lands on a point already sampled, and the loop reported `converged` whatever the width of
   the triple.

Every test here is on the tier-1 cone, milliseconds, except the last, which is the real pencil
the defect was found on.
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.mps.objective import Objective
from chladni.mps.refine import _vertex, _xatol, refine

EPS, C_K, PLATEAU = 1e-15, 0.27, 0.2
LAMS = (10.0, 49.348022005446786, 1e3, 1e5)
SIDES = ((0.1, 3.0), (0.37, 17.0), (3.0, 0.1), (17.0, 0.37), (1.0, 1.0))


def cone(lam0, *, eps=EPS, prec=1e-14):
    """One cone in `k` at `lam0`, with the floor of a good real pencil. No pencil."""
    k0 = np.sqrt(lam0)

    def tensions(k):
        return [min(np.sqrt(eps**2 + (C_K * (float(k) - k0)) ** 2), PLATEAU), PLATEAU]

    return Objective(tensions, 1.0, 1e3, variable="k", prec=prec)


def _width(lam0, rel, a, b):
    """A bracket scale whose ends stay on the cone's walls.

    Below the plateau the cone is `PLATEAU/C_K` wide each side in `k`, about
    `2*sqrt(lam0)*PLATEAU/C_K` in `lam`. An end past that sits on the flat plateau, where no
    parabola closes and `no_parabola` is the CORRECT outcome -- a test of the fit's domain, not
    of the refiner. Half the reach keeps both ends on a wall.
    """
    return min(rel * lam0, np.sqrt(lam0) * PLATEAU / C_K / max(a, b))


def _converged_means_narrow(r, prec):
    """The invariant the defect broke: `converged` is a claim about the WIDTH of the triple."""
    return r.reason != "converged" or r.width <= _xatol(r.lam, prec)


# --- the vertex ---------------------------------------------------------------------------------

@pytest.mark.parametrize("m", [49.35, 493.5, 3208.0, 1e5])
@pytest.mark.parametrize("h", [1e-2, 1e-5, 1e-8, 1e-10, 1e-12])
def test_the_vertex_does_not_cancel_as_the_triple_closes(m, h):
    """An exact parabola, off-centre in an uneven triple. The vertex must be good to a few ulps
    of `m` at every spacing -- not to `ulp(m**2)/h`, which is 2e-7 relative at the tightest."""
    H = h * m
    x_star = m + 0.3 * H
    x = [m - H, m, m + 1.7 * H]
    y = [(xi - x_star) ** 2 for xi in x]
    v = _vertex(x, y)
    assert v is not None
    assert abs(v[0] - x_star) <= 4.0 * np.spacing(m), (
        f"vertex off by {abs(v[0] - x_star) / np.spacing(m):.0f} ulps at h = {h:g}")


def test_the_vertex_curvature_is_the_leading_coefficient():
    """`delta_res` reads `c = sqrt(c2)`, so `c2` must stay the leading coefficient exactly."""
    x = [1.0, 2.0, 4.0]
    y = [3.0 * (xi - 2.5) ** 2 + 1.0 for xi in x]
    x_star, c2 = _vertex(x, y)
    assert x_star == pytest.approx(2.5, rel=1e-15)
    assert c2 == pytest.approx(3.0, rel=1e-15)


# --- the refiner, from asymmetric brackets ------------------------------------------------------

@pytest.mark.parametrize("lam0", LAMS)
@pytest.mark.parametrize("a, b", SIDES)
def test_it_reaches_the_floor_from_an_ASYMMETRIC_bracket(lam0, a, b):
    """The case `test_refine.py` never posed: the minimum off-centre, and no hint. Before the fix
    every one of these read 1e-11 to 4e-10 relative, `converged` and accepted."""
    w = _width(lam0, 1e-3, a, b)
    r = refine(cone(lam0), lam0 - a * w, lam0 + b * w, prec=1e-14)
    assert r.accepted, r.reason
    assert _converged_means_narrow(r, 1e-14), (
        f"converged at width {r.width:.2e} against xatol {_xatol(r.lam, 1e-14):.2e}")
    assert abs(r.lam - lam0) <= _xatol(lam0, 1e-14), (
        f"relative error {abs(r.lam - lam0) / lam0:.2e}, reason {r.reason}, {r.steps} steps")
    assert abs(r.lam - lam0) <= r.delta_res, "and the reported bar still contains the error"


def test_the_invariant_holds_over_a_seeded_sweep_of_brackets():
    """Many brackets, many positions, one invariant. A cheap net for the ways the repeated-point
    exit can be reached -- a clip onto an endpoint, a vertex rounding onto a sample, a nudge
    landing on an older point -- without constructing each by hand."""
    rng = np.random.default_rng(20260914)
    bad = []
    for _ in range(300):
        lam0 = float(10 ** rng.uniform(1, 5))
        a, b = (float(v) for v in 10 ** rng.uniform(-1.5, 1.5, size=2))
        w = _width(lam0, float(10 ** rng.uniform(-6, -1)), a, b)
        prec = float(10 ** rng.uniform(-14, -6))
        r = refine(cone(lam0, prec=prec), lam0 - a * w, lam0 + b * w, prec=prec)
        if not r.accepted:
            bad.append(("rejected", lam0, a, b, prec, r.reason))
        elif not _converged_means_narrow(r, prec):
            bad.append(("wide-converged", lam0, a, b, prec, r.width))
        elif abs(r.lam - lam0) > max(_xatol(lam0, prec), r.delta_res):
            bad.append(("inaccurate", lam0, a, b, prec, abs(r.lam - lam0) / lam0))
    assert not bad, f"{len(bad)} of 300 failed; first: {bad[:3]}"


def test_a_hint_AT_the_answer_does_not_end_the_refinement_at_the_bracket_width():
    """The repeated-point exit fires on the first step when the hint is exact: the vertex lands
    on the middle point. That is right about the location and says nothing about the width,
    which is still the whole bracket. `converged` must mean narrow, so it must keep going."""
    lam0 = 49.348022005446786
    r = refine(cone(lam0), lam0 - 0.37e-2, lam0 + 1.9e-2, prec=1e-14, lam_hint=lam0)
    assert r.accepted
    assert abs(r.lam - lam0) <= _xatol(lam0, 1e-14)
    assert _converged_means_narrow(r, 1e-14), f"converged at width {r.width:.2e}"


def test_on_a_FLAT_bottom_a_narrow_triple_does_not_bound_the_error_and_the_bar_must():
    """A floor that is flat over 1e-9 relative, so every point inside ties. The refiner fences
    whichever tied point it holds and reports `converged` at a triple 4.4e-13 wide -- correctly,
    since `converged` is a claim about the width -- while the true minimizer sits 5e-9 away.

    So the narrow triple is NOT an error bound on a flat bottom, and nothing in the refiner may
    treat it as one. What covers the error is the resolution limit `2*eps/c`, which reads 1e-4
    here. That is the property that has to hold whatever the reason code."""
    lam0 = 49.348022005446786
    k0 = np.sqrt(lam0)
    flat = 1e-9 * lam0

    def tensions(k):
        d = abs(float(k) ** 2 - lam0)
        return [max(np.sqrt(EPS**2 + (C_K * (float(k) - k0)) ** 2), 1e-6) if d < flat
                else min(np.sqrt(EPS**2 + (C_K * (float(k) - k0)) ** 2), PLATEAU), PLATEAU]

    obj = Objective(tensions, 1.0, 1e3, variable="k", prec=1e-14)
    r = refine(obj, lam0 - 1e-3, lam0 + 3e-3, prec=1e-14)
    assert r.accepted, r.reason
    assert _converged_means_narrow(r, 1e-14)
    assert abs(r.lam - lam0) <= r.delta_res, (
        f"error {abs(r.lam - lam0):.2e} outside the bar {r.delta_res:.2e} (width {r.width:.2e})")


# --- the real pencil it was found on ------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("prec", [1e-12, 1e-14])
def test_it_reaches_the_floor_on_a_real_pencil_from_asymmetric_brackets(prec):
    """`rect(1, 1/phi)`, tension ~1.5e-15 at every true eigenvalue. Asymmetric brackets 1e-3
    relative, the scale stage 1 hands on, no hint. Before the fix: 1e-12 to 1e-9 relative, all
    `converged`."""
    import bracket_score as bs
    import rect_suite as rs

    (case,) = rs.family_A()
    obj = bs.objective_for(case, prec)
    worst = []
    levels = np.unique(case.truth)
    for i, lam0 in enumerate(levels):
        below = lam0 - levels[i - 1] if i else np.inf
        above = levels[i + 1] - lam0 if i + 1 < levels.size else np.inf
        for a, b in ((0.37, 1.9), (2.3, 0.21)):
            obj.reset()
            # A bracket holds ONE level: each side stays inside a quarter of the gap to the
            # neighbour, or the fit spans two wells and `no_parabola` is the correct answer.
            w = min(1e-3 * lam0, 0.25 * below / a, 0.25 * above / b)
            r = refine(obj, lam0 - a * w, lam0 + b * w, prec=prec)
            assert r.accepted, f"{lam0}: {r.reason}"
            assert _converged_means_narrow(r, prec)
            worst.append(abs(r.lam - lam0) / lam0)
    assert max(worst) <= max(prec, 1e-14), f"worst relative error {max(worst):.2e}"


# --- a bracket already narrower than the tolerance must still be LOCATED -------------------------

@pytest.mark.parametrize("lam0", [49.348022005446786, 493.48022005446785])
@pytest.mark.parametrize("prec", [1e-8, 1e-12])
@pytest.mark.parametrize("frac", [0.2, 0.35, 0.75])
def test_a_bracket_NARROWER_than_xatol_is_still_located_and_its_bar_holds(lam0, prec, frac):
    """Found porting the multiplicity instrument. A bracket two licences wide is inside `_xatol`
    already, so the first vertex is trivially "within tolerance", both fences clip onto the ends,
    and the refiner reported `converged` at its MIDPOINT without evaluating a vertex -- the root
    sat at 0.5 of the width where it truly sat at 0.75. Within `prec`, so no accuracy check sees
    it; but the bar was the model's `2*eps/c`, orders below the half-bracket error, which is
    clause (3) broken silently -- and the multiplicity stage reads distances from that point.

    Measured on the exact cone: with the root nearer an END than the midpoint (0.2, 0.75) the
    lowest of the three starting points is an endpoint, the fence pass "converges" on it, and
    `_accept` REJECTS the bracket -- the eigenvalue is dropped. With the root nearer the middle
    (0.35) it was already located correctly, and that case stays here as the guard that the fix
    does not trade one position for the other. The pipeline escaped only because it hands the
    refiner the deepest sample as a hint; the multiplicity instrument hands it none.

    The pre-Phase-0 refiner always took vertex steps before any width check. This pins that it
    still locates: the bar contains the error, whatever the width rule allowed."""
    # 0.9 of `_xatol = prec*lam` wide, so the whole bracket is inside the tolerance. Two
    # licences wide was wider than the tolerance and located normally, which is why the first
    # version of this test passed without reproducing anything.
    lo = lam0 * (1 - 0.9 * prec * frac)
    hi = lam0 * (1 + 0.9 * prec * (1 - frac))
    r = refine(cone(lam0), lo, hi, prec=prec)
    assert r.accepted, r.reason
    assert _converged_means_narrow(r, prec)
    assert abs(r.lam - lam0) <= r.delta_res, (
        f"located {abs(r.lam - lam0) / lam0:.2e} relative from the root with a bar of "
        f"{r.delta_res / lam0:.2e}; the root sits at {frac} of the bracket")
