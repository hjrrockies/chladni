"""Design evidence moved out of `tests/test_floor.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: a rank cutoff below machine epsilon barely
truncates at the top of the window, and what that costs is spurious minima, not floor. Both
justify `rect_suite.SUITE_RTOL`; neither is a contract douse needs. `BARELY_TRUNCATES` also counts
rounding-level directions, so its number is a platform pin (81 of 92 on Accelerate, 71 on
OpenBLAS) and jitter moves it by construction (inventory section 8). Moved verbatim, with the
helpers they need. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_floor_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

import rect_suite as rs
from chladni.reference import rect_eig


A_WINDOW = (5.0, 120.0)


def _truth(L, H, a, b):
    return np.sort([rect_eig(m, n, L, H) for m, n in rs.modes_in_window(a, b)
                    if a <= rect_eig(m, n, L, H) <= b])


def _spurious(pencil, truth, a, b, n=900):
    """Sampled local minima of `sigma_1` not near any true eigenvalue.

    This is what a rank cutoff that fails to truncate actually costs. It does NOT move the
    floor -- which is why a floor check alone never noticed -- it makes `sigma_1` rough
    BETWEEN eigenvalues, which is where a search spends all of its time.
    """
    xs = np.linspace(a, b, n)
    v = np.array([pencil.tension(x) for x in xs])
    idx = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1
    h = xs[1] - xs[0]
    hit = sum(1 for i in idx if np.min(np.abs(truth - xs[i])) < 3 * h)
    return len(idx) - hit, len(idx)


# --- the rank cutoff must actually truncate -------------------------------------------------

@pytest.mark.slow
def test_a_cutoff_below_machine_epsilon_BARELY_TRUNCATES_at_the_top_of_the_window():
    """`rtol = 1e-16` is below `eps = 2.2e-16`, so under a RELATIVE threshold it keeps
    essentially everything the arithmetic can distinguish -- and it does so from the TOP down,
    where basis demand is highest. A floor checked only at the low end proves nothing.

    The claim here was once "retains EVERY column", `rank_at(hi) == n_cols`, and that was true
    only with the oversized collocation this suite used to build: 368 boundary points against
    92 columns left all 92 numerically independent, while the sized 138 rows saturate the rank
    at 82. The phenomenon survives the correction and the exact equality does not, so the test
    asserts the phenomenon -- a cutoff below machine epsilon barely truncates at the top, and
    truncates hard at the bottom.

    What the cutoff COSTS is measured in the next test, and that is the part that matters: not
    the floor, which stays healthy, but hundreds of spurious minima between the eigenvalues.

    HOW MANY it keeps at the top is not a property of the pencil. Past index 60 the singular
    values at `b` are rounding, 4e-17 to 1.5e-16 relative, and `1e-16` lands inside that band, so
    the count is decided by how one LAPACK's rounding happens to fall: 81 of 92 on Accelerate,
    71 on OpenBLAS (Linux CI, 2026-10-04), against 51 at `SUITE_RTOL` on Accelerate. The old bounds,
    0.85 n and 1.5x the sound rank, were Accelerate's 81 with a margin, and failed on Linux. What
    holds on both is the phenomenon itself: the cutoff keeps a block of directions that a cutoff
    above rounding drops -- and every one of them sits below `SUITE_RTOL`, which is rounding.
    """
    a, b = A_WINDOW
    p = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=1e-16)
    good = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=rs.SUITE_RTOL)
    extra = p.rank_at(b) - good.rank_at(b)
    assert extra >= 0.1 * p.n_cols, (
        f"a cutoff below eps keeps {extra} directions of rounding a sound one drops; expected a "
        f"block of them (30 on Accelerate)")
    assert p.rank_at(a) < 0.6 * p.n_cols, "while the low end truncates normally"


@pytest.mark.slow
def test_the_inert_cutoff_costs_spurious_minima_and_NOT_the_floor():
    """Which is exactly why a floor check never noticed it. The damage is between the
    eigenvalues, where a search lives."""
    a, b = A_WINDOW
    truth = _truth(1.0, 1.0 / rs.PHI, a, b)
    inert = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=1e-16)
    good = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=rs.SUITE_RTOL)

    assert inert.tension(float(truth[0])) < 1e-13, "the floor is fine at 1e-16"
    n_bad, _ = _spurious(inert, truth, a, b)
    n_good, _ = _spurious(good, truth, a, b)
    assert n_bad > 100, f"but the plateau is rough: {n_bad} spurious minima"
    assert n_good == 0, f"and clean at SUITE_RTOL: {n_good}"
