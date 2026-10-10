"""Design evidence from `tests/test_search.py`, kept as runnable evidence.

Moved out of the suite verbatim on 2026-10-09 by the test-suite prune
(`notes/test_inventory_2026-10-09.md`, decision 6). These record why stage 1 is built as it is --
the seed doubles because a sparse seed underestimates the slope, depth is normalized by the
FURTHER flank because the nearer one rejects real dips, and the search beats a uniform grid on a
real pencil -- rather than a contract douse needs. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_search_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import surrogate_cases as sc
import chladni.mps.bracket as V
from candidates_v1 import v1_brackets


def objective(case, prec):
    return bs.objective_for(case, prec)


def test_the_seed_underestimates_the_slope_and_that_is_the_dangerous_direction():
    """The bootstrap hazard runs OPPOSITE to intuition: an underestimated `L` makes exclusion
    radii too LARGE, so a search that has not seen the true slope excludes everything and
    terminates having found nothing. That is why the seed doubles rather than trusting a
    density."""
    case = sc.isolated()
    obj = objective(case, 1e-8)
    coarse = np.linspace(obj.lo, obj.hi, 8)
    for x in coarse:
        obj.tension(x)
    ks, vs = obj.samples
    sparse = float(np.max(V._secants(ks, vs, obj.hi - obj.lo)))
    obj2 = objective(case, 1e-8)
    _, _, _, _ = V._seed(obj2, obj2.lo, obj2.hi, 1e-8)
    ks2, vs2 = obj2.samples
    settled = float(np.max(V._secants(ks2, vs2, obj2.hi - obj2.lo)))
    assert sparse < settled, f"8 points saw {sparse:.4g}, the settled estimate {settled:.4g}"


def test_ABLATION_a_nearer_flank_depth_rule_rejects_real_dips():
    """The measured failure: 19 of 26 minimizers instead of 26, at every grid density. Pinned
    as an ablation because the nearer flank is the natural thing to reach for."""
    rejected = 0
    ks = np.linspace(0.0, 10.0, 41)
    for offset in np.linspace(0.01, 0.24, 12):
        centre = 5.0 + offset
        vs = np.maximum(np.abs(ks - centre), 1e-13)
        cert = V.descent_certificate(ks, vs, 0.0, 10.0)
        if cert is None:
            continue
        near = vs[cert[0]] / min(vs[cert[0] - 1], vs[cert[0] + 1])
        if near > V.DIP_RATIO:
            rejected += 1
    assert rejected > 0, "the nearer-flank rule must reject some genuine mid-cell cones"


@pytest.mark.slow
def test_it_beats_the_uniform_grid_baseline_on_a_real_pencil():
    """`lipschitz-cone-mps.md` section 3.2 concedes the honest comparison is a well-chosen
    uniform grid. A search that cannot beat one has not earned its complexity."""
    import rect_suite as rs
    from baselines import BASELINES

    (case,) = rs.family_A()
    v1 = bs.run_case(v1_brackets(), case, 1e-8)
    grid = bs.run_case(BASELINES["uniform_grid"](), case, 1e-8)
    assert v1.coverage >= grid.coverage
    assert v1.per_eigenvalue < grid.per_eigenvalue or v1.pure and not grid.pure
