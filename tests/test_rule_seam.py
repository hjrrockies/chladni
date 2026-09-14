"""The multiplicity rule seam, on exact cones. Plan Phase 3b.

`multiplicity(..., rule=)` and `solve(..., rule=, merge=)` exist so the counting rule and the
merge policy can be changed and measured one at a time. The defaults must reproduce the committed
behaviour exactly -- the seam is plumbing, and the wip branch's version of it failed twice,
silently, by not threading `rule` into `_collapse_ambiguous` and by a patch that matched nothing.

Three rules answer three different questions, which is the point of having them side by side:

    gap     (C1)+(C2) over a wide bracket   the shipped rule
    floor   branches coincident with sigma_1   right about COINCIDENCE, blind to the rest
    reach   eigenvalues INSIDE the bracket   `mu*(B)`: a distance `sigma_j/c` against the bracket
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from chladni.mps.multiplicity import multiplicity
from chladni.mps.search import solve

C, EPS, PLATEAU = 1.0, 1e-15, 10.0


def cones(*centres):
    """`sigma` vector from exact cones in `lam`, one branch per centre, slope `C`."""
    centres = np.asarray(centres, dtype=float)

    def tensions(lam):
        return np.sort(np.minimum(np.sqrt(EPS**2 + (C * (centres - float(lam))) ** 2), PLATEAU))

    return tensions


def test_the_defaults_are_the_shipped_behaviour():
    assert inspect.signature(multiplicity).parameters["rule"].default == "gap"
    p = inspect.signature(solve).parameters
    assert p["rule"].default == "gap" and p["merge"].default == "licence"


def test_reach_needs_the_bracket_and_the_slope():
    with pytest.raises(ValueError, match="bracket"):
        multiplicity(cones(10.0), 10.0, rule="reach")


def test_reach_COUNTS_a_pair_inside_the_bracket_and_REJECTS_a_neighbour_outside():
    """Two eigenvalues 0.3 apart inside `[9.5, 10.5]`, a third 2.0 away. `mu*(B) = 2`."""
    t = cones(10.0, 10.3, 12.0)
    m = multiplicity(t, 10.0, rule="reach", bracket=(9.5, 10.5), c=C)
    assert m.mu == 2 and not m.ambiguous


def test_reach_resolves_the_SIDE_on_an_asymmetric_bracket():
    """The partner 0.6 away on the LONG side of `[9.9, 10.8]`: inside, though farther than the
    short side reaches. `min(near, far)` rejected exactly this and under-counted."""
    m = multiplicity(cones(10.0, 10.6), 10.0, rule="reach", bracket=(9.9, 10.8), c=C)
    assert m.mu == 2


def test_floor_counts_COINCIDENCE_only():
    """An exact double reads 2; a pair 1e-3 apart -- inside any sensible bracket but nowhere near
    the floor together -- reads 1. The floor rule answers a different question than `mu*(B)`."""
    assert multiplicity(cones(10.0, 10.0), 10.0, rule="floor").mu == 2
    assert multiplicity(cones(10.0, 10.001), 10.0, rule="floor").mu == 1


def test_the_reach_slope_is_a_choice_and_OUTSIDE_is_still_the_default():
    """`reach_slope` picks where the reach rule reads its wall slope. "outside" -- three bracket
    widths out -- is what the rule was calibrated on and stays the default until a recalibration
    clears acceptance. "cache" reads the nearest cached samples either side of the root, which
    measured within 7% of the true distance on the cells where "outside" read 1.86x: on a cluster
    wider than the standoff, the outside samples sit in other members' wells."""
    p = inspect.signature(solve).parameters
    assert p["reach_slope"].default == "outside"
    with pytest.raises(ValueError, match="reach_slope"):
        from chladni.mps.objective import Objective
        solve(Objective(lambda k: [0.2, 0.3], 3.0, 4.0, variable="k", prec=1e-8),
              rule="reach", reach_slope="sideways")


# --- the reach rule LOCATES a partner it cannot count ----------------------------------------------

_REACH = dict(merge="attained", rule="reach", reach_slope="cache", collapse=False)


def _cluster_total(N, m, band, prec):
    import bracket_score as bs
    import rect_suite as rs

    (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: m})
    truth = rs.complete_truth(case)
    res = solve(bs.objective_for(case, prec), prec=prec, **_REACH)
    near = [L for L in res.levels if abs(L.lam - N * rs.PI2) < 0.02 * N * rs.PI2]
    return sum(L.mult for L in near), m, res.diagnostics


@pytest.mark.slow
@pytest.mark.parametrize("N, m, band, prec", [(50, 3, 0.3, 1e-8), (50, 3, 0.5, 1e-8),
                                              (65, 4, 0.3, 1e-8)])
def test_a_partner_at_the_bracket_edge_is_LOCATED_rather_than_left_uncounted(N, m, band, prec):
    """Every wrong total the recalibrated reach rule left on the cluster grid was one bracket
    holding two distinct eigenvalues: the refiner located one, the partner sat at the bracket
    edge where sigma_j/c cannot place it inside or out, and the rule declined -- honestly, with a
    total one short. Merging is permitted and never required, so the partner need not be counted
    at all: refining the bracket either side of the root, outside half the estimated distance,
    found it as its own root in every such case measured, at 1-2 evaluations a side."""
    total, truth, d = _cluster_total(N, m, band, prec)
    assert total == truth, f"declared {total} of {truth}"
    assert d.certified_counts, d.uncertain


@pytest.mark.slow
@pytest.mark.parametrize("prec, band, cause", [
    (1e-10, 0.5, "side-search dedup: the partner was discarded against the root being split"),
    (1e-8, 1.6, "phantom 4 ulps inside a bracket end, past the ulp tie"),
])
def test_the_fresh_125_grid_has_no_SILENT_wrong_total(prec, band, cause):
    """The two silent wrong totals the candidate left on the held-out 125*pi^2 grid. Pinning them
    makes 125 design data; 145 and 170 stay held out.

    1e-10, band 0.5: the rule declined at the root, the side search FOUND the partner 5.5e-8 away
    -- and discarded it as already located, because the dedup compared it with the root being
    split, whose own bar was 5.6e-8. The sub-bracket excludes that root by construction.
    1e-8, band 1.6: a phantom accepted by the refiner as a fifth level; see
    `test_refine_regression.test_a_phantom_a_FEW_ULPS_INSIDE_the_end_is_judged_by_its_reach...`."""
    total, truth, d = _cluster_total(125, 4, band, prec)
    assert total == truth or not d.certified_counts, (
        f"declared {total} of {truth} with certified counts -- a silent wrong total ({cause})")
    assert total == truth, f"declared {total} of {truth}, flagged ({cause})"


@pytest.mark.slow
def test_a_side_search_that_finds_NOTHING_leaves_a_correct_count_correct():
    """The uneven cluster at 1e-12, band 0.5 is already right under the reach rule, flagged
    because a branch landed in the undecidable band. Side searches there find no second root --
    there is none -- and the total must not move."""
    total, truth, _ = _cluster_total(65, 4, 0.5, 1e-12)
    assert total == truth
