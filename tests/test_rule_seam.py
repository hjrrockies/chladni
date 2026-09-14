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
