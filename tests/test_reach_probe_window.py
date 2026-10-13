"""The reach rule's side probes stay inside the search window.  (2026-10-13)

`reach_multiplicity` settles a branch in its undecided band by evaluating `sigma_1` at
`x - d` and `x + d`, with `d = sigma_j / c` the branch's estimated distance. On a flat,
ill-conditioned objective `c` is tiny and the bracket wide, so `x - d` can fall below the window --
below ZERO, where the Helmholtz layer refuses (`lam must be ... strictly positive`). This was hit
on a thin triangle with lightning sources (`notes/hybrid_bases.py`, round 2), and it crashed the
solve.

The fix is principled, not a clamp. The bracket lies inside the window, so an eigenvalue at
distance `d` on a side that leaves the window cannot be inside the bracket either: that side holds
nothing for this count. It is not evaluated, and the existing logic decides from the other side,
or declines if neither side holds.
"""
import numpy as np
import pytest

from chladni.mps.multiplicity import multiplicity, reach_multiplicity

WINDOW = (0.5, 40.0)


def _tensions(z):
    """sigma_1 dips at 10 and at 25; sigma_2 = 15 everywhere. It RAISES outside the window, as the
    real pencil does at lam <= 0."""
    z = float(z)
    if not WINDOW[0] < z < WINDOW[1]:
        raise ValueError(f"evaluated outside the window at {z}")
    return np.array([min(abs(z - 10.0), abs(z - 25.0)) + 1e-12, 15.0])


def test_a_probe_outside_the_window_is_not_evaluated_and_the_other_side_decides():
    """x = 10 in the bracket [1, 30], c = 1: branch 1 sits d = 15 away, in the band (near 9, far
    20). Its low probe would be at x - d = -5. Not evaluated: the high probe at 25 holds it, and
    it lies inside the long side (d / BIAS_LO = 16.9 <= 20), so it is counted."""
    mu, declined, _, seen = reach_multiplicity(_tensions, 10.0, 1.0, 30.0, 1.0, window=WINDOW)
    assert mu == 2 and not declined, seen


def test_the_window_reaches_the_rule_through_multiplicity():
    m = multiplicity(_tensions, 10.0, rule="reach", bracket=(1.0, 30.0), c=1.0, window=WINDOW)
    assert m.mu == 2


def test_without_a_window_the_old_behaviour_is_unchanged():
    """No window given: both probes are evaluated, as before (here the low one raises)."""
    with pytest.raises(ValueError, match="outside the window"):
        reach_multiplicity(_tensions, 10.0, 1.0, 30.0, 1.0)
