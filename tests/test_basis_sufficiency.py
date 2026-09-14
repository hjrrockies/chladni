"""Can this case's basis serve this `prec`? One answer, used everywhere an excuse is granted.

`tests/basis_sufficiency.py` is the ONLY place a benchmark cell may be excused: a shortfall is
`b-excused` only if the basis cannot serve the accuracy, and a merge or wrong total only if it
cannot serve the multiplicity. It was three separate checks before -- `surrogate_cases.adequate`,
`rect_suite.assert_floor`, `hard_roster.finest_prec` -- each written after its tier learned the
same lesson alone, and the ladder test excused misses by one while the gate skipped cells by
another.

Everything here runs on a fake pencil built from exact cones, so the floor `eps` and the wall
slope `c` are known and the criteria are pinned as formulas, in the default tier.
"""
from __future__ import annotations

import numpy as np
import pytest

from basis_sufficiency import bracketing_prec, multiplicity_prec, sufficient

LAM = 50.0


class FakePencil:
    """`sigma` vector from exact cones: one branch per unit of multiplicity at each level.

    `floors` maps a level to the floor of each of its branches, so a degenerate level can have
    a second branch that is worse resolved than the first -- which is what `sigma_mu` exists to
    catch.
    """

    def __init__(self, floors, c=1.0, plateau=10.0, rtol=1.49e-15, prec=1e-8):
        self.floors = {float(k): tuple(v) for k, v in floors.items()}
        self.c, self.plateau, self.rtol, self.prec = c, plateau, rtol, prec

    def tensions(self, lam):
        out = [self.plateau]
        for at, fl in self.floors.items():
            out += [min(np.sqrt(e * e + (self.c * (lam - at)) ** 2), self.plateau) for e in fl]
        return np.sort(np.array(out))

    def tension(self, lam):
        return float(self.tensions(lam)[0])


class FakeCase:
    def __init__(self, floors, *, digits=15.6, **kw):
        self.pencil = FakePencil(floors, **kw)
        self.truth = np.array(sorted(at for at, fl in floors.items() for _ in fl), dtype=float)
        self.digits = digits
        self.label = "fake"


# --- the two criteria, as formulas ---------------------------------------------------------------

def test_the_bracketing_criterion_is_three_resolution_widths():
    """A bracket of width `prec*lam` is usable while it is wider than the flat bottom, `eps/c`,
    with a margin of 3 -- `rect_suite.assert_floor`'s criterion, now in one place."""
    case = FakeCase({LAM: (1e-11,)})
    assert bracketing_prec(case) == pytest.approx(3 * 1e-11 / LAM, rel=0.05)


def test_the_multiplicity_criterion_is_twenty_resolution_widths_at_SIGMA_MU():
    """`search.md` section 2.2's `gap > 20*eps/c`, read at the level's multiplicity: at a
    double it is `sigma_2`'s floor that decides whether the pair can be told apart."""
    case = FakeCase({LAM: (1e-15, 1e-10)})
    assert multiplicity_prec(case) == pytest.approx(20 * 1e-10 / LAM, rel=0.05)


# --- the verdict ---------------------------------------------------------------------------------

@pytest.mark.parametrize("prec", [1e-4, 1e-8, 1e-12, 1e-14])
def test_a_reference_grade_basis_serves_the_whole_ladder(prec):
    s = sufficient(FakeCase({LAM: (1e-15,), 2 * LAM: (1e-15,)}), prec)
    assert s.accuracy and s.multiplicity, s.reasons


def test_a_floor_of_1e_11_serves_accuracy_to_1e_12_and_multiplicity_to_1e_10():
    case = FakeCase({LAM: (1e-11,)})
    assert sufficient(case, 1e-12).accuracy
    assert not sufficient(case, 1e-14).accuracy
    assert sufficient(case, 1e-10).multiplicity
    assert not sufficient(case, 1e-12).multiplicity


def test_a_badly_resolved_SECOND_branch_withdraws_only_the_multiplicity_excuse():
    case = FakeCase({LAM: (1e-15, 1e-10)})
    s = sufficient(case, 1e-12)
    assert s.accuracy, "sigma_1 reaches the floor, so the location can be served"
    assert not s.multiplicity, "sigma_2 does not, so the count cannot"


def test_truth_limits_accuracy_like_a_basis_does():
    """A 7.8-digit table cannot score an 8-digit request. One spare digit is required, since a
    table good to `d` digits is not a reference for `prec = 10**-d` itself."""
    case = FakeCase({LAM: (1e-15,)}, digits=7.8)
    assert sufficient(case, 1e-6).accuracy
    assert not sufficient(case, 1e-8).accuracy
    assert any("digits" in r for r in sufficient(case, 1e-8).reasons)


def test_an_insufficient_verdict_always_says_why():
    s = sufficient(FakeCase({LAM: (1e-9, 1e-9)}), 1e-12)
    assert not (s.accuracy and s.multiplicity)
    assert s.reasons and all(r.strip() for r in s.reasons)


def test_an_rtol_DERIVED_from_prec_is_flagged_and_never_excuses_anything():
    """A pencil whose `rtol` is `sqrt(prec*eps)` changes with the request, so a cell scored on
    it confounds the basis with the rank policy. That is a benchmark configuration error: it is
    reported, and it must not turn a failure into an excuse."""
    derived = float(np.sqrt(1e-8 * np.finfo(float).eps))
    case = FakeCase({LAM: (1e-15,)}, rtol=derived, prec=1e-8)
    s = sufficient(case, 1e-14)
    assert not s.frozen
    assert s.accuracy and s.multiplicity, "flagging is not excusing"
    assert any("rtol" in r for r in s.reasons)
    assert sufficient(FakeCase({LAM: (1e-15,)}), 1e-14).frozen
