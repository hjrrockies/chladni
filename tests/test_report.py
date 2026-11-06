"""`SolveRecord.report(tier)`: what a solve computed along the way (2026-11-06).

The report must not change a solve, must cost nothing at tiers 0 and 1, and must be a JSON-able
record a sweep can write as it is. Tier 2's Hadamard fields must be the ones `hadamard` reports.
"""
import json
import warnings

import numpy as np
import pytest

from chladni._core import PrecisionWarning
from chladni.eigenproblem import Eigenproblem
from chladni.geometry import rect
from chladni.mps.solver import MPSEigensolver

PREC = 1e-10


@pytest.fixture(scope="module")
def rec():
    return Eigenproblem(rect(1.0, 0.7)).solve(3, prec=PREC)


def _strict_json(d):
    return json.loads(json.dumps(d, allow_nan=False))


@pytest.mark.slow
@pytest.mark.parametrize("tier", [0, 1, 2])
def test_a_report_is_strict_json_at_every_tier(rec, tier):
    rep = rec.report(tier)
    d = _strict_json(rep.as_dict())
    assert d["tier"] == tier and len(d["levels"]) == len(rec.levels)
    assert d["run"]["pencil"]["n_cols"] == rec.solver.n_cols
    assert d["run"]["plan"]["rung"] == rec.plan.rung


@pytest.mark.slow
def test_the_free_tiers_evaluate_nothing(rec, monkeypatch):
    calls = []
    for name in ("tension", "tensions", "rank_diagnostics", "hadamard", "eigenfunctions"):
        orig = getattr(MPSEigensolver, name)

        def counted(self, *a, _orig=orig, _name=name, **k):
            calls.append(_name)
            return _orig(self, *a, **k)
        monkeypatch.setattr(MPSEigensolver, name, counted)
    rec.report(1)
    assert calls == []
    rec.report(2)
    assert {"tension", "rank_diagnostics", "hadamard"} <= set(calls)


@pytest.mark.slow
def test_tier1_keeps_what_the_search_and_planner_threw_away(rec):
    rep = rec.report(1)
    L = rep.levels[0]
    for key in ("search_sigma", "search_delta_res", "search_count_decided_by", "search_mp_bound",
                "search_mp_drift", "search_mp_norm_spread", "search_rule_prec"):
        assert key in L, key
    ladder = rep["ladder"]
    assert [r["accepted"] for r in ladder].count(True) == 1
    accepted = next(r for r in ladder if r["accepted"])
    assert accepted["rung"] == rec.plan.rung and accepted["seconds"] > 0
    assert rep["seconds"] >= sum(r["seconds"] for r in ladder) * 0.99
    assert accepted["why"] == "" and rep["evaluations"] >= accepted["evaluations"] > 0
    assert len(rep["corner_points"]) == 4


@pytest.mark.slow
def test_the_ladder_says_why_each_rejected_rung_was_rejected_and_what_it_cost():
    from chladni.geometry import Polygon
    # a thin triangle the low rungs cannot do at 1e-8: each stops early on an unmet level
    r = Eigenproblem(Polygon([0, 1, 0.1 + 0.25j])).solve(3, prec=1e-8)
    rejected = [x for x in r.ladder if not x["accepted"]]
    assert rejected
    for x in rejected:
        assert x["why"], x
        assert x["evaluations"] > 0 or not x["usable"]
    assert all("unmet" in x["why"] for x in rejected if x["aborted"])


@pytest.mark.slow
def test_tier2_hadamard_fields_are_hadamards_own(rec):
    rep = rec.report(2)

    def V(curve, t):
        return np.ones((1, len(t)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        _, want = rec.hadamard(0, V, report=True)
    L = rep.levels[0]
    assert L["hadamard_attained"] == pytest.approx(want.attained, rel=1e-12)
    assert L["hadamard_identity_residual"] == pytest.approx(want.identity_residual, rel=1e-12)
    # a clean rectangle: the root sits far above its floor, the eigenfunctions are orthonormal
    assert L["tension_floor_rel"] < PREC and L["ef_orthonormal"]
    assert L["svd_rank"] == rec.solver.rank or rec.solver.rank is None


@pytest.mark.slow
def test_a_declined_solve_still_reports():
    from chladni.geometry import Polygon
    # a slit a millionth of the diameter wide: the planner declines before any solve
    dom = Polygon([0, 1, 1 + 0.5j, 0.5 + 0.5j, 0.5 + 0.500001j, 1 + 0.500001j, 1 + 1j, 1j])
    r = Eigenproblem(dom).solve(3, prec=PREC)
    assert r.spectrum.declined
    d = _strict_json(r.report(2).as_dict())
    assert d["run"]["declined"] and d["levels"] == []


def test_tier_must_be_0_1_or_2():
    from chladni.report import solve_report
    with pytest.raises(ValueError, match="tier"):
        solve_report(None, tier=3)
