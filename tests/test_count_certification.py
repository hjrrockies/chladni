"""A multiplicity the pipeline cannot certify must SAY so -- the safety net before the fix.

Plan Phase 3a, ported from `wip/v1-promotion-and-multiplicity` commit `a64229f`. The cluster
defect -- a collapsed cluster reporting fewer eigenvalues than it holds -- was SILENT on main: in
every wrong cell `declined` was False, `unmet` was empty and nothing warned, so
`solve_interval` returned a short array and `Eigensolver.solve(n)` answered it by widening the
window. This does not fix a single count. It makes the doubt visible, which is what a caller
needs before the count is right.

Two independent readings, both free, and neither decides anything:

    mult < n_roots        a level declares fewer eigenvalues than the accepted, refined,
                          certificate-carrying minima folded into it -- the solver contradicting
                          its own locator
    cross-check disagrees `_collapse_ambiguous`'s spanning count and the sum of the fine counts
                          differ, and neither reading is right on both cluster families

Everything here is on hand-built levels, in the default tier. The measurement of which real cells
it catches lives in `tests/test_cluster_totals.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.mps.search import Level, SolveDiagnostics, SolveResult, _flag_uncertain_counts

X = 49.348022005446786


def test_a_level_declaring_FEWER_than_its_folded_roots_is_flagged():
    """Two accepted minima folded into one level that reads multiplicity 1."""
    out, flagged = _flag_uncertain_counts([Level(lam=X, mult=1, n_roots=2)])
    assert len(flagged) == 1 and out[0].count_uncertain
    assert out[0].mult == 1, "the value does not move: flagged, not corrected"
    assert out[0].count_alt == 2
    assert out[0].uncertain_reason.strip()


def test_a_consistent_level_is_left_alone():
    out, flagged = _flag_uncertain_counts([Level(lam=X, mult=2, n_roots=2),
                                           Level(lam=2 * X, mult=3, n_roots=1)])
    assert not flagged
    assert not any(x.count_uncertain for x in out)


def test_a_level_ALREADY_flagged_upstream_is_still_counted():
    """An earlier version skipped these, so the branch that detects a disagreement by
    construction was the one whose levels never reached `n_uncertain`: 14% recall against 93%."""
    upstream = Level(lam=X, mult=1, n_roots=1, count_uncertain=True, count_alt=3,
                     uncertain_reason="the cross-check's two counts disagree")
    out, flagged = _flag_uncertain_counts([upstream])
    assert flagged == [upstream] and out == [upstream]


def test_certified_counts_is_its_own_question_beside_met_request():
    """Both were True in every wrong cell, because only one of them was being asked."""
    d = SolveDiagnostics(n_uncertain=1, uncertain=((X, 1, 2, "why"),))
    assert d.met_request, "the value really was located to prec"
    assert not d.certified_counts
    assert SolveDiagnostics().certified_counts


def test_solve_interval_RAISES_on_an_uncertain_count(monkeypatch):
    """An array with multiplicity repeated cannot say "and possibly one more here", and a short
    array is answered by `solve(n)` widening the window. So it raises, and names the ways on."""
    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave
    from chladni.mps import MPSEigensolver

    s = MPSEigensolver(disk(1.0), basis=PlaneWave.uniform(12, radius=2.0), n_bdry=36, n_int=12,
                       rng=np.random.default_rng(0))
    bad = SolveResult((Level(lam=X, mult=1, n_roots=2, count_uncertain=True, count_alt=2,
                             uncertain_reason="two roots folded, one read"),),
                      SolveDiagnostics(n_uncertain=1, uncertain=((X, 1, 2, "two roots"),)))
    monkeypatch.setattr(type(s), "search_interval", lambda self, a, b, **kw: bad)
    with pytest.raises(RuntimeError, match="could not be certified"):
        s.solve_interval(40.0, 60.0)
