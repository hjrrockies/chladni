"""The scorer for one (case, prec) cell, checked against hand-built results before it scores anything.

`tests/prec_score.py` turns a `SolveResult` and complete truth into ONE verdict, and the ladder,
the gate and the robust tier all read that verdict. So a scorer that is too strict rejects
correct answers everywhere, and one that is too loose passes wrong ones everywhere -- both
silently. Every verdict it can return is pinned here on a result built by hand, with no solve.

The acceptance rule it implements, as agreed:

* Width at most `prec*lam` MAY merge and never MUST. So a merged answer and a split answer
  within the licence both pass, and an exact degeneracy reported as two simples passes.
* A gap larger than `prec*lam` MUST split. Merging it is a failure unless the basis cannot
  resolve multiplicity at this `prec`.
* The total is scored everywhere. There is no unscored band; the only excuse is the basis.
* Falling short of `prec` is allowed only when REPORTED, with a bar that contains the error,
  and only where the basis cannot serve `prec`.
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.mps.search import Level, SolveDiagnostics, SolveResult
from prec_score import ACCEPTABLE, Sufficiency, score_cell

P = 1e-8
X = 49.348022005446786
OK = Sufficiency(accuracy=True, multiplicity=True)


def lvl(lam, mult=1, bar=None):
    """A level that met `P` unless told otherwise: its bar is a tenth of the licence."""
    return Level(lam=float(lam), mult=int(mult),
                 delta_res=float(0.1 * P * lam if bar is None else bar))


def res(*levels, declined=False, reason=""):
    return SolveResult(tuple(levels),
                       SolveDiagnostics(declined=declined, decline_reason=reason))


def verdict(result, truth, suff=OK, prec=P):
    return score_cell(result, np.asarray(truth, dtype=float), prec, suff).verdict


# --- what must pass ------------------------------------------------------------------------------

def test_an_exact_answer_passes():
    assert verdict(res(lvl(X), lvl(2 * X)), [X, 2 * X]) == "pass"


def test_a_cluster_inside_the_licence_passes_MERGED():
    assert verdict(res(lvl(X, 2)), [X, X * (1 + 0.5 * P)]) == "pass"


def test_the_same_cluster_passes_SPLIT():
    """Merging is permitted, never required: resolving more than was asked is not an error."""
    assert verdict(res(lvl(X), lvl(X * (1 + 0.5 * P))), [X, X * (1 + 0.5 * P)]) == "pass"


def test_an_exact_degeneracy_passes_MERGED():
    assert verdict(res(lvl(X, 2)), [X, X]) == "pass"


def test_an_exact_degeneracy_passes_SPLIT():
    """Allowed in the tests by decision. A nearest-level assignment would hand both truth values
    to one of the two levels and call the other spurious, which is why the scorer matches by
    capacity instead."""
    assert verdict(res(lvl(X), lvl(X)), [X, X]) == "pass"
    assert verdict(res(lvl(X), lvl(np.nextafter(X, np.inf))), [X, X]) == "pass"


# --- merging and counting ------------------------------------------------------------------------

def test_merging_ACROSS_the_licence_fails():
    assert verdict(res(lvl(X, 2)), [X, X * (1 + 3 * P)]) == "merge-above-licence"


def test_merging_across_the_licence_is_excused_ONLY_by_the_basis():
    bad = Sufficiency(accuracy=True, multiplicity=False)
    assert verdict(res(lvl(X, 2)), [X, X * (1 + 3 * P)], bad) == "merge-excused"


def test_a_dropped_member_is_a_wrong_total():
    """The failure the collapse defect produced: the well was found, the count was short."""
    assert verdict(res(lvl(X, 1)), [X, X * (1 + 0.5 * P)]) == "wrong-total"


def test_an_inflated_count_is_a_wrong_total():
    assert verdict(res(lvl(X, 2)), [X]) == "wrong-total"


def test_a_wrong_total_is_excused_ONLY_by_the_basis():
    bad = Sufficiency(accuracy=True, multiplicity=False)
    assert verdict(res(lvl(X, 1)), [X, X * (1 + 0.5 * P)], bad) == "wrong-total-excused"


# --- coverage ------------------------------------------------------------------------------------

def test_a_level_at_no_eigenvalue_is_spurious():
    assert verdict(res(lvl(X), lvl(1.5 * X)), [X]) == "spurious"


def test_an_eigenvalue_with_no_level_near_it_is_missed():
    """Distinct from a wrong total: no reported level is anywhere near, so the search never
    found the well at all."""
    assert verdict(res(lvl(X)), [X, 1.5 * X]) == "missed"


# --- accuracy and honesty ------------------------------------------------------------------------

def test_a_REPORTED_shortfall_where_the_basis_could_serve_is_a_failure():
    """The false floor near 1e-10 read as this: reported, honest, and wrong to accept."""
    assert verdict(res(lvl(X * (1 + 10 * P), bar=20 * P * X)), [X]) == "b-fail"


def test_a_reported_shortfall_is_excused_where_the_basis_cannot_serve():
    bad = Sufficiency(accuracy=False, multiplicity=True)
    assert verdict(res(lvl(X * (1 + 10 * P), bar=20 * P * X)), [X], bad) == "b-excused"


def test_a_SILENT_shortfall_is_never_excused():
    """The level claims to have met `prec` and has not. No basis excuses that."""
    bad = Sufficiency(accuracy=False, multiplicity=False)
    assert verdict(res(lvl(X * (1 + 10 * P))), [X], bad) == "shortfall-silent"


def test_a_reported_shortfall_whose_bar_does_not_contain_the_error_is_dishonest():
    bad = Sufficiency(accuracy=False, multiplicity=False)
    assert verdict(res(lvl(X * (1 + 10 * P), bar=5 * P * X)), [X], bad) == "bar-violated"


def test_a_level_with_no_bar_at_all_has_not_met_the_request():
    """NaN is not a bound. A NaN-bar level that is accurate still passes -- it is within `prec`
    -- but one that is not cannot be graded as honest."""
    assert verdict(res(Level(lam=X, mult=1)), [X]) == "pass"
    assert verdict(res(Level(lam=X * (1 + 10 * P), mult=1)), [X]) == "bar-violated"


# --- declining -----------------------------------------------------------------------------------

def test_a_decline_with_a_reason_is_its_own_verdict():
    assert verdict(res(declined=True, reason="slope never settled"), [X]) == "declined"


def test_a_decline_with_no_reason_is_a_failure():
    assert verdict(res(declined=True, reason="  "), [X]) == "decline-unreasoned"


# --- the acceptance set --------------------------------------------------------------------------

def test_the_acceptable_verdicts_are_exactly_these():
    assert ACCEPTABLE == {"pass", "declined", "b-excused", "merge-excused", "wrong-total-excused"}


@pytest.mark.parametrize("v", ["b-fail", "shortfall-silent", "bar-violated", "spurious",
                               "missed", "wrong-total", "merge-above-licence",
                               "decline-unreasoned"])
def test_no_failure_verdict_is_acceptable(v):
    assert v not in ACCEPTABLE


def test_the_score_carries_what_a_report_needs():
    s = score_cell(res(lvl(X), lvl(2 * X)), np.array([X, 2 * X]), P, OK)
    assert s.n_true == 2 and s.found_total == 2
    assert s.worst_rel_err == pytest.approx(0.0, abs=1e-15)
    assert s.acceptable


def test_a_collapsed_level_that_DECLARES_its_width_and_undercounts_is_a_wrong_total():
    """The collapse defect as `solve` actually reports it: one level, `grouping_ambiguous`, whose
    declared width spans the whole cluster, carrying multiplicity 1 where three eigenvalues lie.
    Its members are inside what it declared, so the eigenvalues were not MISSED -- the search
    found the neighbourhood -- they were dropped from the count. A diagnosis that says `missed`
    points at the bracketer when the defect is in the multiplicity stage."""
    members = [X, X * (1 + 0.96e-4), X * (1 + 1.92e-4)]
    collapsed = Level(lam=X, mult=1, delta_res=1e-6 * X, grouping_ambiguous=True,
                      wide_bracket=(X * (1 - 1e-4), X * (1 + 3e-4)))
    assert collapsed.width > 0, "the premise: the level declares a width"
    assert verdict(res(collapsed), members, prec=1e-4) == "wrong-total"
