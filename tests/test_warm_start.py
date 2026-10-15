"""Warm start: `Eigenproblem.solve(n, plan=plan, near=prev)`.  (2026-10-13)

douse solves a sequence of nearby shapes on one plan. Cold, each solve searches the prefix
window up to the Weyl estimate of level `n + HEADROOM`; warm, it searches only
`[min(0.98 FK, lam_1 - h_1), lam_n + h_n]` around the previous values, `h` a third of the local
mean Weyl spacing (lappy's tracking rule). Measured at douse's N = 6, k = 2 optimum after a 1e-3
move: 84 tension evaluations against 177, 2.4x faster, the same values to 1.2e-13
(`notes/warm_start.py`).

Dropping the headroom drops what the count audit leans on, so completeness rests on CONTINUITY,
checked on every warm solve, with a cold solve on any failure (decided with the user, 2026-10-13):
the window is a prefix (it starts below Faber-Krahn, so nothing is missed below it); no level is
unmet or count-uncertain; at least `n` values are found; none moved more than its `h` (no index
shift); the top level sits clear of the window's edge (no level cut at `b`); the audit is not
short or excess. And a cold solve every `WARM_COLD_EVERY` warm solves, as a backstop.
"""
import dataclasses

import numpy as np
import pytest

from chladni import eigenproblem as EP
from chladni.eigenproblem import WARM_COLD_EVERY, Eigenproblem, _warm_verdict
from chladni.geometry import rect
from chladni.reference import rect_eigs

PREC = 1e-10
pytestmark = pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")


def _truth(L, H, n=20):
    return np.asarray(rect_eigs(n, L, H), float)


def _within_bounds(spec, truth):
    for x, b in zip(spec.lam, spec.bound):
        assert np.min(np.abs(truth - x)) <= b + 1e-14 * x, (x, b)


@pytest.fixture
def assemblies(monkeypatch):
    import chladni.mps.pencil as P
    n = {"calls": 0}
    real = P.assemble

    def counting(*a, **k):
        n["calls"] += 1
        return real(*a, **k)
    monkeypatch.setattr(P, "assemble", counting)
    return n


# --- the acceptance rule, as a pure function ---------------------------------------------------

def _verdict(**kw):
    base = dict(found=np.array([10.0, 20.0, 30.0]), prev=np.array([10.1, 19.9, 30.2]), n=3,
                b=34.0, h=np.array([3.0, 3.0, 3.0]), top=30.0, declined=False, unmet=0,
                uncertain=0, audit="ok")
    base.update(kw)
    return _warm_verdict(**base)


def test_a_clean_warm_solve_is_accepted():
    assert _verdict() is None


@pytest.mark.parametrize("kw, word", [
    (dict(declined=True), "declined"),
    (dict(unmet=1), "unmet"),
    (dict(uncertain=1), "uncertain"),
    (dict(found=np.array([10.0, 20.0])), "found"),
    (dict(found=np.array([10.0, 25.0, 30.0])), "moved"),
    (dict(top=33.5), "edge"),
    (dict(audit="short"), "audit"),
    (dict(audit="excess"), "audit"),
])
def test_each_continuity_check_refuses(kw, word):
    why = _verdict(**kw)
    assert why is not None and word in why, why


def test_an_unaudited_count_is_not_a_refusal():
    """With few levels below `b` the audit may have nothing to say; continuity decides."""
    assert _verdict(audit="unaudited") is None


# --- end to end, against closed forms -----------------------------------------------------------

@pytest.mark.slow
def test_a_small_move_solves_warm_and_cheaper(assemblies):
    ep0 = Eigenproblem(rect(1.0, 0.7))
    plan = ep0.plan(4, prec=PREC)
    prev = ep0.solve(4, prec=PREC, plan=plan)
    assert not prev.warm and prev.warm_depth == 0

    moved = Eigenproblem(rect(1.003, 0.7))
    n0 = assemblies["calls"]
    cold = moved.solve(4, prec=PREC, plan=plan)
    n_cold = assemblies["calls"] - n0
    n0 = assemblies["calls"]
    warm = moved.solve(4, prec=PREC, plan=plan, near=prev)
    n_warm = assemblies["calls"] - n0

    assert warm.warm and warm.warm_depth == 1 and warm.warm_note is None, warm.warm_note
    _within_bounds(warm, _truth(1.003, 0.7))
    np.testing.assert_allclose(warm.unfolded[:4], cold.unfolded[:4], rtol=PREC)
    assert n_warm < n_cold, (n_warm, n_cold)


@pytest.mark.slow
def test_a_crossing_comes_back_sorted():
    """Near the square, lam(2,1) and lam(1,2) swap order as H passes 1. The sorted values move
    continuously, so the warm solve accepts and returns them ascending."""
    ep0 = Eigenproblem(rect(1.0, 0.995))
    plan = ep0.plan(3, prec=PREC)
    prev = ep0.solve(3, prec=PREC, plan=plan)
    spec = Eigenproblem(rect(1.0, 1.005)).solve(3, prec=PREC, plan=plan, near=prev)
    assert spec.warm, spec.warm_note
    assert np.all(np.diff(spec.unfolded) >= 0)
    _within_bounds(spec, _truth(1.0, 1.005))


@pytest.mark.slow
def test_the_squares_double_at_n_stays_whole():
    ep0 = Eigenproblem(rect(1.0, 1.0))
    plan = ep0.plan(2, prec=PREC)
    prev = ep0.solve(2, prec=PREC, plan=plan)
    spec = Eigenproblem(rect(1.002, 1.002)).solve(2, prec=PREC, plan=plan, near=prev)
    assert spec.warm, spec.warm_note
    assert spec.count == 3 and spec.mult[-1] == 2
    _within_bounds(spec, _truth(1.002, 1.002))


@pytest.mark.slow
def test_a_large_move_falls_back_to_cold():
    ep0 = Eigenproblem(rect(1.0, 0.7))
    plan = ep0.plan(4, prec=PREC)
    prev = ep0.solve(4, prec=PREC, plan=plan)
    spec = Eigenproblem(rect(1.3, 0.7)).solve(4, prec=PREC, plan=plan, near=prev)
    assert not spec.warm and spec.warm_depth == 0 and "moved" in spec.warm_note, spec.warm_note
    _within_bounds(spec, _truth(1.3, 0.7))


@pytest.mark.slow
def test_the_cold_cadence_and_a_foreign_plan_force_cold():
    ep0 = Eigenproblem(rect(1.0, 0.7))
    plan = ep0.plan(4, prec=PREC)
    prev = ep0.solve(4, prec=PREC, plan=plan)
    moved = Eigenproblem(rect(1.003, 0.7))

    stale = dataclasses.replace(prev, warm_depth=WARM_COLD_EVERY)
    spec = moved.solve(4, prec=PREC, plan=plan, near=stale)
    assert not spec.warm and "cadence" in spec.warm_note, spec.warm_note

    other = dataclasses.replace(prev, plan=Eigenproblem(rect(1.0, 0.7)).plan(4, prec=1e-8))
    spec = moved.solve(4, prec=PREC, plan=plan, near=other)
    assert not spec.warm and "plan" in spec.warm_note, spec.warm_note


def test_near_without_a_plan_uses_the_previous_ones():
    """`near` carries its plan, so `plan=None` with `near` does not re-plan."""
    assert "near" in Eigenproblem.solve.__code__.co_varnames
    assert WARM_COLD_EVERY == 10
    assert hasattr(EP, "_warm_window")


@pytest.mark.slow
def test_a_tracked_warm_solve_on_caller_windows_matches_a_cold_solve_and_costs_less():
    """The caller bounds each level's move; the solve refines a window per level instead of
    searching the prefix. Same values as a cold solve on the moved shape, far fewer
    evaluations than the prefix warm solve (2026-10-14)."""
    from chladni.geometry import rect
    ep0 = Eigenproblem(rect(1.2, 0.8))
    plan, prev = ep0.plan_and_solve(3, prec=1e-10)
    moved = rect(1.2, 0.8 * 1.002)
    ep = Eigenproblem(moved)
    cold = ep.solve(3, prec=1e-10, plan=plan)
    prefix = ep.solve(3, prec=1e-10, plan=plan, near=prev)
    h = 0.05 * prev.unfolded[:3]                                  # a generous move bound
    tracked = ep.solve(3, prec=1e-10, plan=plan, near=prev, windows=h)
    assert tracked.warm and prefix.warm, (tracked.warm_note, prefix.warm_note)
    np.testing.assert_allclose(tracked.unfolded, cold.unfolded, rtol=1e-9)
    assert tracked.achieved and not tracked.flags
    assert tracked.diagnostics.evaluations < 0.6 * prefix.diagnostics.evaluations, (
        tracked.diagnostics.evaluations, prefix.diagnostics.evaluations)


@pytest.mark.slow
def test_a_window_that_lost_its_level_goes_cold_and_says_why():
    """A half-width too small for the move: no well in the window, so the warm solve is refused
    with a `track:` note and the answer is the cold one."""
    from chladni.geometry import rect
    plan, prev = Eigenproblem(rect(1.2, 0.8)).plan_and_solve(3, prec=1e-10)
    rec = Eigenproblem(rect(1.2, 0.8 * 1.05)).solve(3, prec=1e-10, plan=plan, near=prev,
                                                   windows=1e-6 * prev.unfolded[:3])
    assert not rec.warm and rec.warm_note.startswith("track:"), rec.warm_note
    assert rec.achieved
