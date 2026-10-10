"""THE STANDING ACCEPTANCE GATE. Run before tagging anything: `pytest -m gate`.

`CHLADNI_V1.md` section 5.5. About seven minutes.

The crossover sweep and the degeneracy class, on three rectangle windows, at every precision in
`bracket_score.PREC_LADDER_FULL`. Coverage and purity are hard; spurious, fineness, cost and the
DECLINE RATE are reported.

# Why a standing gate, and why it must be fixed in advance

V1 section 7 is unambiguous: **every claim measured on the case that motivated it was
overstated, without exception** -- "2.2-2.7x cheaper" became 1.77-1.79x, "coverage 1.0 at every
precision" became lost eigenvalues on 3 of 7 cases -- until a case set fixed in advance became
the gate. Twice the narrowness cost COVERAGE rather than only credibility: a version passed
every gate that existed and lost three eigenvalues on a window nobody had run.

So this file's case set is fixed, and it is widened only on its own merits -- never because
widening makes a change look good.

# The gate asserts its own coverage

A gate that quietly stops covering the case it was widened for is worse than no gate, because it
still reads green. `test_the_gate_still_covers_the_multiplicity_3_level` checks that `50*pi^2`
is present in the widest window, does split under the perturbation, and is bracketed.

# Declining is reported beside passing

A method that declines whenever it is about to be wrong has a perfect record over what it
accepted and has demonstrated nothing. The decline rate is quoted over the same case set,
always.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import rect_suite as rs
from candidates_v1 import v1_brackets
from chladni.mps.refine import refine

pytestmark = [pytest.mark.gate]

BANDS = (0.5, 0.8, 1.3, 3.0)


def _score(case, prec):
    """One scored case, charged with the REAL refinement its brackets hand on."""
    return bs.run_case(v1_brackets(), case, prec, refine=refine)


def _require_preconditions(case, prec):
    """Both premises, checked BEFORE the case is scored. Skip with the reason if either fails.

    This is the third time the same lesson has arrived in this build: **a case that violates a
    precondition measures the case, not the method.** Tier 1 lost 13 of 66 cells to a surrogate
    floor that could not support `prec = 1e-14`; tier 2's floor check existed but was never
    called; and the gate's first run reported nine failures of which most were windows whose
    tension curve admits no slope bound at all.

    `assert_floor` asks whether the basis resolves the eigenfunctions. `assert_smooth` asks
    whether the curve BETWEEN them supports an exclusion argument. A search whose method rests
    on a slope bound will decline on a curve that has none -- correctly -- so scoring it there
    measures the pencil.
    """
    ok, rows = rs.assert_floor(case, finest_prec=prec)
    if not ok:
        worst = max(r["sigma_mu"] for r in rows)
        pytest.skip(f"UNEXERCISABLE: the basis cannot support prec={prec:.0e} here -- worst "
                    f"sigma_mu {worst:.2e}, min headroom "
                    f"{min(r['headroom'] for r in rows):.2f}")
    smooth, hist = rs.assert_smooth(case, prec)
    if not smooth:
        pytest.skip(f"UNEXERCISABLE: the objective admits no slope bound here -- the largest "
                    f"observed secant runs {['%.3g' % h for h in hist]} across doublings and "
                    f"does not settle, against a wall slope of about 0.7 on the same curve")


# --- the crossover sweep, the whole cross product -----------------------------------------------

@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("prec", bs.PREC_LADDER_FULL)
@pytest.mark.parametrize("window", ["pair", "cluster", "wide"])
def test_the_crossover_sweep(window, prec, band):
    """Two bands that must MERGE and two that must SPLIT, at every precision, on every window.

    The crossover is located per precision rather than sampled three times: section 29 recorded
    results IDENTICAL at 1e-8, 1e-11 and 1e-14 because one global separation ladder left the
    requirement unchanged at all three, so the sweep measured cost while claiming capability.
    """
    a, b = rs.CROSSOVER_WINDOWS[window]
    L = 1.0 + rs.delta_for(band, prec, a, b)
    if L == 1.0:
        pytest.skip(f"UNEXERCISABLE: 1 + delta is not representable at band {band}, "
                    f"prec {prec:.0e} -- this cell of the sweep is empty")
    (case,) = rs.crossover_cases(prec, window=window, bands=(band,))
    _require_preconditions(case, prec)

    sep = bs.classify_separation(case.truth, prec)
    assert sep.exercisable, f"achieved ratio {sep.ratio:.4g}, no band"
    assert sep.band == pytest.approx(band), f"achieved {sep.ratio:.4g}, asked {band}"

    s = _score(case, prec)
    assert not s.declined, f"DECLINED: {s.decline_reason}"
    assert s.coverage == 1.0, f"missed {s.n_true - s.covered} of {s.n_true} at {sep.at}"
    assert s.pure, f"{s.impure} impure of {s.n_brackets} brackets"
    assert s.cost_ok, f"{s.per_eigenvalue:.0f} evaluations per eigenvalue"


@pytest.mark.parametrize("prec", bs.PREC_LADDER_FULL)
@pytest.mark.parametrize("mult", [2, 3, 4])
def test_exact_degeneracy_as_a_class(prec, mult):
    """Swept as a CLASS, not as one case. The crossover ladder varies a separation and never
    reaches zero, while `delta = 0` is the regime a shape optimization crosses whenever
    eigenvalues meet -- and it is where the isolation gate can never fire, because `sigma_2`
    sits on the floor beside `sigma_1`."""
    lam = {2: 25, 3: 50, 4: 65}[mult] * rs.PI2
    (case,) = [c for c in rs.family_D() if f"mult={mult}" in c.label]
    _require_preconditions(case, prec)
    s = _score(case, prec)
    assert not s.declined, f"DECLINED: {s.decline_reason}"
    assert s.coverage == 1.0, f"missed {s.n_true - s.covered} of {s.n_true}"
    assert s.pure
    lv = np.unique(case.truth)
    assert np.sum(np.isclose(lv, lam, rtol=1e-12)) == 1, "the level is present and is one level"
    assert np.sum(np.isclose(case.truth, lam, rtol=1e-12)) == mult, f"of multiplicity {mult}"


@pytest.mark.parametrize("prec", bs.PREC_LADDER_FULL)
@pytest.mark.parametrize("mult", [2, 3, 4])
def test_the_split_counterparts_of_every_degeneracy(prec, mult):
    """Every closed-form multiple gets a split counterpart at each precision, so the merge/split
    decision is exercised on MULTIPLETS and not only on pairs. A cluster of three with unequal
    gaps is measurably harder than a pair."""
    cases = [c for c in rs.split_degeneracy_cases(prec, bands=(1.3,))
             if f"split{mult}" in c.label]
    if not cases:
        pytest.skip("UNEXERCISABLE: the perturbation is not representable at this precision")
    _require_preconditions(cases[0], prec)
    s = _score(cases[0], prec)
    assert not s.declined, f"DECLINED: {s.decline_reason}"
    assert s.coverage == 1.0 and s.pure


# --- the gate asserts its own coverage ------------------------------------------------------------

def test_the_gate_still_covers_the_multiplicity_3_level():
    """The case the widest window exists for. Present, splits, and bracketed -- so the gate
    cannot quietly stop covering what it was widened to cover."""
    a, b = rs.WIDE_WINDOW
    lam = 50 * rs.PI2
    assert a < lam < b, "the level is inside the widest window"
    at_level = [(m, n) for m, n in rs.modes_in_window(a, b) if m * m + n * n == 50]
    assert sorted(at_level) == [(1, 7), (5, 5), (7, 1)], "multiplicity 3"

    prec = 1e-8
    (case,) = rs.crossover_cases(prec, window="wide", bands=(3.0,))
    members = np.sort([x for x in case.truth
                       if abs(x - lam) < 1e-4 * lam])
    assert members.size == 3 and np.unique(members).size == 3, "it does split"

    s = _score(case, prec)
    assert not s.declined and s.coverage == 1.0

    # And every member of that level is inside some bracket. Read the RETURNED list, not
    # `fn.last.brackets`: the latter holds the searcher's internal `k` view, and treating those
    # numbers as `lam` reported the level as loose when it was bracketed all along.
    fn = v1_brackets()
    obj = bs.objective_for(case, prec)
    raw = fn(obj.tensions, obj.lo, obj.hi, prec=prec, objective=obj)
    got = bs.to_lam_brackets(list(raw), obj.variable)
    for x in members:
        assert any(lo <= x <= hi for lo, hi in got), f"member {x!r} of the mult-3 level is loose"


# --- the report: coverage, purity, cost and the DECLINE RATE together --------------------------------

def test_the_gate_reports_its_decline_rate_beside_its_pass_rate():
    """Quoted over the SAME case set, always. A method that declines whenever it is about to be
    wrong has a perfect record over what it accepted and has demonstrated nothing."""
    rows, rough = [], []
    for window in ("pair", "cluster"):
        for prec in (1e-4, 1e-8, 1e-14):
            for band in (0.5, 1.3):
                a, b = rs.CROSSOVER_WINDOWS[window]
                if 1.0 + rs.delta_for(band, prec, a, b) == 1.0:
                    continue
                (case,) = rs.crossover_cases(prec, window=window, bands=(band,))
                if not rs.assert_floor(case, finest_prec=prec)[0]:
                    continue
                if not rs.assert_smooth(case, prec)[0]:
                    rough.append((window, prec, band))
                    continue
                rows.append(_score(case, prec))

    n = len(rows)
    passed = sum(1 for s in rows if s.ok)
    declined = sum(1 for s in rows if s.declined)
    cost = float(np.median([s.per_eigenvalue for s in rows]))
    print(f"\n  gate summary over {n} scored cases: pass {passed}/{n}, declined "
          f"{declined}/{n}, median {cost:.0f} evaluations per eigenvalue; "
          f"{len(rough)} cells unexercisable (objective admits no slope bound): {rough}")
    assert declined == 0, f"{declined} declines: {[s.decline_reason for s in rows if s.declined]}"
    assert passed == n
    assert cost < bs.COST_CAP_PER_EIG


def test_the_roughness_precondition_is_reported_not_hidden():
    """How many cells the objective cannot support, counted. If this number grows, the pencil
    got worse -- and a gate that skipped them silently would read green while covering less."""
    rough = []
    for window in ("pair", "cluster", "wide"):
        for band in (0.5, 1.3):
            (case,) = rs.crossover_cases(1e-8, window=window, bands=(band,))
            if not rs.assert_smooth(case, 1e-8)[0]:
                rough.append(f"{window}/b={band}")
    print(f"\n  unexercisable (no slope bound): {len(rough)} of 6 -- {rough}")
    assert len(rough) <= 2, f"the objective got rougher: {rough}"
    assert "pair/b=1.3" not in rough and "wide/b=1.3" not in rough, (
        "the two windows the gate most depends on must stay smooth")


def test_the_floor_precondition_holds_on_every_gate_case():
    """`CHLADNI_V1.md` section 5.6: assert the floor BEFORE believing any measurement. Three
    separate ways of getting the discretization wrong each degraded the objective by orders
    while looking healthy, and a search scored on a bad objective measures the objective."""
    for window in ("pair", "cluster"):
        for case in rs.crossover_cases(1e-8, window=window, bands=(1.3,)):
            ok, rows = rs.assert_floor(case)
            worst = max(r["sigma_mu"] for r in rows)
            assert ok, f"{case.label}: worst sigma_mu {worst:.2e} over {len(rows)} levels"
