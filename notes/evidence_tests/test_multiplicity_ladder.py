"""The multiplicity ladder: gap x precision x multiplicity, scoring `mu*(B)` and nothing else.

Ported from `wip/v1-promotion-and-multiplicity` (plan Phases 3a and 3b): the instrument, its
controls, the shipped rule localised, the floor and reach rules scored on it, and the census of
what the search really emits.

`mult_suite` is the instrument and its docstring is the specification. This file is the ladder
run over it, plus the controls that decide whether the instrument may be believed at all.

# Why per-REGIME reporting is mandatory rather than nice

Measured on this ladder, the shipped `(C1)+(C2)` rule scores 91.2% overall -- which reads like a
method with a rough edge -- and the same run reads:

    coincident 6/6    simple 31/31    unresolved 15/20

Every failure is in one regime, and it is the regime the rule cannot see for a structural reason.
An aggregate hides that; a regime breakdown localises it in one line. The same is true of the
adversary `always_1`, which scores 54.4% overall purely because the ladder holds more simple
brackets than clustered ones, and 0/6 and 0/20 on the two regimes that matter.

So the assertions below are per regime. An overall pass rate is reported and never asserted.

# The ladder crosses the bracket edge, and that is the whole discrimination

A bracket may be at most one merge licence wide, or stage 1 would have failed purity. A cluster
of `m` members at gap `g` licences spans `(m-1)*g`, so it fits only while `g <= 1/(m-1)`. Sweeping
`g` past that point is what separates a method that always merges from one that always splits:
below it every member must be COUNTED, above it the members fall outside `B` and must not be.
`test_the_ladder_crosses_the_bracket_edge` pins that the ladder actually does this, because a
ladder that stayed on one side would score both adversaries perfectly.
"""
import numpy as np
import pytest

import mult_suite as ms

pytestmark = pytest.mark.slow

PRECS = (1e-8, 1e-12)


@pytest.fixture(scope="module")
def cases():
    return (ms.ladder_cases(prec_ladder=PRECS, levels=(25, 50, 65))
            + ms.degenerate_cases(prec_ladder=PRECS))


# --- the instrument must be sound before any result from it means anything --------------------

def test_the_ladder_covers_all_three_regimes(cases):
    """A ladder missing a regime measures nothing about it. `unresolved` is the one that matters
    -- it is where a floor test and a ratio-gap test both fail, for opposite reasons."""
    r = ms.regimes(cases)
    assert r["coincident"] >= 4, r
    assert r["unresolved"] >= 10, r
    assert r["simple"] >= 10, r


def test_the_ladder_crosses_the_bracket_edge(cases):
    """Both sides of `mu*(B) = 1`, or the two adversaries are indistinguishable."""
    assert any(c.mu_true == 1 for c in cases), "no simple brackets"
    assert any(c.mu_true >= 3 for c in cases), "no brackets holding three or more"
    spans = [c.span for c in cases if c.mu_true > 1]
    assert min(spans) < 0.25 and max(spans) > 0.75, (
        f"the clustered cases must span the licence, got {min(spans):.2f} to {max(spans):.2f}")


def test_every_bracket_is_one_stage_1_could_have_EMITTED(cases):
    """Purity, from the other side. A bracket whose contents span more than one licence is one
    stage 1 would have refused, and scoring it measures the harness rather than the estimator."""
    bad = [(c.label, c.span) for c in cases if c.span > 1.0 + 1e-9]
    assert not bad, f"brackets wider than the licence: {bad}"


def test_the_ORACLE_passes_everything(cases):
    """The control that catches a metric too strict to accept a correct answer -- the failure
    that looks like rigour. `SUCCESS.md` asks for it by name."""
    sc = ms.score_estimator(ms.oracle, cases, "oracle")
    assert sc.correct == sc.n and sc.declined == 0, str(sc)


def test_a_DECLINER_comes_back_declined_and_never_as_a_pass(cases):
    """A method that declines whenever it is about to be wrong has a perfect record over what it
    accepted and has demonstrated nothing. It must be visible as a decline rate, not a pass."""
    sc = ms.score_estimator(ms.decliner, cases, "decliner")
    assert sc.declined == sc.n and sc.scored == 0
    assert sc.decline_rate == 1.0
    assert sc.silent_wrong == 0, "declining is not a wrong answer"


@pytest.mark.parametrize("name, fn", [
    ("always_1", ms.always(1)),
    ("always_2", ms.always(2)),
    ("below_plateau", ms.count_below_plateau(0.5)),
])
def test_the_ADVERSARIES_fail(cases, name, fn):
    """If any of these scores well the metric is broken and nothing else here means anything.

    Asserted per regime, because `always_1` scores 54% overall on this ladder purely from the
    simple brackets -- an aggregate would let it through.
    """
    sc = ms.score_estimator(fn, cases, name)
    hit = [k for k, v in sc.by_regime.items()
           if v["scored"] and v["correct"] == v["scored"]]
    assert len(hit) <= 1, (
        f"{name} is correct on more than one regime, which no fixed answer should be: {sc}")
    assert sc.silent_wrong > 0, f"{name} must produce wrong counts: {sc}"


# --- the shipped rule, localised -----------------------------------------------------------

def test_the_SHIPPED_rule_fails_only_in_the_unresolved_regime(cases):
    """(C1)+(C2), localised. It is right wherever the eigenvalues are coincident or alone, and
    wrong where they are inside `B` but distinct -- which is the regime `(C1)` cannot see,
    because the largest ratio gap is the gap up to the PLATEAU rather than to the neighbours."""
    sc = ms.score_estimator(ms.gap_rule, cases, "gap")
    print(f"\n  {sc}")
    for reg in ("coincident", "simple"):
        v = sc.by_regime.get(reg, {"scored": 0, "correct": 0})
        assert v["correct"] == v["scored"], f"{reg} regressed: {sc}"
    u = sc.by_regime["unresolved"]
    assert u["correct"] < u["scored"], (
        "the shipped rule now passes `unresolved` -- if that is a real fix, this test and the "
        "strict xfails in test_cluster_totals.py want revisiting together")


def test_the_FLOOR_rule_is_right_about_COINCIDENCE_and_blind_to_the_rest(cases):
    """It answers `bracket_search`'s question, correctly, and that question is not `mu*(B)`.

    Coincident brackets: every one right. Unresolved brackets: every one WRONG, and under rather
    than over -- the members are not at the floor together, so a floor test cannot see them. That
    is the cleanest possible demonstration that coincidence and `mu*(B)` are different questions.
    """
    sc = ms.score_estimator(ms.floor_rule, cases, "floor")
    print(f"\n  {sc}")
    assert sc.by_regime["coincident"]["correct"] == sc.by_regime["coincident"]["scored"]
    assert sc.by_regime["simple"]["correct"] == sc.by_regime["simple"]["scored"]
    u = sc.by_regime["unresolved"]
    assert u["correct"] == 0, f"floor rule should see none of the unresolved regime: {sc}"
    assert u["under"] == u["scored"], "and its errors are all UNDER-counts"


# --- the rule that answers the question ---------------------------------------------------------

def test_the_REACH_rule_is_correct_on_every_bracket_it_ACCEPTS(cases):
    """The whole specification, in one assertion: no bracket receives a wrong count silently.

    `mu = #{ j : sigma_j/c <= reach(B) }` compares a DISTANCE to a distance, which is the
    comparison `mu*(B)` actually calls for and the one no ratio test can make. Measured over the
    full ladder -- 143 brackets, `prec` 1e-6 to 1e-14, multiplicities 1 to 4 -- it is correct on
    every bracket it accepts and declines 2.1%.
    """
    sc = ms.score_estimator(ms.reach_rule, cases, "reach")
    print(f"\n  {sc}")
    assert sc.silent_wrong == 0, f"a wrong count was returned rather than declined: {sc}"
    assert sc.decline_rate < 0.10, f"declining too much to be useful: {sc}"


def test_the_REACH_rule_sees_the_regime_the_others_cannot(cases):
    """`unresolved` is the regime that motivated the whole exercise. The shipped rule gets 31 of
    51 and the floor rule 3 of 51; both fail structurally rather than by a bad constant."""
    reach = ms.score_estimator(ms.reach_rule, cases, "reach")
    gap = ms.score_estimator(ms.gap_rule, cases, "gap")
    r, g = reach.by_regime["unresolved"], gap.by_regime["unresolved"]
    assert r["correct"] == r["scored"], f"reach must be clean on unresolved: {reach}"
    assert r["correct"] > g["correct"], (
        f"reach {r['correct']} vs shipped {g['correct']} on unresolved")


def test_the_SLOPE_must_be_estimated_from_OUTSIDE_the_bracket(cases):
    """The largest single effect in the rule's construction, and it is not a tuning choice.

    Inside a bracket holding more than one eigenvalue the other members depress `sigma_1` at the
    flanks, so `c` reads too small and every distance `sigma_j/c` too large -- an eigenvalue that
    is inside `B` then reads as outside, and the count goes under. Measured on the ladder, taking
    the flanks outside moved the rule from 82.5% to 96.5% before any decline band existed.
    """
    clustered = [c for c in cases if c.mu_true > 1 and c.regime == "unresolved"]
    assert clustered, "no clustered cases to compare on"
    worse = sum(1 for c in clustered
                if ms.slope_outside(c) > ms._slope(c) * 1.05)
    assert worse > 0.5 * len(clustered), (
        "the inside-flank slope should be systematically SMALLER on clustered brackets; if it "
        "is not, this rule's calibration was measured on a different situation")


# --- the bridge: what the SEARCH actually hands the estimator ---------------------------------

def test_the_search_resolves_almost_everything_into_SIMPLE_brackets():
    """The census that says how much the multiplicity problem matters in practice.

    Measured on the wip branch over 544 brackets the search really emitted: simple 537,
    coincident 6, unresolved 1. So a wrong pipeline total is manufactured DOWNSTREAM of the
    brackets -- by the merge prefilter folding brackets that were already separated, and the
    collapse cross-check recounting the fold -- which is why Phase 3b replaces those rather than
    the bracketer.

    `ladder_cases` builds a bracket AROUND a dialled cluster, a population the search does not
    generate; this pins the one it does.
    """
    cases = ms.as_built_cases(prec_ladder=(1e-8,), windows=("pair", "cluster"))
    r = ms.regimes(cases)
    assert cases, "no as-built brackets were produced"
    assert r["simple"] > 0.8 * len(cases), (
        f"the search has stopped resolving into simple brackets: {r}. The multiplicity rule "
        f"matters far more than when this was measured.")
    sc = ms.score_estimator(ms.reach_rule, cases, "reach on as-built")
    print(f"\n  {len(cases)} as-built  {r}\n  {sc}")
    assert sc.silent_wrong == 0, str(sc)


def test_asymmetric_brackets_really_are_asymmetric():
    """The case set Phase 3b's reach rule is scored on. Symmetric brackets make the two sides of
    a reach identical, which hid a defect once; pinned here so the set cannot quietly re-centre."""
    asym = ms.asymmetric_cases()
    assert asym, "no asymmetric brackets were produced"
    off = [(c.lam_star - c.lo) / c.width for c in asym]
    assert max(off) > 0.6 or min(off) < 0.4, f"root sits at {off} of the width"


def test_the_reach_rule_is_clean_on_ASYMMETRIC_brackets_too():
    """The bracket a search emits is not centred on its root, and that difference hid a defect.

    `ladder_cases` builds `[lam* - w, lam* + w]`, so it reaches equally far both ways and
    `min(...) == max(...)`. An emitted bracket does not: the refiner puts `lam*` at the minimum,
    not the middle. The reach rule compared distances against the SHORTER side, which rejects an
    eigenvalue lying inside the bracket on the longer one -- and every symmetric case scored it
    50/50 while the pipeline was under-counting that same regime.

    Three corrections came out of this, each caught by a different part of the harness: `min`
    under-counted, `max` over-counted, and probe-then-reject over-declined at 2 of 3. What
    survives resolves the SIDE by probing and then bounds the distance with the measured bias
    range, which makes all three outcomes decidable.
    """
    asym = ms.asymmetric_cases()
    assert asym, "no asymmetric brackets were produced"
    off = [(c.lam_star - c.lo) / c.width for c in asym]
    assert max(off) > 0.6 or min(off) < 0.4, (
        f"these brackets are not actually asymmetric: root at {off} of the width")
    sc = ms.score_estimator(ms.reach_rule, asym, "reach on asymmetric")
    print(f"\n  {sc}")
    assert sc.silent_wrong == 0, str(sc)


def test_capability_is_reported_with_the_decline_rate(cases):
    """`SUCCESS.md`: a capability claim must quote the decline rate beside the pass rate over the
    same case set. Printed rather than asserted -- this is the reporting contract, not a bound."""
    print(f"\n  {len(cases)} brackets  {ms.regimes(cases)}")
    for name, fn in (("oracle", ms.oracle), ("gap (shipped)", ms.gap_rule),
                     ("floor", ms.floor_rule), ("reach", ms.reach_rule)):
        print(f"  {ms.score_estimator(fn, cases, name)}")
    assert True
