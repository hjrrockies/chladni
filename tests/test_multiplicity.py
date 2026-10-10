"""Stage 3: (C1), (C2), and the scorer that can tell resolving from guessing.

`search.md` section 5 Phase C and section 2.3; `notes/landscape_findings.md` sections 9.3, 9.4
and 27.2.

The scorer tests come first on purpose. Section 27.2 measured the previous metric REWARDING a
method for inflating multiplicities instead of resolving levels, so nothing about a multiplicity
rule can be believed until the thing judging it has been broken on purpose.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import level_controls as lc
import level_score as ls
import rect_suite as rs
import chladni.mps.multiplicity as M
from candidates_v1 import v1_brackets


PAIR = 5 * np.pi**2
TRUTH = np.array([2 * np.pi**2, PAIR, PAIR * (1 + 3e-7), 8 * np.pi**2])


# --- the scorer, broken on purpose first ------------------------------------------------------

@pytest.mark.parametrize("name", sorted(lc.ADVERSARIES))
def test_every_level_adversary_fails_by_its_named_criterion(name):
    fn, expected = lc.ADVERSARIES[name]
    s = ls.score_levels(fn(TRUTH, 1e-8), TRUTH, 1e-8, label=name)
    assert not s.ok, f"{name} must fail"
    assert s.verdict == expected, f"{name}: expected {expected}, got {s.verdict}"


def test_RECALL_ALONE_would_pass_every_adversary():
    """The measurement that forces this whole module. Recall-with-multiplicity is one number
    measuring two things, so it cannot tell "split the pair" from "guessed 2" -- and under the
    old metric the guess scored PERFECT while the honest answer was penalised."""
    for name, (fn, _) in lc.ADVERSARIES.items():
        s = ls.score_levels(fn(TRUTH, 1e-8), TRUTH, 1e-8)
        assert s.recall == 1.0, f"{name} scores a perfect recall"
        assert not s.ok, f"and is still rejected, by separation or multiplicity"


def test_separation_is_what_distinguishes_resolving_from_guessing():
    """`always_report_mult_2` never splits the pair and claims 2; `never_split` merges honestly.
    Recall cannot tell either from a correct answer. Separation and the multiplicity check can."""
    guess = ls.score_levels(lc.always_report_mult_2(TRUTH, 1e-8), TRUTH, 1e-8)
    merge = ls.score_levels(lc.never_split(TRUTH, 1e-8), TRUTH, 1e-8)
    lv, mult = np.unique(TRUTH, return_counts=True)
    honest = ls.score_levels(list(zip(lv, mult)), TRUTH, 1e-8)

    assert honest.ok and honest.separation == 1.0
    assert guess.n_mult_over > 0, "the guess inflates"
    assert merge.separation < 1.0, "the merge fails to resolve"
    assert guess.recall == merge.recall == honest.recall, "recall cannot separate the three"


def test_the_matching_tolerance_is_keyed_to_the_CASE():
    """A fixed `rtol = 1e-6` is what let ONE reported location claim TWO distinct truth
    entries. Keyed to the case's own tightest separation it cannot."""
    rtol = ls.match_tolerance(TRUTH, 1e-8)
    d_rel = 3e-7
    assert rtol == pytest.approx(ls.MATCH_FRACTION * d_rel, rel=0.05)
    assert rtol < 1e-6, "tighter than the fixed tolerance it replaces"
    # one location cannot claim both members of the pair
    s = ls.score_levels([(PAIR, 1), (2 * np.pi**2, 1), (8 * np.pi**2, 1)], TRUTH, 1e-8)
    assert s.n_matched == 3 and s.n_missed == 1


def test_the_tolerance_never_falls_below_prec():
    """At an exact degeneracy there is no separation to key to, so `prec` is the floor."""
    exact = np.array([PAIR, PAIR, PAIR])
    assert ls.match_tolerance(exact, 1e-8) == pytest.approx(1e-8)


# --- (C1) and (C2) as rules -------------------------------------------------------------------

def _flat(values):
    """A tensions callable returning a fixed vector, for the rule-level tests."""
    return lambda x: np.asarray(values(x) if callable(values) else values, dtype=float)


def test_C2_uses_PROMINENCE_and_not_a_bare_comparison():
    """Roughness minima reach `1 + 2.4e-10` while a true well reaches `plateau/eps`, worst case
    72. A bare `<` counts the first as a branch; `P = 2` does not."""
    def rough(x):
        return np.array([1e-9, 1e-9 * (1.0 if x == 0.0 else 1.0000000002), 1.0])
    r = M.multiplicity(rough, 0.0, wide_bracket=(-1.0, 1.0))
    assert r.c1_read == 2, "the gap test groups them"
    assert r.mu == 1, "and prominence rejects the second"
    assert r.c2_rejected


def test_C2_takes_the_longest_PREFIX_not_a_walk_down():
    """The multiplicity is a contiguous block `1..mu` by definition. Checking only the top index
    and decrementing accepts a block with a hole in it, because the property is not monotone."""
    def holed(x):
        at = np.array([1e-15, 1e-15, 1e-15, 1.0])
        out = np.array([1e-9, 1e-15, 1e-9, 1.0])       # index 1 does NOT dip
        return at if x == 0.0 else out
    r = M.multiplicity(holed, 0.0, wide_bracket=(-1.0, 1.0))
    assert r.mu == 1, f"the block stops at the hole, got {r.mu}"


# --- the wide bracket -------------------------------------------------------------------------

def test_the_wide_bracket_reaches_past_the_flat_bottom():
    """Inside `eps/c` the higher branches have not moved and (C2) reads the floor against
    itself. That is why an earlier attempt's multiplicity test was reading noise."""
    xs = np.array([0.0, 0.1, 0.9, 1.0, 1.1, 1.9, 2.0])
    lo, hi = M.wide_bracket_for(xs, 1.0, sigma_1=0.05, c=1.0, factor=2.0)
    assert 1.0 - lo >= 0.1 and hi - 1.0 >= 0.1


def test_the_wide_bracket_falls_back_rather_than_failing():
    """A side with nothing far enough takes the outermost sample. Too narrow makes (C2)
    conservative -- it under-reads the multiplicity -- which is the safe direction, since
    over-reading is the double count."""
    xs = np.array([0.999, 1.0, 1.001])
    wb = M.wide_bracket_for(xs, 1.0, sigma_1=0.5, c=1.0, factor=2.0)
    assert wb == (0.999, 1.001)
    assert M.wide_bracket_for(np.array([1.0, 2.0]), 0.5, 1e-15, 1.0) is None, "nothing to the left"


def test_the_factor_is_above_the_cone_bound():
    """On a cone, `sigma` at `f*eps/c` is `eps*sqrt(1+f**2)`, so a genuine branch shows
    prominence `sqrt(1+f**2)` and clearing `P` needs `f > sqrt(P**2 - 1)`."""
    assert np.sqrt(1 + M.WIDE_FACTOR**2) > M.PROMINENCE


# --- the reach rule, as `solve` runs it -------------------------------------------------------
#
# The gap rule's real-pencil reads (`reads_correctly_off_a_smooth_spectrum`, `exact_degeneracy_
# reads`) and its (C1) unit tests (`C1_is_a_RATIO_GAP`, `single_branch_reads_one`) were retired
# with `rule="gap"` on 2026-10-09; the starved-basis (C2) measurements moved to
# `notes/evidence_tests/test_multiplicity_evidence.py`.

#: The reach rule's bounds as `solve` ships them: the cache-slope set, `(0.61, 2.21, 0.61, 2.21)`.
#: `multiplicity(rule="reach")` without `reach_bounds` falls back to the RETIRED outside-slope set
#: (`notes/test_inventory_2026-10-09.md`, "two source issues"), so every read here passes these.
SHIPPED_BOUNDS = (M.COUNT_BELOW_CACHE, M.REJECT_ABOVE_CACHE, M.BIAS_LO_CACHE, M.BIAS_HI_CACHE)


def _reach_read(obj, lam, sigma, bracket, prec, counter=None):
    """Stage 3 as `solve` runs it: the reach rule on the EMITTED bracket, with the wall slope read
    from the two cached samples nearest the root (`search._slope_from_cache`) and the shipped
    bounds."""
    from chladni.mps.search import _samples, _slope_from_cache

    lams, svals = _samples(obj)
    c = _slope_from_cache(lams, svals, lam, floor=sigma)
    assert c is not None and c > 0, f"no wall slope in the cache at {lam}"
    return M.multiplicity(lambda z: obj.tensions(obj.from_lam(z)), lam, bracket=bracket, c=c,
                          prec=prec, rule="reach", reach_bounds=SHIPPED_BOUNDS, counter=counter)


def test_the_shipped_bounds_are_the_cache_slope_set():
    """The premise of every reach read below, pinned so a recalibration has to confront it."""
    assert SHIPPED_BOUNDS == (0.61, 2.21, 0.61, 2.21)


@pytest.mark.slow
def test_the_reach_read_at_a_LOCATED_ROOT_counts_the_degeneracy():
    """`sigma_1` read at a REPORTED root is not the floor -- the root sits within `prec*lam` of
    the true minimum, so on a well-resolved pencil `sigma` there is orders above `eps`.

    Measured on the disk at `prec = 1e-8` under the retired gap rule: the reported root sat ~3e-09
    from the minimum, `sigma` there was 6.2e-10 against a floor of 7.2e-16, the wide bracket's
    flanks straddled asymmetrically, and two of seven double levels read 1. That rule needed
    `prec` to widen its bracket. The reach rule compares DISTANCE to DISTANCE -- `sigma_j/c`
    against the emitted bracket -- so the root's own error is inside the comparison rather than
    corrupting it.

    Re-posed on the reach rule (2026-10-09 prune) as `solve` runs it: stage 1's bracket, the
    refiner's root, the cache slope, the shipped bounds. Every disk double reached reads 2, or
    declines; a SILENT under-count is the failure this guards.
    """
    import hard_roster as hr
    from chladni.mps.bracket import bracket_interval
    from chladni.mps.refine import refine

    prec = 1e-8
    case = hr.ROSTER["disk"]()
    obj = bs.objective_for(case, prec)
    res = bracket_interval(obj, prec=prec)
    located = []
    for b in res.brackets:
        lo, hi = obj.in_lam_bracket(b.lo, b.hi)
        hint = obj.in_lam(b.k_hint) if np.isfinite(b.k_hint) else None
        r = refine(obj, lo, hi, prec=prec, lam_hint=hint,
                   interior=(obj.in_lam(b.lo), obj.in_lam(b.hi)))
        if r.accepted:
            located.append((r, (lo, hi)))

    lv, mult = np.unique(case.truth, return_counts=True)
    doubles = [float(x) for x, m in zip(lv, mult) if m == 2]
    assert doubles, "the disk window must hold a degenerate level"

    reads = []
    for lam in doubles:
        near = [(r, br) for r, br in located if abs(r.lam - lam) / lam < 1e-6]
        if not near:
            continue
        r, br = min(near, key=lambda p: p[0].sigma)
        m = _reach_read(obj, r.lam, r.sigma, br, prec)
        reads.append((lam, m.mu, m.ambiguous))
    print(f"\n  disk doubles, reach rule at the located root: {reads}")
    assert reads, "no degenerate level was reached"
    silent = [x for x in reads if x[1] != 2 and not x[2]]
    assert not silent, f"read wrong and undeclined: {silent}"
    assert any(mu == 2 for _, mu, _ in reads), "and the rule does count a double"


# --- it spends nothing ------------------------------------------------------------------------

@pytest.mark.slow
def test_it_spends_NO_EVALUATIONS_on_a_shared_cache():
    """Every point it reads was paid for by the search. A multiplicity stage that quietly cost
    a GSVD per level would change the cost model it is scored under.

    Re-posed on the reach rule (2026-10-09 prune). The reach rule MAY spend two side probes, but
    only for a branch whose distance lands in the band between "inside whichever side" and
    "outside both ways". At an exact degeneracy every partner sits at distance ~0 and the next
    branch far outside, so the read must cost nothing -- and read 3.
    """
    prec = 1e-8
    (case,) = [c for c in rs.family_D() if "mult=3" in c.label]
    obj = bs.objective_for(case, prec)
    v1_brackets()(obj.tensions, obj.lo, obj.hi, prec=prec, objective=obj)
    lam = float(np.unique(case.truth)[0])
    s1 = obj.tension(obj.from_lam(lam))
    bracket = (lam * (1 - prec), lam * (1 + prec))

    before = obj.n_evals
    r = _reach_read(obj, lam, s1, bracket, prec, counter=obj)
    assert obj.n_evals == before, f"spent {obj.n_evals - before} evaluations"
    assert r.evaluations == 0
    assert r.mu == 3 and not r.ambiguous


@pytest.mark.slow
def test_a_real_degeneracy_is_NOT_bit_identical():
    """`sigma_1` and `sigma_2` at a genuine degeneracy are numerically coincident but distinct
    -- 6.1e-16 against 1.0e-15 at the disk's `j_1,1`. So an equality test would fail on real
    data, and the surrogate, whose `mult` produces exactly equal values, must not be used to
    validate one."""
    import hard_roster as hr

    case = hr.ROSTER["disk"]()
    obj = bs.objective_for(case, 1e-8)
    lv, mult = np.unique(case.truth, return_counts=True)
    lam = float(lv[np.argmax(mult)])
    v = obj.tensions(lam)
    assert v[0] != v[1], "coincident, not identical"
    assert v[1] / v[0] < M.GAP_RATIO, "and within the multiplet by the gap test"
