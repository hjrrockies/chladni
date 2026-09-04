"""Stage 4: the composed pipeline, and what composition costs that the stages did not.

`docs/design/candidates.md` section 1 -- `solve` is the only mandatory seam, so it is where v1
and a method that does not decompose are compared on equal terms.

Every stage passed its own tests before this file existed, and composing them failed on every
exact degeneracy. That is the point of the file: seam defects are invisible from either side.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import level_score as ls
import rect_suite as rs
import v1_multiplicity as M
from objective import Declined, Objective
from v1_solve import solve, v1_solve

pytestmark = pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")


def _cone(lams, eps=1e-13, c_k=0.27, plateau=0.2, lo=15.0, hi=110.0, prec=1e-8):
    ks = np.sqrt(np.asarray(lams, dtype=float))

    def tensions(lam):
        d = np.sort(np.abs(ks - np.sqrt(float(lam))))
        return np.minimum(np.sqrt(eps**2 + (c_k * d) ** 2), plateau)

    return Objective(tensions, lo, hi, prec=prec, variable="lam")


# --- the seam ---------------------------------------------------------------------------------

def test_the_seam_returns_lam_mult_pairs():
    """`[(lam, mult), ...]`, which is what a monolithic candidate must also return."""
    lams = [2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]
    out = v1_solve()(_cone(lams).tensions, 15.0, 110.0, prec=1e-8, objective=_cone(lams))
    assert isinstance(out, list) and out
    for item in out:
        lam, mult = item
        assert lam > 0 and isinstance(mult, int) and mult >= 1


def test_a_decline_propagates_as_a_VALUE_not_an_empty_list():
    """An empty list looks like success. `Declined` is falsey and carries the reason."""
    def spiky(lam):
        return np.array([0.2 + 0.1 * np.sin(1e6 * lam**2), 0.3])

    obj = Objective(spiky, 15.0, 110.0, prec=1e-8, variable="lam")
    out = v1_solve()(obj.tensions, obj.lo, obj.hi, prec=1e-8, objective=obj)
    assert isinstance(out, Declined) and not out
    assert out.reason


def test_the_result_is_sorted_and_unfoldable():
    lams = [2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]
    res = solve(_cone(lams), prec=1e-8)
    got = [x.lam for x in res.levels]
    assert got == sorted(got)
    assert res.unfolded.size == sum(x.mult for x in res.levels)


# --- clause 2: accuracy is reported, not assumed -----------------------------------------------

def test_each_level_reports_the_accuracy_it_ACHIEVED():
    """`search.md` clause (2): within `prec`, OR report `delta_res` as the attained bound.
    There is no predicted-accuracy constant -- it is measured from the parabola fit."""
    res = solve(_cone([2 * np.pi**2, 5 * np.pi**2]), prec=1e-8)
    assert res.levels
    for x in res.levels:
        assert np.isnan(x.attained_prec) or x.attained_prec >= 0.0
    d = res.diagnostics
    assert d.prec_requested == 1e-8
    assert d.met_request == (not d.unmet)


def test_clause_3_reports_what_a_caller_cannot_otherwise_see():
    """`S` is unknown on every domain that matters, so the honesty clause is what stands in for
    truth on a real run: where the evaluations went, what the search declined, what the refiner
    rejected, and how many roots were folded together."""
    res = solve(_cone([2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]), prec=1e-8)
    d = res.diagnostics
    assert d.evaluations == d.search_evals + d.refine_evals + d.multiplicity_evals
    assert d.search is not None and np.isfinite(d.search.L)
    assert d.n_brackets >= len(res.levels)


def test_stage_3_SPENDS_NOTHING_inside_the_pipeline():
    """The contract that makes the cost model honest. It was NOT free at first: estimating the
    wall slope with an explicit `lam +/- h` probe cost two evaluations per level, and charging
    them to stage 3 contradicted its own docstring. The slope now comes from the two cached
    samples either side of the root."""
    res = solve(_cone([2 * np.pi**2, 5 * np.pi**2, 8 * np.pi**2]), prec=1e-8)
    assert res.diagnostics.multiplicity_evals == 0


# --- Phase D's distance prefilter, and what it replaces ----------------------------------------

def test_roots_within_prec_are_ONE_level_and_read_once():
    """Two located roots inside `prec*lam` are one level of `S(prec)`. Reading multiplicity at
    each separately double-counts, because the wide bracket spans the whole group -- correctly,
    since the group IS the level -- so each fragment declares the group's total."""
    lam = 5 * np.pi**2
    res = solve(_cone([2 * np.pi**2, lam, lam * (1 + 1e-10), 8 * np.pi**2]), prec=1e-8)
    near = [x for x in res.levels if abs(x.lam - lam) / lam < 1e-6]
    assert len(near) == 1, f"a sub-licence pair must come back as one level, got {len(near)}"


def test_reading_once_per_group_gets_the_COLLAPSE_branch_without_the_gram_angle():
    """`search.md` Phase D gives two outcomes for roots this close -- the same eigenfunction
    found twice, which must collapse and RECOUNT, or two independent ones, which must merge and
    SUM -- and warns "do NOT sum" on the first, because summation is the double count. A single
    read on the merged group recounts by construction, and is right either way: one well found
    twice reads its own multiplicity, and two genuine eigenvalues within `prec` both dip and are
    both counted.

    What is still missing is the case the angle is really for: two roots at the SAME separation
    where one pair is a duplicate and the other is not. Distance cannot tell those apart and
    neither can this -- but recounting is the safe branch.
    """
    lam = 5 * np.pi**2
    res = solve(_cone([2 * np.pi**2, lam, lam * (1 + 1e-10)]), prec=1e-8)
    near = [x for x in res.levels if abs(x.lam - lam) / lam < 1e-6]
    assert len(near) == 1
    assert near[0].mult == 2, f"recounted, not summed away: got {near[0].mult}"


# --- merge on an ambiguous grouping, for the sake of the total ---------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("band, ratio", [(0.5, 1.0), (0.8, 1.6), (1.0, 2.0)])
def test_an_ambiguous_grouping_COLLAPSES_and_the_total_survives(band, ratio):
    """The regime where the grouping has no answer but the total does.

    Three eigenvalues evenly spaced with a span between one and two merge licences cannot all
    be one level, and either pairing is legal, so the split is arbitrary. A count taken at the
    middle one must look far enough to see its partner and therefore also sees the stranger the
    same distance away on the other side -- measured exactly equidistant at a span of 1.6. The
    counts came out 3 and 1, totalling 4 where 3 exist.

    **A wrong total is the error that corrupts an index**, which is why the total is what is
    defended here: `lam_3` becomes a different eigenvalue and a solver reports an accurate but
    non-consecutive spectrum.
    """
    (case,) = rs.crossover_cases(1e-8, window="cluster", bands=(band,))
    assert rs.cluster_diameter_ratio(case, 1e-8) == pytest.approx(ratio, rel=0.05)
    res = solve(bs.objective_for(case, 1e-8), prec=1e-8)

    assert sum(x.mult for x in res.levels) == len(case.truth), "the total is right"
    assert res.diagnostics.n_ambiguous == 1, "and it says the grouping was arbitrary"
    flagged = [x for x in res.levels if x.grouping_ambiguous]
    assert len(flagged) == 1 and flagged[0].mult == len(case.truth)
    assert flagged[0].spacings, "carrying the spacings, so a caller can choose a prec"
    assert flagged[0].width > 1e-8 * flagged[0].lam, "and declaring that it is over-wide"


@pytest.mark.slow
@pytest.mark.parametrize("band, wrong", [(0.5, 6), (0.8, 4), (1.0, 4)])
def test_ABLATION_without_the_cross_check_the_TOTAL_is_wrong(band, wrong):
    """The guard shown to be load-bearing. V1 section 7.3: if disabling it breaks nothing, it
    is not doing what you think. Disabled, the declared total is 4 or 6 where 3 exist -- and a
    wrong total is what moves an index."""
    (case,) = rs.crossover_cases(1e-8, window="cluster", bands=(band,))
    obj = bs.objective_for(case, 1e-8)
    off = solve(obj, prec=1e-8, collapse=False)
    assert sum(x.mult for x in off.levels) == wrong, "the failure the guard prevents"
    assert sum(x.mult for x in off.levels) != len(case.truth)

    on = solve(bs.objective_for(case, 1e-8), prec=1e-8)
    assert sum(x.mult for x in on.levels) == len(case.truth), "and with it, the total is right"


@pytest.mark.slow
@pytest.mark.parametrize("band", [1.3, 3.0])
def test_a_SELF_CONSISTENT_fine_grouping_is_kept(band):
    """The reason this is a cross-check and not a merge radius. At spans of 2.6 and 6.0 licences
    the roots are still within counting reach of one another, so a radius rule would collapse
    them and throw away a correct fine answer -- but their individual counts agree with the
    total, so they stand."""
    (case,) = rs.crossover_cases(1e-8, window="cluster", bands=(band,))
    res = solve(bs.objective_for(case, 1e-8), prec=1e-8)
    assert res.diagnostics.n_ambiguous == 0
    assert len(res.levels) == 3 and all(x.mult == 1 for x in res.levels)
    assert sum(x.mult for x in res.levels) == len(case.truth)


@pytest.mark.slow
def test_the_collapse_does_not_fire_where_nothing_is_ambiguous():
    """It must not reach for the merge on ordinary cases. Everything on the roster outside the
    band comes back unflagged."""
    import hard_roster as hr

    for case in (hr.ROSTER["disk"](), hr.ROSTER["eq_tri"](), hr.ROSTER["L_shape"](),
                 *rs.family_D(), *rs.family_A()):
        res = solve(bs.objective_for(case, 1e-8), prec=1e-8)
        assert res.diagnostics.n_ambiguous == 0, case.label
        assert not any(x.grouping_ambiguous for x in res.levels), case.label


@pytest.mark.slow
def test_the_cross_check_reads_only_CACHED_points():
    """It costs nothing: the spanning count is taken over samples the search already paid for."""
    (case,) = rs.crossover_cases(1e-8, window="cluster", bands=(0.8,))
    res = solve(bs.objective_for(case, 1e-8), prec=1e-8)
    assert res.diagnostics.n_ambiguous == 1
    assert res.diagnostics.multiplicity_evals == 0


@pytest.mark.slow
def test_the_escape_from_the_band_works_in_BOTH_directions():
    """The band is `d/2 < prec*lam < d`, so a caller can leave it either way and both give a
    correct total: more precision splits the neighbourhood into separate levels, less merges it
    into one. That is what makes the flag actionable rather than merely honest."""
    (case,) = rs.crossover_cases(1e-8, window="cluster", bands=(0.8,))
    t = np.unique(np.asarray(case.truth, dtype=float))
    d = float(t[-1] - t[0])
    lam = float(t[-1])
    tighter, looser = 0.4 * d / lam, 1.5 * d / lam

    for prec in (tighter, looser):
        res = solve(bs.objective_for(case, prec), prec=prec)
        assert not res.declined
        assert sum(x.mult for x in res.levels) == len(case.truth), (
            f"prec={prec:.2e} must give a correct total")
        assert res.diagnostics.n_ambiguous == 0, f"prec={prec:.2e} escapes the band"
    # and the two escapes disagree about the GROUPING, which is the thing that was arbitrary
    fine = solve(bs.objective_for(case, tighter), prec=tighter)
    coarse = solve(bs.objective_for(case, looser), prec=looser)
    assert len(fine.levels) > len(coarse.levels)


def test_a_declared_width_is_HELD_TO_by_the_scorer():
    """A level may span what it declares and no more. Plain tuples declare nothing, which is
    why the adversaries are unaffected by this allowance."""
    lam = 5 * np.pi**2
    truth = np.array([lam, lam * (1 + 4e-8)])

    class Wide:
        def __init__(self, lam, mult, width):
            self.lam, self.mult, self.width = lam, mult, width

    honest = ls.score_levels([Wide(lam, 2, 4e-7)], truth, 1e-8)
    assert honest.n_impure == 0 and honest.n_declared_wide == 1
    assert honest.total_right

    silent = ls.score_levels([(lam, 2)], truth, 1e-8)
    assert not silent.ok, "declaring nothing means getting no allowance"
    assert silent.verdict == "missed", (
        "and the second eigenvalue is not even assigned to it -- 4e-08 away against a match "
        "radius of prec, so the level is not merely impure, it does not reach")


# --- real pencils ------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("name", ["disk", "eq_tri", "L_shape"])
def test_the_pipeline_on_the_hard_roster(name):
    import hard_roster as hr

    case = hr.ROSTER[name]()
    prec = 1e-8
    obj = bs.objective_for(case, prec)
    res = solve(obj, prec=prec)
    assert not res.declined, res.diagnostics.decline_reason
    s = ls.score_levels(list(res), case.truth, prec, label=name)
    assert s.ok, (f"{s.verdict}: coverage {s.coverage:.2f}, separation {s.separation:.2f}, "
                  f"{s.n_mult_right}/{s.n_matched} multiplicities")
    assert res.diagnostics.multiplicity_evals == 0


@pytest.mark.slow
@pytest.mark.parametrize("mult", [2, 3, 4])
def test_the_pipeline_on_an_exact_degeneracy(mult):
    """The case composition failed on: all three scored separation 0.00 before the seam defects
    were fixed, while every stage passed its own tests."""
    (case,) = [c for c in rs.family_D() if f"mult={mult}" in c.label]
    obj = bs.objective_for(case, 1e-8)
    res = solve(obj, prec=1e-8)
    assert not res.declined
    s = ls.score_levels(list(res), case.truth, 1e-8, label=case.label)
    assert s.ok, f"{s.verdict}"
    assert max(x.mult for x in res.levels) == mult


@pytest.mark.slow
@pytest.mark.parametrize("band", [0.5, 0.8, 1.3, 3.0])
@pytest.mark.parametrize("window", ["pair", "cluster"])
def test_the_pipeline_across_the_crossover(window, band):
    """Both sides of the merge licence. A method that always merges and one that always splits
    both look correct on a sweep that stays on one side."""
    cs = rs.crossover_cases(1e-8, window=window, bands=(band,))
    if not cs:
        pytest.skip("not representable at this band")
    case = cs[0]
    if rs.multiplicity_undecided(case, 1e-8):
        pytest.skip(f"UNDECIDED by construction: the cluster's diameter is "
                    f"{rs.cluster_diameter_ratio(case, 1e-8):.3f} of the licence, inside the "
                    f"band [1, {M.WIDE_FACTOR}] where no flank placement is both outside the "
                    f"level and inside the gap to the next one")
    obj = bs.objective_for(case, 1e-8)
    res = solve(obj, prec=1e-8)
    assert not res.declined, res.diagnostics.decline_reason
    s = ls.score_levels(list(res), case.truth, 1e-8, label=case.label)
    assert s.ok, f"{s.verdict}: {s.n_mult_right}/{s.n_matched} multiplicities"


@pytest.mark.slow
def test_the_multiplicity_undecided_band_is_bounded_by_the_REACH():
    """Not a tuning failure but a property of the configuration: a cluster whose diameter sits
    between the merge licence and the wide bracket's reach is too wide to be one level, and its
    members are closer together than the flanks a read needs.

        diameter/licence   0.40   1.00   1.60   2.60   6.00
        verdict            pass   FAIL   FAIL   pass   pass

    Below the licence the cluster is one level and the reach spans it; above the reach the
    members are further apart than the flanks and each is read alone.
    """
    ratios = {}
    for band in (0.2, 0.5, 0.8, 1.3, 3.0):
        cs = rs.crossover_cases(1e-8, window="cluster", bands=(band,))
        if cs:
            ratios[band] = rs.cluster_diameter_ratio(cs[0], 1e-8)
    assert ratios[0.2] < 1.0 and not rs.multiplicity_undecided(
        rs.crossover_cases(1e-8, window="cluster", bands=(0.2,))[0], 1e-8)
    assert ratios[1.3] > M.WIDE_FACTOR, "and above the reach it is decided again"
    for band in (0.5, 0.8):
        case = rs.crossover_cases(1e-8, window="cluster", bands=(band,))[0]
        assert rs.multiplicity_undecided(case, 1e-8), f"band {band} is inside the band"


@pytest.mark.slow
def test_the_cost_stays_near_thirty_per_eigenvalue_end_to_end():
    """The whole pipeline, charged as one budget: bracket, refine and multiplicity through one
    shared cache, so the cost is the union of what they sampled and not the sum."""
    import hard_roster as hr

    for case in (hr.ROSTER["disk"](), hr.ROSTER["eq_tri"](), *rs.family_A()):
        obj = bs.objective_for(case, 1e-8)
        res = solve(obj, prec=1e-8)
        per = res.diagnostics.evaluations / len(case.truth)
        assert per < bs.COST_CAP_PER_EIG, f"{case.label}: {per:.1f} per eigenvalue"
