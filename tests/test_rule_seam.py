"""The multiplicity rule seam, on exact cones. Plan Phase 3b.

`multiplicity(..., rule=)` and `solve(..., rule=, merge=)` exist so the counting rule and the
merge policy can be changed and measured one at a time. The defaults must reproduce the committed
behaviour exactly -- the seam is plumbing, and the wip branch's version of it failed twice,
silently, by not threading `rule` into `_collapse_ambiguous` and by a patch that matched nothing.

Three rules answer three different questions, which is the point of having them side by side:

    gap     (C1)+(C2) over a wide bracket   the shipped rule
    floor   branches coincident with sigma_1   right about COINCIDENCE, blind to the rest
    reach   eigenvalues INSIDE the bracket   `mu*(B)`: a distance `sigma_j/c` against the bracket
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from chladni.mps.multiplicity import Branch, multiplicity
from chladni.mps.search import _claim, solve

C, EPS, PLATEAU = 1.0, 1e-15, 10.0


def cones(*centres):
    """`sigma` vector from exact cones in `lam`, one branch per centre, slope `C`."""
    centres = np.asarray(centres, dtype=float)

    def tensions(lam):
        return np.sort(np.minimum(np.sqrt(EPS**2 + (C * (centres - float(lam))) ** 2), PLATEAU))

    return tensions


def test_the_defaults_are_the_shipped_behaviour():
    """`solve` ships attained merge, the reach rule on the cache slope, and no cross-check --
    promoted on the held-out 145*pi^2 grid. `multiplicity` keeps "gap" as its own default: it is
    the low-level call, and "reach" there needs a bracket and a slope that only `solve` has."""
    assert inspect.signature(multiplicity).parameters["rule"].default == "gap"
    p = inspect.signature(solve).parameters
    assert (p["merge"].default, p["rule"].default, p["reach_slope"].default,
            p["collapse"].default) == ("attained", "reach", "cache", False)


def test_reach_needs_the_bracket_and_the_slope():
    with pytest.raises(ValueError, match="bracket"):
        multiplicity(cones(10.0), 10.0, rule="reach")


def test_reach_COUNTS_a_pair_inside_the_bracket_and_REJECTS_a_neighbour_outside():
    """Two eigenvalues 0.3 apart inside `[9.5, 10.5]`, a third 2.0 away. `mu*(B) = 2`."""
    t = cones(10.0, 10.3, 12.0)
    m = multiplicity(t, 10.0, rule="reach", bracket=(9.5, 10.5), c=C)
    assert m.mu == 2 and not m.ambiguous


def test_reach_resolves_the_SIDE_on_an_asymmetric_bracket():
    """The partner 0.6 away on the LONG side of `[9.9, 10.8]`: inside, though farther than the
    short side reaches. `min(near, far)` rejected exactly this and under-counted."""
    m = multiplicity(cones(10.0, 10.6), 10.0, rule="reach", bracket=(9.9, 10.8), c=C)
    assert m.mu == 2


def test_reach_does_not_resolve_a_side_when_BOTH_sides_hold_an_eigenvalue():
    """Found promoting the reach rule: the even cluster at 50*pi^2, prec 1e-8, band 0.8. The root
    is the middle member, with partners 8.0e-9 away on BOTH sides; the bracket reaches past the
    lower one and not the upper. A sigma_1 probe measures the nearest eigenvalue on its side, so
    both probes read low (0.121 and 0.109 of sigma_j), the "lower" one was the short side by
    noise, the branch was judged outside, and the loop stopped -- mu 1, not declined. A side that
    holds no branch reads the root's own wall, ~1.0 of sigma_j (measured 0.993). Two sides holding
    one leave the side undetermined, and that must decline rather than count."""
    t = cones(10.0, 10.0 - 1.05, 10.0 + 1.0)       # upper partner nearer, lower one inside
    m = multiplicity(t, 10.0, rule="reach", bracket=(10.0 - 1.1, 10.0 + 0.14), c=C,
                     reach_bounds=(0.61, 2.21, 0.61, 2.21))
    assert m.ambiguous or m.mu == 2, f"read {m.mu}, not declined"


def test_reach_always_counts_the_located_root_s_OWN_branch():
    """Found on the held-out grid at 170*pi^2, prec 1e-12, band 0.3. The refiner put the root at
    its bracket's upper edge with its eigenvalue 1.7e-14 relative beyond it, so branch 0's distance
    landed in the undecided band, the side probe read it OUTSIDE on the short side, and the loop
    stopped -- the partner 3.1e-13 inside the long side was never examined, mult 1, not declined.
    Branch 0 is the level's own eigenvalue; the level exists because it was located."""
    t = cones(10.0 + 3e-6, 10.0 - 0.3)        # own eigenvalue just past the upper edge; partner inside
    m = multiplicity(t, 10.0, rule="reach", bracket=(9.0, 10.0 + 1e-7), c=C,
                     reach_bounds=(0.61, 2.21, 0.61, 2.21))
    assert m.mu == 2 or m.ambiguous, f"read {m.mu}, not declined"



def test_reach_keeps_looking_past_a_branch_OUTSIDE_the_short_side():
    """Found on the held-out grid at 170*pi^2, prec 1e-12, band 0.3, once branch 0 was always
    counted. The root sat at its bracket's upper edge; branch 1 was the partner just above, correctly
    rejected on the short side -- and the loop BROKE, so branch 2, a member inside the long side,
    was never examined. Ascending branches justify stopping only when a branch is outside BOTH ways;
    one outside the short side says nothing about the next one on the long side."""
    t = cones(10.0, 11.0, 10.0 - 1.8)          # partner outside above; member inside below
    m = multiplicity(t, 10.0, rule="reach", bracket=(8.0, 10.01), c=C,
                     reach_bounds=(0.61, 2.21, 0.61, 2.21))
    assert m.mu == 2 or m.ambiguous, f"read {m.mu}, not declined"

# --- the branch record, and ownership --------------------------------------------------------
#
# `mu` says HOW MANY eigenvalues the rule believes are in the bracket, never WHICH, so a level the
# side search later locates could not be compared against it and the member was reported twice.
# These pin the record and the claim that reconciles them, on exact cones where the answer is
# known by construction.


def test_the_read_RECORDS_every_branch_it_EXAMINED():
    """One entry per branch looked at, in order, and `mu` is exactly the counted ones."""
    m = multiplicity(cones(10.0, 10.3, 12.0), 10.0, rule="reach", bracket=(9.5, 10.5), c=C)
    assert [b.index for b in m.branches] == list(range(len(m.branches)))
    assert m.mu == sum(b.verdict == "counted" for b in m.branches)
    assert m.branches[0].verdict == "counted", "branch 0 is the level's own eigenvalue"
    assert any(b.verdict == "rejected" for b in m.branches), "the neighbour at 12.0"
    for b in m.branches:
        assert b.d_est == pytest.approx(b.sigma / C, rel=1e-12)


def test_a_branch_the_loop_NEVER_REACHED_is_absent_rather_than_marked():
    """After a break there is no evidence about the branches above, and an "unexamined" entry
    would invite a caller to reason from nothing. Four cones, the far ones ending the loop: the
    record stops there, so `mu` is legibly partial over exactly what was looked at."""
    m = multiplicity(cones(10.0, 20.0, 30.0, 40.0), 10.0, rule="reach", bracket=(9.5, 10.5), c=C)
    assert len(m.branches) < 4, "the loop broke and the record stops with it"
    assert m.branches[-1].verdict in ("rejected", "undecided")


def test_a_branch_the_read_counted_is_owned_ONCE():
    """The band is `[d_est/bias_hi, d_est/bias_lo]`, 3.6x wide under the cache bounds, so two
    claimants can both admit one branch. Exactly one of them gets it."""
    b = Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="counted")
    assert _claim((b,), 10.0, [10.0 + 0.45, 10.0 + 1.60], 0.61, 2.21) == (1, (1,))


def test_a_claimant_OUTSIDE_the_estimate_s_band_claims_nothing():
    """`d_est = beta * d_true` with beta in `[bias_lo, bias_hi]`, so a claimant nearer than
    `d_est/bias_hi` or further than `d_est/bias_lo` is a different eigenvalue."""
    b = Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="counted")
    assert _claim((b,), 10.0, [10.0 + 0.30], 0.61, 2.21)[0] == 0, "nearer than the band"
    assert _claim((b,), 10.0, [10.0 + 2.00], 0.61, 2.21)[0] == 0, "further than the band"


def test_a_claimant_on_the_WRONG_SIDE_of_a_RESOLVED_branch_claims_nothing():
    """Where the probe resolved the side, the side is evidence and a claimant the other way is a
    different eigenvalue. Where it did not (`side == 0`) either way is admissible."""
    resolved = Branch(index=1, sigma=1.0, d_est=1.0, side=+1, verdict="counted")
    assert _claim((resolved,), 10.0, [10.0 - 1.0], 0.61, 2.21)[0] == 0
    assert _claim((resolved,), 10.0, [10.0 + 1.0], 0.61, 2.21)[0] == 1
    unresolved = Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="counted")
    assert _claim((unresolved,), 10.0, [10.0 - 1.0], 0.61, 2.21)[0] == 1


def test_a_REJECTED_or_UNDECIDED_or_ZEROTH_branch_is_NOT_claimable():
    """The safety property, and what keeps `mult >= 1` true by construction rather than by a
    clamp. Only a branch the read ADDED to `mu` may be taken out of it -- capping the count at
    what the search could not reach was measured and lost a member (2 of 3 at prec 1e-6). Branch 0
    is the level's own eigenvalue and is excluded outright, or a claimant a few ulps out could
    claim the root's own existence."""
    for b in (Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="rejected"),
              Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="undecided"),
              Branch(index=0, sigma=1.0, d_est=1.0, side=0, verdict="counted")):
        assert _claim((b,), 10.0, [10.0 + 1.0], 0.61, 2.21)[0] == 0, b.verdict


def test_two_claimants_cannot_take_the_SAME_branch():
    """"Owned ONCE" is the whole content of the fix. Without it the root's count drops twice for
    one eigenvalue, which under-reads exactly as the double count over-read."""
    b = Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="counted")
    assert _claim((b,), 10.0, [10.0 + 0.9, 10.0 + 1.1], 0.61, 2.21)[0] == 1


def test_each_claimant_takes_at_most_one_branch():
    two = (Branch(index=1, sigma=1.0, d_est=1.0, side=0, verdict="counted"),
           Branch(index=2, sigma=1.1, d_est=1.1, side=0, verdict="counted"))
    assert _claim(two, 10.0, [10.0 + 1.0], 0.61, 2.21)[0] == 1


def test_ONE_find_a_counted_and_an_undecided_branch_both_admit_is_AMBIGUOUS():
    """The traced cell's ratios (`notes/smooth_filter.md` section 10.7: 50*pi^2, band 0.3, prec
    1e-10, `tau = 0.02 prec`): counted branch 1 at d_est 1.90e-8, undecided branch 2 at 4.12e-8,
    one find at 2.77e-8 on their side. Both bands admit it. If it is branch 1's member, branch 2's
    is still out there; if it is branch 2's, branch 1's lies inside the side search's exclusion,
    where nothing looked. Nothing decides between them, and the old best-match rule chose one
    SILENTLY -- 2 of 3, wrong. The count keeps its pre-existing reading and says it is undecided."""
    b1 = Branch(index=1, sigma=1.0, d_est=1.0, side=-1, verdict="counted")
    b2 = Branch(index=2, sigma=2.17, d_est=2.17, side=-1, verdict="undecided")
    assert _claim((b1, b2), 10.0, [10.0 - 1.46], 0.61, 2.21,
                  with_ambiguity=True) == (1, (1,), True)


def test_TWO_finds_against_a_counted_and_an_undecided_branch_are_decided_by_ORDER():
    """50*pi^2, band 0.3, prec 1e-6 on the default pencil: counted branch 1 at 0.319 licences,
    undecided branch 2 at 0.574, finds at 0.291 and 0.575, all on one side. On one side a nearer
    eigenvalue has the smaller `sigma_j`, so the nearer find is branch 1's and the further branch
    2's -- one reading, not ambiguous. Preferring the undecided branch outright gave the nearer
    find to it and came back 4 of 3; this is the test that caught that."""
    b1 = Branch(index=1, sigma=0.319, d_est=0.319, side=1, verdict="counted")
    b2 = Branch(index=2, sigma=0.574, d_est=0.574, side=1, verdict="undecided")
    assert _claim((b1, b2), 10.0, [10.0 + 0.291, 10.0 + 0.575], 0.61, 2.21,
                  with_ambiguity=True) == (1, (1,), False)


def test_a_find_ONLY_a_counted_branch_admits_still_claims_it():
    """A find outside the undecided branch's band is the counted branch's member located again --
    the 9-of-8 double count `_claim` exists for -- and it is not ambiguous."""
    b1 = Branch(index=1, sigma=1.0, d_est=1.0, side=-1, verdict="counted")
    b2 = Branch(index=2, sigma=5.0, d_est=5.0, side=-1, verdict="undecided")
    assert _claim((b1, b2), 10.0, [10.0 - 1.0], 0.61, 2.21,
                  with_ambiguity=True) == (1, (1,), False)


def test_a_find_ONLY_an_undecided_branch_admits_subtracts_nothing():
    b1 = Branch(index=1, sigma=0.3, d_est=0.3, side=-1, verdict="counted")
    b2 = Branch(index=2, sigma=2.0, d_est=2.0, side=-1, verdict="undecided")
    assert _claim((b1, b2), 10.0, [10.0 - 2.0], 0.61, 2.21,
                  with_ambiguity=True) == (0, (), False)


@pytest.mark.slow
def test_the_traced_under_count_cell_is_never_SILENTLY_wrong():
    """The cell `notes/trace_undercount.py` traced: 2 of 3, silent, because one side-search find
    was admissible to a counted and an undecided branch and the best-match rule gave it the
    counted one. The evidence there does not decide the total, so the contract is the one that
    can be met: right, or flagged. `tau = 0.02 prec` is the rule the default adopts."""
    import bracket_score as bs
    import rect_suite as rs
    from chladni.mps import MPSPencil
    from chladni.mps.objective import Objective

    prec = 1e-10
    (case,) = rs.split_degeneracy_cases(prec, bands=(0.3,), levels={50: 3})
    p = case.pencil
    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec,
                  rtol=0.02 * prec / np.sqrt(p.n_cols))
    base = bs.objective_for(case, prec)
    obj = Objective(lambda x: q.tensions(float(x)), base.lo, base.hi, prec=prec,
                    variable=case.variable, expected_count=base.expected_count)
    res = solve(obj, prec=prec)
    assert not res.declined
    total = sum(L.mult for L in res.levels)
    assert total == 3 or res.diagnostics.n_uncertain > 0, (
        f"{total} of 3 and SILENT: the one outcome this cell must not have")


@pytest.mark.slow
@pytest.mark.parametrize("N, m, band, prec", [(50, 3, 0.8, 1e-8), (65, 4, 0.3, 1e-8),
                                              (50, 3, 0.3, 1e-8), (170, 4, 0.3, 1e-12)])
def test_the_multiplicity_READ_spends_nothing_even_where_the_count_DECLINES(N, m, band, prec):
    """Stage 3's contract, asserted where it could actually fail.

    `multiplicity_evals` used to be the WHOLE stage-3 window, so it silently included the reach
    rule's two-evaluation side probes and the side search's refinements. It read zero only on
    cells that never decline -- which are exactly the cells that do neither -- so the contract
    was being checked where it was unfalsifiable. Split out, the READ answers from the cache and
    nothing else, and the work it used to hide is reported as `probe_evals` and `side_evals`.

    A cone surrogate cannot stand in here: its branches never land in the undecided band, so it
    declines nothing and spends nothing. That is why this is a real cluster and lives in the slow
    tier.
    """
    _, _, d = _cluster_total(N, m, band, prec)
    assert d.probe_evals + d.side_evals > 0, (
        "this cell no longer exercises the probes or the side search, so it no longer tests what "
        "it was written for -- find a cluster that does")
    assert d.multiplicity_evals == 0, (
        f"the stage-3 read spent {d.multiplicity_evals} evaluations; it must answer from the "
        f"cache (probes {d.probe_evals}, side search {d.side_evals})")


@pytest.mark.slow
@pytest.mark.parametrize("N, m, band, prec, cause", [
    (50, 3, 0.8, 1e-8, "both sides hold a partner, and the side probe picked one"),
    (170, 4, 0.3, 1e-12, "a residual sub-bracket abutting an exclusion, where the refiner "
                         "landed on the excluded member's wall until the standoff reached 0.3"),
    (65, 4, 0.3, 1e-8, "a hump top kept as a level beside the two members found either side"),
])
def test_the_promoted_defaults_leave_no_SILENT_wrong_total(N, m, band, prec, cause):
    """The silent wrong totals that appeared once the crash fix (the QR route around Accelerate's
    dgeqrf) moved the objective. At 65*pi^2, band 0.3, one bracket narrower than the licence held
    two members, the refiner accepted the hump between them (sigma 1.35e-8), the side search found
    both members -- and the hump stayed as a third level beside them."""
    total, truth, d = _cluster_total(N, m, band, prec)
    assert total == truth or not d.certified_counts, (
        f"declared {total} of {truth} with certified counts -- a silent wrong total ({cause})")


def test_floor_counts_COINCIDENCE_only():
    """An exact double reads 2; a pair 1e-3 apart -- inside any sensible bracket but nowhere near
    the floor together -- reads 1. The floor rule answers a different question than `mu*(B)`."""
    assert multiplicity(cones(10.0, 10.0), 10.0, rule="floor").mu == 2
    assert multiplicity(cones(10.0, 10.001), 10.0, rule="floor").mu == 1


def test_the_reach_slope_is_a_choice_and_CACHE_is_the_default():
    """`reach_slope` picks where the reach rule reads its wall slope. "cache" reads the nearest
    cached samples either side of the root, which measured within 7% of the true distance on the
    cells where "outside" -- three bracket widths out, what the rule was first calibrated on --
    read 1.86x: on a cluster wider than the standoff, the outside samples sit in other members'
    wells. "cache" became the default once its recalibrated bounds cleared the held-out grid."""
    p = inspect.signature(solve).parameters
    assert p["reach_slope"].default == "cache"
    with pytest.raises(ValueError, match="reach_slope"):
        from chladni.mps.objective import Objective
        solve(Objective(lambda k: [0.2, 0.3], 3.0, 4.0, variable="k", prec=1e-8),
              rule="reach", reach_slope="sideways")


# --- the reach rule LOCATES a partner it cannot count ----------------------------------------------

_REACH = dict(merge="attained", rule="reach", reach_slope="cache", collapse=False)


def _cluster_total(N, m, band, prec):
    import bracket_score as bs
    import rect_suite as rs

    (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: m})
    truth = rs.complete_truth(case)
    res = solve(bs.objective_for(case, prec), prec=prec, **_REACH)
    lam0, window = N * rs.PI2, 0.02 * N * rs.PI2
    near = [L for L in res.levels if abs(L.lam - lam0) < window]
    # The truth in the SAME window, not `m`: at 170*pi^2 the window also holds the pairs at
    # 1667.96 and 1707.44, so it holds 8 where the cluster has 4.
    want = int(np.sum(np.abs(truth - lam0) < window))
    return sum(L.mult for L in near), want, res.diagnostics


@pytest.mark.slow
@pytest.mark.parametrize("N, m, band, prec", [(50, 3, 0.3, 1e-8), (50, 3, 0.5, 1e-8),
                                              (65, 4, 0.3, 1e-8)])
def test_a_partner_at_the_bracket_edge_is_LOCATED_rather_than_left_uncounted(N, m, band, prec):
    """Every wrong total the recalibrated reach rule left on the cluster grid was one bracket
    holding two distinct eigenvalues: the refiner located one, the partner sat at the bracket
    edge where sigma_j/c cannot place it inside or out, and the rule declined -- honestly, with a
    total one short. Merging is permitted and never required, so the partner need not be counted
    at all: refining the bracket either side of the root, outside half the estimated distance,
    found it as its own root in every such case measured, at 1-2 evaluations a side."""
    total, truth, d = _cluster_total(N, m, band, prec)
    assert total == truth, f"declared {total} of {truth}"
    assert d.certified_counts, d.uncertain


@pytest.mark.slow
@pytest.mark.parametrize("prec, band, cause", [
    (1e-10, 0.5, "side-search dedup: the partner was discarded against the root being split"),
    (1e-8, 1.6, "phantom 4 ulps inside a bracket end, past the ulp tie"),
])
def test_the_fresh_125_grid_has_no_SILENT_wrong_total(prec, band, cause):
    """The two silent wrong totals the candidate left on the held-out 125*pi^2 grid. Pinning them
    makes 125 design data; 145 and 170 stay held out.

    1e-10, band 0.5: the rule declined at the root, the side search FOUND the partner 5.5e-8 away
    -- and discarded it as already located, because the dedup compared it with the root being
    split, whose own bar was 5.6e-8. The sub-bracket excludes that root by construction.
    1e-8, band 1.6: a phantom accepted by the refiner as a fifth level; see
    `test_refine_regression.test_a_phantom_a_FEW_ULPS_INSIDE_the_end_is_judged_by_its_reach...`."""
    total, truth, d = _cluster_total(125, 4, band, prec)
    assert total == truth or not d.certified_counts, (
        f"declared {total} of {truth} with certified counts -- a silent wrong total ({cause})")
    assert total == truth, f"declared {total} of {truth}, flagged ({cause})"


@pytest.mark.slow
@pytest.mark.parametrize("N, m, band, prec, cause", [
    (50, 3, 0.3, 1e-6, "a coarsely located member dropped as a hump: its cell held an unlocated "
                       "member's well, so the cell refine read not_a_minimum -- 1 of 3"),
    (50, 3, 0.8, 1e-14, "two brackets refined to the SAME member, attained merge folded them, and "
                        "the n_roots > mult flag read a duplicate as a possible hidden pair"),
    (125, 4, 0.8, 1e-8, "a side search read the wall of a member already located 4.5e-6 away, "
                        "outside a one-width window, and called the dip unexplained"),
    (65, 4, 1.0, 1e-10, "a side sub-bracket on the hump top between two located members, accepted "
                        "once the refiner's fallback found a parabola there -- 5 of 4"),
])
def test_the_hump_and_side_guards_do_not_break_a_CORRECT_count(N, m, band, prec, cause):
    """The two ways the guards for the silent cells overreached, measured on the 50/65 grid. A
    guard that turns a correct certified count into a wrong one, or into a flagged one, costs more
    than the silent cell it was written for: `solve_interval` raises on a flag."""
    total, truth, d = _cluster_total(N, m, band, prec)
    assert total == truth, f"declared {total} of {truth} ({cause})"
    assert d.certified_counts, f"a correct count came back flagged ({cause}): {d.uncertain}"


@pytest.mark.slow
def test_a_side_search_that_finds_NOTHING_leaves_a_correct_count_correct():
    """The uneven cluster at 1e-12, band 0.5 is already right under the reach rule, flagged
    because a branch landed in the undecidable band. Side searches there find no second root --
    there is none -- and the total must not move."""
    total, truth, _ = _cluster_total(65, 4, 0.5, 1e-12)
    assert total == truth
