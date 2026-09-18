"""Phase D's instrument: can the interior Gram angle tell a duplicate from a genuine pair?

`docs/design/search.md` section 7 specifies this file. It was written BEFORE the rule that uses
it -- `CHLADNI_V1.md` section 5.1's discipline, build the scorer before the searcher -- so the
first half tests the INSTRUMENT (the angle itself, on real pencils) and the second half tests the
RULE built on it (`solve(..., phase_d="angle")`, on controlled inputs). Both halves are here
because the rule is only as good as the separation the instrument gives it.

# Why the angle at all

Distance provably cannot decide it. One eigenvalue found twice and two genuine eigenvalues occur
at the SAME separation, so no rule on `|lam_i - lam_j|` can separate them; only the
eigenfunctions can. That is the whole argument for `modes(x)` riding on the `Objective`.

# What is measured here, and what it corrects in the spec

`notes/landscape_findings.md` section 36, reproduced by `notes/gram_discrimination.py`. Two of
the spec's numbers do not survive contact with the roster, and both are recorded as tests rather
than left to trip the next reader:

* **"distinct <= 0.10" is not a constant.** It is the Monte-Carlo sampling noise of the discrete
  interior inner product, `1/sqrt(n_int)`, so it moves with the POINT COUNT. Measured worst
  distinct reading: rect 0.1402, L_shape 0.0605, eq_tri 0.1678, disk 0.2544 -- the rectangle's
  own number already exceeds the quoted bound.
* **"even at two-thirds of a gap" is a trap.** At two-thirds of a gap the probe is PAST THE
  MIDPOINT, where the candidate has already turned into the neighbour's eigenfunction, so it
  reads ~0.06 and looks like a catastrophic failure of the duplicate test. It is not the
  duplicate case at all: two located roots that are really one eigenvalue sit `prec*lam` apart,
  eight to fourteen orders closer, and there the reading is 1.0000.

The headline the rule may rely on:

    worst distinct 0.2544  <  TAU_GRAM 0.5  <  weakest duplicate 1.0000

# The corner domain is the BEST of the four

Section 15 named the L-shape with a corner-adapted basis as the transfer risk -- the domain whose
eigenfunctions are singular and whose basis is fractional-order, where the distinct-pair floor
might lift. It reads 0.0605, the lowest of the roster. The risk does not materialize, and that is
asserted here so a later basis change has to confront it.
"""
from __future__ import annotations

import numpy as np
import pytest

import hard_roster as hr
import rect_suite as rs
# The verdict boundary, imported rather than restated. It lived in this file while nothing read
# it -- an unused package constant is what `search._SLOPE_FRAC` became -- and moved into the
# package with the rule; `test_TAU_GRAM_lives_in_the_PACKAGE...` below pins that it stayed.
from chladni.mps.multiplicity import TAU_GRAM

pytestmark = pytest.mark.slow

ROSTER = ("L_shape", "disk", "eq_tri")

#: Worst distinct reading measured per domain (section 36). Hard-coded so a basis, collocation or
#: rank change has to confront the number rather than silently move it.
WORST_DISTINCT = {"L_shape": 0.0605, "disk": 0.2544, "eq_tri": 0.1678}


def gram(pencil, a, b):
    """`|<u(a), u(b)>|` on the pencil's own interior points -- a plain dot product.

    No common refinement, no quadrature, no lifting to `L2(Omega)`: both vectors are sampled at
    the SAME points, because the `Sampling` is fixed at construction and only `k` changes.
    """
    return abs(float(pencil.interior_image(float(a)) @ pencil.interior_image(float(b))))


def _levels(case):
    """The distinct true eigenvalues strictly inside the case's window."""
    t = np.unique(np.asarray(case.truth, dtype=float))
    return t[(t > case.a) & (t < case.b)]


# --- the duplicate population ----------------------------------------------------------------


@pytest.mark.parametrize("name", ROSTER)
@pytest.mark.parametrize("prec", [1e-6, 1e-8, 1e-10, 1e-12])
def test_one_eigenvalue_found_TWICE_reads_ONE(name, prec):
    """The case Phase D actually faces: two located roots a MERGE LICENCE apart.

    This is the separation that matters and the one section 15 never probed. Measured 1.0000 at
    every level of every roster domain at every precision on the ladder.
    """
    case = hr.ROSTER[name]()
    for lam in _levels(case)[:3]:
        lam = float(lam)
        half = 0.5 * prec * lam
        got = gram(case.pencil, lam - half, lam + half)
        assert got > TAU_GRAM, (
            f"{name} at lam={lam:.6g}, roots {prec:.0e} apart read {got:.4f}: a duplicate must "
            f"be recognised as one eigenfunction")
        assert got == pytest.approx(1.0, abs=1e-3), f"{name}: {got:.6f}"


# --- the distinct population -----------------------------------------------------------------


@pytest.mark.parametrize("name", ROSTER)
def test_GENUINELY_DISTINCT_eigenvalues_read_near_the_SAMPLING_FLOOR(name):
    """Adjacent true eigenvalues, which must not be mistaken for one another.

    The floor is `1/sqrt(n_int)`, the Monte-Carlo noise of a discrete inner product over `n_int`
    points -- NOT the spec's constant 0.10. Asserted against the measured worst reading per
    domain, with room, and separately against `TAU_GRAM`, which is what the rule will use.
    """
    case = hr.ROSTER[name]()
    p, t = case.pencil, _levels(case)
    vals = [gram(p, float(t[i]), float(t[i + 1])) for i in range(min(len(t) - 1, 6))]
    worst = max(vals)
    assert worst < TAU_GRAM, (
        f"{name}: two distinct eigenvalues read {worst:.4f}, at or above the verdict boundary")
    assert worst <= WORST_DISTINCT[name] * 1.25, (
        f"{name}: worst distinct reading {worst:.4f} against the measured "
        f"{WORST_DISTINCT[name]:.4f} -- the distinct population moved")


def test_the_CORNER_domain_is_not_the_weak_one():
    """Section 15 named the L-shape as the transfer risk: singular eigenfunctions, a
    fractional-order basis, and a distinct-pair floor that might lift above pure sampling noise.

    It is the BEST of the roster. Pinned so a basis change has to confront it rather than
    discovering it later as a Phase D failure."""
    worst = {}
    for name in ROSTER:
        case = hr.ROSTER[name]()
        t = _levels(case)
        worst[name] = max(gram(case.pencil, float(t[i]), float(t[i + 1]))
                          for i in range(min(len(t) - 1, 6)))
    assert worst["L_shape"] == min(worst.values()), worst


def test_the_two_populations_are_SEPARATED_by_the_verdict_boundary():
    """The claim the rule rests on, stated once over everything measured."""
    hi, lo = 0.0, 1.0
    for name in ROSTER:
        case = hr.ROSTER[name]()
        p, t = case.pencil, _levels(case)
        hi = max([hi] + [gram(p, float(t[i]), float(t[i + 1]))
                         for i in range(min(len(t) - 1, 6))])
        lo = min([lo] + [gram(p, float(l) - 0.5e-8 * float(l), float(l) + 0.5e-8 * float(l))
                         for l in t[:3]])
    assert hi < TAU_GRAM < lo, f"worst distinct {hi:.4f}, tau {TAU_GRAM}, weakest duplicate {lo:.4f}"
    assert lo / hi > 3.0, f"separation is only {lo / hi:.1f}x -- too delicate to build a rule on"


# --- the two corrections, pinned so they are not re-derived ----------------------------------


def test_a_probe_past_the_MIDPOINT_measures_the_NEIGHBOUR_not_the_same_well():
    """`search.md`'s "even at two-thirds of a gap" must not be read as the duplicate case.

    Past halfway the nearest eigenvalue IS the other one, so the candidate has turned and the
    angle correctly reads DISTINCT. A test placed there would look like the instrument failing
    catastrophically -- 0.06 where 0.948 was expected -- and would be measuring the wrong thing.
    """
    case = hr.ROSTER["L_shape"]()
    p, t = case.pencil, _levels(case)
    la, gap = float(t[0]), float(t[1]) - float(t[0])
    before = gram(p, la, la + 0.33 * gap)
    after = gram(p, la, la + 0.67 * gap)
    assert before > TAU_GRAM, f"one third in is still the same well: {before:.4f}"
    assert after < TAU_GRAM, f"two thirds in is the neighbour's: {after:.4f}"
    assert before / max(after, 1e-12) > 5.0, "the turn at the midpoint should be sharp"


@pytest.mark.parametrize("name", ROSTER)
def test_the_distinct_floor_TRACKS_THE_POINT_COUNT_rather_than_a_constant(name):
    """Why the spec's 0.10 is the wrong SHAPE of bound, not merely the wrong number.

    The discrete interior product is a Monte-Carlo estimate over `n_int` points, so two
    orthogonal functions read `O(1/sqrt(n_int))`, and a bound stated as a constant silently
    becomes wrong when the collocation changes. Asserted loosely -- within a factor of three of
    the floor -- because it is the SCALING being claimed, not a coefficient.
    """
    case = hr.ROSTER[name]()
    p, t = case.pencil, _levels(case)
    floor = 1.0 / np.sqrt(p.n_int)
    typical = float(np.median([gram(p, float(t[i]), float(t[i + 1]))
                               for i in range(min(len(t) - 1, 6))]))
    assert typical < 3.0 * floor, (
        f"{name}: typical distinct reading {typical:.4f} against a sampling floor "
        f"{floor:.4f} -- higher than sampling noise explains")


# --- the vacuity, which is a rule-level hazard rather than an instrument property -------------


def test_WITHIN_ONE_lam_the_angle_is_VACUOUS_and_must_not_be_consulted():
    """The GSVD gives `A_I x_j = s_j v_j` with `V` orthogonal, so the interior images of two
    directions at the SAME `lam` are orthogonal BY CONSTRUCTION and say nothing about the
    eigenfunctions.

    `tests/test_mps_seam.py` pins this on the pencil. It is repeated here because the hazard is
    not the pencil's -- it is that Phase D's rule could reach for the comparison and get a
    confident, meaningless answer. The rule must only ever compare ACROSS reported roots.
    """
    case = hr.ROSTER["disk"]()
    p = case.pencil
    lam = float(_levels(case)[1])
    _, I = p.pencil(lam)
    c = p.coef(lam, 2)
    u, v = I @ c[:, 0], I @ c[:, 1]
    cos = abs(float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert cos < 1e-10, f"expected orthogonal by construction, got {cos:.2e}"


# --- the rectangle, where section 15 measured ------------------------------------------------


def test_the_rectangle_pair_is_STABLE_across_six_orders_of_separation():
    """`rect(1+delta, 1)`'s (2,1)/(1,2) pair, which is section 15's case. The distinct reading
    does not depend on how far apart the two eigenvalues are -- it is set by the point count, not
    by the separation -- which is the cleanest evidence that the floor is sampling noise."""
    seen = []
    for case in rs.family_B():
        t = np.unique(rs.complete_truth(case))
        near = t[np.argsort(abs(t - 5 * np.pi**2))[:2]]
        if near.size < 2:
            continue
        seen.append(gram(case.pencil, float(min(near)), float(max(near))))
    assert len(seen) >= 5
    assert max(seen) - min(seen) < 0.01, f"the distinct reading moved with separation: {seen}"
    assert max(seen) < TAU_GRAM


# =============================================================================================
# THE RULE, on the seam. `solve(..., phase_d="angle")`.
# =============================================================================================
#
# Everything above measures the INSTRUMENT. These test the rule built on it. They use a
# synthetic objective rather than a pencil, because the point is to force BOTH verdicts on
# demand -- which a real domain will not do -- and because the ambiguous-eigenspace case is
# defined by the verdict being arbitrary.

from chladni.mps.objective import Objective                         # noqa: E402
from chladni.mps.search import Level, _gram_group, solve            # noqa: E402


def _pair_objective(centres, directions, *, eps=1e-13, c_k=0.27, plateau=0.2,
                    lo=15.0, hi=110.0, prec=1e-8):
    """Cones at `centres`, with a prescribed eigenfunction direction at each.

    `modes(x)` returns the direction belonging to the NEAREST centre, which is what a real
    candidate eigenfunction does -- it is the mode of whichever eigenvalue dominates there, and
    it is why the angle turns at the midpoint on a real pencil.
    """
    ks = np.sqrt(np.asarray(centres, dtype=float))
    dirs = np.array(directions, dtype=float)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]

    def tensions(lam):
        d = np.sort(np.abs(ks - np.sqrt(float(lam))))
        return np.minimum(np.sqrt(eps**2 + (c_k * d) ** 2), plateau)

    def modes(lam):
        return dirs[int(np.argmin(np.abs(ks - np.sqrt(float(lam)))))]

    return Objective(tensions, lo, hi, prec=prec, variable="lam", modes=modes)


ORTH = ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])      # |cos| = 0 -> independent
SAME = ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])      # |cos| = 1 -> one eigenfunction


def test_TAU_GRAM_lives_in_the_PACKAGE_now_that_something_reads_it():
    """It lived in this file while nothing read it, so it would not become another
    `_SLOPE_FRAC` -- a constant with a docstring and no caller."""
    import chladni.mps.multiplicity as mult
    assert mult.TAU_GRAM is TAU_GRAM
    assert "TAU_GRAM" in mult.__all__


def test_the_default_is_DISTANCE_and_the_angle_is_OPT_IN():
    """Promoted one variable at a time, as `rule=` and `merge=` were. The default is named for
    what it DOES -- the distance prefilter is not nothing; it already gets collapse-by-recount
    for free -- rather than for what it lacks."""
    import inspect
    assert inspect.signature(solve).parameters["phase_d"].default == "distance"
    obj = _pair_objective([50.0], ORTH[:1])
    with pytest.raises(ValueError, match="phase_d"):
        solve(obj, prec=1e-8, phase_d="gram")


def test_the_angle_REFUSES_an_objective_with_no_eigenfunctions():
    """Silently falling back to the distance rule would tell a caller nothing, and the seam
    exists so the two can be measured one at a time."""
    bare = Objective(lambda x: np.array([abs(x - 50.0) + 1e-9]), 15.0, 110.0, variable="lam")
    with pytest.raises(ValueError, match="modes|eigenfunction"):
        solve(bare, prec=1e-8, phase_d="angle")


def test_a_DUPLICATE_pair_COLLAPSES_and_is_RECOUNTED_not_summed():
    """One eigenvalue reported twice inside a licence. Summation IS the double count."""
    lam, prec = 50.0, 1e-8
    obj = _pair_objective([lam, lam * (1 + 0.4 * prec)], SAME, prec=prec)
    res = solve(obj, prec=prec, phase_d="angle")
    assert sum(x.mult for x in res.levels) == 1, [(x.lam, x.mult) for x in res.levels]
    assert res.diagnostics.n_gram_collapsed == 1
    assert res.diagnostics.n_gram_summed == 0
    assert all(x.count_decided_by == "gram" for x in res.levels if x.n_roots > 1)


def test_an_INDEPENDENT_pair_MERGES_and_is_SUMMED():
    """Two genuine eigenvalues at the SAME separation as the duplicate above. Distance cannot
    tell this case from that one; the angle is the only thing that differs."""
    lam, prec = 50.0, 1e-8
    obj = _pair_objective([lam, lam * (1 + 0.4 * prec)], ORTH, prec=prec)
    res = solve(obj, prec=prec, phase_d="angle")
    assert sum(x.mult for x in res.levels) == 2, [(x.lam, x.mult) for x in res.levels]
    assert res.diagnostics.n_gram_summed == 1
    assert res.diagnostics.n_gram_collapsed == 0


def test_the_two_cases_are_IDENTICAL_in_distance_and_differ_only_in_the_ANGLE():
    """The argument for Phase D existing, as one assertion: same separation, same tensions, same
    everything a distance rule can see -- and the right answers differ."""
    lam, prec = 50.0, 1e-8
    dup = _pair_objective([lam, lam * (1 + 0.4 * prec)], SAME, prec=prec)
    ind = _pair_objective([lam, lam * (1 + 0.4 * prec)], ORTH, prec=prec)
    xs = np.linspace(49.0, 51.0, 11)
    assert np.allclose([dup.tension(x) for x in xs], [ind.tension(x) for x in xs]), (
        "the two cases must be indistinguishable to anything reading only tensions")
    a = solve(dup, prec=prec, phase_d="angle")
    b = solve(ind, prec=prec, phase_d="angle")
    assert sum(x.mult for x in a.levels) == 1
    assert sum(x.mult for x in b.levels) == 2


def test_the_DISTANCE_rule_cannot_separate_them_which_is_why_the_angle_exists():
    """The control. Under the shipped default both cases give the same total, so whatever that
    total is, one of the two is wrong -- and nothing in the result says which."""
    lam, prec = 50.0, 1e-8
    a = solve(_pair_objective([lam, lam * (1 + 0.4 * prec)], SAME, prec=prec), prec=prec)
    b = solve(_pair_objective([lam, lam * (1 + 0.4 * prec)], ORTH, prec=prec), prec=prec)
    assert sum(x.mult for x in a.levels) == sum(x.mult for x in b.levels)
    assert a.diagnostics.n_gram_pairs == 0, "the distance rule asks the angle nothing"


def _levels_at(a, b, mult_a, mult_b):
    """Two hand-built levels inside one licence, with their multiplicity reads fixed."""
    return [Level(lam=a, mult=mult_a, sigma=1e-13, delta_res=1e-12, n_roots=1),
            Level(lam=b, mult=mult_b, sigma=2e-13, delta_res=1e-12, n_roots=1)]


def _obj_with_directions(a, b, dir_a, dir_b):
    """An objective whose `modes` returns a prescribed direction at each of two points."""
    table = {round(float(a), 12): np.array(dir_a, dtype=float),
             round(float(b), 12): np.array(dir_b, dtype=float)}
    table = {k: v / np.linalg.norm(v) for k, v in table.items()}
    return Objective(lambda x: np.array([1e-13]), 1.0, 1000.0, variable="lam",
                     modes=lambda x: table[round(float(x), 12)])


E1, E2 = [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("dir_b, arm, total", [(E1, "collapse", 1), (E2, "sum", 2)])
def test_the_DECISION_TABLE_on_per_root_reads(dir_b, arm, total):
    """The rule in isolation, with the multiplicity reads controlled rather than solved for.

    Two roots inside one licence, each read counting ONE eigenvalue -- which is what attribution
    leaves behind for genuinely distinct roots. Parallel collapses to 1, orthogonal sums to 2,
    and the two inputs are identical in every respect a distance rule can see.
    """
    a, prec = 50.0, 1e-8
    b = a + 0.4 * prec * a
    out, pairs, coll, summed = _gram_group(_obj_with_directions(a, b, E1, dir_b),
                                           _levels_at(a, b, 1, 1), prec)
    assert pairs == 1
    assert (coll, summed) == ((1, 0) if arm == "collapse" else (0, 1))
    assert sum(x.mult for x in out) == total
    assert len(out) == 1, "both arms produce ONE level of S(prec); they differ in its count"


def test_a_DEGENERATE_eigenspace_located_twice_is_NOT_benign_and_says_so():
    """`search.md` claims the ambiguous eigenspace is harmless under both verdicts -- "parallel
    collapses and recounts to 2, orthogonal sums 1 + 1 to 2". **That assumes each member read 1,
    and under the reach rule it does not.**

    A read answers "how many eigenvalues are inside MY bracket", so a degenerate level located
    twice has both members reading the full multiplicity:

        each member reads   parallel (collapse)   orthogonal (sum)   truth
        1                          1                    2            both arms right
        2                          2                    4            the sum DOUBLES it

    Attribution is what makes the first row work, and it cannot help here: both members own the
    same eigenspace, so neither is the other's partner to claim. There is no safe guess -- capping
    by a recount over the union bracket is not it either, since `_collapse_ambiguous` records a
    spanning read folding four correct levels into two -- so the sum is FLAGGED and carries the
    alternative. `solve_interval` raises on a flag, which is loud and honest.
    """
    a, prec = 50.0, 1e-8
    b = a + 0.4 * prec * a

    par, _, coll, _ = _gram_group(_obj_with_directions(a, b, E1, E1),
                                  _levels_at(a, b, 2, 2), prec)
    assert (coll, sum(x.mult for x in par)) == (1, 2), "collapse recounts, and is right"
    assert not par[0].count_uncertain, "a collapse of equal reads is not in doubt"

    orth, _, _, summed = _gram_group(_obj_with_directions(a, b, E1, E2),
                                     _levels_at(a, b, 2, 2), prec)
    assert summed == 1
    assert sum(x.mult for x in orth) == 4, "the sum doubles it -- recorded, not hidden"
    assert orth[0].count_uncertain, "and it must NOT be certified"
    assert orth[0].count_alt == 2, "the other reading is carried"
    assert "degenerate" in orth[0].uncertain_reason


def test_a_collapse_whose_members_DISAGREE_is_flagged():
    """The angle says one eigenfunction; the two reads say different multiplicities. Which count
    the collapse should carry is then not decided, and pretending otherwise would pick one."""
    a, prec = 50.0, 1e-8
    b = a + 0.4 * prec * a
    out, _, coll, _ = _gram_group(_obj_with_directions(a, b, E1, E1),
                                  _levels_at(a, b, 1, 2), prec)
    assert coll == 1
    assert out[0].count_uncertain and out[0].count_alt == 1


def test_a_summed_level_carries_the_span_in_its_BAR():
    """Clause (2) honesty. A level standing for eigenvalues either side of its reported `lam`
    must have a bar that reaches them, and the span is bounded by the licence so this can never
    by itself put the level in `unmet`."""
    a, prec = 50.0, 1e-8
    b = a + 0.8 * prec * a
    out, _, _, summed = _gram_group(_obj_with_directions(a, b, E1, E2),
                                    _levels_at(a, b, 1, 1), prec)
    assert summed == 1
    lvl = out[0]
    assert lvl.delta_res >= 0.5 * (b - a)
    assert min(a, b) - lvl.delta_res <= lvl.lam <= max(a, b) + lvl.delta_res
    assert lvl.met(prec), "the span is inside the licence, so the bar still meets the request"
    assert lvl.spacings == (pytest.approx(b - a),)


def test_the_angle_is_NOT_asked_about_roots_further_apart_than_a_licence():
    """The distance prefilter's surviving role: it decides which pairs are worth testing, and
    anything wider is a separate level whatever the angle would say."""
    prec = 1e-8
    obj = _pair_objective([50.0, 50.0 * (1 + 1000 * prec)], SAME, prec=prec)
    res = solve(obj, prec=prec, phase_d="angle")
    assert res.diagnostics.n_gram_pairs == 0
    assert len(res.levels) == 2, "two levels a thousand licences apart are two levels"


def test_a_gram_decided_level_is_not_re_flagged_as_uncertain():
    """`_flag_uncertain_counts` raises exactly the question the angle answered, and
    `solve_interval` RAISES on a flag -- so re-raising a settled question would turn a correct
    answer into an exception."""
    lam, prec = 50.0, 1e-8
    res = solve(_pair_objective([lam, lam * (1 + 0.4 * prec)], SAME, prec=prec),
                prec=prec, phase_d="angle")
    collapsed = [x for x in res.levels if x.count_decided_by == "gram"]
    assert collapsed, "this cell must exercise the collapse arm"
    assert res.diagnostics.n_uncertain == 0, res.diagnostics.uncertain
    assert all(not x.grouping_ambiguous for x in collapsed), (
        "the grouping is DECIDED, not arbitrary -- saying otherwise is a lie about the one "
        "thing Phase D settles")


@pytest.mark.parametrize("name", ROSTER)
def test_every_ROSTER_case_carries_its_eigenfunctions(name):
    """The roster went without `Case.modes` until Phase D's rule was measured against it, and
    `solve(phase_d="angle")` refused the entire roster outright.

    `rect_suite` has carried it since the harness was written. The transfer domains are exactly
    the ones a grouping rule most needs trying on, so the omission was the wrong way round.
    """
    case = hr.ROSTER[name]()
    assert getattr(case, "modes", None) is not None
    u = case.modes(float(_levels(case)[0]))
    assert u.shape == (case.pencil.n_int,)
    assert np.linalg.norm(u) == pytest.approx(1.0, rel=1e-12)
