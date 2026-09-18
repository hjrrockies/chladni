"""Phase D's instrument: can the interior Gram angle tell a duplicate from a genuine pair?

`docs/design/search.md` section 7 specifies this file. It is written BEFORE the rule that will
use it, which is `CHLADNI_V1.md` section 5.1's discipline -- build the scorer before the searcher
-- and it is why the assertions here are about the ANGLE rather than about `solve`'s output. When
Phase D's rule lands, its own behaviour (collapse-and-recount against merge-and-sum, and the
ambiguous-eigenspace case giving the same total under both verdicts) is tested beside these.

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

pytestmark = pytest.mark.slow

#: The verdict boundary Phase D will use. Defined HERE and not in the package, because the rule
#: that reads it is not built: `search._SLOPE_FRAC` is what an unused package constant becomes.
#: It moves to `chladni.mps.multiplicity` with the rule, and this file imports it from there.
TAU_GRAM = 0.5

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
