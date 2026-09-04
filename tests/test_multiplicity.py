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
import v1_multiplicity as M
from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil
from chladni.reference import rect_eig
from v1_bracket import v1_brackets

pytestmark = pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")

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


def test_C1_is_a_RATIO_GAP_and_not_a_threshold():
    """No absolute level on `sigma` can serve: the floor varies FIVE ORDERS within one scan.
    So the same shape at two wildly different floors must read the same multiplicity."""
    for eps in (1e-15, 1e-10, 1e-5):
        def tens(x, e=eps):
            deep = [e, 1.3 * e, 1.0, 1.2]          # two branches at the floor, then the gap
            return np.array(deep) if x == 0.0 else np.array([9 * e, 12 * e, 1.0, 1.2])
        r = M.multiplicity(tens, 0.0, wide_bracket=(-1.0, 1.0))
        assert r.mu == 2, f"floor {eps:.0e} read {r.mu}"
        assert r.gap_ratio > M.GAP_RATIO


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


def test_a_single_branch_reads_multiplicity_one():
    def simple(x):
        return np.array([1e-15, 1.0, 1.2]) if x == 0.0 else np.array([1e-9, 1.0, 1.2])
    assert M.multiplicity(simple, 0.0, wide_bracket=(-1.0, 1.0)).mu == 1


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


@pytest.mark.slow
def test_the_reach_must_clear_the_LOCATED_ROOTS_OWN_ERROR():
    """`sigma_1` read at a REPORTED root is not the floor -- the root sits within `prec*lam` of
    the true minimum, so on a well-resolved pencil `sigma` there is orders above `eps`.

    Measured on the disk at `prec = 1e-8`: the reported root sits ~3e-09 from the minimum,
    `sigma` there is 6.2e-10 against a floor of 7.2e-16, and `2*sigma/c` gives a reach of
    6.4e-09. The flanks then straddle asymmetrically -- one side moves TOWARD the true minimum
    and one away -- prominence comes out at 1.45 against `P = 2`, (C2) rejects `sigma_1` itself,
    and a genuine multiplicity-2 level reads as 1. Two of seven disk levels went that way, and
    the isolated stage-3 tests could not see it because they read at SAMPLED points.
    """
    import hard_roster as hr
    from v1_bracket import bracket_interval
    from v1_refine import refine

    case = hr.ROSTER["disk"]()
    obj = bs.objective_for(case, 1e-8)
    res = bracket_interval(obj, prec=1e-8)
    roots = [refine(obj, b.lo, b.hi, prec=1e-8, lam_hint=b.k_hint)
             for b in res.brackets]
    roots = [r for r in roots if r.accepted]

    lv, mult = np.unique(case.truth, return_counts=True)
    doubles = [float(x) for x, m in zip(lv, mult) if m == 2]
    assert doubles, "the disk window must hold a degenerate level"

    xs, _ = obj.samples
    checked = 0
    for lam in doubles:
        near = [r for r in roots if abs(r.lam - lam) / lam < 1e-6]
        if not near:
            continue
        r = near[0]
        h = 0.005 * abs(r.lam)
        c = 0.5 * (obj.tension(r.lam + h) + obj.tension(r.lam - h)) / h
        naive = M.wide_bracket_for(xs, r.lam, r.sigma, c)
        aware = M.wide_bracket_for(xs, r.lam, r.sigma, c, prec=1e-8)
        assert min(r.lam - aware[0], aware[1] - r.lam) > min(r.lam - naive[0],
                                                             naive[1] - r.lam), (
            "knowing `prec` must widen the bracket")
        assert M.multiplicity(obj.tensions, r.lam, wide_bracket=aware, prec=1e-8).mu == 2
        checked += 1
    assert checked, "no degenerate level was reached"


# --- it spends nothing ------------------------------------------------------------------------

@pytest.mark.slow
def test_it_spends_NO_EVALUATIONS_on_a_shared_cache():
    """Every point it reads was paid for by the search. A multiplicity stage that quietly cost
    a GSVD per level would change the cost model it is scored under."""
    (case,) = [c for c in rs.family_D() if "mult=3" in c.label]
    obj = bs.objective_for(case, 1e-8)
    v1_brackets()(obj.tensions, obj.lo, obj.hi, prec=1e-8, objective=obj)
    xs, _ = obj.samples
    lam = float(np.unique(case.truth)[0])
    s1 = obj.tension(lam)
    h = 0.005 * lam
    c = 0.5 * (obj.tension(lam + h) + obj.tension(lam - h)) / h
    wb = M.wide_bracket_for(xs, lam, s1, c)

    before = obj.n_evals
    r = M.multiplicity(obj.tensions, lam, wide_bracket=wb, counter=obj)
    assert obj.n_evals == before, f"spent {obj.n_evals - before} evaluations"
    assert r.evaluations == 0
    assert r.mu == 3


# --- real pencils: the nine-for-nine measurement ------------------------------------------------

def _read_all(case, prec=1e-8):
    obj = bs.objective_for(case, prec)
    v1_brackets()(obj.tensions, obj.lo, obj.hi, prec=prec, objective=obj)
    xs, _ = obj.samples
    out = []
    for lam in np.unique(case.truth):
        lam = float(lam)
        s1 = obj.tension(lam)
        h = 0.005 * abs(lam)
        c = 0.5 * (obj.tension(lam + h) + obj.tension(lam - h)) / h
        wb = M.wide_bracket_for(xs, lam, s1, c)
        mu = M.multiplicity(obj.tensions, lam, wide_bracket=wb).mu if wb else 1
        out.append((lam, mu))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("name", ["disk", "eq_tri"])
def test_multiplicity_reads_correctly_off_a_smooth_spectrum(name):
    """The measurement that put this rule in the spec: nine for nine on the disk and the
    equilateral triangle, covering both mechanisms -- the disk's degeneracies are geometric
    (sin/cos pairs), the triangle's arithmetic (`m^2+mn+n^2` collisions)."""
    import hard_roster as hr

    case = hr.ROSTER[name]()
    got = _read_all(case)
    s = ls.score_levels(got, case.truth, 1e-8, label=name)
    assert s.ok, (f"{s.verdict}: separation {s.separation:.2f}, "
                  f"{s.n_mult_over} over / {s.n_mult_under} under")
    assert s.n_mult_right == s.n_levels_true


@pytest.mark.slow
@pytest.mark.parametrize("mult", [2, 3, 4])
def test_an_exact_degeneracy_reads_its_multiplicity(mult):
    (case,) = [c for c in rs.family_D() if f"mult={mult}" in c.label]
    got = _read_all(case)
    s = ls.score_levels(got, case.truth, 1e-8, label=case.label)
    assert s.ok, f"{s.verdict}: read {got}"
    assert max(m for _, m in got) == mult


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


# --- (C2) is load-bearing: the starved-basis regime ----------------------------------------------

def _starved(ndir, target_ratio):
    """A pencil whose basis is deliberately too small, with the pair placed at a chosen
    multiple of `eps/c`. Section 9.4's construction: it is `eps` that sets `eps/c`, so the
    dangerous regime is entered by having a mediocre basis."""
    probe = _starved_pencil(ndir, 1e-9)
    epsc = probe[2] / probe[3]
    delta = target_ratio * epsc / (6 * np.pi**2)
    return _starved_pencil(ndir, delta)


def _starved_pencil(ndir, delta):
    L = 1.0 + delta
    dom = rect(L, 1.0)
    la, lb = rect_eig(2, 1, L, 1.0), rect_eig(1, 2, L, 1.0)
    lam1 = min(la, lb)
    basis = PlaneWave.uniform(ndir, radius=dom.diameter)
    p = MPSPencil.from_basis(basis, dom.bdry_quad(3 * ndir),
                             dom.interior_pts(ndir, rng=np.random.default_rng(0)))
    eps = p.tension(lam1)
    h = 0.02 * lam1
    c = 0.5 * (p.tension(lam1 + h) + p.tension(lam1 - h)) / h
    return p, lam1, eps, c, abs(lb - la)


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.42, 0.86])
def test_C2_REJECTS_the_double_count_that_C1_alone_makes(ratio):
    """Section 9.4's measurement, reproduced. Near a simple eigenvalue whose neighbour is
    unresolved, the second branch is the OTHER cone's descending wall -- small, so it passes any
    gap test, but monotone rather than minimal. (C1) reads 2 at a SIMPLE eigenvalue and would do
    so again at its neighbour: the double count. (C2) rejects it."""
    p, lam1, eps, c, d = _starved(10, ratio)
    epsc = eps / c
    assert d / epsc == pytest.approx(ratio, rel=0.5), "the case landed in the ambiguous band"

    reach = M.WIDE_FACTOR * epsc
    r = M.multiplicity(lambda z: p.tensions(float(z)), lam1,
                       wide_bracket=(lam1 - reach, lam1 + reach))
    assert r.c1_read == 2, "the gap test alone says 2"
    assert r.mu == 1, "and (C2) cuts it to 1"
    assert r.c2_rejected


@pytest.mark.slow
def test_the_dangerous_regime_is_entered_by_having_a_MEDIOCRE_BASIS():
    """The consequence larger than the multiplicity rule: it is `eps` that sets `eps/c`, so a
    black-box solver whose planner produces a merely adequate basis is in the double-counting
    regime by default. Basis quality and multiplicity correctness are the same problem."""
    starved = _starved_pencil(10, 3.28e-5)
    good = _starved_pencil(40, 3.28e-5)
    ratio_bad = starved[4] / (starved[2] / starved[3])
    ratio_good = good[4] / (good[2] / good[3])
    assert ratio_bad < 1.0, f"10 directions: d/(eps/c) = {ratio_bad:.2f}, unresolvable"
    assert ratio_good > 20.0, f"40 directions: {ratio_good:.3g}, inside the contract"
    assert good[2] < starved[2] / 1e6, "and the only thing that changed is the floor"
