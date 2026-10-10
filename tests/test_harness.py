"""The scorer has to detect the failures it exists to detect.

A harness that cannot see a missed eigenvalue is worse than no harness, because it converts
"we did not check" into "we checked and it was fine". So every test here feeds it a search
that is broken in a KNOWN way and asserts the score reflects that break -- and only then is
the real baseline scored, on cases whose answers come from the surrogate by construction.

The four failures that matter, in the order they matter:

    a missed eigenvalue          recall < 1
    a spurious one               precision < 1
    a cluster reported twice     spurious, NOT a second match
    a wrong multiplicity         n_mult_wrong > 0, and recall counts the shortfall
"""
import numpy as np
import pytest

from harness import (Case, Score, levels_agree, run, score, table, uniform_grid_search)
from surrogate import Surrogate, Well

TRUTH = [(3.0, 1), (5.0, 2), (7.0, 1)]


# --- the scorer sees what it must ---------------------------------------------------------


def test_a_perfect_search_scores_perfectly():
    s = score(TRUTH, TRUTH)
    assert s.recall == 1.0 and s.precision == 1.0 and s.perfect
    assert s.n_true == 4 and s.n_matched == 4


def test_a_missed_eigenvalue_shows_up_as_recall():
    s = score([(3.0, 1), (5.0, 2)], TRUTH)
    assert s.n_missed == 1
    assert s.recall == pytest.approx(3 / 4)
    assert s.missed_at == [7.0]


def test_a_missed_DEGENERATE_eigenvalue_costs_its_full_multiplicity():
    """Missing a double is two eigenvalues missed, not one. Counting distinct locations
    would make a solver that never resolves multiplicity look almost perfect."""
    s = score([(3.0, 1), (7.0, 1)], TRUTH)
    assert s.n_missed == 2 and s.recall == pytest.approx(2 / 4)


def test_a_spurious_root_shows_up_as_precision():
    s = score(TRUTH + [(4.0, 1)], TRUTH)
    assert s.n_spurious == 1 and s.recall == 1.0
    assert s.precision == pytest.approx(4 / 5)


def test_reporting_one_root_twice_is_spurious_not_a_second_match():
    """The double-counting failure of `lipschitz-cone-mps.md` section 5: a cluster found
    from two brackets and reported twice, each time with its full multiplicity. A scorer
    that matched greedily without claiming would score this perfect."""
    s = score([(3.0, 1), (5.0, 2), (5.0 + 1e-9, 2), (7.0, 1)], TRUTH)
    assert s.n_matched == 4
    assert s.n_spurious == 2
    assert s.precision < 1.0


def test_a_wrong_multiplicity_is_counted_and_costs_recall():
    s = score([(3.0, 1), (5.0, 1), (7.0, 1)], TRUTH)      # called the double a simple
    assert s.n_mult_wrong == 1
    assert s.n_matched == 3 and s.n_missed == 1


def test_an_overcounted_multiplicity_is_spurious():
    s = score([(3.0, 1), (5.0, 3), (7.0, 1)], TRUTH)      # called the double a triple
    assert s.n_mult_wrong == 1 and s.n_spurious == 1
    assert s.n_matched == 4


def test_undercounting_a_multiplet_is_told_apart_from_missing_an_eigenvalue():
    """Both cost the same recall, and they are different bugs with different fixes. The
    split is what `n_missed_distinct` is for: it counts only the truth entries with NOTHING
    of the search's nearby, so undercounting a level the search did find does not read as
    having failed to find it."""
    under = score([(3.0, 1), (5.0, 1), (7.0, 1)], TRUTH)   # found the double, called it 1
    gone = score([(3.0, 1), (5.0, 2)], TRUTH)              # never saw 7.0 at all

    assert under.n_missed == gone.n_missed == 1            # same cost to recall
    assert under.n_missed_distinct == 0 and under.n_mult_wrong == 1
    assert gone.n_missed_distinct == 1 and gone.n_mult_wrong == 0


def test_both_sides_are_unfolded_to_multisets():
    """Truth arrives from `chladni.reference` already unfolded and a search returns pairs;
    they meet here. A repeated value IS a degeneracy, because a true one is bit-identical."""
    from harness import unfold

    assert list(unfold([(3.0, 1), (5.0, 2)])) == [3.0, 5.0, 5.0]
    assert list(unfold(np.array([5.0, 3.0, 5.0]))) == [3.0, 5.0, 5.0]
    assert score([(5.0, 2)], np.array([5.0, 5.0])).perfect


def test_a_near_pair_is_not_a_degeneracy():
    """Two distinct eigenvalues 1e-9 apart are two eigenvalues. A search reporting one of
    them as a double has both undercounted and overcounted, and neither is hidden."""
    truth = np.array([49.0, 49.0 * (1 + 1e-9)])
    s = score([(49.0, 2)], truth, rtol=1e-12)
    assert s.n_missed == 1 and s.n_mult_wrong >= 1


def test_matching_respects_the_tolerance():
    assert score([(3.0 * (1 + 1e-7), 1)], [(3.0, 1)], rtol=1e-6).n_matched == 1
    assert score([(3.0 * (1 + 1e-5), 1)], [(3.0, 1)], rtol=1e-6).n_matched == 0


def test_the_table_renders():
    assert "recall" in table([score(TRUTH, TRUTH, label="x")])


# --- and now the baseline, on cases whose answers are known by construction ---------------


def _case(label, wells, a, b, **kw):
    s = Surrogate(wells, **kw)
    return Case(label, s.tensions, a, b, s.truth, counter=s)


CASES = [
    _case("simple x3", [Well(3.0, eps=1e-10), Well(5.0, eps=1e-10), Well(7.0, eps=1e-10)],
          2.0, 8.0, n=16),
    _case("with a double", [Well(3.0, eps=1e-10), Well(5.0, eps=1e-10, mult=2)],
          2.0, 6.0, n=16),
    _case("wide separation", [Well(4.0, eps=1e-12), Well(9.0, eps=1e-12)], 2.0, 11.0, n=16),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.label)
def test_the_uniform_grid_baseline_is_not_broken(case):
    """Not a claim that the grid is good -- a claim that it works on easy cases, so that a
    poor score on a hard one means the case is hard rather than the baseline is broken."""
    s = run(uniform_grid_search(), [case])[0]
    assert s.recall == 1.0, f"baseline missed {s.missed_at}"
    assert s.n_spurious == 0


def test_the_baseline_reports_multiplicity_on_a_true_double():
    case = _case("double", [Well(5.0, eps=1e-10, mult=2)], 4.0, 6.0, n=16)
    s = run(uniform_grid_search(), [case])[0]
    assert s.n_mult_wrong == 0 and s.recall == 1.0


def test_evaluations_are_counted():
    s = run(uniform_grid_search(), [CASES[0]])[0]
    assert s.evaluations > 0


# --- the coarse view: did it get the levels and multiplicities right? --------------------


def test_levels_agree_is_a_yes_no_on_values_and_multiplicities():
    assert levels_agree(TRUTH, TRUTH)
    assert not levels_agree([(3.0, 1), (5.0, 1), (7.0, 1)], TRUTH)   # undercounted
    assert not levels_agree([(3.0, 1), (5.0, 2)], TRUTH)             # missed one
    assert not levels_agree(TRUTH + [(4.0, 1)], TRUTH)               # spurious


def test_levels_agree_tolerates_a_degeneracy_that_is_not_bit_identical():
    """The case `snap` cannot reach: a reference whose repeated eigenvalues were arrived at
    separately and agree only to their own precision -- a tabulated spectrum, or a closed
    form with an irrational parameter. Collapsing both sides at the comparison tolerance is
    what makes those comparable at all.

    Measured: `rect(sqrt(2), 1)` has true degeneracies whose computed values differ by
    4.4e-16 before `snap`, and a tabulated pair may differ by far more.
    """
    truth = np.array([49.0, 49.0 * (1 + 3e-13)])       # one double, imperfectly computed
    assert levels_agree([(49.0, 2)], truth, rtol=1e-9)
    assert not levels_agree([(49.0, 2)], truth, rtol=1e-15)


def test_levels_ok_rides_along_on_every_score():
    assert score(TRUTH, TRUTH).levels_ok
    assert not score([(3.0, 1), (5.0, 1), (7.0, 1)], TRUTH).levels_ok


def test_the_reference_hands_back_bit_identical_degeneracies():
    """`snap` exists so this holds even where the arithmetic is not symmetric."""
    from chladni.reference import rect_spectrum

    for L in (1.0, 2.0, np.sqrt(2.0), np.sqrt(3.0)):
        sp = rect_spectrum(0.0, 400.0, L, 1.0)
        close = [(sp[i], sp[i + 1]) for i in range(sp.size - 1)
                 if 0 < abs(sp[i] - sp[i + 1]) / sp[i + 1] < 1e-13]
        assert not close, f"L={L}: degenerate values not made identical: {close}"
