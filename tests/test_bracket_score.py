"""The metric, validated against controls before any search is believed.

`CHLADNI_V1.md` section 5.2, both halves: six adversaries that must fail, each by the criterion
meant for it, and five oracles that must pass. Tier 1 -- a surrogate cone objective, no pencil,
no GSVD -- so it stays in the default run.

The oracles are the half that gets forgotten and the half that matters most here. An adversary
suite alone cannot see a metric that is too STRICT to accept a correct answer, and that is the
failure which looks like rigour.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_controls as bc
import bracket_score as bs
from baselines import BASELINES
from objective import Objective

#: A window with an isolated level, a resolvable pair and a wide-open gap. Real rectangle
#: values: 5*pi**2 is the (2,1)/(1,2) degeneracy, split here by a controlled amount.
PAIR = 5 * np.pi**2
TRUTH = np.array([2 * np.pi**2, PAIR, PAIR * (1 + 3e-7), 8 * np.pi**2, 10 * np.pi**2])
WINDOW_LAM = (15.0, 110.0)


def cone_objective(truth=TRUTH, window=WINDOW_LAM, *, variable="lam", prec=1e-8,
                   eps=1e-13, c_k=0.27, plateau=0.2):
    """The measured local model, one cone per true eigenvalue, on a plateau.

    `sigma_1 ~ sqrt(eps**2 + c**2*(k - k*)**2)` fits the far field to 1-3% on real pencils, so
    a scorer test built on it is testing the scorer rather than a caricature.

    **The cone is built in `k` whatever the objective's variable is, and that is not a
    detail.** `c` has units of length and is measured in `k`; the first version of this helper
    used `c = 0.27` directly in `lam` and produced dips 15x too narrow to sample, at which
    point the uniform-grid baseline scored coverage 0 and looked incompetent. Converting
    properly -- `dsigma/dlam = c_k / (2*sqrt(lam))` -- gives a dip half-width of 5.2 at
    `lam ~ 49` against a mean gap of 19.7, i.e. dips occupying about half the spacing, which
    is the independently measured invariant (0.46-0.52 across four domains). That the change
    of variables reproduces that number is a check on the surrogate, and worth stating: a cone
    written in the wrong variable is wrong by the factor `2*k`, and that factor is exactly what
    made the honest baseline look broken.
    """
    k_star = np.sqrt(np.asarray(truth, dtype=float))

    def tensions(x):
        k = float(x) if variable == "k" else np.sqrt(float(x))
        d = np.sort(np.abs(k_star - k))
        v = np.sqrt(eps**2 + (c_k * d) ** 2)
        return np.minimum(v, plateau)

    lo, hi = window
    if variable == "k":
        lo, hi = np.sqrt(lo), np.sqrt(hi)
    return Objective(tensions, lo, hi, prec=prec, variable=variable)


# --- the adversaries: each must fail, and by the criterion meant for it --------------------

def test_whole_window_fails_purity():
    obj = cone_objective()
    s = bs.run_bracketer(bc.whole_window(), obj, TRUTH, 1e-8, label="whole_window")
    assert not s.ok
    assert s.verdict == "purity"
    # It scores coverage 1.0 and spurious 0 while being useless: purity is the ONLY thing that
    # can reject it, which is why the verdict matters and not merely the failure.
    assert s.coverage == 1.0 and s.spurious == 0


def test_whole_window_is_CORRECT_on_a_single_level():
    """The three cases section 29 records it passing are the exact degeneracies, which hold
    ONE level -- where one bracket is the right answer and passing is not a metric defect."""
    one = np.array([PAIR, PAIR, PAIR])
    obj = cone_objective(one, (48.0, 51.0))
    s = bs.run_bracketer(bc.whole_window(), obj, one, 1e-8, label="whole_window/degenerate")
    assert s.ok and s.verdict == "pass"


def test_unevaluated_partition_is_rejected_only_by_COST():
    """The single most instructive adversary: it satisfies coverage, purity AND a search-only
    budget. Nothing but a cost model charging the handed-on refinement can reject it."""
    prec = 1e-3          # loose enough that a purity-guaranteeing partition materializes
    obj = cone_objective(prec=prec)
    s = bs.run_bracketer(bc.grid_partition_unevaluated(), obj, TRUTH, prec,
                         label="grid_partition_unevaluated")
    assert s.coverage == 1.0, "it does cover everything"
    assert s.pure, "and it is pure, by construction"
    assert s.cost.search_evals == 0, "having evaluated nothing"
    assert not s.ok and s.verdict == "cost"


def test_unevaluated_partition_would_PASS_on_a_search_only_budget():
    """Stated as its own test because it is the reason `CostReport` exists. Score the same
    bracket set charging only what the search spent, and the adversary walks through."""
    prec = 1e-3
    obj = cone_objective(prec=prec)
    raw = bc.grid_partition_unevaluated()(obj.tensions, obj.lo, obj.hi, prec=prec)
    search_only = bs.score_brackets(bs.to_lam_brackets(raw, obj.variable), TRUTH, prec,
                                    cost=bs.CostReport(search_evals=obj.n_evals))
    assert search_only.ok, "which is exactly the hole a flat search-only budget leaves"


def test_unevaluated_partition_fails_on_purity_at_tight_prec():
    """At tight `prec` the purity-guaranteeing count is not materializable (~4e10 cells), the
    partition is truncated, and the rejection moves to purity. Both regimes are correct
    rejections and the sweep should record which one it was in."""
    prec = 1e-9
    obj = cone_objective(prec=prec)
    s = bs.run_bracketer(bc.grid_partition_unevaluated(max_cells=2000), obj, TRUTH, prec,
                         label="grid_partition_unevaluated/tight")
    assert not s.ok
    assert s.verdict in ("purity", "cost")


def test_evaluated_fine_grid_fails_on_cost():
    obj = cone_objective()
    s = bs.run_bracketer(bc.evaluated_fine_grid(n=4000), obj, TRUTH, 1e-8,
                         label="evaluated_fine_grid")
    assert not s.ok
    # It fails purity too, and correctly: a grid's cells are orders wider than `prec * lam`,
    # which is the measured reason a uniform grid "holds coverage and fails purity". So the
    # claim is that COST is among the criteria rejecting it, not that it is the only one.
    assert "cost" in s.failures, s.failures
    assert s.cost.search_evals >= 4000


def test_oracle_minus_one_fails_coverage():
    obj = cone_objective()
    s = bs.run_bracketer(bc.oracle_minus_one(TRUTH), obj, TRUTH, 1e-8,
                         label="oracle_minus_one")
    assert not s.ok and s.verdict == "coverage"
    assert s.covered == s.n_true - 1


def test_oracle_two_merged_fails_purity():
    obj = cone_objective()
    s = bs.run_bracketer(bc.oracle_two_merged(TRUTH), obj, TRUTH, 1e-8,
                         label="oracle_two_merged")
    assert not s.ok and s.verdict == "purity"
    assert s.coverage == 1.0, "everything is still covered -- purity is the discriminator"


def test_brackets_in_the_gaps_fails_coverage():
    obj = cone_objective()
    s = bs.run_bracketer(bc.brackets_in_the_gaps(TRUTH), obj, TRUTH, 1e-8,
                         label="brackets_in_the_gaps")
    assert not s.ok and s.verdict == "coverage"
    assert s.covered == 0 and s.n_brackets > 0, "returned plenty, found nothing"


@pytest.mark.parametrize("name", sorted(bc.ADVERSARIES))
def test_every_adversary_fails_by_its_named_criterion(name):
    """The table itself, so a newly added adversary cannot be added without a verdict."""
    factory, expected = bc.ADVERSARIES[name]
    prec = 1e-3 if expected == "cost" else 1e-8
    obj = cone_objective(prec=prec)
    fn = factory(TRUTH) if "oracle" in name or "gaps" in name else factory()
    s = bs.run_bracketer(fn, obj, TRUTH, prec, label=name)
    assert not s.ok, f"{name} must fail"
    assert expected in s.failures, (
        f"{name} must be rejected by {expected}, was rejected by {s.failures}")


# --- declining: never a pass ---------------------------------------------------------------

def test_a_decliner_reports_declined_and_never_passes():
    """It returns a PERFECT partial result, to prove that a decline cannot be laundered into
    a pass by the partial happening to look good."""
    obj = cone_objective()
    s = bs.run_bracketer(bc.declines_everything(), obj, TRUTH, 1e-8, label="decliner")
    assert s.declined
    assert s.verdict == "declined"
    assert not s.ok
    assert s.decline_reason


def test_a_declined_score_still_reports_its_partial_result():
    """Declining is a third outcome, not an erasure: what it found stays inspectable."""
    obj = cone_objective()
    s = bs.run_bracketer(bc.declines_everything(), obj, TRUTH, 1e-8)
    assert s.n_brackets == 1 and s.coverage == 1.0
    assert not s.ok, "and it is still not a pass"


# --- the oracles: each must pass -----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(bc.ORACLES))
@pytest.mark.parametrize("prec", [1e-4, 1e-6, 1e-8, 1e-10])
def test_every_oracle_passes(name, prec):
    """A metric too strict to accept a correct answer is the failure that looks like rigour."""
    obj = cone_objective(prec=prec)
    s = bs.run_bracketer(bc.ORACLES[name](TRUTH, variable=obj.variable), obj, TRUTH, prec,
                         label=f"{name}@{prec:.0e}")
    assert s.ok, f"{name} at prec={prec:.0e} must pass, got {s.verdict}"
    assert s.spurious == 0


@pytest.mark.parametrize("name", sorted(bc.ORACLES))
def test_every_oracle_passes_in_the_k_variable_too(name):
    """The conversion and the 4-ulp widening are applied by the harness, so an oracle emitting
    in `k` must score identically to one emitting in `lam`. A one-ulp coverage loss on the
    round trip was observed, and its occurrence depended only on seed density."""
    prec = 1e-8
    obj = cone_objective(variable="k", prec=prec)
    s = bs.run_bracketer(bc.ORACLES[name](TRUTH, variable="k"), obj, TRUTH, prec,
                         label=f"{name}/k")
    assert s.ok, f"{name} in k must pass, got {s.verdict}"


def test_the_lattice_ends_are_both_accepted_and_bound_the_count():
    """Fine end = one bracket per true eigenvalue, coarse end = one per level of S(prec).
    Both pass, and the count runs between them."""
    prec = 1e-6      # coarse enough that the 3e-7 pair is ONE level
    obj = cone_objective(prec=prec)
    fine = bs.run_bracketer(bc.oracle_fine(TRUTH), obj, TRUTH, prec, label="fine")
    coarse = bs.run_bracketer(bc.oracle_coarse(TRUTH), obj, TRUTH, prec, label="coarse")
    assert fine.ok and coarse.ok
    assert coarse.n_brackets < fine.n_brackets, "the pair merges at this prec"
    assert coarse.n_brackets == coarse.n_levels_prec
    assert fine.n_brackets == fine.n_true
    assert fine.in_lattice and coarse.in_lattice


def test_splitting_a_within_prec_cluster_is_not_a_failure():
    """Merging is permitted and never required. Penalising a split would contradict the
    multiplicity convention: downstream the refiner finds both members and Phase D returns two
    levels of multiplicity 1."""
    prec = 1e-6
    obj = cone_objective(prec=prec)
    s = bs.run_bracketer(bc.oracle_split_cluster(TRUTH), obj, TRUTH, prec, label="split")
    assert s.ok
    assert s.n_brackets > s.n_levels_prec, "it did split the cluster"


def test_an_exact_degeneracy_is_one_bracket_and_that_is_correct():
    """No interval can separate a level from itself, so asking for one bracket per MODE at an
    exact degeneracy is a scorer bug. Score against distinct locations with multiplicity."""
    truth = np.array([2 * np.pi**2, PAIR, PAIR, PAIR])
    obj = cone_objective(truth, (15.0, 60.0))
    s = bs.run_bracketer(bc.oracle_split_cluster(truth, variable=obj.variable), obj, truth,
                         1e-8, label="degenerate")
    assert s.ok and s.verdict == "pass"


# --- the baselines fail in OPPOSITE ways ---------------------------------------------------

@pytest.mark.parametrize("name", sorted(BASELINES))
def test_a_baseline_never_reads_truth(name):
    """The property that separates a baseline from a control. Called with an objective whose
    truth it has no access to, it must still return something."""
    obj = cone_objective()
    out = BASELINES[name]()(obj.tensions, obj.lo, obj.hi, prec=1e-8)
    assert isinstance(out, list)


def test_the_two_baselines_fail_in_opposite_ways():
    """The check that the scoring is not merely agreeing with itself: if both failed the same
    way, coverage and purity would be one metric rather than two."""
    prec = 1e-9
    grid = bs.run_bracketer(BASELINES["uniform_grid"](), cone_objective(prec=prec),
                            TRUTH, prec, label="uniform_grid")
    bisect = bs.run_bracketer(BASELINES["grid_then_bisect"](), cone_objective(prec=prec),
                              TRUTH, prec, label="grid_then_bisect")
    assert grid.coverage == 1.0 and not grid.pure, "the grid holds coverage, fails purity"
    assert bisect.pure and bisect.coverage < 1.0, "bisect holds purity, loses eigenvalues"


# --- separation classification -------------------------------------------------------------

def test_a_case_is_classified_by_the_separation_it_ACHIEVED():
    """Requesting an unrepresentable separation rounds the perturbation to zero and silently
    yields an exact degeneracy wearing the split-pair label."""
    requested = np.array([PAIR, PAIR * (1 + 1e-30)])   # unrepresentable: snaps to identical
    sep = bs.classify_separation(requested, 1e-14)
    assert sep.degenerate, "achieved zero, whatever was asked for"
    assert sep.exercisable, "an exact degeneracy is a legitimate class, not an empty cell"
    assert not sep.must_split


def test_a_separation_between_bands_is_reported_UNEXERCISABLE():
    """Report the cell of the sweep as empty and say why, rather than dropping it or
    relabelling it. Section 29 claimed prec=1e-14 could not be exercised and 29.1 retracted
    that -- an unexercisable cell is a reportable measurement."""
    sep = bs.classify_separation(np.array([PAIR, PAIR * (1 + 7e-8)]), 1e-8)
    assert sep.band is None and not sep.exercisable
    assert sep.ratio == pytest.approx(7.0, rel=0.1)


@pytest.mark.parametrize("band", bs.SEPARATION_BANDS)
@pytest.mark.parametrize("prec", bs.PREC_LADDER)
def test_every_band_of_every_precision_classifies_to_itself(band, prec):
    """The sweep is a CROSS PRODUCT: each band has to land on its own label at each of the six
    precisions, or the crossover is not being located per precision."""
    truth = np.array([PAIR, PAIR * (1 + band * prec)])
    sep = bs.classify_separation(truth, prec)
    if sep.min_gap == 0.0:
        pytest.skip(f"band {band} at prec {prec:.0e} is below the float64 floor here")
    assert sep.band == pytest.approx(band), f"got ratio {sep.ratio:.3g}"
    assert sep.must_split == (band > 1.0)


def test_the_tightest_pair_is_read_from_the_WHOLE_window():
    """The tightest pair is not the one you are dialling. Relative separation scales as
    |m^2-n^2|/(m^2+n^2): 3/5 for the (2,1)/(1,2) pair but 7/25 for (3,4)/(4,3), so a case
    placed by the former leaves the latter twice as tight as intended."""
    dialled = 1e-6
    truth = np.array([PAIR, PAIR * (1 + dialled), 80.0, 80.0 * (1 + dialled / 10)])
    sep = bs.classify_separation(truth, 1e-6)
    assert sep.at[0] == pytest.approx(80.0), "it found the untended pair, not the dialled one"
    assert sep.ratio == pytest.approx(0.1, rel=0.05)


# --- the cost model ------------------------------------------------------------------------

def test_the_refine_charge_is_not_flat():
    """A flat per-bracket charge overcharged tight brackets threefold and undercharged
    spurious ones by the same factor -- and reversed an architectural conclusion twice."""
    tight = bs.predicted_refine_cost(1e-12, PAIR, 1e-14)
    wide = bs.predicted_refine_cost(20.0, PAIR, 1e-14)
    assert tight < wide, f"tight {tight:.2f} vs wide {wide:.2f}"
    # Fitted on tier 2 (126 real refinements), not on V1's quoted anchors: measured cost runs
    # 4 to 10 over widths 1e-12 to 1e-1, so the model must sit inside that band.
    assert 3.5 <= tight <= 5.0, f"tight bracket {tight:.2f}"
    assert 6.0 <= wide <= 10.0, f"wide bracket {wide:.2f}"
    assert bs.REJECT_COST == 3.0, "measured flat across 10 empty brackets"


def test_a_predicted_charge_is_marked_unmeasured():
    obj = cone_objective()
    s = bs.run_bracketer(bc.oracle_fine(TRUTH), obj, TRUTH, 1e-8)
    assert not s.cost.measured, "tier 1 has no pencil; say so rather than implying otherwise"
    assert len(s.cost.per_bracket) == s.n_brackets


def test_the_budget_is_per_eigenvalue_and_never_flat():
    """A flat cap silently penalises dense windows: section 29.2's flat 1500 turned a
    38-evaluation miss into 'nobody handles family E'."""
    sparse = bs.score_brackets([(48.0, 50.0)], [PAIR], 1e-8,
                               cost=bs.CostReport(search_evals=300))
    dense = bs.score_brackets([(48.0, 50.0)], [PAIR] * 20, 1e-8,
                              cost=bs.CostReport(search_evals=300))
    assert sparse.per_eigenvalue == pytest.approx(300.0)
    assert dense.per_eigenvalue == pytest.approx(15.0)
    assert not sparse.cost_ok and dense.cost_ok


def test_cost_is_measured_per_bracketer_not_inherited():
    """The objective is reset per run: a candidate that inherited a warm cache would be scored
    on someone else's evaluations."""
    obj = cone_objective()
    first = bs.run_bracketer(bc.evaluated_fine_grid(n=200), obj, TRUTH, 1e-8)
    second = bs.run_bracketer(bc.oracle_fine(TRUTH), obj, TRUTH, 1e-8)
    assert first.cost.search_evals >= 200
    assert second.cost.search_evals < 10
