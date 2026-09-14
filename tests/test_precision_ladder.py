"""The full pipeline on every real case, at EVERY precision in the ladder, one test per cell.

`bracket_score.PREC_LADDER` is `(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)`. `test_gate.py` sweeps
it on the BRACKETING stage; this file drives the composed pipeline -- bracket, refine,
multiplicity -- across all of it, on the rectangle families and the hard roster.

# Scoring

Each (case, prec) cell is scored by `tests/prec_score.py` against COMPLETE truth, and the only
excuses it grants come from `tests/basis_sufficiency.py`:

    pass                  complete, unspurious, counted right, every level within prec
    declined              declined WITH a reason (`candidates.md` section 5)
    b-excused             short of prec, reported with an honest bar, where the basis cannot
                          serve prec
    merge-/wrong-total-excused
                          a merge or miscount where the basis cannot resolve multiplicity

Merging within `prec*lam` is permitted and never required, an exact degeneracy may be reported
split, and the total is scored everywhere -- there is no unscored band.

# What an earlier version of this file got wrong

It accepted `b` at every precision and explained it as "these pencils floor near 1e-10". They do
not: the tension at the true eigenvalues is ~1.5e-15 on the rectangles, 1.7e-15 on the disk and
5.8e-15 on the L-shape. The refiner was stopping early -- a vertex formula that cancelled and a
repeated-point exit with no width check -- and `test_the_cost_..._saturates` pinned the flat
evaluation count that produced as the expected behaviour. See `tests/test_refine_regression.py`.

It also scored rectangles against `rect_spectrum`, which snaps values within `1e-13*lam`, so the
(2,1)/(1,2) pair on `rect(1 + 3e-14, 1)` was truth's exact double -- and a solver that split it
correctly at `prec = 1e-14` read as a wrong multiplicity. Truth is now `rect_suite.complete_truth`.

Cells are built lazily: collection names them by family and index, so the default run, which
deselects this file, never assembles a pencil.

Set `CHLADNI_LADDER_LOG=path.jsonl` to append one JSON record per scored cell -- verdict, error,
counts, cost, the basis's excuses, the worst tension at truth, and every level's bar. It is the
report the grid summarizes, and it is off by default. The refiner's own reason code is not in it:
`Level` does not keep it.
"""
from __future__ import annotations

import functools
import json
import os
import warnings

import numpy as np
import pytest

import hard_roster as hr
import rect_suite as rs
from basis_sufficiency import sufficient
from bracket_score import PREC_LADDER
from chladni import asymp
from chladni.mps.bracket import BUDGET_PER_EIG
from chladni.mps.objective import Objective
from chladni.mps.search import SolveResult, solve
from prec_score import score_cell

pytestmark = pytest.mark.slow

_FAMILIES = {"A": rs.family_A, "B": rs.family_B, "C": rs.family_C, "D": rs.family_D,
             "S": rs.family_stress}
_SIZES = {"A": 1, "B": len(rs.B_DELTAS), "C": len(rs._C_LEVELS) * len(rs.C_DELTAS),
          "D": len(rs._D_LEVELS), "S": 1}
CASES = [(f, i) for f, n in _SIZES.items() for i in range(n)] + [("roster", k) for k in hr.ROSTER]

#: Cells known to fail, keyed `(family, index, prec)`, each with the evidence. Strict: when one
#: starts passing the suite goes red until its entry comes off, so a fix cannot land silently.
KNOWN = {
    ("C", 2, 1e-4): (
        "Phase 3: the collapse defect. The three members at 50*pi^2 span 1.92e-04 relative, wider "
        "than the licence, and solve returns ONE level of multiplicity 1 flagged "
        "grouping_ambiguous -- two members dropped. At 1e-06 all three come back."),
    ("C", 6, 1e-4): (
        "Phase 3: the collapse defect. The four members at 65*pi^2 span 1.94e-04 relative and "
        "solve returns one level of multiplicity 1 -- three dropped. At 1e-06 all four return."),
    ("roster", "eq_tri", 1e-14): (
        "Phase 2: the basis serves accuracy only to 2.5e-12 on an rtol derived from prec, and "
        "4 of 6 levels are dropped WITHOUT a warning -- TODO section 4, the silent-drop item."),
}

#: Cases not yet on a frozen basis. `hard_roster` builds its pencils at the default `prec`, so
#: their `rtol` moves with it. Plan Phase 2 freezes them.
UNFROZEN = {("roster", k) for k in hr.ROSTER}

_CODES = {"pass": ".", "declined": "D", "b-excused": "b", "b-fail": "B", "shortfall-silent": "!",
          "bar-violated": "?", "spurious": "S", "missed": "M", "wrong-total": "T",
          "wrong-total-excused": "t", "merge-above-licence": "G", "merge-excused": "g",
          "decline-unreasoned": "X"}


@functools.lru_cache(maxsize=None)
def _family(f):
    return tuple(_FAMILIES[f]())


@functools.lru_cache(maxsize=None)
def _case(f, i):
    return hr.ROSTER[i]() if f == "roster" else _family(f)[i]


@functools.lru_cache(maxsize=None)
def _cell(f, i, prec):
    """`(case, truth, sufficiency, score, evaluations)` for one cell, computed once."""
    case = _case(f, i)
    truth = (np.sort(np.asarray(case.truth, dtype=float)) if f == "roster"
             else rs.complete_truth(case))
    # A prefix table: a level may sit up to the REQUEST past the last entry and still be that
    # entry. A fixed 1e-9 held only while the refiner stopped at ~6e-10 whatever it was asked;
    # fixed, it correctly stops 5.4e-06 from 2*pi**2 at `prec = 1e-4` and was cut away.
    hi = float(case.b) if case.complete else float(truth[-1]) * (1 + max(4 * prec, 1e-9))
    obj = Objective.from_pencil(
        case.pencil, case.a, case.b, variable="lam", prec=prec,
        expected_count=lambda lo, up: asymp.expected_count(lo, up, case.domain))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve(obj, prec=prec)
    if not case.complete:
        # A prefix table: levels above its last entry cannot be scored either way.
        res = SolveResult(tuple(L for L in res.levels if L.lam <= hi), res.diagnostics)
    suff = sufficient(case, prec)
    score = score_cell(res, truth, prec, suff)
    return case, truth, suff, score, res.diagnostics.evaluations, res


_LOG = os.environ.get("CHLADNI_LADDER_LOG")


def _record(f, i, prec):
    """One cell as a JSON-ready dict. Reads the cache; the tension at truth costs a few calls."""
    case, truth, suff, score, evals, res = _cell(f, i, prec)
    floor = max(float(case.pencil.tension(float(x))) for x in np.unique(truth))
    return dict(label=case.label, family=f, index=str(i), prec=prec, verdict=score.verdict,
                acceptable=score.acceptable, detail=score.detail,
                worst_rel_err=score.worst_rel_err, n_true=score.n_true,
                found_total=score.found_total, n_levels=score.n_levels, evaluations=int(evals),
                basis_accuracy=suff.accuracy, basis_multiplicity=suff.multiplicity,
                frozen=suff.frozen, reasons=list(suff.reasons), floor_at_truth=floor,
                levels=[dict(lam=L.lam, mult=L.mult, delta_res=L.delta_res,
                             grouping_ambiguous=L.grouping_ambiguous) for L in res.levels])


def _write_log(f, i, prec, path=None):
    path = path or _LOG
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(_record(f, i, prec)) + "\n")


def _params():
    for f, i in CASES:
        for prec in PREC_LADDER:
            why = KNOWN.get((f, i, prec))
            marks = [pytest.mark.xfail(strict=True, reason=why)] if why else []
            yield pytest.param(f, i, prec, marks=marks, id=f"{f}{i}-{prec:.0e}")


@pytest.mark.parametrize("f, i, prec", list(_params()))
def test_the_cell_is_acceptable(f, i, prec):
    case, truth, suff, score, _, _ = _cell(f, i, prec)
    _write_log(f, i, prec)
    assert score.acceptable, (
        f"{case.label} at prec={prec:.0e}: {score.verdict} -- {score.detail} "
        f"[worst error {score.worst_rel_err:.2e}; basis: accuracy={suff.accuracy}, "
        f"multiplicity={suff.multiplicity}; {'; '.join(suff.reasons)}]")


@pytest.mark.parametrize("f, i", [
    pytest.param(f, i, marks=[pytest.mark.xfail(strict=True, reason="plan Phase 2")]
                 if (f, i) in UNFROZEN else [], id=f"{f}{i}") for f, i in CASES])
def test_every_case_is_scored_on_a_FROZEN_basis(f, i):
    """A pencil whose rank cutoff moves with `prec` is a different objective on every rung, so a
    cell measures the rank policy together with the search."""
    assert sufficient(_case(f, i), PREC_LADDER[0]).frozen


def test_the_cost_rises_with_the_request_and_stays_in_budget():
    """`search.md` section 2.6, not more work than asked, stated the right way round. Tighter
    requests may cost more and must not cost less, and no cell may exceed the per-eigenvalue
    budget the searcher itself declares. The earlier form asserted that cost SATURATED past
    1e-10, which was the refiner stopping early, pinned as intended."""
    per = {p: [] for p in PREC_LADDER}
    over = []
    for f, i in CASES:
        for p in PREC_LADDER:
            case, truth, _, score, evals, _ = _cell(f, i, p)
            n = max(int(truth.size), 1)
            per[p].append(evals / n)
            if evals > BUDGET_PER_EIG * n:
                over.append((case.label, p, evals, n))
    med = [float(np.median(per[p])) for p in PREC_LADDER]
    assert all(b >= 0.95 * a for a, b in zip(med, med[1:])), (
        f"median evaluations per eigenvalue fell as prec tightened: {np.round(med, 1)}")
    assert not over, f"over the per-eigenvalue budget: {over}"


def test_the_ladder_report(capsys):
    """Not an assertion -- the grid, printed with `-s`. A standing report nobody regenerated is
    how both the false floor and the snapped truth stayed invisible."""
    with capsys.disabled():
        print(f"\n\n{'case':24}" + "".join(f"{p:>9.0e}" for p in PREC_LADDER))
        print("-" * (24 + 9 * len(PREC_LADDER)))
        for f, i in CASES:
            label = _case(f, i).label
            row = "".join(f"{_CODES.get(_cell(f, i, p)[3].verdict, '?'):>9}" for p in PREC_LADDER)
            print(f"{label:24}{row}")
        print("\n" + "  ".join(f"{c}={v}" for v, c in _CODES.items()))
        print(f"{'median evals/eig':24}" + "".join(
            f"{np.median([_cell(f, i, p)[4] / max(_cell(f, i, p)[1].size, 1) for f, i in CASES]):>9.1f}"
            for p in PREC_LADDER))


def test_the_log_is_one_parseable_record_per_call(tmp_path):
    """The report the grid summarizes has to be readable by something other than a person."""
    path = tmp_path / "ladder.jsonl"
    _write_log("A", 0, 1e-8, path=str(path))
    _write_log("A", 0, 1e-8, path=str(path))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {"verdict", "worst_rel_err", "evaluations", "reasons", "levels",
            "floor_at_truth"} <= set(rows[0])
    assert rows[0]["levels"] and "delta_res" in rows[0]["levels"][0]

