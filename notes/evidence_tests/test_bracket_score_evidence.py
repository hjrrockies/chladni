"""Design evidence moved out of `tests/test_bracket_score.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: this test records that the two retired
baselines fail in opposite ways, which is why coverage and purity are two metrics and not one.
It is evidence about the scorer's design, not a contract douse needs. Moved verbatim, with the
helper it needs. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_bracket_score_evidence.py
"""
from __future__ import annotations

import numpy as np

import bracket_score as bs
from baselines import BASELINES
from chladni.mps.objective import Objective

#: A window with an isolated level, a resolvable pair and a wide-open gap. Real rectangle
#: values: 5*pi**2 is the (2,1)/(1,2) degeneracy, split here by a controlled amount.
PAIR = 5 * np.pi**2
TRUTH = np.array([2 * np.pi**2, PAIR, PAIR * (1 + 3e-7), 8 * np.pi**2, 10 * np.pi**2])
WINDOW_LAM = (15.0, 110.0)


def cone_objective(truth=TRUTH, window=WINDOW_LAM, *, variable="lam", prec=1e-8,
                   eps=1e-13, c_k=0.27, plateau=0.2):
    """The measured local model, one cone per true eigenvalue, on a plateau. Copied from
    `tests/test_bracket_score.py`, which says why the cone is built in `k`."""
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
