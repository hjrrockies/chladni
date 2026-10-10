"""Design evidence moved out of `tests/test_precision_ladder.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: `test_the_ladder_report` asserts nothing; it
prints the precision-ladder grid with `-s`. It is a report, not a contract, so it lives here. It
reuses the ladder's own cell cache from `tests/test_precision_ladder.py` rather than a copy, so the
grid it prints is the one the suite scores. It prints over `PREC_LADDER_FULL`, the six-rung ladder
it printed before the prune; the suite itself now stops at 1e-10. Run on demand, with `-s`:

    .venv/bin/python -m pytest -m '' -s notes/evidence_tests/test_precision_ladder_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

from bracket_score import PREC_LADDER_FULL as PREC_LADDER
from test_precision_ladder import CASES, _CODES, _case, _cell

_SLOW = pytest.mark.slow


@_SLOW
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
