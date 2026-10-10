# Retired test modules, kept as runnable evidence

Moved out of `tests/` on 2026-10-09 by the test-suite prune (`notes/test_inventory_2026-10-09.md`,
decision 2). Each one either tests a retired or parked feature or pins design-era trajectories, but
records measurements that TODO.md or the notes still cite. `pytest` does not collect this folder
(`testpaths = ["tests"]`); run a module on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_gate.py

| module | why it left the suite | what it still records |
|---|---|---|
| `test_gate.py` | stage 1 only, ladder to 1e-14, a decline counted as failure; 27% of suite time | the standing V1 acceptance gate, `rect_suite`'s wide windows |
| `test_searches.py` | the retired lappy / Lipschitz / swing baselines | why the fixed-step probe drops levels (error over h) |
| `test_multiplicity_ladder.py` | scores the retired `rule="gap"` and `reach_slope="outside"` | the reach rule's calibration populations |
| `test_search_collapse.py` | Phase D (`phase_d="angle"`), parked | TAU_GRAM's populations, the degenerate-eigenspace finding (TODO section 10) |
| `test_sector_floors.py` | rungs below 1e-10 only; its `sector_suite` was never built | more Fourier-Bessel columns DEGRADE a sector's floor |

The two live tests that were in `test_search_collapse.py` moved to `tests/test_solve.py` (the
`phase_d` default) and `tests/test_hard_roster.py` (`Case.modes`).
