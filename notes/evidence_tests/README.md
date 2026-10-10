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

## Single tests moved out of live modules (decision 6 and parked rungs)

`<source>_evidence.py` holds design-evidence tests moved verbatim from `tests/<source>.py`: ablations,
measurements of a premise, and cases below 1e-10.

| file | what moved |
|---|---|
| `test_bracket_score_evidence.py` | two baselines fail in opposite ways (V1 metric design) |
| `test_calibrate_evidence.py` | wall slope scales with length; kappa re-measured at 2.591 (TODO section 2) |
| `test_floor_evidence.py` | a sub-eps cutoff barely truncates (platform pin); inert cutoff costs spurious minima |
| `test_hard_roster_evidence.py` | the L-shape's floor spread is a basis artefact |
| `test_harness_evidence.py` | a symmetrically sampled well (surrogate quirk) |
| `test_moler_payne_evidence.py` | Moler-Payne at `tau = 2e-14`; the starved-basis bound (TODO: `delta_res` admitted at the edge) |
| `test_multiplicity_evidence.py` | C2 rejects a double count; the mediocre-basis dangerous regime |
| `test_precision_ladder_evidence.py` | the ladder report over the full six-rung ladder |
| `test_refine_evidence.py` | the machine-eps `delta_res` formula ablation |
| `test_regularization_evidence.py` | the two rank rules agree on a real scan |
| `test_rule_seam_evidence.py` | the four posterior-penalty tests (parked; `notes/posterior_penalty.md`) |
| `test_search_evidence.py` | seed underestimates slope; nearer-flank ablation; beats a uniform grid |

The two live tests that were in `test_search_collapse.py` moved to `tests/test_solve.py` (the
`phase_d` default) and `tests/test_hard_roster.py` (`Case.modes`).
