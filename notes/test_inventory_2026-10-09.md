# Test-suite inventory for the prune (2026-10-09)

*For the user's review: nothing has been deleted. Sources:*
- *three read-only classification passes over every test file (rubric below);*
- *a full local run with per-test durations, `-m ''`, one BLAS thread, 22.4 min (inflated by a
  concurrent sweep; the proportions hold);*
- *an 8-seed jitter sweep of the per-push tier (`notes/jitter_sweep_slow_tier.json`, 2237 tests,
  control seed 0 clean).*

**Rubric.** Each test is classified as one of:
- **C**: a contract douse needs;
- **M**: pins a mechanism on a chosen edge case;
- **E**: evidence an open TODO item cites;
- **R**: a retired or parked feature, or a rung below 1e-10 only;
- **I**: tests the test apparatus.

**Fair** means the test accepts any honest outcome (right, declined or flagged), rather than
demanding one answer inside a band the solver declares undecidable.

## The decisions, in order of leverage

### 1. The scorers have no "wrong but flagged" outcome (systemic: fixes many tests at once)

- **`tests/prec_score.py`:** `ACCEPTABLE = {pass, declined, b-excused, merge-excused,
  wrong-total-excused}`, and nothing in the file reads `count_uncertain`, `n_uncertain` or
  `certified_counts`.
- **`test_cluster_totals._assert_total`:** passes only when the total is right or the cell declined.

So an honestly flagged miscount, which the 1e-10 direction accepts, fails:
- the precision ladder;
- both cluster grids, on both pencils;
- the refine controls;
- several rule-seam tests;
- `test_solve::…across_the_crossover` at bands 0.8/1.3, where the licence ratio is about 1.

**Proposal:** add a `flagged` verdict (count flagged, or `certified_counts` False) to `prec_score`
and accept it in `_assert_total`. Keep reporting it separately, so the rate stays visible.

### 2. Whole files to retire

| file | runtime | why | evidence to move first |
|---|---|---|---|
| `test_gate.py` (+ the `gate` marker, `nightly.yml` leg) | 357 s, 27% of the suite | stage 1 only, ladder to 1e-14, treats a decline as a failure; superseded by the precision ladder at douse's rungs | TODO §2 names it as part of the "prec ladder"; keep `rect_suite` |
| `test_searches.py` (+ `tests/searches.py` if unused) | 32 s | the retired lappy/Lipschitz/swing baselines | TODO §5 item; `FIXED_STEP_probe_error_over_h` to notes |
| `test_multiplicity_ladder.py` (+ `mult_suite.py`) | 25 s | scores the retired `rule="gap"` and `reach_slope="outside"` | none cited by name |
| `test_search_collapse.py` | small | Phase D (`phase_d="angle"`), parked | move `default_is_DISTANCE` and `ROSTER_carries_eigenfunctions` out; copy the §36 numbers and the degenerate-eigenspace finding (TODO §10) to notes |
| `test_sector_floors.py` | small | all rungs below 1e-10; its `sector_suite` was never built | none |

Optionally `test_refine_controls.py`, if `prec_score` itself goes; it should not go after
decision 1.

### 3. Rungs below 1e-10 (parked) inside otherwise-good tests

**Drop the 1e-12 and 1e-14 params, and add 1e-10 where it is missing**, in:
- `test_cluster_totals` (SLICE, GRID, both pencils; this also retires `MULTIPLICITY_precondition…`);
- `test_precision_ladder` (cells, STRESS, cost);
- `test_hard_roster` (`finest_prec`, coverage, L-shape audit);
- `test_bracket_score` (`every_band…`);
- `test_rect_suite` (`dial`, `quantization`, `rect_spectrum_would_MERGE`, `assert_smooth`);
- `test_refine` (`guard4_exactly_degenerate_pair`, `reaches_floor_real_rectangle`);
- `test_refine_regression` (`real_pencil_asymmetric`, the 1e-12 `NARROWER` params);
- `test_refine_controls` (1e-14);
- `test_rule_seam` (`INNER_END_refusal`, `stage1_bracket_refused_at_END`, `side_search_finds_NOTHING`,
  the 1e-12 cells of `READ_spends_nothing` and `promoted_defaults`);
- `test_regularization` (`penalty_whose_floor_would_blur`, the 1e-12 param of `…inside_its_reach`,
  and optionally the 1e-12/1e-14 rows of `auto_is_the_penalty`);
- `test_moler_payne` (`[threshold]`, and `[tau 2e-14]` to notes; the starved-basis test to notes as
  evidence for the `delta_res` margin item);
- `test_floor` (`suite_rtol_chosen_by_FLOOR`, `suite_rtol_is_derived_rule_at_finest_prec`);
- `test_floor_multiplicity` (re-pose at 1e-10).

In the sweep, the 1e-12 rule-seam cells flip in 2–3 of 8 seeds.

### 4. Retired features still tested

- **The literal `reg="tikhonov"` mode:** all its parametrizations,
  `two_penalty_realizations_agree`, and the literal half of
  `qr_penalty_never_changes_rank…`.
- **Phase D:** the four `interior_image` tests in `test_mps_seam` (only `phase_d="angle"` consumes them).
- **`rule="gap"` / (C1):**
  - `test_multiplicity` (`C1_is_a_RATIO_GAP`, `single_branch_reads_one`,
    `reads_correctly_off_smooth_spectrum`, `exact_degeneracy_reads`);
  - `test_cluster_totals::no_wide_bracket_reach…`.
- **`rule="floor"`:** `test_rule_seam::floor_counts_COINCIDENCE`.
- **The posterior penalty (parked):** the four posterior tests in `test_rule_seam`, to notes.
  Each pins an exact count at 1e-12.

### 5. Rewrite as contracts (unfair as written)

| test | problem | contract |
|---|---|---|
| `test_cluster_totals` SLICE/GRID | a flagged total fails | right, declined or flagged (decision 1) |
| `test_solve::pipeline_across_the_crossover` | bands 0.8/1.3, licence ratio ~1 | right or flagged |
| `test_rule_seam::partner_at_edge_LOCATED`, `hump_side_guards_keep_CORRECT` | demand certified-right at the bracket edge | right or flagged |
| `test_rule_seam::fresh_125_grid` | `total == truth` on top of the no-silent check | keep only "no silent wrong" |
| `test_solve_interval::exact_degeneracy_ONE_level_on_wall` | raising on a flag fails it | accept a flag or a raise |
| `test_solve_interval::basis_cannot_serve_REPORTED`, `starved_basis_not_DROP` | at 1e-12; Linux admitted `delta_res` by 1% | re-pose at 1e-10; narrow `except Exception` |
| `test_floor_multiplicity::exact_double…` | `sum == 2` with no flag accepted; floor 8.8e-13 is near `prec` | at 1e-10: count 2, or flagged |
| `test_regularization::ci_refusal_does_not_happen_under_the_penalty` | pins one refine trajectory: **8/8 seeds** | keep the penalty half; drop the injected-flip reason-string pin |
| `test_regularization::refused_bracket_end_is_smooth…` | **5/8 seeds** | assert smoothness under the penalty only |
| `test_regularization::blas_sized_perturbation_flips…` | `flips >= 1` at 0.98x cutoff | skip when no flip, as its sibling does |
| `test_regularization::rank_at_tracks_lam` | `== [44, 48, 50]` is LAPACK's own tolerance | grows, and stays within `n_cols` |
| `test_regularization::geqrt_route…[shape0]` | 548x274 IS the heap-corruption reproducer | use another shape (TODO §4) |
| `test_regularization::default_tau_proportional` | pins `C_PEN == 0.02` | drop the constant pin |
| `test_gate` (if kept, slimmed) | `declined == 0`, 1e-14 | quote the rate; rungs of 1e-10 and above |
| `test_refine::delta_res_CONTAINS_true_error` | slack < 10 and ≥3-level pins | containment only |
| `test_multiplicity::reach_must_clear…`, `spends_NO_EVALUATIONS` | the final read is the gap rule | re-pose on the reach rule |
| `test_rule_seam::reach_COUNTS_pair…`, `reach_resolves_SIDE` | run on the retired outside-slope bounds (0.60 vs 0.61 under the shipped ones) | pass the shipped bounds |
| `test_rule_seam::defaults_are_shipped_behaviour` | asserts `rule="gap"` and `collapse` still exist | update with the retirements |
| `test_prec_score::acceptable_verdicts…`, `collapsed_level…` | no flagged verdict | decision 1 |

### 6. Move to `notes/` scripts (design evidence, not contracts)

- **`test_floor`:** `BARELY_TRUNCATES` (a platform pin: 81 vs 71 of 92) and `inert_cutoff…`.
- **`test_calibrate`:** `wall_slope_scales_with_LENGTH`, and `kappa_re_measured` (keep the number
  in TODO).
- **`test_hard_roster`:** `FLOOR_SPREAD_IS_A_BASIS_ARTIFACT`. It contradicts TODO §3's "eps varies
  five orders" line; settle which is stale.
- **`test_multiplicity`:** `C2_REJECTS_double_count`, `dangerous_regime_MEDIOCRE_BASIS`.
- **`test_refine`:** `ABLATION_machine_eps_formula`.
- **`test_search`:** `seed_underestimates_slope`, `ABLATION_nearer_flank`, `beats_uniform_grid`.
- **`test_bracket_score`:** `two_baselines_fail_in_opposite_ways`.
- **`test_harness`:** `symmetrically_sampled_well`.
- **`test_regularization`:** `the_two_rules_agree_on_a_real_scan`.
- **`test_precision_ladder`:** `ladder_report` (print only).

### 7. Vacuous or broken (fix or delete regardless)

- **`test_rule_seam::owned_ONCE`** does not test what it says: 0.45 < 1/2.21, so only one claimant
  admits the branch.
- **`test_calibrate::unfindable_warns…`** has an assertion ending in `or True`.
- **`test_multiplicity_ladder::capability_reported`** is `assert True`.
- **`test_cluster_totals::every_wrong_total_in_slice/grid_is_LOUD`** collect zero items (`WRONG`
  is empty).
- **`test_search::gate_uses_c_UPPER_bound`**: `if small and big` can assert nothing.
- **`test_search`:** `finer_ladder_costs_more` and `relative_scale_FLOOR` are tautologies.
- **`test_refine::delta_res_measured_and_reported`** is covered by the next test.
- **`test_refine.py:345`** has a duplicated `@pytest.mark.slow`.

### 8. Harness artifacts in the jitter sweep (not flips; exempt from `ulp_sensitive`)

`test_mps_seam::assemble_is_where_lam_becomes_k` and `…free_functions_and_methods_agree` fail 8/8
because the harness patches `assemble` itself. `test_floor::…BARELY_TRUNCATES` counts
rounding-level directions, which jitter changes by construction (see 6).

## Two source issues the inventory turned up

- **`multiplicity(rule="reach")` without `reach_bounds` falls back to the RETIRED outside-slope
  bounds** (bias 0.89–1.63). `solve` ships the cache bounds, 0.61–2.21. `multiplicity()`'s own
  default is still `rule="gap"`.
- **At douse's `prec = 1e-10`, `reg="auto"` resolves to truncation, not the penalty.** The penalty
  tests guard only `prec >= 1e-8`. The rank-flip exposure is at douse's own precision, which is
  Phase 2's question.

## Everything else: keep

These need no change:
- **The foundation layers** (geometry, Helmholtz, reference, integrals, plotting, examples): all
  contracts, deterministic, cheap.
- **`test_objective`, `test_precision`, `test_count_certification`, `test_stress_domains`,
  `test_eigensolver`, `test_eigenfunctions`, `test_orthonormal`, `test_rellich`,
  `test_integral_rules`, `test_solver_seam`, `test_surrogate`, `test_surrogate_cases`.**
- **Most of `test_regularization`, `test_refine`, `test_refine_regression` and `test_search`.**
  The phantom calibration cases there are TODO §10 evidence: keep them.

Jitter flips at 1/8 inside kept files (`test_collocation` 3/8, `test_refine` OFF_CENTRE and
empty-bracket, a few `test_solve_interval` and `test_solve` cells) need a second look after
decision 1: some are flagged-honest outcomes counted as failures.

## Estimated effect

The runtime estimate, from proportions rather than a measured pruned run:
- Retiring `test_gate` and dropping the sub-1e-10 rungs removes roughly half of the 22-minute
  local suite.
- Per-push CI (already without gate and robust) shrinks less, mostly via the precision ladder,
  the cluster SLICE and `floor_multiplicity`.
- The `ulp_sensitive` set would hold only contracts at 1e-10 and above.
