*Status: catalogue, collected 2026-10-06 from `douse` and `lappy` records (read-only; nothing re-run;
five cases spot-checked against stored data, noted below). The case list for the acceptance
benchmark in `douse_requirements.md`. Ranking by damage to `douse` is the user's call and is not
yet made.*

# `lappy` failures recorded in `douse`

Every recorded case where `lappy` failed `douse`, de-duplicated across the places it is written
down. Paths are relative to `/Users/hringer/projects/python/douse` unless prefixed `lappy/` or
`python/` (the parent directory, which holds `HANDOFF.md` and `LAPPY_WISHLIST.md`).

## Conventions of the source data

* Family `VertexPolygon(N, perim=4.0)`; `p` (or `x`) has length `2N - 4`. Start shapes are
  deterministic: `benchmarks.ratio_multistart.start_shape(N, i, seed0=0)`, then `param_vector`.
* **Ledger naming trap**: ledger `N<N>_k<k>` maximizes `lam_{k+1}/lam_1` (`k` is 0-based). So
  "N=6, `lam_3/lam_1`" is `N6_k2`. Runs request `n_eigs = k + 2`, `lam_max_window = max(8, 2k+4)`,
  `prec = 1e-13`, `replan_tol = 1e-9`.
* `LED` = `benchmarks/run/ratio_multistart/`. `pre_rerun` = `LED/pre_rerun_2026-08-27/` (rows from
  before `lappy` 9bc588c's missed-modes audit), `rerun` = `LED/rerun/`. "L8" is line 8 of a JSONL.
* A refusal record (`refusals[i]`) stores `p`, `plan_at`, `lams`, `deficit`; that pair is enough to
  replay a frozen-plan failure (`benchmarks/replay_refusal.py`). Crash rows store NO shape.
* `worst_sigma` and `worst_weyl_deficit` are lifetime maxima; the shape where they peaked is not
  stored.

Failure kinds: **(a)** dropped mode, **(b)** spurious value, **(c)** wrong multiplicity,
**(d)** exception, **(e)** lost precision.

Reproducibility: **R** = shape (and plan, where it matters) stored; **R-** = shape stored, plan or
nondeterministic state not; **N** = not reproducible from what is stored.

---

## Tier 1 -- reproducible, with ground truth

**T1.1 The 21.9 / 25.0 / 133.1 degree triangle: three modes silently gone.** (a), (b), (c). R.
* `p_bad = [1.02885517693507, 0.4136014913917629]`, plan built at the shape, `n_eigs = 4`.
* Returned `[93.0733, 198.4611, 256.2323, 381.8813]` at tension **2.8e-14**, no warning. FEM (P1,
  upper bounds): `93.1605, 148.0023, 198.8602, 256.8979, 306.8297, 328.5484, 383.3621, ...` --
  true indices 1, 3, 4, 7. Reported `F = 2.7530` against a true 2.1346; Weyl deficit +2.02.
* Mechanism: scan step 3.01 on `[45.9, 347.0]`, wells 0.16-0.32 wide; 13 minima rejected at tension
  2.2-2.8e-3 against `ttol = 1e-3`; the rescue loop filled the slots by count.
* Same shape at `n_eigs = 6`: `UnresolvedSpectrum`, a spurious value at tension 5.07e-4.
* On the bisection toward the clean partner, `p = [0.9402286697638149, 0.42818182971442903]`
  returned `[88.0218, 88.0219, 141.1603, 190.2552]` -- `lam_1` twice, true `lam_4` dropped -- with
  Weyl deficit -0.49 and FEM deficit 0: **both count statistics fooled**.
* Clean partner `p_good = [0.9291503563674081, 0.4300043720047623]`.
* Data: `benchmarks/run/missing_mode/cache.npz` (`p_bad`, `lams_bad`, `fem_bad` to 44 values,
  `p_good`, scan arrays, `meta['bracket']`). Also `LED/N3_k2.jsonl` L8 `refusals[0]`,
  `jupyter_notebooks/missing_mode.ipynb`, `benchmarks/sigma_base_ablation.py`,
  `rtol_single_point.py`. *Spot-checked 2026-10-06.*

**T1.2 `pre_plateau_fix/N5_k2` start 0: the plan decides whether a double loses a copy or gains
one.** (a), (c). R.
* Shape: `LED/pre_plateau_fix/N5_k2.jsonl` start 0.
* Recorded `[21.575233, 39.487109, 68.853674, 88.458931]`, tension 1.6e-10, deficit +0.683.
  Certified: `[21.575211, 39.487156, 68.853605, 68.853605]` -- `lam_3 = lam_4`, gap 8.9e-10; FEM
  deficit +1 at 1984, 12466 and 49842 dofs.
* A plan frozen at the regular pentagon returns `[21.5752, 21.5752, 39.4871, 68.8537]`: an
  **invented double `lam_1`**, `F = 1.8302` against 3.1913, passing every guard (tension 2.9e-8,
  Weyl -0.79, FEM deficit 0). `lappy`'s `cold_start` Leg 3 could not reproduce it inside `lappy`.
* `python/LAPPY_WISHLIST.md:307-364`, `docs/todo.md:704-802`,
  `lappy/docs/WORK_ORDER_solver_reliability.md:184-225`, `benchmarks/completeness_detectors.py`.

**T1.3 N=3 start 4, the 11.2 / 16.3 / 152.6 degree spike: result depends on `n_eigs`, and the
solver later crashes.** (a), (d). R.
* `p = start_shape(3, 4) = [1.1756103887576692, 0.23218091730644705]`, fresh plan.
* `n_eigs = 4`: drops 430.38. `= 5`: drops `lam_1` 248.65. `= 6`: drops 248.65 and 344.59.
  `= 7`: drops 248.65 and ~887.5 (`pre_rerun/N3_k{1..5}.jsonl`). Later code raises
  `EigensolverFailure` at k=3, k=5.
* Reference (consensus of clean solves): `248.653419474, 344.590171803, 430.380005708,
  518.615801753, 611.337416106, 705.749663843, 807.007609817, 887.472441245, 1020.39768`
  (`benchmarks/diagnostics/repro_N3_k2_s4.jsonl`).

**T1.4 `reg_ngon_6` / `reg_ngon_8` reference tables.** (c), in the reference pipeline. R.
* The `reg_ngon_6` table listed the double 32.45185751 once, shifting every later index;
  `reg_ngon_8` was missing `lam_1 = 6.4849335` (caught by Faber-Krahn). Cause:
  `symsolve.solve_sym` under-counts per sector reducing by a proper subgroup -- **still open in
  `lappy`**.
* Corrected `reg_ngon_6` (full domain, 12.1 digits): `7.155339133926042, 18.131677865530694 x2,
  32.45185751440048 x2, 37.49135287679824, 47.62936577385704, 52.63789013914339,
  60.10511209416662 x2`. Note `geo.reg_ngon(n)` is unit CIRCUMRADIUS, not perimeter 4.
* `lappy/benchmarks/suite/run/reference_values.py:78-200`, `lappy/tests/test_reference_tables.py`.
* The other claim in `douse/derivative.py:157-163` -- "`lappy` reported a double at a simple
  eigenvalue on `reg_ngon_6`" -- has **no numbers anywhere**; `lappy`'s full-domain multiplicities
  there measured correct. Treat as unsubstantiated.

**T1.5 Rectangle duplicate roots on the cold path** (`lappy`, fixed). (c). R.
* Perimeter 4, `L/H = 1.633`, k=12: `lam_5 = 94.0813` returned twice; `L/H = 2.0`, k=22: 72.171482
  twice. Triggered by a false Weyl alarm then a loose root. A 106-cell probe found 8 duplicate and
  14 inaccurate cells.
* `lappy/docs/WORK_ORDER_solver_reliability.md:42-100`, `lappy/benchmarks/cold_start/run/`,
  `lappy/tests/test_cold_start_contract.py`. Ground truth closed form.

**T1.6 Scan-resolution drops at low `ppl`** (`lappy`, fixed the default). (a). R.
* `right_trapezoid` dropped 44.9484877814 at k=10 (ppl 5); `H_shape` dropped 14.30522996 at k=7, 9
  (ppl 10); 9/90, 2/90, 0/90 cells wrong at ppl 5, 10, 20.
* `lappy/tests/test_mode_completeness.py`, reference `lappy/benchmarks/suite/run/reference_values.py`.

**T1.7 The Weyl guard misfires on complete spectra.** False positive, not a solver failure, but it
triggered T1.5. R. `sqrt(8/3)` rectangle at k=8 scores +0.577, a 5:1 rectangle +1.324; past k=8
the clean and missed classes overlap (`benchmarks/deficit_calibration.py`, `docs/todo.md:837-861`).
Relevant because `douse_requirements.md` keeps the Weyl count as an independent audit.

## Tier 2 -- reproducible, no ground truth stored

**T2.1 N6_k1 start 2: the 2.6203 row (PPW violated) and its spurious-value rerun.** (a), (b). R-.
* `x = [0.796102460003105, 1.0430649234606624, 0.3045509941484931, -0.2964947023033227,
  0.114354480800215, 0.862628738987791, 1.2240136818145162, 0.6424560268310093]`
  (`pre_rerun/N6_k1_s2_second_pass.json[0]`; also `tests/test_spectral_guard.py`).
* Returned `[17.0828, 44.7624, 74.0909]` at tension 1.3e-9; true `lam_2 = 41.49929` absent. Ran 92
  iterations on `lam_3/lam_1` reporting 2.6203 > 2.5387. Fresh ratio 2.4293. `plan_at` not stored.
* The rerun shape (`rerun/N6_k1_start2.jsonl`, field `x`) returned `[17.0590, 43.1361, 58.1948]`:
  58.1948 at tension **0.62**, not an eigenvalue, under `worst_sigma = 1.3e-9`; Weyl 0.470 (one
  dropped plus one spurious balances the count). The origin of `spurious_tol`.
  *Spot-checked 2026-10-06.*

**T2.2 N6_k2 start 1 (current): a bit-identical triple on an asymmetric hexagon, `verified=True`.**
(c), suspected. R.
* `x = plan_at = [1.5524069062968155, 1.421770613638599, 0.5304091124623859, 0.5053978611322578,
  0.16141219015262093, 0.727170479679781, 0.5527920320397105, 0.13976052872357242]`
  (`LED/N6_k2.jsonl` L8). Angles 15.5 / 91.5 / 91.9 / 104.5 / 165.5 / 251.1.
* Returned `[31.6183, 34.467345990537844 x3]`; `F = 1.090` against the cell's 3.198; tension
  3.3e-7, deficit 0.23 -- passed every guard. *Spot-checked 2026-10-06.*

**T2.3 N5_k4 start 5 (current): a double ground state on an irregular pentagon.** (c), suspected. R.
* `x` and `plan_at` in `LED/N5_k4.jsonl` L8. Returned `lam_1 = lam_2 = 20.33765341129639 /
  20.337653411564297` (relative gap 1.3e-11) and `lam_3 = lam_4 = 48.6884`; `F = 3.9255` against the
  cell's 5.0433; `verified=True`, excluded only on `worst_sigma = 4.9e-5`. A simple `lam_1` is a
  theorem for connected domains, so this is certainly wrong. *Spot-checked 2026-10-06.*
* Same start, `pre_rerun/N5_k4.jsonl` L4: `refusals[0]` drops the ~51.35 mode (deficit 1.31);
  `refusals[3]` also collapses the 100.5 / 101.0 pair (deficit 3.05). Frozen-plan pairs stored.

**T2.4 `ratio_stall` N=5: duplicated and vanishing near-double partners accepted as iterates.** (c). R.
* `benchmarks/run/ratio_stall/N5.jsonl` lines 4-5: `[17.4278, 45.5625, 45.5625]` (ratio 2.614) and
  `[17.5887, 47.9100, 47.9100]` (2.724) -- both above PPW: `lam_2` duplicated, true `lam_2` gone.
  *Spot-checked 2026-10-06.*
* Lines 18-28: from iterate 17 the 44.4468 / 44.4470 pair loses a member; the run stalled at
  2.51530 against the regular pentagon's 2.52684.
* `ratio_stall/N4.jsonl` line 25: `[19.7735, 49.42243851, 49.42243861]`, gap 2.1e-9, on a
  non-square quadrilateral; the run then stopped on 40 failed backtracks.

**T2.5 N6 start 5, the reflex hexagon (203 degrees): fails at every k, in every mode.** (a)-(e). R-.
* `start_shape(6, 5) = [1.6312897734785867, 0.9117377758429616, 0.6231789069106254,
  0.2840791568565929, 0.2333524495837802, 0.904778806970504, 0.6638056912940102,
  0.17752397373425044]`.
* k=3: `UnresolvedSpectrum` at the start (deficit 5.21), later a spurious value at 5.02e-4 and a
  redraw. k=2 (`pre_rerun/N6_k2.jsonl` L6): one copy of the near-double 77.41 dropped, `F`
  inflated 37% (3.637 vs a cell best of 3.198). k=1 (`pre_rerun/N6_k1.jsonl` L8): `lam_1 ~ 39`
  dropped entirely. Endpoints stored, pre-rerun plans not.

**T2.6 Frozen-plan misses with stored `(p, plan_at)` pairs.** (a), (c). R.
* `pre_rerun/N6_k3.jsonl` L4 `refusals[0]` (start 7): a convex sliver (171.8 degrees, edge ratio
  0.27); full solve on the frozen plan drops `lam_5 = 94.7242`, deficit 1.98 -- and a FRESH plan
  agrees bit-for-bit, so it is not only staleness. `lappy`'s missed-modes audit recovers it.
* `rerun/N5_k5_contested_run3.jsonl` start 1 `refusals[0]`: the frozen plan returns one copy of a
  double `lam_1` (`[20.2672, 42.6339, ...]`); a plan at the shape gives `[20.2672, 20.2672, ...]`.
* `pre_rerun/N5_k4.jsonl` L7 (start 3, near-degenerate pentagon): ~50.04 dropped, deficit 1.32.
* `pre_rerun/N3_k2.jsonl` L8 (start 6) and `pre_rerun/N3_k5.jsonl` L6 (start 5): one copy of a
  near-double lost (125.166 / 125.170; 177.458172 / 177.458199).
* `LED/N3_k4.jsonl` L8 `refusals[0]`: thin triangle `p = [1.0352995627987067, 0.3312315145786663]`,
  deficit +4.64, about four modes missing.

**T2.7 The N5_k5 cell (`lam_6/lam_1`): contested.** ~40 refused shapes with `p` and `plan_at`
(`LED/N5_k5.jsonl`, `rerun/N5_k5_contested*.jsonl` -- the two rerun files are identical), deficits
0.485-0.546 straddling the 0.5 threshold, a near-double at ~122.5. Either a Weyl false positive at
k >= 6 or a lost copy; reruns give 6.0617 and 6.0698 against a certified 6.0656540842. Unsettled.

**T2.8 Exceptions at known start shapes.** (d). R- (the shape is known, the failure is not
deterministic).
* `EigensolverFailure: f has too many local minima` (`lappy/opt.py:212`): `start_shape(3, 4)`
  (T1.3), `start_shape(5, 2)`, `start_shape(6, 1)` (N6_k1 start 1 crashed twice). 15 crash rows
  de-duplicate to these plus T2.5 and one unrecorded shape.
* `IndexError` in `basis_plan.realize` when the corner count changes: a plan frozen on a pentagon
  realized at its flat-vertex limit (`benchmarks/kink_sweep.py`, `pentagon(h)`). R.

**T2.9 Precision cases.** (e). R.
* Tension floor limits finite differences: `SEED_SHAPE = [0, 1.3, 1.1+0.9j, 0.15+0.8j]`; target 1e-12
  gives a floor 1.15e-10 and FD error 3.7e-4, target 1e-14 a floor 1.02e-11 (`tests/test_derivative.py`).
* A pair below resolution at the N=4 `lam_4/lam_1` optimum: `[23.6302578136, 55.3495724946,
  57.3573811503, 101.2178122523, 101.2178122638]`, split 1.1e-10 relative; cost 8 of the first 10
  runs (`tests/test_cluster_opt.py:418-454`).
* Tracked `F` disagrees with a fresh solve: N5_k3 start 5, 4.441676010 vs 4.375460168, trust ratios
  -9e7 to -7.5e11 (`tests/test_discontinuity.py`, `LED/detector/`).
* Default `prec` gives ~8.2 digits on regular N-gons N = 5, 8, 12, 16, against 10.0-11.7 at 1e-13
  (`benchmarks/run/ngon_basis/probe.jsonl`).
* `N5_k3_convex.jsonl` L1: a near-triangle (173.9 and 179.6 degree vertices) "certified" at
  `F = 2.3455` after one iteration (cell 4.4824) on a bit-identical double. Low confidence.

## Tier 3 -- recorded, not reproducible

Kept because they are failure CLASSES the benchmark must still cover by construction.

* `SourcesInsideDomain` ("1 of 75 sources landed inside the domain"), N6_k2 start 3 mid-run
  (`LED/N6_k2.jsonl` L4); no shape. The same start's pre-rerun row (`pre_rerun/N6_k2.jsonl` L5, a
  notched ~284 degree shape) lost `lam_2 ~ 98.5` at tension 4.7e-12 and collapsed its trust radius.
* `LinAlgError: SVD did not converge` in `regularize_pencil`, one shape in ~1500 solves, killed 6
  of 10 starts (fixed by a gesvd fallback; `python/LAPPY_WISHLIST.md:135-188`).
* `track` accepting a point 2.9e-3 from the eigenvalue inside `ttol = 1e-3`; the optimizer went
  below 2 pi^2 (`douse/evaluate.py:29-36`). `ttol = 1e-8` returning zero eigenvalues on a triangle.
* Frozen-plan drift: N=5 `lam_3/lam_1` at tension 1.6e-4 vs 5.4e-11 fresh -- the retracted 3.0717
  (`benchmarks/ratio_k3.py:22-30`); 7 of 13 reentrant starts excluded on `worst_sigma`, against 2
  of 66 others (`douse/sampling.py:5-10`); ten current rows with lifetime `worst_sigma` 5e-6 to
  7e-4.
* Per-value track collapses (two seeds on one mode): 29 at N=4 and 135 at N=5 before `track_set`.
* 14 mid-run spurious refusals and ~360 Weyl refusals counted in ledgers without shapes.
* Unsorted tracking through a crossing (2.6936 > PPW) -- a `douse` bug, retired by sorting.

## Aggregates

Ledger rows: 191 current, 105 pre-rerun, 5 rerun, 11 pre-plateau-fix. Crash rows: 15, five
distinct shapes. Stored refusal records: 71, about 45 unique shapes (about 40 of them N5_k5).
`removed.json` (13 cells, pre-rerun): 28 removals -- 9 crashes, 9 drifted, 10 missed modes.

## What this says for the benchmark

* **Silent failures are the core.** T1.1, T1.2, T2.1, T2.2, T2.3 and T2.4 passed tension, and
  several passed the Weyl and FEM counts too. A benchmark that scores only tension or count would
  pass them. Each case needs a REFERENCE SPECTRUM with multiplicities.
* **Most shapes have no ground truth.** Tier 2 needs references generated independently: a
  high-`prec` `chladni` solve on a fresh, generous basis, cross-checked against FEM upper bounds
  and, for multiplicities, a second basis -- per `CLAUDE.md`, `chladni` may generate references
  once trusted, but these cases are exactly where trust has to be earned, so FEM agreement is the
  bar.
* **Frozen-plan cases need `(p, plan_at)` pairs**, so the benchmark exercises `plan.realize`.
  T2.6 and T2.3 supply them.
* **Many cases cluster on three geometries**: obtuse/spike triangles, reflex hexagons, and
  near-degenerate pentagons near the regular one. Short edges and near-collinear vertices appear in
  T2.6 and T2.8; thin shapes in T2.6.

## Not established

* No case was re-run; five were spot-checked against stored data (T1.1, T2.1's first shape, T2.2,
  T2.3, T2.4's lines 4-5). The rest are as the source files record them.
* T2.2 and T2.3 are classified from the returned spectra alone; T2.3 is wrong by theorem (a simple
  `lam_1`), T2.2 only by implausibility.
* Which of these hurt `douse` most is the user's ranking, not yet made.

## Reproduced, with references (2026-10-07)

`benchmarks/douse/` now holds every Tier 1/2 shape above as exact vertices (`cases.json`, with the
discrepancies against this catalogue in each case's `notes`), reference spectra for all of them
(`references.json`), and lappy's cold solve on each (`lappy_runs.json`). See
`benchmarks/douse/scoreboard.md`. What changes here:

* **Reproduced from the stored shapes**: T1.1 (all three shapes, to the catalogue's 4 decimals),
  T1.2 on the frozen plan (the invented double `lam_1`), T2.3 (doubled `lam_1` on the frozen plan;
  fresh correct at the endpoint, wrong at both pre-rerun refusals), T2.5 (fresh and frozen return
  `[77.3959 x2, 199.76, ...]` where the truth is `29.2430, 60.9012, 77.4130, ...`, all simple:
  `lam_1` and `lam_2` dropped and a value near `lam_3` doubled -- the catalogue's "`lam_1 ~ 39`"
  was itself a lappy number), T2.8-kink (`IndexError`), and doubled or dropped levels across T2.6
  and T2.7.
* **T1.3's reference list is missing `lam_9 = 916.0816`** (three chladni bases; FEM 917.39 at
  refine 6). The list's 1020.40 is `lam_10`.
* **T2.2's recorded lappy output does not belong to the stored shape**: on it `lam_1 = 37.1831`
  (three bases and FEM), and a fresh lappy solve gives `[34.4924 x3, 41.3688]` -- a spurious triple
  below `lam_1`, so the catalogue's verdict (spurious triple) stands, the numbers differ.
* **T2.3 now has ground truth**: `20.3376533, 48.6883582, 52.9825808, 79.8345375, ...`, all simple.
* **T2.6 `N5k5_contested_run3`**: `lam_1` is simple by theorem, so the at-shape output
  `[20.2672, 20.2672, ...]` is the wrong one, not the frozen plan's. (The reference has one level
  at 20.26722 but marks it provisional: the three runs did not agree on its count flags.)
