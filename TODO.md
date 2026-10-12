# Outstanding work

Three rules this file lives by:

* **An item is deleted when it is done, not annotated.** The file is kept true at HEAD, the
  same way `docs/conventions.md` is. What was outstanding on a past date is what `git log` is
  for.
* **Every item names the evidence that forces it** -- a measurement, a design document, or a
  line of code. An item with no source is a preference, and belongs in a design record where
  it can be argued with.
* **Citations into `notes/` are deliberate.** The sharpest open items were measured there and
  exist nowhere else, so an item that cites one is pointing at its evidence rather than at a
  file someone happens to have. `notes/` is tracked for that reason -- it was not, and an item
  whose justification lived only in a working directory was an item nobody else could argue
  with. It stays working notes: dated, superseded, addressed to a moment, and excluded from a
  release rather than from the repository.

Priorities are `[blocking]` (something else waits on it), `[next]`, `[later]`.

---

## 0. What `chladni` is for -- read first

`chladni` exists to fix the `lappy` failures that hurt `douse`: dropped eigenvalues and lost
precision (`notes/session_2026-10-05.md`). **Direction since 2026-10-09: work OFTEN at a first
working precision of `prec = 1e-10`, and report shortfalls honestly and early, rather than fight
the floor** (`docs/design/douse_requirements.md`, "Precision per level"). Nearly every defect of
2026-10-06..09 lived where the request sits within a few floor scales of the floor, `R = prec*lam /
(eps/c)` small; the remedy is to stay out of that regime and say so. Work below 1e-10 is in the
"Parked" section at the end.

The path, in order:

* **[blocking]** **Prune the test suite** (agreed 2026-10-09). Much of it pins trajectories,
  constants and ablations from when the search was designed on one rectangle family; it costs
  runtime (the full suite is ~17 min) and flips under `tests/ulp_jitter.py`. First an inventory
  classifying every test as a douse contract, a mechanism pin, evidence for an open item, or a
  retired/parked feature, with runtime and jitter flip rate -- and FAIRNESS: how far its case sits
  from the nearest decision boundary the solver declares (the margin ratio `R`, the merge licence,
  the reach rule's bias interval, `basis_sufficiency`'s precondition). A test that demands one
  answer inside a declared-undecidable band is unfair; it should accept any honest outcome
  (right, declined, flagged) or move off the boundary. The user reviews it, then the
  pruning goes in a few commits that say where any evidence went.
* **[blocking]** **Make the `douse` benchmark a reliable instrument at 1e-10.** It exists
  (`benchmarks/douse/scoreboard.md`: 41 catalogued shapes, references by three chladni bases plus
  FEM counts, `lappy`'s cold solves, a scorer). It is not yet reproducible: the window comes from
  ARPACK FEM eigenvalues with a random start, and counts at near-degeneracies change with ulps of
  the window (`notes/nondeterminism.md`). Needs: a seeded ARPACK start in `fem.py`; a STABILITY
  score (each case on a few jittered windows, how often the count changes); rows at `prec = 1e-10`
  with wall time.
* **DECIDED 2026-10-10: keep truncation at 1e-10** (`notes/regularization_1e10.md`). On the douse
  benchmark (basis D, 5 windows): truncation 161 ok / 1 flagged / 13 silent, stable 36/41;
  `tikhonov_qr` 152 / 10 / 13, 38/41, 1.8x wall, with NEW silent losses on T1.4-oct and ngon8;
  posterior 161 / 3 / 11, 35/41, 2.9x wall, new silent windows on T1.4-hex and ngon8. Cluster
  grids under jitter: truncation 115/115. Neither penalty clears the pre-written rule. Open from
  it: (a) **at 1e-8, where `reg="auto"` ships the penalty, truncation beats it on the benchmark on
  every axis** (169 ok / 1 silent / 39 of 41 stable vs 155 / 9 / 37, 0.6x wall; cluster grids tie at
  115/115), and at 1e-6 too (0 silent vs 5). REVERSED 2026-10-11 on `phase2/auto-truncation`
  (`PENALTY_PREC_MIN = inf`: `"auto"` is the threshold at every `prec`). The Linux rank-flip
  refusal the penalty once removed was on the suite pencil (`SUITE_RTOL`); at the rung's own
  `rtol` the cluster grids are 391/391 at 1e-6 and 1e-8 under 16 jitter seeds. **Linux CI is the
  open check;** (b) the penalty turns truncation's silent
  T1.2/ngon12 rows at 1e-10 into flags; check whether the margin rule does the same for
  truncation; (c) E3: the first dropped singular value sits within 10% of the cutoff at 15% of
  evaluated points, with no outcome yet attributed to it.
* **Phase 3 BUILT 2026-10-11** (branch `phase3/margin-rule`): `Level.margin`, `Level.resolution`,
  `Level.count_shortfall`, and `SolveDiagnostics.n_count_shortfall`, reporting only
  (`tests/test_level_resolution.py`). On the benchmark, rescored (`notes/phase3_rescore.md`): no
  count is silently wrong on bases D or A at 1e-10 or 1e-8. What is left is DROP rows (Phase 4)
  and one value bound (the `delta_res` item below). The scoreboard itself still matches at 1e-6;
  move it to resolution groups when it is next regenerated.
* **[next]** **Phase 4, planned 2026-10-11: the count audit, then `on_shortfall`.**
  Calibration on the 31 complete douse reference spectra: the MEAN Weyl deficit over mid-gap cuts
  (corner-corrected `weyl_count_poly`) sits in [-0.07, 0.04] on complete spectra. A simulated
  whole-level drop scores >= 0.16 when it lies in the lower two-thirds of the spectrum (all
  caught) and is caught 88% of the time in the top third. Today's worst-cut statistic
  (`weyl_deficit`) overlaps: clean up to 1.04, drops from 0.15. In-sample, so:
  1. DONE 2026-10-11 (`notes/count_audit.md`). Validated on the douse benchmark (100 of 100
     drops, no false alarm), then widened to 18 shapes with known spectra. That widening found
     the douse-only threshold too tight: a 5:1 rectangle and the L-shape alarm when complete.
  2. BUILT 2026-10-11: `asymp.count_audit` (`short` above 0.25, `excess` below -0.6; prefix
     windows only), wired as `diagnostics.count_audit` (`tests/test_count_audit.py`). No false
     alarm on any complete spectrum measured; ~94% of single drops caught on spectra of <= 30
     levels with 4 levels of headroom, and weaker deeper. A backstop: drops stay the planner's.
  3. BUILT 2026-10-11: `on_shortfall="flag" | "raise"` on `search_interval` and
     `solve_interval`; `PrecisionShortfall` (`tests/test_on_shortfall.py`).
  4. MEASURED 2026-10-11 (`notes/early_shortfall.md`): the early abort saves a median 57% of a
     failing solve's wall time, so `"raise"` stops at the first refined level that misses
     `prec`. The floor pre-scan predicts unmet levels at AUC 0.59, so it is dropped.
  5. RESCORED 2026-10-11 (`notes/count_audit.md`): 100 of 101 real drops caught, no false alarm
     in 654 clean solves. The one miss, a drop replaced by a duplicate of its neighbour, is
     invisible to counting.
  **Phase 4 done.** Left for the planner: drops on deep spectra (the audit weakens), and the
  Faber-Krahn prefix condition on elongated shapes (start windows low).
* **[later]** **Sharp, thin-wedged douse shapes cap near 1e-7, whatever the basis size**
  (`notes/hard_cases.md` parts 2-3): T1.3, T2.2, T2.6-N3k4-L8 and T2.6-N3k5-L6, with corners of
  11-20 degrees (40-55 for the last). It is not Fourier-Bessel evaluation, which agrees with
  30-digit mpmath to ~1e-14. Enriching any single corner does not help, nor does dropping
  inaccurate columns or doubling collocation. Untested candidates:
  - the corner expansions' convergence regions on thin wedges;
  - too few interior points inside the wedges;
  - TAPERED lightning clusters (untapered ones made solves decline).
  Until then the planner ends these FLAGGED at its top rung.
* **[next]** **Fourier-Bessel's accuracy warning is not substantive at 1e-10.** `Sampling.accuracy`
  bounds read ~1e-7 to 1e-9 on planned bases, while 30-digit mpmath shows the columns good to
  ~1e-14 (`notes/hard_cases.md`, part 3). So `PrecisionWarning: FourierBessel can deliver about
  1e-9 ...` fires on ordinary 1e-10 solves, against the warn-only-when-substantive rule.
  Tighten the bound, or stop warning on it.
* **[next]** **The search can ask the pencil for a NEGATIVE lam.** `notes/hybrid_bases.py`
  round 2, T2.6-N3k4-L8-ref0 with Fourier-Bessel plus corner lightning (default sigma 4,
  corner-graded collocation) at 1e-10. The window is [~25, ~300], and the solve raised
  `ValueError: lam must be finite and strictly positive, got -348.16` from the Helmholtz layer.
  Some stage evaluates outside `[lo, hi]`, probably extrapolating a step on an ill-conditioned,
  flat objective. A search must clip every evaluation to its window, or decline. Reproduce with
  `HYBRID_ROUND=2 .venv/bin/python notes/hybrid_bases.py` (variant "FBs+L16 graded").
* **[later]** **Lightning sources, as lappy set them up, were not finished measuring.** Round 3 of
  `notes/hybrid_bases.py` (`HYBRID_ROUND=3`: sigma 1 or 2, poles out to ~1, order 2, a 50/50
  column split with Fourier-Bessel) was stopped to ship Phase 5. With chladni's defaults (sigma 4,
  L = the corner's clearance, order 1), lightning declined on almost every case, graded
  collocation included: very ill-conditioned columns. The shipped ladder uses Fourier-Bessel plus
  boundary-offset sources instead.
* **[next]** **Rebuild T2.8-N5s2's reference on current code.** Its trusted value 179.302250990678
  (status `fem_resolved`: one run, recipe B, plus a finite-element count) does not reproduce.
  Recipe B on current code gives 179.3022510764, uncertified. Every current basis -- corner-only A
  and C, and the planner's hybrid with sources (Moler-Payne 4.7e-11) -- agrees at 179.3022511196.
  It is the only SILENT row of the Phase 5 benchmark at 1e-10 (`notes/phase5_benchmark.md`), and it
  puts the "biased well" reading in `notes/admission_bound.md` in doubt.
* **[with the planner]** **Re-base the Phase 3 contract test on a fair basis.**
  `tests/test_level_resolution.py` runs on the stand-in basis D (1.0x lappy's per-corner FB
  sizing), so it partly tests the basis. Move it to planner-built bases, and judge only levels
  the basis demonstrably resolves (Moler-Payne meets `prec`), so that a count error left is the
  solver's. Even the references' recipe A (1.5x, FB only) misses 1e-10 on ~40% of levels.
* **[next]** **The planner refuses a basis that fails `basis_sufficiency`'s multiplicity
  precondition.** It is the guard for a double read as simple on a MET level, which only starved
  bases showed (the `count_ratio` grid, R up to 18).
* **Phase 5 BUILT 2026-10-12: the planner and `Eigenproblem`** (`chladni.plan`,
  `chladni.eigenproblem`; `notes/phase5_benchmark.md`). With no basis supplied, the douse
  benchmark comes back right on 33 of 35 cases at 1e-10 (30 unflagged) and 35 of 35 at 1e-8. T2.2
  is flagged. The one silent row, T2.8, is a suspect reference (item above).
* **[next]** **The `douse` seam, then M5 Hadamard** -- to the interface in
  `docs/design/douse_requirements.md`. `Eigenproblem.solve(n, prec, plan, on_shortfall)` is the
  backend call; `ep.plan(...)` plus `Plan.realize` is plan-once-realize-per-iterate.

## 1. The eigenvalue search

The search SHIPS. `chladni.mps.{bracket,refine,multiplicity,search}` are the four stages and
`MPSEigensolver.solve_interval` returns a `Spectrum`, so a `Domain` goes in and eigenvalues
with multiplicities come out. `tests/candidates_v1.py` holds the two adapters that keep V1 on
`candidates.md`'s scored seam, so a fourth candidate is still measured on the same instrument.

What is left here is not construction, and since 2026-09-18 it is not Phase D either. The
`modes(x)` widening landed, the interior Gram angle is built behind `phase_d=` and measured, and
the multiplicity contract is stated in `docs/design/search.md`'s As-built section. The cluster
grids (all rectangles) show no wrong totals on macOS; **off rectangles the search is not yet
trustworthy near the floor**: the L-shape's `10 pi^2` and the spike triangle's x8 (fixed by the (C2)
change, `tests/test_floor_multiplicity.py`), T1.2's near-double (below), and the `douse` benchmark's
silent rows (`benchmarks/douse/scoreboard.md`). What is left is the margin rule (section 0), the
constants still inherited from one rectangle family, and the rank policy that belongs to the
planner.

* **[next]** **T1.2's near-double is located 1, 2 or 3 times depending on ulps of the window.**
  `notes/nondeterminism.md` (2026-10-09): the solve is deterministic, but a one-ulp change to a
  window end 58-115 units away flips the count at douse's T1.2 pair (two eigenvalues 9e-10 apart,
  ~54 floor scales, so resolvable): 7 of 16 perturbations wrong, none count-flagged. Count 1 misses
  a member; count 3 locates one member twice, 4e-12 apart, and the dedup keeps both. At `prec =
  1e-12` the pair sits at `R = 4.1`, inside the margin rule's regime (section 0), which may absorb
  it -- trace stages 1-2 only if it does not.
* **[next]** **Drive the `ulp_sensitive` tests to 0 of N jitter seeds -- one left.** The suite
  pencil is cut for 1e-10 since 2026-10-11 (`rect_suite.FINEST_PREC`; it was 1e-14), and 15 of the
  16 quarantined tests then passed 32 of 32 jitter seeds and went back into per-push CI
  (`notes/suite_rtol_sweep.md`, `notes/jitter_sweep_rung_pencil.json`). Their flips came from
  near-null directions a `sqrt(1e-14 eps)` cut keeps. Left: `test_floor_multiplicity`'s noisy-floor
  case [A], which builds its own pencil (1 of 32 seeds, and Linux Zen 3 returned 916.08 x2).
  Not re-measured on the new pencil: single cluster-grid cells, which flipped at ~1/16 on the old
  one (`test_the_cluster_total_GRID`, unmarked). `solve(collapse=True)` and `rule="gap"` are
  untested options since 2026-10-09 -- retire them or re-test them.
* **[next]** **Under ulp noise, `refine` accepts a bracket in a spectral GAP as a root.** The
  2026-10-09 jitter sweep of the pruned suite (`notes/jitter_sweep_after_prune.json`, seeds 4 and 7):
  `test_refine::test_it_rejects_an_empty_bracket_on_a_real_pencil_cheaply`, re-posed at 1e-10,
  returned `accepted=True, reason='converged'` at `sigma = 0.41` -- an O(1) value nowhere near an
  eigenvalue -- in [217.25, 217.35]. A spurious root a search would report; the test is
  `ulp_sensitive` meanwhile. **Traced 2026-10-10:** not a rank flip (the rank stays 57). Under
  seed 4 the plateau's `sigma_1` near the gap's peak is flat and NOISY -- 0.41075 .. 0.41082 over
  4e-6 in `lam`, ~2e-4 relative, from a 4-ulp perturbation of a pencil truncated at `rtol` 1.5e-15,
  which keeps near-null directions the perturbation shakes -- and `refine` found an interior dip
  below both ends. `_accept` (`mps/refine.py`) has no DEPTH test: "strictly interior and below both
  ends" is all it asks, so a noise dip on a plateau passes. The same plateau under
  `reg="tikhonov_qr"` at 1e-10 wanders ~4e-6 relative, 50x quieter. In a real search stage 1
  rarely brackets a plateau peak, so the exposure is through spurious brackets in noisy regions.
  A depth criterion (relative to the ends or the plateau) is the obvious fix; it touches a
  calibrated function, so it waits for the margin rule.
  On the rung-sized suite pencil (2026-10-11) the test passes 32 of 32 seeds: the noise was the
  1e-14 cut's. The missing depth test is still a defect, with less exposure at the default cut.
* **[next]** **`delta_res` is admitted into `Level.bound` at the very edge of `prec`.** `c1c3f53`
  admits `delta_res` when `delta_res <= prec*lam`. On a Linux EPYC 9V45 runner (2026-10-09) the
  starved-basis test's level at 44.948 had `delta_res = 4.445e-11` against `prec*lam = 4.495e-11`
  -- admitted by 1% -- and the reported bound was 700x under the true error (3.1e-8; Moler-Payne,
  3.0e-6, covered it). Admission needs a margin, `delta_res <= prec*lam / M`. On the 715-level grid
  (`notes/bound_coverage.jsonl`) M = 10 leaves 6 levels under-covered instead of 10 and turns 24 met
  levels unmet; settle M with the margin rule, under jitter. **Second case, on a douse shape and
  an adequate basis (2026-10-11, `notes/phase3_rescore.md`):** T2.8-N5s2, basis A, 1e-10 -- admitted
  at margin R = 3.06 (1.5x the edge), 11x under the error that reference run B's Moler-Payne bound
  pins. It is the only SILENT row left on the benchmark after Phase 3, and it is a value, not a
  count. **Measured 2026-10-11 (`notes/admission_bound.md`): it is basis BIAS, not edge
  admission.** At 1e-8 the same level has R = 56 and is still 1.9x under-covered, so no margin
  M fixes it. Moler-Payne-only is honest but flips ~30% of met levels to unmet at 1e-10, nearly
  all truly within `prec`. A cross-check -- admit `delta_res` only if also `mp_bound <= 20 *
  prec*lam` -- removes the harmful case and the Linux one at a cost of 2-5 met levels per ~1200.
  **Decided: change nothing yet**; revisit with the planner, whose basis choice may make the
  biased well rare.
* **[later]** **The Linux count flips at OpenBLAS's default threading are not explained.** The
  2026-10-05 `main` run failed 7 truncation-scored tests on Python 3.11 and 3.13 and passed on 3.12,
  ~4 in 5 at full-suite scope; single-thread OpenBLAS measured 0 in ~39 runs. CI now pins one
  thread (`.github/workflows/tests.yml`), which HIDES the flips. The lead is
  `notes/nondeterminism.md`: a search discontinuous in its inputs turns last-bit BLAS differences
  into different counts, with no nondeterminism in either library. **Confirmed 2026-10-09**: the
  failure set followed the runner's CPU (EPYC 7763 vs 9V45), `OPENBLAS_CORETYPE` equalized only
  the former, and `tests/ulp_jitter.py` reproduces every Linux failure on macOS.
* **[next]** **`basis_sufficiency` decides the DENOMINATOR at a boundary it clears by 6%, and the
  two platforms therefore score different case sets.** macOS calls 10 cells unexercisable where
  Linux calls 7. (Measured at 1e-14, a rung the suite no longer runs since 2026-10-09; the boundary
  question stands for the planner's own preconditions.) Two of the four `test_cluster_totals` skips
  sat on the precondition boundary:
  separation 4.78e-15 against a multiplicity resolution of 5.09e-15, a ratio of **0.94**, and
  2.84e-15 against 4.67e-15. On Linux those are scored instead of skipped.

  So "216 right, 0 wrong, 35 declined" is not over the same 251 cells on both platforms, and any
  cross-platform tally is a comparison of different denominators. Measured 2026-09-19,
  `notes/session_2026-09-19.md`. The precondition is right in kind -- section 8's rule -- and the
  question is whether a decision this close to its own threshold should be reported rather than
  silently taken.

* **[later]** **A threshold-rank flip at one bracket END makes the refiner refuse a real root.**
  OBSERVED on a Linux runner 2026-09-21, bit-identical over 3 repetitions, and reproducible on
  macOS in a second (`notes/session_2026-09-19.md`, the 09-21 section). This was the mechanism of
  the former blocking CI item. `notes/landscape_findings.md` section 37 filed the `125*pi^2` rejection
  as costing nothing "but ... nothing guarantees it stays harmless"; this is that shape, costing
  the one wrong total CI ever had.

  The chain: stage 1 brackets all four roots. In bracket 0, ONE evaluation -- the upper end --
  retains rank 86 instead of 85, because `s[85]/s[0]` sits at 0.89x the cutoff on macOS and a
  few percent more on OpenBLAS crosses it. Direction 86 is noise, so its effect on `sigma_1` is
  jagged in `lam` -- a 67% drop to none across 8e-9 relative -- and at that end it drops `sigma_1`
  3.1x. The end becomes the deepest point the refiner holds, it walks outward, and it refuses
  `not_a_minimum` at the flipped end.

  **Two claims this file made on 2026-09-20 were wrong.** The mechanism is NOT `no_parabola` --
  that was argued from the code and never observed. And `_deep_rejections` does NOT catch this
  cell: the refused point is a flipped FLANK point, `sigma = 5.18e-07`, shallower than the worst
  located level, so the rule reads `()` on Linux. Its "flags the one wrong cell" came from macOS,
  where the cell is right.

  **Exposure is general, not local to this cell.** In 70 of this solve's 83 evaluations the first
  dropped singular value sits at 0.80-0.90x the cutoff; a flip moves `sigma_1` up to 3.5x on the
  flanks and barely at the roots, which is where `_accept` compares ends. The pencil's own
  retained rank also disagrees with `choose_rank`'s `svdvals` rank at 2 of 83 points on ONE
  platform, where `svdvals` puts the kept value 24-37% above the cutoff.

  The remedy is an objective CONTINUOUS in `lam`, and it exists now as an OPTION:
  `reg="tikhonov_qr"`, a penalty in place of the truncation, so no rank is chosen and none can
  flip (`notes/smooth_filter.md` section 9). On macOS it replayed this refusal deterministically
  and fixed it -- with the flip injected `refine` refused exactly as recorded; under the penalty it
  accepts, 1e-10 from the root -- and solves the cell 4 of 4 (`tests/test_regularization.py`, the
  "2026-09-21 CI refusal" section, which since 2026-10-09 keeps only the penalty half: the injected
  replay pinned refine's reason string and failed 8 of 8 jitter seeds).

  **Run on Linux 2026-10-04, and demoted to `[later]` -- PREMATURELY (see the item above, 10-05:
  the suite row failed again on two of three legs).** The default path at `prec >= 1e-8` is
  the penalty and has no rank to flip. What stays open is the flip itself, on every TRUNCATION:
  `reg="auto"` below 1e-8, and any explicit `rtol=`, `rank=` or `lam_ref=`. Its passing on Linux
  now is a moved bracket end, not a fix, so this item stays until either the penalty's range
  reaches the bottom rung or the planner owns the rank (the item below). Section 12 of
  `notes/smooth_filter.md` added a look past a refused bracket END; it does not cover this case,
  where the root is inside and the flipped end points away from it. Truncation stays the 1e-10 default (decided 2026-10-10), so this stays open.
* **[next]** **The threshold rank makes the objective piecewise, and a search cannot bound a
  slope across a jump.** DEFERRED TO BASIS PLANNING, deliberately: the rank policy is the
  planner's to own, it is entangled with `rtol` and the floor it induces, and section 3's planner
  is where the two halves of the `_UNSAFE_RATIO` question get decided together rather than one at
  a time. Recorded here in full so the planner inherits the measurement rather than rediscovering
  it. `notes/landscape_findings.md` section 34. At `145*pi^2` **30 of
  55 cells decline**, every one of them `_seed`'s "the slope estimate never settled", and the
  rising secants that cause it are RANK FLIPS: at every seed density the largest secant sits
  exactly on a `107 -> 108` boundary with `sigma_1` halving across it, so the secant grows like
  `1/dk` and cannot settle (2.05, 3.38, 6.04, 19.82 at n = 64, 127, 253, 505).

  Freezing the rank (`lam_ref`, nothing else changed) removes every flip, takes the max secant to
  1.52, moves the minimum sampled `sigma` by 2%, and on whole grids:

      grid        threshold rank                    frozen rank
      145*pi^2    22 right, 0 wrong, 23 declined    44 right, 0 wrong,  1 declined
      65*pi^2     47 right, 0 wrong,  6 declined    47 right, 0 wrong,  6 declined
      50*pi^2     53 right, 0 wrong,  0 declined    53 right, 0 wrong,  0 declined

  **This is the SHIPPED default**, not a test artifact: `MPSEigensolver` takes no `lam_ref` and
  `rank_at` re-derives per `lam`, justified as "the only rule that cannot pick an unsafe rank" --
  an argument about picking a rank too HIGH, silent on the objective being discontinuous in `lam`.
  Not every decline is this: `65*pi^2`'s six are unaffected, and the roster pencils are
  threshold-rank too but untested for it.

  **And freezing is not the answer either**, which section 35 measures and section 34 did not:
  its 2% figure came from a 6%-wide window. Over `[5, 300]` on `rect(1, 1/phi)` one frozen rank
  inflates the floor up to **2.0e11x** -- `choose_rank` returns 29 where the lowest eigenvalue
  needs 39 and the highest 61 -- at a steady 2 to 3 retained directions per decade of floor. The
  asymmetry is what makes `choose_rank`'s MINIMUM the wrong end: short is fatal, long costs 264
  spurious minima and no floor at all.

  What separates the two regimes is the WINDOW WIDTH -- the threshold rank is constant below
  about 10% -- so **this is one decision the code lets three places make separately**: the window
  (`weyl_window` or a caller), the rank (`choose_rank` or the threshold rule) and the floor
  (`rtol`). A planner chooses a `(window, rank, floor)` TRIPLE. Section 35 also records the third
  option that dissolves the dichotomy -- a smooth spectral filter in place of the hard cut, since
  a matrix function is continuous where a rank projector is not -- with its three costs, and the
  cheapest thing that needs no new numerics: size the sub-window from the rank, freeze within it,
  and the jumps land on boundaries the search restarts at.

  Two remedies are measured NOT to work and are in section 9. The decline rate is now a scored
  quantity (`test_cluster_totals.MAX_DECLINE_RATE`; 7.6% over 66 exercisable cells at 1e-6..1e-10,
  2026-10-09, where it was 5.7% over 106 with the 1e-12/1e-14 rungs); the "5% the
  robustness plan set" cited here before came from a plan not in this repository, and the "24 of
  45" beside it contradicted `search.solve`'s own table, where 24 is the number RIGHT.
* **[next]** Carry the in-loop stride correction past the seed. The bracketing stage now has
  one -- `v1_bracket._seed` sizes from Weyl and then doubles until the largest observed secant
  stops growing, so no a-priori stride is trusted -- but nothing downstream of it adapts, and
  the correction was argued for the whole solve. `notes/landscape_findings.md` section 20.2;
  `notes/solver_features.md` F3.
* **[later]** Two-level rank from one factorization -- the scan wants a low rank, the
  refinement a high one, and one SVD serves both. A measured pure win; `search.md` section 8,
  `notes/solver_features.md` F6.
* **[later]** We should have a "warm-start" solver for eigenproblems that are perturbations of
  eigenproblems with known solutions: a robust mechanism for taking old estimates and refining
  them. This will have to handle multiplicity correctly, as well as target precision. The
  warm-start solver may require bounds on the new eigs.
* **[later/maybe]** If we cannot make the air-tight search have excellent average-case performance,
  as measured by number of GSVD calls, then we should implement a "fast cold start" solver
  which has excellent average-case performance, with guards that can opt back-in to the slower-
  but-air-tight solver.
## 2. Generalization and validation debt

Every number in `CHLADNI_V1.md` was measured on one family, the `(1+delta) x 1` rectangle, and
the document says so in its own summary. `tests/hard_roster.py` is the second domain that
section 10 item 1 asked for -- the disk and equilateral triangle for provable coverage, the
L-shape for a re-entrant corner -- and it settled the cost question: 22-24 evaluations per
eigenvalue, against 26-27 on rectangles. What it has not settled is below.

* **[next]** Re-measure the three constants that are still inherited: `straddle = 0.02`,
  `margin = 1.5`, `safety = 1.5`. Each carries a `RECALIBRATE` comment in `chladni/mps/bracket.py`
  naming what it was measured on. The others are done: `kappa` re-measured at 2.591 +- 0.226
  against the study's 2.384 (`notes/evidence_tests/test_calibrate_evidence.py`), the wall slope at 0.64-1.41 across
  four domains, and the floor set by `sqrt(prec*eps)` landing on the crossover of Betcke's
  relation (`tests/test_floor.py`). `CHLADNI_V1.md` section 8.
* **[next]** Confirm that stopping at isolation is safe. Section 8 calls it "the change most
  in need of confirmation elsewhere": it trades an unconditional guarantee for one resting on
  the `sigma_2` gate, which has produced coverage failures twice.
* **[next]** Check whether the 5% slope spread in `k` survives corners and re-entrant angles.
  If it does not, the case for a *local* slope model returns. Section 8. **Not yet answered,
  and the diagnostic that looks like it answers it does not**:
  `SearchDiagnostics.slope_spread` is the spread of every sampled secant, plateau to wall, and
  reads 0.97 to 1.17 on the disk, eq_tri and L-shape -- while V1's 5% is the spread of the
  *wall slope across levels*, a different quantity. Either measure `c` per level on those
  domains, or rename the field so it stops looking like an answer.
* **[later]** One of the six test files `search.md` section 7 specifies has no equivalent:
  `test_search_multiplicity.py`. The others exist under different names -- `test_search.py`
  (scan), `test_refine.py`, `test_bracket_score.py` (score), and the `prec` ladder threaded
  through `test_search.py`, `test_precision_ladder.py` and `test_hard_roster.py`. (`test_gate.py` and
  `test_search_collapse.py` moved to `notes/evidence_tests/` on 2026-10-09.)

## 3. Unwritten layers

* **[blocking]** **Basis planning.** `MPSEigensolver` requires an explicit `basis=`, so there
  is no black-box path at all. `notes/handoff_minimization.md` section 4: everything above is
  quietly waiting on it, and search difficulty is downstream of basis quality -- `eps` varies
  five orders *within a single L-shape scan*, structurally -- so a search benchmarked against
  hand-built bases may not predict black-box behaviour. Whatever the planner becomes, its
  output should include a contract: the `eps` and dip width a search can tolerate. Deferred in
  `docs/design/mps.md` and `docs/design/particular_solutions.md`;
  `src/chladni/helmholtz/sources.py:271` defers the taper decision to it.

  It is better equipped than it was, by two measurements made while promoting the search.
  **The target now has a closed form.** `delta_res = DELTA_SAFETY * eps/c`, so
  `delta_res <= prec*lam` means `eps <= prec*lam*c/DELTA_SAFETY` -- a floor to hit, where
  `search.md` section 2.5 previously said no closed form was available. And **it must not
  hill-climb on column count**: on the L-shape, corner Fourier-Bessel buys thirteen orders over
  plane waves alone (3.8e-01 -> 4.3e-14) and then REGRESSES 10x from 40 terms per corner to 80,
  while on the disk plane-wave enrichment from 0.6x to 2.5x `n_useful` moves the floor not at
  all because the rank saturates. More is better until it is not, and where the turn happens is
  a property of the family.
* **[next]** **Boundary integrals, M5: Hadamard** (`docs/design/boundary_integrals.md`), to the
  interface in `docs/design/douse_requirements.md` (`spec.hadamard(levels, V)`). Still open from M2:
  a candidate whose basis mixes smooth columns with corner ones. Also open: the Moler-Payne bound
  is loose on `douse` polygons -- 1e-6..3e-5 relative where three bases agree to ~1e-8
  (`benchmarks/douse/scoreboard.md`) -- so it cannot yet vouch for a precision tier there.
* **[next]** `Eigenproblem`, the black-box entry point `CLAUDE.md` names as the ultimate goal.
  It does not exist; `Eigensolver` plus `weyl_window` is the intended spine. Blocked on the
  planner (section 0).
* **[later]** Eigenfunction and residual plotting. `src/chladni/plotting/_axes.py:3` already
  names them as "still to come" and `prepare` is the seam.
* **[later]** Corner grading. `grading="corner"` / `_grade_flags` is "a placeholder, not a
  design" (`docs/design/geometry.md:449`): it over-grades right-angled Dirichlet and Neumann
  vertices, and node placement wants the integrand, so a quadrature layer should own the
  choice and reach in through `panels=`. Superseded by `chladni.integrals` (item above), whose
  corner rules replace geometric grading for every integral; `grading="corner"` stays for
  collocation only.
* **[later]** Interior cubature, deferred until something asks for it
  (`docs/design/geometry.md:260`). Note that `notes/solver_features.md` A6 argues against ever
  making this a project: the interior block wants non-degeneracy, not accuracy, and anything
  that must be *accurate* belongs on the Rellich boundary integrals.
* **[later]** User-facing documentation -- the "fifth kind" in `docs/README.md`. `docs/_build/`
  is already ignored for it.

## 4. Known gaps in what is built

* **[later]** **The factored Bessel kernel's accuracy advantage over plain `scipy.special.jv` is
  unproven end to end.** `helmholtz/_bessel.py` (series / scipy / Olver-Debye, values as
  `S * exp(sigma)`, with the Airy-regularized envelope in `fourier_bessel.py`) is validated as a
  KERNEL against mpmath, and is faster than `jv` + `jvp` when derivatives are wanted. But with the
  same points, equilibration, regularization and search, raw `jv` columns (underflowed columns
  dropped) give the same eigenvalues: identical on the L-shape (80 terms, 1.6e-14, no underflow),
  and indistinguishable on `douse`'s spike triangle (failure catalogue T1.3) at 1e-9 .. 2e-8,
  including with 16-20 high-order apex columns lost to underflow. Expected from the normwise
  argument in the `PrecisionWarning` item above. One of the most complex parts of `chladni`, so a
  simplification candidate -- decided by the two tests not yet run: a high-precision basis on a
  sharp shape (these bases floor at ~2e-7), and a Neumann/Cauchy-data case, where J' from the series
  may beat `jvp`. Repro: `notes/bessel_kernel_vs_scipy.py` (2026-10-06).
* **[next]** **`FourierBessel`'s `PrecisionWarning` flags per-entry relative error, which is not a
  substantive loss.** The user's rule (2026-10-06): **warn only for precision losses that are
  substantive** -- ones that can move a computed result. `Sampling._check_accuracy`
  (`helmholtz/_core.py`) warns on `max(rel_err)` over all entries, and the series branch's
  `rel_err = eps * largest_term / |S|` (`helmholtz/_bessel.py`, `_j_series`) blows up at entries
  near a zero of `J_mu(k r)`. On the L-shape (80 reentrant-corner terms) the worst bound is 2.1e-8
  at `lam = 42.1`; mpmath gives a true relative error of 2.7e-9 there, on an entry 1.9e-6 of its
  column maximum -- 5e-15 relative to the column, and at most 2.6e-13 over all entries. The solve
  is normwise (equilibration, GSVD), so eigenvalues come out to ~4e-14 whether the family's `prec`
  is 1e-8 (warns occasionally, depending on which `lam` the search visits) or 1e-12 (185 warnings
  in one search). The likely fix is a column-scaled criterion (`rel_err * |entry| / column scale`,
  which `log_scale` already supplies), but `docs/conventions.md` records "`helmholtz` owns
  entry-wise accuracy" as a design choice, so the criterion is the user's decision. Audit the other
  `PrecisionWarning` sites against the same rule. Repro: `notes/precision_warning_entrywise.py`.
  Not established: branch C (Olver/Debye); other domains.
* **[later]** **The two (C2) fixes of 2026-10-07 are measured on two shapes.** `search.solve` now
  (i) uncounts a reach-counted branch that does not dip at the root against the cached wide
  bracket, and (ii) counts a declined branch the side search left unaccounted when it does dip,
  else flags the level (`_c2_holds`; `tests/test_floor_multiplicity.py`). They fixed the spike
  triangle's x8 (`notes/spike_x8.py`: a cache slope read 14000x steep inside a noisy floor) and the
  10 pi^2 double certified simple at 1e-12, and the full suite including the gate and the robust
  tier is unchanged. The cache slope itself is still wrong on a noisy floor -- the guard in
  `_slope_from_cache` is defeated when the floor's noise exceeds its 2x flat-bottom margin -- and
  other consumers of `c` (the side search's exclusion radii, `wide_bracket_for`) still read it.
* **[next]** **The penalty is the default at `prec >= 1e-8`; what is still open about it.**
  SHIPPED 2026-09-24: `reg="auto"` on `MPSPencil` and `MPSEigensolver` is `"tikhonov_qr"` with
  `tau = C_PEN * prec` (`C_PEN = 0.02`) at `prec >= 1e-8`, and the threshold at its derived `rtol`
  below; an explicit `rtol=`, `rank=` or `lam_ref=` means truncation. An explicit penalty outside
  its measured reach warns and runs. `notes/smooth_filter.md` section 11 has the grid, and
  `tests/test_cluster_totals.py` now scores the default pencil beside the suite's.

  **What it buys is the removal of a hazard, not a better score.** Against the threshold at its
  OWN derived `rtol` -- what shipped before -- the 50/65*pi^2 cluster grid is 22 of 22 either way
  at 1e-6 and 1e-8. The "beats the threshold" rows of section 10.6 were against the suite pencil's
  fixed `SUITE_RTOL`. At 1e-10 the penalty lost a member the threshold kept (flagged), which is
  why the range stops at 1e-8 and not at 1e-10 or 1e-12, where a feasible `tau` exists.

  Still open:

  1. **`calibrate_rtol` under a penalty.** `findability(calibrate=True)` on a default pencil at
     `prec >= 1e-8` now calibrates a PENALTY's `rtol` with a model written for a truncation cut.
     Not on the solve path (`src/chladni/mps/solver.py:178`); unmeasured -- except that the
     reach warning now fires when the calibration loosens `rtol` past it, which is how
     `tests/test_calibrate.py` found this (its pencils are now explicitly `reg="svd"`).
  2. **`"tikhonov"`, the literal form, does not remove the rank decision** -- LAPACK's `?ggsvp3`
     decides the interior block's rank itself. Kept as an oracle; whether it stays user-facing is
     open.
  3. **Cost.** 1.5x per evaluation on the objective (section 9). NOT measured as wall time on the
     default path: suite time moved 765 s -> 883 s, almost all of it the new default-pencil grid,
     because nearly every fixture still pins a truncation (section 11).

* **[later]** A starved basis DROPS the levels it resolves worst, and for the levels that are
  never BRACKETED no diagnostic added to `solve` can catch it -- `notes/landscape_findings.md`
  section 38. RECLASSIFIED to basis planning (section 3) rather than an honesty gap in the search.
  **Re-measured at 1e-10 (2026-10-09)**: `solver_for(disk, 60, pad=0.35)` returns 2 of 6 levels and
  `pad=0.45` 5 of 6, no decline, flag or unmet level -- stage 1 never brackets them. At 1e-12 the same
  losses were caught only incidentally. `tests/test_solve_interval.py` carries three
  `xfail(strict=False)` tests that document it; the planner and the Weyl audit are the defence.

  **The claim was stated too broadly, and one case of it is now caught.** It read "no diagnostic
  added to `solve` can catch it", and the `pad = 0.45` case it cited is catchable, because stage 1
  DID bracket the level and the refiner then refused it: `_deep_rejections` (2026-09-20) reports
  the refusal at `lam = 14.68197`, `sigma_1 = 4.94e-13`, reason `not_a_minimum`, and
  `solve_interval` raises. Its strict xfail in `tests/test_solve_interval.py` is retired. The rule
  was built for a wrong total on a rectangle cluster and catches this on the way past, which is
  the evidence it keys on something real.

  What survives is the narrower and still-true version: **a level the search never brackets is
  undetectable from inside `solve`**, because there is nothing to compare against -- "no
  eigenvalue here" and "one my basis cannot resolve" are the same observation. The distinction
  worth keeping is between a level LOST SILENTLY (no bracket, still undetectable) and one
  REFUSED (bracketed and thrown away, now loud).

  Both candidate alarms are measured to fail, and differently. The **Weyl deficit** is
  INSENSITIVE, not inverted: 0.91 with nothing missing, 0.91 with two missing, 0.87 with six,
  every reading inside its own +-1 baseline. Its cuts are derived from the data being audited, so
  a missing level removes the cut AND the values below it; every cut lies below the top returned
  level, which is where a starved basis loses them; and it returns `nan` at the worst pad, where
  one level comes back claiming multiplicity 8. **`findability` is flat, and correctly so** -- it
  keys on the PLATEAU and the catchment width, which starvation does not move, while starvation
  raises the FLOOR.

  The quantity that does separate the ladder is the tension floor AT THE LEVELS (4.0e-03 down to
  3.7e-10), and reading it requires knowing where the levels are -- circular for the ones lost. A
  search cannot distinguish "no eigenvalue here" from "one my basis cannot resolve". That is the
  planner's contract to publish, not the searcher's to detect.

  (Beware the multiplicity trap when re-measuring: `weyl_count_poly` counts WITH multiplicity, so
  audit `SolveResult.unfolded`. Auditing distinct levels reports +4.70 on the TRUE spectrum and
  invalidated the first attempt at this. Invisible on the L-shape, which has no degeneracies and
  is what `test_hard_roster.py` audits.)
* **[next]** Nothing on the solve path calls `MPSEigensolver.findability`
  (`src/chladni/mps/solver.py:128`). The check is built, exported and tested
  (`src/chladni/mps/calibrate.py`), it costs ~40 evaluations, and it
  reports the scan step a domain actually needs -- a sharp elongated triangle needs 8x the
  density of a square. But `solve_interval` raises, so no caller reaches it, and the searcher
  does not read `required_dk` to size its seed. That composition is the note's fix (2) and is
  the whole point of the check.
* **[next]** Weyl asymptotics raise for any `bc_type` other than Dirichlet or Neumann
  (`src/chladni/asymp.py:39`), and there is no curvature term, so a smooth domain gets only
  the two-term law. `docs/design/reference.md:293` records the same limit.
* **[next]** Decide whether `Sampling._check_accuracy`'s shortfall on Fourier-Bessel is real.
  It was invisible until the plane-wave conditioning guard was rescoped: eighteen modules
  suppressed the whole `PrecisionWarning` class to silence that guard, and took this signal
  with it. Uncovered, it fires on eight tests -- **about 1e-7 against a requested 1e-8** on the
  L-shape `pencil` fixture evaluated far from its `lam_ref`, and **about 1e2, i.e. no accuracy
  at all**, on the sector references in `tests/test_tension.py`. Both sites carry an argument
  for why the bound is not their claim, written before the mark that hid it:
  `tests/test_regularization.py` (the module-level filterwarnings, "accuracy at those lam is not the claim, and is tested in
  test_precision.py") and `tests/test_tension.py:43` (with a SINGLE column the common kernel
  scale divides out of the tension). The arguments are plausible and the `< 1e-10` assertions
  do re-check the sector case immediately -- but neither has been verified since the warning
  became visible, and the sector is the project's sharpest ground truth. The suppressions are
  now targeted by message, so the rank and findability signals stay live while this is open.
* **[next]** Enrichment is NOT monotone, and nothing tests that it helps at all. Measured on
  the L-shape, worst tension over the first six levels: plane waves alone **3.8e-01** (the
  basis cannot represent `r**(2/3)` and the solve is meaningless), then corner Fourier-Bessel
  at 5/10/20/40 terms per corner giving 5.7e-03, 3.1e-05, 6.0e-11, **4.3e-14** -- thirteen
  orders, and the whole argument for corner families. But 80 terms per corner reads
  **4.8e-13**, a 10x REGRESSION past the optimum around 40. So "add more basis" is not a safe
  default and a planner cannot hill-climb on column count. Two things follow: the enrichment
  claim in `docs/design/particular_solutions.md` is still untested (section 4 above), and the
  regression above the optimum is not recorded anywhere. Note this is Fourier-Bessel, so it is
  NOT the plane-wave conditioning bound `n_useful` describes -- on the disk, enrichment from
  0.6x to 2.5x `n_useful` moves the floor not at all (rank saturates at 49 from 38 to 160
  columns), which is the case that made the old `log_scale` guard fire on healthy setups.
* **[next]** `_UNSAFE_RATIO` and `SUITE_RTOL` disagree, and neither cites the other.
  `src/chladni/mps/pencil.py:68` calls a cut below `1e-13` unsafe, measured on the disk;
  `tests/rect_suite.py` set the suite's threshold rank to `sqrt(FINEST_PREC*eps)` = 1.5e-15 until
  2026-10-11 (1.5e-13 since, cut for 1e-10; the evidence modules keep 1.5e-15), so every
  threshold-rank pencil in the suite cut two decades BELOW the level the frozen-rank guard calls
  unsafe -- deliberately, and measured to work (0 spurious minima in `[5, 35]` on
  `family_A`, `sigma_1` of 8.6e-16 to 1.5e-15 at all eleven true eigenvalues). Either the
  constant is domain-dependent, or the threshold path is safe at cuts the frozen path is not,
  and which it is has not been measured. Related: `_warn_if_unsafe` fires only for a FROZEN
  rank and only from `rank_diagnostics`, never from `tension`, so a caller looping over
  `tension` with a bad frozen rank is told nothing.
* **[later]** The end-to-end reference values in `docs/design/particular_solutions.md` --
  sector eigenfunctions, square, disk, equilateral triangle, and the L-shape's
  `lam_1 = 9.6397238440219410527` -- are written down but untested, because there is no
  eigensolver to test them with. They are the `mps` layer's first tests. Includes the L-shape
  Fourier-Bessel enrichment regression: 1e-12 to 1e-13 with the corner terms against ~1e-5
  without.
* **[next]** Accelerate's blocked `dgeqrf` corrupts the heap on macOS 26.5 / M4, and only the
  hot path is routed around it. **Its own regression test is intermittent for the same
  reason**: `test_the_geqrt_route_is_an_economic_qr[shape0]` compares the safe route against
  `la.qr` on a 548 x 274 matrix -- which IS the standalone reproducer -- so the guard calls
  the faulty path to check itself. Observed failing once in about ten default-tier runs on
  2026-09-18, passing five for five immediately after. The comparison needs a shape that is
  not the reproducer, or the stock call needs skipping when `_AVOID_GEQRF` is detected
  rather than forced. `regularize`'s svd-mode QR now uses `dgeqrt`
  (`src/chladni/mps/_regularize.py`, `_qr_economic`), which took the rectangle suite's
  `170*pi^2` cluster from exit 139 on every run to 3 of 3 clean with identical levels. Still
  on Accelerate's `dgeqrf` or unmeasured: `choose_rank` and `rank_diagnostics` (`svdvals` of
  the TALL stack, and `gesdd` QR-factors first when `m >= ~1.8n`), the `"qrp"` mode
  (`dgeqp3`), and the GSVD's internal `dggsvp3` at shapes other than the one probed. The
  standalone reproducer is `numpy.linalg.qr` of a random 548 x 274 matrix in a loop. Re-test
  each macOS release with `CHLADNI_ACCELERATE_GEQRF=1` and delete the workaround when it
  passes; scipy/scipy#26145 is the upstream report of the same fault class (`dpotrf`).
  Regression: `tests/test_regularization.py`, the slow `170_pi2` subprocess test.

## 5. Test-suite hygiene

* **[later]** The older public searches (`tests/searches.py`) have no live test: `test_searches.py`
  moved to `notes/evidence_tests/` on 2026-10-09 with the retired baselines. They are reached only
  through `tests/bracketers.py` and `baselines.py`, which `test_bracket_score.py` exercises.

## 6. Small deferred measurements

Each is one experiment. None blocks anything.

* Can a coefficient vector be reused across point sets to build Cauchy data, or does that need
  a GSVD recompute on the new pencil? `docs/design/mps.md:172`. Belongs with the eigenfunction
  work.
* Does the `lam`-over-`k` refinement advantage (17 of 17 cases, median 5.1x) survive at
  `delta_res`-scale brackets, and does it grow at large `lam`? `search.md`, As built.
* `PlaneWave.n_useful_at(k)` has no composite counterpart: `Basis` concatenates `log_scale` and
  `singular_exponents` across families but has no `n_useful_at`, so a mixed basis cannot report
  a bound. Nothing needs one today, because the verdict lives at the pencil and is
  basis-agnostic. A per-family basis planner would.
* The dip-width / gap invariant drifts mildly upward, 0.515 to 0.587 over `k` in `[3.5, 13.5]`.
  Re-measure above `k ~ 15` before anyone relies on it there.
  `notes/handoff_minimization.md` section 3c.
* Give the surrogate a `lam`-dependent rotation, so in-well eigenvector rotation exists and the
  swing angle has a realistic noise floor. A few lines, and it restores the one mechanism a
  swing prototype would need -- as it stands the surrogate's median swing angle is exactly
  zero, so it would report success where the real pencil fails.
  `notes/landscape_findings.md` section 31.
* The surrogate models no `sigma_3` and above, so any (C1) multiplicity conclusion drawn from
  it is provisional until confirmed on a real pencil. Now pinned by a test rather than left as
  a docstring caution.
* Diagnose the narrow plateau spikes that leave one gate cell unexercisable. On the
  multiplicity-3 cluster window the tension reads 0.074 at `k = 25.0555441` against 0.173 one
  part in 1e8 away -- a 2.3x dip about 1e-8 wide, sitting 0.24 from any eigenvalue, on a curve
  whose wall slope is 0.64 and whose floor is a healthy 3.4e-15. The rank is the obvious
  suspect and is not the whole story: `s_r/s_0 = 2.27e-15` sits right at the cutoff with
  `s_{r+1}/s_0 = 1.3e-15` just below, so the threshold rank flips 85 <-> 86 on a one-ulp change
  in `lam` and each flip moves `sigma_1` by 2.3x -- but freezing the rank over the window makes
  it *worse* (max secant 700 against 261). Sizing the collocation to the pencil removed five of
  the six affected cells, so it is partly a point-set effect. Whatever remains is what
  `assert_smooth` is currently reporting as a limitation.
* Derive `rtol` from `prec` directly, for the basis planner. Betcke's relation holds in the
  relative convention -- `eps ~ max(eps_basis, rtol)`, measured log-log slope 1.13 with
  `eps/rtol ~ 1.15` -- so the floor a threshold induces is predictable, and the largest usable
  `rtol` follows from the bracketing adequacy contract rather than from a sweep. That is a
  contract the planner could publish alongside `eps` and the dip width. `tests/test_floor.py`.
* `CHLADNI_V1.md` section 10: asymmetric cone prediction in the locator, fitting `c_left` and
  `c_right` separately (10-20% of the locate phase); a `sigma`-vector completeness audit,
  paired with the decline mechanism because it converts a certificate into an audited
  estimate. Its third item, batched evaluation, is in section 7.
* Everything measured so far sits at low `lam` -- the roster tops out at 90 and mostly at
  30-40 -- and no domain has a hole. `notes/solver_features.md` section 6.

## 7. Parallelization

`CLAUDE.md` asks that embarrassingly-parallel work be threaded where possible. Today exactly
one place does: `MPSEigensolver.tensions` maps an array of `lam` onto a thread pool
(`src/chladni/mps/solver.py:100`), on the argument that the work is LAPACK and releases the
GIL. Everything below is what that leaves open.

* **[later]** Decide whether the candidate instrument scores wall-clock rounds as well as
  evaluations. `candidates.md` sections
  3 and 4 count GSVD-level cache misses and nothing else, but with `p` concurrent GSVDs the
  metric becomes rounds rather than evaluations -- and both `search.md` section 8 and
  `lipschitz-cone-mps.md` section 8 say that if `p` is large a graded uniform grid may beat
  branch-and-bound outright, because the branch-and-bound advantage is adaptivity and
  adaptivity is worth less when points are nearly free. This is an ordering risk, not a
  refinement: an evaluation-count-only instrument can crown a winner that a parallel machine
  would not choose.
* **[next]** `chladni.mps.objective`'s `Objective` is deliberately not thread-safe -- a lock there would make
  the miss count depend on scheduling. So no parallel candidate can be scored on the current
  instrument at all. The way out is a batch seam, `Objective` taking a vector of points and
  counting misses deterministically, rather than a lock; a batch-submitting search is also the
  shape the parallel argument above wants.
* **[next]** Bound the thread pools, and measure the interaction with BLAS.
  `cf.ThreadPoolExecutor()` at `src/chladni/mps/solver.py:100` takes the default worker
  count, each worker calls into a multithreaded BLAS (Accelerate on macOS,
  `scipy-openblas32` elsewhere, per `pyproject.toml:23-33`), and `douse` runs whole solves
  across threads on top of both. That
  is three nested levels of concurrency with no cap at any of them and no measurement
  anywhere. Expose `max_workers`, and measure before choosing a default.
* **[next]** Batch the phases that are already embarrassingly parallel. `CHLADNI_V1.md`
  section 10 item 5: the cover phase is inherently sequential, but the seed is not and is
  ~20% of cost. `lipschitz-cone-mps.md` section 8 adds two more -- the pilot pass that
  calibrates `L`, and Phase B refinements, which are independent across brackets. All three
  can go through the existing array path in `tensions` as soon as a search submits points as
  a vector rather than one at a time.
* **[later]** Batched evaluation *inside* the pencil -- several `lam` in one assembly, if the
  factorizations allow it. This is vectorization rather than threading and is a separate
  question from the pool above; `CHLADNI_V1.md` section 10 item 5.
* **[later]** Extend the concurrency test past the `mps` seam.
  `tests/test_mps_seam.py:98` is the only test in the repo that actually starts threads, and
  it covers the solver. The thread-safety claim is made much more broadly than that -- it is
  the stated reason `Frozen` and `cached_property` are used at all
  (`src/chladni/_core.py:55`, `docs/conventions.md:42`) -- and `test_transforms.py` pins the
  immutability that implies it without ever running concurrently. A geometry- and
  `helmholtz`-level concurrency test would make the claim real rather than argued.
* **[later]** Parallelism across *domains* is `douse`'s to own, not `chladni`'s. The obligation
  here is only that a shared frozen object stays correct under it, which is the item above.
  Recorded so it is not built twice.

## 8. The precondition rule

Three tiers learned this separately and it is the most transferable thing in the build, so it
sits here rather than in a design record: **a case that violates a precondition measures the
case, not the method.** Every tier now checks before it scores, and a cell that fails is
reported UNEXERCISABLE with the reason -- never counted as a failure, never silently dropped.

    tier 1   surrogate_cases.adequate      prec*lam >= margin * eps/c
    tier 2   rect_suite.assert_floor       sigma_mu at every level, and the headroom
             rect_suite.assert_smooth      does the largest observed secant settle?
    tier 3   hard_roster.finest_prec       the tightest prec this basis supports

How each was found: tier 1 scored 13 of 66 cells as coverage failures against a surrogate floor
of 1e-13, which cannot support `prec = 1e-14` at all; tier 2's floor check existed, correct,
and was **never called from a test**; and the gate's first run reported nine failures, most of
them windows whose curve admits no slope bound, where the searcher declines correctly.

The corollary is a design rule for the search itself, and it is now implemented: **being asked
for more precision than the objective can deliver is not a failure.** The stall and budget
paths emit what they hold, flag it `floor_limited`, and report `attained_prec` beside
`prec_requested`, rather than turning "I achieved 3e-12 where you asked for 4e-13" into
"DECLINED, nothing found".

---

## 9. Measured not to work

Kept here because each was tried and measured, and the cost of retrying one is a session.
From `notes/solver_features.md` section 3 and `CHLADNI_V1.md` section 10.

* **No absolute tension threshold, anywhere.** The floor spreads four to five orders *within*
  one scan on three of seven domains. Thresholds are fractions of the measured plateau.
* **No `sigma_2`-value gate for cluster detection.** `sigma_2`'s minimum is `c*d/2` only at the
  crossing point, which is never a sample, so the gate tests the grid rather than the pair.
* **No offset detector expecting early warning.** The `sigma_2` offset relation is real, and it
  fires at exactly the coarseness at which `sigma_1` already shows two minima.
* **No distance-only collapse/merge.** One eigenvalue found twice and two genuine eigenvalues
  occur at the same separation; only the eigenfunctions tell them apart.
* **No structured interior fill.** `method="grid"` annihilates the mode `side+1` exactly, to
  1e-56, because its points sit on that mode's nodal lines.
* **No interior-cubature project.** See section 3.
* **No fixed tolerance expressing resolution or accuracy.** `search.md` section 2.1: a constant
  may be fixed only if it is dimensionless and its margin does not shrink as `prec` tightens.
* **No auto-swap to a symmetry-reduced solve on detected symmetry.** A tolerance threshold makes
  the objective discontinuous in shape, and it fires in the endgame of the optimization.
  `notes/mps_symmetry.md` section 2.
* **Do not design against a narrow dip.** No domain produces one; `c` is 0.445 to 1.725 over 19
  eigenvalues. The real failure mode is a cluster.
* **No handing a non-settled seed to the cover phase.** The argument is good -- `MAX_GROWTHS = 16`
  is already the falsification test, so let cover discover the walls -- and it is wrong. Measured
  across five cluster grids: 215 right / 1 wrong / 35 declined becomes 242 right / **9 wrong** / 0
  declined, three of them absurd (176 of 6, 184 of 6, 72 of 8), and the median cost at `145*pi^2`
  goes 961 -> 3249. With `L` set from a max secant that is a rank-flip artifact the exclusion
  radii are far too small, nothing is excluded, and every live region becomes a bracket. **The
  decline must not be relaxed at the seam that reports it**; the objective is what is wrong.
  `notes/landscape_findings.md` section 34.
* **No raising `MAX_DOUBLINGS` to make a cluster seed settle.** The secant history rises like
  `1/dk` because it is tracking a discontinuity, so there is nothing to converge to. At 12
  doublings the seed is ~260k points and still rising -- the cost the ladder law predicts, paid
  for nothing. Section 34.
* **Nothing that makes the cover phase sample more cleverly.** The ladder law says a wall
  yields to a geometric ladder set by the safety factor and to nothing else. Six attempts have
  failed. `CHLADNI_V1.md` section 10.
* **No raising `sigma_base` by loosening the rank truncation.** The study's primary remedy for
  narrow wells, measured there to move `sigma_base` by a factor of 48. In `chladni` the lever
  is **inert**: 0.3329 to 0.3346, a range of 0.5%, while the retained rank goes 47 -> 19. The
  shipped default is therefore the findability *check*, not the fix.
  `tests/test_calibrate.py::test_loosening_the_truncation_has_NO_LEVER_on_sigma_base_here`.
* **No oversized collocation.** 8x boundary / 3x interior per column against 1.2x/1x leaves the
  floor completely unmoved (1.4e-15 against 6.1e-16) at 1.5x the cost -- and it made the
  objective *rougher*, not just slower: sizing the point sets to the pencil took the gate's
  unexercisable cells from six to one and its runtime from 10:16 to 4:48. Fixed floors are
  worse than the ratios suggest, because they bind hardest on small bases.
* **No positive threshold on the Piyavskii envelope.** `env >= 0` looks unsound, since
  `sigma_1` bottoms at `eps > 0` rather than zero, so a finely sampled well can exclude itself.
  Replacing it with `env >= tau` broke **every** case, 0 of 66: the cover loop stops on purity
  by width long before that regime, so the zero threshold never bites.
* **No `straddle` fallback when the parabola fails to close.** The straddle is an offset from a
  *prediction*, and with no prediction a `STRADDLE * width` step shrinks the bracket by 2% per
  evaluation -- measured creeping 6.2e-08 -> 3.4e-08 over twelve evaluations and stopping.
  Bisect the wider side of the triple instead; the failed fit is itself the signal that there
  is more than one minimum in there.
* **No retiring a region by overlap with an emitted bracket.** It retires too much: a region
  holding two eigenvalues, bracketed around one, is retired whole and the second is never
  bracketed -- coverage 0.75 on every pair. Containment of the *region* retires nothing, since
  a bracket is narrower than the region it came from. Retire on the region's remainder.
* **No collapsing the fast and slow paths.** Sending every region to the locator instead of
  gating it looks like a simplification and loses purity everywhere: the straddle samples at a
  predicted vertex, so on a region holding two wells it piles samples onto one and never lands
  between them. Coverage 1.00 with purity failing at every precision.
* **No treating slope growth as an error.** Exclusions are recomputed from the current `L` every
  pass, so growth is COST. A limit of 4 growths declined a solvable case: on the multiplicity-3
  cluster the seed settles at 0.31 against a true wall slope of 0.683 and needs 7 growths to
  reach the right answer.
* **No dropping a region because its samples are FLAT.** Flatness at the floor is the opposite
  of flatness on the plateau. A region lying entirely inside a well has every sample on the
  floor and an internal ratio of ~1, and the confirmation step dropped it -- losing an
  eigenvalue it had correctly found, which is the worst trade available. The level test has to
  be a fraction of the measured plateau.
* **No budgeting a loop on cache misses.** A refiner re-reading points it has already seen adds
  none, so the budget never advances and the loop spins forever at zero cost.

---

## 10. Parked: precision below 1e-10

Parked by the 2026-10-09 direction (section 0): nothing here is needed for `douse`'s first
integration at `prec = 1e-10`. Each item keeps its evidence so it can be picked up without a
rediscovery session. Revisit when a benchmark case at a tighter `prec` needs it.

* **The posterior penalty: READY at 1e-12, not robust at 1e-14**
  (`notes/posterior_penalty.md`, 2026-10-05). A fixed `tau = C eps sqrt(n)` read through posterior
  (Rayleigh-Ritz) tensions on the 8 lowest penalized vectors. After four search defects it exposed
  were fixed (`463ef6d`, `e469740`), every `C` in [8, 100] is clean at 1e-12 on three basis sizes
  (n = 128-258) and the hard roster, matching the threshold; `C = 3` is the noise jaw and declines
  honestly. At 1e-14 no `C` is clean at every size and losses are SILENT: the penalty bends the
  span off the eigenspace (large `C`) or noise wins (small `C`). Next:
  1. A posterior mode in `src/` at `C = 30`, and `reg="auto"` extended from `prec >= 1e-8` to
     `prec >= 1e-12` -- retiring truncation, and the Linux rank flip, everywhere but 1e-14. Needs:
     the cost (2-2.7x wall time per evaluation), and the reach warning's message, which still says
     no penalty serves below ~150 eps sqrt(n).
  2. The 1e-14 span: a larger span than `MAX_BRANCHES`; posterior for refine/count only; a
     second, smaller-`tau` solve in a window around each root.

* **At `prec = 1e-13` the L-shape's exact double `10 pi^2` comes back x0, x3 or x4,
  unflagged, on its ADEQUATE basis** (80-term recipe; tension 1.5e-13 there). x3 and x4 on [5, 100]
  and [5, 110]: separate simple levels 1e-13..5e-12 apart, roots located inside the floor-flat
  region -- the refiner's phantom class at the floor. x0 on [90, 105] and [95, 100]: not traced.
  At `prec = 1e-12` this double is now counted 2 in every window (the (C2) fixes below).
  Repro: `notes/double_10pi2.py`.

* **`delta_res` still under-covers by up to 2.6x on met levels near the float floor.**
  `notes/bound_coverage.md`: after `c1c3f53` (a `delta_res` short of `prec` is no longer admitted
  into `Level.bound`) the bound covers every unmet level of a 715-level grid, and misses 10 met
  levels by at most 2.6x, all with errors of 8-3100 ulps and within `prec` -- the sector apex
  family and the L-shape at 1e-13. `DELTA_SAFETY = 2` was calibrated on a located-error ratio of
  0.99-1.27; these reach 5.2. Raising it trades against tightness at 1e-13, unmeasured. Neumann and
  mixed levels have no Moler-Payne bound and still report bare `delta_res`.

* **`refine`'s phantom test assumes a cone, and a flat-bottomed well defeats it.** On
  the 50*pi^2 triple at `prec = 1e-6`, band 1.3, under the order-2 penalty (`tau_2 = 2 prec`), a
  bracket lying BETWEEN two members converged 2 ulps inside its lower end, on the wall of the
  member below, and was accepted: 4 of 3. At an end, `refine` probes outward by
  `sigma / c_wall`, where `c_wall` is the steepest secant INSIDE the bracket -- here the steep
  hump side -- so the reach undershoots the gentle descent toward the neighbour, and all three
  probes read 0.54-0.94 of the point against the 0.5 needed to call it a phantom. Only a penalized
  objective with a floor near the licence produces such a well; the threshold never has. The
  phantom logic carries five calibrated cases and a documented overshoot trap, so this is a
  re-derivation, not a constant. Wanted before any penalty whose floor approaches `prec` is
  built into `src/`. The shipped default does not: its floor is about 1.4 `C_PEN` prec = 0.03
  prec, a thirtieth of the licence. `notes/smooth_filter.md` section 10.3.

  (The same cell ALSO reported one root twice, 1e-12 apart, through a separate defect: the
  cache wall slope `_slope_from_cache` ignored the floor and read 8.45e6 against a true 0.02-0.08,
  collapsing the side search's exclusion radius to 1e-13. Fixed 2026-09-22 with a GUARD: a
  floor-aware estimate is used only where the blind one exceeds it by more than 100x, so the
  calibrated slope is unchanged bit for bit everywhere else -- applied unconditionally it moved a
  hump root 2.3x and flipped four calibrated tests. The cell went from 5 of 3 to 4 of 3.)

* **One attribution case the evidence cannot decide, FLAGGED by design.** A single
  side-search find admissible to both a counted and an undecided branch: if it is the counted
  branch's member the undecided one's is still out there, and if not, the counted one's lies inside
  the exclusion where nothing looked. 50*pi^2, 1e-10, band 0.3, `tau = 0.02 prec` reads 2 of 3,
  flagged (`notes/smooth_filter.md` 10.7). Deciding it needs a search inside the exclusion, the
  same move section 12's inner-end retry makes; not tried here.

* Promote Phase D's angle, or retire it. It is BUILT --
  `solve(..., phase_d="angle")`, `search._gram_group`, `multiplicity.TAU_GRAM` -- and off by
  default because it **changes nothing measurable**. Over five cluster grids and the roster it is
  consulted 115 times and agrees with the distance rule on every one: no regression, no
  improvement (`notes/landscape_findings.md` sections 36, 36.1). The prediction that it would be
  inert was wrong; it is exercised, it just never disagrees.

  What would settle it is a case where the two rules differ, and none of the committed grids
  holds one. The roster asks the angle NOTHING at `prec = 1e-8` -- its levels are further apart
  than a licence -- so the grouping regime lives only in the split-degeneracy grids. Either find
  or construct a domain where a duplicate and a genuine pair occur at the same separation, or
  accept that the rule is insurance against a case this roster does not contain and say so.

  One correction it already bought, section 36.2: **the ambiguous eigenspace is NOT benign under
  the reach rule.** The spec's "parallel recounts to 2, orthogonal sums 1+1 to 2" assumes each
  member read 1; a degenerate level located twice has both reading the full multiplicity, and the
  sum doubles it. Flagged rather than guessed, since capping by a spanning recount is measured
  unsafe elsewhere.

* Repair the swing detector or retire it. Its `5 x median` peak rule tracks the
  noise it is meant to reject (the median rises with the stride, so refining *degrades* the
  detector: 3/3 genuine gaps at 160 samples per gap, 2/3 at 2560), and its cell-cutting is a
  uniform scan wearing an adaptive label. `notes/landscape_findings.md` sections 30.3-30.4
  gives three untested repairs.

* Use coarse and fine bases separately to reduce the cost of the search phase. 
  See `notes/coarse_to_fine_basis.md`.

* **The second-order penalty** (`notes/tikhonov_derivations.tex`), the only form measured to get
  the slope loop through at 1e-14 (`notes/smooth_filter.md` section 10). Parked until bases are
  right-sized (`notes/coarse_to_fine_basis.md`).

* **The penalty's reach constants are one family's.** `_REACH_FLOOR`, `_REACH_ROUNDING`, `C_PEN` and
  `PENALTY_PREC_MIN` in `src/chladni/mps/_regularize.py` were measured on near-square rectangles
  with plane waves. The lower jaw should scale with `eps * sqrt(n)`; unchecked across basis sizes.
