# Accuracy-reporting audit, 2026-11-03

**Question.** Are there places where tolerances, accuracy estimates or accuracy claims are
mishandled or misreported, and where fixing the reporting, or calibrating to the accuracy
actually achieved, would buy performance?

**How it was done.** Four read-only reviews ran on branch `boundary-integrals-accuracy`, one per
area:
1. search, bracket, refine and multiplicity;
2. pencil, certification, solver and planner;
3. kernels, geometry and reference;
4. the test suite.

The integrals rework on this branch was already audited in `hadamard_accuracy.md` and was not
reviewed again.

**Status labels.**
- **checked** means the claim was confirmed by reading the code or running it in the main
  session, not only by the reviewer.
- **reviewer** means it rests on the reviewer's reading alone.
- **est.** means the saving has not been measured.

## 1. Claims that can be wrong (fix first)

| # | where | defect | status |
|---|---|---|---|
| 1.1 | `mps/search.py:271-283` `Level.bound` | When the Moler-Payne bound is refused (`mp_bound = inf`: eps ≥ 1, or the norm clipped), `delta_res_admitted` is False but the `isfinite` guard skips the MP branch, so `bound` falls back to `delta_res` and the level reads **met**. This is the worst MP outcome, reported as success. `bound_reason` is set and nothing reads it. | checked |
| 1.2 | `mps/solver.py:390-430` `track_levels` | The merged-window branch keeps only `levels` and `evaluations` from each `search_interval`. `n_uncertain`, `rejected_deep` and `n_ambiguous` are dropped, and `eigenproblem._record` flags only `d.n_uncertain`. **The warm path, which is douse's, accepts uncertain counts and possible lost eigenvalues silently.** | checked |
| 1.3 | `eigenproblem.py:312` | `raise PrecisionShortfall(...)` in `_solve_cold` has no import in scope, so `on_shortfall="raise"` on a geometry-declined shape raises `NameError`. douse catching `PrecisionShortfall` would crash. | checked |
| 1.4 | `search.py:218-227` `Level.margin` | Messages ("margin 4; 2 is needed") and `_rank`'s rung tie-break use a `delta_res`-only margin, while `met` reads `bound`, which may be MP. A level that failed the MP cross-check reports margin ≥ 2, and the planner ranks that rung as nearly met. | checked |
| 1.5 | `solver.py` `certify_levels`, `eigenfunctions` | Neither reads the integral rule's `sizing_prec`/`shortfalls`: the same defect as `hadamard()` before this branch. A demoted corner (sizing inf) feeds the MP norm and the Löwdin step unreported. | checked |
| 1.6 | `bracket.py:380`, `search.py:845` | Purity uses `prec*max(hi, 1.0)` in k, and the merge licence uses `prec*max(lam, 1.0)`. Both are absolute floors, so on large domains (lam < 1, k < 1) brackets and merges loosen to prec/k and prec/lam. The result is not scale-invariant. | checked |
| 1.7 | `refine.py:452` | A `budget` exit doesn't widen `delta_res` to the triple's width; only `stalled` does. The bar can be optimistic on a pathological refinement. | checked |
| 1.8 | `multiplicity.py` `reach_multiplicity` | If every branch up to `MAX_BRANCHES` (8), or up to the retained rank, is counted, mu is reported as certain. That count is really a lower bound. | reviewer |
| 1.9 | `solver.py:618-633` `HadamardReport.attained` | `max(rule, gram_residual)` leaves out the eigenfunction's own error (about `mp_eps`), so `report.met` can claim better than u_n allows. | reasoning, not measured |
| 1.10 | `helmholtz/_bessel.py` / `_core.py:226` | Evaluating a Fourier-Bessel family at its own vertex (r = 0) crashes: `rel_err` is inf or nan, and `int(log10(inf))` raises `OverflowError`. Any plot or eigenfunction grid that includes a vertex fails. | checked |
| 1.11 | `eigensolver.py:114` | `count_uncertain = ... or grouping_ambiguous`, although `grouping_ambiguous` is documented as "the count is trustworthy". Trustworthy counts are reported as uncertain, which risks needless replans in douse. | reviewer |
| 1.12 | `mps/solver.py:255` `findability` | The cache key leaves out the probe count `n`, so a different `n` returns a stale result. | reviewer |

## 2. Honest calibration that should buy speed

| # | where | what | status |
|---|---|---|---|
| 2.1 | `eigenproblem.py:334` | The cold window starts at `0.5 * faber_krahn`. The count audit needs only `lo < FK`, and the warm path already uses `0.98 FK`. Every cold solve, including each rung of the planner's climb, scans k in [0.71, 0.99]·k_FK, which Faber-Krahn proves empty. **One-line change.** | checked; saving est. |
| 2.2 | `solver.py:524-537` | The certification rule is sized at `integral_prec(prec)` (0.1·prec), but the MP bound needs the norm only to a few digits: a norm error δ moves the bound by δ relative. At 1e-10 that means 1e-11 nodes where about 1e-3 would do. Verify on `bound_coverage.py`. | checked; saving est. |
| 2.3 | `solver.py:397, 429` | `track_levels` certifies merged-window levels twice: once inside `search_interval`, then again over all levels. That is one redundant Moler-Payne pass per cluster window on every warm step, and douse is all warm steps. | checked |
| 2.4 | `refine.py` loop | The refiner narrows toward `prec*lam` even after the floor (`delta_res`) has exceeded it, running to `stalled`/`budget` (≤ 40 evaluations) on every unmet level in flag mode. The stop planned in `search.md` §2.6, at `max(prec*lam, delta_res)`, was never implemented. | reviewer; saving est. |
| 2.5 | `geometry/segments.py:238,532,725` | The adaptive polyline tolerance is `prec*length`, so the point count grows like prec^-½: a disk has 1M points at 1e-12 and 16.8M (3 GB) at 1e-14. None of the consumers needs prec-level chords. Cap it at about 1e-10·length. | reviewer (measured there) |
| 2.6 | `helmholtz/_bessel.py:56` `_LOG_MAXTERM` | The series branch is chosen while its largest term is ≤ 1e3 *absolute*. In the oscillatory regime \|J\| ≪ 1, so it loses 3–4 digits where scipy is exact (checked: mu 15.8, x 21.52 gives 5.3e-11 relative). The bound reports it honestly, but the loss is avoidable and matters at prec ≤ 1e-10. | checked |

## 3. Warning noise that hides real warnings

- **3.1 Wrong scale for the warning.** The Fourier-Bessel `PrecisionWarning` is an *entry-wise
  relative* bound, so it fires beside every zero of J or of sin(ν θ). On the L-shape it fires
  at 18 of 73 k values at prec 1e-8 and at 72 of 73 at 1e-12. The worst case was an
  absolute error of 8e-15 reported as 1e-6. As a result, 12 test files ignore it, among them
  the end-to-end suites that assert `achieved` and `met`. Nothing downstream consumes
  `Sampling.accuracy`.
  - Fix: report a column-scaled bound (`rel_err*|S|*scale`). This is already in TODO.md, and
    the new data raises its priority.
  - Two smaller fixes go with it: correct the message's "not the point set" wording, and put
    the corner and order in the text so that one corner's warning doesn't deduplicate
    another's.
- **3.2 The family's prec is never aligned with the solve's.** Each family checks against its own
  build `prec` (`plan.py:172`). Nothing in `mps/` calls `with_prec`, so a 1e-14 solve on a
  1e-8 basis never warns, and a 1e-12 basis in a 1e-8 solve warns spuriously.
- **3.3 The scale-error floor (3–4e-14) is counted as entry error.** It makes the warning fire
  on every evaluation at prec ≤ 2e-14.
- **3.4 `pencil.py:73` `_UNSAFE_RATIO = 1e-13` sits above the default rtol for prec below
  4.5e-11.** `rank_diagnostics` then warns spuriously.

## 4. Tests that would let a regression through

| # | test | defect |
|---|---|---|
| 4.1 | `level_score.match_tolerance` → `test_solve.py:164` | The hard-roster pipeline at prec 1e-8 matches within `max(prec, 0.1·min gap)`, which is about 1e-2 for the disk and the L-shape. Eigenvalues wrong in the 3rd digit pass. **(checked)** |
| 4.2 | `test_hadamard.py:151` | The split pair on rect(1, 1.001) is exact (dλ/dL = −2m²π²/L³), yet it is checked at rtol 5e-3. Five digits of cluster-Hadamard regression would pass. **(checked)** |
| 4.3 | `test_tension.py:149,158,167` | Tension at the true λ is checked as `< 1e-6`, while 1e-12 to 1e-16 is achieved. **(checked)** |
| 4.4 | `test_solve.py:67` "reports the accuracy it ACHIEVED" | It asserts only `attained_prec >= 0` or NaN; the known cone roots are never compared. **(checked)** |
| 4.5 | `test_orthonormal.py:104` | `gram_residual` is never compared with the interior-Gauss Gram, the oracle sitting in the same file. It is the number `hadamard()` refuses on. |
| 4.6 | `test_scaling.py:229`, `test_bessel_kernel.py:160` | Docstrings say "upper bound", but the assertions allow 20× and 10× under-reporting. |
| 4.7 | `test_level_resolution.py:109`, `test_on_shortfall.py:96` | A bare `pytest.warns(PrecisionWarning)` is satisfied by the FB noise warning, so the solve-level shortfall warning could be deleted unnoticed. Add `match="short of the requested"`. **(checked)** |
| 4.8 | `test_stage_policy.py:340` | The reference is chladni rung 2, agreeing with lappy only to about 1e-10, yet it is tested at a 1e-10 bar. |
| 4.9 | `test_integral_rules.py:414` (mine) | `ach <= 1e-14` has under 2× margin over a platform-dependent floor, the same flake as the node cap. |
| 4.10 | xfails | `test_solve_interval.py:172,262` are `strict=False` and slow-only, so a fix to STARVED_SILENT would land unnoticed. |

Also, a few blanket `simplefilter("ignore")` calls hide RuntimeWarnings in the precision-ladder
tests. `test_moler_payne.py:223` can pass while comparing nothing, because it never asserts
`n_checked >= 1`.

## Areas the reviewers found clean

- The MP bound derivation (`certify.py`), `norm_lower`, `_regularize.py`, `bounds.py`, and the
  planner's corner sizing.
- refine's `delta_res` convention, and lam-vs-k handling in bracketing.
- The policy constants, plane waves, the Y kernel, and `check_prec`.
- `reference/tables.py`. The closed forms there claim 15.6 digits against a measured 15.5.
