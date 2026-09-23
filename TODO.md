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

## 1. The eigenvalue search

The search SHIPS. `chladni.mps.{bracket,refine,multiplicity,search}` are the four stages and
`MPSEigensolver.solve_interval` returns a `Spectrum`, so a `Domain` goes in and eigenvalues
with multiplicities come out. `tests/candidates_v1.py` holds the two adapters that keep V1 on
`candidates.md`'s scored seam, so a fourth candidate is still measured on the same instrument.

What is left here is not construction, and since 2026-09-18 it is not Phase D either. The
`modes(x)` widening landed, the interior Gram angle is built behind `phase_d=` and measured, and
the multiplicity contract is stated in `docs/design/search.md`'s As-built section. **There are no
known wrong totals and no known lost eigenvalues on any measured grid, ON macOS/ACCELERATE AT ANY
SCOPE** -- including the full suite, measured 2026-09-19. Linux/OpenBLAS has one, and only in the
full suite; the two platforms also score different case sets. See the two items below. What is
left is the
constants still inherited from one rectangle family, the rank policy that belongs to the planner,
and deciding whether Phase D's angle earns its keep.

* **[blocking]** **CI is red on Linux, and one failure is a WRONG TOTAL from a threshold-rank
  flip.** Everything in this project is measured on
  macOS/Accelerate; CI runs `ubuntu-latest` against `scipy-openblas32` (`pyproject.toml`), and
  there `test_the_cluster_total_GRID[N65-1e-06-b2.6]` declares **3 of 4, silently**. Every other
  cell of all five grids agrees. Blocking because the contract's central claim is stated more
  broadly than the evidence supports, and because neither platform is the trusted reference:
  section 4 already tracks an Accelerate fault that corrupts the heap.

  Caught by `test_the_DECLINE_RATE_is_quoted_beside_the_pass_rate`, which fails with "1 cell(s)
  neither right nor declined" -- what it was written to say.

  **The mechanism is OBSERVED: a rank flip at one bracket end** -- the `[next]` item below has the
  chain, and `notes/session_2026-09-19.md` the trace. Three things this item said earlier are
  superseded:

  * "the mechanism is unknown" -- it was traced on a Linux runner 2026-09-21. The four candidates
    measured to negative still stand, and none was it: pinning `gsvd4py` onto OpenBLAS, jittering
    the tension values across eleven orders, moving the rank UNIFORMLY via `rtol` (a different
    thing from one flip at one point), and the pencil cache. Nor is "a different LAPACK moves the
    GSVD at 1e-15, and the grid sits at that edge": the members sit 2.60 / 5.72 / 2.60 licences
    apart.
  * "needs full-suite scope" and "threading is not it" -- both wrong. Across every Linux run,
    `OPENBLAS_NUM_THREADS=1` has never failed (0 of ~39), and at default threading the rate moves
    with scope: 1 of 8 for the module, 3 of 4 for the nine preceding files plus it, 4 of 5 for the
    full suite. Scope moves the rate by moving where stage 1 puts the bracket end, and a flip only
    fails the cell if that end lands where direction 86 happens to drop `sigma_1` hard.
  * the collection-versus-execution experiment it named was run: import-time state is not the
    trigger, 12 of 12 right with the whole suite collected.

  The rest of the CI board, for scale: 8 failures. Five are pre-existing and all exercise
  `_collapse_ambiguous`, the cross-check now off by default (`test_solve.py` x4, `test_floor.py`
  x1) -- they may be measuring a configuration nothing ships, and they pass on macOS. Seven of the
  eight are Linux-only; the eighth was not, and is fixed.

* **[next]** **`basis_sufficiency` decides the DENOMINATOR at a boundary it clears by 6%, and the
  two platforms therefore score different case sets.** macOS calls 10 cells unexercisable where
  Linux calls 7. Two of the four `test_cluster_totals` skips sit on the precondition boundary:
  separation 4.78e-15 against a multiplicity resolution of 5.09e-15, a ratio of **0.94**, and
  2.84e-15 against 4.67e-15. On Linux those are scored instead of skipped.

  So "216 right, 0 wrong, 35 declined" is not over the same 251 cells on both platforms, and any
  cross-platform tally is a comparison of different denominators. Measured 2026-09-19,
  `notes/session_2026-09-19.md`. The precondition is right in kind -- section 8's rule -- and the
  question is whether a decision this close to its own threshold should be reported rather than
  silently taken.

* **[later]** **Pinning OpenBLAS to one thread is 2.5x to 3.2x FASTER on this suite**, measured on
  the same `ubuntu-latest` image at module scope: 221 s and 284 s at `OPENBLAS_NUM_THREADS=1`
  against 712 s at the runner's default two cores. Thousands of factorizations of a 190-column
  pencil, where per-call thread spawn and synchronization dominate what they parallelize. If it
  holds at full-suite scope it takes CI's legs from ~65 minutes toward ~20. NOT measured at
  full-suite scope, and not a claim about Accelerate versus OpenBLAS. `notes/session_2026-09-19.md`.

* **[next]** **A threshold-rank flip at one bracket END makes the refiner refuse a real root.**
  OBSERVED on a Linux runner 2026-09-21, bit-identical over 3 repetitions, and reproducible on
  macOS in a second (`notes/session_2026-09-19.md`, the 09-21 section). This is the mechanism of
  the blocking item above. `notes/landscape_findings.md` section 37 filed the `125*pi^2` rejection
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
  flip (`notes/smooth_filter.md` section 9). On macOS it replays this refusal deterministically
  and fixes it -- with the flip injected `refine` refuses exactly as recorded; under the penalty
  it accepts, 1e-10 from the root -- and solves the cell 4 of 4
  (`tests/test_regularization.py`, the "2026-09-21 CI refusal" section). What is NOT done: the
  acceptance test is this cell on a LINUX runner in the nine-file scope at default threading
  (`test_asymp.py` through `test_cluster_totals.py`), which reproduced the failure 3 of 3 at
  about eight minutes a repetition. macOS cannot discriminate -- the threshold passes there
  too -- so that run is what closes this item and the blocking one above.

  **BLOCKED ON A RUNNER, not on the work: the account's free GitHub Actions minutes are spent
  (2026-09-23).** Nothing else waits on it -- every other claim above was verified locally at
  every tier -- so this is a pause. When minutes are available again the nine-file scope is the
  cheap decisive run, not the full suite.
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
  quantity (`test_cluster_totals.MAX_DECLINE_RATE`, 5.7% over 106 exercisable cells); the "5% the
  robustness plan set" cited here before came from a plan not in this repository, and the "24 of
  45" beside it contradicted `search.solve`'s own table, where 24 is the number RIGHT.
* **[later]** Promote Phase D's angle, or retire it. It is BUILT --
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
* **[next]** Carry the in-loop stride correction past the seed. The bracketing stage now has
  one -- `v1_bracket._seed` sizes from Weyl and then doubles until the largest observed secant
  stops growing, so no a-priori stride is trusted -- but nothing downstream of it adapts, and
  the correction was argued for the whole solve. `notes/landscape_findings.md` section 20.2;
  `notes/solver_features.md` F3.
* **[later]** Two-level rank from one factorization -- the scan wants a low rank, the
  refinement a high one, and one SVD serves both. A measured pure win; `search.md` section 8,
  `notes/solver_features.md` F6.
* **[later]** Repair the swing detector or retire it. Its `5 x median` peak rule tracks the
  noise it is meant to reject (the median rises with the stride, so refining *degrades* the
  detector: 3/3 genuine gaps at 160 samples per gap, 2/3 at 2560), and its cell-cutting is a
  uniform scan wearing an adaptive label. `notes/landscape_findings.md` sections 30.3-30.4
  gives three untested repairs.
* **[later]** We should have a "warm-start" solver for eigenproblems that are perturbations of
  eigenproblems with known solutions: a robust mechanism for taking old estimates and refining
  them. This will have to handle multiplicity correctly, as well as target precision. The
  warm-start solver may require bounds on the new eigs.
* **[later/maybe]** If we cannot make the air-tight search have excellent average-case performance,
  as measured by number of GSVD calls, then we should implement a "fast cold start" solver
  which has excellent average-case performance, with guards that can opt back-in to the slower-
  but-air-tight solver.
* **[later/maybe]** Use coarse and fine bases separately to reduce the cost of the search phase. 
  See `notes/coarse_to_fine_basis.md`.

## 2. Generalization and validation debt

Every number in `CHLADNI_V1.md` was measured on one family, the `(1+delta) x 1` rectangle, and
the document says so in its own summary. `tests/hard_roster.py` is the second domain that
section 10 item 1 asked for -- the disk and equilateral triangle for provable coverage, the
L-shape for a re-entrant corner -- and it settled the cost question: 22-24 evaluations per
eigenvalue, against 26-27 on rectangles. What it has not settled is below.

* **[next]** Re-measure the three constants that are still inherited: `straddle = 0.02`,
  `margin = 1.5`, `safety = 1.5`. Each carries a `RECALIBRATE` comment in `chladni/mps/bracket.py`
  naming what it was measured on. The others are done: `kappa` re-measured at 2.591 +- 0.226
  against the study's 2.384 (`tests/test_calibrate.py`), the wall slope at 0.64-1.41 across
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
* **[later]** Two of the six test files `search.md` section 7 specifies have no equivalent:
  `test_search_multiplicity.py` and `test_search_collapse.py`, both blocked on stage 3 and on
  the `modes(x)` widening. The other four exist under different names -- `test_search.py`
  (scan), `test_refine.py`, `test_bracket_score.py` (score) and the `prec` ladder threaded
  through `test_search.py`, `test_gate.py` and `test_hard_roster.py`, which is the row the spec
  names as the one to write first.

## 3. Unwritten layers

* **[blocking]** **Basis planning.** `MPSEigensolver` requires an explicit `basis=`, so there
  is no black-box path at all. `notes/handoff_minimization.md` section 4: everything above is
  quietly waiting on it, and search difficulty is downstream of basis quality -- `eps` varies
  five orders *within a single L-shape scan*, structurally -- so a search benchmarked against
  hand-built bases may not predict black-box behaviour. Whatever the planner becomes, its
  output should include a contract: the `eps` and dip width a search can tolerate. Deferred in
  `docs/design/mps.md` and `docs/design/particular_solutions.md`;
  `src/chladni/helmholtz/sources.py:271` defers the taper decision to it.

  **Second, after Phase D** -- see section 1's ordering note: a planner judges a basis by the
  solver's results, and those results are currently a lower bound on a cluster.

  It is better equipped than it was, by two measurements made while promoting the search.
  **The target now has a closed form.** `delta_res = DELTA_SAFETY * eps/c`, so
  `delta_res <= prec*lam` means `eps <= prec*lam*c/DELTA_SAFETY` -- a floor to hit, where
  `search.md` section 2.5 previously said no closed form was available. And **it must not
  hill-climb on column count**: on the L-shape, corner Fourier-Bessel buys thirteen orders over
  plane waves alone (3.8e-01 -> 4.3e-14) and then REGRESSES 10x from 40 terms per corner to 80,
  while on the disk plane-wave enrichment from 0.6x to 2.5x `n_useful` moves the floor not at
  all because the rank saturates. More is better until it is not, and where the turn happens is
  a property of the family.
* **[next]** **Boundary integrals, and Rellich orthonormalization.**
  `docs/design/boundary_integrals.md` is a seven-line scope note and the layer is unwritten.
  `Quadrature` already carries the weights for it (`src/chladni/mps/pencil.py:31`). It blocks
  orthonormal eigenfunctions, and later the Hadamard shape derivatives.
* **[next]** `Eigenproblem`, the black-box entry point `CLAUDE.md` names as the ultimate goal.
  It does not exist; `Eigensolver` plus `weyl_window` is the intended spine. Blocked on the
  search and the planner.
* **[later]** Eigenfunction and residual plotting. `src/chladni/plotting/_axes.py:3` already
  names them as "still to come" and `prepare` is the seam.
* **[later]** Corner grading. `grading="corner"` / `_grade_flags` is "a placeholder, not a
  design" (`docs/design/geometry.md:449`): it over-grades right-angled Dirichlet and Neumann
  vertices, and node placement wants the integrand, so a quadrature layer should own the
  choice and reach in through `panels=`.
* **[later]** Interior cubature, deferred until something asks for it
  (`docs/design/geometry.md:260`). Note that `notes/solver_features.md` A6 argues against ever
  making this a project: the interior block wants non-degeneracy, not accuracy, and anything
  that must be *accurate* belongs on the Rellich boundary integrals.
* **[later]** User-facing documentation -- the "fifth kind" in `docs/README.md`. `docs/_build/`
  is already ignored for it.

## 4. Known gaps in what is built

* **[later]** **`refine`'s phantom test assumes a cone, and a flat-bottomed well defeats it.** On
  the 50*pi^2 triple at `prec = 1e-6`, band 1.3, under the order-2 penalty (`tau_2 = 2 prec`), a
  bracket lying BETWEEN two members converged 2 ulps inside its lower end, on the wall of the
  member below, and was accepted: 4 of 3. At an end, `refine` probes outward by
  `sigma / c_wall`, where `c_wall` is the steepest secant INSIDE the bracket -- here the steep
  hump side -- so the reach undershoots the gentle descent toward the neighbour, and all three
  probes read 0.54-0.94 of the point against the 0.5 needed to call it a phantom. Only a penalized
  objective with a floor near the licence produces such a well; the threshold never has. The
  phantom logic carries five calibrated cases and a documented overshoot trap, so this is a
  re-derivation, not a constant. Wanted before any penalty whose floor approaches `prec` is
  built into `src/`. `notes/smooth_filter.md` section 10.3.

  (The same cell ALSO reported one root twice, 1e-12 apart, through a separate defect: the
  cache wall slope `_slope_from_cache` ignored the floor and read 8.45e6 against a true 0.02-0.08,
  collapsing the side search's exclusion radius to 1e-13. Fixed 2026-09-22 with a GUARD: a
  floor-aware estimate is used only where the blind one exceeds it by more than 100x, so the
  calibrated slope is unchanged bit for bit everywhere else -- applied unconditionally it moved a
  hump root 2.3x and flipped four calibrated tests. The cell went from 5 of 3 to 4 of 3.)

* **[next]** **The penalty regularization is the INTENDED DEFAULT, and these are the gates.**
  DECIDED 2026-09-23: `reg="tikhonov_qr"` is where the project means to end up, replacing the
  threshold rank as `MPSPencil`'s default, even though work remains. It landed 2026-09-21 as an
  option (`notes/smooth_filter.md` section 9): no rank chosen, so none flips; verified continuous
  where the threshold flickers, and immune to the 1e-15 perturbation that flips the threshold.
  The reason to want it is that a rank flip has already cost a real answer -- the blocking item
  in section 1 -- and no amount of tuning a threshold removes a decision that rounding makes.

  The gates, in the order they should be cleared:

  0. **PARTIALLY RUN (2026-09-23, section 10.5), and it narrowed the decision.** Cluster-total
     rows at 1e-10, 1e-12, 1e-14 under the shipped `rtol`, under `tau = C*prec`, against the
     threshold. Ready at `prec >= 1e-10` -- 8 of 8, marginally better than the threshold's 7 and
     a decline. NOT ready at 1e-12 (every variant loses members, 3-6 of 12, where the threshold
     is 12 of 12) and catastrophic at 1e-14 under the shipped rule: **0 of 18, silently wrong.**
     Still to run: the decline grid under the same rules, and the 1e-6/1e-8 rows.

  1. **Its reach, which is limited by NOISE.** The 9e-14 "saturation" in `notes/smooth_filter.md`
     sections 2 and 7.4 was measured on the literal form and is LAPACK's internal cut. The QR
     form's floor tracks `tau` down to 2.1e-15, but its plateau noise grows as `tau` shrinks --
     1e-2 at the `tau` a 1e-14 floor needs (section 10). A SECOND-ORDER penalty built from QRs
     (`notes/tikhonov_derivations.tex`) fixes the noise and gets the slope loop through at 1e-14,
     and `tau_2 = C * prec` is a clean rule down to 1e-10 for C in 0.2-0.9 -- but at 1e-14 no C
     is robustly free of wrong totals (section 10.2). Parked, and not built into `src/`, until
     the search's merge defect (below in this section) is fixed and bases are right-sized
     (`notes/coarse_to_fine_basis.md`).
  2. **The `prec` RANGE the default can serve, which is not all of it.** At `prec = 1e-14` the
     derived `rtol` puts the penalty's floor at 2.4e-14, above the precision asked for, so the
     shipped form cannot serve the bottom rung; gate 1 is what would. Either the default is
     `prec`-dependent -- penalty above some rung, threshold below -- or the bottom rung waits for
     the second-order form. Say which, in the docstring, before flipping the switch.
     MANDATORY, not advisory: at 1e-14 the shipped rule is silently wrong on every cell of the
     grid (gate 0), so a default without a `prec` range trades honest declines for wrong totals.
     Measured range (section 10.6, which corrects 10.5): a feasible `tau` EXISTS for
     `prec >= 1e-12` -- the clean band at 1e-12 is `tau` in [0.007, 0.05] * `prec`, 12 of 12,
     matching the threshold -- and NONE exists at 1e-14, where declines and wrong totals meet
     with no gap. At 1e-6 and 1e-8 the penalty BEATS the threshold, 12 of 12 against 11 and a
     decline. Missing for a rule: the band's `prec`-dependence, measured at 1e-12 only, and a
     single C is not yet uniformly clean (`tau = 0.02*prec` loses one cell at 1e-10).
     `notes/smooth_filter.md` sections 10.4-10.5 and `notes/fig_penalty_prec_limit.png` draw it: the
     rule `rtol = sqrt(prec*eps)` was derived for a truncation, whose floor the cutoff barely
     moves, while a penalty's floor IS `tau ~ sqrt(prec)` -- so `floor/prec` grows like
     `1/sqrt(prec)` and crosses 1 between 1e-12 and 1e-14. A `prec`-proportional rule for the
     penalty's `rtol` is the obvious candidate: measured in 10.5, it holds `floor/prec` constant
     and is one to four decades quieter at loose `prec`, so `C ~ 0.1` is the better rule wherever
     the penalty is used -- but it does NOT fix 1e-12, so it is a better rule, not the fix.

     Section 7 of the note is the evidence to argue against: it found the regimes cleanly split,
     frozen rank winning narrow windows and a filter winning wide windows with a tight pair, with
     nothing to choose on a corner-FB basis because the rank does not move there. A default has
     to survive that table, and the 1.5x per-evaluation cost -- which section 7.4 suggests may pay
     for itself in evaluations saved (42k against 82k, 38 spurious brackets against 807), never
     measured as wall time.
  3. **`calibrate_rtol` under a penalty.** It runs -- raising `rtol` raises the floor either way --
     but its model is written for a truncation cut and is unmeasured here.
  4. **A `PrecisionWarning` below its reach**, like `_warn_if_unsafe` for a frozen rank. There is no
     calibrated threshold to warn at yet.
  5. **`"tikhonov"`, the literal form, does not remove the rank decision** -- LAPACK's `?ggsvp3`
     decides the interior block's rank itself (finite count 41-50 over a scan where the QR form
     holds 50). Kept as an oracle; whether it should stay user-facing is open. It is NOT a
     candidate for the default.
  6. **The refiner's phantom test assumes a cone** (below in this section). It only bites where a
     well's floor approaches the merge licence, which the shipped `rtol` does not do -- floor
     1.7e-11 against `prec = 1e-8` -- so it does not gate this default. It WOULD gate any penalty
     tuned to spend more of `prec` on its floor, the second-order form included.

* **[later]** A starved basis DROPS the levels it resolves worst, and for the levels that are
  never BRACKETED no diagnostic added to `solve` can catch it -- `notes/landscape_findings.md`
  section 38. RECLASSIFIED to basis planning (section 3) rather than an honesty gap in the search.

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
  `tests/test_regularization.py:117` ("accuracy at those lam is not the claim, and is tested in
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
  `tests/rect_suite.py:113` sets the suite's threshold rank to `sqrt(FINEST_PREC*eps)` = 1.5e-15,
  so every threshold-rank pencil in the suite cuts two decades BELOW the level the frozen-rank
  guard calls unsafe -- deliberately, and measured to work (0 spurious minima in `[5, 35]` on
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

* **[later]** No direct `test_searches.py` covering the older public searches. `test_searches.py`
  reaches only three private helpers (`_spacing`, `_descend`, `_local_minima`); `lappy_search`,
  `lappy_repaired_search` and `swing_direct_search` are exercised only through
  `tests/bracketers.py`, which is `notes/`-runner tooling rather than a test module. That is now
  stated in `bracketers.py`'s own docstring, so the arrangement is deliberate rather than
  undiscovered -- but the searches themselves are still uncovered by anything the suite runs.

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

* **[next]** Decide whether the candidate instrument scores wall-clock rounds as well as
  evaluations, and decide it **before** the search winner is picked. `candidates.md` sections
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
