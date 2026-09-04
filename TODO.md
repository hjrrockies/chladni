# Outstanding work

Three rules this file lives by:

* **An item is deleted when it is done, not annotated.** The file is kept true at HEAD, the
  same way `docs/conventions.md` is. What was outstanding on a past date is what `git log` is
  for.
* **Every item names the evidence that forces it** -- a measurement, a design document, or a
  line of code. An item with no source is a preference, and belongs in a design record where
  it can be argued with.
* **Citations into `notes/` are deliberate**, and `notes/` is not tracked. The sharpest open
  items were measured there and exist nowhere else; dropping them to keep this file
  self-contained would make it shorter and less useful.

Priorities are `[blocking]` (something else waits on it), `[next]`, `[later]`.

---

## 1. The eigenvalue search

The layer everything else waits on. `MPSEigensolver.solve_interval` raises
(`src/chladni/mps/solver.py:169`), and the candidates live in `tests/` on purpose -- when one
wins it moves into `src/chladni/mps/`.

All four stages now exist as scored candidates and the composed pipeline passes 15 of 15 cases
it can decide -- disk, equilateral triangle, L-shape, exact degeneracies of multiplicity 2, 3
and 4, and both sides of the crossover on two windows -- at 15 to 33 evaluations per eigenvalue
including refinement. What remains is promotion, not construction.

* **[blocking]** Promote the v1 candidate and implement `solve_interval`. The comparison is
  settled: `tests/v1_bracket.py`, `tests/v1_refine.py`, `tests/v1_multiplicity.py` and
  `tests/v1_solve.py` pass the standing gate -- 111 cases, 1
  unexercisable, at 26-27 evaluations per eigenvalue on real rectangle pencils including the
  refinement each bracket hands on, and 22-24 on the disk, the equilateral triangle and the
  L-shape. The older generation stops splitting at `d/lam` between 1.2e-04 and 3.6e-05
  (`notes/landscape_findings.md` section 29). What moves is `v1_bracket` and `v1_refine` into
  `src/chladni/mps/`, rewritten against a spec that `CHLADNI_V1.md`'s As-built section will
  have corrected.
* **[next]** Phase D's Gram angle, for the case distance cannot settle. `tests/v1_solve.py`
  implements Phase D's DISTANCE PREFILTER and gets the collapse branch for free -- reading
  multiplicity once per merged group RECOUNTS by construction, which `search.md` says is the
  right move and warns against replacing with a sum. What remains is the case the angle is
  really for: two roots at the SAME separation where one pair is a duplicate and the other is
  not. Needs the `modes(x)` widening below. Nothing in the roster exercises it, which is a
  measurement on one roster and not an argument.
* **[next]** Multiplicity is UNDECIDABLE for a cluster whose diameter sits between the merge
  licence and the wide bracket's reach, `1 < diameter/(prec*lam) < WIDE_FACTOR`. Measured on the
  multiplicity-3 cluster: pass at ratios 0.40, 2.60 and 6.00, fail at 1.00 and 1.60. Such a
  cluster is too wide to be one level of `S(prec)` and its members sit closer together than the
  flanks a read needs, so no flank placement is both outside the level and inside the gap to the
  next one. `rect_suite.multiplicity_undecided` reports it and the tests skip those cells.
  Whether a smaller `WIDE_FACTOR` narrows the band without breaking the prominence bound
  (`sqrt(1+f**2) >= P` needs `f > sqrt(3)`) is one experiment.
* **[next]** The open `floor_frac` bug. `lappy_repaired_search` misses a *different* L-shape
  eigenvalue depending on `floor_frac` over `k` in `[3, 5]`: 1e-6 loses 15.197, 1e-9 loses
  9.640, 1e-12 finds all three. "Load-bearing in a way nothing yet explains" -- `search.md`,
  As built. It is the only known correctness failure in a search that otherwise works, and it
  was found by the Weyl audit rather than by prefix truth.
* **[next]** Widen the candidate interface from `tensions(x)` to also carry `modes(x)` -- the
  candidate eigenfunctions on a fixed shared point set. Phase D's collapse-versus-merge is
  decided by the interior Gram angle, which `tensions(x)` cannot supply, so everything about
  multiplicity is un-benchmarkable until this lands.
  `notes/solver_features.md` section 4 argues the widening over post-processing; section 5
  ranks it second.
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

## 2. Generalization and validation debt

Every number in `CHLADNI_V1.md` was measured on one family, the `(1+delta) x 1` rectangle, and
the document says so in its own summary. `tests/hard_roster.py` is the second domain that
section 10 item 1 asked for -- the disk and equilateral triangle for provable coverage, the
L-shape for a re-entrant corner -- and it settled the cost question: 22-24 evaluations per
eigenvalue, against 26-27 on rectangles. What it has not settled is below.

* **[next]** Re-measure the three constants that are still inherited: `straddle = 0.02`,
  `margin = 1.5`, `safety = 1.5`. Each carries a `RECALIBRATE` comment in `tests/v1_bracket.py`
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

* **[next]** `MPSEigensolver`'s `rng=None` default should become a fixed constant. Unseeded, an
  interior fill gives a 110x run-to-run spread in the located eigenvalue, and that goes into a
  shape objective as noise. The docstring at `src/chladni/mps/solver.py:44` already says the
  argument "is not optional in spirit"; the default has not followed.
  `search.md` As built; `notes/landscape_findings.md` section 17.
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
* **[later]** The end-to-end reference values in `docs/design/particular_solutions.md` --
  sector eigenfunctions, square, disk, equilateral triangle, and the L-shape's
  `lam_1 = 9.6397238440219410527` -- are written down but untested, because there is no
  eigensolver to test them with. They are the `mps` layer's first tests. Includes the L-shape
  Fourier-Bessel enrichment regression: 1e-12 to 1e-13 with the corner terms against ~1e-5
  without.

## 5. Test-suite hygiene

* **[next]** Commit the untracked instrument. It is now substantially complete and green --
  1380 default tests in 10 s, 52 slow, 111 gate in 4:48 -- and untracked, CI runs none of it.
* **[next]** `tests/bracketers.py` is orphaned. No `test_*.py` imports it, and it is the only
  importer of `tests/searches.py`. Wire both in or delete both -- and note that
  `tests/baselines.py` now supplies the honest-baseline role on the current seam, so the case
  for keeping `bracketers.py` is only that it holds the older generation for comparison.
* **[next]** Delete `lipschitz_search`. `notes/handoff_minimization.md` section 1 gave it two
  fates -- rebuild against `search.md` sections 4-5, or delete, explicitly and without
  patching. **The rebuild happened**: `tests/v1_bracket.py` is that structure done properly,
  and it passes the gate where `lipschitz_search` scored 7/12. Nothing now depends on the old
  one.
* **[later]** No direct `test_searches.py` covering the older searches; they are exercised
  only through `bracketers.py`, which is itself orphaned.

## 6. Small deferred measurements

Each is one experiment. None blocks anything.

* Can a coefficient vector be reused across point sets to build Cauchy data, or does that need
  a GSVD recompute on the new pencil? `docs/design/mps.md:172`. Belongs with the eigenfunction
  work.
* Does the `lam`-over-`k` refinement advantage (17 of 17 cases, median 5.1x) survive at
  `delta_res`-scale brackets, and does it grow at large `lam`? `search.md`, As built.
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
* **[next]** `tests/objective.py:82` is deliberately not thread-safe -- a lock there would make
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
