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

What is left here is not construction. It is the multiplicity contract, the `modes(x)` widening
Phase D needs, and the constants still inherited from one rectangle family.

**Phase D still goes before basis planning in section 3**, though for a narrower reason than it
did. Since Phase 3b promoted attained merge and the reach rule, a cluster's total is certified or
flagged on every grid measured, with the known exceptions below; what Phase D alone can settle is
two roots at the same separation where one is a duplicate and the other is not. For `douse` the
ordering still holds: a shape objective goes non-smooth exactly where levels collide.

Do the `modes(x)` widening as its OWN step before the angle. It is a seam change across
`Objective`, the pencil and the candidate interface; the angle is a rule. Landing them together
is two kinds of work in one session, which is what made the previous promotion attempt
illegible.

* **[next]** The side search's residual can start ON its find's wall, and the refiner then
  returns that endpoint. This is the one cause behind every wrong total the cluster grids still
  carry, and it is a REFINER interaction rather than a counting one. When the side search finds a
  member and searches on past it, the residual begins one exclusion radius out; that radius is
  `(sigma/c)/bias_lo`, which is right for the dedup it was borrowed from and wrong here, because
  at a CONVERGED find `sigma` is the tension floor rather than a distance to anything. A standoff
  of a fraction of the remaining span was added and fixes one of the two measured cells
  (`search._SIDE_STANDOFF`, flat over 0.05 to 0.2, so the value is not tuned). What it does not
  fix is the case where the refiner lands on the residual's inner end anyway:

  - `170*pi^2`, band 0.3, prec 1e-12, 7 of 8 (strict xfail in `tests/test_rule_seam.py`). The
    residual `(1677.8327481833226, 1677.832748183896)` provably holds the missing member --
    `sigma_1` dips to 2.18e-13 at 1677.8327481833942 -- and the refiner returns the upper end at
    2.47e-13, `not_a_minimum`. Standing the exclusion off by 2x or more recovers it; 1.0 and 1.5
    do not. That factor is measured on one cell and wants the whole grid before it ships.

  Belongs with the refiner-rejection item below: both are `solve` reading a refiner verdict on a
  sub-interval that is not empty.
* **[later]** `solve` still reads every refiner rejection as an empty bracket. Since the
  `no_parabola` fallback, a rescan of the `170*pi^2` band-0.3 row and three other cluster cells
  found no main-stage bracket holding eigenvalues rejected (42 of 42 accepted), so this is latent,
  not live; a rejection other than `not_a_minimum` or `endpoint` should still not certify a count.
* **[next]** Declines at `145*pi^2`, and they have ONE cause. Re-measured over 5 precisions x 11
  bands: **30 of 55 cells decline, and every one of them carries the bracketer's seed message** --
  "the slope estimate never settled" -- with the largest observed secant rising monotonically
  through all six doublings (e.g. `0.71 -> 0.81 -> 2.20 -> 3.92 -> 7.77 -> 16.24`). Not one is the
  `MAX_GROWTHS` path and not one is the multiplicity layer. The cells that do NOT decline are all
  correct: 25 right, 0 wrong.

  The mechanism is already written down one constant away, in `MAX_GROWTHS`'s own comment: on a
  tight cluster "the settling test passes on the PLATEAU's own variation and the cover phase then
  discovers the walls one doubling at a time". Here it does not even pass -- six doublings of
  `n -> 2n-1` never resolve a cluster whose members sit a fraction of a licence apart -- so the
  bootstrap meant to falsify the Lipschitz premise is instead reporting that a SEED GRID is too
  coarse. Candidates, cheapest first: more doublings (the history shows no plateau in sight, so
  this alone will not do it); a settling test that is not fooled by plateau variation on a narrow
  window; or accepting that a narrow cluster window declines at seed and letting the cover phase's
  `MAX_GROWTHS = 16` do the work it was widened for. Do NOT invent a seventh way to sample the
  cover phase more cleverly: `bracket.py`'s cost law retires that family and six attempts have
  failed.

  Note the "5% the robustness plan set" cited here before: that plan is not in the repository, and
  the "24 of 45 declining" it quoted contradicted `search.solve`'s own promotion table, where 24
  is the number RIGHT and 21 declined under both configurations. State a whole-solve decline
  target with its justification; the estimator-level `decline_rate < 0.10` in
  `tests/test_multiplicity_ladder.py` is a different quantity, since these declines are upstream
  of the estimator entirely.
* **[next]** The old Phase D note, for the case distance cannot settle. `chladni.mps.search`
  implements Phase D's DISTANCE PREFILTER and gets the collapse branch for free -- reading
  multiplicity once per merged group RECOUNTS by construction, which `search.md` says is the
  right move and warns against replacing with a sum. What remains is the case the angle is
  really for: two roots at the SAME separation where one pair is a duplicate and the other is
  not. Needs the `modes(x)` widening below. Nothing in the roster exercises it, which is a
  measurement on one roster and not an argument.
* **[next]** State the multiplicity contract for the promoted pipeline: attained merge, the reach
  rule on the cache slope, the side search that locates what the count declines. The old failing
  band of one to eight licences belonged to the cross-check, which is retired from the defaults
  (`tests/test_cluster_totals.py` records the grid); the contract is a statement about the rule
  that replaced it, including when a count declines and what the side search may add.
* **[blocking]** Widen the candidate interface from `tensions(x)` to also carry `modes(x)` --
  the candidate eigenfunctions on a fixed shared point set. **This is the first thing to do**:
  Phase D's collapse-versus-merge is decided by the interior Gram angle, which `tensions(x)`
  cannot supply, so everything about multiplicity is un-benchmarkable until it lands. A seam
  change, so it wants its own step and its own commit, with the angle following separately.
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

* **[next]** A starved basis DROPS the levels it resolves worst, silently. On the disk over
  `[8, 60]` at `prec = 1e-12`, plane waves at `pad = 0.45` (28 columns) return 5 of 6 levels,
  with no decline, no warning, and nothing in `unmet` for the lost one -- the levels that ARE
  found meet the request. At `pad = 0.35` (22 columns) the previous defaults returned 2 of 6 the
  same way; the promoted defaults leave a count the reach rule cannot certify there, and
  `solve_interval` refuses, which is loud but not a completeness diagnostic. `search.md` section 2.4 clause (3)
  says the completeness diagnostics fire whenever clause (1) fails; here none does, and
  `Eigensolver.solve(n)` answers a short return by widening the window. Pinned as a strict
  xfail, `tests/test_solve_interval.py::test_a_starved_basis_does_not_DROP_levels_silently`.
  The Weyl deficit is the obvious candidate alarm; over a window this narrow it is a
  plus-or-minus-one estimator (`notes/landscape_findings.md` section 33.1), so measure it first.

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
  hot path is routed around it. `regularize`'s svd-mode QR now uses `dgeqrt`
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
