# Where a k=2 run's solves go (2026-10-14)

`douse/benchmarks/diagnostics/warm_trace.py`: `maximize_ratio` on chladni, N = 6, k = 2
(lam_3/lam_1), seeds 0-2, constrained, 80 iterations, with the backend's per-solve trace
(`douse.backend.SolveTrace`). Timings are inflated by a concurrent slow test run; the counts
and ratios are the point. Rows: `douse/benchmarks/run/diagnostics/warm_trace.jsonl`.

| seed | wall | solves (plan / cold / replan / warm) | seconds (plan / cold / replan / warm) | evals per warm | evals per cold |
|---|---|---|---|---|---|
| 0 | 1054 s | 1 / 13 / 2 / 68 | 6 / 294 / 33 / 691 | 92 | 181 |
| 1 | 493 s | 3 / 6 / 3 / 25 | 14 / 149 / 89 / 225 | 79 | 182 |
| 2 | 1568 s | 1 / 15 / 5 / 66 | 5 / 365 / 140 / 1029 | 93 | 188 |

Why a warm solve went cold (over the three seeds): `cadence` (the cold solve every 10 warm
ones) 11; `unmet` (a level missed prec, which douse then re-plans) 9; `moved` (a tracked value
moved more than its allowed h, i.e. a large early step) 8; `edge` 1.

## What it says

1. **The hypothesis was half right.** Every replan is an `unmet` (9 of 9), as predicted -- but
   replans are 3-9% of a run, and the planned-record reuse (`Eigenproblem.plan_and_solve`)
   already removed the second solve each one cost. They are not the gap to lappy.
2. **The warm solve is the gap.** 65-70% of a run is warm solves, at ~90 evaluations each --
   half a cold solve's 180. The warm window is a PREFIX, `[min(0.98 FK, lam_1 - h_1),
   lam_n + h_n]`, chosen so the count audit applies; so every warm solve re-seeds and re-covers
   the whole range below lam_n, not a neighbourhood of the previous values. lappy's `track_set`
   scans narrow windows around the tracked values, which is where its 4-9x comes from.
3. **The cadence backstop is a third of the cold solves** (11 of 30), each ~13 s more than a
   warm one: about 5% of a run. It is a guard against drift past the continuity checks, decided
   with the user; lengthening it to 20 is a measurable 2-3%.
4. **Per-evaluation cost** (0.17-0.33 s at 392-561 columns, 93% of assembly in `log_j`) is the
   multiplier on all of the above.

## Next levers, in order (Session 4 of the follow-up plan, revised)

**Decided with the user, 2026-10-14: the windows are set by the OPTIMIZER'S STEP, not by a
prior.** douse knows how far each tracked level will move before it steps -- `g_i . d` from the
Hadamard Jacobian (or the cluster block) and the trust-region step -- so it hands the warm solve
per-level half-widths `h_i = max(c |g_i . d|, resolution_i)`, `c` in 2-3 for the second-order
margin, and caps `delta` so no window exceeds half the gap to its neighbours. Completeness is
then first-order exact (no tracked level can jump its window), the headroom level `k + 2` is the
tracked ceiling an intruder must cross, and the cadence cold solve stays as the backstop. Item 1
below is the solver's half of that; the optimizer's half is in douse's plan.

1. **Per-level warm windows.** In `Eigenproblem._solve_warm`, refine each previous value in
   `[lam_i - h_i, lam_i + h_i]` directly (`mps.refine` on a bracket, then the multiplicity read)
   instead of searching the prefix: ~15-20 evaluations per level against ~90 per solve.
   Completeness then rests on continuity and the cadence cold solve: an eigenvalue entering
   from above must cross lam_n's window (seen as an extra value there), and an interior crossing
   moves a tracked value by more than h. The `moved`/`edge`/`unmet` verdicts stay; the count
   audit does not apply to non-prefix windows and is dropped on warm solves (it already is).
   Measure on the same three seeds; the certified values must agree to 1e-9.
2. Cadence 10 -> 20, measured on the same seeds.
3. The Fourier-Bessel pass: evaluate `log_j` once per distinct `(nu, k r)`.
4. The shared objective cache across the warm/cold pair (now that cold follows warm less often,
   a smaller win than the plan assumed).

## After step-controlled windows (same seeds, same evening)

chladni `fefad6a` (`track_levels`, `solve(windows=)`), douse `691c649` (`predicted_moves`,
`_warm_windows`, the step cap at half the gap to a neighbour, `window_factor = 2.5`). Rows:
`douse/benchmarks/run/diagnostics/warm_trace.jsonl` (before: `warm_trace_before_windows.jsonl`).

| seed | wall before -> after | warm solves | evals per warm before -> after | s per warm before -> after |
|---|---|---|---|---|
| 0 | 1054 -> 825 s | 68 -> 70 | 92 -> 61 | 10.2 -> 6.8 |
| 1 | 493 -> 837 s (*) | 25 -> 73 | 79 -> 67 | 9.0 -> 7.4 |
| 2 | 1568 -> 761 s | 66 -> 72 | 93 -> 61 | 15.6 -> 6.8 |

(*) The before-run of seed 1 took only 37 solves in 80 iterations (many infeasible or refused
trials); its wall is not comparable. All six runs hit the 80-iteration cap, so `fun` is not
comparable either (the step cap changes the path): converged runs are needed for the
1e-9 agreement check.

**What moved.** Evaluations per warm solve fell by a third, to ~61 for four tracked levels --
which is the bracket refiner's own cost, ~15 per level. That is the floor of this design: the
prefix search is gone, and what remains is one refine per level plus certification.

**Why warm solves still went cold** (28 over the three seeds): the cadence backstop 13; `unmet`
4 (each a replan); `track: refine: no well` 10, all on WIDE windows (half-widths of 1-10 on
values of 18-87, i.e. early large steps where the predicted move is large): the refiner is
the wrong tool for a window holding several wells, and those went cold at the cost of a wasted
refine; `found` 1.

**Next, in order.**
1. In `track_levels`, SEARCH a window instead of refining it when its width exceeds a fraction
   of the local mean Weyl spacing (the window may hold more than one well), so the wide early
   windows cost a small search rather than a failed refine plus a cold solve. [built below]
2. The refiner's cost per level (~15 evaluations): start it from the predicted position
   (`lam_hint = prev + signed move`) and let a converged parabola stop earlier; target 6-8.
3. Cadence 10 -> 20: the backstop is now half the cold solves, ~7% of a run. The user decided
   the cadence; a measured 2-3% is the case for changing it.
4. Certification (Moler-Payne per level, no tension evaluations but boundary integrals) is the
   next fixed cost per warm solve; profile it before touching assembly.

## Two more measurements the same evening (micro-benchmark on the N = 6, k = 2 optimum)

Plan once, perturb `p` by a Gaussian step, solve the moved shape cold, prefix-warm, and
windowed (`windows = 2.5 |predicted move|`); evaluations and the relative error against cold:

| step | cold | prefix warm | windowed | note |
|---|---|---|---|---|
| 1e-3 | 182 | 87 | 65, err 1e-12 | |
| 3e-3 | 175 | 81 | went COLD (175) | per-level prediction wrong at a near-double |
| 3e-3, joint-block prediction | 175 | 81 | 66, err 3e-13 | fixed |

- **Refiner hints** (start the refiner at the predicted position): 64 against 65 evaluations.
  No gain; reverted. The refiner's three starting points and parabola steps are its cost.
- **Near-degenerate levels must be predicted together.** At this shape levels 3 and 4 are
  1e-3 apart; their own blocks predicted moves of 0.32 and 0.13 and they moved 0.12 and 0.32 --
  the branch moves on the wrong members, which is degenerate perturbation theory doing what
  it does. douse's `predicted_moves` now groups levels whose gaps are within reach of their
  joint motion and gives every member the spectral radius of the JOINT block (cross terms),
  plus a floor of a quarter of the largest move for near-stationary levels (one predicted
  2e-4 and moved 7e-3, second order). With that the 3e-3 step tracks warm.

## Certification, profiled (2026-10-15, one tracked N = 6 warm solve, 4 levels)

`track_levels` with certification 8.0 s, without 5.8 s (best of 3; 65 evaluations either
way): the Moler-Payne certificates are 28% of a tracked warm solve. Inside `certify_levels`
(1.56 s under cProfile): `moler_payne -> boundary_sup -> sample -> evaluate` 0.94 s, i.e. the
basis evaluated on the sup-norm sample set at each level's own `k` -- all of it in
`_bessel._j_series` (0.67 s self time over 120 calls); the coefficient vectors (`coef`, a GSVD
per level) 0.52 s. So the Bessel series is the common cost of assembly AND certification, and
it is the next thing to make fast. Deduplicating `(nu, k r)` does not apply (one order per
column); the series evaluation itself, or an order-ladder recurrence, does.

**The series kernel is not the problem (2026-10-15).** `_bessel._j_series` on a realistic
block -- 28 orders x 300 points, 8191 entries on the series branch -- takes 1.0 ms, about
0.12 us per entry; an active-set variant that retires converged entries from the term loop is
slower (1.7 ms, the fancy indexing costs more than the spare terms) and changes no value. So
the 0.67 s in the profile is VOLUME: ~120 calls on blocks of hundreds of columns by a thousand
or more points. The lever is fewer entries, not a faster series: a smaller Moler-Payne
sup-norm sample sized by what the bound needs, fewer certified levels when the caller asks for
fewer, and the order-ladder recurrence only if it cuts the per-entry cost below 0.1 us.
