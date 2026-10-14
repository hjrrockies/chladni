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
