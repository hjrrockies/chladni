# Regularization at prec = 1e-10: keep truncation (2026-10-10)

Phase 2 of the 1e-10 plan. Script: `notes/regularization_1e10.py` (`e1`, `e2`, `e3`); raw output in
`notes/regularization_1e10_e{1,2,3}.json`. macOS, `OPENBLAS_NUM_THREADS` unpinned, 8 workers.

## Decision rule (written before running)

> Switch the default at 1e-10 to a penalty mode only if it improves count correctness or
> stability on the benchmark by a clear margin, at no more than 2x wall time, with no new
> silent losses. Otherwise keep truncation and record why.

**Outcome: keep truncation** (`reg="auto"` unchanged; it already picks `svd` below
`PENALTY_PREC_MIN = 1e-8`). Neither penalty mode clears the rule: both add new silent losses, and
the posterior also costs 2.9x wall time.

## Variants

| name | construction |
|---|---|
| truncation | `reg="svd"`, threshold rank, `rtol = sqrt(prec*eps)` (today's 1e-10 default) |
| penalty | `reg="tikhonov_qr"`, tau = `C_PEN*prec` = 2e-12 |
| posterior | `tikhonov_qr` with `rtol = 30*eps`, then Rayleigh-Ritz tensions on the 8 lowest penalized vectors (prototype in the script; no `src/` support) |

## E1: douse benchmark, basis D, 5 jittered windows per case (41 cases, 175 scored solves)

| variant | ok | flagged | SILENT | stable | tightness (med bound/err) | wall med | wall total |
|---|---|---|---|---|---|---|---|
| truncation | 161 | 1 | 13 | 36/41 | 219 | 6.0 s | 24.7 m (1.0x) |
| penalty | 152 | 10 | 13 | 38/41 | 293 | 6.2 s | 43.7 m (1.8x) |
| posterior | 161 | 3 | 11 | 35/41 | 199 | 9.2 s | 71.2 m (2.9x) |

Per-case outcomes, five windows each (`o` ok, `f` flagged, `S` silent); cases where any variant
differs or is silent:

| case | truncation | penalty | posterior |
|---|---|---|---|
| T1.2 | SSooo | fffff | oSooo |
| T1.4-hex | ooooo | ooooo | ooooS |
| T1.4-oct | ooooo | **SooSo** | ooooo |
| T1.5-r2 | SooSS | ooooS | SooSS |
| T2.2 | SSSSS | SSSSS | SSSSS |
| T2.3-endpoint | ooooo | ooooo | oooof |
| T2.9-N5k3convex | oooof | ooooo | oofof |
| T2.9-ngon12 | SSoSo | fffff | ooooo |
| T2.9-ngon8 | ooooo | **SSSSS** | oooSo |

What this says:
- **Penalty** converts truncation's silent T1.2 and ngon12 rows into flags (an honest shortfall,
  and the smoother tension does make them stable), but it **creates new silent losses on T1.4-oct
  and ngon8**, cases truncation gets right on every window. Its bound is also looser (293x vs 219x,
  consistent with `moler_payne_vs_tension.md`). It fails "no new silent losses".
- **Posterior** fixes ngon12 and one T1.2 window, but adds a silent window on T1.4-hex and ngon8,
  is no more stable (35/41 vs 36/41), and costs 2.9x. It fails both the wall-time cap and the
  silent-loss condition.
- The stability hypothesis (a penalty's smoother tension reduces window sensitivity) is weakly
  supported for the plain penalty (38 vs 36) and not for the posterior. Not a clear margin.
- T2.2 is silent under every variant: it is not a regularization problem (basis D is a fixed
  stand-in; see the scoreboard's starved-basis notes).

## E2: cluster grids under ulp jitter (seeds 0..4; seed 0 unjittered)

(50,3) and (65,4) clusters over 11 band/licence ratios, plus the 8-fold 170 pi^2 band 0.3:
115 cells x seeds per variant.

| variant | right | flagged | SILENT | wall |
|---|---|---|---|---|
| truncation | 115 | 0 | 0 | 2.7 m |
| penalty | 110 | 5 | 0 | 4.5 m |
| posterior | 115 | 0 | 0 | 8.9 m |

Truncation is already perfect here, so neither penalty mode can improve on it. (An earlier run
reported 5 silent 170 pi^2 rows for every variant; that was a script bug, the expected count
hard-coded to 4 for an 8-fold level. Fixed before this run, along with a jitter seed leaking into
seed-0 jobs in reused workers.)

## E3: the truncation hazard census

Along every evaluated point of truncation solves on the benchmark (436 points over the cases that
evaluate), the ratio s_(r+1)/cutoff of the first dropped singular value to the cutoff:
median 0.65; >= 0.5 at 76% of points; **>= 0.9 at 15%**.

So the hazard is not rare in the configuration sense: a rank flip at a few-percent perturbation of
the cutoff is available at about one point in seven. But E1 and E2 show no outcome attributable to
it at 1e-10. The silent rows are the same under the penalty (no rank cutoff at all) on T2.2 and
T1.5-r2, and the T1.2/ngon12 rows are window-sensitive under the posterior too. Attribution was
not measured point by point; that would mean re-solving with `rtol` nudged and comparing.

## Not established

- Linux: none of this ran on Zen 3/5 runners. The ulp-jitter harness in E2 is the stand-in.
- Basis D is a fixed stand-in for the planner. With an adequate basis (Phase 5) the silent rows may
  vanish for every variant and the comparison may tighten.
- tau was not swept; only `C_PEN = 0.02` was tested. The plan's tau <= prec/70 (1.4e-12) is
  near this (2e-12) but not equal.
- Whether the penalty's flag-instead-of-silent behaviour on T1.2/ngon12 can be had from
  truncation through the Phase 3 margin rule. That is the next thing to check: if the margin
  rule flags those rows, the penalty's only advantage is gone.

## Follow-up 1: the same comparison at prec = 1e-8, where the penalty IS the default

`CHLADNI_REG_PREC=1e-8 CHLADNI_REG_VARIANTS=truncation,penalty`; outputs `*_prec1e-08.json`.
`penalty` here is exactly what `reg="auto"` ships at 1e-8 (`tikhonov_qr`, tau = 0.02 prec).

| E1, 1e-8 | ok | flagged | SILENT | declined | stable | tightness | wall total |
|---|---|---|---|---|---|---|---|
| truncation | 169 | 4 | 1 | 1 | 39/41 | 143 | 20.3 m |
| penalty (shipped) | 155 | 11 | 9 | 0 | 37/41 | 284 | 35.3 m |

| case | truncation | penalty |
|---|---|---|
| T1.2 | ooooo | oooSo |
| T1.4-oct | ooooo | ooSSS |
| T1.5-r2 | ooood | ooooo |
| T2.2 | fSfff | SSSSf |
| T2.6-N5k4-L7-ref0 | ooooo | fffff |
| T2.9-ngon12 | ooooo | fffff |
| T2.9-ngon8 | ooooo | ooSoo |

E2 at 1e-8 (cluster grids, seeds 0..4): both 115/115 right; truncation 2.2 m, penalty 3.7 m.

At **1e-6** (E1 only, `*_e1_prec1e-06.json`): truncation 170 ok / 5 flagged / 0 silent, 41/41
stable, tightness 3.4, 16.5 m; penalty 162 / 8 / 5, 40/41, 26.4, 29.8 m. Same direction.

On douse shapes **truncation beats the shipped 1e-8 default on every axis**: 8 fewer silent
losses, 2 more stable cases, a 2x tighter bound, 0.6x the wall time. The penalty was promoted at
1e-8 on rectangle evidence (a Linux rank-flip refusal it removed; `tests/test_regularization.py`),
not on this benchmark. Reversing it (`PENALTY_PREC_MIN`, so `"auto"` is the threshold at every
rung) is a default change with history, so it is left for review rather than made here; see TODO.
What a reversal would have to answer: the rectangle-suite rank-flip refusal the penalty fixed
(`tests/test_regularization.py`, "2026-09-21 CI refusal") comes back on the default path at 1e-8,
on Linux.

**Measured 2026-10-11.** That refusal was on the SUITE pencil (`SUITE_RTOL` ~ 1.5e-15) at 1e-6,
not on a truncation at the rung's own `rtol`. E2 at 1e-6 and 1e-8 over 16 jitter seeds (0..16):
- truncation at its default `rtol`: right on 391 of 391 cells × seeds at both rungs, the CI cell
  (65π², band 2.6) included;
- the penalty: the same, at 1.7-1.8x the wall time.

So on the jitter harness the reversal's known risk does not reproduce. A Linux run is still the
real test. The suite-pencil finding (`notes/suite_rtol_sweep.md`) is the likely reason the
original failure needed `SUITE_RTOL`.

## Follow-up 2: the 170 pi^2 stage-1 declines are a suite-pencil artifact

TODO recorded the 170 pi^2 cell (m=4, band 0.3) at 1e-10 declining at stage 1 under 7 of 8 jitter
seeds. E2 did not reproduce it. The difference is the pencil: the test solves on the rectangle
suite's pencil, truncated at the fixed `SUITE_RTOL = sqrt(1e-14 eps)` ~ 1.5e-15
(`tests/rect_suite.py`, `FINEST_PREC`), while a default 1e-10 solve truncates at
`sqrt(1e-10 eps)` ~ 1.5e-13. A/B on the same cell, seeds 0..8 (seed 0 unjittered):

| pencil | outcome |
|---|---|
| suite (`SUITE_RTOL`) | right at seeds 0 and 8; DECLINED at 1..7 (961 evals each) |
| default `rtol` at 1e-10 | right at all 9 (206-221 evals) |

The suite pencil keeps near-null directions that 4-ulp noise shakes. That is also the pencil in the
refine gap-root trace (TODO: "a pencil truncated at `rtol` 1.5e-15"). So part of the suite's 1e-10
`ulp_sensitive` population is a property of a pencil sized for 1e-14, not of what a 1e-10 user
gets. How much was not measured.

## What changes

Nothing in `src/` on this branch. TODO records the result, the 1e-8 reversal question, and the
suite-pencil finding.
