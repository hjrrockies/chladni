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

## What changes

Nothing in `src/`. TODO records the result and the T1.4-oct/ngon8 penalty regressions (relevant if
the penalty is ever reconsidered at 1e-8, where `reg="auto"` still uses it).
