# Stage-wise rank policy: the build, step by step  (2026-10-27)

Plan: `/Users/hringer/.claude/plans/ok-so-there-s-a-cached-diffie.md`. Diagnosis:
`notes/triangle_apex_planner.md` sections 9-12. The baseline numbers every step is gated against are
recorded on the baseline session's branch (`claude/fervent-wiles-d26d63`, `notes/stage_policy.md`).
This file records what each step built and what its gate measured. Invariant throughout: one assembly
per lam, and only VECTORS cached (tension vectors per policy, the stack spectrum, scales).

## Step 1: plumbing (829bd3e)

`chladni.mps.policy.Policy` (the pencil's own = None, a frozen rank); pencil and objective caches
keyed by policy (default keys unchanged); `MPSPencil.spectrum(lam)`, the stack's singular values
the `"svd"` mode already computed and threw away; `Objective.spectrum(x)`, recorded with each
evaluation through `Evaluation`; `coef(..., rank=)`; `Level.rank`.

Gate: default suite passed; golden / pencil-cache / regularization slow tiers 92 passed; a
pencil-backed `search_interval` on the L-shape and the apex triangle bit-identical to the pre-change
code (values, bounds, evaluation counts 194 and 113).

## Step 2: the coarse penalty from the shared factorization (c8078a9)

`Policy.penalty(tau_rel = sqrt(eps))`: `([A_B; tau I], A_I)`, tau relative to the stack's SPECTRAL
norm (an equilibrated stack's Frobenius norm is sqrt(n_cols), which would grow tau with the basis).
Reduced from the `"svd"` factorization (`StackFactors`): a small QR of `[R; tau I]`, then the
tensions by the CS identity -- the reduced blocks are jointly orthonormal, so each generalized
singular value is `c / sqrt(1 - c^2)` for a singular value `c` of the numerator block (one SVD, no
GSVD; reading `c` keeps the small tensions accurate). `regularize(reg="svd")` is now
`reduce_svd(svd_factors(...))`, bit-identical.

Gate (`notes/stage_policy_step2_results.txt`, `_timing.txt`):

| | |
|---|---|
| agreement with `tikhonov_qr` at the same tau | 2-4e-11 relative, 10 smallest tensions |
| apex lightning basis, lam_8 cell, max/median secant | coarse 1.0, threshold 10.8 |
| coarse floor / plateau at lam_1, lam_8, lam_15 | 2.6e-5, 6.7e-6, 5.4e-6 (gate <= 1e-3) |
| cost per evaluation vs the threshold path | 18-50% CHEAPER (budget was +15%) |

## Step 3: certification at the level's rank (c162747)

`certify_levels` and `eigenfunctions` read `coef` at `Level.rank`; `search_interval(objective=)`
refuses an objective not built on the solver's own pencil. Gate: default suite; golden /
count-certification / on-shortfall / eigenproblem / eigenfunctions slow tiers 60 passed; L-shape and
apex bit-identical.

## Step 4: refinement and the count at a stable rank frozen per root cell (c162747, opt-in)

`solve(stable_rank=True)`: after the refine stage, a root whose cell `lam (1 +/- 3e-4)` has a noise
tail -- the stable rank (`policy.stable_rank`, two recorded spectra, TOL 1e-2) below the threshold
rank -- is re-refined in that cell at the frozen stable rank through a `PolicyView` confined to the
cell, with up to two flank evaluations at the (C2) reach (`multiplicity.wide_reach`), and its count
is read there. Its levels carry `Level.rank`.

**A defect caught on the way:** the first version keyed the view's samples by policy only, so roots
frozen at the SAME rank read each other's cells; on the L-shape the double at 49.348 took the
neighbouring eigenvalues 44.96 and 56.69 (tension ~0) as its wide bracket, (C2) cut its second branch,
and it read multiplicity 1, silently. The view is now confined to its cell.

Gate so far:
- **L-shape [9, 120], 1e-10:** 15 of 20 roots re-refined (ranks 110-115 against threshold 119-143);
  every value unchanged within its bound, counts identical (22, both doubles), all met;
  88 extra evaluations, 5.9 per re-refined root (budget 5).
- **Rectangle cluster grids** (`notes/stage_policy_step4_grid.py`, shipping pencil, N = 50, 65, 125,
  145, prec 1e-6/1e-8/1e-10, 11 bands): the stable path re-ranked **0 roots in all 132 cells** --
  plane-wave rectangle pencils have no noise tail (`notes/knee_generalize_results.txt`) -- so the two
  paths are identical: 131 right, 1 SILENT at 125pi^2 (present on the shipped path too; the baseline
  session records the same cell). The grids cannot exercise step 4.
- **douse benchmark, first version (one pair per root)** (`notes/stage_policy_step4_douse.py`,
  basis D, 41 cases x 5 jitters, 1e-10, data `..._1e-10_v1-pair-rule.json`):

  | | ok | flagged | silent | evaluations |
  |---|---|---|---|---|
  | shipped (threshold) | 161 | 1 | 13 | 57,043 |
  | stable rank, one pair | 164 | 2 | 9 | 63,698 (+12%; 9.0 per re-refined root) |

  The shipped row matches the baseline session's exactly. Fixed: T1.5-r2 (2 silent), T1.2 (2 silent
  -> flagged), T2.9-N5k3convex (flag -> ok). **New silent: T1.4-oct, last jitter** -- the regular
  octagon's double at 79.02 split into two levels of multiplicity 1, one unmet.

**Why, and the fix: one pair is one sample of the noise.** The stable rank there came out 92 from the
root and a refine sample 4e-11 away; the same root reads 96 from a pair at any spacing 1e-12 .. 1e-4
in a standalone check. Indices 92-95 (1.5-1.8e-11 x sigma_1) move 1e-4 .. 2e-3 between most pairs and
> 1e-2 between one: they flicker at the tail's edge but carry real content (cut, the floor at the
double rose 20x). The rank is now read from EVERY recorded spectrum within `PAIR_ETA = 1e-6` of the
root (41 there), cutting at the first index whose MEDIAN movement over consecutive pairs exceeds
`STABLE_TOL` (`policy.stable_rank_of`). The tail proper moves 10-40% between every pair, so the median
keeps that cut. With it, the octagon reads the double as 2 at rank 96, met; the apex lightning and
L-shape results of step 5 are unchanged.

Also observed: under multi-threaded BLAS the SHIPPED path is not run-to-run deterministic (L-shape
rung 0, worst bound 8.9e-12 in one run and 2.4e-11 in another, 465 vs 470 evaluations): the threshold
rank's flutter at work. The single-threaded identity checks are bit-exact; CI pins one thread.

## Step 5: bracketing on the coarse penalty (4f8ba47, opt-in)

`solve(coarse_bracket=True)`: stage 1 through `PolicyView(obj, Policy.penalty())`, asked for
`max(prec, 10 tau)`; stage 2 refines each bracket at the stable rank read at its coarse minimum,
frozen over the bracket (brackets holding one level are wide on purpose; the stable rank drifts by a
few over a factor of three in lam); threshold refine where there is no tail or the frozen refine
refuses; stage 3 at each root's rank.

| case (n = 15 prefix, 1e-10) | shipped | coarse bracketing |
|---|---|---|
| apex (0.1, 0.25), lightning rung | DECLINED, 2113 evals | 23 levels, all met, worst bound 3.9e-11, min margin 5.1, max err 4.6e-11, 448 evals |
| L-shape rung 0 | 22 counted, all met, 465-470 evals | 22 counted, all met, worst bound 7.7e-12, 424 evals |

Coarse bracketing alone (`notes/stage_policy_step5_probe.py`): the apex lightning basis settles in
224 evaluations, 23 brackets, every reference value inside one; the L-shape in 242, both doubles
inside single brackets.

### The douse gate, 1e-10 (basis D, 41 cases x 5 jitters; `notes/stage_policy_step4_douse.py`)

| path | ok | flagged | silent | stable across jitter | evaluations |
|---|---|---|---|---|---|
| shipped (threshold) | 161 | 1 | 13 | 37/41 | 57,043 |
| step 4: stable rank, median rule | 165 | 1 | 9 | 39/41 | 63,486 (+11%) |
| step 5, first version | 125 | 9 | 41 | | 55,607 |
| **step 5: rescue + hump stop + widened brackets** | **170** | **0** | **5** | **41/41** | 60,545 (+6%) |

Step 4 adds no silent loss the shipped path does not have (T1.5-r2 fixed; T1.2 2 -> 1). Step 5's
first version DROPPED doubles in bracketing (the cover's exclusion has no floor; on the coarse
penalty's ~1e-7 floor it excludes a double's well, which is never certified single) and lost close
pairs; three fixes, each of which can only add a bracket (commit 3e1dd46): `rescue_floor`
(`RESCUE_DEPTH`), `well_extent`'s hump stop, and widening each bracket to the coarse well's floor
region. The widening walks out to a 10x rise, i.e. +- ~5 res, beyond the +- 2 WIDE_FACTOR f/c the
step-5a measurement (branch claude/vibrant-tharp-c0e58b, `notes/coarse_policy_check.md`) found
necessary for a split pair that reads as one coarse well.

**The only silent case left is T2.2, silent on every path** (the hard roster's 15-degree,
re-entrant shape; `notes/hard_cases.md`). T1.2, T1.5-r2, T2.9-ngon12 and T1.4-oct, silent on the
shipped path, are ok. Every case reads the same at every jitter.

## Through the planner (a0632ae, c71fa9c): `Eigenproblem(dom, coarse_bracket=True)`

The ladder climb, `solve` and the warm path's window search pass `coarse_bracket` to every search.
Measuring it on the apex line exposed three more things, all fixed in c71fa9c:

- **The stable rank costs depth where the threshold does not flutter.** On apex (0.15, 0.25) rung 3
  (shipped accepts it) the stable rank's floors and margins were 3-5x worse on every level and one
  went unmet. The flutter's damage is in BRACKETING; near a root the threshold's extra directions are
  mostly useful. So stage 2 is now **threshold first, stable rank on failure** (refuse or miss `prec`),
  and stage 3 the same: a threshold root whose count is uncertain is recounted at the stable rank
  (apex (0.1, 0.25) lightning, 451.09: a noise direction read as an undecided branch).
- **Empty coarse brackets plus a fluttering fine tension give noise minima.** A refine or side search
  in a bracket that holds no eigenvalue accepted sigma 0.077 at 465.32 as a level. Every root and
  every reported level must now sit in a coarse well (`RESCUE_DEPTH` x the coarse plateau; every true
  eigenvalue does, 175 of 175 in step 5a); dropped levels are counted (`n_noise_dropped`).
- **Certification inherits the flutter.** The threshold rank's coefficient vector is noisy, and so is
  its Moler-Payne bound (apex (0.3, 0.25) rung 3: 4.5e-11 and 3.3e-9 at roots 1e-11 apart).
  `certify_levels` now takes the smaller bound of the level's rank and the stable rank near it (plan
  step 6; each is a valid bound).

`plan_and_solve(15, 1e-10)` along the apex line (`notes/apex_line.py ... coarse`, data
`notes/apex_line_coarse.json`; shipped from `notes/apex_line.json`):

| apex | shipped: rung, met | coarse: rung, met, worst rel. bound |
|---|---|---|
| (0.05, 0.25) | 1, 0/15 | 3, **15/15**, 9.6e-12 |
| (0.1, 0.25) | 0, 0/15 (fix 1: 2, 1/15) | 3, **15/15**, 3.6e-11 |
| (0.2, 0.25) | 0, 0/15 | 3, **15/15**, 8.0e-11 |
| (0.1, 0.2) | 0, 0/15 | 3, **12/15**, 1.3e-9 |
| (0.3, 0.25) | 3, 13/15 | 3, **14/15**, 2.0e-9 |
| (0.15, 0.25) | 3, 15/15 | 3, 15/15 |
| (0.1, 0.3) | 1, 15/15 | 1, 15/15 |
| (0.3, 0.3) | 0, 15/15 | 0, 15/15 |

Rung-3 prefix solves alone (`search_interval`, met of 22-23): apex (0.1,0.25) declined -> 23/23;
(0.2,0.25) 20 -> 23; (0.3,0.25) 18 -> 22; (0.15,0.25) 22 -> 22. Wall times were 380-680 s per
planner call on a machine at load 20-50; not comparable with the shipped runs.

## The douse gate on the final coarse path (c71fa9c), three precisions

`notes/stage_policy_step4_douse.py PREC 6 shipped,coarse`; data `..._<prec>_shipped-coarse.json`.
Basis D, 41 cases x 5 jitters (30 unscored: no reference).

| prec | shipped: ok / flagged / silent / declined | coarse | evals shipped -> coarse | same across jitter |
|---|---|---|---|---|
| 1e-10 | 161 / 1 / 13 / 0 | **169 / 1 / 5 / 0** | 57,043 -> 65,060 (+14%) | 37 -> 40 of 41 |
| 1e-8 | 169 / 4 / 1 / 1 | 170 / 0 / **5** / 0 | 52,028 -> 56,100 (+8%) | 39 -> 41 |
| 1e-6 | 170 / 5 / 0 / 0 | 168 / 5 / **2** / 0 | 42,535 -> 47,420 (+11%) | 41 -> 40 |

**Two regressions to fix before the default can switch:**
- **T2.2 at 1e-8:** shipped reads it flagged at four jitters and silent at one; coarse reads it silent
  at all five. T2.2 is silent on every path at 1e-10 (the hard roster's 15-degree re-entrant shape),
  but at 1e-8 the shipped path at least says so. The coarse path loses the FLAG.
- **T1.5-r2 at 1e-6:** silent at two jitters (shipped: ok at all five). At 1e-10 the coarse path fixes
  this case's shipped silents; at 1e-6 it introduces two.
