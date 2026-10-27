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
- **douse benchmark** (basis D, 41 cases x 5 jitters): see below.
