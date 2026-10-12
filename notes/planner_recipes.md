# Phase 5 step 0: the recipe ladder, measured on the douse benchmark (2026-10-12)

Script: `notes/planner_recipes.py`; data in `notes/planner_recipes.json`. The douse benchmark:
41 cases, 35 scored, on reference windows, at prec 1e-10 and 1e-8. Truncation regularization;
3 boundary and 1 interior collocation point per column.

**PASS** is what the planner accepts: every level met; the count audit not short or excess; no
silent or dropped count by Phase 3's resolution groups; nothing uncertain or declined.
**Bias** is a met level off a TIGHT reference value by more than its own bound.

## Recipes

- **Fx**: Fourier-Bessel at every corner, lappy's per-corner sizing (orders up to
  `k r + 2 (k r)^(1/3) + 3`) times x.
- **Px**: the same with a precision term, orders up to `k r + t(prec) (k r)^(1/3) + 3`, times x.
  `t = 2^(-1/3) (1.5 ln(1/prec))^(2/3)` is where `J_nu(k r)` has decayed to `prec` past its
  turning point: 8.4 at 1e-10 and 7.3 at 1e-8, against lappy's fixed 2.
- **B and +Sn**: a fundamental-solution source layer, n per boundary wavelength.

| recipe | 1e-10: cols / wall / met / pass | 1e-8: cols / wall / met / pass |
|---|---|---|
| F1.0 | 84 / 7.5 s / 38% / 5 | 84 / 6.5 s / 52% / 15 |
| F1.5 | 126 / 12.5 s / 57% / 10 | 126 / 10.2 s / 77% / 25 |
| F2.0 | 168 / 19.9 s / 66% / 17 | 168 / 14.8 s / 82% / 26 |
| F2.5 | 208 / 30.0 s / 69% / 20 | 208 / 24.0 s / 91% / 32 |
| F3.0 | 249 / 35.1 s / 73% / 22 | 249 / 29.1 s / 91% / 32 |
| F4.0 | 332 / 50.7 s / **85%** / **24** | 332 / 41.5 s / 92% / 32 |
| **P1** | **136 / 14.8 s** / 60% / **17** | 126 / 11.0 s / 76% / 25 |
| **P2** | 271 / 42.4 s / 78% / 22 | 251 / 29.8 s / 91% / 32 |
| B (F1.2 + sources) | 120 / 12.1 s / 45% / 8 | 120 / 10.3 s / 68% / 22 |
| F1.5+S3 | 153 / 17.5 s / 65% / 16 | 153 / 13.3 s / 81% / 26 |

(Out of 35 scored cases. Medians for columns and wall time.)

- **Size is what matters, and the precision term spends it well.** P1 passes as many cases as
  F2.0 (17) with 20% fewer columns and 25% less wall time at 1e-10. At 1e-8 it matches F1.5.
- **Sources barely raise the met fraction.** Their value is the one biased case, below.
- **Ladders, by expected wall time to the first pass** (every rung paid until one passes):

  | ladder | 1e-10: mean / never pass | 1e-8: mean / never pass |
  |---|---|---|
  | **P1 > P2 > F4.0** | 53 s / 11 | 27 s / 3 |
  | F2.0 > F4.0 | 50 s / 11 | 33 s / 3 |
  | F2.0 > F3.0 > F4.0 | 61 s / 11 | 33 s / 3 |
  | F4.0 alone | 86 s / 11 | 74 s / 3 |

  No ladder within this family gets past 24 of 35 at 1e-10 or 32 of 35 at 1e-8.
- **Never pass at 1e-10 (11 cases):** T1.3 (declines on every recipe), T2.1-rerun, T2.2, the
  T2.3 trio, the four T2.6 cases and T2.8-N5s2. They are mostly douse's own optimized iterates.
  Most reach 70-90% of levels met at F4.0 (`notes/hard_cases.md`).
- **One SILENT row persists on every large Fourier-Bessel-only recipe: T2.8-N5s2.** It is the
  biased well of `notes/admission_bound.md`, a met level 7e-10 off. Recipe B (with sources) avoids
  it, but B is too small to pass. Either a source layer on the large rungs, or the Moler-Payne
  cross-check on `delta_res` admission (`mp_bound <= 20*prec*lam`), would turn it into an unmet
  level, which the planner then climbs past or flags.
- **Low-margin levels (margin < 20), the starved-basis guard's trigger,** number 86 to 220 per
  recipe at 1e-10. The guard will fire on most cases at the lower rungs.

## Proposal for 5.1-5.2

- **Ladder:** P1 > P2 > P3, a single precision-aware family, rung m = m x the Airy-sized orders.
  P3 (~400 columns) is unmeasured: the 5.4 benchmark measures it. F4.0 is its nearest measured
  relative.
- **Bias:** build the Moler-Payne cross-check in 5.2 rather than leaving it to 5.5. The planner's
  stopping rule reads `met`, so the biased level must not read met. That makes T2.8 a climb or a
  flag, never a silent row. Sources stay out of the ladder: they buy little met fraction for
  their columns.
- **Cost:** about 50 s of planning per case at 1e-10 (27 s at 1e-8), plus the guard's
  confirmation solves, paid once per plan. douse re-uses the plan. Eleven of 35 cases at 1e-10
  end FLAGGED at the top rung.
