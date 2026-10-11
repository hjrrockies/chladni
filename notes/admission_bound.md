# Which bar should `Level.bound` report? Measured; nothing changed yet (2026-10-11)

Script: `notes/admission_bound.py`. Data: `notes/admission_bound.json` (the douse benchmark
re-solved with truncation, bases D and A, at 1e-10 and 1e-8, five windows; both `delta_res` and the
Moler-Payne bound saved per level). The grid check uses `notes/bound_coverage_head_2026-10-11.jsonl`.

**The question.** Today `delta_res` is admitted into `bound` when `delta_res <= prec*lam` (`c1c3f53`).
Two cases showed it under-covering:
- a starved basis on a Linux Zen 5 runner: admitted by 1% (R = 2.02), and 700x under the error;
- T2.8-N5s2 on basis A: R = 3.06, 11x under (`notes/phase3_rescore.md`).

**What the second case is.** It is not edge admission but **basis bias**. The Fourier-Bessel-only
bases (A and C, and these solves) converge to a clean, deep well 7e-10 relative from the
eigenvalue that run B (FB plus sources) pins to 6.8e-13. `delta_res` measures the well, not the
bias. At 1e-8 the same level has R = 56 and is still 1.9x under-covered, so **no admission margin
fixes this class**. Moler-Payne covers it in every run.

## Policies priced

The table counts met levels (out of ~2000 per row) and "harm", meaning a level claimed met while
its true error exceeds `prec*lam`. Coverage is judged only against reference levels at least 10x
tighter than the level.

| policy | D 1e-10 met | A 1e-10 met | D 1e-8 met | A 1e-8 met | harm (all rows) | under-covered (all rows) |
|---|---|---|---|---|---|---|
| today (admit at `delta_res <= prec*lam`) | 849 | 1218 | 1171 | 1622 | 1 | 7 |
| margin M = 10 | 495 | 974 | 1023 | 1473 | 1 | 6 |
| Moler-Payne only | 470 | 881 | 1018 | 1472 | 0 | 0 |
| cross-check K = 20 | 847 | 1213 | 1169 | 1620 | **0** | 6 |

- **A margin is dominated.** It costs as many met levels as Moler-Payne-only and still leaves the
  harmful case.
- **Moler-Payne-only is honest but pessimistic.** It flips 379 / 337 levels (D / A, 1e-10) from met
  to unmet. Of the 336 that can be judged against a tight reference, 335 are truly within `prec`:
  Moler-Payne is about 2x loose there (median MP/(prec*lam) ≈ 2, p90 ≈ 5). Under Phase 3, every
  one of those levels would also raise `count_shortfall`.
- **The cross-check** admits `delta_res` only when `delta_res <= prec*lam` AND
  `mp_bound <= K*prec*lam`, so the two bars must roughly agree.
  - The biased T2.8 level has MP/(prec*lam) = 39-76 at 1e-10; the Linux edge case has ~67,000.
  - K = 20 catches both, at a cost of 2-5 met levels per row.
  - On the grid it costs nothing at 1e-8 and 1e-10.
  - What it leaves under-covered is harmless: bars about 2x small on levels whose error is still
    within `prec`.

## Decision (2026-10-11)

Change nothing yet. The cross-check is the measured candidate. Revisit it with the planner: a
basis that resolves T2.8-N5s2 (FB plus sources, as run B) may make the biased-well case rare
enough that the rule is moot. Or the planner may need it exactly where it plans FB-only bases.
