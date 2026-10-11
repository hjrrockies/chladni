# Would a margin rule catch the wrong counts? Measured before designing it (2026-10-11)

Script: `notes/margin_rule_offline.py`. Data: `notes/margin_rule_offline.json` (basis D) and
`notes/margin_rule_offline_A.json` (basis A).

The study re-solves the douse benchmark with truncation (`reg="svd"`) at 1e-10 and 1e-8, on five
windows per case, on two bases:
- **D**: the scoreboard's stand-in, at 1.0x lappy's per-corner Fourier-Bessel sizing;
- **A**: the references' recipe, at 1.5x.

Each level gets the margin

    R = prec * lam / (eps/c) = 2 * prec * lam / delta_res

using the refiner's `delta_res` as the ε/c proxy (`notes/count_ratio.py`'s definition). **`met`
holds exactly when R >= 2** on all 8062 levels.

Each count **defect** is a reference level miscounted in a solve, or a solved level matching
nothing. It is attributed to the solved level assigned to it, or to the nearest one if the level
was dropped. A rule "flag where R < T" catches a defect when that level has R < T.

> **Correction, 2026-10-11.** The defect attribution below scores a DROPPED level by the nearest
> solved level's R. For T1.2 and ngon12 on basis D that level is 28-78% of λ away, so their R
> (14.4 and 1.52) belongs to an unrelated level. Both are whole doubles dropped outright (class
> (a) in `docs/design/douse_requirements.md`, Phase 3), not count misreads. Re-read by kind, **no
> level that meets `prec` had a wrong count on the douse benchmark, on either basis.** The
> remaining defects on unmet levels are either value errors that the bound covers (and that the
> 1e-6 match window scores as drop + extra), a missed partner of a pair, or a simple level read as
> a double inside its floor. Result 3 below, about T1.2, is superseded.

## Result

| basis | prec | defects | defects with R < 2 (unmet level) | right-count levels with R < 2 |
|---|---|---|---|---|
| D | 1e-10 | 22 | 20 | 61% |
| D | 1e-8 | 10 | 10 | 49% |
| A | 1e-10 | 3 | **3** | 43% |
| A | 1e-8 | 6 | **6** | 24% |

1. **On the references' basis, every wrong count is on a level whose value already misses
   `prec`.** All 9 defects (3 at 1e-10, 6 at 1e-8) have R between 0.03 and 0.75, and 7 of them
   are unflagged as counts today. No level that meets `prec` had a wrong count.
2. **So the cheapest margin rule is T = 2: "a level that misses `prec` has its count flagged too".**
   - On basis A it catches 9 of 9.
   - Its "needless" flags land only on levels that already report a value shortfall, so douse
     sees no new flags on levels it can use.
   - It needs no new machinery: R comes from `delta_res`, which every level already has.
   - The cost is honesty about counts that happen to be right on unmet levels: 43% of right-count
     levels at 1e-10 on basis A, 24% at 1e-8.
3. **A larger T buys nothing on basis A, and on basis D it costs a lot.** On basis D at 1e-10,
   the only defects on levels that DO meet `prec` are T1.2's dropped member at 68.85 (R = 14.4,
   2 of 5 windows). ngon12's four (R = 1.52) are unmet and caught at T = 2. Catching T1.2 needs
   T ≈ 20, which flags about half of the good, resolving levels. T1.2 is right on basis A, so it
   is a starved-basis effect, not a count-logic one.
4. **The dominant limit is the basis, not the count logic.**
   - Even basis A misses 1e-10 on 40% of levels (1218 of 2031 met), and 1e-8 on 20%.
   - Basis D misses 1e-10 on 58% of levels.
   - The silent rows on the benchmark (T2.2, T1.5-r2, ngon12) are all on unmet levels.

## What it means for the plan

- **Phase 3 can be small:** flag the count wherever the level is unmet (T = 2), and report it as a
  shortfall. The neighbourhood-at-a-gap `resolution` machinery is not supported by this data. No
  defect was found on a level that meets `prec` with an adequate basis, so there is nothing for a
  coarser-resolution report to rescue there. It may still matter for douse's UX, as "here is what
  I can resolve"; that is a design call.
- **Phase 5 (the planner's enrichment loop) is where the 1e-10 work is.** "Works often" means
  getting levels to R >= 2, and on the reference recipe that is 60% of levels at 1e-10. Moving
  the minimal planner before, or alongside, the margin rule looks right.

## Cross-check: `notes/count_ratio.py` re-run on HEAD (rectangle-free bound-coverage grid)

`notes/bound_coverage.py` covers the L-shape, 3π/2 sector, equilateral triangle and disk; bases
from starved to generous; prec 1e-8..1e-13. It was regenerated on HEAD (72 s on 8 processes) and
`count_ratio.py` re-read it. The HEAD data is kept beside the pre-(C2) originals, which stay as
they were:
- `notes/bound_coverage_head_2026-10-11.jsonl`;
- `notes/count_ratio_head_2026-10-11.json`.

Changes on HEAD:
- Wrong multiplicities fell from 12 to 8.
- The largest R of a wrong multiplicity fell from 31 to 18.4.
- Correct degenerate counts still have a median R of 116.

**This grid qualifies result 1.** With deliberately starved bases it has wrong multiplicities ON
levels that meet `prec` (R >= 2), all of them a double read as simple:

| domain | basis | prec | R |
|---|---|---|---|
| eq_tri | plane waves at 0.5x | 1e-8 | 18.4 |
| L_shape | FB at the re-entrant corner only, n=40 | 1e-8 | 15.9 |
| L_shape | FB re-entrant only, n=80 | 1e-13 | 9.9 (parked rung) |
| L_shape | FB re-entrant only, n=40 | 1e-10 | 3.2 |

With basis D's T1.2 (R = 14.4) that is a consistent picture: **on a starved basis a double can be
read as simple at a level whose value meets `prec`, with R up to about 20**. A T = 2 rule does not
see it. On this grid, moving T from 2 to 20 costs only 34 of 704 correct records (4 of them
degenerate), but on the douse benchmark (basis D) it costs about half of the resolving levels. The
price of T > 2 is domain-dependent.

So the margin rule alone does not close the starved-basis case. Candidates, for design:
- `basis_sufficiency`'s multiplicity precondition as the guard;
- the planner refusing to stop at a basis that has not cleared it;
- T ≈ 20 only where the basis is known to be marginal.

## Caveats, and what is not established

- `delta_res` is a proxy. It is clamped from below and widened on a stalled refine
  (`count_ratio.py`'s caveats). The plan's ε/c from the cached wide bracket was not computed.
- The references were built in part from basis A (at 1e-12). A level is trusted only when at least
  two of runs A, B and C agree AND the count matches FEM, so the counts are independently checked.
  Values are not independent, though.
- 41 cases × 5 windows. Defects are few on basis A (9), so "every defect has R < 2" is 9 of 9, not
  a rate.
- macOS only, on the shipped code (truncation at both rungs, i.e. NOT the 1e-8 default today,
  which is the penalty).
