# Phase 3 on the douse benchmark: silent count errors gone; drops and one value bound remain (2026-10-11)

Script: `notes/phase3_rescore.py`. It re-scores the solves in
`notes/margin_rule_offline{,_A}.json` (truncation; bases D and A; 1e-10 and 1e-8; 41 cases × 5
windows), with no re-solving. Phase 3's quantities follow from each saved level:
`resolution = max(prec*lam, bound)` and `count_shortfall = not met`.

- **Old scoring:** the scoreboard's, matching at 1e-6 relative, with count flags only.
- **New scoring:** the contract of `tests/test_level_resolution.py`.
  - Group the levels whose resolution intervals overlap.
  - A group's total must equal the multiplicities of the trusted reference levels inside the
    union, or a member must be flagged.
  - A reference level counts as inside when its own interval (value ± reference bound) meets the
    union. The references certify COUNTS; some values are known only to 4e-6 relative (T1.3).
  - A trusted reference level that no group covers is a **DROP**. No level exists to carry a
    flag, so it is Phase 4's count audit.

| basis | prec | old: ok / flagged / SILENT | new: ok / flagged / SILENT / DROP |
|---|---|---|---|
| D | 1e-10 | 161 / 1 / **13** | 164 / 1 / **0** / 10 |
| D | 1e-8 | 169 / 4 / 1 | 169 / 1 / 0 / 4 |
| A | 1e-10 | 172 / 1 / 2 | 173 / 1 / **1** / 0 |
| A | 1e-8 | 170 / 3 / 2 | 170 / 1 / 0 / 4 |

(30 solves per row are unscored: no trusted reference levels. One declined at D, 1e-8.)

## What changed and what is left

- **No count is silently wrong anywhere.** The old SILENT rows were of three kinds:
  - value errors that the 1e-6 match scored as drop + extra, which `resolution` now covers;
  - counts read inside the floor, now flagged by `count_shortfall`;
  - whole drops, now reported as DROP rather than as a wrong count.
- **DROP is Phase 4's.** T1.2's and ngon12's doubles on basis D, and the missed partner of
  T2.2's pair at 118.38 (3.5e-5 apart). Nothing per-level can see them.
- **The one SILENT row is a VALUE bound, not a count:** T2.8-N5s2 on basis A at 1e-10, window
  (-1e-12, -1e-12).
  - The level at 179.30225112 reports `delta_res` = 6.5e-11 relative at margin R = 3.06, so it
    is met.
  - Reference run B (Fourier-Bessel plus sources) puts the eigenvalue at 179.30225099, with a
    Moler-Payne bound of 6.8e-13 relative.
  - Runs A and C (FB only) and this solve agree with each other to 1e-12, and all sit 7e-10
    relative from B.
  - This is the TODO item "`delta_res` is admitted into `Level.bound` at the very edge of
    `prec`": on a basis whose floor is the basis's, `delta_res` under-covers, here by 11x at
    R = 3. Admission needs a margin above R = 2, or Moler-Payne alongside.
- **Three met levels that looked uncovered against the reference were reference precision,**
  not solver error: T1.3 (reference bound 4e-6 relative) and T2.5 (1.1e-7). This only showed up
  once matching moved from 1e-6 down to `resolution`.

## Caveats

- These are the saved solves from the margin study, rescored, and the contract test re-solves
  five of them on HEAD. The rescore assumes HEAD's levels equal the saved ones, which holds
  because Phase 3 changes no search behaviour.
- `width` is 0 throughout (`collapse=False` is the default).
- macOS only.
