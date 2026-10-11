# Do rank flips at the cutoff move outcomes on douse shapes? (2026-10-11)

Script: `notes/rtol_nudge.py`; data in `notes/rtol_nudge.json`, compared against E1's truncation
runs at 1.0x (`notes/regularization_1e10_e1.json`).

The study re-solves the douse benchmark at 1e-10 (basis D, truncation, the five `scoreboard.JITTERS`
windows) with `rtol` at 0.9x and 1.1x of `sqrt(prec*eps)`. E3 had found the first dropped singular
value within 10% of the cutoff at 15% of evaluated points. This asks whether that moves answers.

| rtol | ok | flagged | SILENT | unscored |
|---|---|---|---|---|
| 0.9x | 161 | 4 | 10 | 30 |
| 1.0x | 161 | 1 | 13 | 30 |
| 1.1x | 160 | 1 | 14 | 30 |

Per window, as outcomes at 0.9 / 1.0 / 1.1 (`o` ok, `f` flagged, `S` silent), for cases that move
or are ever silent:

| case | w1 | w2 | w3 | w4 | w5 |
|---|---|---|---|---|---|
| T1.2 | oSo | oSo | ooo | ooS | Soo |
| T1.4-hex | ooo | foo | ooo | ooo | ooo |
| T1.4-oct | ooo | ooo | ooo | ooo | ooS |
| T1.5-r2 | oSo | ooo | ooo | oSS | oSo |
| T2.2 | SSS | SSS | SSS | SSS | SSS |
| T2.3-endpoint | ooo | ooo | ooo | SoS | ooo |
| T2.9-N5k3convex | foo | oof | ooo | foo | ffo |
| T2.9-ngon12 | SSS | SSS | ooS | SSS | ooS |

**The outcome moved in 16 of 205 solves under a 10% nudge, and the count structure in 17.**

There are two kinds of defect:
- **Cutoff-sensitive:** T1.2, T1.5-r2, T2.3-endpoint, T1.4-oct and T1.4-hex. The silent loss
  appears or disappears with a 10% move of the cutoff.
  - T1.2's silent windows are the only defect on basis D on a level that meets `prec`
    (`notes/margin_rule_offline.md`), so the one defect a T = 2 margin rule misses is a rank-flip
    effect.
  - These are the same cases that are unstable under window jitter (Phase 1's stability score). A
    10% `rtol` change and a 1e-13 window change are both small perturbations of a chaotic
    search, and the rank decision is one channel through which they act.
- **Cutoff-robust:** T2.2 (silent at every rtol and window) and most of ngon12. These are basis
  failures; both are unmet levels and are right or flagged on basis A.

No single `rtol` is better: the totals are flat (10, 13 and 14 silent). Retuning the cutoff only
reshuffles which solves fail.

## What it means

- Rank flips are real on douse shapes at 1e-10 and account for roughly half the silent losses on
  the stand-in basis. They do not show up in totals, only in which solve fails. That is the
  instability the stability score measures.
- This does NOT reopen the penalty. The penalty removes the rank decision, but E1 showed it trades
  these losses for others (T1.4-oct and ngon8), at 1.8x the cost.
- The levers that remain are the ones already planned: the T = 2 margin rule for unmet levels,
  and a planner basis that moves levels away from R < 2. Whether T1.2's met-level rank flip
  survives on an adequate basis: on basis A at 1e-10, T1.2 was right on all 5 windows.

## Caveats

macOS only; basis D; one rung; ±10% only, not a scan.
