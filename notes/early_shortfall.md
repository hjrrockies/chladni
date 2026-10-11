# Phase 4 step 4: an early abort pays; a floor pre-scan does not (2026-10-11)

Script: `notes/early_shortfall.py`; data in `notes/early_shortfall.json`. Douse benchmark, bases
D and A, prec 1e-10, reference windows: 82 solves, instrumented from outside (`src/` untouched).

| | measured | verdict |
|---|---|---|
| **Early abort** at the first refined level that misses `prec` | 65 of 82 solves have such a level. Stopping there saves a median **57% of wall time** (quartiles 51-62%) and 38% of tension evaluations; Moler-Payne certification alone is 24% of wall time | **built** (`on_shortfall="raise"`) |
| **Floor pre-scan** from stage 1's samples | over 804 levels (394 unmet), the lowest stage-1 sample inside a level's bracket predicts "unmet" with AUC **0.59** (Spearman with final margin -0.18). A cut catching 60% of unmet levels flags 50% of met ones | **dropped** |

**Why the pre-scan fails, in principle.** A sample away from a well bounds the floor from ABOVE
(the tension there is floor plus wall), and a shortfall needs a bound from BELOW. Only refining
the well reveals its floor, so "as soon as possible" is after the first deficient level is
refined, as `docs/design/douse_requirements.md` already said.

**The early abort's test is the refiner's own `delta_res > prec*lam`** (margin < 2), applied
before Moler-Payne. On the 8062 benchmark levels of `notes/margin_rule_offline.md`, no level with
margin < 2 was certified met afterwards, so stopping on it lost nothing measured. The abort
carries that level, its margin and the values located so far. It carries no audit and no
assembled result: those are the work it skips.

Wall per solve here: median 7.3 s, 90th percentile 21.2 s (8 processes on one machine).
