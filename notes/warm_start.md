# Warm start: what a narrow window saves  (2026-10-13)

`notes/warm_start.py`. The douse N = 6, k = 2 ratio optimum (chladni_vs_lappy.jsonl, seed 0),
planned once (rung 2, 562 columns), vertices moved by ~1e-3, solved on the same plan:

| | window | evaluations (search + refine) | wall | levels |
|---|---|---|---|---|
| cold | [8.58, 157.08] | 177 (102 + 75) | 50.3 s | 10 |
| warm | [13.98, 69.27] | 84 (56 + 28) | 21.1 s | 4 |

The four warm values agree with the cold ones to 1.2e-13 relative; the audit read ok on the warm
window. **2.4x on one solve.** The cold window's top, Weyl(n + HEADROOM) padded by 15%, found 10
levels where 8 were needed.

Per-iterate profile of the cold re-solve that motivated it (52 s): the search 86% (186 tension
evaluations at ~0.24 s, SERIAL: assembly 0.10 s, GSVD 0.10 s, regularization 0.04 s per `lam`),
certification 13%, the count audit ~0.

In a douse run (`benchmarks/chladni_vs_lappy.py`, N = 4, k = 1, seed 0): 9 of 12 solves ran
warm, 3 cold (the first and the cadence), no warm refusals, the same optimum (2.49999999994).
The run's wall time fell 275 s -> 105 s, but the 275 s row ran under heavy contention, so that
ratio is not a clean measurement.

Not established: the per-run speedup at N = 6, how it varies with step size, N and n.
