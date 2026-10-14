# Short edges: a local source patch  (2026-10-13)

`notes/short_edge_sources.py`. douse's N = 6, k = 2 ratio run (seed 1) drives one hexagon edge
toward zero: at the stall it is 0.042 long, against 0.63-1.19 for the others (diameter 1.44,
perimeter 4). Both corners on it are blunt (0.86 pi, 0.61 pi); every corner is singular.

**Why it stalled.** Each rung at 1e-10 (margin, 2 is needed):

| rung | basis | margins lam_1..4 |
|---|---|---|
| 0 | corners | 0.02-0.09 |
| 1 | corners + offset layer | 0.6-0.9 |
| 2 | corners + offset layer, larger | 2.3, **2.0**, 6.8, 9.0 |
| 3 | lightning | 1.1 on lam_1 |
| 4 | plane waves | 0.02-0.09 |

The values agree on every rung; the precision claim is marginal. The offset layer sits 0.05 x
diameter = 0.072 out, farther than the short edge is long. A step further along the collapse
and rung 2 misses too, so the optimizer took 5 refusals in a row and stopped.

**Moving the whole layer closer** (offset `min(0.05 D, ell/3)`, `ell` the smallest corner
clearance, with more sources) fixed the stall shape's first four levels. But it moved the layer
on 11 of the 41 douse benchmark cases, and pushed T2.6-N5k4-L7-ref3 from rung 1 to rung 2.
Closer without enough density is worse: 48 sources at 0.029 gave margin 1.4.

**A local patch** (built): 9 sources over 2e of boundary either side of each short edge's
midpoint (e < 0.15 D), e/3 out. They sit on rungs 1 and 2 only, and the global layer is
untouched. On the stall shape plus the 11 benchmark cases at 1e-10, implemented code:

| case | before | with the patch |
|---|---|---|
| stall hexagon | flagged (no rung met; lam_2 margin 2.0), 179 s | clean, margins 5.0-26, 116 s |
| T2.1-rerun | CLEAN, rung 2, 475 columns, 43 s | CLEAN, rung 1, 339 columns, 21 s |
| the other 10 | | same verdicts and rungs |

No case became wrong or SILENT. T2.2's FLAGGED-WRONG was there before.

Not established: the douse run itself (seed 1) with the patch; shapes with two or more short edges
next to each other; a short edge next to a reflex corner. There, the patch raises if a source
lands inside, which douse reads as a refusal and re-plans.
