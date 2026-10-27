# The triangle apex (0.1, 0.25): why the planner certifies nothing  (2026-10-26)

Dirichlet triangle `0, 1, 0.1+0.25i` (angles 68.2 / 15.5 / 96.3 deg, edges 1.0 / 0.934 /
0.269), `Eigenproblem.plan_and_solve(15, prec=1e-10)` on `fix/douse-pipeline-2026-10-14 @
6d8d4e2`. Found by douse's triangle grid (`douse` PR #2). Scripts: `notes/apex_planner_probe.py`
(each rung, prefix solve), `notes/apex_basis_sweep.py` (rows in `notes/apex_basis_sweep_results.txt`) (any basis, narrow windows at three
reference levels), `notes/apex_fb_radius.py`, `notes/apex_line.py`, `notes/apex_dense_rung.py`;
data `notes/apex_probe_*.json`. The machine was shared (load ~40 on 10 cores): **wall times
here are only comparable within one table.**

## 1. The answer is right on rungs 1-2 and wrong on what the planner returns

| rung | basis | cols | worst / median rel. error vs rung 2 | rel. bounds | met |
|---|---|---|---|---|---|
| 0 | FB at all corners, 1x | 90 | **2.0e-5 / 3.2e-6** | 5e-4 .. 5e-2 | 0/15 |
| 1 | FB 2x + offset layer | 299 | 5.2e-10 / 2.8e-11 | 5e-9 .. 6e-7 | 0/15 |
| 2 | FB 3x + offset layer | 389 | -- | 2e-9 .. | 1/15 (full solve) |
| 3 | lightning | 180 | DECLINED ("slope estimate never settled") | | |
| 4 | plane waves | 318 | DECLINED (same) | | |

- **Independent check: lappy** (its own venv, `Eigenproblem(dom, precision=1e-10).solve(15)`,
  134 columns, self-reported 1e-6) agrees with rung 2 to 1e-13 .. 2e-10 on all 15, 9e-10 on
  lam_1. lam_1 = 288.96438540(3). The rung-1/2 values are correct to ~1e-10; they are only
  **uncertified**.
- **Every bound is honest**: bound >= 120x the actual error on every rung. Nothing is
  silently wrong.
- **But the planner returns rung 0**, whose values are off by up to 2e-5. All rungs tie at
  "0 levels met" in the early-abort ranking (`_aborted_fraction`), `score > best` keeps the
  FIRST, and rung 0 is then solved in full. The tie-break picks the poorest basis on the
  ladder. A grid that reads `lam` and ignores the `unmet` flag gets 1e-6 .. 2e-5 errors where
  1e-10 values were in hand.

## 2. Why the corner terms stall: the short edge belongs to no useful expansion

FB at all three corners stalls at a ~5e-4 bound **whatever its size**:

| FB-only basis (scales at v0, v1, apex) | cols | rel. bound at lam_1 / lam_8 / lam_15 |
|---|---|---|
| 1, 1, 1 | 90 | 5.1e-4 / 2.0e-3 / 3.6e-2 |
| 3, 3, 3 | 270 | 4.6e-4 / 4.8e-4 / 9.9e-4 |
| 5, 5, 5 | 448 | 2.5e-4 / 4.9e-4 / 4.3e-4 |
| 6, 6, 6 | 538 | 1.9e-4 / 4.4e-4 / 3.2e-4 |
| 9, 3, 3 / 3, 9, 3 / 3, 3, 9 | 318-550 | 1.8e-4 .. 1.0e-3 |

The residual of the FB-only eigenfunction (|u| on the boundary over interior rms, 3x basis,
lam_1) is 4.6e-4 on the short edge `apex -> 0` and 1.9e-4 on the base near vertex 0; 9e-6 on
the long side. **Both endpoints of the short edge carry FB expansions that vanish identically
on it**, so the only columns that can cancel residual there are the 15.5-degree corner's. That
corner has `alpha = pi/omega = 11.6`: orders 11.6, 23.2, ..., and at `k r <= 56` only ~5-8 of
them are not exponentially small. Enlarging it (3, 9, 3) helped most (2.5x), and not enough.
This is the "group B" stall of `notes/hard_cases.md` (sharp corners, stagnant floors), now
with a mechanism: in a triangle, an edge is served only by the corner opposite it, and a sharp
opposite corner has almost nothing to give.

Ruled out:
- **Collocation**: bdry x2 or int x4 on rung 1, no change (5.2e-9 at lam_1 all three).
- **FB scaling radius**: rebuilding the columns scaled at 1.0x or 0.5x the nearest-vertex
  distance instead of `evaluation_radius` changes nothing (`notes/apex_fb_radius.py`).
- **Source offset closer**: 0.02 D instead of 0.05 D is 300x worse (1.3e-6 at lam_1).

## 3. What helps: sources, and enough of them (FB + fundamental-solution sources, as the user suggested)

Narrow windows (+-1%) at lam_1, lam_8, lam_15; relative bound, margin (2 is needed):

| basis | cols | lam_1 | lam_8 | lam_15 |
|---|---|---|---|---|
| FB 2x + layer 6/wl (rung 1) | 299 | 5.1e-9, 0.13 | 8.0e-9, 0.31 | 1.0e-7, 0.23 |
| FB 3x + layer 6/wl (rung 2) | 389 | 4.4e-9, 0.16 | 2.0e-9, 0.69 | 8.3e-9, 0.40 |
| FB 4x + layer 6/wl | 478 | 1.0e-9, 0.54 | 9.0e-10, 1.5 | 1.9e-9, 1.6 |
| FB 2x + layer 6 + short-edge patch (9) | 308 | 9.4e-10, 0.52 | 6.5e-9, 0.52 | 5.4e-8, 3.0 |
| FB 3x + layer 6 + short-edge patch (9) | 398 | 7.3e-10, 1.0 | 1.0e-9, 1.1 | 9.4e-11, 2.1 |
| FB 2x + patch, no layer | 189 | 2.5e-4 | 2.1e-4 | 3.7e-3 |
| **FB 2x + layer 9/wl** | 359 | **9.3e-11, 2.2** | **2.8e-11, 7.2** | 3.1e-9, **2.8** |
| FB 2x + layer 12/wl | 418 | 4.1e-12, 49 | none (search failed) | 9.0e-12, 22 |
| FB 3x + layer 12/wl | 508 | 2.0e-12, 100 | none | none |
| FB 1x + lightning (rung 3: 10 poles, sigma 1, 0.5 D) | 180 | 1.0e-11, 19 | none | none |
| FB 2x + lightning | 270 | 9.5e-12, 21 | none | none |
| FB 1x + lightning, 20 poles | 270 | 1.1e-11, 19 | none | none |
| FB 1x + lightning, sigma 2 | 180 | 6.2e-12, 32 | none | none |
| FB 1x + lightning, reach 0.25 D | 180 | 7.1e-12, 28 | none | 2.8e-12, 71 |
| FB 2x + layer 6/wl + lightning | 389 | 1.9e-12, 100 | none | none |
| FB 2x + layer 9/wl at 0.08 D | 359 | 2.4e-12, 84 | none | none |

Prefix solves of the best candidate that searched cleanly, FB 2x + layer 9/wl
(`notes/apex_dense_rung.py`, `notes/apex_dense_rung.json`; met = of the `n + HEADROOM` = 19):

| apex | cols | met | worst rel. bound of the first 15 | min margin |
|---|---|---|---|---|
| (0.05, 0.25) | 367 | 18/19 | 8.4e-11 | 2.4 (all 15 met) |
| (0.1, 0.25) | 359 | 11/19 | 3.3e-9 | 0.90 |
| (0.2, 0.25) | 346 | 12/19 | 1.2e-9 | 1.2 |

So 9/wl is a large step (from 0 met on every shipped rung) but not a whole fix.

- **Source DENSITY is the lever, not corner size.** 6 -> 9 per wavelength on the same
  corners takes lam_1 from margin 0.13 to 2.2 and lam_8 from 0.31 to 7.2.
- **The short-edge patch helps, but does not decide it.** Widening `SHORT_EDGE` to 0.35 D
  (patch on the 0.27 edge) leaves the full planner still failing here, and at apexes
  (0.05, 0.25) / (0.15, 0.25) (table 4).
- **Lightning works as a basis and fails as a search.** See section 4.

## 4. Why the richest bases return nothing: the threshold rank jitters

"none" above is the search declining: "the slope estimate never settled: 0.28, 0.40, 0.64,
0.89, 2.28, 3.37 over 6 doublings". The same decline removes rungs 3 and 4 from this shape's
climb. On the lightning basis (180 cols) at lam_8 the BASIS is excellent: tension 2.8e-11 at
the reference value, a clean cone (3.2e-6 at +1e-6, 3.2e-4 at +1e-4 relative). But on a
401-point grid over +-1%:

- the threshold rank moves between **119 and 122 of 180** with lam,
- the tension JUMPS between neighbouring samples (0.0311 -> 0.0280 across 5e-5 relative,
  then back up), slopes ~10x the median,
- so the seed's max secant across a jump grows as the grid refines (it doubles per doubling
  at the end) and `_seed` (`mps/bracket.py:544`, `SETTLE = 0.10`) declines.

The well is fine; the plateau around it is discontinuous because near-dependent columns
(clustered poles; a 12/wl layer) cross the `reg="auto"` threshold one at a time as lam moves.
`notes/hard_cases.md` saw lightning "DECLINE" and attributed it to near-duplicate columns;
this is the measured path from those columns to the decline.

## 5. How widespread (apex line, full `plan_and_solve(15, 1e-10)`)

`notes/apex_line.json`. "SE 0.35" is the same run with `SHORT_EDGE` widened to 0.35 D in that
process. Angles at (0, 1, apex); the short edge is apex -> 0.

| apex | angles | short edge | returned rung | met | worst / median rel. bound | SE 0.35 |
|---|---|---|---|---|---|---|
| (0.05, 0.25) | 78.7 / 14.7 / 86.6 | 0.255 | 1 | **0/15** | 1.5e-7 / 1.9e-8 | rung 1, 0/15, 1.8e-8 / 5.7e-9 |
| (0.1, 0.2) | 63.4 / 12.5 / 104.0 | 0.224 | **0** | **0/15** | 5.5e-3 / 2.4e-3 | same |
| (0.1, 0.25) | 68.2 / 15.5 / 96.3 | 0.269 | **0** | **0/15** | 4.6e-2 / 5.8e-3 | same |
| (0.2, 0.25) | 51.3 / 17.4 / 111.3 | 0.320 | **0** | **0/15** | 3.5e-3 / 9.6e-5 | same |
| (0.3, 0.25) | 39.8 / 19.7 / 120.5 | 0.391 | 3 | 13/15 | 3.4e-9 / 2.3e-11 | same |
| (0.15, 0.25) | 59.0 / 16.4 / 104.6 | 0.292 | 3 | 15/15 | 6.5e-11 / 1.0e-11 | same |
| (0.1, 0.3) | 71.6 / 18.4 / 90.0 | 0.316 | 1 | 15/15 | 2.1e-11 / 8.6e-12 | same |
| (0.2, 0.3) | 56.3 / 20.6 / 103.1 | 0.361 | -- | | | rung 3, 15/15 |
| (0.3, 0.3) | 45.0 / 23.2 / 111.8 | 0.424 | -- | | | rung 0, 15/15 |

- **4 of the 7 apexes at height <= 0.25 certify nothing**, and 3 of those return rung 0 with
  bounds of 1e-4 .. 5e-2. It is a region, not a point: sharp corner (12-18 deg) at vertex 1
  with all three corners singular.
- **Which shapes pass is decided by whether rung 3 (lightning) searches cleanly**, not by the
  geometry's difficulty: (0.15, 0.25) passes on rung 3 between two failures that decline there.
  This is the rank-jitter decline of section 4.
- (0.1, 0.3) has a 90-degree apex, which is reflection-closed, so the singular-corner rungs
  place nothing there and rung 1 passes at 197 columns.
- The short-edge patch (SE 0.35) changes the outcome nowhere.
- Each failing apex costs 450-550 s (shared machine) for a result that is then wrong to
  1e-4 .. 1e-6: five rungs plus the full re-solve of rung 0.

## 6. Proposed fixes (none implemented; accuracy first, so each needs the user's go-ahead)

1. **Ranking tie-break (`eigenproblem._plan`, small, no numerical change).** When no rung is
   accepted and the met fractions tie, rank by the located levels' `delta_res / (prec lam)`
   (median, lower is better) or prefer the higher of two non-declined rungs. Here it would
   return rung 1 or 2: values right to ~1e-10, bounds 1e-9 .. 1e-7, instead of rung 0 at
   1e-6 .. 2e-5. This is the cheapest change with the largest effect on what a grid stores.
2. **A denser offset layer on the ladder.** FB 2x + 9/wl met lam_1, lam_8, lam_15 at margins
   2.2-7.2 where every shipped rung missed. Options: rung 2 becomes FB 2x + 9/wl (it was not
   measured on its own before shipping, `plan.py`), or a new rung between 2 and 3. As a prefix
   solve it met all 15 at (0.05, 0.25) and 11-12 of 19 at the two harder apexes (section 3), so
   it is a partial fix on its own. Must be run
   on the douse benchmark and the triangle pilot before it ships; a 12/wl layer already starts
   to jitter the rank (section 4), so 9 is near the edge and the rule may need to be
   `density ~ f(prec)` rather than a constant.
3. **Rank-stable search for clustered/dense bases** (`mps/bracket._seed`, numerical core):
   freeze the threshold rank over a window (`lam_ref=(lo, hi)`, which already exists) for
   rungs whose columns are near-dependent, or let the seed tolerate isolated jumps (settle on
   the median secant, or the max over non-adjacent pairs). That would let rung 3 (lightning,
   margin ~20 at lam_1 here, and it solved apex (0.15, 0.25) outright) serve this shape. Larger
   change; needs the benchmark.
4. Not proposed: widening `SHORT_EDGE`. It is not what fails here.

**Recommendation.** Do 1 first: it is small, touches no numerics, and turns this region's
answers from 1e-6-wrong into 1e-10-right-but-flagged. Then 3, because every FB + sources
basis that searched to completion here certified at margins 19-100: the bases already work,
and the search is what refuses them. 2 is the fallback if 3 cannot be made safe.

The flat triangles (height 0.10-0.15) are a different stall: at apex (0.5, 0.12) (angles 13.5 /
13.5 / 153) rung 1 meets 7 of 15 with margins 0.8-5.6, and neither 3x corners nor a patch on
either slanted side moves it (margins 1.2 / 0.8-0.9 / 1.2-1.8 at lam_1, lam_8, lam_13). Not
diagnosed; the density lever (fix 2) was not tried on it.

## 7. Fix 1, built (2026-10-26)

`eigenproblem._rank` replaces `_met_fraction` / `_aborted_fraction` in `_plan`'s fallback:
`(met fraction, log10 margin of the levels that missed)`, compared as a tuple. Two defects
fixed:
- **Aborted rungs always scored 0.** `PrecisionShortfall.located` holds floats
  (`mps/search.py:721`), and `_aborted_fraction` read `.lam` / `.delta_res` off each item, so it
  never counted anything. Now it counts the values located before the stopping one; each of
  them passed `_abort_if_unmet`. **This changes which rung comes back on ANY shape where no rung
  is accepted, not only on ties.** The douse benchmark's flagged cases were not re-run.
- **A tie kept the first rung.** The margin now breaks it. Margins come from `delta_res`, which
  can overstate a poor basis, so they only break ties. The returned answer still carries its own
  Moler-Payne bounds and flags.

Cost, apex (0.1, 0.25), stock and fixed run side by side on the same machine: 2539 s and 3003 s
wall (+18%). The extra time is the full re-solve of rung 2 (389 columns) instead of rung 0 (90).
Both runs got ~35% CPU, so read the ratio, not the absolute times.

Tests: `tests/test_eigenproblem.py`, the last four. The suite's default tier passes (1785), and
so does the slow tier of `test_eigenproblem.py` (8).

## 8. Fix 3 design: freeze the rank LOCALLY, on a lattice in lam (proposed, not built)

What is already known (`notes/landscape_findings.md` sections 27.1, 34, 35):
- A rank frozen over a WIDE window is fatal. The numerical rank grows with lam, the frozen rank
  is the window's minimum, and the top of the window starves: about 2-3 retained directions per
  decade of floor, and up to 2e11x floor inflation over a 60x window.
- A rank frozen over a NARROW window (~6%) removed every flip, recovered 22 of 23 declines with no
  wrong answer, and moved the floor by 2%.
- Relaxing the decline at the seam (handing an unsettled seed to the cover) bought 9 wrong
  answers. The fix belongs in the objective, not in `bracket.py`'s tolerance.

So the target is an objective that is **continuous within cells whose width tracks the rate at
which the rank actually changes**, and that re-reads the rank in every cell.

1. **Cells on a fixed geometric lattice.** `lam_j = lam_0 (1 + delta)^j`. The rank used at `lam` is
   that of the cell containing it, a pure function of `lam`: no hysteresis, no dependence on
   evaluation order, cacheable, reproducible. That matters, because section 17's lesson is that a
   shape optimizer cannot descend on an objective that moves for reasons it cannot see.
2. **Cell width from the local rank drift.** The rank grows roughly like `k`, so over a relative
   width `delta` in lam it drifts by about `r delta / 2`. Choose `delta` so that drift is about
   1 (`delta ~ 2 / r`: 1.7% at r = 120). Measure `dr/dlam` from the threshold rank at the two
   ends of the search window rather than assume it. Large bases and high lam get finer cells
   automatically; cells never cover a decade.
3. **Cell rank = the minimum of the threshold rank at a few nodes in the cell** (3, as
   `_RANK_SAMPLES` does for `lam_ref` today). That is the safe direction: never above what the
   threshold rule allows anywhere in the cell. The cost at the cell's top is at most about one
   drift step (~1/3 decade of floor at 2-3 ranks per decade). That must be checked against
   `prec`. If it bites, fall back to the threshold rank for that cell rather than shipping a
   starved one.
4. **The search is told where the seams are.** The tension is continuous inside a cell and may
   jump at a node. The objective exposes the node positions. `_secants` skip only node-straddling
   pairs. Each node is evaluated with BOTH neighbouring cells' ranks, so each cell's curve is
   continuous on its closed interval. A Piyavskii region is split at nodes and never spans one.
   Unlike the rejected hand-over, nothing is guessed: the skipped pairs are exactly the ones whose
   rank differs, and every other secant still has to settle.
5. **Refine and certify in one cell.** A bracket is refined on the rank of the cell that holds
   its centre. A bracket that straddles a node is refined on both and kept only if they agree
   within the bracket. Moler-Payne is computed from the candidate eigenfunction and does not
   depend on the rank, so the certificate is unchanged.
6. **Opt-in first.** Turn it on only for plans whose columns are near-dependent by construction
   (lightning, a layer of 12/wl or more), keep the threshold rank elsewhere, and widen only after
   the benchmark.

Measure before building (cheap, `notes/apex_*` style): on this triangle's lightning basis
(180 columns, rank 119-122 across +-1% at lam_8), compute the cell rank at `delta = 2/r` and
check that (a) the seed's max secant settles within each cell, (b) the minimum tension at
lam_1, lam_8 and lam_15 moves by under ~1 decade against the threshold rank, (c) the rectangle
grids of section 34 keep their 22 recoveries and 0 wrong. Then the douse benchmark and the
triangle pilot.

## 9. The check, run: the rank COUNT is not the problem; the cut sits in rounding noise

Basis: the shipped rung 3 on apex (0.1, 0.25) (FB 1x + 10 lightning poles per corner, 180
columns). Cells on a geometric lattice in lam with delta = 2/r (1.6-2.0%), the cells within
+-2.5% of lam_1, lam_8 and lam_15. Scripts `notes/rank_freeze_check.py`, `rank_freeze_jump.py`,
`rank_freeze_rtol.py`, `rank_freeze_gap.py`, `rank_stable_cells.py`; data
`notes/rank_freeze_check_*.json`.

**1. Freezing the threshold rank per cell does nothing** (`rank_freeze_check.py`). In the lam_8
and lam_15 cells the threshold rank flips 100-149 times on a 201-point grid (118-131). Frozen at
the cell minimum, the seed's max secant grows exactly as before (lam_8 cell: 0.0027 -> 0.0203
against 0.0027 -> 0.0203), and the same cells decline. lam_1's cells are unaffected either way
(rank 98, no flips, margin 41).

**2. The jumps are rounding** (`rank_freeze_jump.py`). Bisecting the steepest pair with the rank
frozen at 120: over 8e-10 in lam (6e-13 relative) the raw columns move by ~1e-10, yet the
tension jumps 0.0232 -> 0.0248. At the cut, `s_r/s_1` and `s_{r+1}/s_1` are 1.7e-13 and
1.5e-13, and they trade places from one sample to the next.

**3. The spectrum has a noise tail** (`rank_freeze_gap.py`). At lam_8 and lam_15 the singular
values fall 2-4x per index down to ~2e-12, then flatten to ~1.1x per index from 1e-12 to 1e-13.
Moving lam by 1e-12 RELATIVE leaves the head unchanged and moves the tail ~20% (6.2e-13 ->
7.4e-13). The threshold cut (`rtol = sqrt(prec eps)` = 1.5e-13) lands inside that tail, so WHICH
noise directions survive is decided by rounding. That is the "rank flip" of
`landscape_findings.md` section 34 seen from the other side: freezing the count leaves the
choice of directions to rounding. lam_1's spectrum has no tail (fast decay to 1e-15), which is
why lam_1 always searched cleanly. A symptom of how fragile this is: lam_15's cell DECLINED at
the default `rtol` (1.49e-13) and searched cleanly at an explicit 1.5e-13.

**4. A global looser cutoff fixes the smoothness and costs the low end** (`rank_freeze_rtol.py`):

| rtol | lam_8 cell rank | max/median secant | lam_1 margin | lam_8 | lam_15 |
|---|---|---|---|---|---|
| 1.5e-13 (default) | 119-122 | 10.8 | 41 | DECLINED | 31 |
| 1e-12 | 102-105 | 9.7 | 5.0 | DECLINED | 42 |
| 3e-12 | 96 | 1.1 | 4.6 | 15.8 | 52 |
| 5e-12 | 95-96 | 1.5 | 3.9 | 11 | 42 |
| 1e-11 | 94 | 1.0 | 1.1 (unmet) | 6.0 | 26 |
| 3e-11 | 90 | 1.0 | 0.35 (unmet) | 2.8 | 2.7 |

The right cut is below the knee where there is a tail and at the threshold where there is
none. That differs with lam, so no single `rtol` serves it.

**5. A STABLE rank, frozen per cell, does** (`rank_stable_cells.py`). The stable rank at lam
is the number of leading singular values that move by less than TOL (relative) between lam and
lam (1 + 1e-12), capped at the threshold rank. The cell rank is its minimum at the cell's ends
and middle, frozen with `rank=`.

| TOL | lam_1 cells: rank, margin | lam_8 cells | lam_15 cells | max/median secant | seeds | declines |
|---|---|---|---|---|---|---|
| 1e-3 | 91, 4.95 | 90, 2.78 | 94-95, 2.51 | 1.0 | all settle | 0 of 11 |
| **1e-2** | **96-98, 41** | **94-96, 21** | **100-101, 46** | 1.0-1.3 | all settle | 0 of 11 |
| **5e-2** | **98, 41** | **97-99, 36** | **103, 35** | 1.1-2.7 | all settle | 0 of 11 |

(threshold rank there: 98 / 119-123 / 125-129.) Relative bounds at TOL 5e-2: 4.9e-12, 5.5e-12,
5.8e-12, against a shipped rung 3 that declined at lam_8 and lam_15. Values agree with the
references to <= 1e-10.
- It ADAPTS: at lam_1, where there is no tail, it keeps the threshold rank (98). Where there
  is a tail it cuts above it, 20-25 directions below the threshold, and it grows with lam
  (98 -> 97-99 -> 103 at TOL 5e-2).
- TOL is not delicate between 1e-2 and 5e-2. The tail moves ~20% and the head ~0, so the test
  sits in a wide gap. 1e-3 is too strict: it cuts real directions whose singular values move
  a little at this eta.
- Cost: two extra SVDs of the stacked pencil per node, three nodes per cell, against hundreds of
  tension evaluations per cell.

**What this changes in the section 8 design:** the cell rank is the minimum STABLE rank, not
the minimum threshold rank; everything else stands (lattice cells, delta ~ 2/r, the search told
where the cell boundaries are, opt-in first). Not yet measured: the cell boundaries themselves
(ranks differ between neighbouring cells by 1-4 here, so the seams step 4 handles still exist);
the rectangle grids of `landscape_findings.md` section 34 (22 recoveries, 0 wrong, to keep); the
douse benchmark; the pilot triangles. One shape and one basis so far.
