# A fixed penalty read through posterior tensions: clean at one tau, NOT robust at 1e-14

*Updated 2026-10-05, twice. First: the one loss was a search defect, fixed in `search.solve`, and
the row came back clean at every rung. Second, and overriding the first's conclusion: a sweep of
`tau = C eps sqrt(n)` over three basis sizes and the hard roster (section "The tau sweep") shows
that clean row was a LUCKY POINT. At 1e-14 no `C` is clean at every basis size, and the losses are
mostly SILENT. At 1e-12, `C = 100` was clean everywhere it was tried. Read that section first.*

2026-10-05. The question from `notes/moler_payne_vs_tension.md` section 1: the order-1 penalty
has no feasible `tau` at `prec = 1e-14` when `tau` scales with `prec`, because its floor (~1.4
tau) must stay under the licence while its noise jaw (~3 eps sqrt(n)) is fixed by the arithmetic.
Does fixing `tau` near the noise jaw, and reading POSTERIOR tensions, remove the squeeze?

Reproduce with `notes/posterior_penalty.py` (about 6 minutes for the three rungs).

## What was run

Three rows on the cluster-total grid -- the 50*pi^2 triple and 65*pi^2 quadruple, all eleven
bands, cells the multiplicity precondition excuses skipped, scored as `tests/test_cluster_totals.py`
scores them:

* **threshold** -- truncation at its derived `rtol`, what `reg="auto"` ships below 1e-8.
* **fixed tau** -- the penalty at `tau = 2e-14`, independent of `prec`, searched on `sigma_tau`.
* **posterior** -- the same pencil; every tension the search reads is the unpenalized
  Rayleigh-Ritz tension on the span of the 8 (`MAX_BRANCHES`) lowest penalized vectors: the
  singular values of `(B X) R^(-1)`, `I X = Q R`. It can only lower each tension, it chooses no
  rank, and at an eigenvalue it reads 9.7e-16 where `sigma_tau` floors at 2.3e-14.

## Results

    prec      threshold            fixed tau                    posterior
    1e-10     22/22                22/22                        21/22, 1 silent
    1e-12     22/22                22/22                        22/22
    1e-14     17/18 + 1 declined   0/18 (10 flagged, 8 silent)  18/18

* **At 1e-14 the posterior row is the only clean one**, and beats the threshold's decline. Fixing
  `tau` alone collapses there, as the floor-vs-licence argument predicts: the floor is the licence.
* **At 1e-10 and 1e-12, fixing `tau` alone already ties the threshold.** 2e-14 lies inside the
  clean band section 10.6 of `smooth_filter.md` found at 1e-12 ([7e-15, 5e-14]), so that is
  consistent; what is new is that it holds at 1e-10 too, where `tau = 0.02 prec` lost a member.
* **Cost:** the same number of evaluations (2085-2272 against 2209-2867), about 2-2.7x the wall
  time -- each posterior evaluation adds a separate-mode GSVD for 8 vectors and an 8x8 solve.

## The one loss is the search's, and the posterior exposes it

50*pi^2, band 0.3, `prec = 1e-10` (`trace_undercount.py` with the posterior objective swapped in):
the root lands on member 0, the read leaves a branch undecided, and the side search FINDS both
other members -- member 1 at +0.038 licences, member 2 at -0.041, both accepted. Member 2's find is
then discarded as "already a located root": the dedup merges two finds whose distance is within the
LARGER of their two `delta_res` bars, and member 1's find carries a refiner bar of 1.2e-8 (0.24
licences) against a separation of 1.09e-8. 2 of 3, silent.

That is the shape `search.py` already documents for the root's OWN bar (125*pi^2, where `r` was
excluded from `taken` for exactly this reason): a parabola-fit bar wide enough to swallow a genuine
neighbour. The posterior objective produces wider refiner bars, which is why it surfaces here; the
threshold's narrower bars hide it. Not fixed in this experiment.

## After the fix

The dedup now merges two finds only within the NARROWER of their refiner bars; a true duplicate is
still caught by the find's own reach (`(sigma / c) / bias_lo`). Regression:
`tests/test_rule_seam.py::test_a_wide_refiner_bar_does_not_swallow_a_found_neighbour`, which fails
without the fix. On every exercisable 50/65*pi^2 cell from 1e-6 to 1e-14, on the default and the
suite pencils (106 each), the fix changes nothing -- totals and evaluation counts identical. The
grid, rerun:

    prec      threshold            fixed tau                    posterior
    1e-10     22/22                22/22                        22/22
    1e-12     22/22                22/22                        22/22
    1e-14     17/18 + 1 declined   0/18 (10 flagged, 8 silent)  18/18

The threshold and fixed-tau rows are identical to the first run, evaluation counts included.

## What this does and does not establish

Established, on this instrument: a fixed `tau` near the noise jaw, read through posterior tensions,
counts the cluster grid correctly at 1e-10, 1e-12 and 1e-14, beating the threshold at 1e-14. The squeeze of `penalty_prec_limit_explained.md` is a property of
reading `sigma_tau`, not of the penalty.

Not established:

* **One `tau`, one family.** 2e-14 on 126-column plane-wave rectangles -- the family the reach
  constants were fitted on. The noise jaw scales with `eps sqrt(n)`, so `tau` should too; unchecked
  across basis sizes, corner bases, curved boundaries.
* **Whether the reach rule's calibrated constants transfer.** `COUNT_BELOW_CACHE` and its kin were
  calibrated on threshold tensions; they were not re-derived for posterior ones, and the row
  happened to pass.
* **Linux.** Measured on macOS only -- though a path with no rank decision should be the one that
  does not care.

## Next, if this is pursued

1. ~~Fix the dedup~~ -- done: the narrower bar.
2. `tau = C eps sqrt(n)`, and the same grid across basis sizes and on the hard roster (L-shape,
   GWW, disk), the posterior row against the threshold.
3. If both hold: a `reg="posterior"`-style mode on `MPSPencil` (or the posterior as the objective
   `MPSEigensolver` builds), then `PENALTY_PREC_MIN` falls and truncation can retire -- which also
   retires the Linux rank-flip item in `TODO.md` section 1 at its root.

## The tau sweep -- the clean row does not survive

2026-10-05, `notes/posterior_tau_sweep.py` (`size` about 30 minutes, `roster` about 2). `tau = C
eps sqrt(n)`, `C` in {3, 8, 30, 100}; `C = 8` is about the `tau = 2e-14` above at n = 126. The
cluster grid rebuilt at three paddings, n = 128, 172, 258 (the rows above were n = 172, at
`C` = 6.8). Cells: right, then f flagged, s SILENT, d declined.

    prec    n     threshold   C = 3        C = 8          C = 30        C = 100
    1e-12   128   22/22       22           22             22            22
            172   22/22       15 + 7d      21 + 1f        21 + 1s       22
            258   22/22       21 + 1s      22             21 + 1s       22
    1e-14   128   18/18       16 + 2s      17 + 1f        4  (5f 9s)    0  (10f 8s)
            172   17 + 1d     13 + 5d      15 + 1f + 2s   7  (5f 6s)    1  (10f 7s)
            258   17/17       16 + 1s      16 + 1s        6  (4f 7s)    0  (9f 8s)

The hard roster, total count and worst relative error against truth:

* disk (84 cols) and equilateral triangle (90): `C` = 3 and 8 clean at 1e-10, 1e-12 and 1e-14, with
  errors matching the threshold's (2-7e-16 at 1e-14). At 1e-14, `C = 30` and 100 return the
  triangle's six eigenvalues as TWO, silently; the disk takes one flagged level at `C = 30`.
* L-shape (80 pure corner Fourier-Bessel cols): identical at every `C`, and to the threshold, at 1e-10
  and 1e-12 (3 of 3, 1.6e-14 at 1e-12).

What this says:

* **The 18/18 at 1e-14 was a knife edge.** On the very same n = 172 cases, `tau` 2.0e-14 gave 18/18
  and 2.3e-14 (`C = 8`) gives 15, one flagged and two SILENT. No `C` is clean across sizes at 1e-14,
  where the threshold is 17-18 of 17-18 with one honest decline. A posterior mode at 1e-14 would
  trade the threshold's decline for silent wrong totals -- the trade this project refuses.
* **Two jaws again, in a different place.** Small `C` sits on the noise jaw and declines or loses a
  member (`C = 3`: 7 declines at n = 172). Large `C` loses members wholesale at 1e-14: the penalty
  bends the 8-vector span away from the cluster's eigenspace, and the Rayleigh-Ritz on it cannot
  recover what the span does not contain. The posterior moved the squeeze; it did not remove it.
* **At 1e-12, `C = 100` was clean at every size and on the roster**, and the scattered single losses at
  `C` = 3 and 30 are not monotone in `C` -- the signature of search defects like the dedup one, not of
  the regularization. Untraced.

## Where that leaves the goal

The penalty-at-every-`prec` goal is NOT established by this route. What is supported, on this
instrument: a posterior mode at a large fixed `C` (100) matched the threshold at 1e-12 on three basis
sizes and the roster, and would extend the penalty's range from 1e-8 to 1e-12 if it holds wider. At
1e-14 the span is the limit, and the candidates are:

1. **A larger span** (`k` beyond `MAX_BRANCHES`): more room for the eigenspace inside a span the
   penalty has bent. Cheap to try -- one constant in `posterior_tensions`.
2. **Posterior for refine and count only**, with bracketing on `sigma_tau`: the stage-1 sweep reads
   many points far from any root, where the posterior buys nothing and its noise may cost.
3. **A posterior that is not a fixed span**: a second, smaller-`tau` solve restricted to a window
   around each root, or the second-order penalty (`notes/tikhonov_derivations.tex`) as the span.
4. **Trace the 1e-12 singles** first: if they are search defects, they are cheap and the 1e-12
   claim firms up regardless of what happens at 1e-14.

