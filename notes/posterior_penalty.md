# A fixed penalty read through posterior tensions clears the cluster grid down to 1e-14

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

## What this does and does not establish

Established, on this instrument: a fixed `tau` near the noise jaw, read through posterior tensions,
counts the cluster grid correctly at 1e-12 and 1e-14 and loses one member at 1e-10 to a dedup defect
that has a named mechanism. The squeeze of `penalty_prec_limit_explained.md` is a property of
reading `sigma_tau`, not of the penalty.

Not established:

* **One `tau`, one family.** 2e-14 on 126-column plane-wave rectangles -- the family the reach
  constants were fitted on. The noise jaw scales with `eps sqrt(n)`, so `tau` should too; unchecked
  across basis sizes, corner bases, curved boundaries.
* **The dedup defect.** Until the bar-swallows-neighbour rule is fixed, the posterior row has a
  known silent loss.
* **Whether the reach rule's calibrated constants transfer.** `COUNT_BELOW_CACHE` and its kin were
  calibrated on threshold tensions; they were not re-derived for posterior ones, and the row
  happened to pass.
* **Linux.** Measured on macOS only -- though a path with no rank decision should be the one that
  does not care.

## Next, if this is pursued

1. Fix the dedup: two converged finds should not merge on the WIDER of their bars when both read
   at the floor (compare depths, or use the narrower bar), with this cell as the regression.
2. `tau = C eps sqrt(n)`, and the same grid across basis sizes and on the hard roster (L-shape,
   GWW, disk), the posterior row against the threshold.
3. If both hold: a `reg="posterior"`-style mode on `MPSPencil` (or the posterior as the objective
   `MPSEigensolver` builds), then `PENALTY_PREC_MIN` falls and truncation can retire -- which also
   retires the Linux rank-flip item in `TODO.md` section 1 at its root.
