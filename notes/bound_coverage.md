# Does `Spectrum.bound` cover the error? (2026-10-07)

*Working note. Script `notes/bound_coverage.py`, data `bound_coverage.jsonl`, tables
`bound_coverage_read.py`. Measured at the `overnight/2026-10-07` branch point (`9d33361`, no `src/`
change since 2026-10-05), macOS/Accelerate, single-threaded BLAS.*

## Question

`notes/delta_res_undercover.py` (2026-10-06) found `bound = min(delta_res, mp_bound)` below the true
error by up to 4e6x on one starved L-shape basis at `prec = 1e-12`. Is that a starved-basis effect, or
does `delta_res` under-cover on adequate bases too? And what should `bound` be?

## Grid

76 solves, 715 levels, against closed-form or 14-digit truth:

| domain | truth | bases (starved -> generous) |
|---|---|---|
| L-shape, [5, 100] | `L_shape_eigs` (14 digits) | FB at the reentrant corner, n = 10, 20, 40, 80; FB at all six corners, n = 8, 14, 24 each |
| 3pi/2 sector, [5, 100] | closed form | FB at the apex, n = 4, 8, 16; offset sources only, n = 40, 80, 160 |
| eq. triangle, [40, 300] | closed form | plane waves, 0.5, 0.8, 1.3 x `n_useful` |
| disk, [5, 80] | closed form | plane waves, 0.5, 0.8, 1.3 x `n_useful` |

each at `prec` 1e-8 (penalty path), 1e-10, 1e-12 and 1e-13 (threshold), default `rtol`.

## Results

* **Moler-Payne covered the error on all 715 levels**: `err / mp_bound` at most 0.64, median 0.024.
* **`delta_res` failed on 129 of 715**, from 1.0x to 1.6e5x under the error. **Every failure by
  more than 2.6x was on a level whose `delta_res > prec*lam`** -- a level the search already
  reported as unmet. Those are the starved bases: the tension floor is the basis's, and the well can
  sit away from the eigenvalue while still looking sharp.
* The other 10 failures are on met levels, at most 2.6x under, with errors of 8-3100 ulps, all
  within `prec` (sector apex FB n = 4, 8 at 1e-10..1e-13; L-shape n = 80 at 1e-13; L-shape all-corner
  n = 14 at 1e-12). They are the float floor: `DELTA_SAFETY = 2` was calibrated on a located-error
  ratio of 0.99-1.27, and these reach 5.2.
* Disagreement between the two bars predicts failure, but not cleanly: `mp/delta_res` is 5.4 at the
  lowest failure and up to 3.2e9 on levels where `delta_res` was right.

Rules tried on the grid (uncovered levels / levels whose bound got looser / met -> unmet flips):

| rule | uncovered | looser | flips |
|---|---|---|---|
| `min(delta_res, mp)` (shipped) | 129 | -- | -- |
| `mp` alone | 0 | 659 (median 5.2x) | 38 |
| `min(mp, max(delta_res, mp/G))`, G = 3 | 0 | 476 | 17 |
| G = 10 | 7 | 248 | 9 |
| **`delta_res` admitted only when `<= prec*lam`** | **10 (all <= 2.6x, met)** | 262 (unmet levels only) | **0** |

`mp` alone is sound but costs the 1e-13 tier on corner domains: on the L-shape with the 80-term
recipe at `prec = 1e-13` it reports 4.4e-12 against true errors of at most 2.2e-14 and would flip 14
of 19 levels to unmet (`bound_coverage_read.py`, the per-solve table). That tier is `douse`'s working
default (`docs/design/douse_requirements.md`).

## What shipped (`c1c3f53`)

The last rule. `MPSEigensolver.certify_levels` sets `Level.delta_res_admitted = delta_res <=
prec*|lam|`, and `Level.bound` reports `mp_bound` alone when it is not admitted. Met/unmet status is
unchanged on every grid level by construction, the bound now covers the error on every unmet level,
and `attained_prec` for a starved solve is now the Moler-Payne figure. Test:
`test_moler_payne.py::test_a_starved_basis_reports_a_bound_that_covers_its_error`. Full suite
2574 passed, 15 skipped, 3 xfailed.

## Not established

* The 10 residual under-covers (<= 2.6x, near the float floor) are not fixed; raising
  `DELTA_SAFETY` would trade them against tightness at 1e-13, which was not measured.
* Neumann and mixed levels have no Moler-Payne bound and still report bare `delta_res`.
* Four domains; no douse polygon (`benchmarks/douse/` is where those are now). The Moler-Payne
  bound's own looseness on polygons -- 1e-6..3e-5 relative on douse shapes where the runs agree to
  1e-8 -- is a separate finding, in the 2026-10-07 session note.

## Found on the way: count errors on an ADEQUATE basis

The grid's count column showed the L-shape's exact double `10 pi^2` returned with multiplicity 1, 3
or 4, unflagged, on the 80-term recipe. Repro and mechanism: `notes/double_10pi2.py`.
