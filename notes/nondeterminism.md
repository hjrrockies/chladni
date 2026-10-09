# The "nondeterminism" at T1.2 is sensitivity, not nondeterminism (2026-10-09)

*Working note. Script `notes/nondeterminism.py` (modes `rate`, `grid`, `window`, `order`, `sens`);
raw data in `notes/nondeterminism_rate.json`. macOS/Accelerate, at `b481a39` (the (C2) fixes are in;
nothing in this note changes code).*

## Question

Three identical runs of `make_references.py T1.2` (2026-10-07) gave run C a count of 2 (flagged),
2 and 1 at douse's T1.2 near-double: two eigenvalues ~9e-10 apart at 68.8536737, 1.3e-11 relative,
so outside the 1e-12 merge licence. FEM counts 2. Was that BLAS nondeterminism, as guessed (memory
alignment, the Linux rank-flip family)?

## Answer: the solver is deterministic, and the search is chaotic in its inputs

| experiment | result |
|---|---|
| **1a.** The full solve, fixed window, 20 fresh processes + 20 in one process, `VECLIB_MAXIMUM_THREADS=1` and default | **80 of 80 identical**: count 1, one evaluation fingerprint (268 evaluations, bitwise equal tensions, same order) |
| **1b.** Tensions on a fixed 50-point grid, 10 fresh processes | **bitwise identical** at all 50 points |
| **E1.** `make_references.window` from FEM, 5 times in one process | **5 distinct windows**: `lo` spread 2.3e-14, `hi` spread 4.0e-13. ARPACK (`eigsh`) starts from a random vector |
| **E3.** Runs A, B, C in sequence in one process, as `make_references.one` does | C's count unchanged (1). No state leaks between solves |
| **E2.** The count at the pair against perturbations of ONE window end | below |

E2, the fixed window `(10.787616554650096, 183.79944350430173)` with one end scaled by `1 + rel`:

| end | rel | count | | end | rel | count |
|---|---|---|---|---|---|---|
| lo | -1e-9 | 2 | | hi | -1e-9 | **3** |
| lo | -1e-12 | 2 (flagged) | | hi | -1e-12 | 2 |
| lo | -1e-14 | 2 | | hi | -1e-14 | **3** |
| lo | -4e-16 | 2 | | hi | -4e-16 | **1** |
| lo | +4e-16 | **1** | | hi | +4e-16 | **1** |
| lo | +1e-14 | 2 | | hi | +1e-14 | 2 |
| lo | +1e-12 | **1** | | hi | +1e-12 | 2 (flagged) |
| lo | +1e-9 | **1** | | hi | +1e-9 | 2 |

**7 of 16 wrong, none count-flagged.** A one-ulp change to an endpoint 58 or 115 units away
decides whether the second eigenvalue is found. The count-1 runs locate one member only. The
count-3 runs locate one member twice, at roots 4e-12 apart (6e-14 relative): a duplicate the
dedup did not merge, although it is inside the licence.

So the 2026-10-07 spread came from the window, which `make_references` derives from ARPACK FEM
eigenvalues that differ in the last digits every run. **"Nondeterminism" was the wrong word.**
The search is deterministic given its inputs and discontinuous in them.

## What this means

* **This pair is not floor-limited.** The members are 9e-10 apart against a floor of ~1e-12 at a
  slope of ~0.05, so `d / (eps/c)` is about 45. A search at `prec = 1e-12` should resolve it
  every time. Missing it, or doubling it, is a search defect (stage 1/2 locating and the dedup),
  not a basis limit. It has not been traced yet.
* **It is what douse will see.** An optimizer's iterates change every input slightly each step.
  A count that depends on ulps of the window is a count that flickers along an optimization path:
  T2.6's and T2.7's "near-double lost a copy" rows in the failure catalogue are the same
  signature in lappy.
* **The Linux CI flips may be the same thing.** OpenBLAS and Accelerate differ in the last bits;
  under a chaotic search path that is enough to change a count. This would explain "4 in 5, scope-
  and state-dependent" without any nondeterminism in either library. Not tested: no Linux here.
* **Item 1's planned fixes do not apply.** Aligned LAPACK copies would change nothing; the numbers
  are already reproducible.

## Not established

* One pair, one basis, one `prec`. Whether other near-degenerate cases have the same sensitivity
  is not measured. The 16-point sweep is a sample, not a rate.
* The mechanism inside stages 1-2 (why the second well is missed, why the duplicate survives) is
  untraced.
