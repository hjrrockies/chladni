# Overview
Validating the solver pipeline of `chladni`, while we are building it, is of the highest importance. This document gives a description of the pipeline benchmarks that need to be built.

# Solver architecture
The core eigensolver of `chladni` is the GSVD-based Method of Particular Solutions. The pipeline is broken down into the following stages:

1. Geometry: domains via boundary, with boundary conditions
2. Eigenproblem specification: max value of $\lambda$, target relative precision
3. Particular solution basis selection
4. Collocation point selection
    a. If orthonormal eigenfunctions are needed (default `True`), boundary quadrature rule generation
5. MPS problem instance from basis, collocation points, target precision, regularization settings
6. Eigenvalues via tension minimization
7. Orthonormal eigenfunctions via Rellich identity on quadrature

## 1. Geometry
While `chladni` is intended to handle eigenproblems on arbitrary "non-pathological" domains with Dirichlet, Neumann, Zaremba, or Robin boundary conditions, the first priority is to develop a solver that achieves high accuracy for polygonal domains with Dirichlet boundary conditions.

### Domains with closed-form solutions
The following domains and domain families have closed-form solutions for eigenvalues and eigenfunctions:
- Rectangles
- Equilateral triangle
- Isosceles right triangle
- Hemiequilateral (30-60-90) triangle
- Disk
- Disk sectors

These should always be included in our benchmarks. Since rectangles and disk sectors are parameterized families, we should include several cases that span various degeneracy and near-degeneracy cases (rectangles), as well as singular corner cases (sectors).

### Important domains without closed-form solutions
The following domains and domain families are considered important test cases:
- L-shape domain
- Asymmetric L-shapes
- Ellipses
- Bunimovich stadium
- Mushroom shape
- H-shaped domain
- Cut squares
- Regular N-gons
- Random N-gons
- Isosceles triangles
- Chevrons

## 2. Eigenproblem specification
The most important knob here is the target precision. All of our benchmarks should be run with precisions in the family
`[1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]`. The definition of a successful solve (eigenvalue accuracy, legal merging of
clustered eigenvalues, etc) is dependent on the target precision. The choice of particular solution basis, and everything downstream from that choice, should be adapted to target precision.

Generally, `chaldni`'s design prioritizes computing the "low" spectrum of a domain.

## 3. Particular solution basis selection
The role of basis selection in our benchmarks is twofold:

1. For each benchmark domain and target precision, identify a basis which supports computing the eigenvalues accurate to that precision, up to a domain-specific value of $\lambda_{max}$. Generally, $\lambda_{max}$ should be large enough to include at least the first 10 Dirichlet eigenvalues of the domain. These "frozen" bases are used to validate the downstream solver performance.
2. We also need benchmarks for basis selection *heuristics*, but these will follow later.

## 4. Collocation point selection
TODO

## 5. MPS problem instance
Our benchmarks need to include tests which cover how the rank truncation tolerance for regularization affects the solve.

## 6. Tension minimization for eigenvalues
Right now, this is the most important area. We need to have benchmarks based on high-quality "frozen" bases which provide up to 14 digits of precision for the target eigenvalues. We need to test on families of domains which have closed form and tabulated values.
We need to test on domain families which cover degenerate and nearly-degenerate eigenvalues, on both sides of the legal merge policy for a given target precision.

## 7. Orthonormal eigenfunctions
TODO

# As built: deltas from this spec

Written 2026-11-05. The sections above are the plan of 2026-09-17, kept as written; this records
which of its open items are now built, and two claims that never matched the code.

* **7. Orthonormal eigenfunctions is built** (`boundary_integrals.md`, M4):
  `MPSEigensolver.eigenfunctions(level)` returns the level's eigenfunctions Löwdin-orthonormalized
  on a Rellich boundary rule, with `gram_residual` checked at further origins. Its tests compare
  against interior oracles (`tests/test_orthonormal.py`, `tests/test_rellich.py`).
* **3 and 4, basis and collocation selection, are built as the planner** (`chladni.plan`,
  `Eigenproblem.plan`): a ladder of `Recipe` rungs (`LADDER`) sizes the basis and the collocation
  for a Dirichlet polygon at `prec`, and `Eigenproblem.solve` climbs it until a rung meets `prec`
  with the count audit clean (`douse_requirements.md`, Phase 5). "Heuristics will follow later"
  is superseded.
* **The benchmark that exists is `benchmarks/douse/`**: the shapes of the `lappy` failure
  catalogue (`douse_failure_cases.md`) as exact vertices (`cases.json`), reference spectra by
  three `chladni` bases cross-checked against FEM (`references.json`), and a scoreboard
  (`scoreboard.md`). The black-box run against it is `notes/phase5_benchmark.md`. Frozen bases
  for closed-form and tabulated families live in `tests/hard_roster.py` and `tests/rect_suite.py`.
* **No 30-60-90 closed form.** Section 1 lists the hemiequilateral triangle among the domains with
  closed-form solutions; `chladni.reference` has none. The closed forms shipped are the rectangle,
  disk, sector, equilateral triangle, isosceles right triangle and polyomino (`reference.md`).
* **No "orthonormal eigenfunctions needed" flag.** Stage 4a's "default `True`" names a switch that
  does not exist. Eigenfunctions are formed on demand (`eigenfunctions`, `hadamard`); only the
  Moler-Payne certification runs by default, and `MPSEigensolver(certify=False)` turns it off.
