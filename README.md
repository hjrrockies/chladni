# chladni

[![tests](https://github.com/hjrrockies/chladni/actions/workflows/tests.yml/badge.svg)](https://github.com/hjrrockies/chladni/actions/workflows/tests.yml)

**chladni** computes Laplacian eigenvalues and eigenfunctions of planar domains using a
regularized, GSVD-based formulation of the Method of Particular Solutions (MPS).

It is a ground-up rebuild of `lappy`, designed around the lessons learned there: adaptive
geometry, a clean separation between geometry / basis / solver, and a black-box path that
works without hand-tuning.

> **Status: in development.** The geometry layer, the Helmholtz particular-solution families,
> the MPS pencil, the eigenvalue search, the basis planner and Rellich-identity orthonormalization
> are implemented and tested. The black-box path, `Eigenproblem(dom).solve(n, prec)`, plans its
> own basis for Dirichlet polygons and returns a `Spectrum` with a certified bound per level; it
> is the eigensolver `douse` runs its shape optimization on. Curved and multiply-connected
> domains, and Neumann boundaries, are supported by the geometry and basis layers but not yet by
> the planner.

## A first look

```python
import chladni.geometry as G

d = G.L_shape()
d.area, d.perimeter, d.inradius     # 3.0, 8.0, 0.5857864376 (= 2 - sqrt(2))

[c.omega / 3.14159 for c in d.corners]   # five 0.5 pi corners and one reentrant 1.5 pi

q = d.bdry_quad(400)                # boundary quadrature: nodes, weights, provenance
q.wts.sum()                         # 8.0, the perimeter
(q.radial(0j) * q.wts).sum()        # 6.0 = 2 * area, by the divergence theorem

d.plot_diagnostic()                 # colour-coded segments, annotated corners
```

The black-box path, which plans the basis itself:

```python
from chladni.eigenproblem import Eigenproblem

spec = Eigenproblem(G.L_shape()).solve(4, prec=1e-10)
spec.lam, spec.mult      # levels and multiplicities, the level at 4 returned whole
spec.bound / spec.lam    # the certified relative bound per level
spec.flags               # () when every level met prec and the Weyl count audited ok
spec.plan                # frozen and picklable: realize it on a nearby shape, solve warm
```

And the same eigenvalues on a basis supplied by hand, for bespoke work:

```python
import numpy as np
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSEigensolver

d = G.disk(1.0)
basis = PlaneWave.uniform(42, radius=d.diameter)
s = MPSEigensolver(d, basis=basis, n_bdry=3 * len(basis), n_int=len(basis),
                   rng=np.random.default_rng(0))

spec = s.solve_interval(5.0, 40.0)
spec.lam        # [ 5.783186, 14.681971, 26.374616, 30.471262]  -- j_{m,n}^2
spec.mult       # [1, 2, 2, 1]  -- the sin/cos pairs, READ rather than inferred
spec.bound      # [2.9e-08, 8.2e-09, 6.1e-09, 1.2e-08]  -- attained, per level
spec.achieved   # True: every level met the requested prec

s.solve(4)      # [5.783186, 14.681971, 14.681971, 26.374616] -- unfolded, ascending
```

`spec.mult` is the part an array of eigenvalues cannot carry: a computed multiplicity of 2 and
two levels a whisker apart are different claims. If a level cannot be reached to the requested
`prec`, `spec.unmet` says which and `achieved` is False — the solver reports the shortfall
rather than returning a confident wrong answer.

**[`examples/geometry_tour.ipynb`](examples/geometry_tour.ipynb)** tours the whole geometry
layer with figures — segments and arclength parameterization, corners and their boundary
conditions, branch cuts, domains with holes, boundary quadrature and its convergence.

**[`examples/helmholtz_tour.ipynb`](examples/helmholtz_tour.ipynb)** tours the layer above it —
where `scipy.special.jv` gives up and what is done about it, the corner exponent table
including mixed boundary conditions, branch cuts drawn over the field they act on, source
placement, and assembling a boundary matrix.

Both are committed with outputs, so they read without being run, and both assert their own
numbers, so the test suite runs them end to end.

## Goals

- **Black-box solves.** Supply a `Domain`, build an `Eigenproblem`, get eigenvalues — no manual
  choice of basis, collocation points, or regularization required for non-pathological domains.
- **Polygons first.** Corner singularities are the hard case and the primary target, but curved
  boundaries are in scope and should not be designed out.
- **User-specified precision.** Every approximation is tied to a working relative precision
  `prec` (default `1e-8`, down to `1e-15`). Tolerances are derived from `prec` rather than
  hard-coded, and a domain reports when its own geometry cannot support what was asked.
- **Solver-in-a-loop.** The real purpose is to be the inner eigensolver for spectral shape
  optimization and inverse spectral problems, so it must be fast, robust, and reusable across
  algorithmically generated domains.

## Architecture

| Layer | Status | What it provides |
| --- | --- | --- |
| `geometry` | **implemented** | Segments with arclength parameterization; boundaries, orientation and frames; domains with holes; corners carrying angle, boundary conditions, clearance and branch cuts; boundary quadrature with weights and provenance. |
| `helmholtz` | **implemented** | Corner-adapted Fourier-Bessel families with branch cuts and Zaremba (mixed-BC) corners; `Y_m` multipoles at exterior sources, offset layers and lightning clusters; plane waves; a composite `Basis`. A scaled Bessel kernel that keeps full relative accuracy past the point where `scipy.special.jv` underflows to zero, and an analytic per-column scale that leaves the columns O(1). |
| `plotting` | **implemented** | Domains, boundary conditions, normals and tangents, and a colour-coded diagnostic view. Kept in its own layer so `geometry` never imports matplotlib. |
| `mps` | **implemented** | Collocation matrix assembly, column equilibration, index-truncated SVD and column-pivoted-QR regularization, the `gsvd4py` call, the tension and eigenfunction coefficients. `MPSPencil` is domain-free; `MPSEigensolver` owns the domain; both speak `lam`, with the `k` seam at the `helmholtz` boundary. The L-shape reaches 1.7e-12 at its first eigenvalue. The eigenvalue search and multiplicity estimation are built -- `bracket`, `refine`, `multiplicity` and `search`, reached through `MPSEigensolver.solve_interval`, which returns a `Spectrum` of levels with multiplicities and a per-level bound. `MPSEigensolver.eigenfunctions(level)` returns a level's eigenfunctions L2-orthonormalized through the Rellich identity. |
| `integrals` | **implemented** | Corner-adapted boundary quadrature rules, Cauchy data, Rellich norms and Gram matrices, Löwdin orthonormalization, and the Moler-Payne bound that certifies each level. |
| `reference` | **implemented** | Closed-form spectra (rectangle, disk, sector, triangles, polyominoes) and tabulated ones, each carrying its accuracy as data. |
| `asymp`, `bounds` | **implemented** | Weyl's law with the corner correction, the count audit, Faber-Krahn and the other eigenvalue bounds that size search windows. |
| `plan`, `eigenproblem` | **implemented** (Dirichlet polygons) | The basis planner: a ladder of recipes climbed until every requested level meets `prec`, frozen as a picklable `Plan`; `Eigenproblem.solve(n, prec, plan=, near=)`, with warm starts on a nearby shape. |

## Installation

```bash
uv pip install -e ".[dev]"
```

or, with pip:

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. Use the lighter `".[test]"` extra if you only want to run the suite.

## Testing

```bash
pytest
```

Slow tests — notebook execution and a scaling guard — are deselected by default to keep the
inner loop near 1.5 s. CI runs everything:

```bash
pytest -m ""
```

Tests are ground-truth-driven wherever possible: closed forms for polygons, elliptic integrals
for curved arclength, and vector identities such as the divergence theorem as self-checks.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
