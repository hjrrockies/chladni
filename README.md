# chladni

[![tests](https://github.com/hjrrockies/chladni/actions/workflows/tests.yml/badge.svg)](https://github.com/hjrrockies/chladni/actions/workflows/tests.yml)

**chladni** computes Laplacian eigenvalues and eigenfunctions of planar domains using a
regularized, GSVD-based formulation of the Method of Particular Solutions (MPS).

It is a ground-up rebuild of `lappy`, designed around the lessons learned there: adaptive
geometry, a clean separation between geometry / basis / solver, and a black-box path that
works without hand-tuning.

> **Status: in development.** The geometry layer, the Helmholtz particular-solution families
> and the plotting are implemented and tested. The eigensolver is not written yet, so nothing
> here solves an eigenvalue problem so far.

## A first look

```python
import chladni.geometry as G

d = G.L_shape()
d.area, d.perimeter, d.inradius     # 3.0, 8.0, 0.5857864376 (= 2 - sqrt(2))

[c.theta / 3.14159 for c in d.corners]   # five 0.5 pi corners and one reentrant 1.5 pi

q = d.bdry_quad(400)                # boundary quadrature: nodes, weights, provenance
q.wts.sum()                         # 8.0, the perimeter
(q.radial(0j) * q.wts).sum()        # 6.0 = 2 * area, by the divergence theorem

d.plot_diagnostic()                 # colour-coded segments, annotated corners
```

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
| `mps` | planned | The GSVD-regularized MPS eigensolver, plus orthonormal eigenfunctions via Rellich-identity boundary integrals evaluated by quadrature. |

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
