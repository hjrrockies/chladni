# chladni

**chladni** computes Laplacian eigenvalues and eigenfunctions of planar domains using a
regularized, GSVD-based formulation of the Method of Particular Solutions (MPS).

It is a ground-up rebuild of [`lappy`](../lappy), designed around the lessons learned there:
adaptive geometry, a clean separation between geometry / basis / solver, and a black-box path
that works without hand-tuning.

> **Status: pre-alpha.** The design is documented in [`docs/design`](docs/design); the
> implementation has not been written yet.

## Goals

- **Black-box solves.** Supply a `Domain`, build an `Eigenproblem`, get eigenvalues — no manual
  choice of basis, collocation points, or regularization required for non-pathological domains.
- **Polygons first.** Corner singularities are the hard case and the primary target, but curved
  boundaries are in scope and should not be designed out.
- **User-specified precision.** Every approximation is tied to a working relative precision
  `prec` (default `1e-8`, down to `1e-15`). Tolerances are derived from `prec` rather than
  hard-coded.
- **Solver-in-a-loop.** The real purpose is to be the inner eigensolver for spectral shape
  optimization and inverse spectral problems (see [`douse`](../douse)), so it must be fast,
  robust, and reusable across algorithmically generated domains.

## Architecture

| Layer | What it provides |
| --- | --- |
| `geometry` | Segments, boundaries, and domains with attached boundary conditions; adaptive sampling, corner detection, area / perimeter / diameter, simplicity and non-degeneracy checks. |
| `particular_solutions` | Vectorized families of Helmholtz solutions — corner-adapted fractional-order Fourier–Bessel functions, monopole and multipole fundamental solutions, plane waves — with derivatives. |
| `mps` | The GSVD-regularized MPS eigensolver, plus orthonormal eigenfunctions via Rellich-identity boundary integrals evaluated by quadrature. |

Design notes for each layer live in [`docs/design`](docs/design).

## Installation

Development install, from the repository root:

```bash
uv pip install -e ".[dev]"
```

or with pip:

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Testing

```bash
pytest
```

Tests are ground-truth-driven wherever possible (domains with known spectra: rectangles, disks,
the equilateral triangle, the L-shape). Once `chladni` is trustworthy, it may generate its own
reference values for features without a closed form.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
