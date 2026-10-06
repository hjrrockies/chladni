# Examples

- **`geometry_tour.ipynb`** — a guided tour of `chladni.geometry`: segments and the arclength
  parameterization, boundaries and orientation, corners with their boundary conditions,
  clearance and branch cuts, metrics against closed forms, domains with holes, boundary
  quadrature, and the primitives the eigensolver layers will consume.
- **`helmholtz_tour.ipynb`** — `chladni.helmholtz`: factored Bessel values, corner modes, and
  the particular-solution families as columns of a matrix.
- **`lshape_tour.ipynb`** — one problem through every layer: the L-shape from domain to basis,
  collocation, normalization, the solver, the tension curve, a hand minimization, convergence in
  the number of corner terms, the certified search, and Rellich-orthonormal eigenfunctions. It
  follows the workflow of `lappy`'s notebooks step for step, with a table mapping each `lappy`
  call to its `chladni` counterpart.

Outputs are committed, so the notebooks read on GitHub without being run. To run it yourself:

```bash
uv pip install -e ".[dev]"
jupyter lab examples/geometry_tour.ipynb
```

`tests/test_examples.py` executes the notebook so it cannot rot silently. It is marked
`slow`, which this project deselects by default to keep the inner loop fast, so run it
explicitly:

```bash
pytest -m slow
```
