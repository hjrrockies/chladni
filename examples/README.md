# Examples

- **`geometry_tour.ipynb`** — a guided tour of `chladni.geometry`: segments and the arclength
  parameterization, boundaries and orientation, corners with their boundary conditions,
  clearance and branch cuts, metrics against closed forms, domains with holes, boundary
  quadrature, and the primitives the eigensolver layers will consume.

Outputs are committed, so the notebook reads on GitHub without being run. To run it yourself:

```bash
uv pip install -e ".[dev]"
jupyter lab examples/geometry_tour.ipynb
```

`tests/test_examples.py` executes the notebook so it cannot rot silently. It is marked
`slow` and, like the other slow tests here, runs by default; `pytest -m "not slow"` skips it
for a faster inner loop.
