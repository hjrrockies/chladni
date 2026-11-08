"""Where the boundary collocation points go: per segment, from the basis as well as the shape.

lappy's `pts_per_seg(domain, fb_basis)` existed because a split by length alone kept failing:
a Fourier-Bessel column satisfies the boundary condition exactly on its corner's two sides, so
only the OTHER sides constrain it, and a short side opposite a heavily loaded corner starves it.
The failure is a spurious one -- coefficient combinations that vanish at the few points there
and not between them -- so the tension sits on a flat floor near machine epsilon and the search
either declines or, worse, reads a well into it. On the isosceles triangle with apex 0.05 and
100 terms per corner, the base got 24 of 900 points (`notes/collocation_allocation.md`).

The decision lives here, one layer above geometry, because which segments a basis needs points
on is not a geometric question (`geometry.Corner`'s docstring keeps it out of that layer). Each
family states its own need (`HelmholtzFamily.collocation_demand`); this turns the sum into
counts and leaves the length split untouched wherever it already covers it.
"""
from __future__ import annotations

import numpy as np

from ..geometry._adaptive import allocate

__all__ = ["bdry_counts"]


def bdry_counts(domain, basis, n, *, k=None):
    """Boundary points per segment of `domain` (flattened over its boundaries) for `basis`.

    Each segment gets the largest of three counts:
    - its share of `n` by length -- the previous default, and the result wherever the other
      two do not bind, which is every shape the planner was measured on;
    - the basis's demand there (`basis.collocation_demand`), rounded up;
    - with a finite wavenumber `k`, `k * length / pi`: two points per wavelength, so every
      side resolves the oscillation of whatever lives on it.
    The total can therefore exceed `n`, by exactly what the shape and basis need.
    """
    lengths = np.array([s.length for b in domain.boundaries for s in b.segments])
    counts = allocate(int(n), lengths)
    demand = np.asarray(basis.collocation_demand(domain), dtype=float)
    # rounding noise must not cost a point: 400.0000000001 is 400
    counts = np.maximum(counts, np.ceil(demand - 1e-9).astype(int))
    if k is not None and np.isfinite(k):
        counts = np.maximum(counts, np.ceil(float(k) * lengths / np.pi - 1e-9).astype(int))
    return counts
