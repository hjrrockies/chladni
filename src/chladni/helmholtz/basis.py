"""`Basis`: several families as one, and the column bookkeeping that goes with it.

A `Basis` is itself a `HelmholtzFamily`, so a solver never has to ask whether it is holding
one family or ten.

The whole of the interesting content here is `cols`. lappy's `MultiBasis._dispatch_cols`
documents "sorted ascending" as an unenforced precondition, filters each sub-basis by range
and then `hstack`s the pieces in SUB-BASIS order -- so an unsorted or interleaved `cols`
silently returns columns in a different order than `__call__`'s docstring promises. Nothing
raises. A caller doing symmetry projection gets a wrong answer with no signal.

Here the dispatch records, for each family, both which of its own columns are wanted and
WHERE they go in the output, and scatters. Shuffled, reversed and repeated indices all come
back in the order asked for, and the test suite passes all three.

The other thing the dispatch buys is the saving `cols` exists for: a family with no requested
column is never EVALUATED, so a corner block asking for its own five columns does not pay for
a two-hundred-source layer's per-k Bessel pass.

Note what that does not skip. `at(pts)` builds every family's `Sampling`, because that is the
k-independent precompute and it happens once for the life of the solve. `cols` restricts the
per-k transcendental work, which is where the cost is -- lappy's own measurement of the same
split says the per-lambda work in a basis is "purely transcendental".
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from .._core import _set
from ._core import HelmholtzFamily, Sampling, SingularStructure, _as_cols, check_k


class BasisSampling(Sampling):
    """One `Sampling` per family, over the same points, plus the scatter."""

    def __init__(self, fam, pts, normals, bc):
        _set(self, _family=fam, pts=pts, normals=normals, bc=bc,
             _parts=tuple(f._sample(pts, normals, bc) for f in fam.families))

    @property
    def family(self):
        return self._family

    def _eval(self, k, cols, want_grad):
        f = self._family
        n = f.n_cols if cols is None else len(cols)
        vals = np.empty((len(self.pts), n))
        grad = np.empty((len(self.pts), n), dtype=complex) if want_grad else None
        for i, local, dest in f._dispatch(cols):
            v, g = self._parts[i]._eval(k, local, want_grad)
            vals[:, dest] = v
            if want_grad:
                grad[:, dest] = g
        return vals, grad

    def accuracy(self, k, cols=None):
        k = check_k(k)
        f = self._family
        cols = _as_cols(cols)
        out = np.empty(f.n_cols if cols is None else len(cols))
        for i, local, dest in f._dispatch(cols):
            out[dest] = self._parts[i].accuracy(k, local)
        return out


class Basis(HelmholtzFamily):
    """The union of several families, in order.

    `families` is the tuple; `column_family` labels every column with its family's index.
    `basis[i]` is the i-th FAMILY, not the i-th column -- the same choice lappy made, and the
    only sensible one given `values(...)[:, j]` already reads as the j-th column.
    """

    def __init__(self, families):
        families = tuple(families)
        if not families:
            raise ValueError("a Basis needs at least one family")
        for f in families:
            if not isinstance(f, HelmholtzFamily):
                raise TypeError(f"not a HelmholtzFamily: {f!r}")
        # min(prec) over the parts, as a composite Boundary or Domain does one layer down: a
        # composite is only as precise as its least precise part and should say so.
        super().__init__(prec=min(f.prec for f in families))
        _set(self, families=families)

    @classmethod
    def join(cls, *parts):
        """Flatten and concatenate. `a + b + c` is one `Basis` of three families, never a
        nest of two -- lappy's `MultiBasis` flattens for the same reason, and the reason is
        that a nested composite makes `column_family` mean two different things."""
        flat = []
        for p in parts:
            flat.extend(p.families if isinstance(p, Basis) else [p])
        return cls(flat)

    def __add__(self, other):
        return Basis.join(self, other)

    def __getitem__(self, i):
        return self.families[i]

    def __iter__(self):
        return iter(self.families)

    # --- the column map -------------------------------------------------------------------

    @cached_property
    def _offsets(self):
        return np.cumsum([0] + [len(f) for f in self.families])

    @cached_property
    def column_family(self):
        return np.repeat(np.arange(len(self.families)),
                         [len(f) for f in self.families])

    @property
    def n_cols(self):
        return int(self._offsets[-1])

    def _dispatch(self, cols):
        """`(family_index, local_cols, destination_positions)` for each family that has any
        requested column.

        `destination_positions` is what makes an arbitrary `cols` order safe: results are
        scattered back to where the caller asked for them rather than concatenated in family
        order. Families with nothing requested are skipped entirely and never sampled.
        """
        cols = _as_cols(cols)
        if cols is None:
            for i, f in enumerate(self.families):
                lo = int(self._offsets[i])
                yield i, None, np.arange(lo, lo + len(f))
            return
        owner = self.column_family[cols]
        for i in range(len(self.families)):
            dest = np.nonzero(owner == i)[0]
            if len(dest):
                yield i, cols[dest] - int(self._offsets[i]), dest

    # --- the family surface -----------------------------------------------------------------

    def _sample(self, pts, normals, bc):
        return BasisSampling(self, pts, normals, bc)

    def log_scale(self, k, cols=None):
        k = check_k(k)
        cols = _as_cols(cols)
        out = np.empty(self.n_cols if cols is None else len(cols))
        for i, local, dest in self._dispatch(cols):
            out[dest] = self.families[i].log_scale(k, local)
        return out

    def singular_exponents(self):
        parts = [f.singular_exponents() for f in self.families]
        return SingularStructure(
            np.concatenate([p.origin for p in parts]),
            np.concatenate([p.exponent for p in parts]),
        )

    # --- transforms ----------------------------------------------------------------------------

    def _map(self, name, *args, **kwargs):
        return Basis([getattr(f, name)(*args, **kwargs) for f in self.families])

    def translated(self, dz):
        return self._map("translated", dz)

    def rotated(self, theta, about=0j):
        return self._map("rotated", theta, about)

    def scaled(self, s, about=0j):
        return self._map("scaled", s, about)

    def with_prec(self, prec):
        return self._map("with_prec", prec)

    def __repr__(self):
        inner = ", ".join(repr(f) for f in self.families)
        return f"Basis([{inner}])"
