"""Plane waves: `cos(k*dot(z, d))` and `sin(k*dot(z, d))` for each unit direction `d`.

lappy has none. `make_default_basis` routes a corner-free domain to boundary fundamental
solutions instead, and "plane wave" appears in that codebase only as the worst-case integrand
used to size a cubature rule. This is a real gap against the MPS literature, where plane
waves are the standard family for a smooth or convex domain, and it is cheap to fill: exact
for any direction, no Bessel function, and `sigma = 0` -- these columns are already O(1) by
construction, which is the cleanest illustration that the scale is a per-family property and
not a global postprocessing step.

Directions sweep `[0, pi)`, not `[0, 2*pi)`. `d` and `-d` give the same `{cos, sin}` pair up
to sign, so a full sweep duplicates every column exactly.

# The conditioning, which has to be said out loud

A plane-wave basis is exponentially ill-conditioned past a threshold that moves out linearly
with `k*R`. Measured on the unit disk with pre-normalized columns, `cond2` of the `2N`-column
matrix:

    k*R \\ N       5      10      20      30      40      60
      5         8.2   1.9e2  1.0e10  1.2e16  1.4e16  1.7e16
     10         1.9   2.0e1   2.1e4  1.6e11  6.9e15  9.6e15
     20         1.8     2.2   2.6e1   1.4e3   1.8e8  2.9e15
     40         1.3     2.0   1.7e1   2.0e2   2.1e2   1.0e6

The numerical rank saturates at the Jacobi-Anger cutoff: measured ranks 33/45/65/95/149 at
`k*R = 2/5/10/20/40` fit `n_useful ~ k*R + 11*(k*R)**(1/3)`, the 11 being an eps_mach
constant. Past that, columns arrive at machine noise. They do not enrich the space -- they
lower the smallest singular value and corrupt the tension the solver is minimizing. The bound
costs a cube root, so `PlaneWave` checks it whenever it knows the enclosing radius.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._core import PREC, PrecisionWarning, _set, check_prec
from ._core import HelmholtzFamily, Sampling, SingularStructure, _take, check_k

#: The Jacobi-Anger cutoff is `ell_max ~ x + C*x**(1/3)`, and C was first fitted to the
#: measured ranks 33/45/65/95/149 at k*R = 2/5/10/20/40 as "about 11". It is not a fitted
#: constant: `(ln(1/eps_mach))**(2/3) = 10.912`. Writing it that way makes the bound
#: precision-aware for free, and the derived law tracks the measured numerical rank to
#: within 2% over k*R in [5, 40] against relative cuts from 1e-13 to 1e-4.


class PlaneWaveSampling(Sampling):
    """A `PlaneWave` bound to points. `dot(z, d)` is the whole k-independent precompute."""

    def __init__(self, fam, pts, normals, bc):
        d = fam.directions[None, :]
        z = pts[:, None]
        s = z.real * d.real + z.imag * d.imag         # (n_pts, n_directions)
        _set(self, _family=fam, pts=pts, normals=normals, bc=bc, _s=s)

    @property
    def family(self):
        return self._family

    def _eval(self, k, cols, want_grad):
        f = self._family
        f.log_scale(k, cols)                          # the conditioning check lives there
        ks = k * self._s
        c, sn = np.cos(ks), np.sin(ks)
        # Column 2*i is cos, column 2*i+1 is sin, so a column index maps to a direction by
        # halving and to a phase by parity. One index space, as everywhere else.
        vals = np.empty((len(self.pts), 2 * len(f.directions)))
        vals[:, 0::2], vals[:, 1::2] = c, sn
        grad = None
        if want_grad:
            kd = k * f.directions[None, :]
            grad = np.empty_like(vals, dtype=complex)
            grad[:, 0::2], grad[:, 1::2] = -kd * sn, kd * c
        return _take(vals, cols), (None if grad is None else _take(grad, cols))

    def accuracy(self, k, cols=None):
        """`cos` and `sin` of a float64 argument: one ulp in the argument, one ulp out. The
        only error here is in `k*dot(z, d)` itself."""
        n = self._family.n_cols if cols is None else len(cols)
        return np.full(n, np.finfo(float).eps)


class PlaneWave(HelmholtzFamily):
    """A cosine and a sine per direction.

    `radius` is the radius of the smallest disk containing Omega, used only for the
    conditioning check. Without it the check cannot run, and warning on a guess would be
    worse than not warning.
    """

    def __init__(self, directions, radius=None, prec=PREC):
        super().__init__(prec=prec)
        d = np.atleast_1d(np.asarray(directions, dtype=complex))
        if d.ndim != 1 or len(d) == 0:
            raise ValueError(f"directions must be a non-empty 1-D array, got {directions!r}")
        if np.any(d == 0):
            raise ValueError("a direction must be a nonzero complex number")
        _set(self, directions=d / np.abs(d),
             radius=None if radius is None else float(radius))

    @classmethod
    def uniform(cls, n, radius=None, prec=PREC):
        """`n` equally spaced directions over `[0, pi)`, i.e. `2*n` columns."""
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        return cls(np.exp(1j * np.pi * np.arange(n) / n), radius=radius, prec=prec)

    @staticmethod
    def n_useful(kR, prec=PREC):
        """`k*R + (ln(1/prec))**(2/3) * (k*R)**(1/3)`: the Jacobi-Anger cutoff at `prec`.

        Beyond about this many directions the columns are numerically dependent on those
        already present, to within `prec`. The bound loosens as `prec` does -- a k*R = 20
        disk supports 63 columns at 1e-4 against 95 at 1e-13 -- which makes this the one
        place in the layer where a looser precision genuinely buys a smaller basis.
        """
        kR = float(kR)
        return kR + np.log(1.0 / check_prec(prec)) ** (2.0 / 3.0) * kR ** (1.0 / 3.0)

    @property
    def n_cols(self):
        return 2 * len(self.directions)

    def log_scale(self, k, cols=None):
        k = check_k(k)
        if self.radius is not None:
            useful = self.n_useful(k * self.radius, self.prec)
            if len(self.directions) > useful:
                warnings.warn(
                    f"{len(self.directions)} plane-wave directions at k*R = "
                    f"{k * self.radius:.3g}, where about {useful:.0f} are numerically "
                    f"independent at prec={self.prec:.0e}. The extra columns arrive at the noise floor: do not "
                    f"enrich the space, they lower the smallest singular value and corrupt "
                    f"the tension.",
                    PrecisionWarning,
                    stacklevel=3,
                )
        n = self.n_cols if cols is None else len(cols)
        return np.zeros(n)

    def singular_exponents(self):
        return SingularStructure(np.full(self.n_cols, np.nan + 1j * np.nan),
                                 np.zeros(self.n_cols))

    def _sample(self, pts, normals, bc):
        return PlaneWaveSampling(self, pts, normals, bc)

    # --- transforms ---------------------------------------------------------------------

    def translated(self, dz):
        """A translation is a phase, not a new direction set -- and a phase shift mixes the
        cosine and sine of the same direction, which is a rotation WITHIN the span. So the
        family is unchanged, and saying so is more honest than returning a copy."""
        return self

    def rotated(self, theta, about=0j):
        return PlaneWave(self.directions * np.exp(1j * theta), self.radius, prec=self.prec)

    def scaled(self, s, about=0j):
        return PlaneWave(self.directions,
                         None if self.radius is None else self.radius * abs(s),
                         prec=self.prec)

    def with_prec(self, prec):
        return PlaneWave(self.directions, self.radius, prec=prec)

    def __repr__(self):
        return f"PlaneWave(n_directions={len(self.directions)}, n_cols={self.n_cols})"
