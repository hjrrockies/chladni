"""Fundamental solutions: `Y_m(k*r)` multipoles at source points outside Omega.

One object holds an ARRAY of origins sharing one multipole order. That is the other half of
the rule stated in `fourier_bessel`: an atom is a set of origins sharing the same structural
parameters. A corner has its own wedge, so `FourierBessel` is one corner; two hundred sources
on an offset curve share everything, and making them two hundred objects would turn one
`(m, n)` `yv` call into two hundred `(m, 1)` ones.

# Sources must be exterior

lappy's docstring at `bases.py:889` is the sharpest thing in that codebase and is worth
citing rather than re-deriving. A source inside Omega is a POLE inside Omega, so that column
does not solve the Helmholtz equation there, and three things break at once and all of them
silently: the MPS premise, since the basis is no longer made of particular solutions; the
Moler-Payne certificate, whose hypothesis is that u solves the PDE exactly in Omega; and the
tension, which gets BETTER rather than worse, because an interior pole lets the fit absorb
boundary error that a legitimate basis cannot. Measured on a chevron with a 307-degree
reentrant corner: 24 of 240 sources inside took the tension background from 5e-2 to 3e-7,
which made every point pass the acceptance test, produced a spurious eigenvalue and hid a
real one.

Nor can it be fixed by a parameter recommendation. At distance `r` from a reentrant corner
whose exterior wedge is `beta`, the offset crosses to the far side once `d` exceeds about
`r*tan(beta/2)`, and lappy measured the count going as `npts*d` -- 16 offenders at d = 0.4 and
0 at d = 0.02 with 60 points per segment, but 536 and 26 at 2000 points per segment. A small
enough offset is safe at a fixed density and no offset is safe at every density. So this
raises rather than dropping the offenders.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.special import gammaln

from .._core import PREC, _set
from ._bessel import log_y
from ._core import HelmholtzFamily, Sampling, SingularStructure, _take, check_k

#: Sharpness of the soft maximum joining the small- and large-argument envelopes of `Y_m`.
#: Same role as `_BLEND` in fourier_bessel, and the same reason: `max` would put a kink into
#: `sigma(k)`, which sits inside the objective MPS parabola-fits.
_BLEND = 4.0

#: Published Gopal-Trefethen clustering rate. lappy's `by_corners` carries the right formula
#: with `sigma=1`. Measured at n = 25 and L = 1, the innermost pole sits at 1.1e-07 with
#: sigma = 4 and at 1.8e-02 with sigma = 1 -- five orders of magnitude of reach toward the
#: corner, which is the entire point of clustering.
LIGHTNING_SIGMA = 4.0


class SourcesInsideDomain(ValueError):
    """A source point landed inside Omega. See the module docstring for why this is fatal."""


def _y_envelope(m, x):
    """The per-column log scale for `Y_m` at standoff `x = k*rho`.

    `Y_m` blows up at small argument the way `J_mu` underflows at large order, so it needs a
    scale for the same reason and of the same kind: a function of `(column, k)` and of a
    fixed geometric length, never of the evaluation points.

    Two regimes, joined by a soft maximum since the envelope decreases monotonically:
    `Gamma(m)(2/x)**m/pi` while `x < m`, and `sqrt(2/(pi*x))` after. `m = 0` has a log
    singularity rather than a power one, and `log(1 + 2/x)` is the small-argument magnitude
    written so that it also decays past `x ~ 2` and lets the oscillatory branch take over --
    `log(2/x)` alone keeps growing like `log log x` and under-scales `Y_0` by 30x at x = 100.
    """
    m = np.asarray(m, dtype=float)
    p = _BLEND
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.where(
            m >= 1,
            gammaln(np.maximum(m, 1.0)) - m * np.log(x / 2) - np.log(np.pi),
            np.log(2 / np.pi) + np.log(np.log1p(2 / x)),
        )
        large = 0.5 * np.log(2 / (np.pi * x))
    return np.logaddexp(p * small, p * large) / p


class FundamentalSampling(Sampling):
    """A `Fundamental` bound to points. The radii, the angular factors and the two pieces of
    the chain rule are all independent of k and computed once."""

    def __init__(self, fam, pts, normals, bc):
        z = pts[:, None] - fam.origins[None, :]        # (n_pts, n_sources)
        r = np.abs(z)
        th = np.angle(z)
        col = fam._src_idx
        m = fam._m[None, :]
        arg = m * th[:, col]
        sin_col = fam._is_sin[None, :]
        ang = np.where(sin_col, np.sin(arg), np.cos(arg))
        dang = np.where(sin_col, m * np.cos(arg), -m * np.sin(arg))
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_r = np.where(r > 0, z / np.where(r > 0, r, 1.0), 0.0)
            grad_th = np.where(r > 0, 1j * z / np.where(r > 0, r * r, 1.0), 0.0)
        _set(self, _family=fam, pts=pts, normals=normals, bc=bc,
             r=r, _ang=ang, _dang=dang,
             _grad_r=grad_r[:, col], _grad_th=grad_th[:, col])

    @property
    def family(self):
        return self._family

    def _bessel(self, k, cols, derivative=True):
        f = self._family
        m = _take(f._m, cols)
        r = self.r[:, _take(f._src_idx, cols)]
        s = log_y(m[None, :], k * r, derivative)
        scale = np.exp(s.sigma - f.log_scale(k, cols)[None, :])
        return s, scale

    def _eval(self, k, cols, want_grad):
        s, scale = self._bessel(k, cols, want_grad)
        ang, dang = _take(self._ang, cols), _take(self._dang, cols)
        rad = s.S * scale
        vals = rad * ang
        if not want_grad:
            return vals, None
        drad = k * s.Sp * scale
        grad = ((drad * ang) * _take(self._grad_r, cols)
                + (rad * dang) * _take(self._grad_th, cols))
        return vals, grad

    def accuracy(self, k, cols=None):
        return self._bessel(check_k(k), cols, derivative=False)[0].rel_err.max(axis=0)


class Fundamental(HelmholtzFamily):
    """Multipole fundamental solutions at a set of exterior source points.

    Each origin contributes `2*order - 1` columns: `Y_0`, then `Y_m cos(m*theta)` and
    `Y_m sin(m*theta)` for `m = 1 .. order-1`. `radius` is each source's standoff from Omega,
    which is what sets the column scale -- a geometric constant, per the point-set
    independence rule, and scalar or per-source.
    """

    def __init__(self, origins, order=1, radius=None, prec=PREC):
        super().__init__(prec=prec)
        origins = np.atleast_1d(np.asarray(origins, dtype=complex))
        order = int(order)
        if origins.ndim != 1 or len(origins) == 0:
            raise ValueError(f"origins must be a non-empty 1-D array, got {origins!r}")
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        if radius is None:
            raise ValueError(
                "radius is required: it is the source's standoff from Omega and it sets the "
                "column scale. A placement builder (offset, lightning) knows it; a hand-built "
                "family has to state it."
            )
        radius = np.broadcast_to(np.asarray(radius, dtype=float), origins.shape).copy()
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("radius must be finite and positive")
        _set(self, origins=origins, order=order, radius=radius)

    # --- the column layout ---------------------------------------------------------------

    @cached_property
    def _m(self):
        """The multipole order of each column."""
        per = np.repeat(np.arange(self.order), 2)[1:]        # 0, 1, 1, 2, 2, ...
        return np.tile(per, len(self.origins))

    @cached_property
    def _is_sin(self):
        per = np.zeros(2 * self.order - 1, dtype=bool)
        per[2::2] = True                                      # cos, sin, cos, sin, ...
        return np.tile(per, len(self.origins))

    @cached_property
    def _src_idx(self):
        return np.repeat(np.arange(len(self.origins)), 2 * self.order - 1)

    @property
    def n_cols(self):
        return len(self.origins) * (2 * self.order - 1)

    def log_scale(self, k, cols=None):
        k = check_k(k)
        return _y_envelope(_take(self._m, cols),
                           k * self.radius[_take(self._src_idx, cols)])

    def singular_exponents(self):
        """Regular everywhere in Omega: the pole is outside, by construction and by the
        exception in `check_exterior`."""
        return SingularStructure(np.full(self.n_cols, np.nan + 1j * np.nan),
                                 np.zeros(self.n_cols))

    def _sample(self, pts, normals, bc):
        return FundamentalSampling(self, pts, normals, bc)

    # --- exteriority ---------------------------------------------------------------------

    def check_exterior(self, domain):
        """Raise if any source is inside Omega. See the module docstring."""
        inside = np.asarray(domain.contains(self.origins), dtype=bool)
        if inside.any():
            raise SourcesInsideDomain(
                f"{int(inside.sum())} of {len(self.origins)} sources are inside the domain, "
                f"e.g. {self.origins[inside][0]}. Such a column has a pole in Omega, so it is "
                f"not a particular solution there: the MPS premise fails, any Moler-Payne "
                f"bound computed from it is void, and the tension gets BETTER rather than "
                f"worse, which is why this is an exception and not a filter. Reduce the "
                f"offset, or taper it to zero at the reentrant corner responsible."
            )
        return self

    # --- placement -------------------------------------------------------------------------

    @classmethod
    def offset(cls, quad, distance, order=1, domain=None, prec=PREC):
        """Sources one `distance` along the outward normal from each quadrature node.

        Pass `domain` to have the result checked; without it the caller owns the check, and
        `check_exterior` is the same test on demand.
        """
        distance = float(distance)
        if distance <= 0:
            raise ValueError(f"offset distance must be positive, got {distance}")
        fam = cls(np.asarray(quad.pts, dtype=complex)
                  + distance * np.asarray(quad.normals, dtype=complex),
                  order=order, radius=distance, prec=prec)
        return fam if domain is None else fam.check_exterior(domain)

    @classmethod
    def lightning(cls, corner, n, sigma=LIGHTNING_SIGMA, L=None, taper=0.0,
                  order=1, prec=PREC):
        """Gopal-Trefethen root-exponentially clustered poles at a corner.

        `s_j = L*exp(-sigma*(sqrt(n) - sqrt(j)))` for `j = 1..n`, placed along the corner's
        exterior direction. Convergence is root-exponential, `~exp(-C*sqrt(n))`, which is
        what a cluster buys over more terms of a regular expansion.

        `taper` drops that fraction from the OUTER end. The outer poles are where the cluster
        would otherwise overlap an ambient source layer and put near-duplicate columns into
        the pencil, which destabilizes the rank truncation. How much to drop is a
        coordination question between this cluster and that layer, so it is a parameter here
        and a decision for the basis planner, which is deferred. lappy has no taper at all.

        `radius` is the standoff from Omega, not the distance to the corner: on the exterior
        bisector the two differ by `sin` of half the exterior angle.
        """
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if not 0.0 <= taper < 1.0:
            raise ValueError(f"taper must lie in [0, 1), got {taper}")
        if not np.isfinite(corner.branch_angle):
            raise ValueError(
                f"corner at {corner.z} has no exterior direction, so a lightning cluster has "
                f"nowhere to go; geometry could not find a cut that stays out of Omega"
            )
        L = float(corner.clearance if L is None else L)
        keep = n - int(round(taper * n))
        j = np.arange(1, keep + 1)
        s = L * np.exp(-float(sigma) * (np.sqrt(n) - np.sqrt(j)))
        half_exterior = 0.5 * (2 * np.pi - corner.omega)
        return cls(corner.z + s * np.exp(1j * corner.branch_angle), order=order,
                   radius=s * np.sin(half_exterior), prec=prec)

    # --- transforms ----------------------------------------------------------------------------

    def translated(self, dz):
        return Fundamental(self.origins + dz, self.order, self.radius, prec=self.prec)

    def rotated(self, theta, about=0j):
        s = np.exp(1j * theta)
        return Fundamental(about + (self.origins - about) * s, self.order, self.radius,
                           prec=self.prec)

    def scaled(self, s, about=0j):
        return Fundamental(about + (self.origins - about) * s, self.order,
                           self.radius * abs(s), prec=self.prec)

    def with_prec(self, prec):
        return Fundamental(self.origins, self.order, self.radius, prec=prec)

    def __repr__(self):
        return (f"Fundamental(n_sources={len(self.origins)}, order={self.order}, "
                f"n_cols={self.n_cols})")
