"""Corner-adapted Fourier-Bessel functions: `J_mu(k*r) * sin(mu*phi)` and its cosine twin.

One object is ONE corner -- one origin, one wedge, one branch cut, an array of degrees.
Batching across corners happens in `Basis`.

That is the opposite of lappy, whose `FourierBesselBasis` spans every corner behind a flat
`alphak_vec` and an `_src_idx`. Two reasons to split. lappy's compound form silently drops
`orders == 0` sources in the constructor and then has to re-match sources back to
`domain.corners` by `np.isclose` on complex positions to recover what it threw away
(bases.py:421, 551-576). And the performance argument for batching does not apply: the corner
count is small, at most about twenty, and one `jv` call per corner over an `(m, M_i)` block
does the same work as one call over `(m, n)`. Fundamental solutions are genuinely different --
200 sources at order 0 must not become 200 calls -- so `Fundamental` holds an array of
origins. The rule, stated once: an atom is a set of origins sharing the same structural
parameters.

# The angle lift

This is the load-bearing line of the module.

    beta = (branch_angle - angle(t_out)) % (2*pi)                     # in (theta, 2*pi)
    phi  = ((angle((z - origin)/t_out) - beta) % (2*pi)) + beta - 2*pi

`phi = 0` is the outgoing ray, `phi` increases counter-clockwise to `theta` on the incoming
ray, and the 2*pi discontinuity sits exactly on the cut in the exterior wedge.

The obvious `np.angle((z - origin)/t_out) % (2*np.pi)` is wrong, and quietly. A boundary
point that should sit at `phi = 0` comes out of `np.angle` as -1e-17 and wraps to 6.283;
the Dirichlet family then evaluates to 0.11, 0.17, 0.22 on the ray where it must be
identically zero. Anchoring the lift to `beta` is what makes that ray give STRUCTURALLY
exact 0.0 -- `sin(mu*0)` and nothing else.

One asymmetry follows and is worth knowing before writing an assertion against it: the
condition on the `phi = 0` ray holds exactly, the one on `phi = theta` only to about 1e-16
relative, from roundoff in `theta` itself.

Holes need no sign flip. `Domain` reverses them to clockwise, which puts Omega on the left
of travel on every boundary, so `-1j*T` is outward everywhere and one convention covers both.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from .._core import PREC, _set
from ._bessel import log_j
from ._core import HelmholtzFamily, Sampling, SingularStructure, _take, check_k
from .corners import CornerModes

#: The global maximum of `J_mu` over all arguments, as `_JMAX_C * (mu + _JMAX_SHIFT)**(-1/3)`.
#: The 0.6748 is the Airy constant; the 0.3 shift is what makes the same expression right at
#: small order too. Measured against the true maximum: the ratio stays in [0.934, 1.000] over
#: mu from 0 to 352, worst at mu ~ 0.25 and better than 0.99 for mu >= 64.
_JMAX_C = 0.6748
_JMAX_SHIFT = 0.3


def _log_envelope(mu, x):
    """The per-column log scale `sigma`: the Debye envelope at radius R, smoothly capped.

    `x` is `k*R` with R the corner's clearance -- a fixed geometric constant, so `sigma`
    depends on `(column, k)` and on nothing about the evaluation points. That is the property
    that lets a downstream unit-norm rescale compose with this one without double-counting.

    `logaddexp` rather than `min`, and the difference is not cosmetic: `min` puts a kink in
    k, and `sigma(k)` sits inside the objective MPS parabola-fits. Measured column maxima
    with this rule, over mu <= 352, k*R <= 300 and theta in {pi/8, 3*pi/2}: [0.75, 1.35].

    Two alternatives were tried and are worse. Normalizing by `log J_mu(k*R)` puts a POLE in
    `sigma(k)` at every zero of `J_mu`. The leading small-argument term alone leaves a
    residual spread of 4.7x at k*R = 10, growing to 3.0e8x at k*R = 100.

    Because it depends on k and R only through their product, `sigma` is EXACTLY invariant
    under `Domain.normalized(by=s)` with `k -> k*s`.
    """
    mu = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x / mu
        w = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        a = mu * (w + np.log(z / (1.0 + w))) - 0.5 * np.log(2 * np.pi * mu * w)
        # Past k*R = mu the function is oscillatory and R stops being where the maximum is;
        # the cap takes over. a -> +inf continuously as z -> 1, so this is not a seam.
        a = np.where((mu > 0) & (z < 1.0), a, np.inf)
    b = np.log(_JMAX_C * (mu + _JMAX_SHIFT) ** (-1.0 / 3.0))
    return -np.logaddexp(-a, -b)


def _winding(poly, z):
    """Winding number of the closed polygon `poly` about each point of `z`.

    Positive for a counter-clockwise loop. Vectorized over (points, edges) -- lappy's
    `_points_in_polygon` loops over edges in Python, which is fine at plan frequency and not
    at evaluation frequency.
    """
    a = poly[None, :]
    b = np.roll(poly, -1)[None, :]
    zz = z[:, None]
    left = ((b.real - a.real) * (zz.imag - a.imag)
            - (zz.real - a.real) * (b.imag - a.imag))
    up = (a.imag <= zz.imag) & (b.imag > zz.imag) & (left > 0)
    down = (a.imag > zz.imag) & (b.imag <= zz.imag) & (left < 0)
    return up.sum(axis=1) - down.sum(axis=1)


class FourierBesselSampling(Sampling):
    """A `FourierBessel` bound to points. Everything here is independent of k.

    Note what that includes: the angular factors `A = sin(mu*phi)` and their derivatives.
    `mu` depends only on the corner, so the whole `(n_pts, n_cols)` trig block is computed
    once and reused for every k. Only the Bessel call is per-k, which is the shape lappy's
    own measurement pointed at ("per-lambda work in the underlying bases is purely
    transcendental").
    """

    def __init__(self, fam, pts, normals, bc):
        z = pts - fam.origin
        r = np.abs(z)
        # The corner-local frame: phi = 0 on t_out, increasing counter-clockwise. `beta`,
        # the branch vertices and the winding polygon all live here, so the points have to
        # as well -- winding a local-frame polygon against un-rotated points is a bug that
        # hides completely on any corner whose t_out happens to be 1.
        z_local = z / fam.t_out
        beta = fam.modes.beta
        phi = ((np.angle(z_local) - beta) % (2 * np.pi)) + beta - 2 * np.pi
        wind = None
        if fam.branch_vertices:
            wind = _winding(fam._cut_polygon(z_local), z_local)
            phi = phi - 2 * np.pi * wind
        mu = fam.orders[None, :]
        arg = mu * phi[:, None]
        if fam.modes.kind == "sin":
            ang, dang = np.sin(arg), mu * np.cos(arg)
        else:
            ang, dang = np.cos(arg), -mu * np.sin(arg)
        with np.errstate(divide="ignore", invalid="ignore"):
            # grad(r) and grad(phi) in complex form: d/dx + 1j*d/dy.
            grad_r = np.where(r > 0, z / np.where(r > 0, r, 1.0), 0.0)
            grad_phi = np.where(r > 0, 1j * z / np.where(r > 0, r * r, 1.0), 0.0)
        _set(self, _family=fam, pts=pts, normals=normals, bc=bc, r=r, phi=phi,
             winding=wind, _ang=ang, _dang=dang, _grad_r=grad_r, _grad_phi=grad_phi)

    @property
    def family(self):
        return self._family

    def _bessel(self, k, cols):
        mu = _take(self._family.orders, cols)
        s = log_j(mu[None, :], k * self.r[:, None])
        # exp(sigma_pointwise - sigma_column): the only place the two scales meet, and the
        # reason both are canonical analytic numbers rather than logs of computed values.
        scale = np.exp(s.sigma - self._family.log_scale(k, cols)[None, :])
        return s, scale

    def _eval(self, k, cols, want_grad):
        s, scale = self._bessel(k, cols)
        ang, dang = _take(self._ang, cols), _take(self._dang, cols)
        rad = s.S * scale
        vals = rad * ang
        if not want_grad:
            return vals, None
        drad = k * s.Sp * scale
        grad = (drad * ang) * self._grad_r[:, None] + (rad * dang) * self._grad_phi[:, None]
        return vals, grad

    def accuracy(self, k, cols=None):
        """Per-column relative-error bound, reduced over the points. Free: it comes out of
        the series loop, which already knows its own largest term."""
        s, _ = self._bessel(check_k(k), cols)
        return s.rel_err.max(axis=0)


class FourierBessel(HelmholtzFamily):
    """The corner-adapted family at one corner.

    `origin` is the vertex, `t_out` the unit tangent of the outgoing edge (the `phi = 0`
    ray), `clearance` the radius that sets the column scale, and `degrees` the integer
    degrees `l` whose orders are `modes.orders(degrees)`.
    """

    def __init__(self, origin, t_out, modes, degrees, clearance,
                 branch_vertices=(), prec=PREC):
        super().__init__(prec=prec)
        degrees = np.asarray(degrees, dtype=int)
        if degrees.ndim != 1 or len(degrees) == 0:
            raise ValueError(f"degrees must be a non-empty 1-D integer array, got {degrees!r}")
        if not np.isfinite(clearance) or clearance <= 0:
            raise ValueError(f"clearance must be finite and positive, got {clearance!r}")
        t = complex(t_out)
        _set(self, origin=complex(origin), t_out=t / abs(t), modes=modes,
             degrees=degrees, clearance=float(clearance),
             branch_vertices=tuple(complex(v) for v in branch_vertices))

    @classmethod
    def at_corner(cls, corner, n=None, degrees=None, prec=PREC):
        """From a `geometry.Corner`, deriving `nu`, the angular kind and the cut.

        Raises on a hole corner whose family is multivalued -- see `CornerModes`, where the
        topological reason lives.
        """
        modes = CornerModes.from_corner(corner)
        if degrees is None:
            if n is None:
                raise ValueError("give either n (how many degrees) or degrees (which ones)")
            degrees = modes.degrees(n)
        return cls(origin=corner.z, t_out=corner.t_out, modes=modes, degrees=degrees,
                   clearance=corner.clearance, branch_vertices=corner.branch_vertices,
                   prec=prec)

    # --- the columns ------------------------------------------------------------------

    @property
    def n_cols(self):
        return len(self.degrees)

    @cached_property
    def orders(self):
        """`mu` for each column: `(l + offset)*nu`."""
        return self.modes.orders(self.degrees)

    def log_scale(self, k, cols=None):
        return _log_envelope(_take(self.orders, cols), check_k(k) * self.clearance)

    def singular_exponents(self):
        return SingularStructure(np.full(self.n_cols, self.origin), self.orders.copy())

    def _sample(self, pts, normals, bc):
        return FourierBesselSampling(self, pts, normals, bc)

    # --- the branch cut ------------------------------------------------------------------

    def _cut_polygon(self, z_local):
        """The closed loop whose winding number carries the polyline cut's extra sheets.

        Traverse the cut outward -- corner, intermediate vertices, then far along the final
        ray -- and return along the straight ray at `beta`. In the complement of the two, the
        true angle and the ray-anchored lift differ by `2*pi` times the winding of that loop,
        which is zero far away where the two agree and steps by one across either curve.

        A winding number rather than a point-in-polygon test, because a wall-following cut on
        a spiral can wrap more than once and an even-odd test cannot count that. lappy's
        version precomputes a single +-1 sign, which is right only for one wrap.
        """
        v = np.array([(w - self.origin) / self.t_out for w in self.branch_vertices],
                     dtype=complex)
        far = 10.0 * (np.max(np.abs(z_local), initial=0.0) + np.max(np.abs(v), initial=0.0) + 1.0)
        ray = far * np.exp(1j * self.modes.beta)
        return np.concatenate([[0j], v, [v[-1] + ray, ray]])

    # --- transforms ------------------------------------------------------------------------

    def _rebuilt(self, origin, t_out, clearance, branch_vertices):
        return FourierBessel(origin, t_out, self.modes, self.degrees, clearance,
                             branch_vertices, prec=self.prec)

    def translated(self, dz):
        return self._rebuilt(self.origin + dz, self.t_out, self.clearance,
                             tuple(v + dz for v in self.branch_vertices))

    def rotated(self, theta, about=0j):
        s = np.exp(1j * theta)
        return self._rebuilt(about + (self.origin - about) * s, self.t_out * s, self.clearance,
                             tuple(about + (v - about) * s for v in self.branch_vertices))

    def scaled(self, s, about=0j):
        """The family on the scaled geometry. Eigenvalues follow `k -> k/s`, matching
        `Domain.scaled`; the family itself carries no k, so only lengths change here."""
        return self._rebuilt(about + (self.origin - about) * s, self.t_out * (s / abs(s)),
                             self.clearance * abs(s),
                             tuple(about + (v - about) * s for v in self.branch_vertices))

    def with_prec(self, prec):
        return FourierBessel(self.origin, self.t_out, self.modes, self.degrees,
                             self.clearance, self.branch_vertices, prec=prec)

    def __add__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.translated(other)
        return super().__add__(other)

    def __mul__(self, s):
        return self.scaled(s)

    def __repr__(self):
        return (f"FourierBessel(origin={self.origin:.6g}, nu={self.modes.nu:.6g}, "
                f"kind={self.modes.kind!r}, n_cols={self.n_cols})")
