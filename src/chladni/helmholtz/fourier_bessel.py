"""Corner-adapted Fourier-Bessel functions: `J_mu(k*r) * sin(mu*theta)` and its cosine twin.

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

    beta = (branch_angle - angle(t_out)) % (2*pi)                     # in (omega, 2*pi)
    theta  = ((angle((z - origin)/t_out) - beta) % (2*pi)) + beta - 2*pi

`theta = 0` is the outgoing ray, `theta` increases counter-clockwise to `omega` on the incoming
ray, and the 2*pi discontinuity sits exactly on the cut in the exterior wedge.

The obvious `np.angle((z - origin)/t_out) % (2*np.pi)` is wrong, and quietly. A boundary
point that should sit at `theta = 0` comes out of `np.angle` as -1e-17 and wraps to 6.283;
the Dirichlet family then evaluates to 0.11, 0.17, 0.22 on the ray where it must be
identically zero. Anchoring the lift to `beta` is what makes that ray give STRUCTURALLY
exact 0.0 -- `sin(mu*0)` and nothing else.

One asymmetry follows and is worth knowing before writing an assertion against it: the
condition on the `theta = 0` ray holds exactly, the one on `theta = omega` only to about 1e-16
relative, from roundoff in `omega` itself.

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

#: Floor on the Debye amplitude's `w`, at the Airy transition width where the asymptotic
#: stops being valid. Solved, not tuned: it is the value that makes the floored amplitude
#: `-0.5*log(2*pi*mu*w)` equal the global maximum `log(_JMAX_C*(mu+_JMAX_SHIFT)**(-1/3))`
#: exactly at the crossover, so the envelope and the cap MEET there instead of crossing.
_AIRY_FLOOR = np.exp(-2.0 * np.log(_JMAX_C)) / (2 * np.pi)

#: Sharpness of the soft minimum blending the envelope with the cap. 1 would be plain
#: `logaddexp`, costing a factor 2 in the column maximum where the two branches meet;
#: 4 costs 2**(1/4) = 1.19 and keeps every derivative bounded. `min` costs nothing there and
#: puts a kink into the objective MPS parabola-fits, which is the trade being refused.
_BLEND = 4.0

#: mu = 0 (the leading Neumann mode) is not a Debye problem at all -- J_0 is O(1) everywhere
#: and the cap alone is right. This keeps the division finite without a separate branch.
_MU_FLOOR = 1e-12


def evaluation_radius(domain, origin):
    """The largest `|z - origin|` over Omega: the radius that sets a corner family's scale.

    NOT the corner's clearance, which is a different quantity answering a different question.
    `clearance` bounds where the local expansion stays VALID; this bounds where the column is
    EVALUATED. Scaling by the wrong one is a `(r_max/R)**mu` error, and on the L-shape at
    k = 7.3 with 14 degrees per corner that is `2.83**28`: measured column maxima of
    **5.7e+10** against the O(1) the whole scaling design is for.

    Both are fixed geometric constants, so both satisfy point-set independence; only one of
    them makes the columns O(1) over the set the solver actually evaluates on.
    """
    # `Boundary.polyline()` returns (b0, b1, owner); b0 alone covers every vertex, since the
    # loop is closed. The maximum over Omega is attained on the boundary, so this is it.
    zs = np.concatenate([b.polyline()[0] for b in domain.boundaries])
    return float(np.abs(zs - origin).max())


def _log_envelope(mu, x):
    """The per-column log scale `sigma`: the Debye envelope at radius R, smoothly capped.

    `x` is `k*R` with R the family's evaluation radius -- a fixed geometric constant, so
    `sigma` depends on `(column, k)` and on nothing about the evaluation points. That is the
    property that lets a downstream unit-norm rescale compose with this one without
    double-counting.

    Three pieces, and each one is there because a simpler version was measured and failed.

    **The envelope.** `mu*eta` with `eta = w + log(z/(1+w))`, `z = k*R/mu`, `w = sqrt(1-z*z)`:
    the exponent of the Debye asymptotic, which is where all the dynamic range is. It is
    extended by 0 past `z = 1`, and that extension is C1 -- `eta -> 0` and `d(eta)/dz -> 0`
    there, since `d(eta)/dz = 1/z - z/(1+w)` is `0` at `z = 1`.

    **The amplitude, regularized at the Airy width.** The Debye prefactor is
    `-0.5*log(2*pi*mu*w)`, and `w -> 0` at the crossover, so that term diverges and its
    derivative diverges faster. Measured across the crossover at mu = 20, as a second
    difference in k: 8.4e5 unfloored against 6.5 floored. Note that unfloored, the SMOOTH
    blend (8.4e5) is worse than a hard `min` (1.9e3), because `min` clips the divergence
    while `logaddexp` follows it in -- the floor is the load-bearing fix and the soft blend
    is what removes the kink that remains. Replacing `w` by
    `(q**2 + w_floor**4)**0.25` with `q = 1 - z*z` floors it at the Airy transition width,
    where the asymptotic stops being valid anyway. `_AIRY_FLOOR` is chosen so the floored
    amplitude reproduces the global maximum exactly AT the crossover, which is what makes the
    two branches meet rather than merely cross.

    **The cap.** Past `z = 1` the function is oscillatory and R stops being where the maximum
    is, so the global max of `J_mu` takes over. The two are blended by a soft minimum rather
    than by `min`: `min` puts a kink in k, and `sigma(k)` sits inside the objective MPS
    parabola-fits. `_BLEND` sets how sharp the blend is; at 4 the two branches meeting costs
    a factor `2**(1/4) = 1.19` in the column maximum, against `min`'s discontinuous
    derivative.

    Measured, over mu from 0.05 to 320 and k*R from 0.5 to 300: column maxima land in
    [0.802, 1.185], a spread of 1.48x, against raw columns spanning hundreds of orders of
    magnitude over the same grid. The second difference of `sigma` in k is at most 20.8 at
    mu = 2, 3.1 at mu = 20 and 0.5 at mu = 200 -- bounded, and 1e5 times smaller than the
    unregularized form across the crossover.

    Two alternatives were tried and are worse. Normalizing by `log J_mu(k*R)` puts a POLE in
    `sigma(k)` at every zero of `J_mu`. The leading small-argument term alone leaves a
    residual spread of 4.7x at k*R = 10, growing to 3.0e8x at k*R = 100.

    Because it depends on k and R only through their product, `sigma` is EXACTLY invariant
    under `Domain.normalized(by=s)` with `k -> k*s`.
    """
    mu = np.asarray(mu, dtype=float)
    p = _BLEND
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x / np.maximum(mu, _MU_FLOOR)
        q = 1.0 - z * z
        w = np.sqrt(np.maximum(q, 0.0))
        eta = np.where(q > 0.0, w + np.log(z / (1.0 + w)), 0.0)
        floor = _AIRY_FLOOR * (mu + _JMAX_SHIFT) ** (2.0 / 3.0) / np.maximum(mu, _MU_FLOOR)
        # A SMOOTH positive part of q, not `maximum(q, 0)`. Past the crossover q goes
        # negative and `q*q` would start growing again, which drives the amplitude down and
        # the column maximum up -- measured at 27.7 before this line, against 1.4 after.
        # Clamping with `maximum` fixes the size but leaves a jump in the second derivative;
        # this is C-infinity and has the same two limits.
        q_plus = 0.5 * (q + np.sqrt(q * q + floor**4))
        w_eff = (q_plus * q_plus + floor**4) ** 0.25
        a = mu * eta - 0.5 * np.log(2 * np.pi * np.maximum(mu, _MU_FLOOR) * w_eff)
    b = np.log(_JMAX_C * (mu + _JMAX_SHIFT) ** (-1.0 / 3.0))
    return -np.logaddexp(-p * a, -p * b) / p


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

    Note what that includes: the angular factors `A = sin(mu*theta)` and their derivatives.
    `mu` depends only on the corner, so the whole `(n_pts, n_cols)` trig block is computed
    once and reused for every k. Only the Bessel call is per-k, which is the shape lappy's
    own measurement pointed at ("per-lambda work in the underlying bases is purely
    transcendental").
    """

    def __init__(self, fam, pts, normals, bc):
        z = pts - fam.origin
        r = np.abs(z)
        # The corner-local frame: theta = 0 on t_out, increasing counter-clockwise. `beta`,
        # the branch vertices and the winding polygon all live here, so the points have to
        # as well -- winding a local-frame polygon against un-rotated points is a bug that
        # hides completely on any corner whose t_out happens to be 1.
        z_local = z / fam.t_out
        beta = fam.modes.beta
        theta = ((np.angle(z_local) - beta) % (2 * np.pi)) + beta - 2 * np.pi
        wind = None
        if fam.branch_vertices:
            wind = _winding(fam._cut_polygon(z_local), z_local)
            theta = theta - 2 * np.pi * wind
        mu = fam.orders[None, :]
        arg = mu * theta[:, None]
        if fam.modes.kind == "sin":
            ang, dang = np.sin(arg), mu * np.cos(arg)
        else:
            ang, dang = np.cos(arg), -mu * np.sin(arg)
        with np.errstate(divide="ignore", invalid="ignore"):
            # grad(r) and grad(theta) in complex form: d/dx + 1j*d/dy.
            grad_r = np.where(r > 0, z / np.where(r > 0, r, 1.0), 0.0)
            grad_phi = np.where(r > 0, 1j * z / np.where(r > 0, r * r, 1.0), 0.0)
        _set(self, _family=fam, pts=pts, normals=normals, bc=bc, r=r, theta=theta,
             winding=wind, _ang=ang, _dang=dang, _grad_r=grad_r, _grad_phi=grad_phi)

    @property
    def family(self):
        return self._family

    def _bessel(self, k, cols, derivative=True):
        mu = _take(self._family.orders, cols)
        s = log_j(mu[None, :], k * self.r[:, None], derivative)
        # exp(sigma_pointwise - sigma_column): the only place the two scales meet, and the
        # reason both are canonical analytic numbers rather than logs of computed values.
        scale = np.exp(s.sigma - self._family.log_scale(k, cols)[None, :])
        return s, scale

    def _eval(self, k, cols, want_grad):
        s, scale = self._bessel(k, cols, want_grad)
        self._check_accuracy(s.rel_err)
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
        s, _ = self._bessel(check_k(k), cols, derivative=False)
        return s.rel_err.max(axis=0)


class FourierBessel(HelmholtzFamily):
    """The corner-adapted family at one corner.

    `origin` is the vertex, `t_out` the unit tangent of the outgoing edge (the `theta = 0`
    ray), `degrees` the integer degrees `l` whose orders are `modes.orders(degrees)`, and
    `radius` the LARGEST radius at which this family will be evaluated -- see
    `evaluation_radius`, and note that it is not the corner's clearance.
    """

    def __init__(self, origin, t_out, modes, degrees, radius,
                 branch_vertices=(), prec=PREC):
        super().__init__(prec=prec)
        degrees = np.asarray(degrees, dtype=int)
        if degrees.ndim != 1 or len(degrees) == 0:
            raise ValueError(f"degrees must be a non-empty 1-D integer array, got {degrees!r}")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"radius must be finite and positive, got {radius!r}")
        t = complex(t_out)
        _set(self, origin=complex(origin), t_out=t / abs(t), modes=modes,
             degrees=degrees, radius=float(radius),
             branch_vertices=tuple(complex(v) for v in branch_vertices))

    @classmethod
    def at_corner(cls, corner, n=None, degrees=None, radius=None, domain=None, prec=PREC):
        """From a `geometry.Corner`, deriving `nu`, the angular kind and the cut.

        **Pass `domain` when the family will be evaluated over the whole boundary**, which is
        the normal case: the scaling radius is then `evaluation_radius(domain, corner.z)`.
        Without it the fallback is `corner.clearance`, which is correct only for an
        evaluation confined to the corner's own neighbourhood -- on the L-shape the two
        differ by 2.83x in radius and by 5.7e+10 in the resulting column maximum.

        Raises on a hole corner whose family is multivalued; `CornerModes` carries the
        topological reason.
        """
        modes = CornerModes.from_corner(corner)
        if degrees is None:
            if n is None:
                raise ValueError("give either n (how many degrees) or degrees (which ones)")
            degrees = modes.degrees(n)
        if radius is None:
            radius = (corner.clearance if domain is None
                      else evaluation_radius(domain, corner.z))
        return cls(origin=corner.z, t_out=corner.t_out, modes=modes, degrees=degrees,
                   radius=radius, branch_vertices=corner.branch_vertices, prec=prec)

    # --- the columns ------------------------------------------------------------------

    @property
    def n_cols(self):
        return len(self.degrees)

    @cached_property
    def orders(self):
        """`mu` for each column: `(l + offset)*nu`."""
        return self.modes.orders(self.degrees)

    def log_scale(self, k, cols=None):
        return _log_envelope(_take(self.orders, cols), check_k(k) * self.radius)

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

    def _rebuilt(self, origin, t_out, radius, branch_vertices):
        return FourierBessel(origin, t_out, self.modes, self.degrees, radius,
                             branch_vertices, prec=self.prec)

    def translated(self, dz):
        return self._rebuilt(self.origin + dz, self.t_out, self.radius,
                             tuple(v + dz for v in self.branch_vertices))

    def rotated(self, theta, about=0j):
        s = np.exp(1j * theta)
        return self._rebuilt(about + (self.origin - about) * s, self.t_out * s, self.radius,
                             tuple(about + (v - about) * s for v in self.branch_vertices))

    def scaled(self, s, about=0j):
        """The family on the scaled geometry. Eigenvalues follow `k -> k/s`, matching
        `Domain.scaled`; the family itself carries no k, so only lengths change here."""
        return self._rebuilt(about + (self.origin - about) * s, self.t_out * (s / abs(s)),
                             self.radius * abs(s),
                             tuple(about + (v - about) * s for v in self.branch_vertices))

    def with_prec(self, prec):
        return FourierBessel(self.origin, self.t_out, self.modes, self.degrees,
                             self.radius, self.branch_vertices, prec=prec)

    def __add__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.translated(other)
        return super().__add__(other)

    def __mul__(self, s):
        return self.scaled(s)

    def __repr__(self):
        return (f"FourierBessel(origin={self.origin:.6g}, nu={self.modes.nu:.6g}, "
                f"kind={self.modes.kind!r}, n_cols={self.n_cols})")
