"""Cauchy data at quadrature nodes: the one input every boundary integral reads.

**Evaluate first, sandwich never** (lappy, `docs/eigfun_integrals.md`). The basis is contracted
with the coefficient vectors AT the nodes, and only the resulting `(n_nodes, m)` values are ever
integrated. The alternative -- a basis-sized Gram `C^T G C` -- multiplies each entry's independent
rounding error through the coefficients on both sides; lappy measured norms off by ~1e-4 on a
basis of condition 1e16. And no node set serves a basis-sized Gram: corner rules lose 2-4 orders
on columns centred at OTHER corners.

`terms` is carried for the CANCELLATION check (`docs/conventions.md`): `u` is a sum of terms far
larger than itself wherever the coefficients cancel, so `eps * terms` is the floor of anything
computed from it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CauchyData", "cauchy_data"]


@dataclass(frozen=True)
class CauchyData:
    """`u`, `u_n`, `u_t` -- value, outward normal and tangential derivative -- of `m` functions at
    a `Quadrature`'s nodes, each `(n_nodes, m)`. `lam` is the spectral parameter the functions are
    Helmholtz-exact for. `terms` is `(n_nodes, m)`, the sum of `|c_j psi_j|` behind each value, or
    None when the data did not come from a coefficient vector."""

    quad: object
    lam: float
    u: np.ndarray
    u_n: np.ndarray
    u_t: np.ndarray
    terms: object = None

    @property
    def m(self):
        return self.u.shape[1]

    def cancellation(self, norms, area):
        """Per function, `max |terms| / (||u|| / sqrt|Omega|)`: how far the largest term behind
        a boundary value exceeds the function's typical INTERIOR size, so `eps` times it is the
        floor of anything computed from these values.

        The interior size, not the boundary one, and that is the whole point: a good Dirichlet
        candidate is nearly zero on the boundary BY DESIGN, so dividing by its boundary rms reads
        1e15 at an exact eigenvalue and says nothing. `norms` come from `identities.norms`.
        """
        if self.terms is None:
            raise ValueError("no terms: this CauchyData was not built from coefficients")
        rms = np.asarray(norms, dtype=float) / np.sqrt(float(area))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rms > 0, self.terms.max(axis=0) / rms, np.inf)

    @classmethod
    def from_functions(cls, quad, lam, u, grad):
        """From callables `u(z)` and `grad(z) = du_dx + 1j du_dy`, vectorized over complex `z` --
        closed-form references, or a consumer's own function. Each may return `(n,)` or
        `(n, m)`."""
        z = np.asarray(quad.pts)
        vals = np.asarray(u(z))
        g = np.asarray(grad(z))
        if vals.ndim == 1:
            vals, g = vals[:, None], g[:, None]
        n, t = quad.normals[:, None], quad.tangents[:, None]
        u_n = g.real * n.real + g.imag * n.imag
        u_t = g.real * t.real + g.imag * t.imag
        return cls(quad, float(lam), vals.real.astype(float), u_n, u_t)


def cauchy_data(family, coef, lam, quad):
    """Cauchy data of `family @ coef` at `quad`'s nodes.

    `family` is a `helmholtz` family or `Basis`; `coef` is `(n_cols,)` or `(n_cols, m)` in the
    FAMILY's coordinates -- for an MPS solution, `MPSPencil.family_coef(lam, coef(lam))`. One
    transcendental pass gives both the values and the gradient.
    """
    c = np.asarray(coef, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    k = float(np.sqrt(lam))
    V, G = family.at(quad).values_and_grad(k)
    n, t = quad.normals[:, None], quad.tangents[:, None]
    dn = G.real * n.real + G.imag * n.imag
    dt = G.real * t.real + G.imag * t.imag
    return CauchyData(quad, float(lam), V @ c, dn @ c, dt @ c, np.abs(V) @ np.abs(c))
