"""The Rellich identity and the Gram matrix, as boundary integrals of Cauchy data.

For `u` with `Delta u + lam u = 0` in `Omega` -- every MPS candidate, with NO condition on the
boundary -- and `r = x - x0` for ANY origin `x0`, `r_n = r . N`, `r_t = r . T`:

    ||u||^2 = (1/2 lam) oint r_n (u_n^2 - u_t^2) ds  +  1/2 oint r_n u^2 ds
              + (1/2 lam) oint r_t (2 u_t u_n) ds,

derived and checked in `docs/design/boundary_integrals.md` (3e-15 against an exact interior
rule, origin-free). It is lappy's "master identity" (`lappy/docs/rellich.md:27`) with nothing
dropped: lappy kept only the terms that survive for an EXACT eigenfunction, which makes its norm
of a candidate wrong at the size of the boundary residual. Kept whole, it needs no boundary
condition and no knowledge of which edge is which.

The Gram matrix of `m` candidates at one `lam` is the polarized form. Candidates at DIFFERENT
`lam` are never combined here -- Green's identity would divide by `lam_i - lam_j`.

Every formula is a fixed combination of one primitive, `weighted_integral` (lappy's design): a
consumer building its own boundary functional does the same.
"""
from __future__ import annotations

import numpy as np

__all__ = ["KERNELS", "LOWDIN_GUARD", "weighted_integral", "gram", "norms", "lowdin"]

#: The four bilinear Cauchy-data kernels. `'cr'` is the symmetrized cross term.
KERNELS = ("uv", "NN", "TT", "cr")


def weighted_integral(cd, kernel, weight):
    """`A[i, j] = oint weight(s) K(u_i, u_j) ds` over `cd`'s nodes, `(m, m)`.

    `kernel` is `'uv'` (`u v`), `'NN'` (`u_n v_n`), `'TT'` (`u_t v_t`) or `'cr'`
    (`u_t v_n + u_n v_t`); `weight` is a real array at the nodes (`r . N` for Rellich, `V . n` for
    Hadamard)."""
    w = (cd.quad.wts * np.asarray(weight, dtype=float))[:, None]
    if kernel == "uv":
        return (cd.u * w).T @ cd.u
    if kernel == "NN":
        return (cd.u_n * w).T @ cd.u_n
    if kernel == "TT":
        return (cd.u_t * w).T @ cd.u_t
    if kernel == "cr":
        return (cd.u_t * w).T @ cd.u_n + (cd.u_n * w).T @ cd.u_t
    raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")


def _radial(cd, x0):
    r = cd.quad.pts - complex(x0)
    n, t = cd.quad.normals, cd.quad.tangents
    return r.real * n.real + r.imag * n.imag, r.real * t.real + r.imag * t.imag


def gram(cd, x0):
    """`G[i, j] = int_Omega u_i u_j`, `(m, m)`, from the full Rellich identity about `x0`.

    The identity holds for every `x0`, so the spread of `gram` over several origins is a
    DETECTOR of quadrature error -- not an error bar, since it also carries the eigenvalue's own
    error through `1/lam` (lappy). `x0` is required: the rule's default lives on `IntegralQuad`.
    """
    lam = float(cd.lam)
    r_n, r_t = _radial(cd, x0)
    G = (weighted_integral(cd, "NN", r_n) - weighted_integral(cd, "TT", r_n)
         + weighted_integral(cd, "cr", r_t)) / (2.0 * lam)
    G += 0.5 * weighted_integral(cd, "uv", r_n)
    return 0.5 * (G + G.T)


def norms(cd, x0):
    """`||u_i||_{L2(Omega)}` for each function, from the diagonal of `gram`. A negative diagonal
    -- possible only from quadrature or cancellation error -- comes back as NaN, never as a
    silently clipped zero."""
    d = np.diag(gram(cd, x0))
    with np.errstate(invalid="ignore"):
        return np.where(d >= 0, np.sqrt(np.abs(d)), np.nan)


#: The smallest eigenvalue ratio `w_min / w_max` of a cluster's Gram matrix that `lowdin` will
#: invert. Below it the columns are nearly dependent -- the cluster has the wrong multiplicity --
#: and `w^(-1/2)` would amplify a near-zero (or rounding-negative) eigenvalue through every column.
#: lappy's `ttol`.
LOWDIN_GUARD = 1e-3


def lowdin(G, guard=LOWDIN_GUARD):
    """The symmetric (Löwdin) orthonormalizing transform `D = Q diag(w^(-1/2)) Q^T` of a small Gram
    matrix `G = Q diag(w) Q^T`: for coefficient columns `C` whose Gram is `G`, `C @ D` is
    orthonormal.

    Symmetric rather than Cholesky because it treats the columns alike, so no member of a cluster
    is privileged by its position. Only the cluster's SPAN is meaningful; which orthonormal basis
    of it comes back is not promised (lappy states the same contract).

    Raises when `w_min / w_max < guard`: the cluster is deficient, and orthonormalizing it would
    manufacture a direction from rounding.
    """
    G = np.asarray(G, dtype=float)
    w, Q = np.linalg.eigh(0.5 * (G + G.T))
    if not (w[0] > 0 and w[0] / w[-1] >= guard):
        raise ValueError(
            f"the Gram matrix is deficient (eigenvalue ratio {w[0] / w[-1]:.2e} < {guard:.0e}): "
            f"the cluster's columns are nearly dependent, so its multiplicity is likely wrong")
    return (Q * w ** -0.5) @ Q.T

