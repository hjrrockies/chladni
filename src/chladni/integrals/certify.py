"""A posteriori eigenvalue error bounds: Moler-Payne, from boundary data alone.

Let `u` satisfy `Delta u + lam u = 0` in `Omega` -- every MPS candidate does -- with trace `g` on
the boundary, and `v` the harmonic extension of `g`. Then `w = u - v` vanishes on the boundary and
`Delta w + lam w = -lam v`; expanding `w` in the DIRICHLET eigenfunctions gives
`min_k |lam - lam_k| ||w|| <= lam ||v||`. The maximum principle gives
`||v||_L2 <= sqrt|Omega| ||g||_inf`, and `||w|| >= ||u|| - ||v||`, so with

    eps = sqrt|Omega| max_dOmega |u| / ||u||_L2(Omega)

    min_k |lam - lam_k| <= lam eps / (1 - eps)        (eps < 1).

(lappy `benchmarks/reference/certify.py`, which states it the same way.) Dirichlet only: the
argument needs `w` to vanish on the WHOLE boundary. It says SOME eigenvalue is that close, not
which one -- the count is the search's business.

Three ingredients, each taken conservatively:

* `||u||` from the Rellich identity (`identities.norms`), as an UNDER-estimate: the smallest of
  its values over several origins, less their spread. The spread is a detector of quadrature
  error, not a bound on it (lappy), which is why it is subtracted rather than trusted.
* `max |u|` by dense sampling, graded toward every segment end, repeated at double density and
  inflated by twice the drift between the two (lappy). A SAMPLED maximum: an estimate that tracks
  the full bound to 0.16 digits in lappy's audit, not a guarantee (`boundary_integrals.md`, open
  questions).
* the rounding in `u` itself: `u` is a sum of terms up to `cancellation` times its own interior
  size, so the sampled maximum is raised by `4 eps` times the largest term. Below that floor the
  bound would claim more than float64 computed, so the floor is part of the bound rather than a
  refusal beside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import identities

__all__ = ["MolerPayne", "boundary_sup", "x0_probes", "norm_lower", "moler_payne"]

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class MolerPayne:
    """The bound and its ingredients, per function. `rel` is `eps / (1 - eps)` and `abs` is
    `lam * rel`, both infinite where `eps >= 1`. `reason` is empty when the bound is finite and
    says why when it is not."""

    lam: float
    eps: np.ndarray
    rel: np.ndarray
    abs: np.ndarray
    sup: np.ndarray
    drift: np.ndarray
    norm: np.ndarray
    norm_spread: np.ndarray
    cancellation: np.ndarray
    reason: str = ""


def boundary_sup(evaluate, domain, n_per_seg=400, grade=3.0):
    """`(sup, drift, terms_max)` per function over `domain`'s boundary.

    `evaluate(z)` returns `(values, terms)`, each `(len(z), m)`: the function values and the sum of
    absolute terms behind them. Nodes on each segment are `t = s^grade` mirrored about the
    midpoint, clustering at both ends where corner singularities live, with the corner itself
    excluded. Sampled at `n_per_seg` and `2 n_per_seg`; `drift` is their relative difference.
    """
    def sample(n):
        s = np.linspace(0.0, 1.0, n // 2 + 1)[1:]
        t = np.unique(np.clip(np.concatenate([0.5 * s**grade, 1.0 - 0.5 * s**grade]),
                              1e-14, 1.0 - 1e-14))
        z = np.concatenate([seg.p(t) for b in domain.boundaries for seg in b.segments])
        vals, terms = evaluate(z)
        return np.abs(vals).max(axis=0), np.asarray(terms).max(axis=0)

    lo, t_lo = sample(int(n_per_seg))
    hi, t_hi = sample(2 * int(n_per_seg))
    top = np.maximum(lo, hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = np.where(top > 0, np.abs(hi - lo) / top, 0.0)
    return top, drift, np.maximum(t_lo, t_hi)


def x0_probes(domain, x0, n=3):
    """`x0` and `n` more deterministic origins spread over the bounding box, kept off the box's
    diagonals -- an origin on the bisector of a `3pi/2` corner cancels that corner's contribution
    exactly and would hide its error (lappy)."""
    pts = np.concatenate([b.sample(200) for b in domain.boundaries])
    lo_x, hi_x, lo_y, hi_y = pts.real.min(), pts.real.max(), pts.imag.min(), pts.imag.max()
    w, h = hi_x - lo_x, hi_y - lo_y
    cand = [complex(lo_x + fx * w, lo_y + fy * h)
            for fx, fy in ((0.31, 0.62), (0.71, 0.23), (0.44, 0.87))]
    return [complex(x0)] + cand[:n]


def norm_lower(cd, origins):
    """`(norm, spread)`: a conservative under-estimate of each `||u_i||`, the smallest Rellich
    diagonal over `origins` less the spread between them (both squared norms)."""
    D = np.array([np.diag(identities.gram(cd, x0)) for x0 in origins])
    lo, spread = D.min(axis=0), D.max(axis=0) - D.min(axis=0)
    return np.sqrt(np.clip(lo - spread, 0.0, None)), np.sqrt(spread)


def moler_payne(cd, evaluate, domain, origins, n_per_seg=400):
    """The Moler-Payne bound for each function in `cd` (Cauchy data on an integral rule) whose
    values `evaluate(z) -> (values, terms)` gives at arbitrary boundary points.

    Refused -- infinite, with a reason -- where the boundary is not all Dirichlet, or the norm's
    under-estimate is zero.
    """
    lam = float(cd.lam)
    m = cd.m
    nan = np.full(m, np.nan)
    if np.any(cd.quad.bc != 0.0):
        inf = np.full(m, np.inf)
        return MolerPayne(lam, inf, inf, inf, nan, nan, nan, nan, nan,
                          "Moler-Payne needs a Dirichlet condition on the whole boundary")
    norm, spread = norm_lower(cd, origins)
    sup, drift, terms = boundary_sup(evaluate, domain, n_per_seg)
    area = float(domain.area)
    with np.errstate(divide="ignore", invalid="ignore"):
        cancel = np.where(norm > 0, terms / (norm / np.sqrt(area)), np.inf)
        sup_eff = sup * (1.0 + 2.0 * drift) + 4.0 * _EPS * terms
        eps = np.where(norm > 0, np.sqrt(area) * sup_eff / norm, np.inf)
        rel = np.where(eps < 1.0, eps / (1.0 - eps), np.inf)
    reason = "" if np.all(np.isfinite(rel)) else (
        "eps >= 1 or a zero norm: the candidate is not close enough to an eigenfunction for "
        "the inclusion to say anything")
    return MolerPayne(lam, eps, rel, lam * rel, sup, drift, norm, spread, cancel, reason)
