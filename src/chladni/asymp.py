"""Weyl asymptotics: how many eigenvalues there should be, and roughly where.

Forward (`lam -> count`) and inverted (`k -> lam_k`) forms of the two- and three-term laws,
plus two diagnostics that compare a computed spectrum against them.

Like `bounds`, each function takes either a `Domain` or the raw scalars.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["weyl_count", "weyl_est", "weyl_count_poly", "weyl_est_poly", "weyl_deficit",
           "weyl_count_check", "corner_correction", "expected_count"]


def _resolve(domain, area, perim, bc_type, need_perim=True):
    if domain is not None:
        area, perim, bc_type = float(domain.area), float(domain.perimeter), domain.bc_type
    if area is None:
        raise ValueError("pass either a Domain or area=")
    if need_perim and perim is None:
        raise ValueError("pass either a Domain or perim=")
    return float(area), (None if perim is None else float(perim)), bc_type


def _sign(bc_type):
    """`-1` for Dirichlet, `+1` for Neumann: the direction of the perimeter term.

    Dirichlet SUBTRACTS -- clamping the boundary removes modes. Note that `weyl_est` flips
    this again, because inverting the quadratic moves the perimeter term to the other side;
    that double flip is the easiest thing here to get wrong.
    """
    if bc_type == "dir":
        return -1.0
    if bc_type == "neu":
        return 1.0
    if bc_type in ("rob", "mixed"):
        raise NotImplementedError(
            f"Weyl asymptotics are not implemented for bc_type={bc_type!r}; there is no "
            f"single perimeter coefficient for a boundary that is not uniformly one or the "
            f"other"
        )
    raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")


def _angles(domain, angles):
    """Interior angles, from `corners` rather than from an isinstance check.

    lappy gates this on `isinstance(domain, Polygon)` and refuses anything else. Every
    `chladni` Domain reports `corners` carrying `omega`, so the corner correction applies to
    `cut_square` and `mushroom` too -- domains with corners that are not polygons.
    """
    if angles is not None:
        return np.asarray(angles, dtype=float)
    if domain is None:
        raise ValueError("pass either a Domain or angles=")
    return np.array([c.omega for c in domain.corners], dtype=float)


def corner_correction(angles):
    """`sum (pi**2 - omega**2) / (24*pi*omega)`. Independent of `lam`."""
    a = np.asarray(angles, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.sum((np.pi**2 - a**2) / (24 * np.pi * a)))


def weyl_count(lam, domain=None, *, area=None, perim=None, bc_type="dir"):
    """Two-term count `N(lam) ~ (A*lam -/+ P*sqrt(lam)) / (4*pi)`. Forward."""
    area, perim, bc_type = _resolve(domain, area, perim, bc_type)
    s = _sign(bc_type)
    lam = np.asarray(lam, dtype=float)
    return (area * lam + s * perim * np.sqrt(np.maximum(lam, 0.0))) / (4 * np.pi)


def expected_count(lo, hi, domain=None, *, area=None, perim=None, bc_type="dir"):
    """How many eigenvalues the two-term law expects in `[lo, hi]`. **SIZING ONLY.**

    A search may read this to choose a starting sample density and for nothing else. It says
    nothing about WHERE anything is and must not be used as if it did -- the whole failure
    this warning exists for is a rescue loop whose stopping condition was a COUNT rather than
    consecutiveness, which filled its quota with whatever it could reach and returned a
    non-consecutive spectrum with no exception and no warning.

    It is also an estimate of the MEAN gap, while recall is bounded by the MINIMUM LOCAL gap.
    Measured across seven domains, the smallest observed gap runs 0.127 to 0.792 of the Weyl
    mean -- 1.3x to 8x tighter -- so a stride computed from this is an estimate of the wrong
    quantity and has to be corrected in the loop rather than trusted once.

    Clamped at zero: the two-term count is not monotone at very small `lam`, where the
    perimeter term dominates, and a negative width is never a useful answer.
    """
    n_lo = weyl_count(lo, domain, area=area, perim=perim, bc_type=bc_type)
    n_hi = weyl_count(hi, domain, area=area, perim=perim, bc_type=bc_type)
    return np.maximum(n_hi - n_lo, 0.0)


def weyl_est(k, domain=None, *, area=None, perim=None, bc_type="dir"):
    """`lam_k` from the two-term law, inverted in closed form. `k` may be non-integer."""
    area, perim, bc_type = _resolve(domain, area, perim, bc_type)
    s = _sign(bc_type)
    k = np.asarray(k, dtype=float)
    # The sign flips AGAIN here: inverting the quadratic in sqrt(lam) moves the perimeter
    # term across, so Dirichlet (which subtracts in the forward count) carries +P.
    num_sign = -s
    return ((num_sign * perim + np.sqrt(perim**2 + 16 * np.pi * area * k)) / (2 * area)) ** 2


def weyl_count_poly(lam, domain=None, *, area=None, perim=None, angles=None, bc_type="dir"):
    """Three-term count: the two-term law plus the corner sum. Forward."""
    return (weyl_count(lam, domain, area=area, perim=perim, bc_type=bc_type)
            + corner_correction(_angles(domain, angles)))


def weyl_est_poly(k, domain=None, *, area=None, perim=None, angles=None, bc_type="dir"):
    """`lam_k` from the three-term law, inverted numerically.

    Seeded by `weyl_est`, then bracketed outward. Raises rather than handing `brentq` an
    invalid bracket, which is what lappy does after its widening loop gives up.
    """
    scalar = np.ndim(k) == 0
    ks = np.atleast_1d(np.asarray(k, dtype=float))
    ang = _angles(domain, angles)
    out = np.empty(ks.shape)
    for i, kk in enumerate(ks.ravel()):
        lam0 = float(weyl_est(kk, domain, area=area, perim=perim, bc_type=bc_type))
        f = lambda x: weyl_count_poly(x, domain, area=area, perim=perim, angles=ang,
                                      bc_type=bc_type) - kk
        lo, hi = max(lam0 / 4, 1e-12), lam0 * 4
        for _ in range(40):
            if f(lo) * f(hi) <= 0:
                break
            lo, hi = max(lo / 2, 1e-14), hi * 2
        else:
            raise ValueError(
                f"could not bracket lam for k={kk}: the three-term count does not change "
                f"sign on [{lo:.3g}, {hi:.3g}]"
            )
        out.ravel()[i] = brentq(f, lo, hi)
    return float(out.ravel()[0]) if scalar else out.reshape(ks.shape)


def weyl_count_check(eigs, domain=None, *, area=None, perim=None, bc_type="dir"):
    """`(i + 1) - weyl_count(eigs[i])`, pointwise.

    Positive means the actual count runs ahead of the asymptotic prediction. The deviation is
    `O(sqrt(lam))` and oscillates around zero for a typical domain, so this is a shape check,
    not a pass/fail.
    """
    e = np.asarray(eigs, dtype=float)
    return np.arange(1, e.size + 1) - weyl_count(e, domain, area=area, perim=perim,
                                                 bc_type=bc_type)


def weyl_deficit(eigs, domain=None, *, area=None, perim=None, angles=None, bc_type="dir",
                 rtol=1e-6):
    """How many eigenvalues the three-term count says are MISSING, worst over the gaps.

    Cuts in the middle of every gap between DISTINCT levels and compares the corner-corrected
    count there against how many returned values lie below it. Returns the worst shortfall,
    or `nan` when there are fewer than two distinct levels and so nowhere to cut.

    **It cannot see a missing copy of a degenerate level.** Every cut lies below the top
    level by construction, so a caller has to ask for more eigenvalues than it consumes if it
    wants the top of its range audited.
    """
    e = np.sort(np.asarray(eigs, dtype=float))
    if e.size < 2 or not np.all(np.isfinite(e)) or e[0] <= 0:
        return float("nan")
    levels = [e[0]]
    for x in e[1:]:
        if abs(x - levels[-1]) > rtol * max(abs(x), 1.0):
            levels.append(x)
    if len(levels) < 2:
        return float("nan")
    worst = -np.inf
    for a, b in zip(levels, levels[1:]):
        cut = 0.5 * (a + b)
        expected = weyl_count_poly(cut, domain, area=area, perim=perim, angles=angles,
                                   bc_type=bc_type)
        worst = max(worst, float(expected) - float(np.sum(e <= cut)))
    return worst
