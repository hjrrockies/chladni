"""Weyl asymptotics: how many eigenvalues there should be, and roughly where.

Forward (`lam -> count`) and inverted (`k -> lam_k`) forms of the two- and three-term laws,
plus two diagnostics that compare a computed spectrum against them.

Like `bounds`, each function takes either a `Domain` or the raw scalars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

__all__ = ["weyl_count", "weyl_est", "weyl_count_poly", "weyl_est_poly", "weyl_deficit",
           "weyl_count_check", "corner_correction", "expected_count", "count_audit",
           "CountAudit"]


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


#: `count_audit`'s thresholds on the MEAN deficit (`notes/count_audit.md`). On the douse benchmark
#: (819 real solves, FEM-labelled) complete spectra scored [-0.10, 0.04] and every drop >= 0.47.
#: Across all 18 shapes with known spectra, complete prefixes reach +0.23 (a 5:1 rectangle) and
#: -0.48 (an 8-gon cut at 6 levels), so these clear every complete spectrum measured. The price:
#: on spectra of <= 30 levels, a single drop with 4 levels above it is caught ~94% of the time.
AUDIT_SHORT = 0.25
AUDIT_EXCESS = -0.6
#: The window's top when it is NOT known to sit in a gap: a single cut's deficit is not averaged,
#: and on complete reference spectra it spans [-1.57, 1.84] (Weyl's remainder, about one
#: eigenvalue, plus up to half from where the top falls in its gap). So it only catches a window
#: that came back several eigenvalues short -- a basis that found nothing at all.
AUDIT_TOP_COARSE = 2.5
#: Values this close are one level, as in `weyl_deficit`.
_AUDIT_RTOL = 1e-6


@dataclass(frozen=True)
class CountAudit:
    """A spectrum's count against Weyl's corner-corrected law. See `count_audit`."""
    verdict: str                          # "ok", "short", "excess" or "unaudited"
    statistic: float = float("nan")       # mean deficit over the mid-gap cuts
    n_cuts: int = 0
    top_deficit: float = float("nan")     # at the window's top, when one was given
    step_at: float = float("nan")         # the first cut where the deficit reaches +-1/2
    reason: str = ""

    @property
    def ok(self):
        return self.verdict == "ok"


def count_audit(eigs, domain, *, lo, top=None, top_is_gap=False, short=AUDIT_SHORT,
                excess=AUDIT_EXCESS):
    """Does a PREFIX of the spectrum have the right number of eigenvalues? A `CountAudit`.

    The one check for a DROPPED eigenvalue: no returned level exists to carry a flag for it.
    At every mid-gap cut between returned levels, `weyl_count_poly(cut) - #(eigs <= cut)`; the
    verdict reads the MEAN over the cuts. Validated in `notes/count_audit.md` on 819 real solves
    labelled by finite-element counts: 100 of 100 drops caught, one of them balanced by a
    duplicate so that the total was right, and no false alarm in 716 complete solves. The
    WORST cut (`weyl_deficit`, douse's guard) gave 20 false alarms: Weyl's remainder swings
    about half an eigenvalue cut to cut, while a drop moves EVERY cut above it.

    * `"short"` when the mean exceeds `short`, `"excess"` when it falls below `excess` (an
      eigenvalue returned twice). `step_at` is the first cut where the deficit reaches one
      half, which says where to look and certifies nothing.
    * `top` is the window's upper end. With `top_is_gap=True` (it is known to sit in a gap) it
      joins the cuts, as calibrated. Otherwise it is a coarse check only, `+-AUDIT_TOP_COARSE`,
      which catches a window that came back nearly empty.
    * `"unaudited"`, with `reason`, when the count cannot be compared:
      - `lo` is not below the Faber-Krahn bound on `lam_1` (Weyl counts from zero);
      - the boundary is not all Dirichlet (the corner sum and the calibration are Dirichlet's);
      - there is nothing to cut.
    * **It is a backstop, not a certificate.** On spectra of <= 30 levels it catches ~94% of
      single drops with four levels of headroom above them (all 100 on the douse benchmark), and
      none of 320 complete spectra over 18 shapes alarms. It weakens with depth: a drop with `h`
      levels above it moves the mean by about `h/n`, while Weyl's remainder wanders by about two
      eigenvalues over the last cuts of a deep spectrum, so no local reading recovers it. Audit
      four levels past what you consume.
    """
    from .bounds import faber_krahn

    if domain.bc_type != "dir":
        return CountAudit("unaudited", reason=f"the boundary is {domain.bc_type!r}; the corner "
                          f"sum and the calibration are Dirichlet's")
    if not lo < faber_krahn(domain):
        return CountAudit("unaudited", reason=f"not a prefix window: lo = {lo:.6g} is not below "
                          f"the Faber-Krahn bound on lam_1, so the count below it is unknown")
    e = np.sort(np.asarray(eigs, dtype=float).ravel())
    levels = []
    for x in e:
        if not levels or abs(x - levels[-1]) > _AUDIT_RTOL * max(abs(x), 1.0):
            levels.append(x)
    cuts = [0.5 * (a + b) for a, b in zip(levels, levels[1:])]
    if top is not None and top_is_gap:
        cuts.append(float(top))

    def deficit(c):
        return float(weyl_count_poly(c, domain)) - float(np.sum(e <= c))

    d = np.array([deficit(c) for c in cuts])
    top_d = deficit(float(top)) if top is not None else float("nan")
    if d.size == 0 and top is None:
        return CountAudit("unaudited", reason="fewer than two distinct levels and no top: "
                          "nothing to cut")
    stat = float(d.mean()) if d.size else float("nan")
    coarse = top is not None and not top_is_gap
    if (d.size and stat > short) or (coarse and top_d > AUDIT_TOP_COARSE):
        verdict = "short"
        hit = np.flatnonzero(d >= 0.5)
    elif (d.size and stat < excess) or (coarse and top_d < -AUDIT_TOP_COARSE):
        verdict = "excess"
        hit = np.flatnonzero(d <= -0.5)
    else:
        verdict, hit = "ok", np.array([], int)
    step = float(cuts[hit[0]]) if hit.size else float("nan")
    return CountAudit(verdict, stat, int(d.size), top_d, step)
