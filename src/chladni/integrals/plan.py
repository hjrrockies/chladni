"""Where the boundary nodes go, and how many: an integral rule sized for `prec` at `lam_max`.

Ported from lappy's `boundary_quadrature` (`lappy/eigfun_integrals.py:697-901`), which measured
every choice below before keeping it. The rule is a pure function of the geometry, `lam_max` and
the precision asked for -- no basis -- so one rule serves every `lam` of a solve.

# Which corners get a corner rule

A corner's integrand carries `r^gamma` with `gamma = 2 mu - 2`, `mu` its leading exponent
(`helmholtz.CornerModes`, the one place allowed to call a corner singular). It needs the corner
rule when a smooth rule CANNOT INTEGRATE that power to the target -- asked directly, with
`rules.smooth_power_error` at the order its edges already get. That is not "re-entrant": a
135-degree corner (`r^(2/3)`) needs it and lappy's re-entrant-only test missed it (5.6e-9 against
a claimed 1e-13 on a right trapezoid). An even-integer `gamma` is a polynomial and never needs it.

Only Dirichlet/Dirichlet corners are wired. A Neumann corner's `uv` and `TT` kernels carry
different exponents, which no single node set matches; a Zaremba corner with `mu <= 1/2` makes
the integrand non-integrable. Both fall back to the smooth rule and are REPORTED -- the sizing
precision becomes infinite, because a smooth model cannot see the power it is failing on.

# The panels

A segment with an admissible corner at one end gets a corner panel anchored there and a smooth
panel for the rest; with corners at both ends it splits, each corner panel at most half. A corner
panel is at most `CLEARANCE_FRAC` of the corner's clearance: the corner rule clusters its nodes at
the corner, so the far end of a long panel cannot resolve the oscillation. lappy measured up to
twelve orders lost without that cap on a 1 x 16 strip, and a fixed fraction of the EDGE did not
substitute (it grows with the edge).

An eigenfunction on a polygon is singular at the vertices and nowhere else, so along a straight
edge a smooth panel has exactly one more thing to resolve than its wavenumber: each other singular
corner, which continues to `tau = t* +- i y`, `y` the corner's distance from the edge's line. Where
Gauss's convergence toward those points needs more than the wavenumber's order, the panel is halved
toward them until each is admissible (panel no wider than its distance away) and raised to the order
left over (`_smooth_panels`). Sized on wavenumber alone, a 250-degree corner 0.156 from a long edge
integrated that edge to 1e-6 at a claimed 1e-10 (douse's N = 6 stall, 2026-11-08).

# Sizing, and what it is not

Each corner panel's order comes from `rules.corner_order_for_prec` on the corner's own model
integrand; each smooth panel's from `rules.smooth_order_for_prec` at the product's wavenumber
`2 sqrt(lam_max)` over the panel. `sizing_prec` is the worst model bound -- a SIZING claim, not a
certificate. lappy's `chevron_1_2` was sized at 1e-13 and measured 4.9e-8, from smooth panels whose
real requirement is set by the basis, which geometry cannot see. Refinement certifies (M3).

# Candidates, not eigenfunctions

The corner rule is exact on the corner family `{gamma + j mu + 2q}`. An MPS candidate also carries
smooth columns, whose cross terms with the corner series are `r^(mu - 1)` times integer powers --
outside that family. Near an eigenvalue the smooth part IS the corner series to within the
boundary residual, so the error this leaves scales with the residual; it is measured, not assumed,
in `tests/test_rellich.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Panel
from ..geometry.segments import LineSegment
from ..helmholtz import CornerModes
from . import rules

__all__ = ["CLEARANCE_FRAC", "SMOOTH_SAFETY", "NU_MARGIN", "integral_prec", "CornerSpec",
           "IntegralQuad", "corner_specs", "default_x0", "integral_quad"]

#: Cap on a corner panel's arclength, as a multiple of the corner's clearance. lappy, on the 1 x N
#: strip: 1.0 sits at the edge of the expansion's validity and lands an order worse than 0.9;
#: below 0.9 buys nothing.
CLEARANCE_FRAC = 0.9

#: Multiplier on the wavenumber the smooth panels are sized for. The model integrand
#: `exp(i k tau)` at `k = 2 sqrt(lam_max) L` is not the real integrand's bandwidth: lappy found it
#: generous on polygons and short on curved boundaries. On the L-shape it is SHORT: the first
#: eigenfunction, from an 80-column corner Fourier-Bessel basis, measured against a graded interior
#: oracle at a 1e-13 target (2026-10-05):
#:
#:     safety       1          2          4          8
#:     rel err   1.8e-11    3.3e-15    2.0e-15    2.7e-15
#:     nodes        84        100        132        188
#:
#: so 2 is the default: 19% more nodes for four digits. A model, still -- refinement certifies.
SMOOTH_SAFETY = 2.0

#: Margin above the slit limit `nu = 1/2`. At 0.51 the Jacobi exponent is -0.96 and the
#: innermost node meets the coordinate floor by order 32, so nearer corners are demoted.
NU_MARGIN = 0.02


#: Coarsest panel `_smooth_panels` will halve to. A guard: a vertex on an edge is an invalid domain.
NEAR_MIN_PANEL = 1e-6


def integral_prec(prec):
    """The precision the integral rule is sized to: `max(0.1 prec, 1e-14)`. A tenth of the
    eigenvalue's, so quadrature error is never what limits a Gram matrix at `prec`; floored where
    a 270-degree corner lands (~1.9e-14), since below it the target is unreachable."""
    return max(0.1 * float(prec), 1e-14)


@dataclass(frozen=True)
class CornerSpec:
    """One corner's classification. `kind` is the panel rule it gets: `'substitution'`,
    `'interpolatory'`, `'smooth'` (no corner rule needed), or `'fallback'` (one is needed and
    none applies -- `reason` says why, and the rule's sizing precision is infinite)."""

    corner: object
    mu: float
    gamma: float
    straight: bool
    kind: str
    sub: float
    reason: str = ""

    @property
    def needs_rule(self):
        return self.kind in ("substitution", "interpolatory")


@dataclass(frozen=True)
class IntegralQuad:
    """An integral rule and what it was sized for.

    `quad` is a `geometry.Quadrature` whose `weight_family` is `'even'` (or `'integer'`, M5).
    `sizing_prec` is the worst MODEL bound over its panels -- infinite when a singular corner was
    demoted -- and `shortfalls` names every corner or segment short of the target, as
    `(where, achieved_or_None, reason)`. `x0` is the default Rellich origin.
    """

    quad: object
    family: str
    x0: complex
    lam_max: float
    prec: float
    sizing_prec: float
    shortfalls: tuple
    corners: tuple


def _straight(seg):
    return isinstance(seg, LineSegment)


def corner_specs(domain, smooth_orders, prec):
    """A `CornerSpec` per corner of `domain`, in `domain.corners` order. `smooth_orders` maps
    `(bdry_idx, seg_idx)` to the order a smooth panel there gets; it is what a smooth rule could
    deliver on `r^gamma`, and so what decides whether a corner rule is needed at all."""
    specs = []
    for c in domain.corners:
        modes = CornerModes.from_wedge(c.omega, bc_out=c.bc_out, bc_in=c.bc_in)
        mu = modes.leading_exponent
        gamma = 2.0 * mu - 2.0
        straight = _straight(c.segment_out) and _straight(c.segment_in)
        order = max(smooth_orders[(c.bdry_idx, c.seg_out)], smooth_orders[(c.bdry_idx, c.seg_in)])
        if rules.smooth_power_error(gamma, order) <= prec:
            specs.append(CornerSpec(c, mu, gamma, straight, "smooth", np.nan))
            continue
        reason = ""
        if modes.is_zaremba:
            reason = (f"a Zaremba corner (mu = {mu:.4f}): its integrand is not in the corner "
                      f"family the rules are built on, and for mu <= 1/2 not integrable")
        elif modes.kind != "sin":
            reason = ("a Neumann corner: its uv and TT kernels carry different exponents, "
                      "which no single node set matches")
        elif not mu > rules.NU_MIN + NU_MARGIN:
            reason = f"mu = {mu:.4f} is within {NU_MARGIN} of the slit limit 1/2"
        if reason:
            specs.append(CornerSpec(c, mu, gamma, straight, "fallback", np.nan, reason))
            continue
        kind, sub = rules.corner_rule_kind(mu, curved=not straight)
        specs.append(CornerSpec(c, mu, gamma, straight, kind, sub))
    return tuple(specs)


def default_x0(domain, specs):
    """The Rellich origin: the corner-ruled corner of smallest `mu`, where `r . N` vanishes on
    both straight edges and removes that corner from the integrand outright; else the bounding
    box's centre. Free, and it makes the one corner it covers exact."""
    ruled = [s for s in specs if s.needs_rule]
    if ruled:
        return complex(min(ruled, key=lambda s: s.mu).corner.z)
    pts = np.concatenate([b.sample(200) for b in domain.boundaries])
    return complex(0.5 * (pts.real.min() + pts.real.max()), 0.5 * (pts.imag.min() + pts.imag.max()))


def _corner_nodes(rule_tau_w, anchor_at_start):
    """A `Panel.nodes` callable for a corner rule anchored at one end of `[tau0, tau1]`. The rule
    lives on `[0, 1]` with the corner at 0; weights come back in tau, as the seam requires."""
    u, w = rule_tau_w

    def build(tau0, tau1):
        h = tau1 - tau0
        tau = tau0 + h * u if anchor_at_start else tau1 - h * u
        order = np.argsort(tau)
        return tau[order], (h * w)[order]

    return build


def _graded_nodes(rule_d_w):
    """A `Panel.nodes` callable for an ANCHORED graded panel: called with the panel's span in the
    distance from its corner, `[0, frac]`, it returns nodes in that distance."""
    u, w = rule_d_w

    def build(d0, d1):
        h = d1 - d0
        return d0 + h * u, h * w

    return build


def _legendre(order):
    def build(tau0, tau1):
        x, w = rules.gauss_legendre(int(order))
        return tau0 + (tau1 - tau0) * x, (tau1 - tau0) * w
    return build


def _near_singularities(specs, bdry_idx, seg_idx, seg):
    """Where the singular corners not on this straight segment sit in its `tau`, as `t* + i y`
    (`y >= 0`; the conjugate is the same distance). A corner the smooth rule integrates anyway
    (`kind == 'smooth'`) has nothing to resolve, and a curved segment has no such closed form:
    both give none."""
    if not _straight(seg):
        return np.array([], dtype=complex)
    a, b = seg.p(0.0), seg.p(1.0)
    out = []
    for s in specs:
        c = s.corner
        if s.kind == "smooth" or (c.bdry_idx == bdry_idx and seg_idx in (c.seg_out, c.seg_in)):
            continue
        w = (complex(c.z) - a) / (b - a)
        out.append(complex(w.real, abs(w.imag)))
    return np.array(out, dtype=complex)


def _smooth_panels(lo, hi, near, size, target):
    """`[(a, b, order, achieved), ...]` covering `[lo, hi]`, Gauss-Legendre on each. `size(h)` is
    the wavenumber's `(order, achieved)` for a panel of width `h`; `near` the singularities in
    tau. Gauss on `[lo, hi]` converges like `rho^(-2 n)`, `rho` the Bernstein parameter of the
    nearest of them, so a panel whose wavenumber order already covers that stays exactly as it
    was. Otherwise it is halved while wider than its distance to `near`, then raised."""
    order, ach = size(hi - lo)
    if near.size == 0:
        return [(lo, hi, order, ach)]
    x = (2.0 * near - (lo + hi)) / (hi - lo)
    s = np.sqrt(x * x - 1.0 + 0j)
    rho = float(np.maximum(np.abs(x + s), np.abs(x - s)).min())
    need = 2 * int(np.ceil(np.log(1.0 / target) / (4.0 * np.log(rho))))
    if need <= order:
        return [(lo, hi, order, ach)]
    dist = float(np.abs(near - np.clip(near.real, lo, hi)).min())
    if hi - lo > max(dist, NEAR_MIN_PANEL):
        mid = 0.5 * (lo + hi)
        return (_smooth_panels(lo, mid, near, size, target)
                + _smooth_panels(mid, hi, near, size, target))
    return [(lo, hi, need, ach)]


def integral_quad(domain, lam_max, prec=1e-8, *, x0=None, clearance_frac=CLEARANCE_FRAC,
                  smooth_safety=SMOOTH_SAFETY, family="even", corners="graded"):
    """An `IntegralQuad` for `domain`, sized so the Rellich and Gram integrals of functions with
    spectral parameter up to `lam_max` are accurate to `integral_prec(prec)` -- as a MODEL bound.

    `corners` picks the corner-panel rule. `'graded'` (the default since 2026-11-03) is
    `rules.graded_corner_rule`, laid out on ANCHORED panels so that `cauchy.cauchy_data`
    evaluates its nodes in the corner's own frame: one node set serves both families, and it
    reaches the floor where the fitted rule stalls (`notes/hadamard_accuracy.md`). A graded panel
    on a segment without an exact `offset` is named in `shortfalls`. `'interpolatory'` is the
    fitted rule of `rules.corner_interp_rule` (or the substitution rule where that is exact),
    kept to be measured against: its model bound can be honest and still miss -- it integrated
    one edge of a flat triangle 30% wrong at a claimed 1e-8.

    `clearance_frac=None` removes the clearance cap; it exists to be measured, not used.
    """
    if corners not in ("interpolatory", "graded"):
        raise ValueError(f"corners must be 'interpolatory' or 'graded', got {corners!r}")
    graded = corners == "graded"
    # 'integer' is the HADAMARD rule (M5): the integrand `(d_n u)^2 V` carries the velocity's
    # integer powers of `r` on top of the corner family -- `V` is linear along a straight edge of
    # a moving polygon -- so every ruled corner takes the interpolatory rule fitted to the CURVED
    # exponent set `{gamma + j nu + m}`, every integer `m`, which contains them. The substitution
    # shortcut is exact only on the straight, even-power set, and is not used for it.
    if family not in ("even", "integer"):
        raise ValueError(f"family must be 'even' or 'integer', got {family!r}")
    hadamard = family == "integer"
    target = integral_prec(prec)
    k = 2.0 * np.sqrt(max(float(lam_max), 0.0))
    segs = {(bi, si): seg for bi, b in enumerate(domain.boundaries)
            for si, seg in enumerate(b.segments)}
    smooth_orders, smooth_ach = {}, {}
    for key, seg in segs.items():
        smooth_orders[key], smooth_ach[key] = rules.smooth_order_for_prec(
            k * seg.length * smooth_safety, target)
    specs = corner_specs(domain, smooth_orders, target)
    if x0 is None:
        x0 = default_x0(domain, specs)

    starts, ends = {}, {}                 # (bdry, seg) -> the spec anchored at that end
    for s in specs:
        if s.needs_rule:
            starts[(s.corner.bdry_idx, s.corner.seg_out)] = s
            ends[(s.corner.bdry_idx, s.corner.seg_in)] = s

    panels, achieved, inexact = [], {}, set()
    for (bi, si), seg in segs.items():
        s0, s1 = starts.get((bi, si)), ends.get((bi, si))
        share = 0.5 if (s0 is not None and s1 is not None) else 1.0
        lo, hi = 0.0, 1.0
        for spec, at_start in ((s0, True), (s1, False)):
            if spec is None:
                continue
            frac = share
            if clearance_frac is not None:
                frac = min(frac, clearance_frac * spec.corner.clearance / seg.length)
            frac = float(np.clip(frac, 1e-6, share))
            k_corner = np.sqrt(max(float(lam_max), 0.0)) * seg.length * frac
            idx = specs.index(spec)
            if graded:
                (lv, o, n_top), ach = rules.graded_order_for_prec(spec.mu, target, k_corner)
                nodes = _graded_nodes(rules.graded_corner_rule(spec.mu, lv, o,
                                                               rules.GRADED_SIGMA, n_top))
                if not seg.exact_offset:
                    inexact.add((idx, (bi, si)))
            else:
                kind = "interpolatory" if hadamard else spec.kind
                curved = True if hadamard else not spec.straight
                order, ach = rules.corner_order_for_prec(kind, spec.mu, spec.gamma, spec.sub,
                                                         curved, target, scale=frac, k=k_corner)
                rule = (rules.corner_rule(order, spec.mu, spec.gamma, spec.sub)
                        if kind == "substitution" else
                        rules.corner_interp_rule(order, spec.mu, spec.gamma, None, curved))
                nodes = _corner_nodes(rule, at_start)
            achieved[idx] = max(achieved.get(idx, 0.0), ach)
            anchor = ("start" if at_start else "end") if graded else None
            if at_start:
                panels.append(Panel(bi, si, 0.0, frac, nodes, anchor=anchor))
                lo = frac
            else:
                panels.append(Panel(bi, si, 1.0 - frac, 1.0, nodes, anchor=anchor))
                hi = 1.0 - frac
        if hi - lo > 1e-12:
            pieces = _smooth_panels(
                lo, hi, _near_singularities(specs, bi, si, seg),
                lambda h: rules.smooth_order_for_prec(k * seg.length * h * smooth_safety, target),
                target)
            panels.extend(Panel(bi, si, a, b, _legendre(o)) for a, b, o, _ in pieces)
            smooth_ach[(bi, si)] = max(a for *_, a in pieces)
        elif s0 is None and s1 is None:
            raise AssertionError("a segment with no corner panel must get a smooth one")

    quad = domain.bdry_quad(panels=panels, weight_family=family)
    shortfalls = tuple(
        [(f"corner {i}", a, "model bound above target") for i, a in achieved.items() if a > target]
        + [(f"corner {i}", None, f"demoted to the smooth rule: {s.reason}")
           for i, s in enumerate(specs) if s.kind == "fallback"]
        + [(f"segment {key}", a, "smooth model bound above target")
           for key, a in smooth_ach.items() if a > target]
        + [(f"corner {i}", None, f"segment {key} has no exact offset: its corner-local frame "
            f"rounds at eps * |corner|") for i, key in sorted(inexact)])
    sizing = max([target] + list(achieved.values()) + list(smooth_ach.values()))
    if any(s.kind == "fallback" for s in specs):
        sizing = float("inf")
    return IntegralQuad(quad=quad, family=family, x0=complex(x0), lam_max=float(lam_max),
                        prec=target, sizing_prec=sizing, shortfalls=shortfalls, corners=specs)
