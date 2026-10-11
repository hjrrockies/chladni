"""TIER 3: domains harder than a rectangle.

TEST SUPPORT ONLY. The most expensive of the three tiers and the last a candidate reaches.
`CHLADNI_V1.md` section 10 puts this first in future work: "**A second domain.** Every number
here is one rectangle... **This is worth more than any remaining speedup**, and it is the only
item here that could invalidate the design rather than tune it."

# What each domain is for

    L_shape    a REENTRANT corner. The transfer test that matters: the eigenfunction carries
               an `r**(2/3)` singularity, the tension floor varies FIVE ORDERS within one scan
               (3.1e-13 at k = 4.443 against 5.5e-08 at k = 5.649), and the wall slope may not
               show the 5% spread in `k` that the whole global-slope argument rests on.
               Truth is a 14-digit table -- a PREFIX, so coverage is audited by Weyl and not
               proved.
    disk       complete closed-form truth AND genuine degeneracies, so coverage and
               multiplicity are both PROVABLE. No corner singularity to confound the reading.
    eq_tri     the same, at a different aspect and with three 60-degree corners.

The two smooth domains are here because a prefix table cannot prove coverage: it can only fail
to contradict it. The L-shape is here because it is the case that could invalidate the design.

# Basis adequacy is the binding constraint, and it differs per domain

**A domain's basis sets the finest `prec` it can be asked for**, and asking for more measures
the basis. From the recipes that reproduce `chladni.reference`'s own tables:

    domain      basis                        tension range        finest prec
    L_shape     80 pure FB at the corner     2.1e-15 .. 2.1e-14   ~1e-14
    eq_tri      90 plane waves               8.1e-16 .. 1.3e-15   1.5e-15   (2026-09-14)
    disk        84 plane waves               7.2e-16 .. 1.7e-15   1.2e-15   (2026-09-14)
    mushroom    240 mixed                    2.4e-11 .. 3.4e-11   ~1e-10
    GWW1        320 mixed                    6.3e-12 .. 2.7e-11   ~1e-10
    H_shape     320 mixed                    6.1e-13 .. 5.2e-10   ~1e-9
    chevron     160 mixed                    1.4e-10 .. 2.0e-09   ~1e-8

So the L-shape, eq_tri and the disk can be swept to the bottom of the `prec` ladder and the
tabulated multi-corner domains cannot.

**Every pencil here carries `SUITE_RTOL`.** Built at the default `prec` the rank cutoff was
`sqrt(1e-8 * eps) = 1.5e-12`, which moved with the request and held eq_tri at 2.4e-12 -- read
for a long time as "plane waves stop short on eq_tri". They do not: the same 90 columns at
`SUITE_RTOL` reached 1.3e-15 while it was cut for 1e-14 (it is cut for 1e-10 since 2026-10-11,
`rect_suite.FINEST_PREC`). The disk and the L-shape did not move, because their bases were
already the binding term.

The roster is the L-shape plus the two smooth domains, and every case carries the finest `prec`
it supports -- checked, not assumed, by `finest_prec` below. Scoring a domain past that
point measures the basis.

**The generic basis is not good enough here.** `basis_for`-style plane waves plus a fixed
number of FB terms measured 15x to 7e+05 worse than the recipe on this roster, and most of the
"floor spread" once attributed to these domains was that basis rather than the domains.
"""
from __future__ import annotations

import numpy as np

from chladni.geometry import L_shape, disk, eq_tri
from chladni.helmholtz import FourierBessel, PlaneWave
from chladni.mps import MPSEigensolver, MPSPencil
from chladni.reference import L_shape_eigs, disk_spectrum, eq_tri_spectrum

from harness import Case
from rect_suite import SUITE_RTOL

__all__ = ["nonanalytic", "recipe_basis", "pencil_for", "solver_for", "make_case",
           "finest_prec",
           "l_shape_case", "disk_case", "eq_tri_case", "all_cases", "ROSTER"]

_PENCILS: dict = {}


def nonanalytic(dom):
    """Corners whose Dirichlet expansion `r**(l*pi/omega)` has a fractional exponent.

    A corner is singular when its local orders are not all integers, which under a single
    boundary condition is just "`pi/omega` is not an integer". The dangerous ones are
    REENTRANT, `omega > pi`, where `pi/omega < 1` and the gradient blows up.
    """
    return [c for c in dom.corners if not c.is_reflection_closed]


def recipe_basis(dom, n_fb=80):
    """Pure Fourier-Bessel at every non-analytic corner. Plane waves where there are none.

    **Pass `domain`** to `at_corner`, which is not optional: the scaling radius is then the
    evaluation radius over the whole boundary rather than the corner's own clearance, and on
    the L-shape those differ by 2.83x in radius and by 5.7e+10 in the resulting column maximum.
    """
    sing = nonanalytic(dom)
    if not sing:
        return None
    per = max(int(n_fb / len(sing)), 4)
    b = None
    for c in sing:
        f = FourierBessel.at_corner(c, n=per, domain=dom)
        b = f if b is None else b + f
    return b


def pencil_for(dom, lam_hi, *, label="", n_fb=80, pad=1.3, seed=0):
    """A reference-grade pencil, memoized. Recipe FB where there is a singular corner."""
    key = (label, float(lam_hi), int(n_fb), float(pad), int(seed))
    hit = _PENCILS.get(key)
    if hit is not None:
        return hit
    basis = recipe_basis(dom, n_fb)
    if basis is None:
        nw = int(pad * PlaneWave.n_useful(np.sqrt(lam_hi) * dom.diameter))
        basis = PlaneWave.uniform(nw, radius=dom.diameter)
    n = len(basis)
    # 3x boundary, 1x interior, sized to the pencil rather than to a fixed floor. See
    # `rect_suite.pencil_for` for the measurement: the floor does not move between 8x/3x and
    # 1.2x/1x, so the extra points cost 1.5x for nothing.
    p = MPSPencil.from_basis(
        basis, dom.bdry_quad(3 * n),
        dom.interior_pts(n, rng=np.random.default_rng(seed)), rtol=SUITE_RTOL)
    _PENCILS[key] = p
    return p


def solver_for(dom, lam_hi, *, label="", n_fb=80, pad=1.3, seed=0, prec=None):
    """An `MPSEigensolver` on the SAME recipe basis as `pencil_for`, memoized.

    The point of sharing the recipe is attribution. `solve_interval` is the one piece of this
    layer that was newly written rather than promoted, so when an end-to-end solve is wrong the
    first question is whether the wiring or the basis is at fault. Built on a basis whose
    tension floor is already pinned by `finest_prec` and by `chladni.reference`'s own tables,
    the basis is ruled out and the wiring is what is left.

    `MPSEigensolver` builds its own quadrature and interior fill from the basis size, on the
    same 3x/1x rule, so this passes the basis and the counts and lets the solver do the rest.
    """
    key = ("solver", label, float(lam_hi), int(n_fb), float(pad), int(seed),
           None if prec is None else float(prec))
    hit = _PENCILS.get(key)
    if hit is not None:
        return hit
    basis = recipe_basis(dom, n_fb)
    if basis is None:
        nw = int(pad * PlaneWave.n_useful(np.sqrt(lam_hi) * dom.diameter))
        basis = PlaneWave.uniform(nw, radius=dom.diameter)
    n = len(basis)
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * n, n_int=n,
                       rng=np.random.default_rng(seed), rtol=SUITE_RTOL,
                       **({} if prec is None else {"prec": prec}))
    _PENCILS[key] = s
    return s


def finest_prec(case, *, margin=3.0, h_frac=0.005):
    """The tightest `prec` this case's objective can actually support.

    The bracketing form of the adequacy contract: a bracket of width `prec*lam` is usable only
    while it is wider than the well's flat bottom, whose scale is `eps/c`. Returned as a
    relative precision, so a caller compares `prec >= finest_prec(case)` before scoring.

    This is the third place the same precondition appears -- `surrogate_cases.adequate` on tier
    1, `rect_suite.assert_floor` on tier 2, and here -- because each tier learned separately
    that scoring a case past its objective's limit measures the objective.
    """
    p = case.pencil
    worst = 0.0
    for x in np.unique(case.truth):
        h = h_frac * abs(x)
        eps = p.tension(float(x))
        c = 0.5 * (p.tension(x + h) + p.tension(x - h)) / h
        if c <= 0:
            return float("inf")
        worst = max(worst, margin * (eps / c) / abs(x))
    return float(worst)


def make_case(label, dom, a, b, truth, *, complete, n_fb=80, seed=0):
    p = pencil_for(dom, b, label=label, n_fb=n_fb, seed=seed)

    def tensions(lam):
        return p.tensions(float(lam))

    # `Case.modes`, which the roster went without until Phase D's rule was measured against it
    # and `solve(phase_d="angle")` refused the whole roster outright. `rect_suite` has carried
    # this since the harness was written; the transfer domains are exactly the ones a grouping
    # rule most needs to be tried on, so the omission mattered.
    def modes(lam):
        return p.interior_image(float(lam))

    case = Case(label=label, tensions=tensions, a=float(a), b=float(b),
                truth=np.sort(np.asarray(truth, dtype=float)), domain=dom,
                complete=complete, variable="lam", modes=modes)
    case.pencil = p
    return case


def l_shape_case(a=5.0, b=26.0, n_fb=80):
    """The transfer test. A reentrant corner, `r**(2/3)`, and a five-order floor spread.

    Truth is `chladni.reference`'s 14-digit table, which is a PREFIX: `complete=False`, so
    coverage here is AUDITED by Weyl rather than proved. That is not a weakness of the case --
    it is the situation on every domain that matters, and it is why the audit exists.
    """
    lam = np.asarray(L_shape_eigs(12), dtype=float)
    inside = lam[(lam >= a) & (lam <= b)]
    case = make_case("L_shape", L_shape(), a, b, inside, complete=False, n_fb=n_fb)
    # The table's own accuracy, which `basis_sufficiency` reads: a 14-digit table cannot
    # referee a request past 1e-13, whatever the basis does.
    case.digits = float(L_shape_eigs.digits)
    return case


def disk_case(a=5.0, b=60.0, r=1.0):
    """Complete closed-form truth WITH degeneracies, so coverage and multiplicity are provable."""
    return make_case("disk", disk(r), a, b, disk_spectrum(a, b, r), complete=True)


def eq_tri_case(a=50.0, b=300.0, l=1.0):
    """The same, at a different aspect and with three 60-degree (analytic) corners."""
    return make_case("eq_tri", eq_tri(l), a, b, eq_tri_spectrum(a, b, l), complete=True)


#: The roster, cheapest first. Deliberately small: two provable-coverage domains and the one
#: case that could invalidate the design.
ROSTER = {"disk": disk_case, "eq_tri": eq_tri_case, "L_shape": l_shape_case}


def all_cases():
    return [f() for f in ROSTER.values()]
