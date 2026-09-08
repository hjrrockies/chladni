"""Tension curves on REAL domains with REAL corner-adapted bases.

Everything measured before this used either the surrogate, a rectangle with plane waves, or
the L-shape. The solver has to work on domains with SEVERAL singular corners, and
Fourier-Bessel at those corners is what makes any of it work. This module is the lab for
that: one basis builder, one scanner, and the domain roster.

A corner is singular when its local orders are not all integers, which under a single
boundary condition is just "`pi/omega` is not an integer". `CornerModes.is_singular` answers
that; the geometry layer deliberately does not, and offers `Corner.is_reflection_closed` --
`omega = pi/m` -- as the angle half of the test. The dangerous ones are REENTRANT,
`omega > pi`, where `pi/omega < 1` and the gradient blows up.

Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

from chladni.helmholtz import FourierBessel, PlaneWave
from chladni.mps import MPSPencil

__all__ = ["nonanalytic", "basis_for", "make", "scan", "dips", "ROSTER"]


def nonanalytic(dom):
    """Corners whose Dirichlet expansion `r**(l*pi/omega)` has a fractional exponent."""
    return [c for c in dom.corners if not c.is_reflection_closed]


def basis_for(dom, lam_hi, fb=40, pw_pad=1.15):
    """Fourier-Bessel at every non-analytic corner, plane waves for the rest.

    `fb` terms per corner. The plane-wave count comes from `n_useful`, which is the number of
    directions that carry information at this wavenumber -- more is not better, it is just
    rank the truncation then discards.
    """
    k = np.sqrt(lam_hi)
    R = dom.diameter
    b = PlaneWave.uniform(int(pw_pad * PlaneWave.n_useful(k * R)), radius=R)
    for c in nonanalytic(dom):
        b = b + FourierBessel.at_corner(c, n=fb, domain=dom)
    return b


#: The basis specs recorded by `chladni.reference.recipe()`, i.e. what lappy needed to
#: produce the tables. (n_basis, kind, fb_frac, offset_distance/diameter). `fb_frac` and the
#: offset distance are not in the recipe text and were fitted here against the tables.
#:
#: USE THESE whenever testing anything precision-sensitive on these domains. The generic
#: `basis_for` below is 20x to 5e+05 worse -- see `landscape_findings` section 28.
RECIPES = {
    "L_shape":  (80,  "pure",  1.00, 0.00),
    "mushroom": (240, "mixed", 0.75, 0.30),
    "GWW1":     (320, "mixed", 0.60, 0.10),
    "H_shape":  (320, "mixed", 0.60, 0.10),
    "chevron":  (160, "mixed", 0.60, 0.30),
    "cut_sq":   (320, "pure",  1.00, 0.00),
}


def recipe_basis(dom, label):
    """The reference-grade basis for a tabulated domain, or None if there is no recipe.

    Fourier-Bessel at every non-analytic corner plus, for the mixed specs, fundamental
    solutions offset from the boundary. Verified against the tables:

        L_shape   n_cols  80   tension 2.1e-15 .. 2.1e-14   spread 1.0e+01
        mushroom          240          2.4e-11 .. 3.4e-11          1.4e+00
        GWW1              320          6.3e-12 .. 2.7e-11          4.2e+00
        H_shape           320          6.1e-13 .. 5.2e-10          8.6e+02
        chevron           160          1.4e-10 .. 2.0e-09          1.5e+01
    """
    from chladni.helmholtz import Fundamental

    spec = RECIPES.get(label)
    if spec is None:
        return None
    n_basis, kind, fb_frac, dist = spec
    sing = nonanalytic(dom)
    if not sing:
        return None
    per = max(int((fb_frac if kind == "mixed" else 1.0) * n_basis / len(sing)), 4)
    b = None
    for c in sing:
        f = FourierBessel.at_corner(c, n=per, domain=dom)
        b = f if b is None else b + f
    if kind == "mixed":
        n_fs = max(n_basis - per * len(sing), 8)
        b = b + Fundamental.offset(dom.bdry_quad(n_fs), dist * dom.diameter, domain=dom)
    return b


def make(dom, lam_lo, lam_hi, *, fb=40, nb=None, ni=None, seed=0, freeze=False,
         label=None):
    """A pencil sized for `[lam_lo, lam_hi]`.

    `label` selects the reference-grade `recipe_basis` when one exists; without it you get
    the generic `basis_for`, which is 20x to 5e+05 weaker on every tabulated corner domain.
    Threshold rank by default: a frozen rank over a wide window is set by the LOW end and
    starves the high end -- section 27.1.
    """
    b = (recipe_basis(dom, label) if label else None) or basis_for(dom, lam_hi, fb=fb)
    nb = nb if nb is not None else max(8 * len(b), 800)
    ni = ni if ni is not None else max(2 * len(b), 300)
    return MPSPencil.from_basis(
        b, dom.bdry_quad(nb), dom.interior_pts(ni, rng=np.random.default_rng(seed)),
        lam_ref=(lam_lo, lam_hi) if freeze else None)


def scan(p, lam_lo, lam_hi, n, nbranch=3):
    lams = np.linspace(lam_lo, lam_hi, n)
    S = np.full((n, nbranch), np.inf)
    for i, x in enumerate(lams):
        t = p.tensions(x)
        S[i, :min(nbranch, len(t))] = t[:nbranch]
    return lams, S


def dips(lams, s1, frac=0.5):
    """Discrete local minima that are actually dips, not plateau wobble."""
    i = np.nonzero((s1[1:-1] < s1[:-2]) & (s1[1:-1] < s1[2:]))[0] + 1
    return i[s1[i] < frac * np.median(s1)]


def half_width(lams, s1, i, plateau):
    """Width of the dip at `i` where sigma_1 crosses plateau/2, in lam. NaN if it runs out."""
    lvl = 0.5 * plateau
    L = R = None
    for j in range(i, 0, -1):
        if s1[j] >= lvl:
            L = np.interp(lvl, [s1[j - 1] if False else s1[j], s1[j + 1]],
                          [lams[j], lams[j + 1]]) if False else lams[j]
            break
    for j in range(i, len(s1)):
        if s1[j] >= lvl:
            R = lams[j]
            break
    if L is None or R is None:
        return np.nan
    return 0.5 * (R - L)


# label, factory, the lam window to study, and what makes it interesting
ROSTER = [
    ("L_shape",  "L_shape()",      (5.0, 60.0),   "1 reentrant, the reference case"),
    ("mushroom", "mushroom()",     (5.0, 45.0),   "2 reentrant, a near-decoupled neck"),
    ("GWW1",     "GWW1()",         (2.0, 20.0),   "2 reentrant + 3pi/4, dense spectrum"),
    ("H_shape",  "H_shape()",      (2.0, 30.0),   "4 reentrant, near-symmetric"),
    ("plus",     "plus_shape()",   (3.0, 40.0),   "4 reentrant, D4 symmetry"),
    ("chevron",  "chevron()",      (10.0, 90.0),  "a 18-degree spike and a reentrant"),
    ("rect81",   "rect(8.0, 1.0)", (2.0, 40.0),   "no singular corner, clustered by aspect"),
]


def domain(expr):
    import chladni.geometry as g
    return eval(expr, {"__dict__": None}, g.__dict__)
