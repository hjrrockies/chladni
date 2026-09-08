"""Landscape lab: measure the tension objective before designing a search for it.

Exploratory. Confirmed facts get promoted to tests/ and docs/design/solver.md; this file
is a run log and is not tracked.
"""
import sys, warnings
sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
from chladni.geometry import Polygon, disk, eq_tri, rect
from chladni.helmholtz import CornerModes, FourierBessel, PlaneWave
from chladni.mps import MPSPencil
from conftest import (L_SHAPE, disk_eigs, eq_tri_eigs, rect_eigs, sector_eigs)


def basis_for(dom, kmax, fb_degrees=14):
    """A deliberately generic basis: plane waves sized by n_useful, plus the corner family
    at every singular corner. NOT the deferred planner -- just something defensible."""
    R = dom.diameter
    b = PlaneWave.uniform(int(PlaneWave.n_useful(kmax * R)), radius=R)
    sing = [c for c in dom.corners if CornerModes.from_corner(c).is_singular]
    for c in sing:
        b = b + FourierBessel.at_corner(c, n=fb_degrees, domain=dom)
    return b


def make(dom, klo, khi, nb=600, ni=200, fb=14, seed=0):
    b = basis_for(dom, khi, fb)
    return MPSPencil.from_basis(b, dom.bdry_quad(nb),
                                dom.interior_pts(ni, rng=np.random.default_rng(seed)),
                                lam_ref=(klo**2, khi**2))


def scan(p, klo, khi, n):
    ks = np.linspace(klo, khi, n)
    # the pencil takes `lam`; `k` stays this lab's scan variable, so square at the call
    return ks, np.array([p.tensions(k**2)[:3] for k in ks])


def dips(ks, s1):
    i = np.nonzero((s1[1:-1] < s1[:-2]) & (s1[1:-1] < s1[2:]))[0] + 1
    return i[s1[i] < 0.5 * np.median(s1)]
