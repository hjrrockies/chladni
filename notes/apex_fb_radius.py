"""Does the FB columns' scaling radius matter on the apex-(0.1, 0.25) triangle? (2026-10-26)

`Plan.realize` scales each corner's columns at `evaluation_radius` (the farthest boundary point,
~1 here). The FB-only residual sits on the 0.27-long edge between vertex 0 and the apex, which is
inside each of those two corners' expansions' convergence disk only out to the OTHER corner. This
rebuilds the same columns with the scaling radius at each corner set to a multiple of its
distance to the nearest other vertex, and compares at lam_1.
usage: python notes/apex_fb_radius.py scale src_density [radius_mode ...]
  radius_mode: "eval" (shipped) or a float f (radius = f x nearest-vertex distance)
"""
import sys, numpy as np
from chladni.geometry import Polygon
from chladni.plan import plan_for, corner_order, _offset_sources
from chladni.helmholtz import FourierBessel
from chladni.mps import MPSEigensolver

prec = 1e-10
scale, dens = float(sys.argv[1]), float(sys.argv[2])
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
k_max = plan_for(dom, 15, prec, 0).k_max
zs = np.array([c.z for c in dom.corners])
refs = [288.9643854029, 1293.9127901997, 2220.6128169992]
for mode in sys.argv[3:]:
    parts = []
    for c in dom.corners:
        d = np.abs(zs - c.z)
        n = corner_order(k_max, float(d.max()), c.omega, prec, scale)
        rad = None if mode == "eval" else float(mode) * float(d[d > 0].min())
        parts.append(FourierBessel.at_corner(c, n=n, domain=dom, radius=rad, prec=prec))
    if dens:
        parts.append(_offset_sources(dom, int(np.ceil(dens * k_max * dom.perimeter / (2 * np.pi))),
                                     0.05, prec))
    basis = parts[0]
    for f in parts[1:]:
        basis = basis + f
    m = len(basis)
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * m, n_int=m, prec=prec, rng=0)
    out = []
    for lam in refs:
        res = s.search_interval(0.99 * lam, 1.01 * lam)
        L = min(res.levels, key=lambda L: abs(L.lam - lam))
        out.append(f"err {abs(L.lam - lam) / lam:.1e} relb {L.bound / L.lam:.1e} mg {L.margin:.2g}")
    print(f"scale {scale} src {dens} radius {mode:5s} cols {m} | " + " | ".join(out), flush=True)
