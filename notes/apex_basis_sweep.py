"""Basis sweep on one triangle at a few reference levels (2026-10-26).

Each basis is searched in narrow windows around reference eigenvalues, so a basis costs a few
seconds per level instead of a full prefix solve. A spec is `s0,s1,s2[:density,offset]`: the
corner-order scale at each vertex (0 = none; the precision rule of `chladni.plan.corner_order`),
plus an optional offset source layer as in the ladder's rungs 1-2, plus an optional edge patch
`/i,m` (`chladni.plan._edge_patch`: m sources around segment i, segment 0 = vertex 0 -> 1), plus
optional collocation ratios `@bdry,int` (default 3,1), plus optional lightning `~n,sigma,reach`
(n poles per FB corner, `chladni.plan.LIGHTNING_SIGMA`/`LIGHTNING_REACH` overridden).

usage: python notes/apex_basis_sweep.py ax ay "lam1,lam2,..." spec [spec ...]
"""
import sys, time
import numpy as np
from chladni.geometry import Polygon
import chladni.plan as cp
from chladni.plan import plan_for, corner_order, Plan

ax, ay = float(sys.argv[1]), float(sys.argv[2])
refs = [float(x) for x in sys.argv[3].split(",")]
prec, n = 1e-10, 15
dom = Polygon(np.array([0, 1, ax + 1j * ay]), bc="dir")
k_max = plan_for(dom, n, prec, 0).k_max
zs = np.array([c.z for c in dom.corners])
for spec in sys.argv[4:]:
    spec_, _, light = spec.partition("~")
    n_light = 0
    if light:
        nl, sg, rc = light.split(",")
        n_light, cp.LIGHTNING_SIGMA, cp.LIGHTNING_REACH = int(nl), float(sg), float(rc)
    spec_, _, colloc = spec_.partition("@")
    br, ir = (float(x) for x in colloc.split(",")) if colloc else (3.0, 1.0)
    spec_, _, patch = spec_.partition("/")
    cs, _, src = spec_.partition(":")
    scales = [float(s) for s in cs.split(",")]
    orders = tuple(0 if s == 0 else corner_order(k_max, float(np.max(np.abs(zs - c.z))), c.omega,
                                                 prec, s) for s, c in zip(scales, dom.corners))
    n_src, off = 0, 0.0
    if src:
        dens, off = (float(x) for x in src.split(","))
        n_src = int(np.ceil(dens * k_max * dom.perimeter / (2 * np.pi)))
    patches = (tuple(int(x) for x in patch.split(",")),) if patch else ()
    p = Plan(prec=prec, k_max=k_max, corner_orders=orders, n_sources=n_src, source_offset=off,
             edge_patches=patches, bdry_ratio=br, int_ratio=ir, lightning_n=n_light)
    t = time.time()
    solver = p.solver(dom)
    out = []
    for lam in refs:
        h = 0.01 * lam
        res = solver.search_interval(lam - h, lam + h, on_shortfall="flag")
        L = min(res.levels, key=lambda L: abs(L.lam - lam), default=None)
        out.append("none" if L is None else
                   f"{L.lam:.10f} err {abs(L.lam - lam) / lam:.1e} relb {L.bound / L.lam:.1e} "
                   f"mg {L.margin:.2g}")
    print(f"{spec:24s} cols {p.n_cols:4d} {orders} src {n_src} {time.time() - t:5.0f}s | "
          + " | ".join(out), flush=True)
