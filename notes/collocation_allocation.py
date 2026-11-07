"""Basis-aware boundary collocation: the measurements behind `chladni.mps.bdry_counts`.

    python notes/collocation_allocation.py starve 0.05 0.2   # length vs per-side, thin isosceles
    python notes/collocation_allocation.py conditioning      # edge-on vs face-on FB columns
    python notes/collocation_allocation.py census            # which plans the new split changes

`starve`: the isosceles triangle with apex `omega`, unit legs, 100 Fourier-Bessel terms per
corner, prec 1e-14, 900 boundary points. Split by length (the old default) and with the base
raised to 2 x 100 (what `bdry_counts` gives); tension over the window and the search's lam_1.

`conditioning`: one corner's FB family (N columns, k = N), 2N points on a radial segment along
the bisector (edge-on) and on an arc across the wedge (face-on) of the SAME length. The ratio
of extreme singular values of the column-normalized matrix.

`census`: `plan_for` on every rung for library shapes, thin triangles and douse's stalled
hexagon; per plan, whether the frozen counts differ from `allocate(n_bdry, lengths)`.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")


def starve(omegas):
    from chladni.bounds import faber_krahn, inradius_upper
    from chladni.geometry import Polygon
    from chladni.geometry._adaptive import allocate
    from chladni.mps import MPSEigensolver
    from chladni.plan import Plan

    prec = 1e-14
    for w in omegas:
        dom = Polygon([0.0, 1.0, np.exp(1j * w)], prec=prec)
        basis, nb, ni = Plan(prec=prec, k_max=np.nan, corner_orders=(100,) * 3).realize(dom)
        share = allocate(nb, np.array([s.length for s in dom.outer.segments]))
        a, b = 0.5 * faber_krahn(dom), 1.05 * inradius_upper(dom)
        for name, counts in (("length", share), ("per-side", np.maximum(share, 200))):
            s = MPSEigensolver(dom, basis=basis, bdry_quad=dom.outer.quad(n_per_seg=counts),
                               n_int=ni, prec=prec)
            t0 = time.perf_counter()
            t = s.tension(np.linspace(a, b, 200))
            res = s.search_interval(a, b)
            lv = min(res.levels, key=lambda L: L.lam) if res.levels else None
            out = (f"lam={lv.lam:.14g} mp/lam={lv.mp_bound / lv.lam:.1e}" if lv else "no level")
            print(f"omega={w:.3f} {name:8s} counts={[int(c) for c in counts]} tension min/med "
                  f"{t.min():.1e}/{np.median(t):.1e} | {out}"
                  f"{' DECLINED' if res.diagnostics.declined else ''} "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)


def conditioning():
    from chladni.helmholtz import FourierBessel
    from chladni.helmholtz.corners import CornerModes

    for omega in (np.pi / 2, 0.75 * np.pi):
        modes = CornerModes.from_wedge(omega, 0.0, 0.0)
        arc = 0.9 * 0.96 * omega                    # the face-on arc's length, at r = 0.9
        for N in (20, 40):
            fam = FourierBessel(0.0, 1.0, modes, np.arange(1, N + 1), radius=0.3 + arc)
            s = np.linspace(0, 1, 2 * N)
            edge = (0.2 + arc * s) * np.exp(0.5j * omega)
            face = 0.9 * np.exp(1j * (0.02 + 0.96 * s) * omega)
            row = []
            for name, z in (("edge-on", edge), ("face-on", face)):
                V = fam.values(float(N), z)
                sv = np.linalg.svd(V / np.linalg.norm(V, axis=0), compute_uv=False)
                row.append(f"{name} {sv[-1] / sv[0]:.1e}")
            print(f"omega={omega / np.pi:.2f}pi N={N} k={N} 2N points: " + "   ".join(row))


def census():
    from chladni.geometry import (GWW1, H_shape, L_shape, Polygon, chevron, eq_tri, iso_right_tri,
                                  parallelogram, plus_shape, rect, reg_ngon, right_trapezoid)
    from chladni.geometry._adaptive import allocate
    from chladni.plan import LADDER, plan_for

    def tri(deg):
        a = np.radians(deg)
        return Polygon([0.0, 1.0, np.exp(1j * a)])

    shapes = {
        "L": L_shape(), "eq_tri": eq_tri(), "iso_right": iso_right_tri(), "square": rect(),
        "rect 4:1": rect(4.0, 1.0), "pentagon": reg_ngon(5), "octagon": reg_ngon(8),
        "GWW1": GWW1(), "plus": plus_shape(), "H": H_shape(), "chevron": chevron(),
        "trapezoid": right_trapezoid(), "parallelogram": parallelogram(),
        "apex 30": tri(30), "apex 13": tri(13), "apex 5": tri(5), "apex 150": tri(150),
        "apex 170": tri(170),
        "stall hexagon": Polygon([0.0417, 1.1091 + 0.5169j, 0.9097 + 1.1191j, 0.2018 + 1.2364j,
                                  -0.2318 + 0.6575j, 0.0]),
    }
    for name, dom in shapes.items():
        L = np.array([s.length for b in dom.boundaries for s in b.segments])
        cells = []
        for rung in range(len(LADDER)):
            try:
                p = plan_for(dom, 10, 1e-10, rung=rung)
            except Exception as e:  # noqa: BLE001
                cells.append(f"r{rung}: {type(e).__name__}")
                continue
            if not p.bdry_counts:
                cells.append(f"r{rung}: empty")
                continue
            n_bdry = int(np.ceil(p.bdry_ratio * p.n_cols))
            old = allocate(n_bdry, L)
            new = np.array(p.bdry_counts)
            if np.array_equal(old, new):
                cells.append(f"r{rung}: same")
            else:
                i = int(np.argmax(new - old))
                cells.append(f"r{rung}: +{int((new - old).sum())} pts (side {i}: "
                             f"{int(old[i])}->{int(new[i])})")
        print(f"{name:14s} " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "census"
    if what == "starve":
        starve([float(a) for a in sys.argv[2:]] or [0.05, 0.2])
    elif what == "conditioning":
        conditioning()
    else:
        census()
