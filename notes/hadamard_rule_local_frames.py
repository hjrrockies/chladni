"""Per-corner LOCAL FRAMES for the Hadamard integral near a re-entrant corner.  (2026-11-03)

Each node on an edge touching the re-entrant vertex z_c is evaluated by `basis.translated(-z_c)`
at its local offset `d e` (d the exact planned distance), with the SAME coefficient vector.
Elsewhere, absolute coordinates as usual. Substitution tau = s^p, p = 1/(2 nu - 1), Gauss-Legendre
in s. Compared against the same rule in absolute coordinates.

Cases: the L-shape (270 deg, nu = 2/3, p = 3) translated off the origin, and a 315-deg notch
(nu = 4/7, p = 7: a square [-2,2]^2 minus the wedge between rays 0 and -45 deg at the vertex).

Run:  PYTHONPATH=src python notes/hadamard_rule_local_frames.py
"""
import sys
import warnings

import numpy as np

sys.path.insert(0, "notes")
from hadamard_accuracy import velocity, identities  # noqa: E402
from chladni.eigenproblem import Eigenproblem  # noqa: E402
from chladni.geometry import Polygon  # noqa: E402

warnings.filterwarnings("ignore")


def rule(dom, zc, p, n_gl=60, n_panels=4):
    """Nodes per segment; on corner edges: (local offset, tau, weight); elsewhere absolute."""
    x, w = np.polynomial.legendre.leggauss(n_gl)
    x, w = 0.5 * (x + 1), 0.5 * w
    out = []
    for ci, s in enumerate([s for b in dom.boundaries for s in b.segments]):
        p0, p1 = complex(s.p0), complex(s.p1)
        Lg = abs(p1 - p0)
        tan = (p1 - p0) / Lg
        nrm = -1j * tan
        for k in range(n_panels):
            u = (k + x) / n_panels
            wu = w / n_panels
            at0, at1 = abs(p0 - zc) < 1e-12, abs(p1 - zc) < 1e-12
            if at0 or at1:
                d = u ** p                                  # distance from zc, in units of Lg
                wt = p * u ** (p - 1) * wu * Lg
                dirn = (p1 - p0) if at0 else (p0 - p1)
                local = d * dirn                            # exact-ish offset from zc
                out.append(dict(curve=ci, tau=d if at0 else 1 - d, w=wt, nrm=nrm,
                                local=local, absolute=zc + local))
            else:
                pts = p0 + u * (p1 - p0)
                out.append(dict(curve=ci, tau=u, w=wu * Lg, nrm=nrm, local=None, absolute=pts))
    return out


def hadamard(rec, level, V, nodes, zc, local_frames):
    ef = rec.solver.eigenfunctions(level)
    c = ef.coef[:, 0]
    k = np.sqrt(float(level.lam))
    B = rec.solver._pencil.basis
    Bt = B.translated(-zc)
    H = 0.0
    for nd in nodes:
        if local_frames and nd["local"] is not None:
            _, G = Bt.at(nd["local"]).values_and_grad(k)
        else:
            _, G = B.at(nd["absolute"]).values_and_grad(k)
        un = (G.real * nd["nrm"].real + G.imag * nd["nrm"].imag) @ c
        H = H - V(nd["curve"], nd["tau"]) @ (un ** 2 * nd["w"])
    return H


def run():
    Lshape = np.array([0, 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1], complex)
    notch = np.array([2, 2 + 2j, -2 + 2j, -2 - 2j, 2 - 2j, 0], complex)
    for name, base, nu, ci, levels in (("L-shape 270", Lshape, 2 / 3, 0, (0, 8)),
                                       ("notch 315", notch, 4 / 7, 5, (0, 1, 2))):
        p = 1.0 / (2 * nu - 1)
        for shift in (0j, 3 + 2j):
            z = base + shift
            zc = z[ci]
            dom = Polygon(z, bc="dir")
            try:
                rec = Eigenproblem(dom).solve(10, prec=1e-10)
            except Exception as e:                        # noqa: BLE001
                print(f"{name} shift {shift}: solve failed: {e}")
                continue
            V = velocity(dom, z)
            for n_gl, n_p in ((40, 3), (60, 4)):
                nodes = rule(dom, zc, p, n_gl, n_p)
                rmin = min(np.abs(nd["local"]).min() for nd in nodes if nd["local"] is not None)
                for lf in (False, True):
                    out = []
                    for i in levels:
                        L = rec.spectrum.levels[i]
                        if L.mult != 1:
                            continue
                        try:
                            with np.errstate(all="ignore"):
                                g = hadamard(rec, L, V, nodes, zc, lf)
                        except (OverflowError, FloatingPointError, ValueError) as e:
                            out.append(f"lvl {i}: FAILS ({type(e).__name__}: nodes round onto "
                                       f"the vertex)")
                            continue
                        if not np.all(np.isfinite(g)):
                            out.append(f"lvl {i}: NON-FINITE")
                            continue
                        r = identities(z, float(L.lam), g)
                        out.append(f"lvl {i}: transl {max(r['tx'], r['ty']) * len(z):.0e} "
                                   f"dil {r['dil']:.0e}")
                    print(f"{name} shift {shift!s:7s} rule {n_gl}x{n_p} min r {rmin:.0e} "
                          f"{'LOCAL ' if lf else 'absolute'} | " + " | ".join(out), flush=True)


if __name__ == "__main__":
    run()
