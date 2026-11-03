"""Is the Hadamard rule what limits the L-shape's gradient?  (2026-11-03)

`integral_quad(..., family="integer")` reports a model shortfall at the re-entrant corner of
3e-7 to 9e-6 at every prec, and `MPSEigensolver.hadamard` does not read it. Here the SAME
eigenfunction coefficients are integrated on an independent rule that is exact for the corner's
exponents: at nu = 2/3 every exponent of `(d_n u)^2 V` near the re-entrant vertex is a multiple
of 1/3, so `tau = s^3` makes the integrand analytic in `s`, and plain Gauss-Legendre in `s`
converges geometrically. The other corners are right angles (nu = 2, integer powers), so plain
composite Gauss-Legendre is exact there.

If the gradient on this rule satisfies the translation identity and matches finite differences
to ~1e-13, the shipped rule is the limit.

Run:  PYTHONPATH=src python notes/hadamard_exact_rule.py
"""
import sys
import json
import warnings
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "notes")
from hadamard_accuracy import SHAPES, velocity, identities  # noqa: E402
from chladni.eigenproblem import Eigenproblem  # noqa: E402
from chladni.geometry import Polygon  # noqa: E402

warnings.filterwarnings("ignore")


def exact_rule(dom, reentrant, n_gl=60, n_panels=4):
    """Nodes on every segment: `tau = s^3` graded toward `reentrant` on the two edges touching
    it, composite Gauss-Legendre elsewhere. Returns a Quadrature-like namespace with the
    provenance `hadamard`'s velocity callable needs (curve index, tau)."""
    x, w = np.polynomial.legendre.leggauss(n_gl)
    x, w = 0.5 * (x + 1), 0.5 * w
    segs = [s for b in dom.boundaries for s in b.segments]
    P, N, T, W, TAU, CUR = [], [], [], [], [], []
    for ci, s in enumerate(segs):
        p0, p1 = complex(s.p0), complex(s.p1)
        Lseg = abs(p1 - p0)
        tan = (p1 - p0) / Lseg
        for pnl in range(n_panels):
            a, b = pnl / n_panels, (pnl + 1) / n_panels
            u = a + (b - a) * x
            wu = (b - a) * w
            # Points are formed FROM THE CORNER END: p0 + tau (p1 - p0) with tau = 1 - O(1e-12)
            # cancels to nothing next to a corner at tau = 1 (rules.NODE_FLOOR's warning).
            if abs(p0 - reentrant) < 1e-12:          # corner at tau = 0
                tau, dt = u ** 3, 3 * u ** 2 * wu
                pts = p0 + tau * (p1 - p0)
            elif abs(p1 - reentrant) < 1e-12:        # corner at tau = 1
                d = (1 - u) ** 3                     # distance from the corner, in tau
                tau, dt = 1 - d, 3 * (1 - u) ** 2 * wu
                pts = p1 + d * (p0 - p1)
            else:
                tau, dt = u, wu
                pts = p0 + tau * (p1 - p0)
            P.append(pts)
            N.append(np.full(len(tau), -1j * tan))
            T.append(np.full(len(tau), tan))
            W.append(dt * Lseg)
            TAU.append(tau)
            CUR.append(np.full(len(tau), ci))
    cat = np.concatenate
    return SimpleNamespace(pts=cat(P), normals=cat(N), tangents=cat(T), wts=cat(W),
                           tau=cat(TAU), curve=cat(CUR), bc=np.zeros(sum(map(len, P))))


def hadamard_on(rec, level, V, q):
    ef = rec.solver.eigenfunctions(level)
    k = np.sqrt(float(level.lam))
    _, G = rec.solver._pencil.basis.at(q).values_and_grad(k)
    un = (G.real * q.normals.real[:, None] + G.imag * q.normals.imag[:, None]) @ ef.coef[:, 0]
    vals = np.zeros((V(0, np.zeros(1)).shape[0], len(q.wts)))
    for c in np.unique(q.curve):
        m = q.curve == c
        vals[:, m] = V(int(c), q.tau[m])
    return -(vals * (un ** 2 * q.wts)).sum(axis=1)


def run(prec=1e-12, levels=(0, 1, 2, 3, 8)):
    z = np.asarray(SHAPES["L-shape"], dtype=complex)
    dom = Polygon(z, bc="dir")
    rec = Eigenproblem(dom).solve(10, prec=prec)
    V = velocity(dom, z)
    rows = []
    for i in levels:
        L = rec.spectrum.levels[i]
        g_ship = rec.solver.hadamard(L, V)[:, 0, 0]
        g60 = hadamard_on(rec, L, V, exact_rule(dom, 0j, 60, 4))
        g90 = hadamard_on(rec, L, V, exact_rule(dom, 0j, 90, 6))
        s = np.abs(g90).max()
        n = len(z)
        idn = {t: identities(z, float(L.lam), g) for t, g in
               (("shipped", g_ship), ("exact", g90))}
        row = dict(level=i, lam=float(L.lam),
                   exact_rule_converged=float(np.abs(g60 - g90).max() / s),
                   shipped_vs_exact=float(np.abs(g_ship - g90).max() / s),
                   worst_coord=int(np.argmax(np.abs(g_ship - g90))),
                   translation_shipped=max(idn["shipped"]["tx"], idn["shipped"]["ty"]) * n,
                   translation_exact=max(idn["exact"]["tx"], idn["exact"]["ty"]) * n,
                   dilation_exact=idn["exact"]["dil"], g_exact=list(map(float, g90)),
                   g_shipped=list(map(float, g_ship)))
        rows.append(row)
        print(f"level {i} lam {row['lam']:.6f}: exact rule settled {row['exact_rule_converged']:.1e}"
              f" | shipped vs exact {row['shipped_vs_exact']:.1e} (worst coord "
              f"{row['worst_coord']}) | translation: shipped {row['translation_shipped']:.1e}, "
              f"exact {row['translation_exact']:.1e} | dilation exact {row['dilation_exact']:.1e}",
              flush=True)
    return rows




# --- convex corners: an hp-graded reference rule ----------------------------------------------
#
# At a convex corner the integrand is ~ r^gamma, gamma = 2 nu - 2 > 0 irrational: bounded but not
# smooth. Geometric panels toward each end (ratio `sigma`, down to `floor` in tau) with
# `n_gl`-point Gauss-Legendre on each converge root-exponentially; the piece below `floor` is
# O(floor^(gamma+1)) < floor, negligible. Points are formed from the nearer end, as above.

def graded_rule(dom, n_gl=24, sigma=0.15, floor=1e-15, n_mid=4):
    x, w = np.polynomial.legendre.leggauss(n_gl)
    x, w = 0.5 * (x + 1), 0.5 * w
    segs = [s for b in dom.boundaries for s in b.segments]
    edges = [0.5]
    while edges[-1] > floor:
        edges.append(edges[-1] * sigma)
    edges = np.array(edges[::-1])                         # distances from an end, increasing
    P, N, T, W, TAU, CUR = [], [], [], [], [], []
    for ci, s in enumerate(segs):
        p0, p1 = complex(s.p0), complex(s.p1)
        Lseg = abs(p1 - p0)
        tan = (p1 - p0) / Lseg
        for end, (pc, sgn) in enumerate(((p0, 1.0), (p1, -1.0))):
            for a, b in zip(edges[:-1], edges[1:]):
                d = a + (b - a) * x                       # distance from this end, in tau
                P.append(pc + sgn * d * (p1 - p0))
                TAU.append(d if end == 0 else 1 - d)
                W.append((b - a) * w * Lseg)
                CUR.append(np.full(n_gl, ci))
                N.append(np.full(n_gl, -1j * tan))
                T.append(np.full(n_gl, tan))
    cat = np.concatenate
    return SimpleNamespace(pts=cat(P), normals=cat(N), tangents=cat(T), wts=cat(W),
                           tau=cat(TAU), curve=cat(CUR), bc=np.zeros(sum(map(len, P))))


def run_convex(shapes=("triangle", "hexagon (irregular)", "octagon (irregular)",
                       "flat triangle"), prec=1e-10, n=10):
    rows = []
    for name in shapes:
        z = np.asarray(SHAPES[name], dtype=complex)
        dom = Polygon(z, bc="dir")
        rec = Eigenproblem(dom).solve(n, prec=prec)
        V = velocity(dom, z)
        qa, qb = graded_rule(dom, 24, 0.15), graded_rule(dom, 32, 0.1)
        for i, L in enumerate(rec.spectrum.levels[:n]):
            if L.mult != 1:
                continue
            g_ship = rec.solver.hadamard(L, V)[:, 0, 0]
            ga, gb = hadamard_on(rec, L, V, qa), hadamard_on(rec, L, V, qb)
            s = np.abs(gb).max()
            nv = len(z)
            ide, ids = identities(z, float(L.lam), gb), identities(z, float(L.lam), g_ship)
            row = dict(shape=name, prec=prec, level=i, lam=float(L.lam),
                       rel_bound=float(L.bound / L.lam),
                       ref_settled=float(np.abs(ga - gb).max() / s),
                       shipped_vs_ref=float(np.abs(g_ship - gb).max() / s),
                       identity_ref=max(ide["tx"] * nv, ide["ty"] * nv, ide["rot"], ide["dil"]),
                       identity_shipped=max(ids["tx"] * nv, ids["ty"] * nv, ids["rot"],
                                            ids["dil"]))
            rows.append(row)
            print(f"{name:20s} lvl {i} lam {row['lam']:10.4f} bound {row['rel_bound']:.0e} | "
                  f"ref settled {row['ref_settled']:.0e} | SHIPPED vs REF "
                  f"{row['shipped_vs_ref']:.1e} | identity: shipped "
                  f"{row['identity_shipped']:.0e}, ref {row['identity_ref']:.0e}", flush=True)
    return rows


if __name__ == "__main__":
    out = {"L-shape": run(), "convex": run_convex()}
    with open("notes/hadamard_exact_rule.json", "w") as f:
        json.dump(out, f, indent=1)
