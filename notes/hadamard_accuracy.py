"""How accurate are chladni's Hadamard eigenvalue derivatives?  (2026-11-03)

The vertex-coordinate gradient `d lam / d (x_k, y_k)` exactly as douse computes it
(`douse.chladni_backend.ChladniSolver._velocity` + `SolveRecord.hadamard`), checked three ways:

  A. closed form   -- a rectangle's per-vertex partials. With the linear hat velocity, each vertex
                      partial is exactly half the moving side's derivative (sin^2 is symmetric
                      about the side's midpoint): d lam / d x = -/+ pi^2 m^2 / L^3,
                      d lam / d y = -/+ pi^2 n^2 / H^3. All 2N partials, every mode.
  B. identities    -- on ANY polygon: translation (sum_k d/dx_k = sum_k d/dy_k = 0), rotation
                      (sum_k -y_k d/dx_k + x_k d/dy_k = 0), dilation (sum_k z_k . grad_k = -2 lam).
                      Exact, so they need no reference solve.
  C. finite diffs  -- per-vertex partials on generic shapes against Richardson-extrapolated
                      central differences of chladni's own eigenvalues at a tight prec.

Run:  PYTHONPATH=src python notes/hadamard_accuracy.py [A|B|C ...]
"""
import sys
import json
import warnings

import numpy as np

from chladni.eigenproblem import Eigenproblem
from chladni.geometry import Polygon

warnings.filterwarnings("ignore")


def velocity(dom, verts):
    """douse's `ChladniSolver._velocity`, verbatim in substance: the 2N vertex-coordinate fields."""
    segs = [s for b in dom.boundaries for s in b.segments]
    n = len(verts)
    ends = [(int(np.argmin(np.abs(verts - s.p0))), int(np.argmin(np.abs(verts - s.p1))))
            for s in segs]
    normals = [-1j * (s.p1 - s.p0) / abs(s.p1 - s.p0) for s in segs]

    def V(curve, t):
        t = np.asarray(t, dtype=float)
        i0, i1 = ends[curve]
        nrm = normals[curve]
        out = np.zeros((2 * n, len(t)))
        for axis, d in ((0, 1.0 + 0j), (1, 1j)):
            vn = d.real * nrm.real + d.imag * nrm.imag
            out[axis * n + i0] += (1.0 - t) * vn
            out[axis * n + i1] += t * vn
        return out
    return V


def solve(verts, n, prec, plan=None):
    """A solve at `prec`; `plan` freezes the basis (the finite differences reuse the base's)."""
    verts = np.asarray(verts, dtype=complex)
    dom = Polygon(verts, bc="dir")
    rec = Eigenproblem(dom).solve(n, prec=prec, plan=plan)
    return rec, dom


def gradients(verts, n, prec):
    """`(lam, mult, rel_bound, J)` with J `(levels, 2N)` (nan rows for multiple levels)."""
    verts = np.asarray(verts, dtype=complex)
    rec, dom = solve(verts, n, prec)
    V = velocity(dom, verts)
    sp = rec.spectrum
    J = np.full((len(sp.lam), 2 * len(verts)), np.nan)
    for i in range(len(sp.lam)):
        if sp.mult[i] == 1:
            J[i] = rec.hadamard(i, V)[:, 0, 0]
    return np.asarray(sp.lam), np.asarray(sp.mult), np.asarray(sp.bound) / np.asarray(sp.lam), J


# --- A: the rectangle's closed form --------------------------------------------------------------

def rect_exact(L, H, lam):
    """(m, n) and the exact 2N partials for a CCW rectangle (0,0),(L,0),(L,H),(0,H)."""
    best = None
    for m in range(1, 30):
        for k in range(1, 30):
            v = np.pi ** 2 * (m * m / L ** 2 + k * k / H ** 2)
            if best is None or abs(v - lam) < abs(best[0] - lam):
                best = (v, m, k)
    v, m, k = best
    gx, gy = np.pi ** 2 * m * m / L ** 3, np.pi ** 2 * k * k / H ** 3
    sx = np.array([-1, 1, 1, -1])         # x_k: the right side moves outward under +x
    sy = np.array([-1, -1, 1, 1])
    return v, (m, k), -np.concatenate([sx * gx, sy * gy])


def run_A(precs=(1e-8, 1e-10, 1e-12, 1e-13), n=12, L=1.0, H=0.7):
    verts = np.array([0, L, L + 1j * H, 1j * H])
    rows = []
    for prec in precs:
        lam, mult, rb, J = gradients(verts, n, prec)
        for i in range(len(lam)):
            if mult[i] != 1:
                continue
            v, mk, g = rect_exact(L, H, lam[i])
            err = np.max(np.abs(J[i] - g)) / np.max(np.abs(g))
            rows.append(dict(prec=prec, i=i, mn=mk, lam=lam[i], lam_err=abs(lam[i] - v) / v,
                             rel_bound=rb[i], grad_err=err))
            print(f"prec {prec:.0e}  level {i:2d} {mk}  lam rel err {abs(lam[i]-v)/v:.1e}  "
                  f"bound {rb[i]:.1e}  grad rel err {err:.1e}", flush=True)
    return rows


# --- B: invariance identities ------------------------------------------------------------------

def identities(verts, lam, g):
    """Relative residuals of translation (x, y), rotation, dilation, scaled by |z| |grad|."""
    z = np.asarray(verts, dtype=complex)
    n = len(z)
    gx, gy = g[:n], g[n:]
    scale = np.sum(np.abs(z) * np.hypot(gx, gy))           # size of the terms that cancel
    gs = np.max(np.hypot(gx, gy))
    return dict(tx=abs(gx.sum()) / gs / n, ty=abs(gy.sum()) / gs / n,
                rot=abs(np.sum(-z.imag * gx + z.real * gy)) / scale,
                dil=abs(np.sum(z.real * gx + z.imag * gy) + 2 * lam) / (2 * lam))


SHAPES = {
    "rect 1x0.7": [0, 1, 1 + 0.7j, 0.7j],
    "triangle": [0, 1, 0.3 + 0.8j],
    "flat triangle": [0, 1, 0.45 + 0.15j],
    "hexagon (irregular)": [0, 1.0 - 0.1j, 1.6 + 0.5j, 1.3 + 1.3j, 0.4 + 1.5j, -0.3 + 0.7j],
    "L-shape": [0, 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1],
    "octagon (irregular)": [np.exp(2j * np.pi * k / 8) * (1 + 0.15 * np.cos(3 * k + 1))
                            for k in range(8)],
}


def run_B(precs=(1e-8, 1e-10, 1e-12), n=10, shapes=None):
    rows = []
    for name in shapes or SHAPES:
        verts = np.asarray(SHAPES[name], dtype=complex)
        for prec in precs:
            try:
                lam, mult, rb, J = gradients(verts, n, prec)
            except Exception as e:                       # noqa: BLE001 -- a study, report it
                print(f"{name:22s} prec {prec:.0e}: {type(e).__name__}: {e}")
                continue
            for i in range(len(lam)):
                if mult[i] != 1:
                    continue
                r = identities(verts, lam[i], J[i])
                rows.append(dict(shape=name, prec=prec, i=i, lam=lam[i], rel_bound=rb[i], **r))
            ok = [r for r in rows if r["shape"] == name and r["prec"] == prec]
            worst = {k: max(r[k] for r in ok) for k in ("tx", "ty", "rot", "dil")}
            print(f"{name:22s} prec {prec:.0e}  levels {len(ok):2d}  worst bound "
                  f"{max(r['rel_bound'] for r in ok):.1e}  translation "
                  f"{max(worst['tx'], worst['ty']):.1e}  rotation {worst['rot']:.1e}  "
                  f"dilation {worst['dil']:.1e}", flush=True)
    return rows


# --- C: per-vertex partials against extrapolated finite differences ----------------------------

def fd_partials(verts, n, prec, levels, hs=(4e-3, 2e-3, 1e-3)):
    """Richardson-extrapolated central differences of `lam[levels]` for all 2N coordinates.
    Returns (fd_extrapolated, fd_raw_smallest_h)."""
    z = np.asarray(verts, dtype=complex)
    N = len(z)
    plan = solve(z, n, prec)[0].plan
    out = np.empty((len(hs), len(levels), 2 * N))
    for a, h in enumerate(hs):
        for c in range(2 * N):
            d = np.zeros(N, dtype=complex)
            d[c % N] = h if c < N else 1j * h
            lp = solve(z + d, n, prec, plan)[0].spectrum.lam
            lm = solve(z - d, n, prec, plan)[0].spectrum.lam
            out[a, :, c] = (np.asarray(lp)[levels] - np.asarray(lm)[levels]) / (2 * h)
    # two Richardson steps on h, h/2, h/4 (error ~ h^2, h^4)
    r1 = (4 * out[1:] - out[:-1]) / 3
    r2 = (16 * r1[1:] - r1[:-1]) / 15
    return r2[0], r1[-1], out[-1]


def run_C(shapes=("L-shape", "hexagon (irregular)"), n=4, prec_ref=1e-12,
          precs=(1e-10, 1e-12)):
    rows = []
    for name in shapes:
        verts = np.asarray(SHAPES[name], dtype=complex)
        lam, mult, rb, _ = gradients(verts, n, prec_ref)
        levels = [i for i in range(len(lam)) if mult[i] == 1][:n]
        fd2, fd1, fd0 = fd_partials(verts, n, prec_ref, levels)
        for prec in precs:
            lam_p, mult_p, rb_p, J = gradients(verts, n, prec)
            for j, i in enumerate(levels):
                gs = np.max(np.abs(fd2[j]))
                e = np.max(np.abs(J[i] - fd2[j])) / gs
                rich = np.max(np.abs(fd2[j] - fd1[j])) / gs        # how settled the FD is
                rows.append(dict(shape=name, prec=prec, i=i, lam=lam[i], rel_bound=rb_p[i],
                                 err=e, fd_settle=rich,
                                 per_coord=list(np.abs(J[i] - fd2[j]) / gs)))
                print(f"{name:22s} prec {prec:.0e} level {i}  lam {lam[i]:.6f}  bound "
                      f"{rb_p[i]:.1e}  |H - FD|/|g| {e:.1e}  (FD settled to {rich:.1e})",
                      flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    res = {}
    for w in which:
        print(f"=== {w} ===", flush=True)
        res[w] = globals()[f"run_{w}"]()
    with open(f"notes/hadamard_accuracy_{''.join(which)}.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
