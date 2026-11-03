"""How accurate is the Rellich-identity normalization of eigenfunctions?  (2026-11-03)

Phase 2 of the boundary-integrals plan (`notes/hadamard_accuracy.md`). `MPSEigensolver.
eigenfunctions` forms the Gram of a level's MPS columns from the full Rellich identity on the
`'even'` integral rule and Löwdin-orthonormalizes. Every Hadamard matrix scales with `1/||u||^2`,
and the Moler-Payne norm lower bound reads the same identity, so its error matters twice.

Ground truth here never touches a chladni boundary rule:

  A. INTERIOR GRAM. The returned (supposedly orthonormal) eigenfunctions integrated over Omega
     on an interior cubature: a fan of triangles from the centroid, each a Duffy square graded
     toward its two polygon corners and the far edge (convex shapes); a tensor rule graded toward
     the re-entrant corner (the L-shape, from `tests/test_rellich.py`). `|G_int - I|` is the true
     normalization/orthogonality error; it is set beside `gram_residual`, the detector.
  B. CLOSED-FORM SECTOR NORMS. `J_nu(k r) sin(nu theta)` on a sector of opening alpha*pi, its
     norm in closed form, integrated by the Rellich identity on the shipped rule -- at the origin
     AND translated by 3+2i, to see whether the rule's accuracy depends on where the apex sits.
  C. THE FLAT TRIANGLE's gram_residual 0.16 at prec 1e-12: the Gram split into its four terms,
     cancellation on the rule, and the boundary residual |u| at the rule's nodes.

Run:  PYTHONPATH=src python notes/gram_accuracy.py [A] [B] [C]
"""
import sys
import json
import warnings

import numpy as np

sys.path.insert(0, "notes")
from hadamard_accuracy import SHAPES  # noqa: E402
from chladni import reference as ref  # noqa: E402
from chladni.eigenproblem import Eigenproblem  # noqa: E402
from chladni.geometry import Polygon, disk_sector, polyomino  # noqa: E402
from chladni.geometry._adaptive import gauss_legendre  # noqa: E402
from chladni.integrals import cauchy as C  # noqa: E402
from chladni.integrals import identities as I  # noqa: E402
from chladni.integrals import plan as P  # noqa: E402
from chladni.integrals import certify  # noqa: E402

warnings.filterwarnings("ignore")


# --- interior cubatures ------------------------------------------------------------------------

def _graded_both(layers=12, ratio=0.25, order=20):
    """Composite Gauss on [0, 1], panels shrinking geometrically toward BOTH ends."""
    half = np.concatenate([[0.0], 0.5 * ratio ** np.arange(layers, 0, -1), [0.5]])
    edges = np.concatenate([half, 1.0 - half[::-1][1:]])
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = gauss_legendre(order)
        xs.append(a + (b - a) * x)
        ws.append((b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def fan_cubature(verts, **kw):
    """Convex polygon: triangles (c, z_k, z_k+1), Duffy-mapped from [0,1]^2 with s toward the
    edge, graded at both ends of s and t (the corners sit at s = 1, t = 0 and t = 1)."""
    z = np.asarray(verts, dtype=complex)
    c = z.mean()
    s, ws = _graded_both(**kw)
    t, wt = _graded_both(**kw)
    P_, W_ = [], []
    for a, b in zip(z, np.roll(z, -1)):
        A, B = a - c, b - c
        jac = abs((A.conjugate() * B).imag)
        S, T = np.meshgrid(s, t, indexing="ij")
        P_.append((c + S * ((1 - T) * A + T * B)).ravel())
        W_.append((jac * S * ws[:, None] * wt[None, :]).ravel())
    return np.concatenate(P_), np.concatenate(W_)


def lshape_cubature(shift=0j, layers=14, ratio=0.25, order=20):
    """`tests/test_rellich.py`'s oracle: three unit squares graded toward the re-entrant corner
    (1, 1) of `polyomino([(0, 0), (1, 0), (0, 1)])`."""
    def graded(toward_end):
        edges = np.concatenate([[0.0], ratio ** np.arange(layers, 0, -1), [1.0]])
        xs, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            x, w = gauss_legendre(order)
            xs.append(a + (b - a) * x)
            ws.append((b - a) * w)
        x, w = np.concatenate(xs), np.concatenate(ws)
        return (x, w) if toward_end == 0 else (1.0 - x[::-1], w[::-1])
    pts, wts = [], []
    for (i, j), (ex, ey) in {(0, 0): (1, 1), (1, 0): (0, 1), (0, 1): (1, 0)}.items():
        (xs, wx), (ys, wy) = graded(ex), graded(ey)
        pts.append(((i + xs[:, None]) + 1j * (j + ys[None, :])).ravel() + shift)
        wts.append((wx[:, None] * wy[None, :]).ravel())
    return np.concatenate(pts), np.concatenate(wts)


def interior_gram(solver, ef, pts, wts, chunk=4000):
    k = np.sqrt(ef.lam)
    G = 0.0
    B = solver._pencil.basis
    for a in range(0, len(pts), chunk):
        U = B.values(k, pts[a:a + chunk]) @ ef.coef
        G = G + (U * wts[a:a + chunk, None]).T @ U
    return G


def check_cubature(pts, wts, area):
    return abs(wts.sum() - area) / area


# --- A: interior Gram of the returned eigenfunctions -------------------------------------------

def cases_A():
    out = [("rect 1x0.7", np.array(SHAPES["rect 1x0.7"]), "fan"),
           ("rect 1x0.7 +(3+2i)", np.array(SHAPES["rect 1x0.7"]) + 3 + 2j, "fan"),
           ("square (5pi^2 double)", np.array([0, 1, 1 + 1j, 1j]), "fan"),
           ("L-shape", None, "L"), ("L-shape +(3+2i)", None, "L+"),
           ("triangle", np.array(SHAPES["triangle"]), "fan"),
           ("flat triangle", np.array(SHAPES["flat triangle"]), "fan"),
           ("hexagon (irregular)", np.array(SHAPES["hexagon (irregular)"]), "fan"),
           ("hexagon +(3+2i)", np.array(SHAPES["hexagon (irregular)"]) + 3 + 2j, "fan"),
           ("octagon (irregular)", np.array(SHAPES["octagon (irregular)"]), "fan")]
    return out


def run_A(precs=(1e-10, 1e-12), n=8):
    rows = []
    for name, z, kind in cases_A():
        if kind == "fan":
            dom = Polygon(z, bc="dir")
            pts, wts = fan_cubature(z)
        else:
            shift = 0j if kind == "L" else 3 + 2j
            dom = polyomino([(0, 0), (1, 0), (0, 1)])
            dom = dom.translated(shift) if shift else dom
            pts, wts = lshape_cubature(shift)
        cub = check_cubature(pts, wts, dom.area)
        for prec in precs:
            try:
                rec = Eigenproblem(dom).solve(n, prec=prec)
            except Exception as e:                       # noqa: BLE001 -- a study, report it
                print(f"{name:24s} prec {prec:.0e}: solve failed: {e}", flush=True)
                continue
            if rec.solver is None:
                print(f"{name:24s} prec {prec:.0e}: no solver ({rec.spectrum.flags[:1]})")
                continue
            sp = rec.spectrum
            for i, L in enumerate(sp.levels[:n]):
                ef = rec.solver.eigenfunctions(L, prec=prec)
                G = interior_gram(rec.solver, ef, pts, wts)
                true = float(np.abs(G - np.eye(L.mult)).max())
                # the detector over more origins than the shipped one uses
                cd = C.cauchy_data(rec.solver._pencil.basis, ef.coef, ef.lam, ef.rule.quad)
                probes = certify.x0_probes(dom, ef.rule.x0, n=3)
                spread = max(float(np.abs(I.gram(cd, x0) - np.eye(L.mult)).max())
                             for x0 in probes)
                row = dict(shape=name, prec=prec, level=i, lam=float(L.lam), mult=int(L.mult),
                           rel_bound=float(L.bound / L.lam), true_err=true,
                           gram_residual=float(ef.gram_residual), probe_max=spread,
                           orthonormal=bool(ef.orthonormal),
                           rule_sizing=float(ef.rule.sizing_prec), cubature_area_err=cub,
                           rung=int(rec.plan.rung))
                rows.append(row)
                print(f"{name:24s} prec {prec:.0e} lvl {i} x{L.mult} lam {L.lam:10.4f} rung "
                      f"{rec.plan.rung} | bound {row['rel_bound']:.0e} | TRUE |G-I| "
                      f"{true:.1e} | gram_residual {ef.gram_residual:.1e} | 4-origin max "
                      f"{spread:.1e} | rule model {ef.rule.sizing_prec:.0e} | cub {cub:.0e}",
                      flush=True)
    return rows


# --- B: the rule alone, on closed-form sector norms, at the origin and translated ---------------

def run_B(alphas=(1.25, 1.5, 1.6, 1.75, 1.9), precs=(1e-10, 1e-13), shifts=(0j, 3 + 2j)):
    rows = []
    for alpha in alphas:
        omega = alpha * np.pi
        lam = ref.sector_eigs(1, omega=omega)[0]
        u, norm2 = ref.sector_eigfun(1, 1, omega=omega)
        g = ref.sector_eigfun_grad(1, 1, omega=omega)
        for shift in shifts:
            dom = disk_sector(1.0, omega)
            dom = dom.translated(shift) if shift else dom
            for prec in precs:
                iq = P.integral_quad(dom, lam, prec=prec)
                cd = C.CauchyData.from_functions(iq.quad, lam, lambda z: u(z - shift),
                                                 lambda z: g(z - shift))
                errs = [abs(I.gram(cd, x0)[0, 0] - norm2) / norm2
                        for x0 in (iq.x0, 0.2 + 0.1j + shift, -0.3 + 0.4j + shift)]
                row = dict(alpha=alpha, shift=str(shift), prec=prec, err_at_x0=errs[0],
                           err_max=max(errs), sizing=float(iq.sizing_prec),
                           min_dist=float(np.abs(iq.quad.pts - shift).min()))
                rows.append(row)
                print(f"sector {alpha:.2f}pi shift {shift!s:7s} prec {prec:.0e} | norm err at "
                      f"x0 {errs[0]:.1e}, max over 3 origins {max(errs):.1e} | rule model "
                      f"{iq.sizing_prec:.0e} | closest node to apex {row['min_dist']:.0e}",
                      flush=True)
    return rows


# --- C: the flat triangle's 0.16 -----------------------------------------------------------------

def run_C(prec=1e-12, level=8):
    z = np.asarray(SHAPES["flat triangle"], dtype=complex)
    dom = Polygon(z, bc="dir")
    rec = Eigenproblem(dom).solve(10, prec=prec)
    sp = rec.spectrum
    L = sp.levels[level]
    s = rec.solver
    p = s._pencil
    fc = p.family_coef(float(L.lam), p.coef(float(L.lam), int(L.mult),
                                             rank=getattr(L, "rank", None)))
    iq = s._integral_quad(float(L.lam), prec)
    cd = C.cauchy_data(p.basis, fc, float(L.lam), iq.quad)
    out = dict(lam=float(L.lam), rel_bound=float(L.bound / L.lam), flags=list(sp.flags),
               rung=int(rec.plan.rung), coef_norm=float(np.linalg.norm(fc)))
    pts, wts = fan_cubature(z)
    k = np.sqrt(float(L.lam))
    Uint = p.basis.values(k, pts) @ fc
    out["interior_norm2"] = float(np.sum(wts * Uint[:, 0] ** 2))
    for name, x0 in (("x0", iq.x0), ("probe", certify.x0_probes(dom, iq.x0, n=1)[1])):
        r_n, r_t = I._radial(cd, x0)
        lam = float(L.lam)
        terms = dict(NN=I.weighted_integral(cd, "NN", r_n)[0, 0] / (2 * lam),
                     TT=-I.weighted_integral(cd, "TT", r_n)[0, 0] / (2 * lam),
                     cr=I.weighted_integral(cd, "cr", r_t)[0, 0] / (2 * lam),
                     uv=0.5 * I.weighted_integral(cd, "uv", r_n)[0, 0])
        out[f"terms_{name}"] = {k_: float(v) for k_, v in terms.items()}
        out[f"gram_{name}"] = float(sum(terms.values()))
    out["boundary_u_max_over_interior_rms"] = float(
        np.abs(cd.u).max() / np.sqrt(out["interior_norm2"] / dom.area))
    out["cancellation"] = float(cd.cancellation(np.sqrt([out["interior_norm2"]]), dom.area)[0])
    out["rule_sizing"] = float(iq.sizing_prec)
    out["rule_shortfalls"] = [list(map(str, s_)) for s_ in iq.shortfalls]
    for key, v in out.items():
        print(f"  {key}: {v}", flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    res = {}
    for w in which:
        print(f"=== {w} ===", flush=True)
        res[w] = globals()[f"run_{w}"]()
    with open(f"notes/gram_accuracy_{''.join(which)}.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
