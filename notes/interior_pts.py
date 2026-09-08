"""How good is the interior point set as an L2(Omega) inner product?

The interior block's job is to stand in for `||u||_{L2(Omega)}`, and `interior_pts` fills it
by REJECTION SAMPLING from the bounding box. That makes the discrete inner product a
Monte-Carlo estimate, with a random realization -- which is what made the section 16
coalescence residual move 43x with the seed, and what sets the section 15 PAIR floor at
1/sqrt(n_int).

There is an exact measurement available. On rect(1+delta, 1) the modes (2,1) and (1,2) are
EXACTLY L2-orthogonal, so

    |cos| = |<u_a, u_b>| / (||u_a|| ||u_b||)

over the interior points is the quadrature error of the discrete inner product on a known-zero
integral. It is a direct read of the point set's quality, with no reference to any search.

Three fills, all through the same pencil:

    random   rejection sampling from the bbox      MC, O(1/sqrt(n))
    grid     `method="grid"`, already in Domain    deterministic, aliasing risk
    sobol    scipy's Sobol sequence, rejected      QMC, O(log(n)^2/n) before rejection

Run log for section 17 of `landscape_findings.md`. Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
from scipy.stats import qmc

from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil

PI2 = np.pi**2


def lams(delta):
    a = 1.0 + delta
    return PI2 * (4 / a**2 + 1), PI2 * (1 / a**2 + 4), PI2 * (1 / a**2 + 1)


def sobol_pts(dom, n, seed=None, oversample=4.0):
    """Sobol points in the bbox, rejected to the domain. Deterministic when seed is None."""
    lo, hi = dom.bbox
    box = (hi.real - lo.real) * (hi.imag - lo.imag)
    factor = max(oversample * box / dom.area, 2.0)
    m = int(2 ** np.ceil(np.log2(max(n * factor, 16))))
    eng = qmc.Sobol(2, scramble=seed is not None, seed=seed)
    p = eng.random(m)
    z = (lo.real + p[:, 0] * (hi.real - lo.real)
         + 1j * (lo.imag + p[:, 1] * (hi.imag - lo.imag)))
    z = z[dom.contains(z)]
    if len(z) < n:
        raise RuntimeError(f"sobol_pts: got {len(z)} of {n}")
    return z[:n]


def points(dom, method, n, seed=0):
    if method == "random":
        return dom.interior_pts(n, rng=np.random.default_rng(seed))
    if method == "grid":
        return dom.interior_pts(n, method="grid")
    if method == "sobol":
        return sobol_pts(dom, n)
    if method == "sobol_s":
        return sobol_pts(dom, n, seed=seed)
    raise ValueError(method)


def build(delta, method, ni, seed=0, nw=40, nb=800, theta=0.0):
    dom = rect(1.0 + delta, 1.0)
    if theta:
        dom = dom.rotated(theta)
    _, lam_b, lam_c = lams(delta)
    b = PlaneWave.uniform(nw, radius=dom.diameter)
    return MPSPencil.from_basis(b, dom.bdry_quad(nb), points(dom, method, ni, seed),
                                lam_ref=(lam_c * 0.9, lam_b * 1.1))


def img(p, lam):
    _, I = p.pencil(lam)
    u = I @ p.coef(lam, 1)[:, 0]
    return u / np.linalg.norm(u)


def cos_ab(delta, method, ni, seed=0, theta=0.0):
    p = build(delta, method, ni, seed, theta=theta)
    lam_a, lam_b, _ = lams(delta)
    return abs(float(img(p, lam_a) @ img(p, lam_b)))


SEEDS = range(12)


def orthogonality_error():
    """|cos| between two exactly-L2-orthogonal modes: the discrete inner product's error."""
    print("A. quadrature error of the interior inner product, rect(1.01, 1)\n")
    print(f"{'n_int':>7} | {'random, 12 seeds: med':>21} {'min':>10} {'max':>10} "
          f"{'spread':>8} | {'grid':>10} {'sobol':>10}")
    med = {"random": [], "grid": [], "sobol": []}
    ns = [100, 200, 400, 800, 1600, 3200, 6400]
    for ni in ns:
        v = np.array([cos_ab(1e-2, "random", ni, s) for s in SEEDS])
        g, so = cos_ab(1e-2, "grid", ni), cos_ab(1e-2, "sobol", ni)
        med["random"].append(np.median(v))
        med["grid"].append(g)
        med["sobol"].append(so)
        print(f"{ni:7d} | {np.median(v):21.2e} {v.min():10.2e} {v.max():10.2e} "
              f"{v.max()/v.min():8.0f}x | {g:10.2e} {so:10.2e}")

    print("\n   rate from the medians, |cos| ~ n**(-r), n = 400 .. 6400")
    x = np.log(np.array(ns[2:], dtype=float))
    for m in med:
        r = -np.polyfit(x, np.log(np.array(med[m][2:])), 1)[0]
        print(f"      {m:>8}  r = {r:5.2f}")


def rotation_control():
    """grid on an AXIS-ALIGNED rectangle is a tensor rule against separable integrands.

    Rotating the domain destroys that alignment and is the honest test: a fill whose quality
    depends on the domain happening to line up with the bounding box is useless in a black box
    iterating over generated shapes.
    """
    print("\n\nB. the same error under rotation, n_int = 1600\n")
    print(f"{'theta':>8} {'bbox/area':>10} | {'random (med of 12)':>19} "
          f"{'grid':>11} {'sobol':>11}")
    for deg in (0, 5, 15, 30, 45):
        th = np.deg2rad(deg)
        dom = rect(1.01, 1.0).rotated(th) if th else rect(1.01, 1.0)
        lo, hi = dom.bbox
        ratio = (hi.real - lo.real) * (hi.imag - lo.imag) / dom.area
        v = np.array([cos_ab(1e-2, "random", 1600, s, theta=th) for s in SEEDS])
        g = cos_ab(1e-2, "grid", 1600, theta=th)
        so = cos_ab(1e-2, "sobol", 1600, theta=th)
        print(f"{deg:7d}d {ratio:10.3f} | {np.median(v):19.2e} {g:11.2e} {so:11.2e}")


def coalescence():
    """The payoff test: does a better fill stabilize section 16's degeneracy residual?

    rect(1,1), modes (2,1) and (1,2) EXACTLY degenerate. |argmin sigma_1 - argmin sigma_2| is
    a splitting induced purely by the discretization, and section 16 measured it moving 43x
    with the interior seed. `random` is varied over seeds; the deterministic fills have no
    seed, so they are varied over ROTATION, which is the perturbation a black-box solver
    actually meets.
    """
    from scipy.optimize import minimize_scalar
    print("\n\nC. the exact-degeneracy residual |argmin s1 - argmin s2|, rect(1, 1)\n")
    lam_a = PI2 * 5.0

    def resid(method, seed=0, theta=0.0):
        p = build(0.0, method, 400, seed, theta=theta)
        c = 0.5 * (p.tension(lam_a + 0.6) + p.tension(lam_a - 0.6)) / 0.6
        xs = [minimize_scalar(lambda x: p.tensions(x)[j]**2,
                              bounds=(lam_a - 0.5, lam_a + 0.5), method="bounded",
                              options={"xatol": 1e-14}).x for j in (0, 1)]
        return abs(xs[0] - xs[1]) / (p.tension(lam_a) / c)

    v = np.array([resid("random", seed=s) for s in SEEDS])
    print(f"   random, 12 seeds        |D|/(eps/c)  med {np.median(v):7.1f}  "
          f"min {v.min():7.1f}  max {v.max():7.1f}   spread {v.max()/v.min():5.0f}x")
    for m in ("grid", "sobol"):
        w = np.array([resid(m, theta=np.deg2rad(d)) for d in (0, 5, 15, 30, 45)])
        print(f"   {m:>6}, 5 rotations     |D|/(eps/c)  med {np.median(w):7.1f}  "
              f"min {w.min():7.1f}  max {w.max():7.1f}   spread {w.max()/w.min():5.0f}x")
    u = np.array([resid("random", seed=s, theta=np.deg2rad(d))
                  for s, d in zip(SEEDS, [0, 5, 15, 30, 45] * 3)])
    print(f"   random, 5 rotations     |D|/(eps/c)  med {np.median(u):7.1f}  "
          f"min {u.min():7.1f}  max {u.max():7.1f}   spread {u.max()/u.min():5.0f}x")


if __name__ == "__main__":
    orthogonality_error()
    rotation_control()
    coalescence()


# --- the L-shape, which is the case that decides whether the default moves ------------------
#
# The rectangle-mode-pair test does not generalize: the L-shape has no closed form. But
# eigenfunctions at DISTINCT eigenvalues of the same Dirichlet Laplacian are exactly
# L2-orthogonal, whatever the domain, so any two reference eigenvalues give the same
# known-zero integral. That instrument works anywhere `reference` has a table -- and it is
# validated below by reproducing the rectangle numbers it was derived from.

from chladni.helmholtz import FourierBessel  # noqa: E402
import chladni.reference as ref              # noqa: E402
from chladni.geometry import Polygon         # noqa: E402

L_SHAPE = np.array([0, 2, 2 + 1j, 1 + 1j, 1 + 2j, 2j], dtype=complex)


def l_shape():
    return Polygon(L_SHAPE)


def pencil_at(dom, basis, method, ni, seed=0, nb=1400, lam_ref=None):
    return MPSPencil.from_basis(basis, dom.bdry_quad(nb), points(dom, method, ni, seed),
                                lam_ref=lam_ref)


def cross_cos(p, x, y):
    return abs(float(img(p, x) @ img(p, y)))


def cross_mode_check():
    """Validate the instrument: two DISTINCT rectangle eigenvalues must reproduce section 17."""
    print("\n\nD. instrument check -- distinct eigenvalues on rect(1.01, 1), n_int = 1600\n")
    dom = rect(1.01, 1.0)
    b = PlaneWave.uniform(40, radius=dom.diameter)
    e = np.sort(ref.rect_eigs(6, 1.01, 1.0))
    print(f"   reference lam: {np.array2string(e, precision=4)}")
    print(f"{'method':>10} {'|cos|(lam1,lam2)':>18} {'|cos|(lam1,lam3)':>18} "
          f"{'|cos|(lam2,lam4)':>18}")
    for meth, seeds in (("random", range(12)), ("grid", [0]), ("sobol", [0])):
        vals = []
        for (i, j) in ((0, 1), (0, 2), (1, 3)):
            v = [cross_cos(pencil_at(dom, b, meth, 1600, s, nb=800), e[i], e[j])
                 for s in seeds]
            vals.append(np.median(v))
        print(f"{meth:>10} {vals[0]:18.2e} {vals[1]:18.2e} {vals[2]:18.2e}")


def l_shape_fill():
    """The number that decides the default. area/bbox = 3/4, and a reentrant corner."""
    dom = l_shape()
    lo, hi = dom.bbox
    ratio = (hi.real - lo.real) * (hi.imag - lo.imag) / dom.area
    e = ref.L_shape_eigs(6)
    corner = [c for c in dom.corners if c.is_reentrant][0]
    b = FourierBessel.at_corner(corner, n=80, domain=dom)
    print(f"\n\nE. L-shape, bbox/area = {ratio:.3f}, pure Fourier-Bessel n=80 "
          f"(the reference recipe)\n")
    print(f"   reference lam: {np.array2string(e, precision=5)}")
    print(f"{'n_int':>7} | {'random, 12 seeds: med':>21} {'min':>10} {'max':>10} "
          f"{'spread':>8} | {'grid':>10} {'sobol':>10}")
    pairs = ((0, 1), (0, 2), (1, 2), (0, 3), (2, 4))
    for ni in (200, 400, 800, 1600, 3200):
        def med_over(meth, seeds):
            out = []
            for s in seeds:
                p = pencil_at(dom, b, meth, ni, s)
                out.append(np.median([cross_cos(p, e[i], e[j]) for i, j in pairs]))
            return np.array(out)
        v = med_over("random", range(12))
        g = med_over("grid", [0])[0]
        so = med_over("sobol", [0])[0]
        print(f"{ni:7d} | {np.median(v):21.2e} {v.min():10.2e} {v.max():10.2e} "
              f"{v.max()/v.min():8.0f}x | {g:10.2e} {so:10.2e}")
