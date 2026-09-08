"""Can the interior Gram tell "two eigenfunctions" from "one, found twice"?

The question is Phase D's, in `docs/design/search.md`: two roots reported at nearby lam have
to be classified as ONE well found twice (collapse, then recount) or TWO independently
resolved eigenvalues (merge, then sum). The spec decides that by DISTANCE -- diameter against
delta_res, then against tol_merge*x -- and distance cannot separate the two cases, because
they occur at the same separation.

The angle between the candidate eigenfunctions can. Both are sampled at the SAME interior
collocation points, since the `Sampling` is fixed at construction and only k changes inside
`assemble`, so the comparison is a plain dot product -- no common refinement, no quadrature,
no lifting to L2(Omega).

Two arms at matched separation d, on rect(1+delta, 1) whose modes (2,1) and (1,2) are
degenerate at delta = 0 and split like 6*pi**2*delta:

    PAIR   u(lam_a) vs u(lam_b)         two genuinely distinct eigenfunctions, d apart
    DUP    u(lam_c) vs u(lam_c + d)     ONE isolated well, sampled twice, d apart

Run log for section 15 of `landscape_findings.md`. Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil

PI2 = np.pi**2


def lams(delta):
    """(lam of mode (2,1), lam of mode (1,2), lam of the isolated mode (1,1))."""
    a = 1.0 + delta
    return PI2 * (4 / a**2 + 1), PI2 * (1 / a**2 + 4), PI2 * (1 / a**2 + 1)


def build(delta, nw=40, nb=800, ni=400, seed=0):
    dom = rect(1.0 + delta, 1.0)
    _, lam_b, lam_c = lams(delta)
    b = PlaneWave.uniform(nw, radius=dom.diameter)
    return MPSPencil.from_basis(b, dom.bdry_quad(nb),
                                dom.interior_pts(ni, rng=np.random.default_rng(seed)),
                                lam_ref=(lam_c * 0.9, lam_b * 1.1))


def img(p, lam):
    """The candidate eigenfunction at the (fixed) interior points, normalized.

    `pencil` is the EQUILIBRATED pair and `coef` lives in that same scaled column space, so
    `I @ c` is the function values directly -- no unscaling, which matters because the
    unscaled coefficient vector does not exist in float64 at a sharp corner.
    """
    _, I = p.pencil(lam)
    u = I @ p.coef(lam, 1)[:, 0]
    return u / np.linalg.norm(u)


def cos_between(p, x, y):
    return abs(float(img(p, x) @ img(p, y)))


def sweep():
    print("A. the two arms at matched separation (40 plane waves, 800 bdry, 400 int)\n")
    print(f"{'delta':>9} {'d(lam)':>10} {'eps_a':>9} {'eps_b':>9} "
          f"{'PAIR cos':>10} {'DUP cos':>10}")
    for delta in [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        p = build(delta)
        lam_a, lam_b, lam_c = lams(delta)
        d = lam_b - lam_a
        print(f"{delta:9.1e} {d:10.3e} {p.tension(lam_a):9.2e} {p.tension(lam_b):9.2e} "
              f"{cos_between(p, lam_a, lam_b):10.3e} "
              f"{cos_between(p, lam_c, lam_c + d):10.3e}")

    p0 = build(0.0)
    lam_a, _, _ = lams(0.0)
    _, I = p0.pencil(lam_a)
    U = I @ p0.coef(lam_a, 2)
    U = U / np.linalg.norm(U, axis=0)
    print(f"\n   within-lam at the exact degeneracy, coef(lam, 2): "
          f"|cos| = {abs(U[:, 0] @ U[:, 1]):.3e}")


def floor_scaling():
    """PAIR's floor is the Monte-Carlo overlap of two L2-orthogonal modes at n_int points."""
    print("\nB. the PAIR floor against interior point count, delta = 1e-2\n")
    print(f"{'n_int':>7} {'seed':>5} {'PAIR cos':>11} {'x sqrt(n_int)':>14}")
    lam_a, lam_b, _ = lams(1e-2)
    for ni in [100, 200, 400, 800, 1600, 3200]:
        for seed in (0, 1, 2):
            c = cos_between(build(1e-2, ni=ni, seed=seed), lam_a, lam_b)
            print(f"{ni:7d} {seed:5d} {c:11.3e} {c * np.sqrt(ni):14.3f}")


def dup_decay():
    """DUP's departure from 1 inside a single well: the minimizer rotates linearly in d."""
    print("\nC. DUP against the offset, one isolated well "
          f"(mean gap 4*pi/area = {4 * np.pi / 1.01:.2f})\n")
    p = build(1e-2)
    _, _, lam_c = lams(1e-2)
    print(f"{'offset d':>11} {'DUP cos':>11} {'1 - cos':>11} {'/d**2':>11}")
    for d in [8.0, 4.0, 2.0, 1.0, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4]:
        c = cos_between(p, lam_c, lam_c + d)
        print(f"{d:11.2e} {c:11.6f} {1 - c:11.3e} {(1 - c) / d**2:11.3e}")


if __name__ == "__main__":
    sweep()
    floor_scaling()
    dup_decay()
