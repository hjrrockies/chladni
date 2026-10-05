"""Is the eigenvalue better than the tension says? And how tight is a Moler-Payne bound?

`notes/moler_payne_vs_tension.md` is the record this reproduces. Two measurements on
`rect(1, 1/phi)` (`rect_suite.family_A()[0]`, Dirichlet, 126 plane-wave columns), whose
eigenvalues are known exactly:

    posterior   at a simple eigenvalue, the penalty tension sigma_tau split into its two terms,
                and the UNPENALIZED ratio t(x) = ||A_B x|| / ||A_I x|| on the same minimizing
                vector, against the offset from the eigenvalue.
    bound       five eigenvalues, three pencils, each refined with `refine`: the true relative
                error, the refiner's bar, sigma, t(x), and the Moler-Payne bound

                    |lam_k - lam| / lam_k <= sqrt(A) max_{dOmega} |u| / ||u||_{L2(Omega)}

                with max|u| over 8000 boundary points and ||u|| from an 80 x 60 tensor
                Gauss-Legendre rule -- exact to rounding for these u on a rectangle, which is
                what makes this the reference the general layer must reproduce.

Nothing in `src/` is modified. Run from the repository root:

    .venv/bin/python notes/moler_payne_vs_tension.py posterior
    .venv/bin/python notes/moler_payne_vs_tension.py bound
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")          # explicit penalties outside the reach warn, on purpose

import numpy as np

import rect_suite as rs
from chladni.eigensolver import lam_to_k
from chladni.mps import MPSPencil
from chladni.mps.objective import Objective
from chladni.mps.refine import refine

CASE = rs.family_A()[0]
P = CASE.pencil
N = P.n_cols
W, H = CASE.dims
FAM = P.bdry.family


def penalty(tau, prec):
    """An explicit `tikhonov_qr` pencil with absolute `tau` on the equilibrated stack."""
    return MPSPencil(P.bdry, P.interior, reg="tikhonov_qr", prec=prec, rtol=tau / np.sqrt(N))


def parts(q, lam):
    """`(sigma_tau, t(x), penalty term, x)` at `lam`. The penalty is in the NUMERATOR:
    sigma_tau^2 = (||B x||^2 + tau^2 ||x||^2) / ||I x||^2, so t(x) <= sigma_tau always."""
    B, I = q.pencil(lam)
    x = q.coef(lam)[:, 0]
    tau = q.rtol * np.linalg.norm(np.vstack([B, I]))
    nb, ni, nx = np.linalg.norm(B @ x), np.linalg.norm(I @ x), np.linalg.norm(x)
    return q.tension(lam), nb / ni, tau * nx / ni, x


def u_at(q, lam, x, pts):
    """The function `x` names, at arbitrary points. A column of the equilibrated pencil is the
    family's column divided by the pencil's equilibration scale, so divide `x` by it."""
    scale = np.exp(q._equilibrated(lam)[2])
    return FAM.at(pts).values(lam_to_k(lam)) @ (x / scale)


def posterior():
    lam0 = float(CASE.truth[0])
    print(f"lam* = {lam0!r}, n = {N}, threshold at SUITE_RTOL: {P.tension(lam0):.2e}")
    for prec in (1e-8, 1e-10, 1e-12, 1e-14):
        q = penalty(0.02 * prec, prec)
        s, t, pen, _ = parts(q, lam0)
        print(f"\nprec {prec:.0e}  tau {0.02 * prec:.0e}  at lam*: sigma_tau {s:.2e}  "
              f"penalty term {pen:.2e}  t(x) {t:.2e}")
        print("   offset/lam   sigma_tau    t(x)       threshold")
        for d in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
            lam = lam0 * (1 + d)
            s, t, _, _ = parts(q, lam)
            print(f"   {d:+.0e}      {s:.3e}   {t:.3e}   {P.tension(lam):.3e}")
        zs = lam0 * (1 + np.linspace(-3e-13, 3e-13, 61))
        ts = np.array([parts(q, z)[1] for z in zs])
        print(f"   over +-3e-13 rel: min t(x) {ts.min():.2e} at "
              f"{(zs[ts.argmin()] - lam0) / lam0:+.1e} rel")


def bound():
    gx, gw = np.polynomial.legendre.leggauss(80)
    hy, hw = np.polynomial.legendre.leggauss(60)
    X, Y = 0.5 * W * (gx + 1), 0.5 * H * (hy + 1)
    gpts = (X[:, None] + 1j * Y[None, :]).ravel()
    gwts = (0.25 * W * H * gw[:, None] * hw[None, :]).ravel()
    bpts = CASE.domain.bdry_quad(8000).pts
    pencils = {
        "pen 1e-8 ": (MPSPencil(P.bdry, P.interior, prec=1e-8), 1e-8),     # the shipped default
        "pen 2e-14": (penalty(2e-14, 1e-12), 1e-12),
        "threshold": (P, 1e-12),                                           # SUITE_RTOL
    }
    print("lam*      pencil      err/lam   bar/lam   sigma     t(x)      MP bound  MP/err")
    for lam0 in (float(t) for t in CASE.truth[[0, 1, 2, 4, 7]]):
        for name, (q, prec) in pencils.items():
            lo, hi = lam0 * (1 - 1e-6), lam0 * (1 + 1.3e-6)
            obj = Objective(lambda z, q=q: q.tensions(float(z)), lo, hi, prec=prec,
                            variable="lam")
            r = refine(obj, lo, hi, prec=prec)
            _, t, _, x = parts(q, r.lam)
            ub, ui = u_at(q, r.lam, x, bpts), u_at(q, r.lam, x, gpts)
            mp = np.sqrt(W * H) * np.abs(ub).max() / np.sqrt(np.sum(gwts * ui ** 2))
            err = abs(r.lam - lam0) / lam0
            print(f"{lam0:8.3f}  {name}  {err:.1e}   {r.delta_res / r.lam:.1e}   {r.sigma:.1e}   "
                  f"{t:.1e}   {mp:.1e}   {mp / max(err, 1e-17):.1e}"
                  f"{'' if r.accepted else '  REFUSED'}")


if __name__ == "__main__":
    {"posterior": posterior, "bound": bound}[sys.argv[1] if len(sys.argv) > 1 else "bound"]()
