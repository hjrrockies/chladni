"""Does `chladni`'s factored Bessel kernel beat plain `scipy.special.jv` END TO END?

Asked 2026-10-06. The kernel (`helmholtz/_bessel.py`: series / scipy / Olver-Debye, values
returned as `S * exp(sigma)`) is validated as a kernel against mpmath (`tests/test_bessel_kernel.py`,
`tests/test_bessel_branches.py`). This script asks whether that buys any eigenvalue accuracy over a
plain implementation: the same collocation points, the same equilibration and `svd` threshold
regularization, the same golden-section search -- columns from `Sampling.values` versus raw
`jv(mu, k r) * sin(mu theta)`, with all-zero (underflowed) columns dropped, as any plain
implementation would.

Run from the repository root (~1 minute):

    .venv/bin/python notes/bessel_kernel_vs_scipy.py

Measured 2026-10-06 (macOS/Accelerate, `42c2a77`), relative errors:

    L-shape, 80 reentrant terms, vs the 14-digit table: identical, 1.6e-14 on every level -- the
      resolution of the search tolerance and the table, not of either kernel; no column underflows
      (orders <= 53).
    douse's spike triangle (failure catalogue T1.3; 11.2/16.3/152.6 degrees, exact vertices), vs
      its 12-digit consensus spectrum:
      150 terms by angle (orders <= 155):       kernel 1.0e-9 .. 4.6e-9, scipy 1.6e-9 .. 6.7e-9,
        0 columns lost.
      40 terms at the spike (orders <= 644):    kernel 3.2e-9 .. 1.9e-8, scipy 1.9e-9 .. 1.0e-8,
        16-20 columns lost to underflow.
    Neither consistently better; differences within the basis's own error.

Not established: the triangle bases floor at tension ~2e-7, so an advantage at 1e-12 could not
show -- a fair test needs a high-precision basis on a sharp shape (e.g. with fundamental
solutions). Derivatives were not tested: Neumann/Robin traces and the integrals layer's Cauchy data
use J', which the kernel takes from the same series and a plain implementation from `jvp`.
"""
import warnings

import numpy as np
from scipy.special import jv

from chladni.geometry import L_shape, Polygon
from chladni.helmholtz import FourierBessel, PrecisionWarning
from chladni.mps import equilibrate, lam_to_k, tensions
from chladni.reference import L_shape_eigs

warnings.simplefilter("ignore", PrecisionWarning)
RTOL = np.sqrt(1e-12 * np.finfo(float).eps)        # MPSPencil's threshold cutoff at prec=1e-12


def samplings(dom, alloc, bmult):
    fams = [FourierBessel.at_corner(c, n=n, domain=dom) for c, n in alloc]
    ncol = sum(len(f) for f in fams)
    q = dom.bdry_quad(bmult * ncol)
    ip = dom.interior_pts(ncol, rng=np.random.default_rng(0))
    return fams, [(f.at(q), f.at(ip)) for f in fams], len(q)


def tension(samps, lam, plain):
    k = lam_to_k(lam)
    blocks = [[jv(s.family.orders[None, :], k * s.r[:, None]) * s._ang if plain else s.values(k)
               for s in pair] for pair in samps]
    A_B = np.hstack([b[0] for b in blocks])
    A_I = np.hstack([b[1] for b in blocks])
    zero = np.all(A_B == 0, axis=0) & np.all(A_I == 0, axis=0)
    B, I, _ = equilibrate(A_B[:, ~zero], A_I[:, ~zero])
    return tensions(B, I, reg="svd", rtol=RTOL)[0], int(zero.sum())


def golden(f, a, b, tol):
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def compare(label, dom, alloc, ref, bmult, window):
    fams, samps, nb = samplings(dom, alloc, bmult)
    print(f"\n{label}: {sum(len(f) for f in fams)} columns, max order "
          f"{max(f.orders.max() for f in fams):.0f}, {nb} boundary points")
    print("   reference         kernel err   scipy err   cols lost")
    for lam in ref:
        out = []
        for plain in (False, True):
            x = golden(lambda t: tension(samps, t, plain)[0], lam * (1 - window),
                       lam * (1 + window), 1e-13 * lam)
            out.append((abs(x - lam) / lam, tension(samps, x, plain)[1]))
        print(f"   {lam:16.10f}  {out[0][0]:9.1e}   {out[1][0]:9.1e}   {out[1][1]:5d}")


dom = L_shape()
reentrant = next(c for c in dom.corners if c.is_reentrant)
compare("L-shape, 80 reentrant terms", dom, [(reentrant, 80)],
        np.asarray(L_shape_eigs(4), dtype=float), bmult=3, window=5e-3)

# douse T1.3: VertexPolygon(3, perim=4.0) at p = start_shape(3, 4), exact float vertices.
tri = Polygon([1.972069685870855 + 0j, 1.1756103887576692 + 0.23218091730644705j, 0j])
REF = np.array([248.653419474, 344.590171803, 430.380005708, 518.615801753, 611.337416106])
angles = np.array([c.omega for c in tri.corners])
spike = min(tri.corners, key=lambda c: c.omega)
compare("spike triangle, 150 terms by angle", tri,
        [(c, max(int(round(150 * c.omega / angles.sum())), 3)) for c in tri.corners],
        REF, bmult=4, window=2e-3)
compare("spike triangle, 40 terms at the 11-degree corner", tri,
        [(c, 40 if c is spike else int(round(150 * c.omega / angles.sum()))) for c in tri.corners],
        REF, bmult=4, window=2e-3)
