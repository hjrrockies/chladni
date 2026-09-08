"""Does the swing cell decomposition separate a PAIR that sigma_1 shows as ONE dip?

Section 22 measured A19b's cells against gaps between DETECTED dips, so it has never been
shown to split a pair sigma_1 cannot see. That is the whole failure mode sections 9.4 and 19
call real, and it decides whether the composed method is a capability win or only a cost win.

The hypothesis is sharper than "does it still work". The swing peaks where the MINIMISING
EIGENVECTOR CHANGES IDENTITY. Between two eigenvalues at lam_a < lam_b the minimiser is
eigenfunction A below the crossing and B above it, and section 15 measured those two staying
nearly orthogonal (|cos| ~ 1/sqrt(n_int)) down to d = 5.9e-04 and below -- far past where
sigma_1 stops showing two dips. So:

    a swing peak INSIDE a single sigma_1 dip  <=>  that dip holds two eigenvalues

If that holds it is a pair DETECTOR at coarse stride, and section 3 of `search.md` records
that no existing detector fires before sigma_1 itself does -- the section 3.4 offset fires at
exactly the same coarseness. This would be the first that does.

rect(1+delta, 1), modes (2,1) and (1,2), splitting like 6*pi**2*delta. Exploratory.
"""
import sys
import warnings

sys.path.insert(0, "notes")
sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil

PI2 = np.pi**2


def lams(delta):
    a = 1.0 + delta
    return PI2 * (4 / a**2 + 1), PI2 * (1 / a**2 + 4)


def build(delta, nw=40, nb=800, ni=400, seed=0):
    dom = rect(1.0 + delta, 1.0)
    la, lb = lams(delta)
    b = PlaneWave.uniform(nw, radius=dom.diameter)
    return MPSPencil.from_basis(
        b, dom.bdry_quad(nb),
        dom.interior_pts(ni, rng=np.random.default_rng(seed)),
        lam_ref=(la * 0.85, lb * 1.15))


def probe(p, xs):
    """sigma_1 and the normalized interior image of the minimiser, at each sample."""
    s1, U = [], []
    for x in xs:
        s1.append(p.tension(x))
        _, I = p.pencil(x)
        u = I @ p.coef(x, 1)[:, 0]
        U.append(u / np.linalg.norm(u))
    return np.array(s1), np.array(U)


def peaks(v, thresh):
    j = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1
    return j[v[j] > thresh * np.median(v)]


def run(strides=(0.30, 0.15, 0.08), thresh=5.0):
    """`strides` are h as a fraction of the MEAN gap (4*pi/area ~ 6.2 here)."""
    print("rect(1+delta, 1): does a swing peak appear inside a single sigma_1 dip?\n")
    print(f"{'delta':>9} {'d(lam)':>10} {'h':>7} {'h/d':>9} "
          f"{'sig1 dips':>10} {'swing pks':>10} {'pk in pair':>11} {'verdict':>22}")
    gap = 4 * np.pi / 1.0
    for delta in (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 0.0):
        p = build(delta)
        la, lb = lams(delta)
        d = lb - la
        mid = 0.5 * (la + lb)
        W = 0.45 * gap                        # a window holding the pair and no neighbour
        for frac in strides:
            h = frac * gap
            n = int(np.ceil(2 * W / h)) + 1
            xs = np.linspace(mid - W, mid + W, n)
            s1, U = probe(p, xs)
            ang = 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))
            midpts = 0.5 * (xs[:-1] + xs[1:])
            dips = np.nonzero((s1[1:-1] < s1[:-2]) & (s1[1:-1] <= s1[2:]))[0] + 1
            dips = dips[s1[dips] < 0.5 * np.median(s1)]
            pk = midpts[peaks(ang, thresh)]
            # a peak "in the pair" lies between the two true eigenvalues, or within one
            # stride of the midpoint when d is far below h
            inpair = int(np.sum((pk > la - h) & (pk < lb + h)))
            if d == 0:
                verdict = "degenerate control"
            elif len(dips) >= 2:
                verdict = "sigma_1 already splits"
            elif inpair >= 1:
                verdict = "SWING SEES IT, dip does not"
            else:
                verdict = "neither"
            print(f"{delta:9.1e} {d:10.3e} {h:7.3f} {d/h if h else np.nan:9.2e} "
                  f"{len(dips):10d} {len(pk):10d} {inpair:11d} {verdict:>22}")
        print()


if __name__ == "__main__":
    run()
