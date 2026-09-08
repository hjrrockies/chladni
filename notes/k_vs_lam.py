"""Regenerate notes/k_vs_lam.png: is sigma_1^2 quadratic in k, or in lam?

    python notes/k_vs_lam.py

Working script, not tracked. Answers a question the Lipschitz proposal settles for the SCAN
(section 2.2, work in k) but never asks for the REFINEMENT, where Brent fits a parabola.
"""
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chladni import reference as ref  # noqa: E402
from chladni.geometry import L_shape, rect  # noqa: E402
from chladni.helmholtz import FourierBessel, PlaneWave  # noqa: E402
from chladni.mps import MPSPencil  # noqa: E402

warnings.simplefilter("ignore")

CASES = [
    ("L-shape",  L_shape(),      ref.L_shape_eigs(1)[0],       40),
    ("rect 2x1", rect(2.0, 1.0), ref.rect_eigs(1, 2.0, 1.0)[0], 0),
    ("rect 5x1", rect(5.0, 1.0), ref.rect_eigs(1, 5.0, 1.0)[0], 0),
]


def build(dom, lam0, fb):
    k0 = np.sqrt(lam0)
    if fb:
        corner = next(c for c in dom.corners if c.is_reentrant)
        b = FourierBessel.at_corner(corner, n=fb, domain=dom)
    else:
        R = dom.diameter
        b = PlaneWave.uniform(int(PlaneWave.n_useful(1.4 * k0 * R)), radius=R)
    return k0, MPSPencil.from_basis(b, dom.bdry_quad(1200),
                                    dom.interior_pts(400, rng=np.random.default_rng(0)))


fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# --- residual of the quadratic fit, in each variable, for one case -------------------
a = ax[0]
k0, p = build(*CASES[0][1:])
H = 0.10
ks = k0 + np.linspace(-H, H, 121)
f = np.array([p.tension(x**2) ** 2 for x in ks])
for x, lbl, col in ((ks, r"fit against $k$", "C0"),
                    (ks**2, r"fit against $\lambda$", "C1")):
    r = f - np.polyval(np.polyfit(x, f, 2), x)
    a.plot((ks - k0) / H, r / np.ptp(f), "o-", ms=3, lw=1, color=col, label=lbl)
a.axhline(0, color="0.6", lw=.8)
a.set_xlabel(r"$(k-k^*)\,/\,h$   (identical physical points either way)")
a.set_ylabel(r"residual of a quadratic fit to $\sigma_1^2$, relative to range")
a.set_title(r"L-shape $\lambda_1$: $\sigma_1^2$ is the better parabola in $\lambda$",
            fontsize=11)
a.legend(fontsize=9)
a.grid(alpha=.25)

# --- and it is not one domain -------------------------------------------------------
b_ = ax[1]
hs = np.logspace(-2.2, -0.5, 12)
for i, (name, dom, lam0, fb) in enumerate(CASES):
    k0, p = build(dom, lam0, fb)
    rk, rl = [], []
    for h in hs:
        kk = k0 + np.linspace(-h * k0, h * k0, 41)
        ff = np.array([p.tension(x**2) ** 2 for x in kk])
        rk.append(np.std(ff - np.polyval(np.polyfit(kk, ff, 2), kk)) / np.ptp(ff))
        rl.append(np.std(ff - np.polyval(np.polyfit(kk**2, ff, 2), kk**2)) / np.ptp(ff))
    b_.loglog(hs, np.array(rk) / np.array(rl), "o-", ms=4, color=f"C{i}", label=name)
b_.axhline(1.0, color="0.5", ls="--", lw=1)
b_.text(1.1e-2, 1.15, "below this line, $k$ would be better", fontsize=8.5, color="0.4")
b_.set_xlabel(r"bracket half-width, as a fraction of $k^*$")
b_.set_ylabel(r"residual against $k$  /  residual against $\lambda$")
b_.set_title(r"$\lambda$ wins on every domain and every window", fontsize=11)
b_.legend(fontsize=9)
b_.grid(alpha=.25, which="both")
b_.set_ylim(0.7, 30)

fig.tight_layout()
fig.savefig("notes/k_vs_lam.png", dpi=145)
print("wrote notes/k_vs_lam.png")
