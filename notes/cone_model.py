"""Regenerate notes/cone_model.png: sigma_1 near an eigenvalue against the local model.

    python notes/cone_model.py

Working script, not tracked. The figure it makes is the clearest single statement of what
`eps` and `c` are and why they are separate parameters -- see the section on it in
notes/handoff_minimization.md.
"""
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chladni import reference as ref  # noqa: E402
from chladni.geometry import L_shape  # noqa: E402
from chladni.helmholtz import FourierBessel  # noqa: E402
from chladni.mps import MPSPencil  # noqa: E402

warnings.simplefilter("ignore")

NS = (20, 40, 60)
dom = L_shape()
corner = next(c for c in dom.corners if c.is_reentrant)
k0 = np.sqrt(ref.L_shape_eigs(1)[0])

d = np.logspace(-14, -1, 220)
fits, curves, p40 = {}, {}, None
for n in NS:
    b = FourierBessel.at_corner(corner, n=n, domain=dom)
    p = MPSPencil.from_basis(b, dom.bdry_quad(1200),
                             dom.interior_pts(400, rng=np.random.default_rng(0)))
    eps = p.tension(k0**2)
    c = float(np.median([p.tension((k0 + x) ** 2) / x for x in (0.02, 0.03, 0.04)]))
    fits[n], curves[n] = (eps, c), np.array([p.tension((k0 + x) ** 2) for x in d])
    print(f"n={n}: eps={eps:.3e}  c={c:.4f}  eps/c={eps/c:.3e}")
    if n == 40:
        p40 = p

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

a = ax[0]
for i, n in enumerate(NS):
    eps, c = fits[n]
    a.loglog(d, curves[n], "o", ms=2.6, color=f"C{i}", alpha=.75,
             label=f"measured, {n} FB terms")
    a.loglog(d, np.sqrt(eps**2 + (c * d) ** 2), "-", color=f"C{i}", lw=1.3, alpha=.9)
    a.axhline(eps, color=f"C{i}", ls=":", lw=.9)
    a.plot([eps / c], [np.sqrt(2) * eps], "*", ms=13, color=f"C{i}",
           markeredgecolor="k", markeredgewidth=.5, zorder=5)
a.loglog(d, 1.6 * d, "k--", lw=.9, alpha=.5)
a.set_xlabel(r"$\delta = k - k^*$")
a.set_ylabel(r"$\sigma_1$")
a.set_title(r"solid = model $\sqrt{\epsilon^2+c^2\delta^2}$;  star = knee at $\delta=\epsilon/c$",
            fontsize=10)
a.legend(loc="lower right", fontsize=8.5)
a.grid(alpha=.25, which="both")
a.set_ylim(1e-13, 1)

b_ = ax[1]
eps, c = fits[40]
dd = np.linspace(-8 * eps / c, 8 * eps / c, 241)
meas = np.array([p40.tension((k0 + x) ** 2) for x in dd])
b_.plot(dd / (eps / c), meas / eps, "o", ms=3, color="C1", label="measured, 40 FB terms")
b_.plot(dd / (eps / c), np.sqrt(eps**2 + (c * dd) ** 2) / eps, "-", color="C1", lw=1.4,
        label="model")
b_.plot(dd / (eps / c), np.abs(c * dd) / eps, "k--", lw=1, alpha=.6,
        label=r"cone $c|\delta|$ alone")
b_.axhline(1.0, color="0.6", ls=":", lw=.9)
b_.set_xlabel(r"$\delta \,/\, (\epsilon/c)$")
b_.set_ylabel(r"$\sigma_1 \,/\, \epsilon$")
b_.set_title("the rounded bottom: the cone is wrong within about one knee-width", fontsize=10)
b_.legend(fontsize=8.5)
b_.grid(alpha=.25)
b_.set_ylim(0, 8)

fig.tight_layout()
fig.savefig("notes/cone_model.png", dpi=145)
print("wrote notes/cone_model.png")
