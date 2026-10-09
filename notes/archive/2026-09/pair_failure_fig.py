"""What breaks for a close pair: the tension merges them, and a coarse swing misses the crossing.

rect(1.0001, 1), modes (2,1) and (1,2) at lam = 49.3441 and 49.3500, so d = 5.92e-03. The
search scores recall 0.750 at per_gap = 6 and 1.000 at per_gap = 24 (section 26). This draws
the reason.
"""
import sys
import warnings

sys.path.insert(0, "notes")
sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil

PI2 = np.pi**2
DELTA = 1e-4
A, B = 10.0, 88.0

dom = rect(1.0 + DELTA, 1.0)
a_ = 1.0 + DELTA
LA, LB = PI2 * (4 / a_**2 + 1), PI2 * (1 / a_**2 + 4)
D = LB - LA
p = MPSPencil.from_basis(PlaneWave.uniform(40, radius=dom.diameter), dom.bdry_quad(800),
                         dom.interior_pts(400, rng=np.random.default_rng(0)),
                         lam_ref=(A, B))


def img(x):
    _, I = p.pencil(x)
    u = I @ p.coef(x, 1)[:, 0]
    return u / np.linalg.norm(u)


def locate(lo, hi, ulo, uhi, tol=1e-9, depth=40):
    """Bisect toward the crossing: it lies in whichever equal-width half carries more angle."""
    path = [(lo, hi)]
    for _ in range(depth):
        if hi - lo <= tol:
            break
        m = 0.5 * (lo + hi)
        um = img(m)
        if 1.0 - abs(float(ulo @ um)) >= 1.0 - abs(float(um @ uhi)):
            hi, uhi = m, um
        else:
            lo, ulo = m, um
        path.append((lo, hi))
    return 0.5 * (lo + hi), path


def swing(xs):
    U = np.array([img(x) for x in xs])
    return 0.5 * (xs[:-1] + xs[1:]), 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))


# the strides the search actually uses: gap is estimated at ~26 from a pilot that has
# already merged the pair, so per_gap = 6 gives h = 4.33
GAP_EST = 26.0
grids = {}
for pg, style in ((6, dict(color="#c0392b", marker="o", ms=5)),
                  (24, dict(color="#1a5276", marker="s", ms=3))):
    n = int(np.ceil((B - A) / (GAP_EST / pg))) + 1
    xs = np.linspace(A, B, n)
    grids[pg] = (xs, *swing(xs), style, (B - A) / (n - 1))

wide = np.linspace(44.0, 55.0, 500)
s_wide = np.array([p.tension(x) for x in wide])
zoom = np.linspace(LA - 6 * D, LB + 6 * D, 400)
s_zoom = np.array([p.tension(x) for x in zoom])

fig, ax = plt.subplots(3, 1, figsize=(10.5, 11.5))
fig.suptitle(f"A pair at $d = {D:.2e}$ in $\\lambda$: what the search sees\n"
             f"rect(1.0001, 1), modes (2,1) and (1,2), mean gap $\\approx$ 26",
             fontsize=13, y=0.975)

# --- 1. the tension, at the stride the search uses --------------------------------------
ax[0].semilogy(wide, s_wide, color="0.35", lw=1.3, label=r"$\sigma_1(\lambda)$, fine")
for pg in (6, 24):
    xs, _, _, st, h = grids[pg]
    m = (xs >= 44) & (xs <= 55)
    ax[0].semilogy(xs[m], [p.tension(x) for x in xs[m]], ls="none",
                   label=f"samples, per_gap={pg} ($h$={h:.2f})", **st)
ax[0].axvline(LA, color="green", lw=1, ls=":")
ax[0].axvline(LB, color="green", lw=1, ls=":")
ax[0].annotate("the pair: TWO eigenvalues,\n$5.9\\times10^{-3}$ apart",
               xy=(LA, 2e-7), xytext=(50.6, 3e-5), fontsize=10,
               arrowprops=dict(arrowstyle="->", color="green"))
ax[0].set_title("1. The tension merges the pair. At either stride $\\sigma_1$ shows ONE dip.",
                fontsize=11, loc="left")
ax[0].set_ylabel(r"$\sigma_1$")
ax[0].legend(fontsize=8, loc="upper right")

# --- 2. the swing, at both strides ------------------------------------------------------
for pg in (6, 24):
    xs, mid, ang, st, h = grids[pg]
    m = (mid >= 44) & (mid <= 55)
    ax[1].semilogy(mid[m], ang[m], lw=1.1, ls="-", **{k: v for k, v in st.items()
                                                      if k != "ms"},
                   label=f"per_gap={pg} ($h$={h:.2f})", ms=4)
    ax[1].axhline(5 * np.median(ang), color=st["color"], lw=0.8, ls="--", alpha=0.6)
ax[1].axvline(LA, color="green", lw=1, ls=":")
ax[1].axvline(LB, color="green", lw=1, ls=":")
ax[1].annotate("crossing at the pair's midpoint:\nvisible at $h$=1.08, invisible at $h$=4.33",
               xy=(0.5 * (LA + LB), 1e-2), xytext=(50.4, 2e-5), fontsize=10,
               arrowprops=dict(arrowstyle="->", color="black"))
ax[1].set_title("2. The swing $1-|\\langle u_i,u_{i+1}\\rangle|$. Dashed = the "
                "$5\\times$median detection threshold.", fontsize=11, loc="left")
ax[1].set_ylabel("swing angle")
ax[1].set_xlabel(r"$\lambda$")
ax[1].legend(fontsize=8, loc="upper right")

# --- 3. the zoom: detection brackets, bisection separates -------------------------------
ax[2].semilogy(zoom, s_zoom, color="0.35", lw=1.4)
ax[2].axvline(LA, color="green", lw=1.3, ls=":")
ax[2].axvline(LB, color="green", lw=1.3, ls=":")
ax[2].plot([], [], color="green", ls=":", lw=1.3, label="true eigenvalues")
print(f"lam_a={LA:.6f}  lam_b={LB:.6f}  d={D:.4e}")
for pg in (6, 24):
    xs, mid, ang, st, h = grids[pg]
    jj = np.nonzero((ang[1:-1] > ang[:-2]) & (ang[1:-1] >= ang[2:]))[0] + 1
    jj = jj[ang[jj] > 5 * np.median(ang)]
    near = (mid[jj][np.argmin(np.abs(mid[jj] - 0.5 * (LA + LB)))] if jj.size
            else np.inf)
    if not np.isfinite(near) or abs(near - 0.5 * (LA + LB)) > 3.0:
        ax[2].plot([], [], color=st["color"], lw=2.4,
                   label=f"per_gap={pg}: no peak near the pair $\\rightarrow$ ONE cell")
        print(f"  per_gap={pg:3d}  h={h:5.2f}  nearest peak {near:10.4f} -- "
              f"the pair's crossing was NEVER DETECTED")
        continue
    i_ = jj[np.argmin(np.abs(mid[jj] - 0.5 * (LA + LB)))]
    edge, path = locate(xs[i_], xs[i_ + 1], img(xs[i_]), img(xs[i_ + 1]))
    ax[2].axvspan(xs[i_], xs[i_ + 1], color=st["color"], alpha=0.10)
    ax[2].axvline(edge, color=st["color"], lw=2.4,
                  label=f"per_gap={pg}: bisected edge {edge:.5f} $\\rightarrow$ TWO cells")
    print(f"  per_gap={pg:3d}  h={h:5.2f}  raw peak {near:10.5f} (outside the pair)  "
          f"-> bisected to {edge:.7f}  inside? {LA < edge < LB}  ({len(path)} levels)")
ax[2].set_xlim(zoom[0], zoom[-1])
ax[2].set_title("3. Zoomed. Both minima are present. Detection only BRACKETS the crossing; "
                "bisection puts the edge between them.", fontsize=11, loc="left")
ax[2].set_ylabel(r"$\sigma_1$")
ax[2].set_xlabel(r"$\lambda$")
ax[2].legend(fontsize=8, loc="upper right")

for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("notes/pair_failure.png", dpi=150)
print("wrote notes/pair_failure.png")
