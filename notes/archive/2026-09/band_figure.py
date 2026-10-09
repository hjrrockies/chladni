"""The solvability band, on one rectangle, from measured data.

rect(1+delta, 1), modes (2,1) and (1,2) at lam = 5*pi^2. Sweeping delta moves the pair's
relative separation d/lam over twelve decades while nothing else about the problem changes.
For a requested `prec` there are three regimes:

    d/lam < prec        the pair may legally be merged   -> solved, cheaply
    d/lam > ~1e-4       it must be split, and can be     -> solved
    prec < d/lam < 1e-4 it must be split, and cannot     -> BEYOND THE LIMIT

The band's width is `1e-4 / prec`: empty at prec = 1e-4, four decades at 1e-8, ten at 1e-14.

Markers are measured, from `all_run.log`: filled where at least one of the six bracketers
reached coverage 1.0 with zero impure brackets inside 1500 evaluations, open where none did.
"""
import json
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil
from chladni.reference import rect_eig

S = ("/private/tmp/claude-501/-Users-hringer-projects-python-chladni/"
     "50c64400-9259-42d4-80e9-db814c63727f/scratchpad")
rows = json.load(open(f"{S}/band_data.json"))
PRECS = [1e-4, 1e-8, 1e-11, 1e-14]
SPLIT_LIMIT = 1e-4          # measured: no method splits below this

fig = plt.figure(figsize=(12, 13.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 0.85], hspace=0.72, wspace=0.30)
fig.suptitle("The solvability band on rect$(1+\\delta,\\,1)$: modes (2,1) and (1,2) at "
             "$\\lambda=5\\pi^2$\nsweeping $\\delta$ moves $d/\\lambda$ over twelve "
             "decades; nothing else about the problem changes", fontsize=13, y=0.965)

# --- Panel A: the band diagram ----------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
gx = np.logspace(-15, -1.3, 500)          # d/lam
gy = np.logspace(-15, -3.3, 500)          # prec
X, Y = np.meshgrid(gx, gy)
cls = np.where(X > SPLIT_LIMIT, 0, np.where(X < Y, 0, 1))   # 0 solvable, 1 band
ax.pcolormesh(gx, gy, cls, cmap=matplotlib.colors.ListedColormap(["#c8e6c9", "#ffcdd2"]),
              shading="auto", vmin=0, vmax=1, zorder=0)
ax.axvline(SPLIT_LIMIT, color="#c62828", lw=1.8, ls="--", zorder=2)
ax.plot(gx, gx, color="#2e7d32", lw=1.8, ls="--", zorder=2)
for r in rows:
    dl = r[1]
    for i, p_ in enumerate(PRECS):
        n = r[2 + i][0]
        ax.plot(dl, p_, "o", ms=9, mfc=("#1b5e20" if n else "white"),
                mec=("#1b5e20" if n else "#b71c1c"), mew=1.7, zorder=5)
        if n:
            ax.annotate(str(n), (dl, p_), fontsize=6.5, color="white", ha="center",
                        va="center", zorder=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(3e-15, 5e-2); ax.set_ylim(3e-15, 5e-4)
ax.invert_yaxis()
ax.set_xlabel(r"pair separation $d/\lambda$")
ax.set_ylabel(r"requested $\mathrm{prec}$   (tighter $\rightarrow$ up)")
ax.set_title("A.  Measured. Filled = at least one of six bracketers reached coverage 1.0 with "
             "no impure bracket\nin $\\leq$1500 evaluations (digit = how many of the six); "
             "open = all six failed.", fontsize=10.5, loc="left")
ax.annotate("must SPLIT,\nand can", xy=(3.0e-2, 3e-7), fontsize=10, ha="center",
            color="#1b5e20", weight="bold")
ax.annotate("may MERGE\n($d/\\lambda<\\mathrm{prec}$)", xy=(1.5e-14, 6e-6),
            fontsize=10, ha="center", color="#1b5e20", weight="bold")
ax.annotate("THE BAND\nmust split, cannot", xy=(3e-8, 1.5e-9), fontsize=11,
            ha="center", color="#b71c1c", weight="bold")
ax.annotate("search limit $d/\\lambda\\approx10^{-4}$", xy=(1e-4, 6e-15),
            xytext=(1.5e-7, 5.5e-15), fontsize=9, color="#c62828",
            arrowprops=dict(arrowstyle="->", color="#c62828"))
ax.grid(alpha=0.3, which="both", zorder=1)

# --- Panel B: what the search sees, at three separations ---------------------------------
dom = rect(1.0, 1.0)
lam0 = 5 * np.pi**2
nw = int(1.6 * PlaneWave.n_useful(np.sqrt(88.0) * dom.diameter))
titles = [("solvable: must split, and does", 1e-4, "#1b5e20"),
          ("in the band at prec=1e-8: must split, cannot", 1e-6, "#b71c1c"),
          ("may merge at prec=1e-8", 1e-11, "#1b5e20")]
for j, (lab, delta, col) in enumerate(titles):
    axb = fig.add_subplot(gs[1, j])
    L = 1.0 + delta
    d2 = rect(L, 1.0)
    p = MPSPencil.from_basis(PlaneWave.uniform(nw, radius=d2.diameter), d2.bdry_quad(1200),
                             d2.interior_pts(600, rng=np.random.default_rng(0)), rtol=1e-16)
    la, lb = rect_eig(2, 1, L, 1.0), rect_eig(1, 2, L, 1.0)
    d = abs(lb - la); mid = 0.5 * (la + lb)
    zoom = np.linspace(mid - 3 * d, mid + 3 * d, 220)
    s1 = np.array([p.tension(x) for x in zoom])
    axb.semilogy((zoom - mid) / d, s1, color="0.3", lw=1.4)
    for v in (la, lb):
        axb.axvline((v - mid) / d, color=col, ls=":", lw=1.2)
    axb.set_title(f"B{j+1}.  $d/\\lambda$={d/mid:.1e}\n{lab}", fontsize=9.5, color=col)
    axb.set_xlabel(r"$(\lambda-\lambda_{mid})\,/\,d$")
    if j == 0:
        axb.set_ylabel(r"$\sigma_1$")
    axb.grid(alpha=0.25)
fig.text(0.5, 0.305, "B.  The curve is the SAME shape in all three — the x-axis is in units "
         "of $d$. Only the SCALE differs, by seven decades.\nNothing about the objective gets "
         "harder; the search simply has to look at a finer scale to see it.",
         ha="center", fontsize=10, style="italic")

# --- Panel C: measured cost against separation -------------------------------------------
axc = fig.add_subplot(gs[2, :])
for i, (p, c, mk) in enumerate(zip(PRECS, ["#1565c0", "#6a1b9a", "#ef6c00", "#c62828"],
                                   ["o", "s", "^", "D"])):
    dl = [r[1] for r in rows if r[2 + i][1] is not None]
    cost = [r[2 + i][1] for r in rows if r[2 + i][1] is not None]
    o = np.argsort(dl)
    axc.plot(np.array(dl)[o], np.array(cost)[o], mk + "-", color=c, ms=5, lw=1.2,
             label=f"prec={p:.0e}")
    fails = [r[1] for r in rows if r[2 + i][1] is None]
    axc.plot(fails, [1500] * len(fails), "x", color=c, ms=7, mew=1.8)
axc.axhline(1500, color="k", ls="--", lw=1, alpha=0.6)
axc.annotate("budget cap 1500;  $\\times$ = no setting succeeded", xy=(4e-13, 1700),
             fontsize=9, color="0.3")
axc.axvline(SPLIT_LIMIT, color="#c62828", lw=1.4, ls="--")
axc.set_xscale("log"); axc.set_yscale("log")
axc.set_xlabel(r"pair separation $d/\lambda$"); axc.set_ylabel("evaluations (cheapest setting)")
axc.set_title("C.  Measured cost of the cheapest successful setting. The flat shelf at 86 "
              "evaluations is the MERGE regime:\nthe pair is inside prec, so nothing has to be "
              "split and the whole window is cheap.", fontsize=10.5, loc="left")
axc.set_ylim(60, 3000)
axc.legend(fontsize=9, ncol=4, loc="lower right", framealpha=0.95)
axc.grid(alpha=0.25, which="both")

fig.savefig("notes/solvability_band.png", dpi=150, bbox_inches="tight")
print("wrote notes/solvability_band.png")
