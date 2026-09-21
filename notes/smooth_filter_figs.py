"""Figures for `notes/smooth_filter.md`: what a smooth filter does to the tension curve.

Data comes from `notes/smooth_filter.py`'s machinery on `rect(1, 1/phi)` over [5, 300],
126 plane-wave columns -- the same pencil `landscape_findings.md` section 35 is built on.
EXPERIMENTAL; nothing in `src/` uses any of this.

Run from the repo root:  .venv/bin/python notes/smooth_filter_figs.py
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_S = ("/private/tmp/claude-501/-Users-hringer-projects-python-chladni/"
      "465cac43-5c87-4f75-aafb-75a65d03d14c/scratchpad/")
_A, _B = np.load(_S + "figdata.npz"), np.load(_S + "figdata2.npz")
D = {k: _A[k] for k in _A.files}
D.update({k: _B[k] for k in _B.files})

MUS = D["mus"]
#: threshold = the thing with the good floor; frozen = the thing with the smooth curve.
C_THRESH, C_FROZEN = "#c62828", "#37474f"
C_MU = ["#b39ddb", "#7986cb", "#1e88e5", "#00897b"]
C_HYB = "#ef6c00"
LBL = [f"filter $\\mu=10^{{{int(np.log10(m))}}}$" for m in MUS]
EPS = np.finfo(float).eps


def series(tag):
    """(threshold, frozen, [per-mu]) for a scan tagged `tag`."""
    return (D[tag + "_thresh"], D[tag + "_frozen"],
            [D[f"{tag}_mu{i}"] for i in range(len(MUS))])


def plot_all(ax, x, thr, frz, mus, lw=1.3, mu_idx=None):
    for i in (range(len(MUS)) if mu_idx is None else mu_idx):
        ax.semilogy(x, mus[i], color=C_MU[i], lw=lw, label=LBL[i], zorder=3)
    ax.semilogy(x, frz, color=C_FROZEN, lw=lw, ls="--", label="frozen rank", zorder=4)
    ax.semilogy(x, thr, color=C_THRESH, lw=lw, label="threshold rank", zorder=5)


# ============================================================ 1. the tension curve
def fig_tension():
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 9.2))
    fig.suptitle("What the filter does to the tension curve\n"
                 r"rect$(1,1/\varphi)$, 126 plane waves, frozen rank "
                 f"{int(D['frozen_rank'])}, threshold rank "
                 f"{int(D['rank_truth'].min())}\u2013{int(D['rank_truth'].max())} "
                 "over [5, 300]", fontsize=13, y=0.975)

    x, (thr, frz, mus) = D["wideA_x"], series("wideA")
    a = ax[0]
    a.semilogy(x, thr, color=C_THRESH, lw=1.0, label="threshold rank", zorder=5)
    a.semilogy(x, frz, color=C_FROZEN, lw=1.2, ls="--", label="frozen rank", zorder=4)
    for i in range(len(MUS)):
        a.semilogy(x, mus[i], color=C_MU[i], lw=1.1, label=LBL[i], zorder=3)
    a.set_title("A.  Context: three eigenvalues, linear axis. Every policy agrees on the "
                "plateau and on where the wells are, and at this\nscale they are "
                "indistinguishable \u2014 the wells are narrower than a pixel. The whole "
                "difference is in their last decades.",
                fontsize=10.5, loc="left")
    a.set_xlabel(r"$\lambda$")
    a.set_ylabel(r"$\sigma_1$")
    a.set_xlim(x[0], x[-1])
    a.set_ylim(2e-16, 3.0)
    a.legend(ncol=3, fontsize=9, loc="lower left", framealpha=0.95)
    a.annotate("panel B magnifies this one", xy=(261.3, 1e-13), xytext=(253.0, 2e-11),
               fontsize=9.5, color="#424242", ha="center",
               arrowprops=dict(arrowstyle="->", color="#424242", lw=1.2))

    a = ax[1]
    l0 = float(D["wellB_lam0"])
    rel = (D["wellB_x"] - l0)/l0
    thr, frz, mus = series("wellB")
    for i in range(len(MUS)):
        a.loglog(rel, mus[i], color=C_MU[i], lw=1.6, label=LBL[i], zorder=3)
    a.loglog(rel, frz, color=C_FROZEN, lw=1.8, ls="--", label="frozen rank", zorder=4)
    a.loglog(rel, thr, color=C_THRESH, lw=1.8, label="threshold rank", zorder=5)
    a.set_title(f"B.  The same well at $\\lambda={l0:.4f}$ (the frozen rank's worst), one "
                "side, log\u2013log. The V is a straight line of slope 1 and every policy\n"
                "rides it down until it hits its own floor. WHERE it leaves the line is the "
                "locating uncertainty \u2014 nine decades separate the incumbents.",
                fontsize=10.5, loc="left")
    a.set_xlabel(r"$(\lambda - \lambda_*)\,/\,\lambda_*$")
    a.set_ylabel(r"$\sigma_1$")
    a.set_ylim(1e-15, 3e-2)
    a.set_xlim(3e-16, 1.2e-2)
    a.legend(ncol=2, fontsize=9, loc="lower right", framealpha=0.95)
    a.annotate("", xy=(2.2e-4, 5.5e-4), xytext=(1.2e-15, 5.5e-4),
               arrowprops=dict(arrowstyle="<->", color="#616161", lw=1.3))
    a.annotate("nine decades of locating power", xy=(2e-10, 8e-4), fontsize=10,
               color="#424242", ha="center", weight="bold")
    for y, col, txt, xx in ((2.25e-4, C_FROZEN, "frozen rank leaves here", 2e-6),
                            (1.3e-9, C_MU[2], r"$\mu=10^{-9}$", 4e-12),
                            (2.9e-15, C_THRESH, "threshold", 2e-12)):
        a.axhline(y, color=col, lw=0.7, ls=":", zorder=1)
    for a in ax:
        a.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig("notes/fig_filter_tension.png", dpi=150)
    print("notes/fig_filter_tension.png")


# ============================================================ 2. the noise
def fig_jump():
    fig, ax = plt.subplots(2, 2, figsize=(12.2, 8.6))
    c = float(D["flick_c"])
    fig.suptitle("Not a jump \u2014 NOISE, at every scale\n"
                 f"a plateau point ($\\lambda \\approx {c:.4f}$), far from any eigenvalue, "
                 "zoomed in by four decades", fontsize=13, y=0.975)

    for i, a in enumerate((ax[0, 0], ax[0, 1], ax[1, 0])):
        x, (thr, frz, mus) = D[f"zoom{i}_x"], series(f"zoom{i}")
        w = float(D[f"zoom{i}_w"])
        d = (x - c)/w
        a.plot(d, thr, color=C_THRESH, lw=1.1, label="threshold rank")
        a.plot(d, frz, color=C_FROZEN, lw=1.6, ls="--", label="frozen rank")
        for j in (1, 2):
            a.plot(d, mus[j], color=C_MU[j], lw=1.6, label=LBL[j])
        a.set_xlabel(f"$(\\lambda-\\lambda_c)$  /  ${w:g}$")
        a.set_ylabel(r"$\sigma_1$")
        a.set_title(f"{'ABC'[i]}.  half-width {w:g}", fontsize=10.5, loc="left")
        a.grid(alpha=0.25, lw=0.5)
        a.ticklabel_format(axis="y", style="plain", useOffset=False)
        if i == 0:
            a.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax[1, 0].annotate("the grass never thins: the threshold rank flickers\n"
                      "between 39 and 40 at EVERY scale, 60-90 changes\n"
                      "per 200 samples whatever the interval",
                      xy=(0.02, 0.06), xycoords="axes fraction", fontsize=9,
                      color=C_THRESH, ha="left", va="bottom")

    a = ax[1, 1]
    W = D["flick_w"]
    a.loglog(W, D["flick_thresh"], "^-", color=C_THRESH, lw=1.7, ms=4,
             label="threshold rank")
    a.loglog(W, D["flick_frozen"], "s--", color=C_FROZEN, lw=1.7, ms=4, label="frozen rank")
    for i in range(len(MUS)):
        a.loglog(W, D[f"flick_mu{i}"], "o-", color=C_MU[i], lw=1.4, ms=3.2, label=LBL[i])
    for tag, ls, mk in (("hybA", "-", "D"), ("hybB", "--", "D")):
        a.loglog(W, D[tag+"_flick"], ls, color=C_HYB, lw=1.9, marker=mk, ms=4,
                 label=f"hybrid $c=${float(D[tag+'_c']):.0e}, $p=2$")
    a.loglog(W, 0.8*W, ":", color="#2e7d32", lw=1.4)
    a.annotate("slope 1 = smooth\n(spread $\\propto$ width)", xy=(1e-7, 8e-8),
               xytext=(4e-9, 2e-6), fontsize=9, color="#2e7d32", ha="center",
               arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.1))
    a.set_xlabel("half-width of the interval   (zooming in $\\rightarrow$)")
    a.set_ylabel(r"spread of $\sigma_1$  (ptp / median)")
    a.set_title("D.  The test for differentiability. A smooth function's spread shrinks\n"
                "with the interval; the threshold's FLATTENS at 4e-3 and stays there.",
                fontsize=10.5, loc="left")
    a.invert_xaxis()
    a.legend(fontsize=7.4, loc="lower left", ncol=2, framealpha=0.95)
    a.grid(alpha=0.25, which="both", lw=0.5)
    a.annotate("irreducible 0.4% noise floor", xy=(1e-9, 4e-3), xytext=(1e-6, 3e-2),
               fontsize=9, color=C_THRESH, ha="center",
               arrowprops=dict(arrowstyle="->", color=C_THRESH, lw=1.1))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("notes/fig_filter_jump.png", dpi=150)
    print("notes/fig_filter_jump.png")


# ============================================================ 3. the filter itself
def fig_filter():
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 6.3))
    fig.suptitle("The filter, drawn on the spectrum it acts on   "
                 r"(stacked pencil at $\lambda=150$, 126 columns)", fontsize=13, y=0.98)

    s = D["spectrum"]
    r = s / s[0]
    rtol = float(D["rtol"])
    a = ax[0]
    a.semilogy(np.arange(1, len(r) + 1), r, "o", ms=3.2, color="#1e88e5", zorder=3)
    a.axhline(rtol, color=C_THRESH, lw=1.4, ls="-")
    a.axhline(EPS, color="#6a1b9a", lw=1.2, ls=":")
    a.axhline(1e-11, color=C_MU[1], lw=1.2, ls="--")
    a.set_xlabel("index $j$")
    a.set_ylabel(r"$s_j\,/\,s_1$")
    a.set_title("A.  A smooth signal ramp with NO gap in it (so any cut inside it is\n"
                "arbitrary, and moves with $\\lambda$) — meeting a flat noise floor at a\n"
                "clear KNEE around $j=55$. That knee is not arbitrary.",
                fontsize=10.5, loc="left")
    a.annotate(f"threshold cut, rtol$={rtol:.1e}$", xy=(4, rtol), xytext=(62, 4e-14),
               fontsize=9, color=C_THRESH)
    a.annotate(r"$\varepsilon_{\rm mach}$", xy=(4, EPS), xytext=(4, 6e-17), fontsize=9,
               color="#6a1b9a")
    a.annotate(r"$\mu=10^{-11}$", xy=(4, 1e-11), xytext=(4, 2.2e-11), fontsize=9,
               color=C_MU[1])
    a.set_ylim(2e-17, 3)
    a.grid(alpha=0.25, which="both", lw=0.5)

    a = ax[1]
    t = np.logspace(-17, 0, 900)                    # s_j / s_1
    a.axvspan(1e-17, EPS, color="#ffcdd2", alpha=0.55, zorder=0)
    a.axvspan(EPS, 1e-13, color="#fff9c4", alpha=0.7, zorder=0)
    a.axvspan(1e-13, 1, color="#c8e6c9", alpha=0.55, zorder=0)
    a.plot(t, (t >= rtol).astype(float), color=C_THRESH, lw=2.2,
           label="hard truncation (today)")
    for i, mu in ((1, 1e-11), (2, 1e-9)):
        g = mu / (t * s[0])
        a.plot(t, 1.0 / (1.0 + g**2), color=C_MU[i], lw=1.8, label=LBL[i])
    for cc, pw, col, ls in ((rtol, 2.0, "#ef6c00", "-"), (rtol, 8.0, "#8d6e63", "--")):
        g = (cc / t) ** pw
        a.plot(t, 1.0 / (1.0 + g**2), color=col, lw=1.8, ls=ls,
               label=f"hybrid  $c={cc:.1e}$, $p={pw:g}$")
    a.plot(r, np.full_like(r, -0.055), "|", ms=9, color="#1e88e5", mew=1.1, clip_on=False,
           label="the actual $s_j/s_1$")
    a.set_xscale("log")
    a.set_xlim(1e-17, 1)
    a.set_ylim(-0.09, 1.52)
    a.axhline(1.0, color="#bdbdbd", lw=0.6, zorder=1)
    a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel(r"$s_j\,/\,s_1$")
    a.set_ylabel(r"filter factor $f_j = 1/(1+g_j^2)$")
    a.set_title("B.  Keep weight 1, drop weight 0. Truncation is a STEP, a filter a ramp\n"
                "\u2014 and WHERE the ramp sits is the whole design question.",
                fontsize=10.5, loc="left")
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3,
             framealpha=0.97)
    a.annotate("the hybrid puts its ramp HERE,\non the noise floor: smooth, and\n"
               "it costs the good directions nothing",
               xy=(2e-15, 0.55), xytext=(2.5e-12, 1.36), fontsize=8.5, color="#ef6c00",
               ha="left", va="top", weight="bold",
               arrowprops=dict(arrowstyle="->", color="#ef6c00", lw=1.2))
    a.annotate("plain Tikhonov's ramp sits out HERE, in\nthe middle of the signal \u2014 "
               "which is exactly\nwhy its floor is $\\mu$ and not $\\varepsilon$",
               xy=(1.2e-10, 0.45), xytext=(6e-9, 0.13), fontsize=8.5, color=C_MU[2],
               ha="left", va="bottom",
               arrowprops=dict(arrowstyle="->", color=C_MU[2], lw=1.2))
    a.annotate("null zone:\nhard-cut,\nnothing\nto protect", xy=(1.4e-17, 0.62),
               fontsize=8.5, color="#b71c1c", ha="left", va="center", weight="bold")
    a.annotate("kept: $f\\approx1$, untouched", xy=(2e-4, 1.10), fontsize=9.5,
               color="#1b5e20", ha="center", va="center", weight="bold")
    a.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout(rect=(0, 0.01, 1, 0.935))
    fig.savefig("notes/fig_filter_shape.png", dpi=150)
    print("notes/fig_filter_shape.png")


# ============================================================ 4. the trade
def fig_trade():
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4))
    fig.suptitle("The trade: one dial, and where it puts you", fontsize=13, y=0.98)

    a = ax[0]
    truth, rk = D["truth"], D["rank_truth"]
    a.semilogy(truth, D["floor_thresh"], "o-", color=C_THRESH, lw=1.6, ms=5,
               label="threshold rank")
    a.semilogy(truth, D["floor_frozen"], "s--", color=C_FROZEN, lw=1.6, ms=5,
               label="frozen rank")
    for i in range(len(MUS)):
        a.semilogy(truth, D[f"floor_mu{i}"], "^-", color=C_MU[i], lw=1.4, ms=4, label=LBL[i])
    for tag, ls in (("hybA", "-"), ("hybB", "--")):
        a.semilogy(truth, D[tag+"_floor"], ls, color=C_HYB, lw=2.0, marker="D", ms=4.5,
                   label=f"hybrid $c=${float(D[tag+'_c']):.0e}", zorder=6)
    a.set_xlabel(r"true eigenvalue $\lambda$")
    a.set_ylabel(r"$\sigma_1(\lambda_*)$   (the floor)")
    a.set_title("A.  The floor across a 60$\\times$ window. Both filters are FLAT; the frozen\n"
                "rank inflates by $2\\times10^{11}$, and the hybrid at $c=\\mathrm{rtol}$ sits "
                "on the threshold's own floor.", fontsize=10.5, loc="left")
    a.legend(fontsize=7.6, ncol=2, loc="center left", framealpha=0.95)
    a.grid(alpha=0.25, which="both", lw=0.5)
    a.set_ylim(3e-16, 3e-3)
    ax2 = a.twinx()
    ax2.plot(truth, rk, ":", color="#9e9e9e", lw=1.2)
    ax2.axhline(float(D["frozen_rank"]), color="#9e9e9e", lw=1.0, ls="--")
    ax2.set_ylabel("threshold rank (grey)", color="#757575", fontsize=9)
    ax2.tick_params(axis="y", colors="#757575", labelsize=8)

    a = ax[1]
    tm, tf, tn = D["trade_mu"], D["trade_floor"], D["trade_noise"]
    a.loglog(tf, tn, "-", color="#1e88e5", lw=1.8, zorder=3)
    sc = a.scatter(tf, tn, c=np.log10(tm), cmap="viridis", s=46, zorder=4,
                   edgecolor="white", lw=0.6)
    for mu, f_, n_ in zip(tm, tf, tn):
        if abs(np.log10(mu) - round(np.log10(mu))) < 1e-6 and int(np.log10(mu)) % 2:
            a.annotate(f"$10^{{{int(np.log10(mu))}}}$", (f_, n_), fontsize=8.5,
                       textcoords="offset points", xytext=(7, -10), color="#1565c0")
    a.loglog(D["hyb_floor"], D["hyb_noise"], "-", color=C_HYB, lw=2.0, zorder=6)
    a.plot(D["hyb_floor"], D["hyb_noise"], "D", color=C_HYB, ms=6, mec="white", mew=0.7,
           zorder=7, label="hybrid, $p=2$")
    a.annotate("HYBRID\nhard-cut the null zone,\nramp across the band", xy=(8e-13, 1.1e-9),
               fontsize=9.5, color=C_HYB, weight="bold", ha="left", va="center")
    for nm, col, mk, lab in (("thresh", C_THRESH, "*", "threshold rank"),
                             ("frozen", C_FROZEN, "X", "frozen rank")):
        a.plot(D["pt_floor_" + nm], D["pt_noise_" + nm], mk, color=col, ms=17,
               mec="white", mew=1.0, zorder=5, label=lab)
        a.annotate(lab, (D["pt_floor_" + nm], D["pt_noise_" + nm]), fontsize=10,
                   textcoords="offset points",
                   xytext=((6, -18) if nm == "thresh" else (-10, 16)), color=col,
                   weight="bold", ha=("left" if nm == "thresh" else "right"))
    a.set_xlabel("worst floor over the window   (= deliverable precision)")
    a.set_ylabel("worst plateau noise   (= what a search fights)")
    a.set_title("B.  Both filters are CURVES through the plane the two incumbents are\n"
                "points in. The hybrid runs below and two decades further left.",
                fontsize=10.5, loc="left")
    a.grid(alpha=0.25, which="both", lw=0.5)
    a.annotate("plain Tikhonov saturates here:\na penalty cannot remove a\nnull direction",
               xy=(float(tf[0]), float(tn[0])), xytext=(8e-12, 1.5e-4), fontsize=9,
               color="#1565c0", ha="left",
               arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.1))
    a.annotate("best corner: low floor AND low noise", xy=(0.03, 0.045),
               xycoords="axes fraction", fontsize=9, color="#424242", weight="bold")
    cb = fig.colorbar(sc, ax=a, pad=0.02)
    cb.set_label(r"$\log_{10}\mu$", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig("notes/fig_filter_trade.png", dpi=150)
    print("notes/fig_filter_trade.png")


if __name__ == "__main__":
    fig_tension()
    fig_jump()
    fig_filter()
    fig_trade()
