"""Regenerate the figures in `docs/bracket_search.pdf`.

    uv run python docs/figures/make_bracket_figures.py            # all of them
    uv run python docs/figures/make_bracket_figures.py --only ladder

Every figure is computed from the code it illustrates, not drawn by hand: the exclusion
picture calls `v1_bracket.live_regions`, the ladder calls `wall_ladder_ratio` and then
measures the ratio it predicts, the straddle sweep drives `v1_bracket._locator_step`, and
the tension curves come from real `MPSPencil` evaluations. Where a caption states a number,
the script prints it, so a caption that has drifted from the code is visible.

The searcher candidates live in `tests/`, which is where a scored candidate lives until it
wins (`docs/design/search.md`, As built). That is why this script puts `tests/` on the path.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
HERE = Path(__file__).resolve().parent

warnings.simplefilter("ignore")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

#: The document is 11pt article at 1.1in margins, so the text block is 6.3in.
TEXTWIDTH = 6.3

C_LIVE = "#c0392b"
C_DEAD = "#95a5a6"
C_CURVE = "#2c3e50"
C_MODEL = "#2980b9"
C_ACCENT = "#8e44ad"


def _save(fig, name):
    out = HERE / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# =================================================================================================
# Figure: the exclusion ladder
# =================================================================================================

def _measured_ladder_ratio(safety, c=0.7, n=40, lo=1e-9, hi=1e3):
    """The largest geometric ratio whose ladder still excludes every pair on a cone's wall.

    Bisected against `live_regions` itself, so what is measured is the code's own verdict and
    not a re-derivation of it.
    """
    from v1_bracket import live_regions

    def covered(r):
        # One wall of a cone at the origin, sampled at u = r**j. Exclusion is scale-free, so
        # the absolute placement does not matter.
        u = r ** np.arange(n, dtype=float)
        ks, vs = u, c * u
        return not live_regions(ks, vs, safety * c)

    a, b = 1.0 + 1e-12, 1e6
    if not covered(a):
        return float("nan")
    for _ in range(200):
        m = 0.5 * (a + b)
        if covered(m):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def fig_ladder():
    """A wall is covered by a geometric ladder, and its ratio follows from `safety` alone."""
    from v1_bracket import SAFETY, live_regions, wall_ladder_ratio

    safeties = np.array([1.2, 1.5, 2.0, 3.0, 5.0])
    predicted = np.array([wall_ladder_ratio(s) for s in safeties])
    measured = np.array([_measured_ladder_ratio(s) for s in safeties])
    print("  safety    predicted   measured")
    for s, pr, m in zip(safeties, predicted, measured):
        print(f"  {s:5.2f}    {pr:9.5f}   {m:9.5f}")

    c, L = 1.0, SAFETY * 1.0
    r_crit = wall_ladder_ratio(SAFETY)
    fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15))

    # --- (a) one pair on the wall: the envelope test at the critical ratio, and past it ---
    a = ax[0]
    for r, style, col, lab in ((r_crit, "-", C_MODEL, rf"$r={r_crit:.0f}$: excluded"),
                               (1.45 * r_crit, "--", C_LIVE, rf"$r={1.45*r_crit:.1f}$: live")):
        ub, ua = 1.0, r
        vb, va = c * ub, c * ua
        xs = np.linspace(ub, ua, 400)
        env = np.maximum(vb - L * (xs - ub), va - L * (ua - xs))
        a.plot(xs / r_crit, env, style, color=col, lw=1.3, label=lab)
    xs = np.linspace(0.0, 1.55 * r_crit, 400)
    a.plot(xs / r_crit, c * xs, color=C_CURVE, lw=1.1, alpha=.5, zorder=1)
    a.axhline(0, color="k", lw=.8)
    a.plot([1.0 / r_crit, 1.0], [c, c * r_crit], "o", ms=4, color=C_ACCENT, zorder=5)
    a.set_ylim(-2.2, 8)
    a.set_xlabel(r"$u$, in units of $r_{\mathrm{crit}}$")
    a.set_ylabel("Piyavskii bound")
    a.set_title("(a) one pair on the wall")
    a.legend(loc="lower right", frameon=False, handlelength=1.4)

    # --- (b) the ladder on one wall: geometric, so equispaced in log u ---
    a = ax[1]
    for j, (rat, col, lab) in enumerate(((r_crit, C_MODEL, rf"$r={r_crit:.0f}$"),
                                         (1.45 * r_crit, C_LIVE,
                                          rf"$r={1.45*r_crit:.1f}$"))):
        u = rat ** np.arange(0.0, 6.0)
        vs = c * u
        n_live = len(live_regions(u, vs, L))
        a.plot(u, vs, "o-", ms=4, lw=.8, color=col, mfc="w",
               label=f"{lab}: {n_live} live")
        for i in range(len(u) - 1):
            if live_regions(u[i:i + 2], vs[i:i + 2], L):
                a.axvspan(u[i], u[i + 1], color=C_LIVE, alpha=.12, lw=0)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"distance from the minimizer, $u$")
    a.set_ylabel(r"$\sigma_1$")
    a.set_title("(b) a ladder up one wall")
    a.legend(loc="upper left", frameon=False, handlelength=1.4)

    # --- (c) predicted against measured ---
    a = ax[2]
    ss = np.linspace(1.05, 5.2, 400)
    a.plot(ss, [wall_ladder_ratio(s) for s in ss], color=C_MODEL, lw=1.2,
           label=r"$(s{+}1)/(s{-}1)$")
    a.plot(safeties, measured, "o", ms=5.5, color=C_ACCENT, mfc="none", mew=1.4,
           label="measured")
    a.set_yscale("log")
    a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlabel(r"safety factor $s$")
    a.set_ylabel("coarsest ratio")
    a.set_title("(c) the law, measured")
    a.legend(frameon=False, handlelength=1.4)
    _save(fig, "ladder")


# =================================================================================================
# Figure: cover by exclusion
# =================================================================================================

def _rect_case(L=1.0, H=None, lam_lo=30.0, lam_hi=200.0):
    """A plane-wave rectangle pencil and its complete closed-form truth, in `k`."""
    from rect_suite import PHI, pencil_for
    from chladni.reference import rect_spectrum

    H = 1.0 / PHI if H is None else H
    pen = pencil_for(L, H, lam_lo, lam_hi)
    truth = np.asarray(rect_spectrum(lam_lo, lam_hi, L, H), dtype=float)
    return pen, np.sqrt(truth)


def fig_exclusion():
    """The Piyavskii sawtooth on a real tension curve, and what it certifies empty."""
    from v1_bracket import SAFETY, _envelope_min, _secants, live_regions

    pen, k_true = _rect_case()
    k_lo, k_hi = np.sqrt(30.0), np.sqrt(200.0)
    sig = lambda k: pen.tension(k * k)

    n = 41
    ks = np.linspace(k_lo, k_hi, n)
    vs = np.array([sig(k) for k in ks])
    L = SAFETY * float(np.max(_secants(ks, vs, k_hi - k_lo)))
    live = live_regions(ks, vs, L)
    n_excluded = sum(1 for i in range(n - 1) if _envelope_min(ks, vs, i, L) >= 0.0)
    covered = sum(1 for t in k_true if any(lo <= t <= hi for lo, hi in live))
    print(f"  {n} samples, L = {L:.4f}, {n_excluded} of {n-1} cells certified empty, "
          f"{len(live)} live regions holding {covered} of {len(k_true)} eigenvalues")

    kd = np.linspace(k_lo, k_hi, 900)
    dense = np.array([sig(k) for k in kd])

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.7))
    for lo, hi in live:
        ax.axvspan(lo, hi, color=C_LIVE, alpha=.16, lw=0, zorder=0)
    ax.semilogy(kd, dense, color=C_CURVE, lw=.9, zorder=3, label=r"$\sigma_1(k)$")

    # the sawtooth: two cones of slope L dropping from each pair of adjacent samples
    for i in range(n - 1):
        xs = np.linspace(ks[i], ks[i + 1], 60)
        env = np.maximum(vs[i] - L * (xs - ks[i]), vs[i + 1] - L * (ks[i + 1] - xs))
        pos = env > 0
        ax.plot(xs[pos], env[pos], color=C_MODEL, lw=.8, alpha=.85, zorder=2,
                label=r"Piyavskii bound" if i == 0 else None)
    ax.plot(ks, vs, "o", ms=3.4, color=C_ACCENT, zorder=4, label="samples")
    for t in k_true:
        ax.axvline(t, color="k", lw=.6, ls=":", alpha=.55, zorder=1)
    ax.set_ylim(max(dense.min() * 0.3, 1e-13), 3.0)
    ax.set_xlim(k_lo, k_hi)
    ax.set_xlabel(r"$k=\sqrt{\lambda}$")
    ax.set_ylabel(r"$\sigma_1$")
    ax.set_title(f"{n_excluded} of {n-1} cells certified empty; "
                 f"the {len(live)} live regions hold all {len(k_true)} eigenvalues "
                 "(dotted)")
    ax.legend(loc="lower left", frameon=False, ncols=3)
    _save(fig, "exclusion")


# =================================================================================================
# Figure: why the wavenumber
# =================================================================================================

#: `u = lam**p` for each `p`, plus `log lam` as the natural non-power comparison.
_TRANSFORMS = ((1.00, r"$\lambda$"), (0.75, r"$\lambda^{3/4}$"),
               (0.50, r"$\sqrt{\lambda}=k$"), (0.25, r"$\lambda^{1/4}$"),
               (None, r"$\log\lambda$"))


def _wall_slopes(pen, truth, gap, fracs=(0.03, 0.05, 0.07)):
    """The wall slope in `lam` at each level clear of its neighbours, from the straight flank.

    Offsets are fractions of the Weyl gap rather than of `lam`: the spacing is uniform in
    `lam`, so a relative offset would walk off the dip at the top of the window.
    """
    hs = np.array(fracs) * gap
    lams, cs = [], []
    for t in truth:
        others = np.delete(truth, int(np.argmin(np.abs(truth - t))))
        if others.size and np.min(np.abs(others - t)) < 4.0 * hs[-1]:
            continue                      # too close to a neighbour to read one wall
        lams.append(t)
        cs.append(float(np.median([pen.tension(t + h) / h for h in hs])))
    return np.array(lams), np.array(cs)


def fig_whyk():
    """`dsigma/du` is constant in `lam**p` only for `p = 1/2`, and the data agree."""
    from rect_suite import PHI, pencil_for
    from chladni.reference import rect_spectrum

    L, H = 1.0, 1.0 / PHI
    lam_lo, lam_hi = 10.0, 520.0
    pen = pencil_for(L, H, lam_lo, lam_hi)
    truth = np.unique(np.asarray(rect_spectrum(lam_lo, lam_hi, L, H), dtype=float))
    lams, c_lam = _wall_slopes(pen, truth, gap=4.0 * np.pi / (L * H))
    print(f"  {len(lams)} isolated levels over lam in [{lam_lo:g}, {lam_hi:g}]")

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    spreads = []
    for j, (pw, lab) in enumerate(_TRANSFORMS):
        # dsigma/du = c_lam * dlam/du.  u = lam**p -> dlam/du = lam**(1-p)/p;  u = log lam -> lam
        cu = c_lam * (lams if pw is None else lams ** (1.0 - pw) / pw)
        spread = float(cu.std() / cu.mean())
        spreads.append(spread)
        best = (pw == 0.5)
        ax[0].plot(np.full_like(cu, j) + np.linspace(-.16, .16, cu.size),
                   cu / np.median(cu), "o", ms=3.5,
                   color=C_ACCENT if best else C_DEAD,
                   mfc=C_ACCENT if best else "none", zorder=3 if best else 2)
        print(f"  {lab:22s} ratio {cu.max()/cu.min():5.2f}   spread {100*spread:4.1f}%")

    ax[0].axhline(1.0, color="k", lw=.7, alpha=.5)
    ax[0].set_xticks(range(len(_TRANSFORMS)))
    ax[0].set_xticklabels([t[1] for t in _TRANSFORMS])
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$d\sigma/du$, relative to its median")
    ax[0].set_title("wall slope across levels, per variable")

    ax[1].bar(range(len(_TRANSFORMS)), 100 * np.array(spreads),
              color=[C_ACCENT if t[0] == 0.5 else C_DEAD for t in _TRANSFORMS], width=.6)
    for j, sp in enumerate(spreads):
        ax[1].text(j, 100 * sp + 1.2, f"{100*sp:.0f}%", ha="center", fontsize=8)
    ax[1].set_xticks(range(len(_TRANSFORMS)))
    ax[1].set_xticklabels([t[1] for t in _TRANSFORMS])
    ax[1].set_ylabel("spread (std / mean)")
    ax[1].set_ylim(0, 100 * max(spreads) * 1.22)
    ax[1].set_title(r"$p=1/2$ is the exponent that cancels")
    _save(fig, "whyk")


# =================================================================================================
# Figure: the isolation gate
# =================================================================================================

class _PencilView:
    """The `.sigma(k, j)` seam `v1_bracket.isolation` reads, backed by a real pencil.

    The searcher never sees a pencil -- it is handed an `Objective` (`tests/objective.py`) and
    works in `k`. This is that same seam with the cache and the counter left out, which is all
    the gate needs.
    """

    def __init__(self, pen):
        self.pen = pen

    def sigma(self, k, j):
        v = self.pen.tensions(float(k) ** 2)
        return float(v[j]) if j < len(v) else float("inf")


def _gate_reading(pen, k_star, half, n=9):
    """What `isolation` reports at a level, from a symmetric descent triple around it."""
    from v1_bracket import isolation

    ks = k_star + np.linspace(-half, half, n)      # odd `n`, so `k_star` is a sample
    vs = np.array([pen.tension(k * k) for k in ks])
    return isolation(_PencilView(pen), ks, vs, ks[0], ks[-1], c_bound=0.0)


def fig_gate():
    """`sigma_2/c` is a free distance meter, accurate where the decision is hard."""
    from rect_suite import PHI, pencil_for
    from chladni.reference import rect_spectrum

    # A pair the pencil resolves, and a wide window whose levels are far apart.
    cases = [(1.0 + 8.4e-4, 1.0, 40.0, 60.0), (1.0, 1.0 / PHI, 30.0, 200.0)]
    est, true = [], []
    for L, H, lo, hi in cases:
        pen = pencil_for(L, H, lo, hi)
        kt = np.sqrt(np.unique(np.asarray(rect_spectrum(lo, hi, L, H), dtype=float)))
        for j, k in enumerate(kt):
            others = np.delete(kt, j)
            if not others.size:
                continue
            d = float(np.min(np.abs(others - k)))
            iso = _gate_reading(pen, k, half=0.30 * d)
            if iso is None:
                continue
            est.append(iso["neighbour"])
            true.append(d)
    est, true = np.array(est), np.array(true)
    ratio = est / true
    close = true <= 1e-2
    print(f"  {len(est)} levels: close neighbours read "
          f"{ratio[close].min():.2f}-{ratio[close].max():.2f}, "
          f"far ones {ratio[~close].min():.2f}-{ratio[~close].max():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))

    # --- (a) the two branches over a resolvable pair ---
    L, H, lo, hi = cases[0]
    pen = pencil_for(L, H, lo, hi)
    kt = np.sqrt(np.asarray(rect_spectrum(lo, hi, L, H), dtype=float))
    mid, span = kt.mean(), 6.0 * (kt[1] - kt[0])
    kd = np.linspace(mid - span, mid + span, 500)
    sv = np.array([pen.tensions(k * k)[:2] for k in kd])
    a = ax[0]
    a.semilogy(kd, sv[:, 0], color=C_CURVE, lw=1.1, label=r"$\sigma_1$")
    a.semilogy(kd, sv[:, 1], color=C_MODEL, lw=1.1, ls="--", label=r"$\sigma_2$")
    for t in kt:
        a.axvline(t, color="k", lw=.6, ls=":", alpha=.6)
    a.set_xlabel(r"$k$")
    a.set_ylabel(r"$\sigma_j$")
    a.set_title(rf"a pair $\Delta k={kt[1]-kt[0]:.1e}$ apart")
    a.legend(frameon=False, loc="lower left")

    # --- (b) the meter against the truth ---
    a = ax[1]
    lim = np.array([0.5 * true.min(), 2.0 * true.max()])
    a.fill_between(lim, lim * 1e-3, lim, color=C_MODEL, alpha=.10, lw=0)
    a.loglog(lim, lim, color="k", lw=.8, label="exact")
    a.loglog(true[close], est[close], "o", ms=5, color=C_ACCENT,
             label=r"close ($\leq 10^{-2}$)")
    a.loglog(true[~close], est[~close], "s", ms=4.5, color=C_DEAD, mfc="none", mew=1.2,
             label="far")
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel("true distance to the next level")
    a.set_ylabel(r"$\sigma_2/c$")
    a.set_title("under-reads (shaded) where it errs")
    a.legend(frameon=False, loc="upper left")
    _save(fig, "gate")


# =================================================================================================
# Figure: predict, then straddle
# =================================================================================================

def _straddle_run(pen, k_star, half, straddle, target, cap=60):
    """Evaluations for `_locator_step` to contract a bracket to `target`, or `None`.

    The locator is a pure function of the samples, so nothing here reimplements it: the module
    constant is set, the step is asked where to sample, and the answer is evaluated.
    """
    import v1_bracket as v1

    old, v1.STRADDLE = v1.STRADDLE, straddle
    try:
        ks = k_star + np.array([-half, -0.21 * half, half])   # an off-centre starting triple
        vs = np.array([pen.tension(k * k) for k in ks])
        widths = []
        for it in range(cap):
            step = v1._locator_step(ks, vs, ks[0], ks[-1], 1e-16)
            if step is None:
                return None, np.array(widths)        # the descent was lost: no bracket left
            x, a, b = step
            widths.append(b - a)
            if b - a <= target:
                return it, np.array(widths)
            i = int(np.searchsorted(ks, x))
            ks = np.insert(ks, i, x)
            vs = np.insert(vs, i, pen.tension(x * x))
        return None, np.array(widths)
    finally:
        v1.STRADDLE = old


#: Offsets swept, as a fraction of the bracket. `v1_bracket.STRADDLE` is 0.02.
_STRADDLES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15)


def _prediction_errors(pen, levels, half, w_min=1e-4):
    """|vertex - k*| as a fraction of the bracket, over a locator run at each level.

    This is the quantity the offset has to beat: an offset smaller than the prediction error
    lands on the wrong side of the minimizer and the triple stops being a descent. The vertex
    is clipped to the triple exactly as `_locator_step` clips it.

    Brackets below `w_min` are excluded, and deliberately: once the triple is tight enough that
    `sigma_1**2` differences fall to the arithmetic's own noise the fit degrades badly (the
    error reaches 12% of the bracket), and the search never gets there -- it stops at isolation
    and hands the bracket to the refiner, which works in `lam`. Including that regime would
    measure the stall rather than the prediction.
    """
    import v1_bracket as v1

    out = []
    for k_star in levels:
        ks = k_star + np.array([-half, -0.21 * half, half])
        vs = np.array([pen.tension(k * k) for k in ks])
        for _ in range(30):
            cert = v1.descent_certificate(ks, vs, ks[0], ks[-1])
            if cert is None:
                break
            j, a, mid, b = cert
            if b - a < w_min:
                break
            x = np.array([a, mid, b])
            y = np.array([vs[j - 1], vs[j], vs[j + 1]]) ** 2
            c = np.polyfit(x, y, 2)
            if c[0] > 0:
                vtx = float(np.clip(-0.5 * c[1] / c[0], x[0], x[2]))
                out.append(abs(vtx - k_star) / (b - a))
            step = v1._locator_step(ks, vs, ks[0], ks[-1], 1e-16)
            if step is None:
                break
            xx = step[0]
            i = int(np.searchsorted(ks, xx))
            ks = np.insert(ks, i, xx)
            vs = np.insert(vs, i, pen.tension(xx * xx))
    return np.array(out)


def fig_straddle():
    """Sampling AT the prediction leaves one flank fixed; a small offset contracts both."""
    import v1_bracket as v1
    from rect_suite import PHI, pencil_for
    from chladni.reference import rect_spectrum

    L, H = 1.0, 1.0 / PHI
    lam_lo, lam_hi = 30.0, 200.0
    pen = pencil_for(L, H, lam_lo, lam_hi)
    kt = np.sqrt(np.unique(np.asarray(rect_spectrum(lam_lo, lam_hi, L, H), dtype=float)))
    gaps = np.diff(np.concatenate(([0.0], kt)))
    levels = [float(k) for k, g in zip(kt, gaps) if g > 0.25][:5]
    half, target = 0.12, 1e-6

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))

    # --- (a) one step: the parabola on sigma_1**2, its vertex, and the offset sample ---
    k_star = levels[0]
    ks = k_star + np.array([-half, -0.21 * half, half])
    vs = np.array([pen.tension(k * k) for k in ks])
    x, _, _ = v1._locator_step(ks, vs, ks[0], ks[-1], 1e-16)
    y = vs ** 2
    coef = np.polyfit(ks, y, 2)
    vertex = -0.5 * coef[1] / coef[0]
    kd = np.linspace(ks[0], ks[-1], 400)
    a = ax[0]
    a.plot(kd, np.array([pen.tension(k * k) for k in kd]) ** 2, color=C_CURVE, lw=1.1,
           label=r"$\sigma_1^2$")
    a.plot(kd, np.polyval(coef, kd), color=C_MODEL, lw=1.1, ls="--", label="the parabola")
    a.plot(ks, y, "o", ms=5, color=C_ACCENT, label="the triple")
    a.axvline(vertex, color=C_MODEL, lw=.8, ls=":")
    a.set_ylim(bottom=-0.03 * y.max())
    a.set_xlabel(r"$k$")
    a.set_ylabel(r"$\sigma_1^2$")
    a.set_title("(a) predict, then straddle")

    # The offset is 2% of a bracket, so it is invisible at full scale: zoom on the bottom.
    ins = a.inset_axes([0.06, 0.42, 0.40, 0.52])
    span = 0.06 * (ks[-1] - ks[0])
    kz = np.linspace(vertex - span, vertex + span, 200)
    ins.plot(kz, np.array([pen.tension(k * k) for k in kz]) ** 2, color=C_CURVE, lw=1.0)
    ins.axvline(vertex, color=C_MODEL, lw=1.0, ls=":")
    ins.axvline(x, color=C_LIVE, lw=1.2)
    ins.axvline(k_star, color="k", lw=.8, ls="--", alpha=.6)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("vertex, sample, truth", fontsize=7, pad=2)
    a.indicate_inset_zoom(ins, edgecolor=C_DEAD, lw=.7)
    a.legend(frameon=False, loc="lower right", fontsize=7.5)

    # --- (b) the sweep: cost rises with the offset, and it is floored by the prediction ---
    a = ax[1]
    err = _prediction_errors(pen, levels, half)
    lo_e, hi_e = np.percentile(err, [50, 90])
    print(f"  prediction error over {len(err)} steps: median {lo_e:.1e}, "
          f"90th percentile {hi_e:.1e} of the bracket")
    means = []
    for st in _STRADDLES:
        got = [_straddle_run(pen, k, half, st, target)[0] for k in levels]
        ok = [n for n in got if n is not None]
        means.append(np.mean(ok) if ok else np.nan)
        print(f"  straddle {st:>6g}: {means[-1]:5.1f} evaluations to width {target:g}"
              + ("" if len(ok) == len(got) else
                 f"  ({len(got)-len(ok)} of {len(got)} lost the descent)"))
    a.axvspan(lo_e, hi_e, color=C_LIVE, alpha=.16, lw=0)
    a.text(hi_e, 0.04, " prediction\n error", transform=a.get_xaxis_transform(),
           fontsize=7.5, color=C_LIVE, va="bottom", ha="left")
    a.plot(_STRADDLES, means, "-o", ms=5, color=C_ACCENT)
    a.axvline(v1.STRADDLE, color=C_DEAD, lw=.9, ls=":")
    a.annotate("shipped", xy=(v1.STRADDLE, means[_STRADDLES.index(v1.STRADDLE)]),
               xytext=(0.60, 0.20), textcoords="axes fraction", fontsize=7.5, color=C_DEAD,
               arrowprops=dict(arrowstyle="->", color=C_DEAD, lw=.9))
    a.set_xscale("log")
    a.set_xlabel("straddle, as a fraction of the bracket")
    a.set_ylabel(f"evaluations to width {target:g}")
    a.set_title("(b) what sets the offset")
    _save(fig, "straddle")


# =================================================================================================
# Figure: cover in k, refine in lambda
# =================================================================================================

#: Rectangles the fit comparison runs over: `(L, H, lam_lo, lam_hi)`.
_FIT_CASES = ((1.0, 1.0, 10.0, 200.0), (2.0, 1.0, 10.0, 120.0),
              (1.0, None, 30.0, 200.0), (5.0, 1.0, 10.0, 60.0))


def _fit_pair(pen, k_star, span):
    """Fit `sigma_1**2` by a quadratic in `k` and in `lam`, over the SAME physical points.

    Returns the two relative RMS residuals and the two located vertices, in `lam`.
    """
    ks = k_star + np.linspace(-1.0, 1.0, 9) * span
    lams = ks ** 2
    y = np.array([pen.tension(l) for l in lams]) ** 2
    rng = float(y.max() - y.min())
    ck, cl = np.polyfit(ks, y, 2), np.polyfit(lams, y, 2)
    res = (float(np.sqrt(np.mean((np.polyval(ck, ks) - y) ** 2)) / rng),
           float(np.sqrt(np.mean((np.polyval(cl, lams) - y) ** 2)) / rng))
    vtx = (float((-0.5 * ck[1] / ck[0]) ** 2), float(-0.5 * cl[1] / cl[0]))
    return res, vtx, (ks, lams, y, ck, cl)


def fig_refinevar():
    """The scan wants `k`; the fit the refiner makes is better in `lam`."""
    from rect_suite import PHI, pencil_for
    from chladni.reference import rect_spectrum

    res_k, res_l, err_k, err_l, shown = [], [], [], [], None
    for L, H, lo, hi in _FIT_CASES:
        H = 1.0 / PHI if H is None else H
        pen = pencil_for(L, H, lo, hi)
        truth = np.unique(np.asarray(rect_spectrum(lo, hi, L, H), dtype=float))
        kt = np.sqrt(truth)
        for j, t in enumerate(truth):
            others = np.delete(kt, j)
            if not others.size:
                continue
            d = float(np.min(np.abs(others - kt[j])))
            if d < 0.2:                      # too close for one level to own the sample span
                continue
            res, vtx, data = _fit_pair(pen, kt[j], 0.15 * d)
            res_k.append(res[0]); res_l.append(res[1])
            err_k.append(abs(vtx[0] - t) / t); err_l.append(abs(vtx[1] - t) / t)
            if shown is None and L == 1.0 and H == 1.0:
                shown = (t, data, res)
    res_k, res_l = np.array(res_k), np.array(res_l)
    err_k, err_l = np.array(err_k), np.array(err_l)
    wins = int(np.sum(res_l < res_k))
    print(f"  lam fits better in {wins} of {len(res_k)} levels; "
          f"median residual ratio {np.median(res_k/res_l):.2f}, "
          f"median vertex-error ratio {np.median(err_k/err_l):.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))

    # --- (a) the two fits at one level, each in its own variable, same physical points ---
    lam_star, (ks, lams, y, ck, cl), res = shown
    a = ax[0]
    u_k = (ks - ks.mean()) / (ks[-1] - ks[0])          # a shared, dimensionless abscissa
    dense = np.linspace(ks[0], ks[-1], 300)
    u_d = (dense - ks.mean()) / (ks[-1] - ks[0])
    rng = float(y.max() - y.min())
    a.axhline(0, color="k", lw=.7)
    a.plot(u_k, (np.polyval(ck, ks) - y) / rng, "-o", ms=4.5, lw=1.1, color=C_LIVE,
           label=rf"quadratic in $k$   ({res[0]:.1e})")
    a.plot(u_k, (np.polyval(cl, lams) - y) / rng, "--s", ms=4, lw=1.1, color=C_MODEL,
           mfc="none", label=rf"quadratic in $\lambda$   ({res[1]:.1e})")
    a.set_xlabel("sample position within the span")
    a.set_ylabel("fit residual, relative to the range")
    a.set_title(rf"(a) one level, $\lambda^*={lam_star:.2f}$")
    a.legend(frameon=False, fontsize=7.5, loc="lower center")

    # --- (b) every level ---
    a = ax[1]
    lim = np.array([0.5 * min(res_k.min(), res_l.min()), 2.0 * max(res_k.max(), res_l.max())])
    a.fill_between(lim, lim, lim[1] * np.ones(2), color=C_MODEL, alpha=.09, lw=0)
    a.loglog(lim, lim, color="k", lw=.8)
    a.loglog(res_l, res_k, "o", ms=4.5, color=C_ACCENT, mfc="none", mew=1.2)
    a.set_xlim(*lim); a.set_ylim(*lim)
    a.set_xlabel(r"residual, quadratic in $\lambda$")
    a.set_ylabel(r"residual, quadratic in $k$")
    a.set_title(rf"(b) $\lambda$ wins in {wins} of {len(res_k)} levels")
    a.text(.06, .90, r"$\lambda$ better", transform=a.transAxes, fontsize=8, color=C_MODEL)
    _save(fig, "refinevar")


# =================================================================================================
# Figure: the cone, and its two independent parameters
# =================================================================================================

#: Fourier-Bessel orders at the reentrant corner. The floor moves seven orders across these;
#: the wall slope moves 0.4%.
_FB_ORDERS = (20, 40, 60)


def _l_shape_pencil(n_fb):
    from chladni.geometry import L_shape
    from chladni.helmholtz import FourierBessel
    from chladni.mps import MPSPencil

    dom = L_shape()
    corner = next(c for c in dom.corners if c.is_reentrant)
    return MPSPencil.from_basis(
        FourierBessel.at_corner(corner, n=n_fb, domain=dom),
        dom.bdry_quad(1200),
        dom.interior_pts(400, rng=np.random.default_rng(0)))


def fig_cone():
    """`eps` belongs to the basis and `c` to the domain, and only `eps` moves with the basis."""
    from chladni import reference as ref

    k0 = float(np.sqrt(ref.L_shape_eigs(1)[0]))
    d = np.logspace(-14, -1, 200)
    fits, curves = {}, {}
    for n in _FB_ORDERS:
        pen = _l_shape_pencil(n)
        eps = float(pen.tension(k0 ** 2))
        c = float(np.median([pen.tension((k0 + x) ** 2) / x for x in (0.02, 0.03, 0.04)]))
        fits[n] = (eps, c)
        curves[n] = np.array([pen.tension((k0 + x) ** 2) for x in d])
        print(f"  {n:3d} FB terms:  eps = {eps:.3e}   c = {c:.4f}   eps/c = {eps/c:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))

    a = ax[0]
    for i, n in enumerate(_FB_ORDERS):
        eps, c = fits[n]
        col = plt.get_cmap("viridis")(0.12 + 0.36 * i)
        a.loglog(d, curves[n], "o", ms=2.4, color=col, alpha=.8, label=f"{n} terms")
        a.loglog(d, np.sqrt(eps ** 2 + (c * d) ** 2), "-", color=col, lw=1.1, alpha=.9)
        a.axhline(eps, color=col, ls=":", lw=.8)
        a.axvline(eps / c, color=col, ls=":", lw=.8)
    a.set_xlabel(r"$|k - k^*|$")
    a.set_ylabel(r"$\sigma_1$")
    a.set_title(r"$\sqrt{\varepsilon^2 + c^2(k-k^*)^2}$, three bases")
    a.legend(frameon=False, loc="lower right", fontsize=7.5)

    a = ax[1]
    ns = np.array(_FB_ORDERS, dtype=float)
    eps = np.array([fits[n][0] for n in _FB_ORDERS])
    cs = np.array([fits[n][1] for n in _FB_ORDERS])
    a.semilogy(ns, eps, "-o", ms=5, color=C_ACCENT, label=r"floor $\varepsilon$")
    a.set_xlabel("Fourier--Bessel terms at the reentrant corner")
    a.set_ylabel(r"$\varepsilon$")
    a.set_xticks(ns)
    tw = a.twinx()
    tw.plot(ns, cs, "-s", ms=5, color=C_MODEL, mfc="none", mew=1.4)
    tw.set_ylabel(r"wall slope $c$", color=C_MODEL)
    tw.tick_params(axis="y", colors=C_MODEL)
    tw.set_ylim(0, 2.0)
    tw.spines["right"].set_visible(True)
    tw.spines["right"].set_color(C_MODEL)
    a.set_title(rf"$\varepsilon$ falls {np.log10(eps[0]/eps[-1]):.0f} orders; "
                rf"$c$ moves {100*abs(cs[0]-cs[-1])/cs[0]:.1f} percent")
    _save(fig, "cone")


FIGURES = {"ladder": fig_ladder, "exclusion": fig_exclusion,
           "whyk": fig_whyk, "gate": fig_gate, "straddle": fig_straddle,
           "refinevar": fig_refinevar, "cone": fig_cone}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()
    for name in (args.only or sorted(FIGURES)):
        print(f"{name}:")
        FIGURES[name]()


if __name__ == "__main__":
    main()
