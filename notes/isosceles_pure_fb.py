"""Pure Fourier-Bessel on isosceles triangles: lam_1 across the apex angle, planner bypassed.

The triangle has its apex `omega` at the origin and unit legs, `(0, 1, e^{i omega})`, Dirichlet.
Each of the three corners carries a flat 100 Fourier-Bessel terms (`Plan(corner_orders=(100,)*3)`
realized directly, so the planner's sizing and its shape refusals are skipped), collocation at
the plan defaults (3x boundary, 1x interior), `prec = 1e-14`. Only the first eigenvalue is read.

What is recorded per angle is what measures expected performance without a reference value:
the Moler-Payne bound (relative, `mp_bound / lam`) and the tension at the minimum (`sigma`). The
window is `[FK/2, 1.05 j0^2 / r_in^2]`, which holds lam_1 for every triangle (Faber-Krahn below,
the inradius bound above).

Two readings per angle. SOLVER: `search_interval` on the window, its lowest level -- or its
decline, which is itself the result (near both ends the search refuses its own premises before
it locates anything). DIRECT: the tension scanned on the window, the lowest deep local minimum
refined by golden section, and Moler-Payne on that minimizer, computed as `certify_levels` does.
DIRECT exists so the curves continue past where the solver declines; it is a diagnostic of the
basis, not a solver path, and nothing it returns is certified to be lam_1 rather than a deeper
spurious well.

    python notes/isosceles_pure_fb.py            # sweep, writes isosceles_pure_fb.json
    python notes/isosceles_pure_fb.py --plot     # figures from the JSON
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from chladni.bounds import faber_krahn, inradius_upper
from chladni.geometry.domain import Polygon
from chladni.plan import Plan

PREC = 1e-14
N_FB = 100
HERE = Path(__file__).parent
OUT = HERE / "isosceles_pure_fb.json"


def omegas(n_half=36, lo=1e-2):
    """Log-spaced toward both ends: `lo .. pi/2` geometric, mirrored as `pi - (lo .. pi/2)`."""
    left = np.geomspace(lo, np.pi / 2, n_half)
    return np.concatenate([left, (np.pi - left[:-1])[::-1]])


def triangle(omega):
    return Polygon([0.0, 1.0, np.exp(1j * omega)], prec=PREC)


N_SCAN = 400


def direct(solver, dom, a, b):
    """The lowest deep tension minimum on `[a, b]` and its Moler-Payne bound (see module doc)."""
    from chladni.eigensolver import lam_to_k
    from chladni.integrals import cauchy, certify

    grid = np.linspace(a, b, N_SCAN)
    t = solver.tension(grid)
    scan = {"scan_lam": grid.tolist(), "scan_sigma": t.tolist()}
    i = np.flatnonzero((t[1:-1] < t[:-2]) & (t[1:-1] < t[2:])) + 1
    if i.size == 0:
        return {"direct_note": "no local minimum on the scan grid", **scan}
    # The lowest DEEP minimum is lam_1's candidate; with none deep (a flat tension, as at the
    # ends) the global one is taken and `direct_depth` says how little it stands out.
    deep = i[t[i] < 1e-2 * np.median(t)]
    j = deep[0] if deep.size else i[np.argmin(t[i])]
    h = grid[1] - grid[0]
    lam, sigma = _golden(solver.tension, grid[j] - h, grid[j] + h)
    p = solver._pencil
    fc = p.family_coef(lam, p.coef(lam, 1))
    iq = solver._integral_quad(1.05 * lam, PREC)
    cd = cauchy.cauchy_data(p.basis, fc, lam, iq.quad)
    k = lam_to_k(lam)

    def evaluate(z):
        V = p.basis.values(k, z)
        return V @ fc, np.abs(V) @ np.abs(fc)

    mp = certify.moler_payne(cd, evaluate, dom, certify.x0_probes(dom, iq.x0))
    rd = solver.rank_diagnostics(lam)
    return {"direct_lam": lam, "direct_sigma": sigma,
            "direct_depth": float(sigma / np.median(t)),
            "direct_mp_bound": float(np.max(mp.abs)), "direct_mp_eps": float(np.max(mp.eps)),
            "direct_cancellation": float(np.max(mp.cancellation)),
            "direct_mp_reason": str(mp.reason), "direct_n_deep": int(deep.size),
            "rank": int(rd.rank) if rd.rank is not None else None, "n_cols": int(len(rd.s)),
            "s_ratio": float(rd.s[-1] / rd.s[0]), **scan}


def _golden(f, a, b, rtol=1e-15):
    """Golden-section minimum of `f` on `[a, b]` to `rtol` relative. Not scipy's bounded Brent,
    whose stopping rule carries a `sqrt(eps)|x|` term and stops near 1e-8 relative."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > rtol * abs(b):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (c, float(fc)) if fc < fd else (d, float(fd))


def run(omega):
    dom = triangle(omega)
    row = {"omega": float(omega), "area": float(dom.area)}
    t0 = time.perf_counter()
    try:
        plan = Plan(prec=PREC, k_max=np.nan, corner_orders=(N_FB,) * len(dom.corners))
        solver = plan.solver(dom)
        a = 0.5 * float(faber_krahn(dom))
        b = 1.05 * float(inradius_upper(dom))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = solver.search_interval(a, b)
        row["warnings"] = sorted({str(w.message)[:160] for w in caught})
        d = res.diagnostics
        row["declined"] = bool(d.declined)
        row["decline_reason"] = str(d.decline_reason or "")
        if res.levels:
            L = min(res.levels, key=lambda L: L.lam)
            row.update(lam=float(L.lam), mult=int(L.mult), sigma=float(L.sigma),
                       delta_res=float(L.delta_res), mp_bound=float(L.mp_bound),
                       mp_eps=float(L.mp_eps), cancellation=float(L.cancellation),
                       bound=float(L.bound), bound_reason=str(L.bound_reason),
                       n_levels=len(res.levels))
    except Exception as e:  # noqa: BLE001 -- a failure at an extreme angle is a data point
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["seconds"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row.update(direct(solver, dom, a, b))
    except Exception as e:  # noqa: BLE001
        row["direct_error"] = f"{type(e).__name__}: {e}"[:300]
    row["direct_seconds"] = time.perf_counter() - t1
    return row


def sweep(ws):
    rows = []
    for w in ws:
        r = run(w)
        rows.append(r)
        g = lambda key: r.get(key, np.nan)  # noqa: E731
        print(f"omega={w:.5f}  solver: lam={g('lam'):.14g} mp/lam={g('mp_bound') / g('lam'):.1e}"
              f" sigma={g('sigma'):.1e} {'DECLINED ' if r.get('declined') else ''}"
              f"{r['seconds']:.0f}s | direct: lam={g('direct_lam'):.14g} "
              f"mp/lam={g('direct_mp_bound') / g('direct_lam'):.1e} "
              f"sigma={g('direct_sigma'):.1e} {r['direct_seconds']:.0f}s "
              f"{r.get('error', '')}{r.get('direct_error', '')}", flush=True)
        OUT.write_text(json.dumps(rows, indent=1))
    return rows


BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"


def _arr(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def plot():
    import matplotlib.pyplot as plt

    rows = json.loads(OUT.read_text())
    w = _arr(rows, "omega")
    x = w / np.pi
    lam = _arr(rows, "direct_lam")
    mp = _arr(rows, "direct_mp_bound") / lam
    sig = _arr(rows, "direct_sigma")
    med = np.array([np.median(r["scan_sigma"]) if "scan_sigma" in r else np.nan for r in rows])
    declined = np.array([bool(r.get("declined")) or "error" in r for r in rows])
    refused = ~np.isfinite(mp)
    ceiling = 1.0                              # where a refused bound is drawn

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, constrained_layout=True)
    a = ax[0]
    a.semilogy(x[~refused], mp[~refused], "o-", ms=4, lw=1.5, color=BLUE,
               label="Moler-Payne bound / $\\lambda_1$")
    a.semilogy(x[refused], np.full(refused.sum(), ceiling), "x", ms=6, color=ORANGE,
               label="Moler-Payne refused ($\\varepsilon \\geq 1$)")
    a.axhline(PREC, color=MUTED, ls=":", lw=1, label="prec")
    a.set_ylabel("relative error bound")
    a.legend(frameon=False, loc="best", fontsize=8)
    a = ax[1]
    a.semilogy(x, sig, "o-", ms=4, lw=1.5, color=BLUE, label="$\\sigma$ at the minimum")
    a.semilogy(x, med, "-", lw=1.5, color=MUTED, label="median $\\sigma$ over the window")
    a.set_ylabel("tension")
    a.set_xlabel("apex angle $\\omega_1/\\pi$")
    a.legend(frameon=False, loc="best", fontsize=8)
    for a in ax:
        for xi in x[declined]:
            a.axvline(xi, color=ORANGE, alpha=0.12, lw=3, zorder=0)
        a.grid(True, which="major", color="#e6e6e3", lw=0.6)
    ax[0].set_title("shaded: the solver declined; points are the direct minimum", fontsize=9,
                    color=MUTED)
    fig.suptitle(f"Isosceles triangle, unit legs: pure FB, {N_FB} terms per corner, "
                 f"prec={PREC:g}")
    fig.savefig(HERE / "isosceles_pure_fb.png", dpi=150)

    # The two ends, against the distance to each end on a log axis.
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharey="row", constrained_layout=True)
    for c, (sel, dist, lab) in enumerate(((x < 0.5, w, "$\\omega_1$"),
                                          (x > 0.5, np.pi - w, "$\\pi - \\omega_1$"))):
        ok = sel & ~refused
        ax[0, c].loglog(dist[ok], mp[ok], "o-", ms=4, lw=1.5, color=BLUE)
        ax[0, c].loglog(dist[sel & refused], np.full((sel & refused).sum(), ceiling), "x",
                        ms=6, color=ORANGE)
        ax[0, c].axhline(PREC, color=MUTED, ls=":", lw=1)
        ax[1, c].loglog(dist[sel], sig[sel], "o-", ms=4, lw=1.5, color=BLUE,
                        label="$\\sigma$ at the minimum")
        ax[1, c].loglog(dist[sel], med[sel], "-", lw=1.5, color=MUTED, label="median $\\sigma$")
        ax[1, c].set_xlabel(lab)
        for a in ax[:, c]:
            for d in dist[sel & declined]:
                a.axvline(d, color=ORANGE, alpha=0.12, lw=3, zorder=0)
            a.grid(True, which="major", color="#e6e6e3", lw=0.6)
    ax[0, 0].set_ylabel("Moler-Payne / $\\lambda_1$")
    ax[1, 0].set_ylabel("tension")
    ax[1, 0].legend(frameon=False)
    fig.suptitle("Approaching the ends (x: refused bound; shaded: solver declined)")
    fig.savefig(HERE / "isosceles_pure_fb_ends.png", dpi=150)

    # Tension curves over the window, a few angles from each end inward.
    picks = [k for k in np.linspace(0, len(rows) - 1, 9).round().astype(int)
             if "scan_sigma" in rows[k]]
    n = len(picks)
    fig, ax = plt.subplots(1, n, figsize=(2.1 * n, 2.8), sharey=True, constrained_layout=True)
    for a, k in zip(np.atleast_1d(ax), picks):
        r = rows[k]
        L = np.array(r["scan_lam"])
        a.semilogy(L / L[0], r["scan_sigma"], lw=1.2, color=BLUE)
        if "lam" in r:
            a.axvline(r["lam"] / L[0], color=AQUA, lw=1, ls="--")
        a.set_title(f"$\\omega_1/\\pi$={r['omega'] / np.pi:.3g}", fontsize=9)
        a.set_xlabel("$\\lambda / (FK/2)$", fontsize=8)
        a.tick_params(labelsize=7)
    np.atleast_1d(ax)[0].set_ylabel("tension $\\sigma_1(\\lambda)$")
    fig.suptitle("Tension over the window (dashed: the solver's $\\lambda_1$, where it returned one)",
                 fontsize=10)
    fig.savefig(HERE / "isosceles_pure_fb_scans.png", dpi=150)

    ok = np.isfinite(mp)
    print(f"{len(rows)} angles; solver declined at {declined.sum()}, Moler-Payne refused at "
          f"{refused.sum()}; bound <= prec at {np.sum(mp[ok] <= PREC)}, <= 1e-8 at "
          f"{np.sum(mp[ok] <= 1e-8)}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    elif "--smoke" in sys.argv:
        sweep([0.05, np.pi / 3, np.pi / 2, np.pi - 0.05])
    else:
        sweep(omegas())
