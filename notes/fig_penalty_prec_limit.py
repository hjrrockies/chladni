"""Why `reg="tikhonov_qr"` at its derived `rtol` cannot serve `prec = 1e-14`.

`rtol = sqrt(prec * eps)` was derived for a TRUNCATION, whose floor is set by the basis and
barely moves with the cutoff. A PENALTY's floor IS its `tau = rtol * sqrt(n)`, so it scales as
`sqrt(prec)` while the requirement scales as `prec`: the ratio grows like `1/sqrt(prec)` and
must cross 1. It does, between 1e-12 and 1e-14.

OBJECTIVE-LEVEL. The cluster-total and decline grids have never been run under the shipped
option (`TODO.md` section 4, gate 0); this shows the floor against what was asked for, not a
grid result.

Run from the repo root:  .venv/bin/python notes/fig_penalty_prec_limit.py
"""
import sys
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hard_roster as hr
import rect_suite as rs
from chladni.mps import MPSPencil

EPS = np.finfo(float).eps
PRECS = np.array([1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
C_PEN, C_THR = "#1e88e5", "#c62828"


def domains():
    c = rs.family_A()[0]
    yield "rect(1, 1/phi)", c.pencil, np.unique(rs.complete_truth(c))
    d = hr.ROSTER["disk"]()
    yield "disk", d.pencil, np.unique(np.asarray(d.truth, dtype=float))
    e = hr.ROSTER["eq_tri"]()
    yield "eq_tri", e.pencil, np.unique(np.asarray(e.truth, dtype=float))


def floors():
    """Worst `sigma_1` at the true levels, under the SHIPPED option at each `prec`."""
    out = {}
    for name, p, truth in domains():
        lv = truth[np.linspace(0, truth.size - 1, min(6, truth.size)).astype(int)]
        pen = [max(float(MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=P).tension(float(l)))
                   for l in lv) for P in PRECS]
        out[name] = (np.array(pen), max(float(p.tension(float(l))) for l in lv), p.n_cols)
    return out


def cluster(prec=1e-14, band=1.0):
    """The 50*pi^2 triple dialled to `band * prec`, sampled under both modes."""
    (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={50: 3})
    p = case.pencil
    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec)
    t = np.sort(np.asarray(case.truth, dtype=float))
    xs = t.mean() + np.linspace(-3.0, 3.0, 241) * (t[-1] - t[0])
    return (xs - t.mean()) / t.mean(), np.array([float(p.tension(float(x))) for x in xs]), \
        np.array([float(q.tension(float(x))) for x in xs]), (t - t.mean()) / t.mean(), \
        np.diff(t)[0] / t[0]


def main():
    fl = floors()
    rel, thr, pen, members, sep = cluster()

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.suptitle('Why `reg="tikhonov_qr"` at its derived rtol stops short of prec = 1e-14\n'
                 "the rtol rule was derived for a truncation, whose floor the cutoff barely "
                 "moves; a penalty's floor IS its tau", fontsize=13, y=0.99)

    a = ax[0]
    a.fill_between([3e-15, 3e-5], [3e-15, 3e-5], 1e-3, color="#ffcdd2", alpha=0.5, zorder=0)
    a.plot([3e-15, 3e-5], [3e-15, 3e-5], color="#424242", lw=1.4, ls="--", zorder=2)
    for (name, (pen_fl, thr_fl, n)), mk in zip(fl.items(), "os^"):
        a.loglog(PRECS, pen_fl, mk + "-", color=C_PEN, lw=1.6, ms=6, label=f"penalty, {name}")
        a.axhline(thr_fl, color=C_THR, lw=1.0, ls=":", zorder=1)
    a.loglog([], [], "-", color=C_THR, ls=":", label="threshold floors (basis-set)")
    a.set_xlabel("requested prec"), a.set_ylabel("worst floor over the true levels")
    a.set_title("A.  The floor against what was asked for. Above the dashed\n"
                "line the floor exceeds prec: the precision cannot be delivered.",
                fontsize=10.5, loc="left")
    a.annotate("cannot deliver", xy=(3e-13, 2e-10), fontsize=11, color="#b71c1c", weight="bold")
    a.set_xlim(3e-15, 3e-5), a.set_ylim(3e-16, 1e-8)
    a.invert_xaxis(), a.grid(alpha=0.25, which="both", lw=0.5), a.legend(fontsize=8.5, loc="lower right")

    a = ax[1]
    for (name, (pen_fl, _, n)), mk in zip(fl.items(), "os^"):
        a.loglog(PRECS, pen_fl / PRECS, mk + "-", color=C_PEN, lw=1.6, ms=6, label=name)
    law = 1.45 * np.sqrt(fl["rect(1, 1/phi)"][2] * EPS / PRECS)
    a.loglog(PRECS, law, "-", color="#2e7d32", lw=1.5, alpha=0.8,
             label=r"$1.45\,\sqrt{n\,\varepsilon/\mathrm{prec}}$  (rect)")
    a.axhline(1.0, color="#424242", lw=1.4, ls="--")
    a.set_xlabel("requested prec"), a.set_ylabel("floor / prec")
    a.set_title("B.  Why it must cross. The floor is tau ~ sqrt(prec*eps), so\n"
                "floor/prec grows like 1/sqrt(prec) and passes 1.", fontsize=10.5, loc="left")
    a.annotate("floor = prec", xy=(2e-6, 1.0), xytext=(2e-6, 1.8), fontsize=9, color="#424242")
    a.annotate("crosses between\n1e-12 and 1e-14", xy=(2.2e-13, 1.0), xytext=(2e-10, 0.35),
               fontsize=9.5, color="#424242", ha="center",
               arrowprops=dict(arrowstyle="->", color="#424242", lw=1.1))
    a.set_xlim(3e-15, 3e-5), a.set_ylim(5e-5, 8.0), a.invert_xaxis()
    a.grid(alpha=0.25, which="both", lw=0.5), a.legend(fontsize=8.5, loc="lower right")

    a = ax[2]
    a.semilogy(rel, thr, color=C_THR, lw=1.5, label="threshold")
    a.semilogy(rel, pen, color=C_PEN, lw=1.5, label='penalty, prec=1e-14')
    for m in members:
        a.axvline(m, color="#9e9e9e", lw=0.8, ls=":")
    a.axhline(thr.min(), color=C_THR, lw=0.9, ls="--")
    a.axhline(pen.min(), color=C_PEN, lw=0.9, ls="--")
    a.annotate("", xy=(members[0], 1.5e-13), xytext=(members[1], 1.5e-13),
               arrowprops=dict(arrowstyle="<->", color="#424242", lw=1.2))
    a.annotate(f"members {sep:.0e} apart\n(one merge licence)", xy=(-3.2e-14, 1.6e-13),
               fontsize=9, ha="center", color="#424242")
    a.set_xlabel(r"$(\lambda - \lambda_c)\,/\,\lambda_c$"), a.set_ylabel(r"$\sigma_1$")
    a.set_title("C.  The 50$\\pi^2$ triple at prec = 1e-14, one licence apart. The\n"
                "penalty's floor is WIDER than the spacing it must resolve.",
                fontsize=10.5, loc="left")
    a.annotate(f"penalty floor {pen.min():.1e}\nWIDER than the {sep:.0e} spacing\n"
               "barrier/floor 2.1: unresolvable",
               xy=(2.1e-14, pen.min()), xytext=(2.4e-14, 6.5e-14), fontsize=8.5, color=C_PEN,
               ha="left", arrowprops=dict(arrowstyle="->", color=C_PEN, lw=1.1))
    a.annotate(f"threshold floor {thr.min():.1e}\nbarrier/floor 24: resolvable",
               xy=(-2.0e-14, thr.min()), xytext=(-6.2e-14, 3.0e-15), fontsize=8.5, color=C_THR,
               ha="left", arrowprops=dict(arrowstyle="->", color=C_THR, lw=1.1))
    a.set_ylim(1.2e-15, 1.1e-12)
    a.grid(alpha=0.25, which="both", lw=0.5)
    a.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.95)

    fig.text(0.5, 0.005,
             "Objective-level: the floor against the precision requested, NOT a grid result -- the grids "
             "have never been run under the shipped option (TODO section 4, gate 0).  At prec = 1e-14 "
             "floor/prec is 2.4 on the rect and 2.1 on eq_tri; the disk lands at 1.0, marginal rather "
             "than failing, because its basis is comfortable and its floor sits below tau.",
             ha="center", fontsize=8.5, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig("notes/fig_penalty_prec_limit.png", dpi=150)
    print("notes/fig_penalty_prec_limit.png")


if __name__ == "__main__":
    main()
