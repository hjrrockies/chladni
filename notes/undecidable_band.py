"""Regenerate notes/undecidable_band.png: the band where a cluster's GROUPING has no answer.

    python notes/undecidable_band.py

Working script, not tracked. The figure states one regime and one fix, both as consequences of
two lengths rather than of chosen thresholds:

    the MERGE LICENCE   prec*lam        -- inside it, eigenvalues may be reported as one level
    the WIDE REACH      f*prec*lam      -- the flanks a count needs, f = WIDE_FACTOR

A cluster whose SPREAD falls between them cannot be one level, and either way of splitting it
is legal, so the grouping is arbitrary -- and then so are the individual counts. A count taken
at the middle member has to look far enough to see its partner and therefore also sees the
member belonging to the other group, exactly equidistant, so no radius separates them.

What is NOT arbitrary is the total. `v1_solve` cross-checks the sum of the individual counts
against one count over a bracket spanning the neighbourhood; disagreement means the grouping is
arbitrary, so the neighbourhood collapses to a single level carrying the spanning count. The
total survives, which is what an eigenvalue INDEX depends on -- a wrong total is how a solver
comes to report an accurate but non-consecutive spectrum. The price is a level wider than
`prec` asked for, declared rather than hidden.

Everything plotted is measured on the multiplicity-3 level at 50*pi^2 of rect(1+delta, 1),
perturbed so the minimum adjacent gap lands at a chosen multiple of `prec`. The "without the
cross-check" series is the real ablation, `solve(..., collapse=False)`, not a recorded number.
"""
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import bracket_score as bs  # noqa: E402
import level_score as ls  # noqa: E402
import rect_suite as rs  # noqa: E402
import v1_multiplicity as M  # noqa: E402
from v1_solve import solve  # noqa: E402

PREC = 1e-8
BANDS = (0.2, 0.35, 0.5, 0.8, 1.0, 1.3, 2.0, 3.0)
REACH = M.WIDE_FACTOR

rows = []
for band in BANDS:
    cases = rs.crossover_cases(PREC, window="cluster", bands=(band,))
    if not cases:
        continue
    case = cases[0]
    ratio = rs.cluster_diameter_ratio(case, PREC)
    obj = bs.objective_for(case, PREC)
    res = solve(obj, prec=PREC)
    off = solve(bs.objective_for(case, PREC), prec=PREC, collapse=False)
    score = ls.score_levels(res.levels, case.truth, PREC)
    rows.append(dict(band=band, ratio=ratio, case=case, res=res, score=score,
                     declared=sum(x.mult for x in res.levels),
                     was=sum(x.mult for x in off.levels),
                     n_true=len(case.truth), amb=res.diagnostics.n_ambiguous,
                     ok=score.ok, verdict=score.verdict))
    r = rows[-1]
    print(f"band {band:4.2f}  spread/licence {ratio:6.3f}  S(prec) levels "
          f"{score.n_levels_true}  levels {len(res.levels)}  total {r['declared']}"
          f"/{r['n_true']}  without the check {r['was']}/{r['n_true']}"
          f"{'  COLLAPSED' if r['amb'] else ''}")

fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.4))

# --- A: the total, with and without the cross-check --------------------------------------------
a = ax[0]
a.axvspan(0.12, 1.0, color="#d8ece0", zorder=0)
a.axvspan(1.0, REACH, color="#f6d9d4", zorder=0)
a.axvspan(REACH, 12.0, color="#dae4f2", zorder=0)
a.axvline(1.0, color="#2f7d4f", lw=1.6)
a.axvline(REACH, color="#2f5f9e", lw=1.6)
a.axhline(1.0, color="#666", lw=0.9, ls=":")

for r in rows:
    was = r["was"] / r["n_true"]
    now = r["declared"] / r["n_true"]
    if abs(was - now) > 1e-9:
        a.annotate("", xy=(r["ratio"], now), xytext=(r["ratio"], was),
                   arrowprops=dict(arrowstyle="->", color="#888", lw=1.2,
                                   shrinkA=3, shrinkB=5))
        a.plot(r["ratio"], was, marker="X", ms=11, color="#c0392b", mec="white", mew=1.1,
               zorder=3)
        dx = -14 if 1.2 < r["ratio"] < 1.9 else 14
        a.annotate(f"{r['was']}/{r['n_true']}", (r["ratio"], was), textcoords="offset points",
                   xytext=(dx, 0), ha="right" if dx < 0 else "left", va="center",
                   fontsize=8.5, color="#a03024")
    a.plot(r["ratio"], now, marker="o", ms=10, color="#1a7f37", mec="white", mew=1.2, zorder=4)

a.set_xscale("log")
a.set_xlim(0.13, 11.0)
a.set_ylim(0.72, 2.30)
a.set_xlabel("cluster spread / merge licence     $d\\,/\\,(\\mathrm{prec}\\cdot\\lambda)$")
a.set_ylabel("eigenvalues declared / eigenvalues present")
a.set_title("A.  the total is what an index depends on, and it is restored", fontsize=11)
a.text(0.34, 2.19, "ONE LEVEL\nreach spans it", ha="center", fontsize=9, color="#1d5c38")
a.text(np.sqrt(REACH), 2.19, "GROUPING\nARBITRARY", ha="center", fontsize=9.5,
       color="#a03024", fontweight="bold")
a.text(5.2, 2.19, "SEPARATE LEVELS\neach counted alone", ha="center", fontsize=9,
       color="#264d7e")
a.text(1.0, 0.755, " licence", fontsize=8.5, color="#2f7d4f", rotation=90, va="bottom")
a.text(REACH, 0.755, f" reach = {REACH:g}$\\times$", fontsize=8.5, color="#2f5f9e",
       rotation=90, va="bottom")
a.legend(handles=[Line2D([], [], marker="X", ls="", color="#c0392b", mec="white",
                         label="without the cross-check"),
                  Line2D([], [], marker="o", ls="", color="#1a7f37", mec="white",
                         label="with it: total preserved")],
         loc="lower right", fontsize=8.5, framealpha=0.95)

# --- B: the mechanism, as a ruler -------------------------------------------------------------
#
# One row per case, in units of the licence: where the three members actually sit, against the
# licence a merge is allowed within and the reach a multiplicity read needs. The failure is
# visible as a geometric fact -- in the middle rows the REACH covers a member that the LICENCE
# excludes, so the read counts an eigenvalue belonging to a different level.
b = ax[1]
show = [r for r in rows if r["band"] in (0.2, 0.35, 0.5, 0.8, 1.0, 1.3, 3.0)]
for i, r in enumerate(show):
    y = len(show) - 1 - i
    case, res = r["case"], r["res"]
    t = np.unique(np.asarray(case.truth, dtype=float))
    lic = PREC * abs(t[-1])
    L0 = min(res.levels, key=lambda x: abs(x.lam - float(np.mean(t))))
    origin = L0.lam
    colour = "#c0392b" if r["amb"] else "#1a7f37"

    # the reach the read actually used, and the licence around the same origin
    if L0.wide_bracket:
        lo, hi = ((L0.wide_bracket[0] - origin) / lic, (L0.wide_bracket[1] - origin) / lic)
        # Clipped for drawing, so a wide collapsed bracket does not run through the label
        # column; a caret marks a bar that continues past the cut.
        cut = 4.6
        b.plot([max(lo, -cut), min(hi, cut)], [y, y], color="#2f5f9e", lw=2.2,
               solid_capstyle="butt", zorder=2)
        for edge, inside in ((max(lo, -cut), lo >= -cut), (min(hi, cut), hi <= cut)):
            b.plot([edge], [y], marker="|" if inside else
                   ("<" if edge < 0 else ">"), ms=10 if inside else 5.5, ls="",
                   color="#2f5f9e", mew=2.0, zorder=3)
    b.plot([-0.5, 0.5], [y + 0.22, y + 0.22], color="#2f7d4f", lw=6, alpha=0.35,
           solid_capstyle="butt", zorder=1)

    b.plot((t - origin) / lic, np.full(t.size, y), marker="o", ms=7.5, ls="",
           color=colour, mec="white", mew=1.1, zorder=4)
    b.text(-7.6, y, f"{r['ratio']:.1f}", ha="right", va="center", fontsize=9)
    if r["amb"]:
        b.axhspan(y - 0.40, y + 0.40, xmin=0.0, xmax=0.79, color="#f6d9d4", zorder=0)
    b.text(5.6, y, f"{r['declared']}/{r['n_true']}", ha="left", va="center", fontsize=9,
           color="#1a7f37")
    if r["amb"]:
        b.text(6.7, y, "collapsed", ha="left", va="center", fontsize=7.8,
               color="#a03024", style="italic")

# name the mechanism once, on the clearest undecidable row
row_16 = next((len(show) - 1 - i for i, r in enumerate(show) if abs(r["ratio"] - 1.6) < 0.05),
              None)
if row_16 is not None:
    b.annotate("the reach spans all three;\nthe licence holds one",
               xy=(-2.3, row_16), xytext=(-4.9, row_16), va="center",
               fontsize=8.2, color="#a03024", ha="center",
               arrowprops=dict(arrowstyle="->", color="#a03024", lw=1.0,
                               shrinkA=0, shrinkB=2))

b.set_yticks([])
b.set_xlim(-9.6, 9.6)
b.set_ylim(-1.85, len(show) - 0.30)
b.axvline(0.0, color="#999", lw=0.8, ls=":")
b.set_xlabel("distance from the located root, in units of the licence "
             r"$(\lambda - \hat\lambda)/(\mathrm{prec}\cdot\lambda)$")
b.set_title("B.  why: in the shaded rows the reach covers a member the licence excludes",
            fontsize=11)
b.text(-7.6, len(show) - 0.58, r"$d\,/\,\mathrm{lic}$", ha="right", fontsize=9)
b.text(5.6, len(show) - 0.58, "declared", ha="left", fontsize=9)
b.legend(handles=[Line2D([], [], marker="o", ls="", color="#1a7f37", mec="white",
                         label="member, grouping decided"),
                  Line2D([], [], marker="o", ls="", color="#c0392b", mec="white",
                         label="member, grouping arbitrary"),
                  Line2D([], [], color="#2f7d4f", lw=6, alpha=0.35,
                         label=r"merge licence, $\pm\,0.5$"),
                  Line2D([], [], color="#2f5f9e", lw=2.2, label="reach used by (C2)")],
         loc="lower center", fontsize=8.2, framealpha=0.95, ncol=2,
         bbox_to_anchor=(0.5, -0.005))

fig.suptitle("A cluster whose spread falls between the merge licence and the counting reach: "
             "the grouping is arbitrary, the total is not", fontsize=12.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("notes/undecidable_band.png", dpi=170)
print("\nwrote notes/undecidable_band.png")
