"""The count audit's threshold, re-calibrated on every reference spectrum the repo has.  (2026-10-11)

`notes/count_audit.md` set `short` at 0.1 from the douse benchmark (41 shapes, 6-26 eigenvalues
each). The L-shape's own table then gave a COMPLETE spectrum scoring 0.13-0.14 when cut off at 16
or 17 levels: Weyl's remainder runs +0.5 .. +0.9 over several consecutive cuts (lam 78-97) before
a double swings it back. A window that ends inside such a run looks short. This measures how far
that goes on everything with known truth -- tables and closed forms, to depth 100 where the
closed forms allow:

    clean   every complete prefix (cut off after each level), with and without a mid-gap top
    drops   each level removed in turn, with `h` levels of headroom above it

    .venv/bin/python notes/count_audit_wide.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.simplefilter("ignore")

from count_audit import deficits  # noqa: E402

DEPTH = 100


def spectra():
    from chladni import geometry as G
    from chladni import reference as R

    def table(fn, *a):
        for k in range(200, 0, -1):
            try:
                return np.asarray(fn(k, *a), float)
            except ValueError:
                continue

    phi = (1 + 5 ** 0.5) / 2
    yield "L_shape", G.L_shape(), table(R.L_shape_eigs)
    yield "H_shape", G.H_shape(), table(R.H_shape_eigs)
    yield "chevron", G.chevron(), table(R.chevron_eigs)
    yield "GWW1", G.GWW1(), table(R.gww_eigs)
    yield "iso_tri", G.iso_tri(), table(R.iso_tri_eigs)
    for n in (5, 6, 8):
        try:
            yield f"ngon{n}", G.reg_ngon(n), table(R.reg_ngon_eigs, n)
        except Exception:  # noqa: BLE001 -- a table that does not cover this N
            pass
    for name, H in (("square", 1.0), ("rect_phi", 1 / phi), ("rect_sqrt8/3", (3 / 8) ** 0.5),
                    ("rect_2", 0.5), ("rect_5", 0.2)):
        yield name, G.rect(1.0, H), np.asarray(R.rect_eigs(DEPTH, 1.0, H), float)
    yield "eq_tri", G.eq_tri(), np.asarray(R.eq_tri_eigs(DEPTH), float)
    yield "iso_right_tri", G.iso_right_tri(), np.asarray(R.iso_right_tri_eigs(DEPTH), float)
    yield "disk", G.disk(), np.asarray(R.disk_eigs(DEPTH), float)
    for th, label in ((np.pi / 2, "sector_90"), (3 * np.pi / 2, "sector_270")):
        yield label, G.disk_sector(theta=th), np.asarray(R.sector_eigs(DEPTH, omega=th), float)


def levels(e):
    lv = [e[0]]
    for x in e[1:]:
        if abs(x - lv[-1]) > 1e-6 * x:
            lv.append(x)
    return np.array(lv)


def main():
    clean, clean_top, by_depth = [], [], {}
    drops = {h: [] for h in (2, 3, 4, 5)}
    worst = []
    for name, dom, e in spectra():
        e = np.sort(e)
        lv = levels(e)
        for n in range(6, len(lv)):                 # cut off after level n (index n-1)
            sub = e[e <= lv[n - 1] * (1 + 1e-9)]
            top = 0.5 * (lv[n - 1] + lv[n])
            m, mt = deficits(sub, dom).mean(), deficits(sub, dom, top).mean()
            clean.append(m)
            clean_top.append(mt)
            by_depth.setdefault(min(n // 25, 3), []).append(mt)
            worst.append((mt, m, name, n))
            for h in drops:
                if n - h < 1:
                    continue
                for i in range(n - h):
                    d = sub[np.abs(sub - lv[i]) > 1e-9 * lv[i]]
                    drops[h].append(deficits(d, dom, top).mean())
    clean, clean_top = np.array(clean), np.array(clean_top)
    print(f"complete prefixes: {len(clean)} (18 shapes)")
    print(f"  mean, no top:      max {clean.max():.3f}  99.9th {np.percentile(clean, 99.9):.3f}  "
          f"min {clean.min():.3f}")
    print(f"  mean, mid-gap top: max {clean_top.max():.3f}  99.9th "
          f"{np.percentile(clean_top, 99.9):.3f}  min {clean_top.min():.3f}")
    for k in sorted(by_depth):
        v = np.array(by_depth[k])
        print(f"    depth {25 * k:3d}-{25 * k + 24:3d}: n={len(v):4d}  max {v.max():.3f}  min {v.min():.3f}")
    print("  highest, with top:")
    for mt, m, name, n in sorted(worst, reverse=True)[:8]:
        print(f"    {name:14s} n={n:3d}  with top {mt:.3f}  no top {m:.3f}")
    print("\ndrops caught (mean with a mid-gap top), by threshold and headroom:")
    for th in (0.1, 0.15, 0.2, 0.25, 0.3):
        fa = np.mean(clean_top > th)
        cells = "  ".join(f"h{h}: {np.mean(np.array(v) > th):6.1%}" for h, v in drops.items())
        print(f"  threshold {th:4.2f}: false alarms {fa:6.2%}   {cells}")


if __name__ == "__main__":
    main()
