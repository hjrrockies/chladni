"""Does the single-spectrum knee generalize? (2026-10-27)

Compares, at the Weyl estimates of levels 1, 8 and 15, three ranks of the planner's own pencils on
several domains and rungs: the threshold rank (shipped), the pair-test stable rank (singular values
that move < 1e-2 relative under lam -> lam (1 + 1e-8)), and the knee of `notes/stable_rank_free.py`
(below 1e-8 sigma_1, first index whose mean decay over the next 4 falls under 1.25x). Also where the
tail starts: sigma/sigma_1 at the stable rank, and the head's decay just above it.
"""
import numpy as np
import scipy.linalg as la
from chladni import asymp
from chladni.geometry import L_shape, Polygon, rect
from chladni.plan import plan_for

prec = 1e-10
doms = [("apex (0.1,0.25)", Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir"), (0, 1, 2, 3)),
        ("flat (0.5,0.12)", Polygon(np.array([0, 1, 0.5 + 0.12j]), bc="dir"), (1, 3)),
        ("L-shape", L_shape(), (0, 1, 3)),
        ("rect 1.6x0.4", rect(1.6, 0.4), (4,)),
        ("pentagon", Polygon(np.exp(2j * np.pi * np.arange(5) / 5) * (1 + 0.15 * np.cos(np.arange(5))), bc="dir"), (0, 1))]


def knee(v, cap, K=4, RHO=1.25, FLOOR=1e-8):
    x = np.log(v / v[0])
    dec = (x[:-K] - x[K:]) / K
    bad = (dec < np.log(RHO)) & (x[:-K] < np.log(FLOOR))
    return min(int(np.argmax(bad)) if bad.any() else len(x), cap)


for name, dom, rungs in doms:
    for rung in rungs:
        try:
            p = plan_for(dom, 15, prec, rung)
            if p.n_cols == 0:
                continue
            s = p.solver(dom)
        except Exception as e:  # noqa: BLE001
            print(f"{name} rung {rung}: {e}"[:120]); continue
        row = []
        for j in (1, 8, 15):
            lam = float(asymp.weyl_est_poly(j, dom))
            a = la.svdvals(np.vstack(s.pencil.pencil(lam)))
            b = la.svdvals(np.vstack(s.pencil.pencil(lam * (1 + 1e-8))))
            thr = int(s.rank_diagnostics(lam)[0])
            moved = np.abs(a - b) / a > 1e-2
            st = min(int(np.argmax(moved)) if moved.any() else len(a), thr)
            kn = knee(a, thr)
            tail = a[st] / a[0] if st < len(a) else float("nan")
            row.append(f"l{j}: thr {thr} pair {st} knee {kn} (tail at {tail:.0e})")
        print(f"{name:16s} rung {rung} ({p.n_cols} cols): " + " | ".join(row), flush=True)
