"""Reference spectra, with multiplicities, for the douse failure shapes in `cases.json`.

`docs/design/douse_failure_cases.md`: most Tier 2 shapes have no stored ground truth, and the
ones that do were produced by `lappy` -- the solver under test. So every reference here is made
by THREE methods sharing as little as possible:

    A    chladni, Fourier-Bessel at every corner, sized per corner like lappy's planner
         (`ceil(1.5 (k r + 2 (k r)^(1/3) + 3) / alpha)`, `r` the corner's farthest vertex)
    B    chladni, the same corners at 1.2x (halved at sharp corners), plus an offset
         fundamental-solution layer -- a different basis, so a basis artefact does not repeat
    C    chladni, as A at 2.0x
    FEM  P2 on two uniform refinements (`fem.py`): counts only, see there

every chladni solve at `prec = 1e-12` with the Moler-Payne bound on every level. A level is
`certified` when two of the three runs agree within their bounds with the same multiplicity and
no count flag, and the FEM count below the mid-gaps either side of it agrees with the consensus
count at the finer mesh (`compare`). Otherwise `provisional`, with the reason. Nothing is
dropped: a level only one run sees is recorded with what each saw. Three runs, not two, because
the trial run met two chladni defects on one case (T1.3: A declined, B returned a simple level
as x8), and two runs can only disagree.

    .venv/bin/python benchmarks/douse/make_references.py [ids...]     # all cases, ~30-60 min

Writes `references.json` (and `references.log`). Requires the `bench` extra (scikit-fem).
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PREC = 1e-12
FEM_REFINE = (6, 7)
SCALE = {"A": 1.5, "B": 1.2, "C": 2.0}


def load_cases():
    return json.load(open(os.path.join(HERE, "cases.json")))["cases"]


def polygon(vertices):
    from chladni.geometry import Polygon
    v = np.asarray(vertices, dtype=float)
    return Polygon(v[:, 0] + 1j * v[:, 1])


def basis_for(dom, which, k):
    """`A` or `B` above. `k` is the window's top wavenumber."""
    from chladni.helmholtz import FourierBessel, Fundamental
    scale = SCALE[which]
    zs = np.array([c.z for c in dom.corners])
    b = None
    for c in dom.corners:
        alpha = np.pi / c.omega
        r = float(np.max(np.abs(zs - c.z)))
        n = int(np.ceil(scale * (k * r + 2 * (k * r) ** (1 / 3) + 3) / alpha))
        if which == "B" and c.omega < np.pi / 2:
            n //= 2          # sharp corners: a different split, the layer covers the rest
        f = FourierBessel.at_corner(c, n=max(n, 3), domain=dom)
        b = f if b is None else b + f
    if which == "B":       # A and C are FB-only, at two sizes; B mixes in sources
        perim = float(np.sum(np.abs(np.diff(np.append(zs, zs[:1])))))
        nsrc = int(np.ceil(scale * 3 * k * perim / (2 * np.pi)))
        for frac in (0.05, 0.02, 0.01):
            try:
                b = b + Fundamental.offset(dom.bdry_quad(nsrc), frac * dom.diameter, domain=dom)
                break
            except Exception:  # noqa: BLE001 -- sources inside a thin domain: move closer
                continue
    return b


def chladni_solve(vertices, which, lo, hi):
    from chladni.helmholtz import PrecisionWarning
    from chladni.mps import MPSEigensolver
    warnings.simplefilter("ignore", PrecisionWarning)
    dom = polygon(vertices)
    b = basis_for(dom, which, np.sqrt(hi))
    n = len(b)
    t0 = time.perf_counter()
    try:
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=PREC)
        res = s.search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001 -- recorded, not fatal
        return dict(which=which, n_cols=n, error=repr(e), wall=time.perf_counter() - t0)
    d = res.diagnostics
    return dict(which=which, n_cols=n, wall=time.perf_counter() - t0,
                declined=bool(d.declined), decline_reason=d.decline_reason,
                levels=[dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                             mp_bound=float(L.mp_bound), delta_res=float(L.delta_res),
                             uncertain=bool(L.count_uncertain or L.grouping_ambiguous),
                             reason=L.uncertain_reason)
                        for L in res.levels])


def window(case, fem):
    """`(lo, hi)`: from below the first eigenvalue to a mid-gap cut past the case's indices.

    FEM eigenvalues are upper bounds, so `lo = fem[0] / 2` is safely below `lam_1`. The cut is
    the widest relative gap among indices K..K+3 (K = requested + 3, at least 8) of the finer FEM
    spectrum, placed at its geometric middle."""
    K = max(8, (case.get("n_eigs_requested") or 4) + 3)
    w = fem
    gaps = [(w[i + 1] / w[i], i) for i in range(K - 1, min(K + 3, len(w) - 1))]
    g, i = max(gaps)
    return 0.5 * float(w[0]), float(np.sqrt(w[i] * w[i + 1])), i + 1


#: Clusters closer than this (relative) are one GROUP for the FEM count: the finer FEM mesh is
#: good to ~1e-4 relative away from reentrant corners and ~1e-3 near them, so it cannot place a
#: cut between levels closer than that.
FEM_RESOLVES = 3e-3


def compare(runs, fems, hi):
    """Cluster the runs' levels by value and score each cluster.

    Levels from any runs chain into one cluster while they lie within `_same`; a run's multiplicity for a cluster is the sum of its levels there,
    so a double returned as two simple levels and as one double count the same. Clusters within
    `FEM_RESOLVES` of each other form a GROUP, and FEM's count between the mid-gap cuts around a
    group is its true count once the mesh resolves the modes.

    Status per cluster:
      certified     two or more runs agree on its multiplicity, unflagged, and the group's
                    consensus count equals FEM's
      fem_resolved  the group's FEM count settles the multiplicity, and one unflagged run agrees
      provisional   anything else, with the reason
    One run's defect -- a decline, a miscount -- therefore cannot make a reference on its own."""
    pts = sorted(((L["lam"], w, L) for w, r in runs.items() for L in r.get("levels", [])),
                 key=lambda p: p[0])
    clusters = []
    for lam, w, L in pts:
        c = clusters[-1] if clusters else None
        if c is not None and any(abs(lam - M["lam"]) <= _same(L, M, lam) for _, M in c):
            c.append((w, L))
        else:
            clusters.append([(w, L)])
    if not clusters:
        return []

    lam_c = np.array([min((L for _, L in c), key=lambda L: L["bound"])["lam"] for c in clusters])
    groups, g = [], [0]
    for i in range(1, len(clusters)):
        if lam_c[i] / lam_c[i - 1] - 1 < FEM_RESOLVES:
            g.append(i)
        else:
            groups.append(g)
            g = [i]
    groups.append(g)

    levels = []
    for gi, g in enumerate(groups):
        lo_cut = np.sqrt(lam_c[groups[gi - 1][-1]] * lam_c[g[0]]) if gi > 0 else 0.0
        hi_cut = np.sqrt(lam_c[g[-1]] * lam_c[groups[gi + 1][0]]) if gi + 1 < len(groups) else hi
        n_fem = int(np.sum((fems[-1] > lo_cut) & (fems[-1] < hi_cut)))
        n_fem_coarse = int(np.sum((fems[0] > lo_cut) & (fems[0] < hi_cut)))
        for i in g:
            c = clusters[i]
            per_run = {}
            for w, L in c:
                x = per_run.setdefault(w, dict(mult=0, uncertain=False, lams=[], bound=0.0))
                x["mult"] += L["mult"]
                x["uncertain"] |= L["uncertain"]
                x["lams"].append(L["lam"])
                x["bound"] = max(x["bound"], L["bound"])
            clean = {w: x for w, x in per_run.items() if not x["uncertain"]}
            mults = [x["mult"] for x in clean.values()]
            m_major = max(set(mults), key=mults.count) if mults else None
            agree = [w for w, x in clean.items() if x["mult"] == m_major]
            why = []
            if len(g) == 1:
                m_fem = n_fem
                status = None
                if len(agree) >= 2 and m_major == m_fem:
                    status = "certified"
                elif any(x["mult"] == m_fem for x in clean.values()):
                    status, m_major = "fem_resolved", m_fem
                    agree = [w for w, x in clean.items() if x["mult"] == m_fem]
                    why.append(f"multiplicity from FEM ({m_fem}); runs {per_run_summary(per_run)}")
                else:
                    why.append(f"no run matches FEM's count {m_fem}: {per_run_summary(per_run)}")
                if n_fem != n_fem_coarse:
                    why.append(f"FEM count not settled: {n_fem_coarse} -> {n_fem}")
                    if status == "fem_resolved":
                        status = None        # FEM alone decided it, and FEM is still moving
            else:
                # A tight group: FEM checks the group total, the runs the split inside it.
                status = "certified" if len(agree) >= 2 else None
                why.append(f"in a group of {len(g)} levels closer than {FEM_RESOLVES:g}; FEM "
                           f"checks the group total ({n_fem})")
            if status is None:
                status = "provisional"
            good = [L for w, L in c if w in agree] or [L for _, L in c]
            best = min(good, key=lambda L: L["bound"])
            vals = [L["lam"] for L in good]
            levels.append(dict(lam=best["lam"], mult=m_major if m_major is not None else
                               max(x["mult"] for x in per_run.values()),
                               bound=best["bound"],
                               spread=float(max(vals) - min(vals)) if len(vals) > 1 else float("nan"),
                               runs={w: dict(mult=x["mult"], lams=x["lams"], bound=x["bound"],
                                             uncertain=x["uncertain"]) for w, x in per_run.items()},
                               group=gi, fem_group_count=n_fem, status=status, why=why))
    # Group totals: the consensus multiplicities against FEM.
    for gi, g in enumerate(groups):
        tot = sum(L["mult"] for L in levels if L["group"] == gi)
        n_fem = levels[g[0]]["fem_group_count"]
        if tot != n_fem:
            for L in levels:
                if L["group"] == gi:
                    L["why"].append(f"group total {tot} vs FEM {n_fem}")
                    L["status"] = "provisional"
    return levels


def _same(L, M, lam):
    """How close two runs' levels must be to be one level: twice the larger bound, floored at
    1e-8 and CAPPED at 1e-6 relative. Uncapped, one refused or starved bound (Moler-Payne reads
    `inf` when refused) chained three distinct levels 1.4 apart into one cluster; the runs'
    agreement on a genuine level was measured at <= 1e-7 relative on the trial cases."""
    return min(2 * max(L["bound"], M["bound"], 1e-8 * lam), 1e-6 * lam)


def per_run_summary(per_run):
    return ", ".join(f"{w}:{x['mult']}{'?' if x['uncertain'] else ''}"
                     for w, x in sorted(per_run.items()))


def one(case):
    from fem import fem_eigs
    v = case["vertices"]
    t0 = time.perf_counter()
    K = max(8, (case.get("n_eigs_requested") or 4) + 3)
    fems = [fem_eigs(v, K + 6, r)[0] for r in FEM_REFINE]
    lo, hi, n_below = window(case, fems[-1])
    runs = {w: chladni_solve(v, w, lo, hi) for w in SCALE}
    levels = compare(runs, fems, hi)
    return dict(id=case["id"], window=[lo, hi], fem_count_below_hi=n_below,
                fem={str(r): f.tolist() for r, f in zip(FEM_REFINE, fems)},
                runs={w: {k: x for k, x in r.items() if k != "levels"} for w, r in runs.items()},
                levels=levels, wall=time.perf_counter() - t0)


def safe_one(case):
    try:
        return one(case)
    except Exception as e:  # noqa: BLE001 -- one bad case must not lose the others
        import traceback
        return dict(id=case["id"], error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-1500:], levels=[], wall=0.0)


def main():
    from concurrent.futures import as_completed
    cases = load_cases()
    want = set(sys.argv[1:])
    if want:
        cases = [c for c in cases if c["id"] in want]
    out_path = os.path.join(HERE, "references.json" if not want else "references_partial.json")
    ckpt = out_path + "l"                      # one JSON line per finished case
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line)
            if "error" not in r:
                done[r["id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    with ProcessPoolExecutor(8) as ex, open(ckpt, "a") as fh:
        futs = {ex.submit(safe_one, c): c["id"] for c in todo}
        for f in as_completed(futs):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[r["id"]] = r
            if "error" in r:
                print(f"{r['id']:22s} ERROR {r['error']}", flush=True)
                continue
            n_c = sum(L["status"] != "provisional" for L in r["levels"])
            cnt = sum(L["mult"] for L in r["levels"])
            print(f"{r['id']:22s} {r['wall']:6.0f}s window [{r['window'][0]:.4g}, "
                  f"{r['window'][1]:.4g}]  levels {len(r['levels'])} (count {cnt}, FEM "
                  f"{r['fem_count_below_hi']})  trusted {n_c}"
                  + "".join(f"\n      {L['lam']:.10g} x{L['mult']} {L['status']}: {L['why']}"
                            for L in r["levels"] if L["status"] != "certified"), flush=True)
    order = [c["id"] for c in cases]
    json.dump(dict(prec=PREC, fem_refine=FEM_REFINE, scale=SCALE,
                   cases=[done[i] for i in order if i in done]),
              open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
