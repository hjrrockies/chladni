"""Out-of-sample: Hadamard derivatives on the shapes douse found hardest.  (2026-11-03)

The graded corner rule (`notes/hadamard_accuracy.md`, "Fix built") was tested on shapes chosen
while building it. This script runs it on shapes it never saw: the final iterates of douse ratio
runs on ARC (`thesis_ratio_v1/v2`, chladni e1ebaf8) whose trust radius collapsed to ~6e-15 --
the noise-limited signature -- plus two plateau controls (`hadamard_hard_shapes_cases.json`).
Mostly k = 4..5 cells with re-entrant corners up to 228 deg and edges down to 0.05.

Each shape is solved with the chladni on `sys.path`, and every level gets

  IDENTITIES. Translation in x and y and rotation give H = 0; dilation gives eig(H) = -2 lam.
     Exact, needing no reference solve: max error / lam.
  VERTEX GRADIENTS. tr H for each vertex's hat field in x and y, set beside
  FD. Central differences of the eigenvalues (a cluster's sum) along two fixed random vertex
     directions, at prec 1e-12, steps h and h/2, Richardson-combined.

Old (e1ebaf8, what the runs used) against new (this branch) is two checkouts running the SAME
script; `task` picks the checkout by putting its `src` first on PYTHONPATH and records which
chladni it imported.

Run:
  python notes/hadamard_hard_shapes.py run H <case> <prec>       one Hadamard probe, JSON to stdout
  python notes/hadamard_hard_shapes.py run FD <case>             one FD reference
  python notes/hadamard_hard_shapes.py task <i> --old SRC --new SRC --out DIR
                                                                 Slurm array task i of N_TASKS
  python notes/hadamard_hard_shapes.py analyze DIR               the table
"""
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = json.load(open(HERE / "hadamard_hard_shapes_cases.json"))["cases"]
PRECS = ("1e-10", "1e-12")
N_DIRS, H_STEP, FD_PREC = 2, 4e-3, 1e-12


def tasks():
    """Every (impl, mode, case, prec): the FD references first, they are the long ones."""
    out = [("new", "FD", c, None) for c in range(len(CASES))]
    out += [(impl, "H", c, p) for p in PRECS for c in range(len(CASES)) for impl in ("old", "new")]
    return out


N_TASKS = len(tasks())


def fields(dom, verts, center=0j):
    """Translation x, y; rotation; dilation (both about `center`); then each vertex's hat field in
    x and in y."""
    segs = [s for b in dom.boundaries for s in b.segments]

    def V(curve, t):
        s = segs[curve]
        t = np.asarray(t, float)
        p, n = s.p(t), s.N(t)
        a, b = s.p(np.array([0.0]))[0], s.p(np.array([1.0]))[0]
        out = [np.ones_like(p), 1j * np.ones_like(p), 1j * (p - center), p - center]
        for v in verts:
            for w in (1.0, 1j):
                d = np.zeros_like(p)
                if abs(a - v) < 1e-12:
                    d = d + w * (1 - t)
                if abs(b - v) < 1e-12:
                    d = d + w * t
                out.append(d)
        return np.array([(f * n.conjugate()).real for f in out])
    return V


def directions(N):
    rng = np.random.default_rng(12345)
    d = rng.normal(size=(N_DIRS, N)) + 1j * rng.normal(size=(N_DIRS, N))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _verts(c):
    return np.array([complex(*v) for v in c["verts"]])


def _solve(verts, n, prec):
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    return Eigenproblem(Polygon(verts, bc="dir")).solve(n, prec=prec)


def _expanded(rec):
    return np.concatenate([[float(L.lam)] * int(L.mult) for L in rec.spectrum.levels])


def run_H(c, prec, centered=False):
    verts = _verts(c)
    center = complex(np.mean(verts)) if centered else 0j
    t0 = time.time()
    rec = _solve(verts, len(c["lams"]), prec)
    t_solve = time.time() - t0
    V = fields(rec.solver.domain, verts, center)
    out = []
    for i, L in enumerate(rec.spectrum.levels):
        lam = float(L.lam)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                try:
                    H, rep = rec.hadamard(i, V, report=True)
                    rep = dict(sizing=rep.sizing_prec, attained=rep.attained,
                               gram=list(np.atleast_1d(rep.gram_residual)),
                               shortfalls=[str(s)[:80] for s in rep.shortfalls])
                except TypeError:                       # e1ebaf8 has no report
                    H, rep = rec.hadamard(i, V), None
            except Exception as e:                      # a refusal is a result
                out.append(dict(i=i, lam=lam, error=f"{type(e).__name__}: {str(e)[:200]}"))
                continue
        H = np.asarray(H)
        out.append(dict(i=i, lam=lam, m=H.shape[1],
                        ident=[float(np.abs(H[f]).max() / lam) for f in range(3)],
                        dil=float(np.abs(np.linalg.eigvalsh(H[3]) + 2 * lam).max() / lam),
                        grad=[float(np.trace(H[f])) for f in range(4, H.shape[0])],
                        report=rep, warnings=sorted({str(x.message)[:120] for x in w})))
    return dict(t_solve=t_solve, lams=[float(L.lam) for L in rec.spectrum.levels],
                mults=[int(L.mult) for L in rec.spectrum.levels], levels=out,
                center=[center.real, center.imag],
                radius=float(np.abs(verts - center).max()),
                rung=int(getattr(getattr(rec, "plan", None), "rung", -1)),
                met=[bool(L.met) for L in rec.spectrum.levels],
                bounds=[float(L.bound) for L in rec.spectrum.levels])


def run_FD(c):
    verts = _verts(c)
    n = len(c["lams"]) + 1
    lam0 = _expanded(_solve(verts, n, FD_PREC))
    res = []
    for d in directions(len(verts)):
        vals = {s: _expanded(_solve(verts + s * (H_STEP / 2) * d, n, FD_PREC))
                for s in (-2, -1, 1, 2)}
        m = min(len(v) for v in vals.values())
        D1 = (vals[2][:m] - vals[-2][:m]) / (2 * H_STEP)
        D2 = (vals[1][:m] - vals[-1][:m]) / H_STEP
        rich = (4 * D2 - D1) / 3
        res.append(dict(d=[[z.real, z.imag] for z in d], D=rich.tolist(),
                        richardson_gap=np.abs(rich - D2).tolist()))
    return dict(lams=lam0.tolist(), dirs=res)


# --- FDref: an FD gradient reference with its own error bar (2026-11-05) ------------------------
#
# The FD column above cannot judge a gradient: its step (4e-3) is the size of these shapes'
# shortest edges, its Richardson gap reaches 3e-3 of lam, and it disagreed with H by up to 0.11
# identically on two chladnis (`accuracy_audit_2026-11-03.md`, "Hard-shapes Hadamard, FD
# column"). FDref changes three things:
#   - ONE PLAN, made at the base shape, solves every perturbed shape (`Plan.solver`), so no rung
#     switch puts a jump into a difference quotient;
#   - a STEP LADDER, each step Richardson-combined from +-h and +-2h, so truncation is measured
#     (consecutive steps) rather than assumed;
#   - every value carries a NOISE bar from the levels' own certified bounds, (b(+h) + b(-h)) / 2h.
# Directions: the two seeded random ones, then x and y at the most re-entrant vertex, where u_n
# is most singular.

FDREF_STEPS = (1e-3, 3e-4, 1e-4)
FDREF_N_DIRS = 4


def fdref_directions(verts):
    N = len(verts)
    e = np.roll(verts, -1) - verts
    turn = np.angle(e / np.roll(e, 1))                      # exterior turn at each vertex
    ccw = np.sum(turn) > 0
    interior = np.pi - (turn if ccw else -turn)
    j = int(np.argmax(interior))
    ex = np.zeros(N, complex)
    ex[j] = 1.0
    return list(directions(N)) + [ex, 1j * ex], j, float(np.degrees(interior[j]))


def _fixed_plan_solve(plan, verts, lo, hi):
    """Expanded values and their absolute bars on `plan`, plus each level's `met`."""
    from chladni.geometry import Polygon
    res = plan.solver(Polygon(verts, bc="dir"), prec=FD_PREC).search_interval(lo, hi)
    lam, bar, met = [], [], []
    for L in res.levels:
        m = int(L.mult)
        lam += [float(L.lam)] * m
        bar += [float(L.bound)] * m
        met += [bool(L.met(FD_PREC))] * m
    return np.array(lam), np.array(bar), met


def run_FDref(c, di):
    verts = _verts(c)
    n = len(c["lams"])
    t0 = time.time()
    rec = _solve(verts, n + 1, FD_PREC)
    lam0 = _expanded(rec)
    lo = 0.9 * lam0[0]
    hi = 0.5 * (lam0[n - 1] + lam0[n]) if len(lam0) > n else lam0[n - 1] * (1 + 1e-3)
    dirs, j, angle = fdref_directions(verts)
    d = dirs[di]
    base = _fixed_plan_solve(rec.plan, verts, lo, hi)
    steps = []
    for h in FDREF_STEPS:
        pts = {s: _fixed_plan_solve(rec.plan, verts + s * h * d, lo, hi) for s in (-2, -1, 1, 2)}
        m = min(len(v[0]) for v in pts.values())
        f = {s: v[0][:m] for s, v in pts.items()}
        b = {s: v[1][:m] for s, v in pts.items()}
        D1 = (f[1] - f[-1]) / (2 * h)
        D2 = (f[2] - f[-2]) / (4 * h)
        n1 = (b[1] + b[-1]) / (2 * h)
        n2 = (b[2] + b[-2]) / (4 * h)
        steps.append(dict(h=h, D=((4 * D1 - D2) / 3).tolist(), noise=((4 * n1 + n2) / 3).tolist(),
                          counts={str(s): len(v[0]) for s, v in pts.items()},
                          unmet=sum(not all(v[2]) for v in pts.values()),
                          raw={str(s): dict(lam=v[0].tolist(), bar=v[1].tolist())
                               for s, v in pts.items()}))
    return dict(lams=base[0].tolist(), bars=base[1].tolist(), base_met=all(base[2]),
                rung=int(rec.plan.rung), window=[lo, hi], n=n, dir_index=di,
                d=[[z.real, z.imag] for z in d], vertex=j, vertex_angle=angle,
                steps=steps, t=time.time() - t0)


def run(mode, ci, prec=None):
    import chladni
    c = CASES[ci]
    if mode == "FDref":
        r = run_FDref(c, int(prec))
    elif mode in ("H", "Hc"):
        r = run_H(c, float(prec), centered=mode == "Hc")
    else:
        r = run_FD(c)
    r.update(mode=mode, case=ci, source=c["source"], prec=prec, chladni=chladni.__file__)
    return r


def tasks_fdref():
    """The FDref array: (case, direction) references first, then main's centred H probes."""
    out = [("new", "FDref", c, str(d)) for c in range(len(CASES)) for d in range(FDREF_N_DIRS)]
    out += [("new", "Hc", c, p) for p in PRECS for c in range(len(CASES))]
    return out


def task(i, old, new, out, table=tasks):
    impl, mode, ci, prec = table()[i]
    rest = [q for q in os.environ.get("PYTHONPATH", "").split(os.pathsep) if q]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(old if impl == "old" else new).resolve())] + rest))
    args = [sys.executable, __file__, "run", mode, str(ci)] + ([prec] if prec else [])
    p = subprocess.run(args, env=env, capture_output=True, text=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    name = Path(out) / f"{impl}_{mode}{prec or ''}_case{ci}.json"
    if p.returncode != 0:
        name.write_text(json.dumps(dict(impl=impl, mode=mode, case=ci, prec=prec,
                                        failed=p.stderr[-2000:])))
        sys.exit(p.returncode)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r["impl"] = impl
    expect = Path(old if impl == "old" else new).resolve()
    if not Path(r["chladni"]).resolve().is_relative_to(expect):
        raise RuntimeError(f"task {i} imported chladni from {r['chladni']}, not {expect}")
    name.write_text(json.dumps(r))


def _groups(lams, mults, rtol=1e-6):
    """Levels closer than rtol go together (a cluster's sum is smooth); with eigenvalue indices."""
    out, start = [], 0
    for i, (l, m) in enumerate(zip(lams, mults)):
        if out and l - lams[out[-1][0][-1]] < rtol * l:
            out[-1][0].append(i); out[-1][1].extend(range(start, start + m))
        else:
            out.append(([i], list(range(start, start + m))))
        start += m
    return out


#: Grouping for the FD comparison. FD differences SORTED eigenvalues, so two levels closer than
#: the step can move cross between the perturbed shapes and swap indices; a ratio optimum
#: drives lam_k toward exactly such a cluster. Only a group's SUM is smooth across a crossing,
#: so the grouped column sums any levels within a few steps of each other.
FD_GROUP_RTOL = 5 * H_STEP


def _summary(r, fd, rtol=1e-6):
    lv = {L["i"]: L for L in r["levels"]}
    ok = [L for L in r["levels"] if "error" not in L]
    errs = [L["error"] for L in r["levels"] if "error" in L]
    ident = max((max(L["ident"] + [L["dil"]]) for L in ok), default=np.nan)
    warned = sum(any("Hadamard matrix" in w for w in L["warnings"]) for L in ok)
    fderr = np.nan
    if fd:
        fderr = 0.0
        for levs, eigs in _groups(r["lams"], r["mults"], rtol):
            if any("error" in lv[i] for i in levs) or max(eigs) >= len(fd["lams"]):
                continue
            g = np.sum([lv[i]["grad"] for i in levs], axis=0)
            lam = r["lams"][levs[-1]]
            for d in fd["dirs"]:
                dz = np.array([complex(*z) for z in d["d"]])
                had = np.sum(dz.real * g[0::2] + dz.imag * g[1::2])
                fderr = max(fderr, abs(had - np.sum(np.array(d["D"])[eigs])) / (lam * len(eigs)))
    return ident, fderr, warned, errs


def analyze(d):
    rows = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]
    for r in rows:
        if "failed" in r:
            print("FAILED", r["impl"], r["mode"], r["case"], r["prec"], r["failed"][-300:])
    fds = {r["case"]: r for r in rows if r.get("mode") == "FD" and "failed" not in r}
    hs = {(r["impl"], r["case"], r["prec"]): r for r in rows if r.get("mode") == "H" and "failed" not in r}
    print(f"{'case':24s} {'impl':4s} {'prec':6s} {'ident/lam':>9s} {'FD/lam':>9s} {'FDgrp/lam':>9s}"
          f" warn refused  solve")
    for ci, c in enumerate(CASES):
        for p in PRECS:
            for impl in ("old", "new"):
                r = hs.get((impl, ci, p))
                if r is None:
                    continue
                ident, fderr, warned, errs = _summary(r, fds.get(ci))
                fdgrp = _summary(r, fds.get(ci), FD_GROUP_RTOL)[1]
                print(f"{c['source'][13:]:24s} {impl:4s} {p:6s} {ident:9.1e} {fderr:9.1e} {fdgrp:9.1e} "
                      f"{warned:4d} {len(errs):7d} {r['t_solve']:6.0f}s  {errs[0][:70] if errs else ''}")
        if ci in fds:
            gap = max(max(d["richardson_gap"]) for d in fds[ci]["dirs"]) / max(fds[ci]["lams"])
            print(f"{'':24s} FD Richardson gap / lam_max {gap:.1e}")


def _fdref_values(r):
    """Per expanded eigenvalue: the derivative from the step whose bar is smallest, and that bar.
    A step's bar is its noise plus its truncation, measured as the change to the next step (the
    last step borrows the previous difference)."""
    st = r["steps"]
    m = min(len(x["D"]) for x in st)
    D = np.array([x["D"][:m] for x in st])
    noise = np.array([x["noise"][:m] for x in st])
    trunc = np.abs(np.diff(D, axis=0))
    trunc = np.vstack([trunc, trunc[-1:]])
    bar = noise + trunc
    k = np.argmin(bar, axis=0)
    cols = np.arange(m)
    return D[k, cols], bar[k, cols], np.array([st[i]["h"] for i in k])


def analyze_fdref(d):
    rows = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]
    for r in rows:
        if "failed" in r:
            print("FAILED", r["mode"], r["case"], r["prec"], r["failed"][-300:])
    fd = {}
    for r in rows:
        if r.get("mode") == "FDref" and "failed" not in r:
            fd.setdefault(r["case"], {})[r["dir_index"]] = r
    hs = {(r["case"], r["prec"]): r for r in rows if r.get("mode") == "Hc" and "failed" not in r}
    print(f"{'case':22s} dir  h-mix         FDbar/lam  |H12-FD|/lam  |H10-FD|/lam  "
          f"id12/lam  id10/lam  notes")
    for ci, c in enumerate(CASES):
        for di, r in sorted(fd.get(ci, {}).items()):
            vals, bar, hh = _fdref_values(r)
            notes = []
            if any(len(set(x["counts"].values())) > 1 or set(x["counts"].values()) != {len(r["lams"])}
                   for x in r["steps"]):
                notes.append("counts differ")
            if not r["base_met"] or any(x["unmet"] for x in r["steps"]):
                notes.append("unmet solve")
            dz = np.array([complex(*z) for z in r["d"]])
            line = {}
            for p in PRECS:
                h = hs.get((ci, p))
                if h is None:
                    continue
                lv = {L["i"]: L for L in h["levels"]}
                err, fbar, idr = 0.0, 0.0, 0.0
                for levs, eigs in _groups(h["lams"], h["mults"], FD_GROUP_RTOL):
                    if any("error" in lv[i] for i in levs) or max(eigs) >= len(vals):
                        continue
                    g = np.sum([lv[i]["grad"] for i in levs], axis=0)
                    had = np.sum(dz.real * g[0::2] + dz.imag * g[1::2])
                    lam = h["lams"][levs[-1]]
                    err = max(err, abs(had - vals[eigs].sum()) / lam)
                    fbar = max(fbar, bar[eigs].sum() / lam)
                    R = h["radius"]
                    idr = max(idr, max(max(lv[i]["ident"][0], lv[i]["ident"][1],
                                           lv[i]["ident"][2] / R, lv[i]["dil"] / R) for i in levs))
                line[p] = (err, fbar, idr)
            if not line:
                continue
            e12, f12, i12 = line.get("1e-12", (np.nan,) * 3)
            e10, f10, i10 = line.get("1e-10", (np.nan,) * 3)
            hmix = "/".join(f"{np.sum(hh == x)}" for x in FDREF_STEPS)
            tag = f"{di}" + (f" v{r['vertex']}({r['vertex_angle']:.0f})" if di >= 2 else "")
            print(f"{c['source'][13:]:22s} {tag:12s} {hmix:8s} {max(f12, f10):9.1e} {e12:12.1e} "
                  f"{e10:12.1e} {i12:9.1e} {i10:9.1e}  {', '.join(notes)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        print(json.dumps(run(a[1], int(a[2]), a[3] if len(a) > 3 else None)), flush=True)
    elif a[0] == "task":
        kw = dict(zip(a[2::2], a[3::2]))
        task(int(a[1]), kw.get("--old", kw["--new"]), kw["--new"], kw["--out"],
             tasks_fdref if kw.get("--table") == "fdref" else tasks)
    elif a[0] == "analyze":
        analyze(a[1])
    elif a[0] == "analyze_fdref":
        analyze_fdref(a[1])
    else:
        sys.exit(__doc__)
