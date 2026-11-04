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


def fields(dom, verts):
    """Translation x, y; rotation; dilation; then each vertex's hat field in x and in y."""
    segs = [s for b in dom.boundaries for s in b.segments]

    def V(curve, t):
        s = segs[curve]
        t = np.asarray(t, float)
        p, n = s.p(t), s.N(t)
        a, b = s.p(np.array([0.0]))[0], s.p(np.array([1.0]))[0]
        out = [np.ones_like(p), 1j * np.ones_like(p), 1j * p, p]
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


def run_H(c, prec):
    verts = _verts(c)
    t0 = time.time()
    rec = _solve(verts, len(c["lams"]), prec)
    t_solve = time.time() - t0
    V = fields(rec.solver.domain, verts)
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
                mults=[int(L.mult) for L in rec.spectrum.levels], levels=out)


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


def run(mode, ci, prec=None):
    import chladni
    c = CASES[ci]
    r = run_H(c, float(prec)) if mode == "H" else run_FD(c)
    r.update(mode=mode, case=ci, source=c["source"], prec=prec, chladni=chladni.__file__)
    return r


def task(i, old, new, out):
    impl, mode, ci, prec = tasks()[i]
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


def _summary(r, fd):
    lv = {L["i"]: L for L in r["levels"]}
    ok = [L for L in r["levels"] if "error" not in L]
    errs = [L["error"] for L in r["levels"] if "error" in L]
    ident = max((max(L["ident"] + [L["dil"]]) for L in ok), default=np.nan)
    warned = sum(any("Hadamard matrix" in w for w in L["warnings"]) for L in ok)
    fderr = np.nan
    if fd:
        fderr = 0.0
        for levs, eigs in _groups(r["lams"], r["mults"]):
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
    print(f"{'case':24s} {'impl':4s} {'prec':6s} {'ident/lam':>9s} {'FD/lam':>9s} warn refused  solve")
    for ci, c in enumerate(CASES):
        for p in PRECS:
            for impl in ("old", "new"):
                r = hs.get((impl, ci, p))
                if r is None:
                    continue
                ident, fderr, warned, errs = _summary(r, fds.get(ci))
                print(f"{c['source'][13:]:24s} {impl:4s} {p:6s} {ident:9.1e} {fderr:9.1e} "
                      f"{warned:4d} {len(errs):7d} {r['t_solve']:6.0f}s  {errs[0][:70] if errs else ''}")
        if ci in fds:
            gap = max(max(d["richardson_gap"]) for d in fds[ci]["dirs"]) / max(fds[ci]["lams"])
            print(f"{'':24s} FD Richardson gap / lam_max {gap:.1e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        print(json.dumps(run(a[1], int(a[2]), a[3] if len(a) > 3 else None)), flush=True)
    elif a[0] == "task":
        kw = dict(zip(a[2::2], a[3::2]))
        task(int(a[1]), kw["--old"], kw["--new"], kw["--out"])
    elif a[0] == "analyze":
        analyze(a[1])
    else:
        sys.exit(__doc__)
