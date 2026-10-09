"""Run-to-run nondeterminism at a near-degeneracy: where does it enter?  (2026-10-09)

Fixture: douse's T1.2 pentagon, run C's basis (`benchmarks/douse/make_references.basis_for`,
2.0x lappy's per-corner FB sizing), window (10.7876, 183.799), `prec = 1e-12`. Its near-double at
68.8536737 (two eigenvalues ~9e-10 apart; FEM counts 2) came back 2, 2, 1 across three identical
reference runs on 2026-10-07.

    .venv/bin/python notes/nondeterminism.py rate [N]       # 1a: N fresh processes + N in-process
    .venv/bin/python notes/nondeterminism.py grid [N]       # 1b: bitwise tensions on a fixed grid
    .venv/bin/python notes/nondeterminism.py child          # one solve, JSON on stdout (internal)

Every run records the count near the pair, and two SHA-256 fingerprints of the tension
evaluations the solve made: `fp_seq` over the sequence in call order, `fp_set` over the sorted
set (a thread pool can reorder calls without changing any number).
"""
import hashlib
import json
import os
import subprocess
import sys
import warnings

import numpy as np

sys.path.insert(0, "benchmarks/douse")
warnings.simplefilter("ignore")

PAIR = 68.8536736647
WINDOW = (10.787616554650096, 183.79944350430173)


def _fixture():
    import make_references as M
    case = {c["id"]: c for c in M.load_cases()}["T1.2"]
    dom = M.polygon(case["vertices"])
    return dom, M.basis_for(dom, "C", np.sqrt(WINDOW[1]))


def solve_once(dom, basis):
    from chladni.mps import MPSEigensolver
    from chladni.mps import pencil as P
    log = []
    orig = P.MPSPencil.tensions

    def spy(self, lam):
        t = orig(self, lam)
        log.append((float(lam), np.asarray(t, dtype=float).tobytes()))
        return t

    P.MPSPencil.tensions = spy
    try:
        n = len(basis)
        s = MPSEigensolver(dom, basis=basis, n_bdry=3 * n, n_int=n, prec=1e-12)
        res = s.search_interval(*WINDOW)
    finally:
        P.MPSPencil.tensions = orig
    near = [L for L in res.levels if abs(L.lam - PAIR) < 1e-6]
    seq = hashlib.sha256(b"".join(np.float64(l).tobytes() + t for l, t in log)).hexdigest()[:12]
    st = hashlib.sha256(b"".join(np.float64(l).tobytes() + t for l, t in sorted(log))).hexdigest()[:12]
    return dict(count=int(sum(L.mult for L in near)), levels=[(L.lam, L.mult) for L in near],
                flagged=any(L.count_uncertain for L in near), n_eval=len(log),
                fp_seq=seq, fp_set=st, total=int(sum(L.mult for L in res.levels)))


def child():
    dom, basis = _fixture()
    print(json.dumps(solve_once(dom, basis)))


def rate(n):
    out = {}
    for threads in ("1", None):
        env = dict(os.environ)
        if threads:
            env["VECLIB_MAXIMUM_THREADS"] = threads
        else:
            env.pop("VECLIB_MAXIMUM_THREADS", None)
        tag = f"VECLIB={threads or 'default'}"
        fresh = []
        for _ in range(n):
            r = subprocess.run([sys.executable, __file__, "child"], env=env, capture_output=True,
                               text=True, check=True)
            fresh.append(json.loads(r.stdout.strip().splitlines()[-1]))
        r = subprocess.run([sys.executable, __file__, "inproc", str(n)], env=env,
                           capture_output=True, text=True, check=True)
        inproc = json.loads(r.stdout.strip().splitlines()[-1])
        out[tag] = dict(fresh=fresh, inproc=inproc)
        for name, rows in (("fresh processes", fresh), ("one process", inproc)):
            counts = [x["count"] for x in rows]
            print(f"{tag:16s} {name:16s} counts {dict(sorted({c: counts.count(c) for c in counts}.items()))}"
                  f"  distinct fp_seq {len({x['fp_seq'] for x in rows})}"
                  f"  fp_set {len({x['fp_set'] for x in rows})}"
                  f"  n_eval {sorted({x['n_eval'] for x in rows})}", flush=True)
    json.dump(out, open("notes/nondeterminism_rate.json", "w"), indent=1)


def inproc(n):
    dom, basis = _fixture()
    print(json.dumps([solve_once(dom, basis) for _ in range(n)]))


GRID = PAIR + np.linspace(-3e-9, 3e-9, 25)
GRID = np.concatenate([GRID, np.linspace(60.0, 80.0, 25)])


def grid_child():
    from chladni.mps import MPSEigensolver
    dom, basis = _fixture()
    n = len(basis)
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * n, n_int=n, prec=1e-12)
    rows = [np.asarray(s._pencil.tensions(float(x)), dtype=float) for x in GRID]
    print(json.dumps([hashlib.sha256(r.tobytes()).hexdigest()[:12] for r in rows]))


def grid(n):
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1")
    runs = []
    for _ in range(n):
        r = subprocess.run([sys.executable, __file__, "gridchild"], env=env, capture_output=True,
                           text=True, check=True)
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    differ = [i for i in range(len(GRID)) if len({run[i] for run in runs}) > 1]
    print(f"{n} fresh processes, {len(GRID)} grid points: {len(differ)} points differ bitwise "
          f"across processes; first few lam: {[float(GRID[i]) for i in differ[:5]]}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "child":
        child()
    elif mode == "inproc":
        inproc(int(sys.argv[2]))
    elif mode == "rate":
        rate(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif mode == "gridchild":
        grid_child()
    elif mode == "grid":
        grid(int(sys.argv[2]) if len(sys.argv) > 2 else 10)


# --- 1b': the solve is deterministic (rate: 80/80 identical), so test what feeds it --------------

def window_repro(n):
    """E1: does `make_references.window` (from ARPACK FEM eigenvalues) repeat bitwise?"""
    import make_references as M
    from fem import fem_eigs
    case = {c["id"]: c for c in M.load_cases()}["T1.2"]
    K = max(8, (case.get("n_eigs_requested") or 4) + 3)
    wins = []
    for _ in range(n):
        f = fem_eigs(case["vertices"], K + 6, 7)[0]
        lo, hi, _ = M.window(case, f)
        wins.append((lo, hi))
    print(f"E1: {n} in-process FEM windows, distinct: {len(set(wins))}; "
          f"lo spread {max(w[0] for w in wins) - min(w[0] for w in wins):.1e}, "
          f"hi spread {max(w[1] for w in wins) - min(w[1] for w in wins):.1e}")


def window_sensitivity():
    """E2: the count at the pair against tiny relative perturbations of the window ends."""
    global WINDOW
    dom, basis = _fixture()
    base = WINDOW
    rows = []
    for end in (0, 1):
        for rel in (-1e-9, -1e-12, -1e-14, -4e-16, 4e-16, 1e-14, 1e-12, 1e-9):
            w = list(base)
            w[end] = w[end] * (1 + rel)
            WINDOW = tuple(w)
            r = solve_once(dom, basis)
            rows.append((("lo", "hi")[end], rel, r["count"], r["flagged"], r["n_eval"]))
            print(f"E2: {('lo', 'hi')[end]} x (1{rel:+.0e}): count {r['count']} "
                  f"flagged {r['flagged']} n_eval {r['n_eval']} levels {r['levels']}", flush=True)
    WINDOW = base


def order_effect():
    """E3: run A and B first in the same process, then C, as make_references.one() does."""
    import make_references as M
    case = {c["id"]: c for c in M.load_cases()}["T1.2"]
    out = {w: M.chladni_solve(case["vertices"], w, *WINDOW) for w in ("A", "B", "C")}
    for w, r in out.items():
        near = [L for L in r["levels"] if abs(L["lam"] - PAIR) < 1e-6]
        print(f"E3: run {w} (after {'nothing' if w == 'A' else 'A' if w == 'B' else 'A, B'}): "
              f"count {sum(L['mult'] for L in near)}")


if __name__ == "__main__" and sys.argv[1] in ("window", "sens", "order"):
    {"window": lambda: window_repro(5), "sens": window_sensitivity, "order": order_effect}[sys.argv[1]]()
