"""Source layer sized by the smallest feature, not the diameter: measured before it is built.
(2026-10-13)

The douse N = 6, k = 2 ratio run (seed 1) stalls on a hexagon with one edge 0.042 long (perimeter
4, diameter 1.44): every rung but 2 misses 1e-10, and rung 2 meets it at margin 2.0 exactly, so the
next step refuses. The offset layer sits 0.05 x diameter = 0.072 out -- farther than the short edge
is long. Measured on that shape (`ell` = the smallest corner clearance = 0.042):

    offset            sources   margins (lam_1..4)      met
    0.072 (5% D)      48        2.3  2.0  6.8  9.0      yes, barely
    0.029             48        2.0  1.4  14   8.9      no
    0.014             48        0.8  4.6  6.5  4.1      no
    0.029             96        6.0  3.7  4.7  17       yes
    0.014 (~ell/3)    96        9.0  8.0  9.6  6.8      yes

Closer alone is worse; closer AND denser is comfortable. The candidate rule, for rungs with an
offset layer: offset = min(recipe offset x D, ell / 3), and enough sources that their spacing is at
most 3x the offset (perimeter / (3 offset)), capped at 4x the recipe's count. For an ordinary shape
`ell / 3` exceeds 5% of D and nothing changes; 11 of the 41 douse benchmark cases have
`ell / 3 < 0.05 D`. This script solves those 11 (plus the stall shape) under both rules, scored as
`notes/phase5_benchmark.py` scores them.

    .venv/bin/python notes/short_edge_sources.py [WORKERS]
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "benchmarks", "douse"), HERE]
warnings.simplefilter("ignore")

PREC = 1e-10
OFFSET_PER_FEATURE = 1.0 / 3.0
SPACING_PER_OFFSET = 3.0
MAX_SOURCE_GROWTH = 4.0

#: The stalled hexagon (douse chladni_vs_lappy.jsonl, chladni N=6 k=2 seed 1, last iterate).
STALL = [[0.0417, 0.0], [1.1091, 0.5169], [0.9097, 1.1191], [0.2018, 1.2364],
         [-0.2318, 0.6575], [0.0, 0.0]]


def feature_rule():
    """Patch `plan_for` with the candidate rule (in this worker only)."""
    import dataclasses
    import chladni.plan as PL
    real = PL.plan_for

    def plan_for(dom, n, prec=PL.PREC, rung=0):
        p = real(dom, n, prec, rung)
        if not p.n_sources:
            return p
        ell = min(c.clearance for c in dom.corners)
        off = min(p.source_offset * dom.diameter, OFFSET_PER_FEATURE * ell)
        n_src = max(p.n_sources, int(np.ceil(dom.perimeter / (SPACING_PER_OFFSET * off))))
        n_src = min(n_src, int(MAX_SOURCE_GROWTH * p.n_sources))
        return dataclasses.replace(p, source_offset=off / dom.diameter, n_sources=n_src)
    PL.plan_for = plan_for
    import chladni.eigenproblem as EP
    EP.plan_for = plan_for


SHORT_EDGE = 0.15          # x the diameter: an edge this short gets a local patch
PATCH_REACH = 2.0          # x the edge: boundary within this of its midpoint is covered
PATCH_OFFSET = 1.0 / 3.0   # x the edge
PATCH_SPACING = 0.5        # x the edge


def edge_patches(dom):
    """Local source patches at the short edges: boundary points within PATCH_REACH x e of each
    short edge's midpoint, PATCH_SPACING x e apart, offset PATCH_OFFSET x e outward."""
    from chladni.helmholtz import Fundamental
    segs = [s for b in dom.boundaries for s in b.segments]
    quad = dom.bdry_quad(4000)
    pts, nrm = np.asarray(quad.pts), np.asarray(quad.normals)
    fams = []
    for s in segs:
        e = abs(s.p1 - s.p0)
        if e >= SHORT_EDGE * dom.diameter:
            continue
        mid = 0.5 * (s.p0 + s.p1)
        near = np.where(np.abs(pts - mid) <= PATCH_REACH * e)[0]
        chosen = []
        for i in near[np.argsort(np.abs(pts[near] - mid))]:
            if all(abs(pts[i] - pts[j]) >= PATCH_SPACING * e for j in chosen):
                chosen.append(i)
        org = pts[chosen] + PATCH_OFFSET * e * nrm[chosen]
        fam = Fundamental(org, radius=PATCH_OFFSET * e, extent=dom.diameter + e, prec=dom_prec())
        try:
            fams.append(fam.check_exterior(dom))
        except Exception:   # noqa: BLE001 -- a reflex neighbour put one inside: skip the patch
            pass
    return fams


def dom_prec():
    return PREC


def patch_rule():
    """Patch `Plan.realize` to add the local patches (in this worker only)."""
    import chladni.plan as PL
    real = PL.Plan.realize

    def realize(self, dom):
        basis, nb, ni = real(self, dom)
        if not self.n_sources:
            return basis, nb, ni
        for f in edge_patches(dom):
            basis = basis + f
        m = len(basis)
        return basis, int(np.ceil(self.bdry_ratio * m)), int(np.ceil(self.int_ratio * m))
    PL.Plan.realize = realize


def run(args):
    case, ref, rule = args
    if rule == "feature":
        feature_rule()
    if rule == "patch":
        patch_rule()
    if ref is None:                                   # the stall shape: no reference
        import make_references as MR
        from chladni.eigenproblem import Eigenproblem
        dom = MR.polygon(case["vertices"])
        t0 = time.perf_counter()
        spec = Eigenproblem(dom).solve(4, prec=PREC)
        return dict(id=case["id"], rule=rule, wall=time.perf_counter() - t0,
                    rung=spec.plan.rung, n_cols=spec.plan.n_cols, flags=list(spec.flags),
                    margins=[round(L.margin, 2) for L in spec.levels],
                    lam=[float(x) for x in spec.unfolded])
    import phase5_benchmark as P5
    r = P5.run_one((case, ref, PREC))
    r["rule"] = rule
    if "levels" in r:
        r["verdict"] = P5.score(r, ref)[0]
    return r


def main(workers=4):
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    changed = []
    for i, c in cases.items():
        dom = MR.polygon(c["vertices"])
        if OFFSET_PER_FEATURE * min(cc.clearance for cc in dom.corners) < 0.05 * dom.diameter:
            changed.append(i)
    rules = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("old", "feature")
    jobs = [(dict(id="stall-N6k2s1", vertices=STALL), None, rule) for rule in rules]
    jobs += [(cases[i], refs[i], rule) for i in changed if i in refs for rule in rules]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run, jobs))
    out = os.path.join(HERE, f"short_edge_sources.{'_'.join(rules)}.json")
    json.dump(dict(runs=runs), open(out, "w"), indent=1)
    by = {}
    for r in runs:
        by.setdefault(r["id"], {})[r["rule"]] = r
    print(f"{'case':24s} " + "   ".join(f"{r:>34s}" for r in rules))
    for i, d in by.items():
        cells = []
        for rule in rules:
            r = d.get(rule, {})
            v = r.get("verdict") or ("ERR" if "error" in r else "flags" if r.get("flags") else "clean")
            extra = f" m={r['margins']}" if "margins" in r else ""
            cells.append(f"{v:13s} rung {r.get('rung')} {r.get('n_cols')}c {r.get('wall', 0):5.0f}s{extra}")
        print(f"{i:24s} " + "   ".join(f"{c:>34s}" for c in cells))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
