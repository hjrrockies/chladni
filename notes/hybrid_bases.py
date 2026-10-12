"""Phase 5 step 0d: hybrid bases on the hard douse shapes.  (2026-10-12)

The user's experience on thin triangles: hybrid bases are ESSENTIAL. Fractional-order Fourier-
Bessel terms at the SINGULAR corners, plus fundamental-solution charges, either
  - clustered at the corners (Gopal-Trefethen "lightning"): rapid convergence to a ~1e-12 floor;
  - offset from the boundary: slower convergence to a deeper floor.
`notes/hard_cases.md` did not test that recipe. It put Fourier-Bessel at EVERY corner at 2x
lappy's size, then added untapered lightning at every corner (which declined) or a coarse
5%-offset layer.

At prec 1e-10 on the reference windows: group B (sharp, thin, stalled near 1e-7), a group A case,
and two controls. Fourier-Bessel sized by the planner's precision rule (`chladni.plan.corner_order`,
scale 1) at singular corners only, plus:

    L16, L32        lightning, n poles per singular corner (default sigma, untapered)
    L32t            the same with the outer 30% dropped (taper 0.3)
    S6@5%, S6@2%    boundary-offset sources, 6 per wavelength, at 5% / 2% of the diameter
    S10@2%          10 per wavelength at 2%

against the planner's rung 1 (`FB P2`: Fourier-Bessel at every corner, 2x the precision rule).

    .venv/bin/python notes/hybrid_bases.py [WORKERS]
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
sys.path[:0] = [os.path.join(ROOT, "tests"), os.path.join(ROOT, "benchmarks", "douse"), HERE]
warnings.simplefilter("ignore")

PREC = 1e-10
CASES = ("T1.3", "T2.6-N3k4-L8-ref0", "T2.6-N3k5-L6-ref0", "T2.2", "T2.3-pre-ref0",
         "T1.4-oct", "T2.9-N5k3convex")
VARIANTS = ("FB P2", "FBs+L16", "FBs+L32", "FBs+L32t", "FBs+S6@5%", "FBs+S6@2%", "FBs+S10@2%")
#: Round 2 (`HYBRID_ROUND=2`): larger corner expansions with the source layer, and lightning with
#: corner-graded boundary points plus interior points clustered at the singular corners -- the
#: sampling Gopal-Trefethen pair with clustered poles.
VARIANTS2 = ("FBs2+S6@5%", "FBs+L16 graded", "FBs+L24 graded")
#: Round 3 (`HYBRID_ROUND=3`): lightning as lappy set it up (`lappy/bases.py`
#: `FundamentalBasis.by_corners`, `jupyter_notebooks/corner_basis_tests.ipynb`). Poles at
#: C exp(-sigma (sqrt(n) - sqrt(j))) on the exterior bisector, ORDER 2 (3 columns each), a 50/50
#: split of columns between Fourier-Bessel and the poles, Legendre boundary points. C = 1 is an
#: absolute distance (domains of size ~1-10), sigma = 1 on the triangle. chladni's defaults --
#: sigma = 4, L = the corner's clearance, order 1 -- pack the innermost pole ~1e-7 L from the
#: vertex, where rounds 1-2 declined.
VARIANTS3 = ("FBs+FS s1 C1", "FBs+FS s2 C1", "FBs+FS s1 Cd/2", "FBs2+FS s1 C1")
ROUND = int(os.environ.get("HYBRID_ROUND", "1"))
OUT = os.path.join(HERE, {1: "hybrid_bases.json", 2: "hybrid_bases2.json",
                          3: "hybrid_bases3.json"}[ROUND])


def build(dom, k, variant):
    from chladni.helmholtz import FourierBessel, Fundamental
    from chladni.plan import corner_order
    zs = np.array([c.z for c in dom.corners])
    every = variant == "FB P2"
    scale_s = 2.0 if variant.startswith("FBs2") else 1.0
    b = None
    for c in dom.corners:
        if not every and c.is_reflection_closed:
            continue
        r = float(np.max(np.abs(zs - c.z)))
        f = FourierBessel.at_corner(c, n=corner_order(k, r, c.omega, PREC,
                                                      2.0 if every else scale_s), domain=dom)
        b = f if b is None else b + f
    if every:
        return b
    sing = [c for c in dom.corners if not c.is_reflection_closed]
    if "+FS" in variant:            # lappy-style: order 2, a 50/50 column split
        sigma = 2.0 if " s2 " in variant else 1.0
        L = 0.5 * dom.diameter if "Cd/2" in variant else 1.0
        n_src = max(int(round(len(b) / (3 * len(sing)))), 2)
        for c in sing:
            b = b + Fundamental.lightning(c, n_src, sigma=sigma, L=L, order=2)
        return b
    if "+L" in variant:
        n = 16 if "L16" in variant else 24 if "L24" in variant else 32
        taper = 0.3 if variant.endswith("t") else 0.0
        for c in sing:
            b = b + Fundamental.lightning(c, n, taper=taper)
    else:
        dens = 10.0 if "S10" in variant else 6.0
        frac = 0.05 if "@5%" in variant else 0.02
        nsrc = int(np.ceil(dens * k * dom.perimeter / (2 * np.pi)))
        for f in (frac, frac / 2, frac / 4):
            try:
                b = b + Fundamental.offset(dom.bdry_quad(nsrc), f * dom.diameter, domain=dom)
                break
            except Exception:  # noqa: BLE001 -- sources inside a thin domain: move closer
                continue
    return b


def clustered_interior(dom, n, rng):
    """`n` uniform interior points plus `n/2` clustered at the singular corners: radii
    log-spaced from 1e-4 to 0.3 of the diameter, random angles, kept if inside."""
    pts = [dom.interior_pts(n, rng=rng)]
    sing = [c for c in dom.corners if not c.is_reflection_closed]
    per = max(n // (2 * max(len(sing), 1)), 4)
    for c in sing:
        got = []
        while len(got) < per:
            rad = dom.diameter * 10 ** rng.uniform(-4, np.log10(0.3), 4 * per)
            z = c.z + rad * np.exp(2j * np.pi * rng.uniform(size=4 * per))
            got.extend(z[dom.contains(z)].tolist())
        pts.append(np.array(got[:per]))
    return np.concatenate(pts)


def run_one(args):
    import make_references as MR
    from chladni.mps import MPSEigensolver
    case, window, variant = args
    dom = MR.polygon(case["vertices"])
    k = float(np.sqrt(window[1]))
    t0 = time.perf_counter()
    try:
        b = build(dom, k, variant)
        n = len(b)
        if variant.endswith("graded"):
            kw = dict(bdry_quad=dom.bdry_quad(3 * n, grading="corner"),
                      int_pts=clustered_interior(dom, n, np.random.default_rng(0)))
        else:
            kw = dict(n_bdry=3 * n, n_int=n)
        res = MPSEigensolver(dom, basis=b, prec=PREC, rng=0, reg="svd",
                             **kw).search_interval(*window)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], variant=variant, error=repr(e)[:160])
    lv = res.levels
    return dict(id=case["id"], variant=variant, n_cols=n, wall=time.perf_counter() - t0,
                declined=bool(res.diagnostics.declined), audit=res.diagnostics.count_audit.verdict,
                met=[bool(L.met(PREC)) for L in lv], margin=[float(L.margin) for L in lv],
                floor=[float(L.sigma) for L in lv])


def main(workers=8):
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    jobs = [(cases[i], refs[i]["window"], v) for i in CASES
            for v in {1: VARIANTS, 2: VARIANTS2, 3: VARIANTS3}[ROUND]]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)
    for i in CASES:
        dom = MR.polygon(cases[i]["vertices"])
        sing = sum(not c.is_reflection_closed for c in dom.corners)
        print(f"\n{i}: {len(dom.corners)} corners ({sing} singular), angles/pi "
              f"{[round(c.omega / np.pi, 3) for c in dom.corners]}")
        for r in runs:
            if r["id"] != i:
                continue
            if "error" in r:
                print(f"  {r['variant']:12s} ERROR {r['error']}")
                continue
            m = np.array(r["margin"]) if r["margin"] else np.array([np.nan])
            print(f"  {r['variant']:12s} cols {r['n_cols']:4d} {r['wall']:6.1f}s  met "
                  f"{np.mean(r['met']) if r['met'] else 0:5.0%}  margin min {np.nanmin(m):8.2g} "
                  f"med {np.nanmedian(m):8.2g}  floor med "
                  f"{np.median(r['floor']) if r['floor'] else np.nan:8.1e}  audit {r['audit']}"
                  + ("  DECLINED" if r["declined"] else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
