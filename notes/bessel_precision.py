"""Does Fourier-Bessel EVALUATION cap the floor on sharp-cornered douse shapes?  (2026-10-12)

`notes/hard_cases.md` part 2: on T2.2, T1.3 and T2.6-N3k5-L6 the tension floor stalls at
1.5-3e-7 whatever the basis size, about the size of the worst columns' own error bounds
(`Sampling.accuracy`). This tests that directly on T2.2. The basis is the 3x Fourier-Bessel one,
and the point sets are exactly the solver's (3n boundary, n interior, rng 0). At three reference
eigenvalues:

    float64   `chladni`'s collocation matrices (`MPSPencil.matrices`), equilibrated
    mpmath    the same columns with every J_mu(k r) evaluated to 30 digits, times the same
              angular factors, each column normalized in extended precision over the stacked
              pencil (raw high-order values underflow float64), then rounded to float64

Both go through the same regularization (`tensions`, truncation at sqrt(1e-10 eps)). If the
extended-precision floor is well below float64's, the cap is the evaluation of the Bessel
functions. If not, it is the basis or the geometry.

    .venv/bin/python notes/bessel_precision.py [WORKERS]
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "tests"), os.path.join(ROOT, "benchmarks", "douse"), HERE]
warnings.simplefilter("ignore")

CASE = "T2.2"
PREC = 1e-10
LAMS = (37.183126, 68.260292, 152.295845)     # simple, certified reference levels of T2.2
DPS = 30


def mp_block(args):
    """Extended-precision J_mu(k r) * angular factor for one family's sampling, unnormalized,
    as mpf strings (so the parent can normalize per column without underflow)."""
    import mpmath as mp
    mp.mp.dps = DPS
    orders, r, ang, k = args
    out = []
    for j, mu in enumerate(orders):
        col = []
        for i in range(len(r)):
            v = mp.besselj(mp.mpf(mu), mp.mpf(k) * mp.mpf(r[i])) * mp.mpf(ang[i, j])
            col.append(v)
        out.append([mp.nstr(v, DPS) for v in col])
    return out


def main(workers=8):
    import mpmath as mp
    import make_references as MR
    from hard_cases2 import build
    from chladni.mps import MPSPencil, equilibrate, tensions

    mp.mp.dps = DPS
    cases = {c["id"]: c for c in MR.load_cases()}
    ref = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}[CASE]
    dom = MR.polygon(cases[CASE]["vertices"])
    k_top = float(np.sqrt(ref["window"][1]))
    b, _ = build(dom, k_top, [3.0] * len(dom.corners))
    n = len(b)
    quad = dom.bdry_quad(3 * n)
    ipts = dom.interior_pts(n, rng=np.random.default_rng(0))
    pencil = MPSPencil.from_basis(b, quad, ipts, reg="svd", prec=PREC)
    bd, it = pencil.bdry, pencil.interior
    rtol = float(np.sqrt(PREC * np.finfo(float).eps))
    print(f"{CASE}: {n} columns, {len(quad.pts)} boundary + {n} interior points, k_top {k_top:.2f}")
    for lam in LAMS:
        k = float(np.sqrt(lam))
        A_B, A_I = pencil.matrices(lam)
        B, I, _ = equilibrate(A_B, A_I)
        s64 = tensions(B, I, reg="svd", rtol=rtol)[:3]
        # extended precision, family by family, in the basis's column order
        jobs = []
        for pb, pi in zip(bd._parts, it._parts):
            orders = np.asarray(pb.family.orders, float)
            jobs.append((orders, pb.r, pb._ang, k))
            jobs.append((orders, pi.r, pi._ang, k))
        with ProcessPoolExecutor(workers) as ex:
            blocks = list(ex.map(mp_block, jobs))
        cols_B, cols_I = [], []
        for fb, fi in zip(blocks[0::2], blocks[1::2]):
            for cb, ci in zip(fb, fi):
                vb = [mp.mpf(x) for x in cb]
                vi = [mp.mpf(x) for x in ci]
                nrm = mp.sqrt(mp.fsum(v * v for v in vb + vi))
                cols_B.append([float(v / nrm) if nrm else 0.0 for v in vb])
                cols_I.append([float(v / nrm) if nrm else 0.0 for v in vi])
        Bm, Im = np.array(cols_B).T, np.array(cols_I).T
        smp = tensions(Bm, Im, reg="svd", rtol=rtol)[:3]
        # column agreement (sign-insensitive)
        diff = np.minimum(np.abs(B - Bm).max(axis=0), np.abs(B + Bm).max(axis=0))
        diffI = np.minimum(np.abs(I - Im).max(axis=0), np.abs(I + Im).max(axis=0))
        worst = np.maximum(diff, diffI)
        print(f"\n  lam {lam}: sigma_1..3  float64 {np.array2string(s64, precision=3)}   "
              f"mpmath {np.array2string(smp, precision=3)}")
        print(f"    normalized-column disagreement: max {worst.max():.1e}, columns > 1e-10: "
              f"{int(np.sum(worst > 1e-10))}/{n}, > 1e-8: {int(np.sum(worst > 1e-8))}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
