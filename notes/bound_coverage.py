"""Does `Level.bound = min(delta_res, mp_bound)` cover the true error?  (2026-10-07)

`notes/delta_res_undercover.py` found `delta_res` under the true error by up to 4e6x on ONE
starved L-shape basis at `prec = 1e-12`. This grid asks where that happens: basis adequacy
(starved -> generous) x request (`prec` 1e-8 on the penalty path, 1e-10 .. 1e-13 on the
threshold) x domains with closed-form or 14-digit truth -- the L-shape (reentrant corner,
table), the 3pi/2 sector (reentrant apex, closed form), the equilateral triangle and the disk
(closed form, degenerate).

Every row is one returned level: its value, `sigma`, `delta_res`, `mp_bound`, the true error
(distance to the nearest true eigenvalue), and whether each bar covers it. Writes
`notes/bound_coverage.jsonl`; `notes/bound_coverage.md` reads it.

    .venv/bin/python notes/bound_coverage.py            # ~10 min on 8 processes
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

OUT = os.path.join(os.path.dirname(__file__), "bound_coverage.jsonl")
PRECS = (1e-8, 1e-10, 1e-12, 1e-13)


def _domain(label):
    from chladni.geometry import L_shape, disk, disk_sector, eq_tri
    return {"L_shape": L_shape, "sector270": lambda: disk_sector(1.0, 1.5 * np.pi),
            "eq_tri": eq_tri, "disk": disk}[label]()


def _truth(label, lo, hi):
    from chladni import reference as R
    if label == "L_shape":
        t = np.asarray(R.L_shape_eigs(25), dtype=float)
    elif label == "sector270":
        t = np.asarray(R.sector_spectrum(lo, hi, omega=1.5 * np.pi), dtype=float)
    elif label == "eq_tri":
        t = np.asarray(R.eq_tri_spectrum(lo, hi), dtype=float)
    else:
        t = np.asarray(R.disk_spectrum(lo, hi), dtype=float)
    return np.sort(t)


def _basis(label, dom, kind, n, lam_hi):
    from chladni.helmholtz import FourierBessel, Fundamental, PlaneWave
    if kind == "fb_reentrant":        # FB at every reentrant corner, n each
        b = None
        for c in dom.corners:
            if c.is_reentrant:
                f = FourierBessel.at_corner(c, n=n, domain=dom)
                b = f if b is None else b + f
        return b
    if kind == "fb_all":              # FB at every corner, n each
        b = None
        for c in dom.corners:
            f = FourierBessel.at_corner(c, n=n, domain=dom)
            b = f if b is None else b + f
        return b
    if kind == "pw":                  # plane waves, n = pad * n_useful
        nw = int(n * PlaneWave.n_useful(np.sqrt(lam_hi) * dom.diameter))
        return PlaneWave.uniform(nw, radius=dom.diameter)
    if kind == "mfs":                 # n fundamental solutions offset from the boundary
        return Fundamental.offset(dom.bdry_quad(n), 0.1 * dom.diameter, domain=dom)
    raise ValueError(kind)


# (domain, window, basis kind, size). Sizes chosen to span starved -> generous for each.
CASES = (
    [("L_shape", (5.0, 100.0), "fb_reentrant", n) for n in (10, 20, 40, 80)]
    + [("L_shape", (5.0, 100.0), "fb_all", n) for n in (8, 14, 24)]
    + [("sector270", (5.0, 100.0), "fb_reentrant", n) for n in (4, 8, 16)]
    + [("sector270", (5.0, 100.0), "mfs", n) for n in (40, 80, 160)]
    + [("eq_tri", (40.0, 300.0), "pw", p) for p in (0.5, 0.8, 1.3)]
    + [("disk", (5.0, 80.0), "pw", p) for p in (0.5, 0.8, 1.3)]
)


def run(case, prec):
    from chladni.helmholtz import PrecisionWarning
    from chladni.mps import MPSEigensolver
    warnings.simplefilter("ignore", PrecisionWarning)
    warnings.simplefilter("ignore", UserWarning)
    label, (lo, hi), kind, n = case
    dom = _domain(label)
    truth = _truth(label, lo, hi)
    basis = _basis(label, dom, kind, n, hi)
    nb = len(basis)
    t0 = time.perf_counter()
    try:
        s = MPSEigensolver(dom, basis=basis, n_bdry=3 * nb, n_int=nb, prec=prec)
        res = s.search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001 -- a crash is a row too
        return [dict(domain=label, kind=kind, n=n, n_cols=nb, prec=prec, error=repr(e))]
    wall = time.perf_counter() - t0
    inwin = truth[(truth > lo) & (truth < hi)]
    rows = [dict(row="summary", domain=label, kind=kind, n=n, n_cols=nb, prec=prec, wall=wall,
                 n_truth=int(len(inwin)), n_found=int(sum(x.mult for x in res.levels)),
                 attained=float(res.diagnostics.attained_prec))]
    for L in res.levels:
        err = float(np.min(np.abs(truth - L.lam)))
        rows.append(dict(
            row="level", domain=label, kind=kind, n=n, n_cols=nb, prec=prec, wall=wall,
            n_truth=int(len(inwin)), n_found=int(sum(x.mult for x in res.levels)),
            lam=float(L.lam), mult=int(L.mult), sigma=float(L.sigma),
            delta_res=float(L.delta_res), mp_bound=float(L.mp_bound), bound=float(L.bound),
            mp_eps=float(L.mp_eps), err=err, met=bool(L.met(prec)),
            dr_covers=bool(err <= L.delta_res), mp_covers=bool(err <= L.mp_bound),
            covers=bool(err <= L.bound)))
    return rows


def main():
    jobs = [(c, p) for c in CASES for p in PRECS]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with ProcessPoolExecutor(workers) as ex, open(OUT, "w") as fh:
        futs = [ex.submit(run, c, p) for c, p in jobs]
        for (c, p), f in zip(jobs, futs):
            rows = f.result()
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            bad = sum(not r.get("covers", True) for r in rows)
            sm = rows[0]
            print(f"{c[0]:10s} {c[2]:13s} n={c[3]:<5} prec={p:.0e}: "
                  f"{sm.get('n_found')}/{sm.get('n_truth')} found, {sm.get('wall', 0):.0f} s, "
                  f"{bad} uncovered", flush=True)


if __name__ == "__main__":
    main()
