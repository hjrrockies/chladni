"""Data behind the multiplicity-estimation figures (2026-10-07).

Samples the tension spectrum sigma_1..sigma_8 around the cases discussed in
`notes/session_2026-10-07.md`, `notes/double_10pi2.py` and `notes/spike_x8.py`, and records what
stage 3 saw at each. Writes one JSON file the figures page embeds:

    .venv/bin/python notes/multiplicity_visuals_data.py OUT.json        # ~3 min
"""
import json
import os
import sys
import warnings

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks/douse")
warnings.simplefilter("ignore")

from chladni.geometry import L_shape, disk, rect  # noqa: E402
from chladni.helmholtz import PlaneWave  # noqa: E402
from chladni.mps import MPSEigensolver  # noqa: E402
from chladni.reference import disk_spectrum  # noqa: E402
import chladni.mps.search as S  # noqa: E402
from hard_roster import recipe_basis  # noqa: E402
import make_references as MR  # noqa: E402

K = 6  # branches recorded


def tens(s, x):
    return [float(v) for v in np.asarray(s._pencil.tensions(float(x)))[:K]]


def sweep(s, center, offsets):
    return [dict(d=float(d), s=tens(s, center + d)) for d in offsets]


def pw_solver(dom, lam_hi, prec, pad=1.3):
    nw = int(pad * PlaneWave.n_useful(np.sqrt(lam_hi) * dom.diameter))
    b = PlaneWave.uniform(nw, radius=dom.diameter)
    return MPSEigensolver(dom, basis=b, n_bdry=3 * len(b), n_int=len(b), prec=prec)


out = {}

# 1. The clean case: the disk's double j_{1,1}^2, and the simple j_{0,2}^2 near it. Linear window.
dk = disk(1.0)
s = pw_solver(dk, 40.0, 1e-12)
dbl = float(sorted(set(np.round(disk_spectrum(14, 15.5), 9)))[0])
grid = np.linspace(11.0, 34.0, 461)
out["disk"] = dict(center=0.0, double=dbl, truth=[float(x) for x in disk_spectrum(11, 34)],
                   rows=[dict(d=float(x), s=tens(s, x)) for x in grid])

# 2. A close DISTINCT pair: rect(1+delta, 1), modes (1,2) and (2,1), split ~ 6 pi^2 delta.
delta = 2e-5
rc = rect(1 + delta, 1.0)
lam12 = np.pi ** 2 * (1 / (1 + delta) ** 2 + 4)
lam21 = np.pi ** 2 * (4 / (1 + delta) ** 2 + 1)
s = pw_solver(rc, 60.0, 1e-12)
mid = 0.5 * (lam12 + lam21)
gap = abs(lam12 - lam21)
off = np.linspace(-4 * gap, 4 * gap, 401)
out["pair"] = dict(lam_a=float(min(lam12, lam21)), lam_b=float(max(lam12, lam21)), mid=float(mid),
                   gap=float(gap), rows=sweep(s, mid, off))

# 3. The floor: L-shape 10 pi^2 on the 80-term recipe, log-spaced distances, both sides.
ls = L_shape()
b = recipe_basis(ls, 80)
sl = MPSEigensolver(ls, basis=b, n_bdry=3 * len(b), n_int=len(b), prec=1e-12)
t = 10 * np.pi ** 2
ds = np.logspace(-15, -1, 57)
out["lshape_floor"] = dict(center=float(t), plus=sweep(sl, t, ds), minus=sweep(sl, t, -ds))
# ... and the dense view of the floor itself, where 1e-13 searches found phantom roots.
fine = np.linspace(-8e-12, 8e-12, 321)
out["lshape_fine"] = dict(center=float(t), rows=sweep(sl, t, fine),
                          located_1e13=[-4.7e-12, -2.9e-12, -8.4e-13, 1.1e-13],
                          located_1e12=[-4.5e-13])

# What stage 3 saw at 10 pi^2 (prec 1e-12, window [5, 100]): the reach read and the wide bracket.
seen = {}
o_mult = S.multiplicity


def spy_mult(tns, x, **kw):
    m = o_mult(tns, x, **kw)
    if abs(x - t) < 1e-6 and kw.get("rule") == "reach":
        seen["reach"] = dict(x=float(x), bracket=[float(v) for v in kw["bracket"]],
                             c=float(kw["c"]),
                             branches=[dict(j=br.index, sigma=br.sigma, d_est=br.d_est,
                                            verdict=br.verdict) for br in m.branches])
    return m


o_c2 = S._c2_holds


def spy_c2(tns, x, wb, j):
    r = o_c2(tns, x, wb, j)
    if abs(x - t) < 1e-6:
        seen.setdefault("c2", []).append(dict(j=j, wb=[float(v) for v in wb], at=tens(sl, x)[j],
                                              left=tens(sl, wb[0])[j], right=tens(sl, wb[1])[j],
                                              holds=r))
    return r


S.multiplicity, S._c2_holds = spy_mult, spy_c2
sl.search_interval(5.0, 100.0)
S.multiplicity, S._c2_holds = o_mult, o_c2
out["lshape_stage3"] = seen

# 4. The noisy floor: douse's spike triangle (T1.3), run B, at 807.0076.
case = {c["id"]: c for c in MR.load_cases()}["T1.3"]
dom = MR.polygon(case["vertices"])
win = (124.32878268649651, 966.949607249548)
bb = MR.basis_for(dom, "B", np.sqrt(win[1]))
ss = MPSEigensolver(dom, basis=bb, n_bdry=3 * len(bb), n_int=len(bb), prec=1e-12)
x0 = 807.0076084482336
cache = {}
o_slope = S._slope_from_cache


def spy_slope(lams, vs, lam, floor=0.0):
    c = o_slope(lams, vs, lam, floor)
    if abs(lam - x0) < 1e-3 and "samples" not in cache:
        o = np.argsort(np.abs(lams - lam))[:10]
        cache["samples"] = sorted([dict(d=float(lams[i] - lam), s=float(vs[i])) for i in o],
                                  key=lambda r: r["d"])
        cache["blind"] = float(S._cone_slope(lams, vs, lam, np.ones(np.shape(vs), bool), 0.0))
        cache["aware"] = float(S._cone_slope(lams, vs, lam,
                                             vs > S._SLOPE_FLOOR_MARGIN * floor, floor))
        cache["floor"] = float(floor)
        cache["margin"] = float(S._SLOPE_FLOOR_MARGIN)
        cache["guard"] = float(S._SLOPE_FLOOR_GUARD)
        cache["used"] = float(c)
    return c


S._slope_from_cache = spy_slope
ss.search_interval(*win)
S._slope_from_cache = o_slope
dsp = np.logspace(-10, 0.5, 43)
out["spike"] = dict(center=x0, plus=sweep(ss, x0, dsp), minus=sweep(ss, x0, -dsp), cache=cache,
                    true_slope=float(np.asarray(ss._pencil.tensions(x0 + 1e-2))[0] / 1e-2))

# 5. T1.2's near-double (two eigenvalues ~1e-9 apart at 68.8536737), run C's basis.
c12 = {c["id"]: c for c in MR.load_cases()}["T1.2"]
d12 = MR.polygon(c12["vertices"])
w12 = (10.787616554650096, 183.79944350430173)
bc = MR.basis_for(d12, "C", np.sqrt(w12[1]))
s12 = MPSEigensolver(d12, basis=bc, n_bdry=3 * len(bc), n_int=len(bc), prec=1e-12)
c0 = 68.8536736647
out["t12"] = dict(center=c0, rows=sweep(s12, c0, np.linspace(-6e-9, 6e-9, 241)),
                  plus=sweep(s12, c0, np.logspace(-13, -2, 45)),
                  minus=sweep(s12, c0, -np.logspace(-13, -2, 45)))

json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
