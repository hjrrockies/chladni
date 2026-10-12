"""Phase 5 step 0: which recipes should the planner's ladder hold, and in what order?  (2026-10-11)

Candidate bases on the douse benchmark (41 cases, reference windows, prec 1e-10 and 1e-8):

    F1.0  F1.5  F2.0     Fourier-Bessel at EVERY corner, lappy's per-corner sizing x scale
                         (`make_references.basis_for`'s rule: n = s*(k r + 2 (k r)^(1/3) + 3)/alpha)
    B                    recipe B as built: F1.2, sharp corners (omega < pi/2) halved, plus
                         fundamental-solution sources at 3.6 per boundary wavelength
    F1.5+S3              F1.5 plus sources at 3 per wavelength
    F1.0+S2  F1.0+S3     F1.0 plus sources at 2 or 3 per wavelength

Sources sit at 5% of the diameter outside the boundary, falling back to 2% then 1% on a thin
domain, as recipe B does. Collocation: 3 boundary points and 1 interior point per column.

Per solve: every level's value, multiplicity, margin, `met`, bound and count flags, the count
audit, columns and wall time. The report scores each recipe on:
  - the fraction of levels met (below each case's trusted top);
  - count outcomes by Phase 3's resolution groups (`notes/phase3_rescore.py`);
  - BIAS: a met level whose value is off a TIGHT reference value by more than its own bound;
  - PASS: every level met, the audit not short/excess, nothing silent -- what the planner accepts;
  - columns and wall time;
and orders candidate ladders by expected cost to the first passing rung.

    .venv/bin/python notes/planner_recipes.py run [WORKERS]
    .venv/bin/python notes/planner_recipes.py report
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

OUT = os.path.join(HERE, "planner_recipes.json")
PRECS = (1e-10, 1e-8)
#: name -> (FB scale, halve sharp corners, sources per wavelength[, sizing rule])
#: Rule "lappy" (default): orders up to k r + 2 (k r)^(1/3) + 3. Rule "airy": up to
#: k r + t(prec) (k r)^(1/3) + 3 with t = 2^(-1/3) (1.5 ln(1/prec))^(2/3) -- where J_nu(k r) has
#: decayed to `prec` past its turning point (Airy asymptotics); t = 8.4 at 1e-10, 7.2 at 1e-8.
#: The 2026-10-12 additions (`notes/hard_cases.md`: the hard douse shapes need ~4x lappy's size).
RECIPES = {
    "F1.0": (1.0, False, 0.0),
    "F1.5": (1.5, False, 0.0),
    "F2.0": (2.0, False, 0.0),
    "B": (1.2, True, 3.6),
    "F1.5+S3": (1.5, False, 3.0),
    "F1.0+S2": (1.0, False, 2.0),
    "F1.0+S3": (1.0, False, 3.0),
    "F2.5": (2.5, False, 0.0),
    "F3.0": (3.0, False, 0.0),
    "F4.0": (4.0, False, 0.0),
    "P1": (1.0, False, 0.0, "airy"),
    "P2": (2.0, False, 0.0, "airy"),
}


def basis(dom, k, scale, halve, density, rule="lappy", prec=1e-10):
    from chladni.helmholtz import FourierBessel, Fundamental
    zs = np.array([c.z for c in dom.corners])
    t = 2.0 if rule == "lappy" else 2 ** (-1 / 3) * (1.5 * np.log(1 / prec)) ** (2 / 3)
    b = None
    for c in dom.corners:
        r = float(np.max(np.abs(zs - c.z)))
        n = int(np.ceil(scale * (k * r + t * (k * r) ** (1 / 3) + 3) / (np.pi / c.omega)))
        if halve and c.omega < np.pi / 2:
            n //= 2
        f = FourierBessel.at_corner(c, n=max(n, 3), domain=dom)
        b = f if b is None else b + f
    if density > 0:
        nsrc = int(np.ceil(density * k * dom.perimeter / (2 * np.pi)))
        for frac in (0.05, 0.02, 0.01):
            try:
                b = b + Fundamental.offset(dom.bdry_quad(nsrc), frac * dom.diameter, domain=dom)
                break
            except Exception:  # noqa: BLE001 -- sources inside a thin domain: move closer
                continue
    return b


def run_one(args):
    import make_references as MR
    from chladni.mps import MPSEigensolver
    case, window, recipe, prec = args
    lo, hi = window
    dom = MR.polygon(case["vertices"])
    t0 = time.perf_counter()
    try:
        spec = RECIPES[recipe]
        b = basis(dom, np.sqrt(hi), *spec[:3], *(spec[3:] or ("lappy",)), prec=prec)
        n = len(b)
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=prec, rng=0, reg="svd")
        res = s.search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], recipe=recipe, prec=prec, error=repr(e)[:200])
    d = res.diagnostics
    return dict(id=case["id"], recipe=recipe, prec=prec, n_cols=n, wall=time.perf_counter() - t0,
                declined=bool(d.declined), audit=d.count_audit.verdict,
                n_uncertain=int(d.n_uncertain),
                levels=[dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                             delta_res=float(L.delta_res), mp_bound=float(L.mp_bound),
                             margin=float(L.margin), met=bool(L.met(prec)),
                             count_flag=bool(L.count_uncertain or L.grouping_ambiguous))
                        for L in res.levels])


def run(workers=8, only=None):
    """`only`: run just these recipes and MERGE them into the saved runs."""
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = json.load(open(os.path.join(ROOT, "benchmarks", "douse", "references.json")))["cases"]
    todo = [rc for rc in RECIPES if only is None or rc in only]
    jobs = [(cases[r["id"]], r["window"], rc, p) for r in refs for rc in todo for p in PRECS]
    # largest bases first, so the pool does not end on a straggler
    jobs.sort(key=lambda j: -(RECIPES[j[2]][0] + RECIPES[j[2]][2]))
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    if only is not None and os.path.exists(OUT):
        old = [r for r in json.load(open(OUT))["runs"] if r["recipe"] not in todo]
        runs = old + runs
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)


def score(r, ref):
    """Per solve: met fraction (below the trusted top), Phase 3 outcome, bias count, pass."""
    import phase3_rescore as P3
    trusted, top = P3.trusted_and_top(ref)
    if not trusted or "levels" not in r:
        return None
    lv = [L for L in r["levels"] if L["lam"] < top]
    met = [L["met"] for L in lv]
    outcome, drops = P3.new_score(r, ref)
    # bias: a met level off a tight reference by more than its own bound
    tl = np.array([T["lam"] for T in trusted])
    tb = np.array([T["bound"] if np.isfinite(T.get("bound", np.nan)) else np.inf for T in trusted])
    bias = 0
    for L in lv:
        if not L["met"]:
            continue
        j = int(np.argmin(np.abs(tl - L["lam"])))
        err = abs(tl[j] - L["lam"])
        if tb[j] <= 0.1 * L["bound"] and L["bound"] < err <= 1e-4 * L["lam"]:
            bias += 1
    passed = (bool(lv) and all(met) and r["audit"] not in ("short", "excess")
              and outcome not in ("SILENT", "DROP") and not r["n_uncertain"] and not r["declined"])
    low = sum(1 for L in lv if np.isfinite(L["margin"]) and L["margin"] < 20)
    return dict(met=float(np.mean(met)) if met else 0.0, n_levels=len(lv), outcome=outcome,
                drops=drops, bias=bias, passed=passed, low_margin=low)


def report():
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    runs = [r for r in json.load(open(OUT))["runs"]]
    for p in PRECS:
        print(f"\n=== prec {p:g} ===")
        print(f"{'recipe':9s} {'cols med':>8s} {'wall med':>8s} {'met':>6s} {'pass':>9s} "
              f"{'SILENT':>6s} {'DROP':>5s} {'bias':>5s} {'margin<20':>9s} {'errors':>6s}")
        table = {}
        for rc in RECIPES:
            rr = [r for r in runs if r["recipe"] == rc and r["prec"] == p]
            sc = {r["id"]: score(r, refs[r["id"]]) for r in rr if "error" not in r}
            sc = {k: v for k, v in sc.items() if v is not None}
            table[rc] = (rr, sc)
            ok = [r for r in rr if "error" not in r]
            lv = sum(v["n_levels"] for v in sc.values())
            met = sum(v["met"] * v["n_levels"] for v in sc.values()) / max(lv, 1)
            print(f"{rc:9s} {np.median([r['n_cols'] for r in ok]):8.0f} "
                  f"{np.median([r['wall'] for r in ok]):7.1f}s {met:6.1%} "
                  f"{sum(v['passed'] for v in sc.values()):4d}/{len(sc):<4d} "
                  f"{sum(v['outcome'] == 'SILENT' for v in sc.values()):6d} "
                  f"{sum(v['outcome'] == 'DROP' for v in sc.values()):5d} "
                  f"{sum(v['bias'] for v in sc.values()):5d} "
                  f"{sum(v['low_margin'] for v in sc.values()):9d} {len(rr) - len(ok):6d}")
        # ladders: for an ordering, the expected wall to the first passing rung (failing all = all paid)
        print("\n  ladders (cases scored on every rung): expected wall to the first pass, cases never passing")
        ids = set.intersection(*(set(table[rc][1]) for rc in RECIPES))
        walls = {rc: {r["id"]: r["wall"] for r in table[rc][0] if "error" not in r} for rc in RECIPES}
        for ladder in (("F1.0", "F1.5", "F2.0"), ("F1.5", "F2.5", "F4.0"),
                       ("F2.0", "F3.0", "F4.0"), ("F2.0", "F4.0"), ("F1.5", "F3.0"),
                       ("F4.0",), ("P1", "P2", "F4.0"), ("P2", "F4.0"),
                       ("F1.5", "B", "F1.5+S3")):
            cost, never, rung_hist = 0.0, 0, [0] * len(ladder)
            for i in ids:
                c = 0.0
                for j, rc in enumerate(ladder):
                    c += walls[rc][i]
                    if table[rc][1][i]["passed"]:
                        rung_hist[j] += 1
                        break
                else:
                    never += 1
                cost += c
            print(f"    {' > '.join(ladder):28s} mean {cost / len(ids):6.1f}s  first pass at rung "
                  f"{rung_hist}  never {never}/{len(ids)}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
            sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    report()
