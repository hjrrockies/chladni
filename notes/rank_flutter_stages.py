"""Which stage does the rank flutter hurt? (2026-10-27). Apex (0.1, 0.25), shipped rung 3.
In each lattice cell holding lam_1, lam_8, lam_15, `search_interval` with the threshold rank against
the rank frozen at the cell's single-spectrum knee (`notes/stable_rank_free.py`, K 4, rho 1.25):
declined or not; evaluations by stage (seed/cover = search, refine, multiplicity, other); per level the
value's error against the references, sigma at the well, delta_res, margin, multiplicity and its
deciding rule, and the Moler-Payne bound."""
import numpy as np
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec = 1e-10
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, nb, ni = plan.realize(dom)
mk = lambda **kw: MPSEigensolver(dom, basis=basis, n_bdry=nb, n_int=ni, prec=prec, rng=0, **kw)
cases = [("lam_1", 285.152, 289.904, 288.9643854029, 98),
         ("lam_8", 1283.297, 1304.685, 1293.9127901997, 96),
         ("lam_15", 2214.218, 2251.122, 2220.6128169992, 101)]
for name, a, b, ref, rk in cases:
    for label, s in (("threshold", mk()), (f"knee {rk}", mk(rank=rk))):
        res = s.search_interval(a, b, on_shortfall="flag")
        d = res.diagnostics
        ev = (f"evals: search {d.search_evals}, refine {d.refine_evals}, mult {d.multiplicity_evals}, "
              f"other {d.evaluations - d.search_evals - d.refine_evals - d.multiplicity_evals}")
        if d.declined:
            print(f"{name} {label:10s} DECLINED in the seed ({ev}): {d.decline_reason[:70]}", flush=True)
            continue
        for L in res.levels:
            print(f"{name} {label:10s} {ev} | lam err {abs(L.lam - ref) / ref:.1e} sigma {L.sigma:.2e} "
                  f"delta_res/lam {L.delta_res / L.lam:.1e} margin {L.margin:.3g} mult {L.mult} "
                  f"({L.count_decided_by or 'no question'}, uncertain {L.count_uncertain}) "
                  f"MP/lam {L.mp_bound / L.lam:.1e}", flush=True)
