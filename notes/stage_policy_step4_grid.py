"""Stage-policy step 4 gate (2026-10-27): the rectangle cluster grids, threshold vs stable rank.

The split-degeneracy clusters of `tests/test_cluster_totals.py` (`rect_suite.split_degeneracy_cases`)
at N*pi^2 for N in 50, 65, 125, 145 (cluster multiplicity 3, 4, 4, 4: ordered pairs with
m^2+n^2 = N; the TRUTH scored is every eigenvalue in the case's +-3% window, which at 125 and 145
holds other levels too),
prec 1e-6, 1e-8, 1e-10, the eleven bands. Each cell is solved on the SHIPPING pencil (no rtol knob,
as `MPSEigensolver` builds it) through `Objective.from_pencil`, twice: `solve(stable_rank=False)`
(the shipped path) and `solve(stable_rank=True)` (step 4). Unexercisable cells (the basis cannot
resolve the cluster's multiplicity, `basis_sufficiency.sufficient`) are skipped, as the tests do.
Verdicts as `test_cluster_totals`: right, FLAGGED (wrong but uncertain), SILENT (wrong, unflagged),
DECLINED. Cost: evaluations, and the stable path's rerank_evals.

usage: python notes/stage_policy_step4_grid.py N [out.json]
"""
import json, sys, time
sys.path.insert(0, "tests")
import numpy as np
import bracket_score as bs
import rect_suite as rs
from basis_sufficiency import sufficient
from chladni.mps import MPSPencil
from chladni.mps.objective import Objective
from chladni.mps.search import solve

N = int(sys.argv[1])
M = sum(1 for m in range(1, 20) for n in range(1, 20) if m * m + n * n == N)
BANDS = (0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.0, 4.0, 6.0)
rows = []
for prec in (1e-6, 1e-8, 1e-10):
    for band in BANDS:
        cases = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: M})
        if not cases:
            continue
        case = cases[0]
        truth = int(np.asarray(case.truth, dtype=float).size)   # every eigenvalue in the window
        if not sufficient(case, prec).multiplicity:
            rows.append(dict(N=N, prec=prec, band=band, skip=True)); continue
        row = dict(N=N, prec=prec, band=band, skip=False, truth=truth)
        for mode in (False, True):
            q = MPSPencil(case.pencil.bdry, case.pencil.interior, prec=prec)
            base = bs.objective_for(case, prec)
            obj = Objective.from_pencil(q, base.lo, base.hi, variable=case.variable, prec=prec,
                                        expected_count=base.expected_count, max_workers=1)
            t = time.time()
            res = solve(obj, prec=prec, stable_rank=mode)
            d = res.diagnostics
            total = sum(x.mult for x in res.levels)
            flagged = (not d.certified_counts or d.n_ambiguous > 0
                       or any(x.count_uncertain or x.grouping_ambiguous for x in res.levels))
            verdict = ("DECLINED" if d.declined else "right" if total == truth
                       else "FLAGGED" if flagged else "SILENT")
            row["stable" if mode else "threshold"] = dict(
                verdict=verdict, total=total, evals=d.evaluations, rerank=d.rerank_evals,
                reranked=d.n_reranked, refused=d.n_rerank_refused, wall=time.time() - t,
                worst_rel=max((x.bound / x.lam for x in res.levels), default=None))
        rows.append(row)
        th, st = row["threshold"], row["stable"]
        print(f"N{N} truth {truth} {prec:.0e} b{band:<4g} thr {th['verdict']:8s} {th['total']} ev {th['evals']:5d} | "
              f"stable {st['verdict']:8s} {st['total']} ev {st['evals']:5d} rerank {st['rerank']:3d} "
              f"({st['reranked']} roots, {st['refused']} refused)", flush=True)
json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else f"notes/stage_policy_step4_grid_N{N}.json", "w"),
          indent=1)
