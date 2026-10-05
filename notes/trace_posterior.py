"""Trace one cell of the posterior-penalty tau sweep, stage by stage (`notes/posterior_penalty.md`,
"The 1e-12 singles, traced"). `trace_undercount.py` with its cell builder swapped for the sweep's
padded cluster case and posterior objective. Run from the repo root:

    .venv/bin/python notes/trace_posterior.py PAD C N BAND        # e.g. 1.6 30 65 0.3
"""
import sys; sys.path.insert(0,"notes"); sys.path.insert(0,"tests")
import warnings; warnings.simplefilter("ignore")
import numpy as np
import trace_undercount as T, posterior_tau_sweep as SW
from posterior_penalty import posterior_tensions
import bracket_score as bs
from chladni.mps.objective import Objective
pad, C, N, band = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
prec = 1e-12
def cell(prec_, C_, N_, band_):
    case = SW._cluster_case(N_, T.LEVELS[N_], band_, prec_, pad)
    q = SW.pencil(case.pencil, prec_, C)
    base = bs.objective_for(case, prec_)
    obj = Objective(lambda x: posterior_tensions(q, x), base.lo, base.hi, prec=prec_, variable=case.variable,
                    expected_count=getattr(base, "expected_count", None))
    return case, obj, []
T.cell = cell
T.main(prec, C, N, band)
