"""Does a Hadamard matrix's error track its level's Moler-Payne eps?  (2026-11-03)

`HadamardReport.attained` bounds the rule and the normalization, not the eigenfunction itself.
`u_n` from an MPS candidate is wrong to roughly first order in its boundary residual, whose
relative size is the Moler-Payne `eps`, so the audit (`accuracy_audit_2026-11-03.md`, 1.9) asked
whether `attained` should include it. The exact identities (translation, rotation: H = 0;
dilation: eig H = -2 lam) give H's true error with no reference solve; set it beside `eps`.

usage: python notes/hadamard_vs_mp_eps.py
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))


def cases():
    from test_hadamard import IDENTITY_CASES
    out = [(name, verts, p) for name, verts, _ in IDENTITY_CASES for p in (1e-10, 1e-12)]
    hard = json.load(open(os.path.join(HERE, "hadamard_hard_shapes_cases.json")))["cases"]
    for c in hard[:3]:
        out.append((c["source"], np.array([complex(*z) for z in c["verts"]]), 1e-10))
    return out


def mp_eps(solver, level, ef):
    from chladni.eigensolver import lam_to_k
    from chladni.integrals import cauchy, certify
    p = solver._pencil
    iq = solver._integral_quad(float(level.lam), solver.prec)
    k = lam_to_k(level.lam)
    out = []
    for j in range(ef.coef.shape[1]):
        fc = ef.coef[:, j]
        cd = cauchy.cauchy_data(p.basis, fc, level.lam, iq.quad)

        def evaluate(z, fc=fc):
            V = p.basis.values(k, z)
            return V @ fc, np.abs(V) @ np.abs(fc)
        mp = certify.moler_payne(cd, evaluate, solver.domain, certify.x0_probes(solver.domain, iq.x0))
        out.append(float(np.max(mp.eps)))
    return max(out)


def main():
    warnings.simplefilter("ignore")
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    sys.path.insert(0, HERE)
    from hadamard_hard_shapes import fields
    rows = []
    for name, verts, prec in cases():
        dom = Polygon(np.asarray(verts), bc="dir")
        try:
            spec = Eigenproblem(dom).solve(5, prec=prec)
        except ValueError as e:                    # e.g. no source offset fits: a row, not a stop
            rows.append(dict(case=name, prec=prec, error=str(e)[:160]))
            print(f"{name[:26]:26s} {prec:.0e} solve failed: {str(e)[:80]}", flush=True)
            continue
        V = fields(spec.solver.domain, np.asarray(verts))
        for i, L in enumerate(spec.spectrum.levels):
            try:
                H, rep = spec.hadamard(i, V, report=True)
            except ValueError as e:
                rows.append(dict(case=name, prec=prec, i=i, refused=str(e)[:100]))
                continue
            lam = float(L.lam)
            ident = max(max(np.abs(H[f]).max() for f in range(3)) / lam,
                        np.abs(np.linalg.eigvalsh(H[3]) + 2 * lam).max() / lam)
            ef = spec.solver.eigenfunctions(L, prec=prec)
            rows.append(dict(case=name, prec=prec, i=i, lam=lam, ident=float(ident),
                             attained=float(rep.attained), mp_eps=mp_eps(spec.solver, L, ef)))
            r = rows[-1]
            print(f"{name[:26]:26s} {prec:.0e} {i} ident {r['ident']:.1e}  attained "
                  f"{r['attained']:.1e}  mp_eps {r['mp_eps']:.1e}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "hadamard_vs_mp_eps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
