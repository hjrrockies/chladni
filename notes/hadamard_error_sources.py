"""Where the Hadamard-gradient floor comes from: rounding in the basis, or approximation?  (2026-11-03)

Follows `notes/hadamard_accuracy.py`, which found gradient floors 2-6 orders above the
eigenvalue's on corner domains, untouched by refining the Hadamard quadrature. Three candidate
sources, from the basis:

  (1) the per-term normal derivatives d_n psi_j are evaluated less accurately than the values;
  (2) cancellation in u_n = sum_j c_j d_n psi_j: kappa(x) = sum |c_j d_n psi_j| / |u_n| per node,
      so rounding at relative level delta per term costs ~ kappa * delta;
  (3) the eigenfunction itself is a worse approximation where the Hadamard weight looks (near a
      moving corner) than the eigenvalue reveals.

Measured here: kappa at the Hadamard nodes and the rounding bound it implies on the integral
(tests 2, and 1 through the delta it would take), and the gradient's identity residuals on
RICHER bases at the same prec (tests 3: an approximation floor moves with the basis, a
rounding floor does not, or gets worse).

Run:  PYTHONPATH=src python notes/hadamard_error_sources.py
"""
import sys
import json
import warnings
from dataclasses import replace

import numpy as np

sys.path.insert(0, "notes")
from hadamard_accuracy import SHAPES, velocity, identities  # noqa: E402
from chladni.eigenproblem import Eigenproblem  # noqa: E402
from chladni.geometry import Polygon  # noqa: E402
from chladni.integrals import cauchy, integral_quad  # noqa: E402

warnings.filterwarnings("ignore")
EPS = np.finfo(float).eps

CASES = [("L-shape", 1e-12, [3, 8]), ("hexagon (irregular)", 1e-12, [7, 9]),
         ("flat triangle", 1e-10, [5, 8])]


def rounding_bound(solver, level, prec, V):
    """kappa summary and the rounding bound on each vertex partial, per unit delta.

    For H_f = -int u_n^2 V_f, a perturbation of u_n by at most delta * T(x), T = sum |c_j d_n
    psi_j|, moves H_f by at most 2 delta int |u_n| T |V_f|. Divided by max_f |H_f| it is the
    relative rounding error of the gradient per unit delta. The normalization (Gram) carries the
    same kind of term; it is reported separately on the eigenfunctions' own rule."""
    lam = float(level.lam)
    ef = solver.eigenfunctions(level, prec=prec)
    basis = solver._pencil.basis
    q = integral_quad(solver.domain, lam, prec=prec, family="integer").quad
    k = np.sqrt(lam)
    Vb, G = basis.at(q).values_and_grad(k)
    n = q.normals[:, None]
    dn = G.real * n.real + G.imag * n.imag
    c = ef.coef[:, 0]
    un = dn @ c
    T = np.abs(dn) @ np.abs(c)
    kappa = T / np.maximum(np.abs(un), 1e-300)
    w = np.asarray(q.wts)
    # the Hadamard velocity fields at these nodes, as hadamard() builds them
    segs_off = np.cumsum([0] + [len(b.segments) for b in solver.domain.boundaries])
    curve = segs_off[np.asarray(q.bdry_idx, int)] + np.asarray(q.seg_idx, int)
    vals = None
    for cv in np.unique(curve):
        mask = curve == cv
        v = np.atleast_2d(V(int(cv), np.asarray(q.tau)[mask]))
        if vals is None:
            vals = np.zeros((v.shape[0], len(curve)))
        vals[:, mask] = v
    H = -(vals * (un ** 2 * w)).sum(axis=1)
    dH = 2 * (np.abs(vals) * (np.abs(un) * T * w)).sum(axis=1)
    # where the cancellation sits: kappa at the node carrying the largest u_n^2 |V| mass
    mass = un ** 2 * np.abs(vals).max(axis=0) * w
    # the Gram's rule (eigenfunction normalization): same bound on int u_n^2 (x.n)-type terms
    cd = cauchy.cauchy_data(basis, ef.coef, lam, ef.rule.quad)
    kap_rule = (cd.terms[:, 0] / np.maximum(np.abs(cd.u[:, 0]), 1e-300))
    return dict(H_check=float(np.abs(H).max()), n_cols=int(len(c)), coef_norm=float(np.linalg.norm(c)),
                kappa_max=float(kappa.max()),
                kappa_massweighted=float((kappa * mass).sum() / mass.sum()),
                grad_round_per_delta=float(dH.max() / np.abs(H).max()),
                grad_round_at_eps=float(EPS * dH.max() / np.abs(H).max()),
                gram_rule_terms_over_interior=float(
                    cd.cancellation(np.ones(1), solver.domain.area)[0]),
                gram_residual=float(ef.gram_residual), _unused=float(kap_rule.max()))


def richer_plans(plan):
    """The base plan, its corner expansions x1.5 and x2, and the other ladder rungs."""
    from chladni.plan import plan_for  # noqa: F401  (rungs come through Eigenproblem below)
    out = [("base", plan)]
    for s in (1.5, 2.0):
        out.append((f"FB x{s}", replace(plan, corner_orders=tuple(
            int(np.ceil(s * o)) for o in plan.corner_orders))))
    return out


def run():
    rows = []
    for name, prec, levels in CASES:
        z = np.asarray(SHAPES[name], dtype=complex)
        dom = Polygon(z, bc="dir")
        base = Eigenproblem(dom).solve(10, prec=prec)
        V = velocity(dom, z)
        print(f"\n== {name}, prec {prec:.0e}: rung {base.plan.rung}, corner orders "
              f"{base.plan.corner_orders}, n_cols {base.plan.n_cols}", flush=True)
        variants = richer_plans(base.plan)
        from chladni.plan import plan_for
        for r in range(5):
            if r != base.plan.rung:
                try:
                    variants.append((f"rung {r}", plan_for(dom, 10, prec, r)))
                except Exception:                              # noqa: BLE001
                    pass
        for label, plan in variants:
            try:
                rec = base if label == "base" else Eigenproblem(dom).solve(10, prec=prec,
                                                                          plan=plan)
            except Exception as e:                             # noqa: BLE001
                print(f"  {label:9s} solve failed: {type(e).__name__}: {str(e)[:80]}")
                continue
            if rec.solver is None:
                print(f"  {label:9s} no solver ({rec.spectrum.flags[:1]})")
                continue
            sp = rec.spectrum
            for i in levels:
                # match the base level by value, the richer solve may index differently
                j = int(np.argmin(np.abs(np.asarray(sp.lam) - base.spectrum.lam[i])))
                L = sp.levels[j]
                if L.mult != 1:
                    continue
                try:
                    g = rec.solver.hadamard(L, V)[:, 0, 0]
                    rb = rounding_bound(rec.solver, L, prec, V)
                    if abs(rb.pop("H_check") - np.abs(g).max()) > 1e-6 * np.abs(g).max():
                        print("  !! my H disagrees with hadamard(): the bound is mis-scaled")
                except Exception as e:                         # noqa: BLE001
                    print(f"  {label:9s} lvl {i}: {type(e).__name__}: {str(e)[:80]}")
                    continue
                idn = identities(z, float(L.lam), g)
                worst_id = max(idn["tx"] * len(z), idn["ty"] * len(z), idn["rot"], idn["dil"])
                row = dict(shape=name, prec=prec, variant=label, level=i, lam=float(L.lam),
                           rel_bound=float(L.bound / L.lam),
                           identity=worst_id, grad=[float(x) for x in g], **idn, **rb)
                rows.append(row)
                print(f"  {label:9s} lvl {i} cols {plan.n_cols:4d} |c| {rb['coef_norm']:.1e}  "
                      f"lam-bound {row['rel_bound']:.1e}  gram {rb['gram_residual']:.1e}  "
                      f"kappa max {rb['kappa_max']:.1e} wtd {rb['kappa_massweighted']:.1e}  "
                      f"rounding@eps {rb['grad_round_at_eps']:.1e}  IDENTITY {worst_id:.1e}",
                      flush=True)
    return rows


if __name__ == "__main__":
    rows = run()
    with open("notes/hadamard_error_sources.json", "w") as f:
        json.dump(rows, f, indent=1)
