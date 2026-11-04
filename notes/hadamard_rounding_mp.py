"""The rounding part of the Hadamard gradient's error, measured exactly with mpmath.  (2026-11-03)

For a solved level, the Fourier-Bessel radial factors J_mu(kr) and k J_mu'(kr) are recomputed in
mpmath at every Hadamard node (the angular and geometric factors are elementary and kept in
double), u_n = sum_j c_j d_n psi_j is summed in extended precision, and the Hadamard integrals
are formed from both. The SAME coefficient vector and quadrature go into both, so the
difference is precisely sources (1) per-term derivative evaluation and (2) cancellation, and
nothing else: no approximation error, no quadrature error.

FB-only bases (rung 0) are handled; a plan with sources or plane waves is refused.

Run:  PYTHONPATH=src python notes/hadamard_rounding_mp.py
"""
import sys
import json
import warnings

import mpmath as mp
import numpy as np

sys.path.insert(0, "notes")
from hadamard_accuracy import SHAPES, velocity, identities  # noqa: E402
from chladni.eigenproblem import Eigenproblem  # noqa: E402
from chladni.geometry import Polygon  # noqa: E402
from chladni.helmholtz.fourier_bessel import FourierBessel  # noqa: E402
from chladni.integrals import integral_quad  # noqa: E402

warnings.filterwarnings("ignore")
mp.mp.dps = 34


def fields_at(q, dom, V):
    off = np.cumsum([0] + [len(b.segments) for b in dom.boundaries])
    curve = off[np.asarray(q.bdry_idx, int)] + np.asarray(q.seg_idx, int)
    vals = None
    for cv in np.unique(curve):
        mask = curve == cv
        v = np.atleast_2d(V(int(cv), np.asarray(q.tau)[mask]))
        if vals is None:
            vals = np.zeros((v.shape[0], len(curve)))
        vals[:, mask] = v
    return vals


def dn_columns_mp(basis, q, k):
    """(n_nodes, n_cols) normal derivatives, the Bessel factors in mpmath, as float64 entries,
    plus the same in double from chladni."""
    samp = basis.at(q)
    _, G = samp.values_and_grad(k)
    nrm = q.normals
    dn_dbl = G.real * nrm.real[:, None] + G.imag * nrm.imag[:, None]
    cols = []
    for fam, part in zip(basis.families, samp._parts):
        if not isinstance(fam, FourierBessel):
            raise NotImplementedError(f"{type(fam).__name__} is not handled here")
        mu = fam.orders
        ls = fam.log_scale(k)
        out = np.empty((len(q.pts), len(mu)))
        for j, (m_, l_) in enumerate(zip(mu, ls)):
            s = mp.exp(-mp.mpf(float(l_)))
            for a, r in enumerate(part.r):
                x = mp.mpf(float(k)) * mp.mpf(float(r))
                rad = mp.besselj(float(m_), x) * s
                drad = mp.mpf(float(k)) * mp.besselj(float(m_), x, 1) * s
                g = (complex(drad) * part._ang[a, j] * part._grad_r[a]
                     + complex(rad) * part._dang[a, j] * part._grad_phi[a])
                out[a, j] = g.real * nrm[a].real + g.imag * nrm[a].imag
        cols.append(out)
    return dn_dbl, np.hstack(cols)


def run(cases=(("L-shape", 1e-12, [3, 8]), ("hexagon (irregular)", 1e-12, [9]))):
    rows = []
    for name, prec, levels in cases:
        z = np.asarray(SHAPES[name], dtype=complex)
        dom = Polygon(z, bc="dir")
        rec = Eigenproblem(dom).solve(10, prec=prec)
        print(f"== {name} prec {prec:.0e}: rung {rec.plan.rung}, n_cols {rec.plan.n_cols}, "
              f"sources {rec.plan.n_sources}", flush=True)
        V = velocity(dom, z)
        for i in levels:
            L = rec.spectrum.levels[i]
            lam = float(L.lam)
            k = np.sqrt(lam)
            ef = rec.solver.eigenfunctions(L, prec=prec)
            c = ef.coef[:, 0]
            q = integral_quad(dom, lam, prec=prec, family="integer").quad
            try:
                dn_dbl, dn_mp = dn_columns_mp(rec.solver._pencil.basis, q, k)
            except NotImplementedError as e:
                print(f"  level {i}: skipped, {e}")
                continue
            un_dbl = dn_dbl @ c
            cm = [mp.mpf(float(x)) for x in c]
            un_mp = np.array([float(mp.fsum(mp.mpf(float(a)) * b for a, b in zip(row, cm)))
                              for row in dn_mp])
            vals = fields_at(q, dom, V)
            w = q.wts
            H_dbl = -(vals * (un_dbl ** 2 * w)).sum(axis=1)
            H_mp = -(vals * (un_mp ** 2 * w)).sum(axis=1)
            g_had = rec.solver.hadamard(L, V)[:, 0, 0]
            scale = np.abs(H_mp).max()
            entry = np.abs(dn_dbl - dn_mp).max(axis=0) / np.maximum(np.abs(dn_mp).max(axis=0),
                                                                     1e-300)
            row = dict(shape=name, level=i, lam=lam, n_nodes=int(len(w)),
                       entry_rel_err_max=float(entry.max()),
                       un_rel_err=float(np.abs(un_dbl - un_mp).max() / np.abs(un_mp).max()),
                       grad_rounding=float(np.abs(H_dbl - H_mp).max() / scale),
                       sanity_vs_hadamard=float(np.abs(H_dbl - g_had).max() / scale),
                       identity_dbl=identities(z, lam, H_dbl),
                       identity_mp=identities(z, lam, H_mp))
            rows.append(row)
            print(f"  level {i} lam {lam:.6f}: nodes {len(w)}, worst column d_n entry rel err "
                  f"{row['entry_rel_err_max']:.1e}, u_n rel err {row['un_rel_err']:.1e}, "
                  f"GRADIENT ROUNDING {row['grad_rounding']:.1e} (sanity "
                  f"{row['sanity_vs_hadamard']:.1e})", flush=True)
            for tag in ("identity_dbl", "identity_mp"):
                r = row[tag]
                print(f"    {tag}: tx {r['tx'] * len(z):.1e} ty {r['ty'] * len(z):.1e} "
                      f"rot {r['rot']:.1e} dil {r['dil']:.1e}", flush=True)
    return rows


if __name__ == "__main__":
    rows = run()
    with open("notes/hadamard_rounding_mp.json", "w") as f:
        json.dump(rows, f, indent=1)
