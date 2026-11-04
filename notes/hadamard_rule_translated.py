import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "notes")
import numpy as np
from types import SimpleNamespace
from hadamard_exact_rule import hadamard_on
from hadamard_accuracy import velocity, identities
from chladni.eigenproblem import Eigenproblem
from chladni.geometry import Polygon

def rule(dom, corner, n_gl, n_panels, fix):
    x, w = np.polynomial.legendre.leggauss(n_gl); x, w = 0.5*(x+1), 0.5*w
    segs = [s for b in dom.boundaries for s in b.segments]
    P, N, W, TAU, CUR = [], [], [], [], []
    for ci, s in enumerate(segs):
        p0, p1 = complex(s.p0), complex(s.p1); Lg = abs(p1-p0); tan = (p1-p0)/Lg
        for k in range(n_panels):
            a, b = k/n_panels, (k+1)/n_panels
            u = a + (b-a)*x; wu = (b-a)*w
            at0, at1 = abs(p0-corner) < 1e-12, abs(p1-corner) < 1e-12
            if at0 or at1:
                d = u**3; dt = 3*u**2*wu
                pc, other = (p0, p1) if at0 else (p1, p0)
                pts = pc + d*(other-pc)
                if fix:      # Jacobian at the rounded node's true distance: s_true = (r/L)^(1/3)
                    dt = dt * (np.abs(pts-pc)/(d*Lg))**(2/3)
                tau = d if at0 else 1-d
            else:
                tau, dt = u, wu; pts = p0 + u*(p1-p0)
            P.append(pts); N.append(np.full(n_gl, -1j*tan)); W.append(dt*Lg); TAU.append(tau); CUR.append(np.full(n_gl, ci))
    c = np.concatenate
    return SimpleNamespace(pts=c(P), normals=c(N), tangents=c(N)*1j, wts=c(W), tau=c(TAU), curve=c(CUR), bc=np.zeros(len(c(P))))

base = np.array([0, 1j, -1+1j, -1-1j, 1-1j, 1], complex)
for shift in (0j, 0.7+0.4j, 3+2j):
    z = base + shift; dom = Polygon(z, bc="dir")
    rec = Eigenproblem(dom).solve(10, prec=1e-12); V = velocity(dom, z)
    for fix in (False, True):
        for n_gl, n_p in ((60, 4), (90, 6)):
            q = rule(dom, shift, n_gl, n_p, fix); out = []
            for i in (0, 8):
                L = rec.spectrum.levels[i]; g = hadamard_on(rec, L, V, q); r = identities(z, float(L.lam), g)
                out.append(f"lvl {i}: transl {max(r['tx'], r['ty'])*6:.0e} dil {r['dil']:.0e}")
            print(f"shift {shift!s:10s} fix {fix!s:5s} rule {n_gl}x{n_p} | " + " | ".join(out), flush=True)
