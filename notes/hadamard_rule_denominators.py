"""Substitute-then-grade corner rule on [0,1]: tau = s^p, p = max(1, 1/(2 nu - 1)), then geometric
Gauss-Legendre panels in s toward 0 (ratio sigma, `levels`, n nodes each; innermost [0, sigma^L]
is a plain GL panel). Integrand: sum_{j,m} rho^-(j nu + m) r^(gamma + j nu + m), gamma = 2nu - 2,
rho = 2 (corner panel = half the convergence radius). Error relative to the total."""
import numpy as np
from fractions import Fraction
from numpy.polynomial.legendre import leggauss

def rule(nu, sigma=0.15, levels=8, n=14):
    p = max(1.0, 1.0/(2*nu - 1))
    x, w = leggauss(n); x, w = 0.5*(x+1), 0.5*w
    S, W = [], []
    hi = 1.0
    for _ in range(levels):
        lo = hi*sigma
        S.append(lo + (hi-lo)*x); W.append((hi-lo)*w); hi = lo
    S.append(hi*x); W.append(hi*w)
    s, ws = np.concatenate(S), np.concatenate(W)
    return s**p, ws*p*s**(p-1)            # nodes and weights in tau

def err(nu, r, rho=2.0):
    t, w = r
    g = 2*nu - 2
    e_tot = q_tot = 0.0
    E, C = [], []
    for j in range(80):
        for m in range(80):
            a = j*nu + m
            c = rho**(-a)
            if c < 1e-20: break
            E.append(g + a); C.append(c)
    E, C = np.array(E), np.array(C)
    exact = np.sum(C/(E+1))
    quad = np.sum(C[:, None] * w[None, :] * t[None, :]**E[:, None])
    return abs(quad - exact)/exact

cases = [("2/3 (L-shape)", 2/3), ("7/10", 0.7), ("71/101", 71/101), ("0.7 + irrational", 0.7 + 1e-3*np.sqrt(2)),
         ("4/7", 4/7), ("53/93", 53/93), ("0.57 irr", 0.57 + 1e-4*np.pi),
         ("1 (straight)", 1.0), ("101/100", 1.01), ("1.0399 (octagon)", 1.0399),
         ("3/2", 1.5), ("149/100", 1.49), ("1.5 - 1e-6", 1.5 - 1e-6), ("1.3922 (hexagon)", 1.3922),
         ("2", 2.0), ("199/100", 1.99), ("2.592 (triangle)", 2.592), ("11.7993 (flat apex)", 11.7993)]
print(f"{'nu':22s} {'denom':>6s} {'p':>6s}  {'min r node':>10s}   L=6 n=12  |  L=8 n=14  |  L=10 n=16")
for lab, nu in cases:
    q = Fraction(nu).limit_denominator(1000).denominator
    p = max(1.0, 1/(2*nu-1))
    row = []
    for L, n in ((6, 12), (8, 14), (10, 16)):
        r = rule(nu, 0.15, L, n)
        row.append(f"{err(nu, r):.0e} ({len(r[0])})")
    rmin = rule(nu, 0.15, 8, 14)[0].min()
    print(f"{lab:22s} {q:6d} {p:6.2f}  {rmin:10.1e}   " + "  |  ".join(row))

print("\n=== re-entrant corners: fewer grading levels in s, so nodes stay representable ===")
print(f"{'nu':16s} {'p':>5s}   " + "   ".join(f"L={L} n={n}: err (nodes, min r)" for L, n in ((0, 30), (1, 24), (2, 20), (3, 18))))
for lab, nu in (("2/3 (L-shape)", 2/3), ("7/10", 0.7), ("71/101", 71/101), ("0.7 irr", 0.7 + 1e-3*np.sqrt(2)),
                ("0.8", 0.8), ("0.9", 0.9), ("4/7", 4/7), ("53/93", 53/93), ("0.53 (near slit)", 0.53)):
    p = max(1.0, 1/(2*nu-1))
    row = []
    for L, n in ((0, 30), (1, 24), (2, 20), (3, 18)):
        r = rule(nu, 0.15, L, n)
        row.append(f"{err(nu, r):.0e} ({len(r[0])}, {r[0].min():.0e})")
    print(f"{lab:16s} {p:5.2f}   " + "   ".join(row))
