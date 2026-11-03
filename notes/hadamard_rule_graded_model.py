"""Model study: a graded corner panel (geometric Gauss-Legendre panels + a Gauss-Jacobi innermost
panel with weight r^gamma) on the Hadamard exponent lattice r^(gamma + j nu + m), j, m >= 0.
Reports worst relative error over the lattice monomials and the node count."""
import numpy as np
from scipy.special import roots_jacobi
from numpy.polynomial.legendre import leggauss

def graded(gamma, sigma, levels, n, n_in):
    xs, ws = [], []
    x, w = leggauss(n); x, w = 0.5*(x+1), 0.5*w
    hi = 1.0
    for _ in range(levels):
        lo = hi*sigma
        xs.append(lo + (hi-lo)*x); ws.append((hi-lo)*w); hi = lo
    t, wj = roots_jacobi(n_in, 0.0, gamma)          # weight (1+t)^gamma on [-1, 1]
    d = hi
    xin = d*(1+t)/2
    # integrand f = x^gamma g(x): sum wj * g(x) * (d/2)^(gamma+1)
    return np.concatenate(xs), np.concatenate(ws), xin, wj*(d/2)**(gamma+1)

def worst(gamma, nu, rule, J=40, M=30):
    xg, wg, xi, wi = rule
    err = 0.0
    for j in range(J):
        for m in range(M):
            e = gamma + j*nu + m
            if e > 60: break
            ex = 1.0/(e+1)
            q = np.sum(wg*xg**e) + np.sum(wi*xi**(e-gamma))
            err = max(err, abs(q-ex)/ex)
    return err

for label, nu in (("L-shape re-entrant, nu=2/3", 2/3), ("octagon near-flat, nu=1.04", 1.0399),
                  ("hexagon, nu=1.39", 1.3922), ("flat-tri apex, nu=1.23", 1.2303),
                  ("triangle, nu=2.59", 2.592)):
    gamma = 2*nu - 2
    print(label)
    for sigma, levels, n, n_in in ((0.15, 8, 12, 8), (0.15, 10, 14, 10), (0.2, 12, 16, 12),
                                   (0.15, 14, 16, 12), (0.1, 12, 20, 14)):
        r = graded(gamma, sigma, levels, n, n_in)
        print(f"   sigma {sigma} levels {levels:2d} GL {n:2d} GJ {n_in:2d}: nodes {len(r[0])+len(r[2]):4d}"
              f"  worst rel err {worst(gamma, nu, r):.1e}")

print("\n=== weighted by (h/R)^(e - gamma): relative to the total integral ===")
def worst_w(gamma, nu, rule, rho):
    xg, wg, xi, wi = rule
    err, tot = 0.0, 0.0
    for j in range(60):
        for m in range(60):
            e = gamma + j*nu + m
            c = rho**(-(e-gamma)) / np.math.factorial(min(m, 170)) if False else rho**(-(e-gamma))
            if c < 1e-18: break
            ex = 1.0/(e+1)
            q = np.sum(wg*xg**e) + np.sum(wi*xi**(e-gamma))
            err += c*abs(q-ex); tot += c*ex
    return err/tot
for rho in (2.0, 4.0):
    print(f"-- corner panel = R/{rho:g}")
    for label, nu in (("L nu=2/3", 2/3), ("oct nu=1.04", 1.0399), ("hex nu=1.39", 1.3922), ("tri nu=2.59", 2.592)):
        g = 2*nu-2
        out = []
        for sigma, levels, n, n_in in ((0.15, 6, 8, 6), (0.15, 8, 10, 8), (0.15, 10, 12, 10), (0.15, 12, 14, 10)):
            r = graded(g, sigma, levels, n, n_in)
            out.append(f"{len(r[0])+len(r[2]):3d} nodes {worst_w(g, nu, r, rho):.0e}")
        print(f"   {label:12s} " + " | ".join(out))
