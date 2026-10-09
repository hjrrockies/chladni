"""An independent FEM cross-check for reference spectra: P2 Dirichlet eigenvalues on a polygon.

Conforming Galerkin eigenvalues are UPPER bounds, converging from above at `O(h^(2 alpha))` near
a reentrant corner (`alpha = pi/omega`), so on a corner domain FEM is not a precision check. What
it is good for, and the only thing `make_references.py` uses it for, is the COUNT: below a cut in
a clear spectral gap, the number of FEM eigenvalues equals the true count once the mesh resolves
the modes, and it is computed by a method sharing nothing with MPS. A dropped or invented level,
or a wrong multiplicity of a separated level, shows up as a count mismatch.

The mesh is an ear-clipping triangulation of the polygon, uniformly refined; element shape is the
initial triangulation's, which is poor on spike-like shapes but uniform refinement never makes it
worse.

    from benchmarks.douse.fem import fem_eigs
    lam, ndofs = fem_eigs(vertices, n_eigs=12, refine=5)
"""
import numpy as np
import scipy.sparse.linalg as sla


def _ear_clip(z):
    """Triangles (index triples) of a simple CCW polygon `z` (complex)."""
    idx = list(range(len(z)))
    tris = []

    def cross(a, b, c):
        return ((b - a).conjugate() * (c - a)).imag

    def inside(p, a, b, c):
        # Closed: a vertex ON the ear's diagonal would leave a degenerate triangle behind.
        tol = -1e-13 * max(abs(b - a), abs(c - b), abs(a - c)) ** 2
        return cross(a, b, p) >= tol and cross(b, c, p) >= tol and cross(c, a, p) >= tol

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        best = None
        for i in range(n):
            ia, ib, ic = idx[i - 1], idx[i], idx[(i + 1) % n]
            a, b, c = z[ia], z[ib], z[ic]
            if cross(a, b, c) <= 1e-14 * abs(b - a) * abs(c - b):
                continue                     # reflex or flat: not an ear
            if any(inside(z[j], a, b, c) for j in idx if j not in (ia, ib, ic)):
                continue
            # The best-shaped ear first: its smallest angle.
            ang = min(abs(np.angle((c - b) / (a - b))), abs(np.angle((a - c) / (b - c))),
                      abs(np.angle((b - a) / (c - a))))
            if best is None or ang > best[0]:
                best = (ang, i, (ia, ib, ic))
        if best is None:
            raise ValueError("ear clipping failed: polygon not simple or not CCW")
        tris.append(best[2])
        idx.pop(best[1])
    tris.append(tuple(idx))
    t = np.array(tris)
    area = np.array([cross(z[i], z[j], z[k]) for i, j, k in t])
    if np.any(area <= 0):
        raise ValueError("ear clipping produced a degenerate triangle")
    return t


def mesh_of(vertices, refine):
    from skfem import MeshTri
    v = np.asarray(vertices, dtype=float)
    z = v[:, 0] + 1j * v[:, 1]
    # Drop exactly flat vertices (180 degrees): a triangle there would be degenerate.
    keep = []
    for i in range(len(z)):
        a, b, c = z[i - 1], z[i], z[(i + 1) % len(z)]
        if abs(((b - a).conjugate() * (c - b)).imag) > 1e-13 * abs(b - a) * abs(c - b):
            keep.append(i)
    z = z[keep]
    t = _ear_clip(z)
    m = MeshTri(np.array([z.real, z.imag]), t.T)
    return m.refined(refine) if refine else m


def fem_eigs(vertices, n_eigs, refine=5):
    """The `n_eigs` smallest P2 Dirichlet eigenvalues, and the number of free dofs."""
    from skfem import Basis, ElementTriP2, condense
    from skfem.models.poisson import laplace, mass
    m = mesh_of(vertices, refine)
    ib = Basis(m, ElementTriP2())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs()
    Kc, Mc, _, I = condense(K, M, D=D)
    # A FIXED start vector. ARPACK's default is random, so the eigenvalues -- and every window
    # make_references.py derives from them -- changed in the last digits on every run, and counts
    # at near-degeneracies change with ulps of the window (notes/nondeterminism.md).
    v0 = np.random.default_rng(0).standard_normal(Kc.shape[0])
    w = sla.eigsh(Kc, k=n_eigs, M=Mc, sigma=0.0, which="LM", return_eigenvectors=False, v0=v0)
    return np.sort(w), int(len(I))
