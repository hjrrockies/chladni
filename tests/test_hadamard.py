"""M5: Hadamard shape derivatives, `spec.hadamard(i, V)`.  (2026-10-12)

`docs/design/douse_requirements.md`, "Hadamard integrals". For a level of multiplicity `m` with
Dirichlet eigenfunctions orthonormalized in L2 (Rellich Gram, Löwdin), and a normal velocity `V`
on the boundary, the shape derivative is the `m x m` matrix

    H_ij = - oint d_n u_i  d_n u_j  V ds,

whose eigenvalues are the derivatives of the level's members. `V` is a callable
`V(curve, t) -> (n_fields, len(t))`, with `curve` the boundary curve's index (segments in
`dom.boundaries` order, flattened) and `t` in [0, 1] its own parameter. Tested against closed
forms (a rectangle's moving side) and against finite differences of `chladni`'s own eigenvalues
(the L-shape's re-entrant vertex, where `d_n u` is singular).
"""
import numpy as np
import pytest

from chladni.eigenproblem import Eigenproblem
from chladni.geometry import Polygon, rect

PREC = 1e-10
pytestmark = pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")


def _edge_velocity(dom, pick):
    """`V(curve, t)`: 1 on the segments `pick(p0, p1)` selects, 0 elsewhere; one field."""
    segs = [s for b in dom.boundaries for s in b.segments]

    def V(curve, t):
        s = segs[curve]
        on = pick(s.p0, s.p1)
        return np.full((1, len(t)), 1.0 if on else 0.0)
    return V


@pytest.mark.slow
def test_a_rectangles_moving_side_matches_the_closed_form():
    """`lam_mn = pi^2 (m^2/L^2 + n^2/H^2)`; moving the side x = L outward at unit speed gives
    `d lam / dL = -2 pi^2 m^2 / L^3`."""
    L, H = 1.0, 0.7
    dom = rect(L, H)
    spec = Eigenproblem(dom).solve(3, prec=PREC)
    V = _edge_velocity(dom, lambda p0, p1: abs(p0.real - p1.real) < 1e-12
                       and abs(p0.real - max(z.real for z in dom.vertices)) < 1e-12)
    ms = {1: 1, 2: 2, 3: 1}            # lam_1 (1,1), lam_2 (2,1), lam_3 (1,2) for L=1, H=0.7
    for i in range(3):
        H_ = spec.hadamard(i, V)
        assert H_.shape == (1, 1, 1)
        want = -2 * np.pi ** 2 * ms[i + 1] ** 2 / L ** 3
        assert H_[0, 0, 0] == pytest.approx(want, rel=1e-8), (i, H_[0, 0, 0], want)


@pytest.mark.slow
def test_a_double_splits_by_its_members_derivatives():
    """The square's 5 pi^2 double, (1,2) and (2,1): moving the side x = 1 gives -2 pi^2 for the
    first and -8 pi^2 for the second. The matrix's EIGENVALUES are those, whichever orthonormal
    basis of the eigenspace the solver returned."""
    dom = rect(1.0, 1.0)
    spec = Eigenproblem(dom).solve(3, prec=PREC)
    i = int(np.argmin(np.abs(spec.lam - 5 * np.pi ** 2)))
    assert spec.mult[i] == 2
    V = _edge_velocity(dom, lambda p0, p1: abs(p0.real - 1.0) < 1e-12
                       and abs(p1.real - 1.0) < 1e-12)
    H_ = spec.hadamard(i, V)
    assert H_.shape == (1, 2, 2)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(H_[0])),
                               [-8 * np.pi ** 2, -2 * np.pi ** 2], rtol=1e-8)


@pytest.mark.slow
def test_many_fields_at_once():
    dom = rect(1.0, 0.7)
    spec = Eigenproblem(dom).solve(1, prec=PREC)
    segs = [s for b in dom.boundaries for s in b.segments]

    def V(curve, t):
        return np.vstack([np.full(len(t), 1.0), np.full(len(t), 2.0)])
    H_ = spec.hadamard(0, V)
    assert H_.shape == (1 + 1, 1, 1) and H_[1, 0, 0] == pytest.approx(2 * H_[0, 0, 0], rel=1e-12)
    assert len(segs) == 4


@pytest.mark.slow
def test_the_L_shapes_reentrant_vertex_matches_finite_differences():
    """The hard case: `d_n u ~ r^(-1/3)` at the re-entrant corner, which the vertex's own motion
    weights. Move the re-entrant vertex (0, 0) along x by h; the normal velocity on each edge is
    linear in t, with the vertex's velocity dotted with the edge normal at the end that moves.
    Reference: central differences of `chladni`'s own lam_1 at 1e-12."""
    base = [0, 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1]       # the L-shape, re-entrant vertex at 0

    def shape(h):
        v = list(base)
        v[0] = v[0] + h
        return Polygon(v, bc="dir")

    dom = shape(0.0)
    spec = Eigenproblem(dom).solve(1, prec=PREC)
    segs = [s for b in dom.boundaries for s in b.segments]
    vel = np.array([1.0 + 0j] + [0j] * 5)              # only vertex 0 moves, along +x
    verts = [complex(z) for z in base]

    def V(curve, t):
        s = segs[curve]
        i0 = int(np.argmin([abs(s.p0 - z) for z in verts]))
        i1 = int(np.argmin([abs(s.p1 - z) for z in verts]))
        d = s.p1 - s.p0
        n = -1j * d / abs(d)                            # outward for a CCW polygon
        v = (1 - t) * vel[i0] + t * vel[i1]
        return (v.real * n.real + v.imag * n.imag)[None, :]

    h = 1e-4
    lp = Eigenproblem(shape(h)).solve(1, prec=1e-12).lam[0]
    lm = Eigenproblem(shape(-h)).solve(1, prec=1e-12).lam[0]
    fd = (lp - lm) / (2 * h)
    H_ = spec.hadamard(0, V)
    # Measured 2026-10-12: rel diff 1.5e-6, 1.3e-7, 1.5e-8 at h = 1e-3, 3e-4, 1e-4 -- shrinking as
    # h^2, so the residue is the finite difference's, not the Hadamard value's.
    assert H_[0, 0, 0] == pytest.approx(fd, rel=1e-7), (H_[0, 0, 0], fd)


# --- several levels at once: a NEARLY degenerate cluster (douse's cluster_matrices) -------------

@pytest.mark.slow
def test_two_distinct_levels_form_one_block_with_their_cross_terms():
    """`hadamard([i, j], V)` over two distinct levels: the diagonal is each level's own
    derivative and the off-diagonal the cross term. On a 1 x 0.7 rectangle, moving x = 1: lam_2
    is (2,1), lam_3 is (1,2); their normal derivatives on that side are orthogonal in y, so the
    cross term vanishes."""
    dom = rect(1.0, 0.7)
    spec = Eigenproblem(dom).solve(3, prec=PREC)
    V = _edge_velocity(dom, lambda p0, p1: abs(p0.real - 1.0) < 1e-12
                       and abs(p1.real - 1.0) < 1e-12)
    H_ = spec.hadamard([1, 2], V)
    assert H_.shape == (1, 2, 2)
    np.testing.assert_allclose(np.diag(H_[0]), [-8 * np.pi ** 2, -2 * np.pi ** 2], rtol=1e-8)
    assert abs(H_[0, 0, 1]) < 1e-8 * np.pi ** 2


@pytest.mark.slow
def test_a_split_pair_near_a_double_gives_the_doubles_derivatives():
    """1 x 1.001: the square's 5 pi^2 double split into two close levels. Taken TOGETHER they
    give a block whose eigenvalues are within O(1e-3) of the square's -2 pi^2 and -8 pi^2 --
    the model a cluster optimizer needs on the way into a degeneracy."""
    dom = rect(1.0, 1.001)
    spec = Eigenproblem(dom).solve(3, prec=PREC)
    assert spec.count >= 3 and len(spec.lam) >= 3 and spec.mult[1] == 1
    V = _edge_velocity(dom, lambda p0, p1: abs(p0.real - 1.0) < 1e-12
                       and abs(p1.real - 1.0) < 1e-12)
    ev = np.sort(np.linalg.eigvalsh(spec.hadamard([1, 2], V)[0]))
    np.testing.assert_allclose(ev, [-8 * np.pi ** 2, -2 * np.pi ** 2], rtol=5e-3)
