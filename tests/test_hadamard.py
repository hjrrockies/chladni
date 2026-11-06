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
import warnings

import numpy as np
import pytest

from chladni._core import PrecisionWarning
from chladni.eigenproblem import Eigenproblem
from chladni.geometry import Polygon, rect

PREC = 1e-10

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
    basis of the eigenspace the solver returned.

    The estimate 3 eps/alpha sits at prec here (eps ~ 1.4e-11, alpha 0.375): it warns on some
    BLAS and not others, so the warning is ignored and the estimate checked against the truth."""
    dom = rect(1.0, 1.0)
    spec = Eigenproblem(dom).solve(3, prec=PREC)
    i = int(np.argmin(np.abs(spec.lam - 5 * np.pi ** 2)))
    assert spec.mult[i] == 2
    V = _edge_velocity(dom, lambda p0, p1: abs(p0.real - 1.0) < 1e-12
                       and abs(p1.real - 1.0) < 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        H_, rep = spec.hadamard(i, V, report=True)
    assert H_.shape == (1, 2, 2)
    want = np.array([-8 * np.pi ** 2, -2 * np.pi ** 2])
    got = np.sort(np.linalg.eigvalsh(H_[0]))
    np.testing.assert_allclose(got, want, rtol=1e-8)
    assert np.abs(got - want).max() / spec.lam[i] <= rep.attained


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
    """1 x 1.001: the square's 5 pi^2 double split into two close levels, (1,2) and (2,1).
    Taken TOGETHER they give the block a cluster optimizer needs on the way into a degeneracy.
    It is EXACT here, not O(1e-3) off the square's: moving the x = 1 edge, d(lam)/da =
    -2 m^2 pi^2 / a^3 with a = 1, so -2 pi^2 and -8 pi^2, and the cross term vanishes because
    sin(pi y/b) and sin(2 pi y/b) are orthogonal on the edge. Checked to the solve's prec."""
    dom = rect(1.0, 1.001)
    spec = Eigenproblem(dom).solve(3, prec=PREC)
    assert spec.count >= 3 and len(spec.lam) >= 3 and spec.mult[1] == 1
    V = _edge_velocity(dom, lambda p0, p1: abs(p0.real - 1.0) < 1e-12
                       and abs(p1.real - 1.0) < 1e-12)
    # Not a cluster (gap 1.2e-3), but `attained` warns: Moler-Payne's eps/alpha is loose on a
    # resolved pair (4.9e-8 claimed at prec 1e-10), the price of the calibrated estimate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        H, rep = spec.hadamard([1, 2], V, report=True)
    H = H[0]
    assert spec.spectrum.clusters == ()
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(H)), [-8 * np.pi ** 2, -2 * np.pi ** 2],
                               rtol=10 * PREC)
    assert rep.attained >= 10 * PREC                  # the estimate errs on the safe side
    assert abs(H[0, 1]) < 10 * PREC * np.pi ** 2


# --- what the matrix is good for: the report and the refusal (2026-11-03) ----------------------
#
# `notes/hadamard_accuracy.md`: the rule's shortfall was dropped and a 16% normalization error
# passed silently. Both are now reported, and the second is refused.

def _square_record(prec=PREC):
    dom = rect(1.0, 1.0)
    return dom, Eigenproblem(dom).solve(1, prec=prec)


def _one_field(dom):
    return _edge_velocity(dom, lambda p0, p1: abs(p0.real - 1.0) < 1e-12
                          and abs(p1.real - 1.0) < 1e-12)


@pytest.mark.slow
def test_a_bad_normalization_is_refused(monkeypatch):
    """A level whose Gram residual is far above prec cannot give a derivative: refused, with the
    same exception as a level that could not be orthonormalized."""
    from dataclasses import replace
    from chladni.mps.solver import MPSEigensolver

    dom, spec = _square_record()
    real = MPSEigensolver.eigenfunctions

    def bad(self, level, *, prec=None):
        return replace(real(self, level, prec=prec), gram_residual=0.16)
    monkeypatch.setattr(MPSEigensolver, "eigenfunctions", bad)
    with pytest.raises(ValueError, match="gram_residual"):
        spec.hadamard(0, _one_field(dom))


@pytest.mark.slow
def test_a_clean_square_reports_its_precision_met():
    """No ruled corner, an exact normalization: the report says the matrix meets prec, and the
    default return is still the bare matrix."""
    dom, spec = _square_record()
    H = spec.hadamard(0, _one_field(dom))
    H2, rep = spec.hadamard(0, _one_field(dom), report=True)
    np.testing.assert_array_equal(H, H2)
    assert rep.met and rep.attained <= PREC and rep.shortfalls == ()
    assert H[0, 0, 0] == pytest.approx(-2 * np.pi ** 2, rel=1e-8)


@pytest.mark.slow
def test_a_rule_shortfall_is_reported_and_warned(monkeypatch):
    """A ruled corner whose rule falls short of prec: the report names the corner and the call
    warns. Until 2026-11-03 the L-shape supplied one for free -- the fitted rule stalled at every
    corner. The graded rule meets prec there, and the corners it still misses (within `NU_MARGIN`
    of the slit) are expensive to solve on, so the shortfall is injected at the sizing."""
    from chladni._core import PrecisionWarning
    from chladni.integrals import rules

    real = rules.graded_order_for_prec

    def short(nu, prec=1e-14, k=1.0, sigma=rules.GRADED_SIGMA):
        rule, _ = real(nu, prec, k, sigma)
        return rule, 1e-6

    monkeypatch.setattr(rules, "graded_order_for_prec", short)
    base = [0, 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1]
    dom = Polygon(base, bc="dir")
    spec = Eigenproblem(dom).solve(1, prec=PREC)
    V = _edge_velocity(dom, lambda p0, p1: True)
    with pytest.warns(PrecisionWarning, match="Hadamard matrix"):
        _, rep = spec.hadamard(0, V, report=True)
    assert not rep.met and any("corner" in w for w, *_ in rep.shortfalls)


# --- closed-form identities on corner domains, anywhere in the plane (2026-11-03) ---------------
#
# For a Dirichlet eigenpair, a rigid motion leaves `lam` fixed and a dilation scales it:
#     V = e . n  (translation)  ->  0,      V = (i x) . n  (rotation)  ->  0,
#     V = x . n  (dilation)     ->  -2 lam  (every eigenvalue of the level's block).
# Dilation also tests the normalization, since it is `-2 lam ||u||^2`. Measured on the graded rule
# at prec 1e-12, worst over 6 levels relative to lam: L-shape at the origin 3.7e-14, shifted
# 3+2i 8.7e-14, the 315-degree notch shifted 9.2e-14, irregular hexagon 8.7e-13 (its eigenvalues'
# own bound). The fitted rule this replaced read 2.8e-9 on the L-shape's translation identity.

def _linear_fields(dom):
    segs = [s for b in dom.boundaries for s in b.segments]
    maps = (lambda z: np.ones_like(z), lambda z: 1j * np.ones_like(z),
            lambda z: 1j * z, lambda z: z)

    def V(curve, t):
        s = segs[curve]
        t = np.asarray(t)
        p, n = s.p(t), s.N(t)
        return np.array([(f(p) * n.conjugate()).real for f in maps])
    return V


IDENTITY_CASES = [
    ("L-shape +(3+2i)", np.array([0, 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1]) + 3 + 2j, 1e-12),
    ("315 notch +(3+2i)", np.array([2, 2 + 2j, -2 + 2j, -2 - 2j, 2 - 2j, 0]) + 3 + 2j, 1e-12),
    ("irregular hexagon", np.array([0, 1.0 - 0.1j, 1.6 + 0.5j, 1.3 + 1.3j, 0.4 + 1.5j,
                                    -0.3 + 0.7j]), 1e-11),
]


@pytest.mark.slow
@pytest.mark.parametrize("name, verts, bar", IDENTITY_CASES, ids=[c[0] for c in IDENTITY_CASES])
def test_rigid_motions_and_dilation_hold_on_corner_domains(name, verts, bar):
    dom = Polygon(verts, bc="dir")
    spec = Eigenproblem(dom).solve(6, prec=1e-12)
    V = _linear_fields(dom)
    for i in range(len(spec.lam)):
        # `attained` is a calibrated, conservative estimate (3 x max(identity residual, Moler-
        # Payne eps/alpha), 2026-11-05). It sits above prec where a neighbour is close -- the
        # notch's level 4 reads eps/alpha 4.6e-11 against identities of 1e-14, mixing the
        # identities cannot see -- an honest warning. Checked instead: it covers their error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            H_, rep = spec.hadamard(i, V, report=True)
        lam = spec.lam[i]
        assert rep.shortfalls == (), (name, rep.shortfalls)
        assert rep.identity_residual <= rep.attained, (name, i, rep)
        for f in range(3):
            assert np.abs(H_[f]).max() < bar * lam, (name, i, f)
        np.testing.assert_allclose(np.linalg.eigvalsh(H_[3]), -2 * lam, rtol=bar)


# --- the identity residuals, reported (2026-11-05) ----------------------------------------------
#
# `hadamard` integrates the four exact identities above on its own nodes alongside the caller's
# fields, and reports their residuals: translation in x and y, rotation and dilation about the
# boundary's centroid divided by its radius, so every field has unit size; each relative to
# lam. `notes/accuracy_audit_2026-11-03.md`, "ARC accuracy studies": `attained` under-reported
# H's error on the douse shapes by a median 2x, and the identities are the error itself.

@pytest.mark.slow
@pytest.mark.parametrize("name, verts, bar", IDENTITY_CASES, ids=[c[0] for c in IDENTITY_CASES])
def test_the_report_carries_the_identity_residuals(name, verts, bar):
    dom = Polygon(verts, bc="dir")
    spec = Eigenproblem(dom).solve(6, prec=1e-12)
    V = _linear_fields(dom)
    for i in range(len(spec.lam)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            H_, rep = spec.hadamard(i, V, report=True)
        lam = spec.lam[i]
        assert len(rep.identity_fields) == 4
        assert rep.identity_residual == max(rep.identity_fields) <= bar, (name, i, rep)
        # translation needs no centre: the caller's own fields give the same residual
        for f in range(2):
            assert rep.identity_fields[f] == pytest.approx(np.abs(H_[f]).max() / lam,
                                                           rel=1e-6, abs=1e-16), (name, i, f)


@pytest.mark.slow
def test_a_block_of_distinct_levels_reads_the_identities_level_by_level():
    """The identities hold within a level, not across levels: between eigenfunctions of
    eigenvalues lam != mu, the Rellich form is (lam + mu) <u, v> + (lam - mu)/2 int (v x.grad u
    - u x.grad v), and translation keeps the second term alone. So over a block of distinct
    levels the residual is the worst of the levels' own blocks; the cross-level entries are
    content, not error."""
    verts = dict((c[0], c[1]) for c in IDENTITY_CASES)["L-shape +(3+2i)"]
    dom = Polygon(verts, bc="dir")
    spec = Eigenproblem(dom).solve(3, prec=1e-12)
    V = _linear_fields(dom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        H_, rep = spec.hadamard([0, 1, 2], V, report=True)
        own = [spec.hadamard(i, V, report=True)[1].identity_residual for i in range(3)]
    assert np.abs(H_[0, 0, 1:]).max() > 1e-3 * spec.lam[0]        # the cross terms are there
    assert rep.identity_residual <= 1e-12, rep
    # the same size as the levels' own (the block divides by its largest lam, on that lam's rule)
    assert rep.identity_residual <= 10 * max(own) + 1e-13, (rep.identity_residual, own)


@pytest.mark.slow
def test_a_loose_solve_reports_a_larger_identity_residual():
    """The residual measures the eigenfunctions, not a constant: the irregular hexagon at prec
    1e-6 reads well above its 1e-12 value, and still matches the caller's translation fields."""
    verts = dict((c[0], c[1]) for c in IDENTITY_CASES)["irregular hexagon"]
    dom = Polygon(verts, bc="dir")
    V = _linear_fields(dom)
    worst = {}
    for prec in (1e-6, 1e-12):
        spec = Eigenproblem(dom).solve(4, prec=prec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            reps = [spec.hadamard(i, V, report=True) for i in range(len(spec.lam))]
        worst[prec] = max(rep.identity_residual for _, rep in reps)
        for (H_, rep), lam in zip(reps, spec.lam):
            assert rep.identity_fields[0] == pytest.approx(np.abs(H_[0]).max() / lam,
                                                           rel=1e-6, abs=1e-16)
    assert worst[1e-6] > 100 * worst[1e-12], worst


@pytest.mark.slow
def test_a_starved_level_is_not_called_orthonormal():
    """`gram_residual` alone reads 1e-15 .. 3e-9 on a starved basis whose true Gram defect is
    6e-8 .. 4e-5 (`accuracy_studies.py` starved, T1.4-hex at factor <= 0.25); `mp_eps` covers it.
    A level whose `gram_residual + mp_eps` exceeds the refusal bar is not `orthonormal`, and
    `hadamard` refuses it."""
    from dataclasses import replace
    from chladni.plan import plan_for

    verts = dict((c[0], c[1]) for c in IDENTITY_CASES)["irregular hexagon"]
    dom = Polygon(verts, bc="dir")
    plan = plan_for(dom, 6, 1e-10)
    plan = replace(plan, corner_orders=tuple(max(round(0.25 * m), 1) if m else 0
                                             for m in plan.corner_orders),
                   n_sources=round(0.25 * plan.n_sources))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = plan.solver(dom, prec=1e-10)
        res = s.search_interval(5.0, 60.0)
    bad = [L for L in res.levels if np.isfinite(L.mp_eps) and L.mp_eps > 1e-7]
    assert bad, [L.mp_eps for L in res.levels]
    for L in bad:
        ef = s.eigenfunctions(L)
        assert not ef.orthonormal and "Moler-Payne" in ef.reason, (L.lam, L.mp_eps, ef)
        with pytest.raises(ValueError, match="Moler-Payne"):
            s.hadamard(L, _linear_fields(dom))
