"""A cheap, controllable stand-in for the tension curves `sigma_j(k)`.

TEST SUPPORT ONLY. Nothing in `chladni` imports this, and nothing should: it exists so a
search algorithm can be exercised against known answers without paying for a GSVD, and so
that configurations the real solver cannot easily be steered into -- an exact triple
eigenvalue, an avoided crossing with a chosen gap, a pair separated by exactly `delta_res`
-- can be constructed by fiat.

# What it is

`sigma_j(k)**2` are the eigenvalues of a parameter-dependent Hermitian positive-semidefinite
family, which is what they are in the real problem too: for a simple smallest generalized
singular value, `sigma_1**2` is an eigenvalue of an analytic Hermitian family and is
therefore analytic. So the surrogate is built the same way round -- assemble a Hermitian
`M(k)` whose eigenvalues are the squared tensions -- rather than by writing down `sigma`
directly, which would not be a realizable family.

Each well is a parabola in the SQUARE,

    f_j(k) = eps_j**2 + c_j**2 * (k - k_j)**2

which is exactly the local model `sigma ~ sqrt(eps**2 + c**2 delta**2)` measured on real
pencils to 1-3% over the far field. Non-well directions sit at a plateau.

# The warning that comes with it

A surrogate is only worth what its validation is worth. Earlier in this project a synthetic
pencil "proved" that threshold rank truncation deletes eigenfunctions; it did, but only
because the construction put the minimizer somewhere the real problem never puts it, and the
conclusion was false for MPS. `tests/test_surrogate.py` therefore checks this family against
measured behaviour of real pencils -- cone slope, floor, two-dip resolution, the sigma_2
midpoint signature -- and any search result obtained here should be confirmed on a real
domain before it is believed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Well", "Surrogate"]


class Well(tuple):
    """One eigenvalue: where it is, how deep the floor, how steep the walls.

    `mult` places that many coincident wells, i.e. an exact degeneracy.

    `coupling` is the off-diagonal `g` joining this well to the next curve, turning a true
    crossing into an avoided one. It is in SIGMA-SQUARED units, because that is where the
    Hermitian family lives: at the crossing the two squared eigenvalues become `a -/+ g`
    with `a = eps**2`, so the observable gap in `sigma` is about `g / eps`. It must not
    exceed `eps**2`, or `a - g` goes negative and the family is no longer
    positive-semidefinite -- which is checked, because getting it wrong silently produced a
    `sigma_1` of exactly zero.

    That ceiling is not an artifact of the construction; it is the statement that an avoided
    crossing cannot open a gap larger than the floor it sits on. `lipschitz-cone-mps.md`
    section 9 says the same thing from the physics side: for near-degenerate MPS branches the
    coupling is exponentially small and the gap falls far below `eta`.
    """

    __slots__ = ()

    def __new__(cls, k, eps=1e-9, c=1.0, mult=1, coupling=0.0):
        eps, coupling = float(eps), float(coupling)
        if coupling < 0.0 or coupling > eps**2:
            raise ValueError(
                f"coupling must lie in [0, eps**2] = [0, {eps**2:.3e}], got {coupling:.3e}. "
                f"It is in sigma-squared units; a larger value makes the 2x2 block "
                f"indefinite and sigma_1 collapses to exactly zero."
            )
        return super().__new__(cls, (float(k), eps, float(c), int(mult), coupling))

    k = property(lambda s: s[0])
    eps = property(lambda s: s[1])
    c = property(lambda s: s[2])
    mult = property(lambda s: s[3])
    coupling = property(lambda s: s[4])


class Surrogate:
    """`tensions(k)` from a Hermitian family with eigenvalue curves chosen by fiat.

    `plateau` is where the non-well directions sit, `n` the total number of curves (padding
    the wells out to a realistic vector length), and `roughness` the relative amplitude of a
    deterministic high-frequency perturbation -- measured at 1e-10 on a disk pencil whose
    rank is safe, and 1e-2 on one whose cut sits at machine epsilon.

    `calls` counts evaluations, which is the metric a search is scored on.
    """

    def __init__(self, wells, plateau=0.3, n=24, roughness=0.0, seed=0, rotate=True):
        self.wells = [w if isinstance(w, Well) else Well(*np.atleast_1d(w)) for w in wells]
        self.plateau = float(plateau)
        self.n = int(n)
        self.roughness = float(roughness)
        self.calls = 0
        rng = np.random.default_rng(seed)
        self._phase = rng.uniform(0, 2 * np.pi, self.n)
        self._freq = rng.uniform(3e3, 1e4, self.n)
        # A fixed rotation, so the eigenvectors are not the coordinate axes and a solver
        # cannot accidentally exploit structure the real problem does not have.
        self._Q = np.linalg.qr(rng.standard_normal((self.n, self.n)))[0] if rotate else None
        total = sum(w.mult for w in self.wells)
        if total > self.n:
            raise ValueError(f"{total} well curves do not fit in n = {self.n}")

    # --- the family -------------------------------------------------------------------

    def _diag(self, k):
        """The squared-tension curves before coupling: wells, then plateau."""
        vals, i = np.full(self.n, self.plateau**2), 0
        for w in self.wells:
            for _ in range(w.mult):
                vals[i] = w.eps**2 + (w.c * (k - w.k)) ** 2
                i += 1
        return vals

    def matrix(self, k):
        """The Hermitian PSD matrix whose eigenvalues are `tensions(k)**2`."""
        M = np.diag(self._diag(k))
        i = 0
        for w in self.wells:
            if w.coupling and i + w.mult < self.n:
                a, b = i, i + w.mult          # couple this well to the next curve
                M[a, b] = M[b, a] = w.coupling
            i += w.mult
        if self._Q is not None:
            M = self._Q @ M @ self._Q.T
        return M

    def tensions(self, k):
        """Ascending `sigma_j(k)`. One evaluation, counted.

        Computed ANALYTICALLY, not by `eigvalsh` of `matrix(k)`, and the difference matters.
        The surrogate's curves are eigenvalues of the SQUARED quantity, so a general
        eigensolver delivers them to absolute accuracy `eps_mach * ||M||` -- which becomes
        `sqrt(eps_mach * ||M||)` once the square root is taken, a floor of about 5e-9. The
        real solver reaches 5.8e-16, because a GSVD works on the matrices rather than on
        their squares and so has relative accuracy near zero. Taking the closed form here
        keeps the surrogate from being artificially blunter than the thing it stands in for.

        Coupled pairs still go through the 2x2 closed form, which is accurate.
        """
        self.calls += 1
        k = float(k)
        vals, i = [], 0
        pending = None
        for w in self.wells:
            f = w.eps**2 + (w.c * (k - w.k)) ** 2
            for _ in range(w.mult):
                vals.append(f)
            if w.coupling:
                pending = (len(vals) - 1, w.coupling)
            i += w.mult
        vals += [self.plateau**2] * (self.n - len(vals))
        vals = np.array(vals, dtype=float)
        if pending is not None:
            j, g = pending
            if j + 1 < len(vals):
                a, b = vals[j], vals[j + 1]
                m, h = 0.5 * (a + b), 0.5 * (a - b)
                r = np.hypot(h, g)   # g is already in sigma-squared units
                vals[j], vals[j + 1] = m - r, m + r
        return np.sort(np.sqrt(np.maximum(vals, 0.0))) * self._rough(k)

    def _rough(self, k):
        if not self.roughness:
            return 1.0
        return 1.0 + self.roughness * np.sin(self._freq * k + self._phase)

    def tension(self, k):
        return float(self.tensions(k)[0])

    @property
    def truth(self):
        """`(k, multiplicity)` for every well, ascending -- the answer a search must find."""
        out = []
        for w in sorted(self.wells, key=lambda w: w.k):
            out.append((w.k, w.mult))
        return out
