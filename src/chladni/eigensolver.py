"""The method-agnostic eigensolver contract, and the spectral parameter.

This module sits above `geometry` and below every method package -- the same position
`helmholtz` occupies. Deliberately NOT in `chladni/mps/`: a later `chladni/fem/` must not
import its base class out of the MPS package. And deliberately not in `chladni/_core.py`,
which is dependency-free by design, `geometry` importing from it rather than the other way.

What is on `Eigensolver` is thin, and the thinness is the point. MPS scans a scalar tension
across `lam` and minimizes it. FEM assembles a sparse symmetric pencil and solves it ONCE for
many eigenvalues, with no scan and nothing to minimize. A boundary-integral method has its own
residual with its own shape. The only surface all three share is "given a domain and an
interval, return the eigenvalues in it" -- which is exactly what lappy's `BaseEigensolver`
settled on, and it was right.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ._core import PREC, PrecisionWarning, _set, check_prec

__all__ = ["SpectrumLevel", "Eigensolver", "Spectrum", "lam_to_k", "k_to_lam", "weyl_window",
           "PREC", "PrecisionWarning"]

#: Relative slack below the Faber-Krahn bound. Faber-Krahn is an EQUALITY for the disk, so an
#: unpadded lower endpoint lands exactly ON lam_1 -- and a grid search that ignores its
#: endpoints then misses the first mode entirely. lappy hit this: "The unit disk returned
#: modes 2..k+1: every value correct to ~14 digits, every certificate valid, and nothing in
#: the pipeline able to detect the omission."
_WINDOW_PAD = 1e-3

#: Relative slack above the Weyl estimate. Weyl's law is asymptotic and under-counts at small
#: n, so the window is padded and `solve` widens further when even that came up short. Kept
#: modest because widening is cheap and an over-wide first window is not.
_WEYL_PAD = 0.15

#: How many times `solve` may widen before giving up. A bound, not a tuning knob: without one
#: a domain whose search is broken loops until the machine dies, which is the failure lappy
#: recorded when an unbounded recursion "drove a 16GB machine to a 59.8GB footprint".
_MAX_WIDEN = 8


def lam_to_k(lam):
    """`k = sqrt(lam)`, validated.

    The validation is the reason this is a function and not an expression. A line search can
    hand `lam` a negative iterate, and `sqrt` of that is a complex number arriving three
    functions later as a confusing error about array dtypes -- the same argument `check_k`
    makes one layer down, at the other end of the same conversion.
    """
    value = np.asarray(lam, dtype=float)
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(
            f"lam must be finite and strictly positive, got {lam!r}. The Helmholtz layer "
            f"is parameterized by k = sqrt(lam) and cannot accept a non-positive iterate."
        )
    return np.sqrt(value) if value.ndim else float(np.sqrt(value))


def k_to_lam(k):
    """`lam = k**2`. The other direction needs no validation: every real k gives a real lam."""
    value = np.asarray(k, dtype=float)
    return value**2 if value.ndim else float(value**2)


def weyl_window(domain, n, pad=_WEYL_PAD):
    """A `lam` interval expected to contain the first `n` eigenvalues, as `(a, b)`.

    The upper end is `asymp.weyl_est`, the two-term Weyl law inverted; the lower end is
    `bounds.faber_krahn`, padded by `_WINDOW_PAD` for the reason recorded there. Both used to
    be inlined here, along with a hard-coded first zero of `J_0`.

    For a boundary that is not purely Dirichlet, Faber-Krahn does not apply -- the Neumann
    spectrum starts at `lam = 0`, whose constant eigenfunction no MPS basis represents and
    whose `k = 0` the helmholtz layer rejects by construction -- so the window starts just
    above zero and the zero mode is simply not found.
    """
    from . import asymp, bounds

    b = float(asymp.weyl_est(n, domain)) * (1.0 + pad)
    if domain.bc_type == "dir":
        a = float(bounds.faber_krahn(domain)) * (1.0 - _WINDOW_PAD)
    else:
        a = b * 1e-6
    return a, b


@dataclass(frozen=True)
class SpectrumLevel:
    """One level of a `Spectrum` from the black-box path. `margin` is how many local floor
    scales the request spans (2 is needed), `resolution` the neighbourhood the count stands
    for, `count_shortfall` that the count was read where the floor blurs it
    (`chladni.mps.search.Level`)."""
    lam: float
    mult: int
    bound: float
    margin: float
    resolution: float
    met: bool
    count_shortfall: bool
    count_uncertain: bool = False


@dataclass(frozen=True)
class Spectrum:
    """What an eigensolver found in an interval: LEVELS, their multiplicities, and its bar.

    Method-agnostic on purpose, and the line is drawn where it is for a reason. Every solver
    knows which levels it found, how many independent eigenfunctions sit at each, and how well
    it thinks it located them. None of them shares a tension, a bracket or a gap ratio, so
    those stay on the MPS-side result and out of this type -- the same rule that keeps
    `Eigensolver` itself free of `basis` and `collocation`.

    **Why not just the array.** `unfolded` is one attribute away, and for a long time it was
    the whole contract. Two things do not survive it. A COMPUTED multiplicity of 2 and two
    distinct levels 1e-12 apart unfold to the same three numbers, and they are different
    claims -- `douse` cares, because a crossing is exactly where a shape objective stops being
    smooth. And a short answer is ambiguous between "there are only this many", "some were
    missed" and "the search gave up", which `solve` used to resolve by widening the window
    eight times and then blaming the spectrum. `declined` says which, so it no longer guesses.

    `bound` is the attained ABSOLUTE bar per level, `nan` where the solver does not measure
    one. It is compared against `prec` -- the request this answers -- by `unmet`, so the
    honesty clause is a comparison the caller can redo rather than a flag to be trusted.
    """

    lam: np.ndarray
    mult: np.ndarray
    bound: np.ndarray
    prec: float = PREC
    declined: bool = False
    reason: str = ""
    #: Per level, what the black-box path (`chladni.eigenproblem`) knows beyond value,
    #: multiplicity and bound: `SpectrumLevel`s in `lam` order. Empty from older solvers.
    levels: tuple = ()
    #: The returned count against Weyl's law (`asymp.CountAudit`), or None.
    count_audit: object = None
    #: Everything the caller should know before trusting this answer, as short strings. Empty
    #: means: every level met `prec`, the count audited ok, nothing uncertain.
    flags: tuple = ()
    #: The `chladni.plan.Plan` that produced it, for re-use on a nearby shape.
    plan: object = None

    def __post_init__(self):
        lam = np.atleast_1d(np.asarray(self.lam, dtype=float))
        mult = np.atleast_1d(np.asarray(self.mult, dtype=int))
        bound = np.atleast_1d(np.asarray(self.bound, dtype=float))
        if not (lam.shape == mult.shape == bound.shape):
            raise ValueError(
                f"lam, mult and bound must agree in shape, got {lam.shape}, {mult.shape} "
                f"and {bound.shape}")
        if np.any(mult < 1):
            raise ValueError(f"every multiplicity must be at least 1, got {mult!r}")
        if lam.size and np.any(np.diff(lam) <= 0):
            raise ValueError("levels must be strictly ascending and distinct")
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "mult", mult)
        object.__setattr__(self, "bound", bound)
        object.__setattr__(self, "prec", check_prec(self.prec))

    @classmethod
    def from_unfolded(cls, lam, prec=PREC, bound=None):
        """Group an ascending unfolded multiset into levels, by exact equality.

        Exact, not tolerant, and that is the point: collapsing near-equal values is a
        COMPARISON-TIME choice, so a solver that wants it does the grouping itself and says so
        through `mult`. `chladni.reference` returns unfolded multisets for the same reason.
        """
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        levels, mult = np.unique(lam, return_counts=True)
        if bound is None:
            bound = np.full(levels.shape, np.nan)
        return cls(levels, mult, bound, prec=prec)

    @classmethod
    def declining(cls, reason, prec=PREC):
        """The third outcome: the solver detected that its premises do not hold and stopped.

        Neither a pass nor a failure (`docs/design/candidates.md` section 5), and a VALUE
        rather than an exception, so a caller can inspect it.
        """
        empty = np.empty(0)
        return cls(empty, np.empty(0, dtype=int), empty, prec=prec,
                   declined=True, reason=str(reason))

    def __bool__(self):
        """False only when DECLINED. An empty non-declined spectrum is a usable answer --
        "there are none in this interval" -- and must not read as a failure."""
        return not self.declined

    @property
    def unfolded(self):
        """Ascending, with each level repeated by its multiplicity. The array contract."""
        return np.repeat(self.lam, self.mult)

    @property
    def count(self):
        """Eigenvalues found, counted WITH multiplicity."""
        return int(self.mult.sum())

    @property
    def unmet(self):
        """Per level: is the attained bar wider than `prec` asked for?

        An unmeasured bound is `nan` and compares False here, so `unmet` alone never claims a
        level is bad. `achieved` is the one that requires the bound to exist.
        """
        with np.errstate(invalid="ignore"):
            return self.bound > self.prec * self.lam

    @property
    def achieved(self):
        """Every level located to `prec`, and MEASURED to have been. An unmeasured bound is
        not a met one, so a solver that reports no bar does not get to claim success."""
        return bool(not self.declined
                    and self.bound.size
                    and np.all(np.isfinite(self.bound))
                    and not self.unmet.any())


class Eigensolver(ABC):
    """Anything that computes Laplacian eigenvalues of a planar domain.

    Subclasses implement `solve_interval`; `solve` comes free. There is no tension here, no
    basis and no collocation -- see the module docstring for why none of it survives contact
    with a second method.
    """

    def __init__(self, domain, prec=None):
        _set(self, domain=domain,
             prec=check_prec(domain.prec if prec is None else prec))

    @property
    def bc_type(self):
        return self.domain.bc_type

    @abstractmethod
    def solve_interval(self, a, b):
        """The `Spectrum` of `[a, b]`: every level in it, with multiplicity and a bar."""

    def solve(self, n):
        """The `n` smallest eigenvalues, ascending, with multiplicity repeated.

        A Weyl window, then `solve_interval`, widening upward if the window came up short.
        The lower endpoint never moves: it is a true bound, and lowering it would only
        invite spurious roots below the spectrum.

        A DECLINE is not a short answer and is not widened past. Widening asks the same
        search the same question over a larger interval; if it has already reported that its
        premises do not hold, eight more attempts cost eight more searches and end in a
        message naming the spectrum, which is the wrong cause.
        """
        a, b = weyl_window(self.domain, n)
        for _ in range(_MAX_WIDEN):
            spectrum = self.solve_interval(a, b)
            if spectrum.declined:
                raise RuntimeError(
                    f"the search declined on lam in [{a:.6g}, {b:.6g}]: {spectrum.reason}. "
                    f"This is the solver reporting that its premises do not hold, not a "
                    f"statement about the spectrum -- widening the window would ask the same "
                    f"question again. Call solve_interval directly to inspect what it found "
                    f"before it stopped."
                )
            if spectrum.count >= n:
                return np.sort(spectrum.unfolded)[:n]
            b = a + 2.0 * (b - a)
        raise RuntimeError(
            f"found {spectrum.count} eigenvalues, short of the {n} requested, after "
            f"{_MAX_WIDEN} widenings out to lam = {b:.6g}. Either the domain's spectrum is "
            f"sparser than Weyl's law predicts, or solve_interval is missing roots."
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.domain!r}, prec={self.prec:.0e})"
