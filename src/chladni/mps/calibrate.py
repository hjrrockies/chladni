"""Is this pencil's tension landscape FINDABLE at the resolution a scan can afford?

A cheap check, run before a search, that answers a question the solver's own quality indicator
cannot: not "how well is the basis resolving what I found", but "could I have found it at all".

# The failure this exists for

During a shape optimization maximizing `lam_3/lam_1` over perimeter-4 triangles, the optimizer
stepped onto a triangle with angles 21.9 / 25.0 / 133.1 degrees. A fresh full solve returned

    lams  = [93.07, 198.46, 256.23, 381.88]        tension at lam_1: 2.8e-14

with no exception and no warning. The true spectrum is `[93.16, 148.00, 198.86, 256.90, 306.83,
328.55, 383.36]`: the returned set is `(lam_1, lam_3, lam_4, lam_7)` -- **every value accurate,
the set not consecutive**. The recorded objective was 2.7530 against a true 2.1346.

Three properties of that failure shape this module:

* **The tension at the returned eigenvalues was excellent.** It measures how well the basis
  resolves what was FOUND and is structurally blind to what was not, so no floor check, however
  strict, could have caught this.
* **Counting detectors are one-sided in the wrong way.** A Weyl deficit scored +2.02 against
  three modes actually absent: no cut can be placed above the top returned value, so the topmost
  gap is invisible. A *duplicated* root -- the same eigenvalue returned twice, observed at a
  neighbouring shape -- adds a value exactly where one is missing, so every counting test
  balances.
* **It is not reproducible by replaying the run.** Re-running the identical start diverged at a
  trust-radius shrink and converged cleanly, never revisiting the shape. A known-bad shape has
  to be stored and re-solved, not replayed.

# The mechanism, and the criterion

The dip around an eigenvalue is a clipped V, `sigma ~ min(sigma_base, C_k*|k - k_j|)`, whose
width at a threshold `q*sigma_base` is `w_k = 2*q*sigma_base/C_k`. Measured widths reproduce
that to about 2%: **only `sigma_base` and `C_k` matter** -- not the floor, not the mode index,
not `lam`. And `C_k` is a slope in `k`, so it carries units of LENGTH and scales with the
domain's length scale. Across 11 domains and 61 measurements:

    C_k                      spread 46%
    C_k / area               spread 35%
    C_k / sqrt(area)         spread 19%
    C_k / (area/perimeter)   spread 10%        2.384 +- 0.231

The dimensional argument says only the last can be right -- `sigma` is a ratio of norms and so
dimensionless, `k` is an inverse length, so `C_k = dsigma/dk` is a length, and area is not. A
scale test separates them: the same 3:1 rectangle at perimeters 2, 4 and 8 gives `C_k` of 0.225,
0.451 and 0.901, and `C_k/(A/P)` of 2.403 three times over.

Requiring the visible catchment to be at least one scan step wide, `w_k >= dk`, gives

    sigma_base  >=  (kappa / 2q) * (|Omega| / |dOmega|) * dk

with everything on the right known **before any mode is located**. On the failing triangle that
demanded `sigma_base >= 1.6e-2` against a measured 2.5e-3 -- a 6.5x shortfall, derived from
geometry and the grid alone. It would have flagged the shape as unfindable before the scan ran.

# Two fixes, and why the check matters more than either

Read the inequality either way and it prices both:

1. **Raise `sigma_base`** by loosening the rank truncation. Free at scan time, costs basis
   conditioning. `sigma_base` is monotone in the threshold, so this is a short bisection onto a
   computed target rather than a search.
2. **Shrink `dk`.** A guarantee rather than a tuning gamble, and it costs linearly in time. On
   the failing triangle it asked for 650 scan points against the 100 being used.

They compose, and the CHECK is worth more than either: a solver that measures these can report
"this domain needs 650 points at this precision" instead of silently returning a short spectrum.

# What does NOT port from the study, and why the procedure does

The study measured `lappy`, whose `rtol` is an **absolute** cutoff on singular values. `chladni`
thresholds **relative to the largest**, which is the better convention -- invariant under scaling
of the domain and of the basis normalization, where an absolute cutoff silently changes meaning
when either moves. So `rtol_lappy ~ rtol_chladni * s_max(R)`, and `s_max(R)` depends on the basis
normalization, the collocation count and `k`. **There is no fixed conversion factor and no number
here is ported.** What ports is the procedure: calibrate against the OBSERVABLE `sigma_base` and
its computed target, which sidesteps the convention mismatch entirely and yields a threshold
meaningful in `chladni`'s own units by construction.

# Dead ends, measured, so they are not retried

* **`ttol` is never the binding constraint.** Across 44 configurations the well floor stayed at
  or below 3.8e-4 -- every mode that got bracketed got accepted. The failure is findability,
  never acceptance, and raising an acceptance threshold achieves nothing.
* **"Use a coarser basis" is wrong**, and non-monotone: `target=1e-12` was worse (2/6 and 1/6)
  than either `1e-10` or `1e-14`.
* **Curvature-based auto-selection of the threshold** is reported unreliable by its author.
  Estimating a noise floor from the smoothness of `sigma` is a harder problem than targeting
  `sigma_base`, which is a direct measurement of the quantity that actually matters.
* **The interior-point draw is a symptom, not a knob.** Changing only the RNG seed moved
  `sigma_base` by 14x at fixed basis, and the DEFAULT seed was the lucky one on the shape that
  happened to solve. That is fragility to remove, not tuning to exploit -- hence the sampling
  below is deterministic and takes no seed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .._core import PrecisionWarning
from ..eigensolver import lam_to_k

__all__ = ["sigma_base", "findability_target", "required_dk", "calibrate_rtol",
           "RtolCalibration", "warn_if_unfindable",
           "KAPPA", "KAPPA_LAPPY", "Q_VISIBLE"]

#: `C_k / (|Omega|/|dOmega|)`, RE-MEASURED IN `chladni`. The upper end of the measured range,
#: not the mean, because the criterion should err toward flagging: a larger `kappa` raises the
#: target and so declares more domains unfindable, and for a solver preferring a reliable answer
#: to a fast one that is the safe direction.
#:
#: Measured on six domains whose plane-wave basis reaches a floor below 1e-6 at the level used
#: (`tests/test_calibrate.py`), taking `C_k` as `sigma_1(k* +/- dk)/dk` on the straight flank:
#:
#:     domain      A/P      C_k     C_k/(A/P)
#:     square    0.2500   0.7033      2.813
#:     rect 2x1  0.3333   0.8582      2.575
#:     rect 3x1  0.3750   0.8936      2.383
#:     rect 5x1  0.4167   0.9243      2.218
#:     eq_tri    0.1443   0.3951      2.737
#:     disk      0.5000   1.4107      2.821
#:
#:     kappa_chladni = 2.591 +- 0.226   (spread 23%)
#:     kappa_lappy   = 2.384 +- 0.231   (spread 10%, 61 measurements, 11 domains)
#:
#: **The constant ports, which the originating note expected NOT to happen.** It warned that
#: "kappa = 2.38 is specific to lappy's normalization of sigma; expect a different constant in
#: chladni". The two agree within one standard deviation, so the two normalizations of `sigma`
#: evidently coincide closely enough. Recorded because the caution was reasonable and the
#: measurement is what settled it -- and because the number would have been *silently* wrong by
#: a factor had it not been checked.
#:
#: **The `A/P` scaling is confirmed independently and exactly.** The same 3:1 rectangle at three
#: sizes, which is the test that separates `A/P` from area (holding perimeter fixed makes them
#: proportional, so only scaling distinguishes them):
#:
#:     scale        0.5      1.0      2.0
#:     A/P       0.1875   0.3750   0.7500
#:     C_k       0.4613   0.9225   1.8448      exactly proportional to the length scale
#:     C_k/(A/P) 2.4600   2.4600   2.4598      identical to four significant figures
#:
#: `C_k/area` over those three would read 2.46, 1.23, 0.61. The dimensional argument -- `sigma`
#: is a ratio of norms and so dimensionless, `k` is an inverse length, so `C_k = dsigma/dk` is a
#: LENGTH, and area is not -- is therefore not merely tidy; it is measured.
#:
#: The residual spread is systematic rather than noise, and in the same direction the study
#: found: fat domains sit high (disk 2.82, square 2.81) and elongated ones low (rect 5x1 2.22),
#: against the study's "fat 2.44-2.58, sharp or elongated 2.24-2.31". A wider family would
#: probably resolve a weak aspect-ratio dependence. Carry the safety factor until it does.
#:
#: 2.85 rather than the mean 2.591: it must sit at or above every measured ratio (the largest
#: is the disk at 2.8214), so that no domain in the calibration set would be waved through by a
#: constant fitted to the others.
KAPPA = 2.85

#: The study's value, kept for comparison rather than for use. See `KAPPA` above.
KAPPA_LAPPY = 2.384

#: Fraction of `sigma_base` at which the dip's catchment is called visible. `0.9` keeps the
#: measurement in the straight-flank regime where the clipped-V model was fitted; nearer the
#: floor the V rounds over into the cone and the width formula would need the cone form.
Q_VISIBLE = 0.9

#: Floor on the calibrated threshold: below machine epsilon a RELATIVE cutoff barely truncates
#: at all, keeping every direction the arithmetic can distinguish.
_RTOL_LO = 1e-16

#: **`rtol` is safe to raise up to `prec` itself.** Above the working precision the truncation
#: starts discarding directions the answer depends on; below it, it is discarding directions
#: whose contribution is smaller than the precision asked for, which is by definition
#: negligible. So the calibration searches `[sqrt(prec*eps), prec]` and needs no constant of
#: its own -- and the derived default `sqrt(prec*eps)` is the geometric mean of that interval's
#: two ends, sitting conservatively far from the ceiling.
def _rtol_ceiling(prec):
    return float(prec)

#: Probe points for `sigma_base`. ~40 was enough in the study, and the measurement is a median
#: so it is insensitive to a few probes landing in a dip.
_N_PROBE = 40

#: Golden-ratio offset for the probe grid. Deterministic and irrational: an evenly spaced grid
#: risks landing on a resonance systematically, and a random one reintroduces exactly the
#: seed-sensitivity this module exists to remove.
_PHI_CONJ = 0.6180339887498949


@dataclass(frozen=True)
class RtolCalibration:
    """What the findability check found, and what it did about it."""
    rtol: float                  # the threshold to use
    sigma_base: float            # measured plateau at that threshold
    target: float                # what the criterion demands
    dk: float                    # the scan step the target was computed for
    achieved: bool               # sigma_base >= target
    calibrated: bool             # False when the default already passed and nothing was done
    required_dk: float           # the dk this sigma_base COULD support -- the other fix
    evaluations: int = 0
    history: tuple = field(default_factory=tuple)   # (rtol, sigma_base, abs_cut) per step

    @property
    def shortfall(self):
        """How many times too small the plateau is. `<= 1` means findable."""
        return self.target / self.sigma_base if self.sigma_base > 0 else float("inf")

    def __str__(self):
        verdict = "findable" if self.achieved else f"UNFINDABLE ({self.shortfall:.1f}x short)"
        return (f"{verdict}: sigma_base {self.sigma_base:.2e} vs target {self.target:.2e} "
                f"at rtol {self.rtol:.2e}, dk {self.dk:.3e} "
                f"(needs dk <= {self.required_dk:.3e})")


def _probe_lams(lo, hi, n=_N_PROBE):
    """Deterministic, irrationally-offset probe points in `[lo, hi]`.

    No RNG: `sigma_base` moved 14x on a seed change alone in the study, with the default seed
    being the lucky one on the shape that solved. A quantity that swings 14x on a seed is not a
    measurement.
    """
    i = np.arange(int(n))
    return lo + (hi - lo) * np.mod(_PHI_CONJ * (i + 1), 1.0)


def sigma_base(pencil, lam_lo, lam_hi, *, n=_N_PROBE):
    """The plateau: median `sigma_1` over off-resonance points in `[lam_lo, lam_hi]`.

    One cheap measurement -- about `n` evaluations -- needing no knowledge of the spectrum. The
    median is what makes that true: a probe landing in a dip pulls one sample far down and the
    median does not move, whereas a mean would.
    """
    lams = _probe_lams(float(lam_lo), float(lam_hi), n)
    return float(np.median([pencil.tension(float(x)) for x in lams]))


def findability_target(domain, dk, *, kappa=KAPPA, q=Q_VISIBLE):
    """The `sigma_base` a scan of step `dk` needs in order to SEE every dip.

        sigma_base >= (kappa / 2q) * (|Omega|/|dOmega|) * dk

    Everything on the right is known before any mode is located, which is the whole point: this
    is a precondition, not a diagnostic.
    """
    a_over_p = float(domain.area) / float(domain.perimeter)
    return (kappa / (2.0 * q)) * a_over_p * float(dk)


def required_dk(domain, sigma_base_value, *, kappa=KAPPA, q=Q_VISIBLE):
    """The scan step this plateau CAN support -- the other way to read the criterion.

    This is the number worth reporting to a user: "this domain needs `(k_hi - k_lo)/dk` scan
    points at this precision" is actionable, where "the solve may be incomplete" is not.
    """
    a_over_p = float(domain.area) / float(domain.perimeter)
    return float(sigma_base_value) * 2.0 * q / (kappa * a_over_p)


def calibrate_rtol(pencil, domain, lam_lo, lam_hi, *, dk, steps=5, n=_N_PROBE,
                   kappa=KAPPA, q=Q_VISIBLE, rtol_hi=None):
    """Check findability; if it fails, bisect `rtol` upward onto the computed target.

    Cost: `n` evaluations for the check. When it passes -- the common case -- nothing further
    is spent and `calibrated` is False. When it fails, `steps` more measurements of `n` probes
    each.

    The bisection is in `log10(rtol)` and is a bisection rather than a search because
    `sigma_base` is monotone in the truncation threshold: discarding more directions raises the
    smallest retained singular value and so raises the plateau.
    """
    rtol_hi = _rtol_ceiling(pencil.prec) if rtol_hi is None else float(rtol_hi)
    target = findability_target(domain, dk, kappa=kappa, q=q)
    s0 = sigma_base(pencil, lam_lo, lam_hi, n=n)
    used = n
    hist = [(float(pencil.rtol), s0, float("nan"))]

    if s0 >= target:
        return RtolCalibration(
            rtol=float(pencil.rtol), sigma_base=s0, target=target, dk=float(dk),
            achieved=True, calibrated=False,
            required_dk=required_dk(domain, s0, kappa=kappa, q=q),
            evaluations=used, history=tuple(hist))

    lo_e, hi_e = np.log10(max(float(pencil.rtol), _RTOL_LO)), np.log10(rtol_hi)
    best = (float(pencil.rtol), s0)
    for _ in range(int(steps)):
        mid_e = 0.5 * (lo_e + hi_e)
        trial = pencil.with_rtol(10.0 ** mid_e)
        s = sigma_base(trial, lam_lo, lam_hi, n=n)
        used += n
        hist.append((10.0 ** mid_e, s, float("nan")))
        if s >= target:
            best, hi_e = (10.0 ** mid_e, s), mid_e     # good enough; try to give back margin
        else:
            lo_e = mid_e
            if s > best[1]:
                best = (10.0 ** mid_e, s)

    return RtolCalibration(
        rtol=best[0], sigma_base=best[1], target=target, dk=float(dk),
        achieved=bool(best[1] >= target), calibrated=True,
        required_dk=required_dk(domain, best[1], kappa=kappa, q=q),
        evaluations=used, history=tuple(hist))


def warn_if_unfindable(cal, span_k=None):
    """Raise a `PrecisionWarning` naming the concrete remedy, or return quietly.

    The remedy is the point. "The solve may be incomplete" is not actionable; "this window
    needs 650 scan points at this precision" is, and it is exactly what the failing triangle
    was never told.
    """
    if cal.achieved:
        return
    need = (f", i.e. about {int(np.ceil(span_k / cal.required_dk))} scan points"
            if span_k else "")
    warnings.warn(
        f"the tension wells here are narrower than the scan can see: sigma_base "
        f"{cal.sigma_base:.2e} against a required {cal.target:.2e} at dk = {cal.dk:.3e} "
        f"({cal.shortfall:.1f}x short). Eigenvalues may be MISSED, and the tension at the "
        f"ones returned will not show it. Use dk <= {cal.required_dk:.3e}{need}.",
        PrecisionWarning, stacklevel=3)
