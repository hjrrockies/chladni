"""Ulp-level jitter of the MPS pencil: a stand-in for "another platform's LAPACK".  TEST SUPPORT ONLY.

**Why this models a platform.** LAPACK's factorizations are backward stable: the result one
kernel computes is the EXACT result for an input perturbed by about `eps * ||A||`, and a different
kernel (another CPU's OpenBLAS, Accelerate) is the exact result for a different such perturbation.
So perturbing the assembled collocation matrices by a few ulps, with a pattern fixed per seed,
gives a family of deterministic "platforms" to run the solver on.

Linux CI on 2026-10-09 showed two discrete failure sets alternating across legs and reruns of
identical code with one BLAS thread -- the signature of a search that is discontinuous in its
inputs (`notes/nondeterminism.md`) meeting kernels that differ in the last bits. This lets that be
reproduced and measured on macOS, in minutes rather than one CI draw at a time.

    CHLADNI_ULP_JITTER=7 pytest ...          # seed 7, the default scale of 4 ulps
    CHLADNI_ULP_JITTER=7:16 pytest ...       # seed 7, 16 ulps

or from a script: `import ulp_jitter; ulp_jitter.install(seed, scale)`.

**Deterministic per seed and per `k`**, as a real kernel is: the same seed evaluated at the same
wavenumber always sees the same perturbation, so caches, repeated reads and the search's own
consistency checks behave as they would on one platform. Only `chladni.mps.pencil.assemble` is
patched -- every pencil read (tensions, coefficients, rank decisions) goes through it. The
Moler-Payne certificate's own boundary evaluations (`chladni.integrals.cauchy`) are NOT perturbed.
"""
from __future__ import annotations

import os

import numpy as np

_ORIG = None
_STATE = {}


def _perturb(A, rng, scale):
    eps = np.finfo(float).eps
    return A * (1.0 + scale * eps * rng.standard_normal(A.shape))


def install(seed, scale=4.0):
    """Patch `chladni.mps.pencil.assemble` with a seeded, per-`k` ulp perturbation."""
    import chladni.mps.pencil as P
    global _ORIG
    if _ORIG is None:
        _ORIG = P.assemble
    orig = _ORIG
    _STATE.update(seed=int(seed), scale=float(scale))

    def assemble(bdry, interior, k):
        A_B, A_I = orig(bdry, interior, k)
        kbits = int.from_bytes(np.float64(k).tobytes(), "little")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), kbits]))
        return _perturb(np.asarray(A_B, dtype=float), rng, scale), \
            _perturb(np.asarray(A_I, dtype=float), rng, scale)

    P.assemble = assemble


def uninstall():
    import chladni.mps.pencil as P
    global _ORIG
    if _ORIG is not None:
        P.assemble = _ORIG
        _ORIG = None
    _STATE.clear()


def active():
    """`(seed, scale)` when installed, else None."""
    return (_STATE["seed"], _STATE["scale"]) if _STATE else None


def install_from_env(var="CHLADNI_ULP_JITTER"):
    """Install from `SEED` or `SEED:SCALE` in the environment; a no-op when unset or empty."""
    spec = os.environ.get(var, "").strip()
    if not spec:
        return None
    seed, _, scale = spec.partition(":")
    install(int(seed), float(scale) if scale else 4.0)
    return active()
