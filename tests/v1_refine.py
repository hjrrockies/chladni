"""Stage 2 on `candidates.md`'s scored seam -- a re-export of `chladni.mps.refine`.

TEST SUPPORT ONLY, and TEMPORARY. The implementation moved into the package; this shim exists
so that the promotion changed no test file. It goes when the candidates are repointed at
`chladni.mps`.

The whole module namespace is carried across rather than an explicit list, matching
`v1_bracket.py` and `v1_solve.py`. It started as a list and that list went stale the first time
a constant was added upstream, which is the argument.
"""
import chladni.mps.refine as _promoted
from chladni.mps.refine import *  # noqa: F401,F403

# Every module-level name, privates included; dunders skipped so this file keeps its own
# `__doc__` and the `__all__` below.
globals().update({k: v for k, v in vars(_promoted).items() if not k.startswith("__")})

__all__ = ["RefineResult", "refine", "DELTA_SAFETY"]
