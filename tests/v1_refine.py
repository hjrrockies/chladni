"""Stage 2 on `candidates.md`'s scored seam -- a re-export of `chladni.mps.refine`.

TEST SUPPORT ONLY, and TEMPORARY. The implementation moved into the package; this shim exists
so that the promotion changed no test file, which is what makes an unchanged suite the proof
that the move was a move. It goes when the candidates are repointed at `chladni.mps`.

Re-exports, never redefines: `test_refine.py` reaches two privates and `test_searches.py`
argues about a third, so the whole surface is carried across rather than the public half.
"""
from chladni.mps.refine import (  # noqa: F401
    MAX_EVALS,
    RefineResult,
    _accept,
    _deepest_inside,
    _vertex,
    _xatol,
    refine,
)

__all__ = ["RefineResult", "refine"]
