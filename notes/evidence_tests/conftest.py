"""Run the retired test modules on demand: their support code lives in `tests/`."""
import os
import sys

# The evidence here runs `PREC_LADDER_FULL` to 1e-14, so it keeps the suite pencil cut for 1e-14
# (`rect_suite.FINEST_PREC`, which defaults to 1e-10 since 2026-10-11). Set before anything
# imports `rect_suite`.
os.environ.setdefault("CHLADNI_SUITE_FINEST_PREC", "1e-14")

_TESTS = os.path.join(os.path.dirname(__file__), "..", "..", "tests")
sys.path.insert(0, os.path.abspath(_TESTS))
