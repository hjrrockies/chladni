"""Run the retired test modules on demand: their support code lives in `tests/`."""
import os
import sys

_TESTS = os.path.join(os.path.dirname(__file__), "..", "..", "tests")
sys.path.insert(0, os.path.abspath(_TESTS))
