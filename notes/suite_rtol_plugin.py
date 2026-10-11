"""pytest plugin: override `rect_suite.SUITE_RTOL` from CHLADNI_SUITE_RTOL.  (2026-10-11)

For `notes/suite_rtol_sweep.md`: does the suite's ulp sensitivity come from a pencil truncated
for 1e-14? Load with `PYTEST_ADDOPTS="-p suite_rtol_plugin"` and `notes/` on PYTHONPATH.
"""
import os
import sys


def pytest_configure(config):
    v = os.environ.get("CHLADNI_SUITE_RTOL")
    if not v:
        return
    sys.path.insert(0, os.path.join(str(config.rootpath), "tests"))
    import rect_suite
    rect_suite.SUITE_RTOL = float(v)


def pytest_report_header(config):
    v = os.environ.get("CHLADNI_SUITE_RTOL")
    return f"rect_suite.SUITE_RTOL overridden: {v}" if v else None
