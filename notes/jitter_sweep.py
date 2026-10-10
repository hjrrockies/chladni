"""Run chosen tests under N ulp-jitter seeds and tabulate which fail where.  (2026-10-09)

Each seed is a deterministic stand-in for another platform's LAPACK (`tests/ulp_jitter.py`). The
question: do the failure sets Linux CI showed (two discrete sets, alternating across legs and
reruns with one BLAS thread) reproduce on macOS under ulp perturbations alone?

    .venv/bin/python notes/jitter_sweep.py TARGETS.txt [N_SEEDS] [SCALE] [WORKERS] [MARKEXPR] [OUT]

TARGETS.txt holds pytest node ids, one per line. Writes `notes/jitter_sweep.json` and prints, per
test, how many seeds failed it, and per seed, its failure set.
"""
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def run_seed(seed, scale, targets, markexpr=""):
    env = dict(os.environ, CHLADNI_ULP_JITTER=f"{seed}:{scale}" if seed else "",
               VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as fh:
        xml = fh.name
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", markexpr, "-p", "no:randomly",
                    "-p", "no:cacheprovider", f"--junitxml={xml}", *targets],
                   env=env, capture_output=True, text=True)
    failed, passed = [], 0
    for tc in ET.parse(xml).getroot().iter("testcase"):
        name = f"{tc.get('classname').split('.')[-1]}::{tc.get('name')}"
        if tc.find("failure") is not None or tc.find("error") is not None:
            failed.append(name)
        elif tc.find("skipped") is None:
            passed += 1
    os.unlink(xml)
    return seed, sorted(failed), passed


def main():
    targets = [t.strip() for t in open(sys.argv[1]) if t.strip()]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    markexpr = sys.argv[5] if len(sys.argv) > 5 else ""
    out = sys.argv[6] if len(sys.argv) > 6 else os.path.join(os.path.dirname(__file__),
                                                               "jitter_sweep.json")
    seeds = [0] + list(range(1, n + 1))          # 0 = unjittered control
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda s: run_seed(s, scale, targets, markexpr), seeds))
    per_test = defaultdict(list)
    for seed, failed, _ in results:
        for t in failed:
            per_test[t].append(seed)
    print(f"scale {scale:g} ulps, {n} seeds (+ control seed 0)\n")
    print("per test (seeds that failed it):")
    for t, ss in sorted(per_test.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len([s for s in ss if s]):2d}/{n}  {t}   seeds {ss}")
    print("\nper seed (failure set):")
    sets = defaultdict(list)
    for seed, failed, passed in results:
        sets[tuple(failed)].append(seed)
        print(f"  seed {seed:2d}: {len(failed)} failed, {passed} passed")
    print(f"\n{len(sets)} distinct failure sets:")
    for fs, ss in sorted(sets.items(), key=lambda kv: -len(kv[1])):
        print(f"  seeds {ss}: {list(fs) or 'none'}")
    json.dump(dict(scale=scale, n=n, targets=targets, markexpr=markexpr,
                   results=[dict(seed=s, failed=f, passed=p) for s, f, p in results]),
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
