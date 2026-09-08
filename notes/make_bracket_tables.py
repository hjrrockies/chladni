"""Turn the filter run's log into the LaTeX tables the report includes.

Generated rather than transcribed, so every number in `bracket_report.pdf` is traceable to
`filter_run.log` and cannot drift from it.

Exploratory; not tracked.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

CASE_RE = re.compile(r"^--- (\S+)\s+(\d+) eigenvalues\s+\[([^\]]+)\]\s+\((\d+)s\)")
ROW_RE = re.compile(r"^\s*(\w+) \| (.+)$")

FAMILY = {"A": "A --- isolated", "B": "B --- pair ladder", "C": "C --- clusters of 3 and 4",
          "D": "D --- exact degeneracy", "E": "E --- mixed window", "S": "S --- multiplicity 6"}
PRECS = ["1e-04", "1e-08", "1e-11", "1e-14"]
METHOD_TEX = {"uniform_grid": r"\texttt{uniform\_grid}", "lappy": r"\texttt{lappy}",
              "lappy_repaired": r"\texttt{lappy\_repaired}", "lipschitz": r"\texttt{lipschitz}",
              "swing_cells": r"\texttt{swing\_cells}", "swing_direct": r"\texttt{swing\_direct}",
              "single_bracket": r"\texttt{single\_bracket}", "fine_grid": r"\texttt{fine\_grid}"}


def parse(path):
    """-> [(label, n_eig, window, seconds, {method: [(oracle, auto_ok, auto_cost), ...]})]"""
    cases, cur = [], None
    for line in open(path):
        m = CASE_RE.match(line)
        if m:
            cur = dict(label=m.group(1), n=int(m.group(2)), window=m.group(3),
                       secs=int(m.group(4)), rows={})
            cases.append(cur)
            continue
        if cur is None or "|" not in line or "prec=" in line or "oracle" in line:
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        cells = []
        for chunk in rest.split("|"):
            toks = chunk.split()
            if len(toks) < 2:
                continue
            o = None if toks[0] == "FAIL" else int(toks[0])
            a_tok = toks[1]
            a_ok = a_tok.endswith("*")
            a = int(a_tok.rstrip("*")) if a_ok else None
            cells.append((o, a_ok, a))
        if cells:
            cur["rows"][name] = cells
    return cases


def _cell(o, a_ok, a):
    o_s = r"\textbf{--}" if o is None else f"{o}"
    a_s = f"{a}" if a_ok and a is not None else r"\textbf{--}"
    return f"{o_s} & {a_s}"


def case_table(case):
    lab = case["label"].replace("_", r"\_")
    out = [r"\begin{center}", r"\small",
           r"\begin{tabular}{l" + "rr" * len(PRECS) + "}", r"\toprule",
           r"& \multicolumn{2}{c}{$\pr=10^{-4}$} & \multicolumn{2}{c}{$10^{-8}$}"
           r" & \multicolumn{2}{c}{$10^{-11}$} & \multicolumn{2}{c}{$10^{-14}$} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
           r"method & orc & aut & orc & aut & orc & aut & orc & aut \\", r"\midrule"]
    for name in ["uniform_grid", "lappy", "lappy_repaired", "lipschitz",
                 "swing_cells", "swing_direct", "single_bracket", "fine_grid"]:
        if name not in case["rows"]:
            continue
        if name == "single_bracket":
            out.append(r"\midrule")
        cells = case["rows"][name]
        out.append(METHOD_TEX[name] + " & " +
                   " & ".join(_cell(*c) for c in cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


def summary_table(cases):
    """AUTONOMOUS pass count per method per prec. The only column that may be aggregated."""
    # seed with every method SEEN, or one that never passes autonomously vanishes from the
    # table entirely -- which would silently hide the worst result rather than reporting it
    seen = {name for c in cases for name in c["rows"]}
    tot = {name: [0] * len(PRECS) for name in seen}
    n = 0
    for c in cases:
        if c["label"].startswith("S:"):
            continue
        n += 1
        for name, cells in c["rows"].items():
            for i, (_, a_ok, _) in enumerate(cells[:len(PRECS)]):
                if a_ok:
                    tot[name][i] += 1
    out = [r"\begin{center}", r"\begin{tabular}{lrrrr}", r"\toprule",
           r"method & $\pr=10^{-4}$ & $10^{-8}$ & $10^{-11}$ & $10^{-14}$ \\", r"\midrule"]
    for name in ["uniform_grid", "lappy", "lappy_repaired", "lipschitz",
                 "swing_cells", "swing_direct", "single_bracket", "fine_grid"]:
        if name not in tot:
            continue
        if name == "single_bracket":
            out.append(r"\midrule")
        out.append(METHOD_TEX[name] + " & " +
                   " & ".join(f"{v}/{n}" for v in tot[name]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out), n


def merge(cases):
    """One entry per case LABEL, with the method rows merged.

    The controls are run in a second pass (they have no knobs to sweep), so every case appears
    twice in the concatenated log -- once with the six candidates and once with the two
    controls. Merging by label is what puts them in one table instead of two, and stops the
    aggregate counting each case twice.
    """
    out = {}
    for c in cases:
        cur = out.get(c["label"])
        if cur is None:
            out[c["label"]] = dict(c)
            out[c["label"]]["rows"] = dict(c["rows"])
        else:
            cur["rows"].update(c["rows"])
            cur["secs"] += c["secs"]
    return list(out.values())


def main(log, dest):
    cases = merge(parse(log))
    if not cases:
        sys.exit(f"no cases parsed from {log}")
    fams = defaultdict(list)
    for c in cases:
        fams[c["label"][0]].append(c)

    L = [r"\section{Results}", "",
         r"\texttt{orc} is the cheapest of the tested settings that reached coverage $1.0$ with "
         r"no impure bracket; \texttt{aut} is the single setting fixed across the whole suite "
         r"before any per-case result was seen. \textbf{--} means no setting succeeded within "
         r"the budget of 1500 evaluations. Only the \texttt{aut} column may be aggregated: "
         r"\texttt{orc} is a best-of-$K$ statistic and is biased upward in $K$.", ""]
    summ, ncase = summary_table(cases)
    L += [r"\subsection{Aggregate: autonomous passes}", "",
          f"Across the {ncase} swept cases (the multiplicity-6 stress case excluded), "
          r"counting only the autonomous setting:", "", summ, ""]
    for key in ["A", "B", "C", "D", "E", "S"]:
        if key not in fams:
            continue
        L += [r"\subsection{Family " + FAMILY.get(key, key) + "}", ""]
        for c in sorted(fams[key], key=lambda c: c["label"]):
            lab = c["label"].replace("_", r"\_")
            L += [f"\\paragraph{{{lab}}} {c['n']} eigenvalues in "
                  f"$[{c['window']}]$; sweep took {c['secs']}\\,s.", "",
                  case_table(c), ""]
    open(dest, "w").write("\n".join(L) + "\n")
    print(f"wrote {dest}: {len(cases)} cases, {ncase} swept")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
