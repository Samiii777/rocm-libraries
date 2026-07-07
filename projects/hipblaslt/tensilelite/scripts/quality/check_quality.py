#!/usr/bin/env python3
"""Non-gating quality + AI-friendliness report for the tensilelite Tensile/ tree.

This single check folds two measurements into one table:

  * **complexity / size** via `lizard <https://github.com/terryyin/lizard>`_ --
    function cyclomatic-complexity (CCN) and file-NLOC violator counts,
    Python-only.
  * **AI-friendliness** -- a stdlib AST scanner (see :mod:`ai_friendliness`)
    producing 21 readability signals (file size, deep nesting, swallowed
    errors, missing seam tests, duplicated literals, cross-feature imports,
    typing escape hatches, etc.).

It is **report-only**: it always exits 0 so it can be wired into CI without
gating.  Targets start at the current measurement and are meant to be tightened
deliberately over time.  Pass ``--enforce`` to opt into gating (exit 1 when any
metric exceeds its target).

The report is Markdown so it renders directly in a GitHub Actions job summary
(``$GITHUB_STEP_SUMMARY``); pass ``--summary-file`` to append it there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ai_friendliness import scan_tree  # noqa: E402

# Default: the Tensile/ Python tree, two levels up from scripts/quality/.
DEFAULT_ROOT = _HERE.parent.parent / "Tensile"

# lizard thresholds (Python-only).
CCN_LIMIT = 15          # a function with CCN above this is a "complexity violator"
NLOC_LIMIT = 509        # a file with NLOC above this is a "size violator"

# Targets file lives next to this script.  It records the agreed-upon current
# baseline; metrics at or below target are "OK", above target are over-budget.
TARGETS_FILE = _HERE / "quality_targets.json"


# --------------------------------------------------------------------------- #
# lizard complexity / size
# --------------------------------------------------------------------------- #


def lizard_metrics(root: Path) -> Dict[str, int]:
    """Return complexity / size violator counts using lizard (Python only).

    Falls back to sentinel -1 values if lizard is not importable so the report
    still renders (report-only tooling must never hard-fail on a missing dep).
    """
    try:
        import lizard
    except ImportError:
        return {
            "functions_analyzed": -1,
            "ccn_violators": -1,
            "max_ccn": -1,
            "avg_ccn": -1,
            "nloc_violators": -1,
            "max_file_nloc": -1,
            "total_nloc": -1,
        }

    functions = 0
    ccn_violators = 0
    max_ccn = 0
    ccn_sum = 0
    nloc_violators = 0
    max_file_nloc = 0
    total_nloc = 0

    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        info = lizard.analyze_file(str(py))
        total_nloc += info.nloc
        max_file_nloc = max(max_file_nloc, info.nloc)
        if info.nloc > NLOC_LIMIT:
            nloc_violators += 1
        for fn in info.function_list:
            functions += 1
            ccn_sum += fn.cyclomatic_complexity
            max_ccn = max(max_ccn, fn.cyclomatic_complexity)
            if fn.cyclomatic_complexity > CCN_LIMIT:
                ccn_violators += 1

    avg_ccn = round(ccn_sum / functions) if functions else 0
    return {
        "functions_analyzed": functions,
        "ccn_violators": ccn_violators,
        "max_ccn": max_ccn,
        "avg_ccn": avg_ccn,
        "nloc_violators": nloc_violators,
        "max_file_nloc": max_file_nloc,
        "total_nloc": total_nloc,
    }


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

# Metrics that act as gate candidates (lower is better).  Purely informational
# counts (file counts, totals, maxima) are excluded from enforcement.
GATED_METRICS = {
    "ccn_violators",
    "nloc_violators",
    "very_large_files",
    "large_files",
    "deeply_nested_functions",
    "wide_signatures",
    "many_exit_functions",
    "swallowed_errors",
    "broad_except_clauses",
    "bare_except_clauses",
    "mutable_default_args",
    "global_statements",
    "star_imports",
    "typing_any_uses",
    "type_ignore_comments",
    "cross_feature_imports",
    "duplicated_literals",
    "missing_seam_tests",
}


def load_targets() -> Dict[str, int]:
    if TARGETS_FILE.exists():
        try:
            return json.loads(TARGETS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def collect(root: Path) -> Dict[str, int]:
    metrics: Dict[str, int] = {}
    metrics.update(lizard_metrics(root))
    metrics.update(scan_tree(root))
    return metrics


def render_markdown(
    root: Path, metrics: Dict[str, int], targets: Dict[str, int], enforce: bool
) -> Tuple[str, bool]:
    """Return (markdown, over_budget)."""
    lines: List[str] = []
    lines.append("## tensilelite Tensile/ quality + AI-friendliness report")
    lines.append("")
    lines.append(f"Scanned tree: `{root}`  ")
    lines.append(
        "_Report-only: this check never blocks merges unless run with "
        "`--enforce`._"
    )
    lines.append("")
    lines.append("| Metric | Current | Target | Status |")
    lines.append("| --- | ---: | ---: | :---: |")

    over_budget = False
    for name in metrics:
        cur = metrics[name]
        tgt = targets.get(name)
        if tgt is None:
            status = "info"
            tgt_disp = "-"
        else:
            tgt_disp = str(tgt)
            if name in GATED_METRICS and cur > tgt:
                status = "OVER"
                over_budget = True
            else:
                status = "ok"
        lines.append(f"| {name} | {cur} | {tgt_disp} | {status} |")

    lines.append("")
    if over_budget:
        lines.append(
            "> Some gated metrics exceed their target. In report-only mode this "
            "is informational; run with `--enforce` to gate."
        )
    else:
        lines.append("> All gated metrics are at or below target.")
    lines.append("")

    fail = enforce and over_budget
    return "\n".join(lines), fail


def render_targets_snapshot(metrics: Dict[str, int]) -> str:
    """Emit a targets JSON seeded at the current measurement (for --write-targets)."""
    snap = {k: metrics[k] for k in metrics if k in GATED_METRICS}
    return json.dumps(snap, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="Python tree to scan (default: tensilelite Tensile/).",
    )
    parser.add_argument(
        "--enforce", action="store_true",
        help="Gate: exit 1 if any gated metric exceeds its target.",
    )
    parser.add_argument(
        "--summary-file", type=Path, default=None,
        help="Append the Markdown report to this file (e.g. $GITHUB_STEP_SUMMARY).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also print the raw metrics as JSON to stderr.",
    )
    parser.add_argument(
        "--write-targets", action="store_true",
        help="Write quality_targets.json seeded at the current measurement and exit.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.exists():
        print(f"error: scan root does not exist: {root}", file=sys.stderr)
        # Report-only tooling: do not hard-fail unless enforcing.
        return 1 if args.enforce else 0

    metrics = collect(root)

    if args.write_targets:
        TARGETS_FILE.write_text(render_targets_snapshot(metrics))
        print(f"wrote {TARGETS_FILE}")
        return 0

    targets = load_targets()
    markdown, fail = render_markdown(root, metrics, targets, args.enforce)

    print(markdown)
    if args.summary_file:
        with args.summary_file.open("a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True), file=sys.stderr)

    # Report-only unless --enforce.
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
