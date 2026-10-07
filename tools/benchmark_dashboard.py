"""Benchmark dashboard calculations with synthetic records, never the live ledger."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import date, timedelta, timezone
from datetime import time as clock_time
from pathlib import Path

from repo_layout import REPO_ROOT, SOURCE_ROOT

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import dashboard_metrics
from chronicle_rollup import local_boundaries
from dashboard_metrics import activity_score, time_of_day


def synthetic_analysis(day: date, segments: int) -> dict:
    start, end = local_boundaries(day)
    rows = []
    spacing = (end - start) / max(1, segments)
    for index in range(segments):
        lower = start + spacing * index
        upper = min(end, lower + timedelta(seconds=5))
        rows.append({"start_utc": lower.isoformat(), "end_utc": upper.isoformat(),
                     "support_runs": [{"start_utc": lower.isoformat(), "end_utc": upper.isoformat()}],
                     "sampled_seconds": (upper - lower).total_seconds(),
                     "state": ("active", "unattributed", "idle")[index % 3],
                     "app": "Synthetic editor", "window": "Synthetic project"})
    phones = []
    for index in range(120):
        lower = start + timedelta(minutes=10 * index)
        phones.append({"start_utc": lower.isoformat(),
                       "end_utc": (lower + timedelta(minutes=15)).isoformat(),
                       "app": "Synthetic phone app"})
    return {"day_local": day.isoformat(), "mac": {"segments": rows},
            "iphone": {"sessions": phones}}


def measure(analysis: dict, iterations: int, score=activity_score, hourly=time_of_day) -> dict:
    _, end = local_boundaries(date.fromisoformat(analysis["day_local"]))
    samples = []
    for _ in range(iterations):
        before = time.perf_counter()
        result = score(analysis, {}, end)
        hours = hourly(analysis)
        samples.append(time.perf_counter() - before)
    return {"dataset": "synthetic", "segments": len(analysis["mac"]["segments"]),
            "segments_with_exact_support": sum("support_runs" in row for row in analysis["mac"]["segments"]),
            "phone_sessions": len(analysis["iphone"]["sessions"]),
            "iterations": iterations, "median_seconds": statistics.median(samples),
            "minimum_seconds": min(samples), "score_bins": len(result),
            "hourly_bins": len(hours["mac"]),
            "scope": "dashboard_calculations_only_not_model_inference"}


def baseline_functions(revision: str) -> dict:
    """Use calculation functions from an explicitly reviewed local Git revision."""
    if not re.fullmatch(r"[A-Za-z0-9_./~^{}-]{1,100}", revision) or revision.startswith("-"):
        raise ValueError("Invalid baseline revision")
    for filename in ("src/your_time/dashboard_metrics.py", "dashboard_metrics.py", "src/your_time/local_dashboard.py", "local_dashboard.py"):
        result = subprocess.run(["git", "show", f"{revision}:{filename}"],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            break
    else:
        raise ValueError("Baseline source is unavailable in local Git history")
    # Do not import the old controller or execute its top-level runtime code.
    names = {"app_name", "time_of_day", "activity_score", "_interval", "_overlapping_bins", "_observed_intervals"}
    nodes = [node for node in ast.parse(result.stdout).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if not {"app_name", "time_of_day", "activity_score"}.issubset({node.name for node in nodes}):
        raise ValueError("Baseline calculation functions are unavailable")
    namespace = vars(dashboard_metrics).copy()
    namespace.update(clock_time=clock_time, timezone=timezone)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, "<reviewed-dashboard-baseline>", "exec"), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--day", type=date.fromisoformat, default=date(2026, 9, 29))
    parser.add_argument("--baseline-ref", help="Reviewed calculation code in local Git history")
    args = parser.parse_args()
    if not 1 <= args.segments <= 10000 or not 1 <= args.iterations <= 10:
        parser.error("segments must be 1–10000 and iterations 1–10")
    analysis = synthetic_analysis(args.day, args.segments)
    report = measure(analysis, args.iterations)
    if args.baseline_ref:
        baseline = baseline_functions(args.baseline_ref)
        _, end = local_boundaries(args.day)
        if (baseline["activity_score"](analysis, {}, end) != activity_score(analysis, {}, end)
                or baseline["time_of_day"](analysis) != time_of_day(analysis)):
            raise ValueError("Synthetic projections differ from the reviewed baseline")
        previous = measure(analysis, args.iterations, baseline["activity_score"], baseline["time_of_day"])
        report.update(baseline_ref=args.baseline_ref,
                      baseline_median_seconds=previous["median_seconds"],
                      equivalent_outputs=True,
                      speedup=previous["median_seconds"] / report["median_seconds"])
    print(json.dumps(report))


if __name__ == "__main__":
    main()
