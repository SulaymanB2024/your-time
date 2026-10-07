"""Benchmark dashboard calculations with synthetic records, never the live ledger."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
            "phone_sessions": len(analysis["iphone"]["sessions"]),
            "iterations": iterations, "median_seconds": statistics.median(samples),
            "minimum_seconds": min(samples), "score_bins": len(result),
            "hourly_bins": len(hours["mac"]),
            "scope": "dashboard_calculations_only_not_model_inference"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--day", type=date.fromisoformat, default=date(2026, 9, 29))
    args = parser.parse_args()
    if not 1 <= args.segments <= 10000 or not 1 <= args.iterations <= 10:
        parser.error("segments must be 1–10000 and iterations 1–10")
    print(json.dumps(measure(synthetic_analysis(args.day, args.segments), args.iterations)))


if __name__ == "__main__":
    main()
