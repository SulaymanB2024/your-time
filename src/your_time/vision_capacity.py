"""Estimate nightly frame capacity from local timings and actual selected images."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from private_io import write_json
from secure_store import DB_PATH, STATE_DIR
from vision_batch import select_images

ZONE = ZoneInfo("America/Chicago")
RESULTS_PATH = STATE_DIR / "vision-quality-eval/results.json"
NIGHTLY_RECEIPT = STATE_DIR / "vision-nightly-receipt.json"
LATEST_RECEIPT = STATE_DIR / "vision-latest-receipt.json"
PRIMARY_RECEIPT = STATE_DIR / "vision-fallback-latest-receipt.json"
HISTORY_DIR = STATE_DIR / "vision-run-receipts"
CAPACITY_RECEIPT = STATE_DIR / "vision-capacity-plan.json"
from overnight_schedule import TEXT_SECONDS, TOTAL_SECONDS, VISION_SECONDS

WINDOW_SECONDS = VISION_SECONDS  # 00:30–07:00 vision; 07:00–08:00 text analysis.
UTILIZATION = 0.80
CONFIGURED_LIMIT = 2000  # Safety ceiling, not an expected throughput.
CONFIGURED_INTERVAL_SECONDS = 10


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise ValueError("nonempty values and percentile in (0,1] required")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def frame_capacity(seconds_per_image: float, *, window_seconds: int = WINDOW_SECONDS,
                   utilization: float = UTILIZATION) -> int:
    if seconds_per_image <= 0 or not 0 < utilization <= 1:
        raise ValueError("positive image time and utilization in (0,1] required")
    return math.floor(window_seconds * utilization / seconds_per_image)


def sustained_primary_nights() -> list[dict]:
    """Keep successful historical measurements when a later launch skips."""
    nights = {}
    if not HISTORY_DIR.is_dir():
        return []
    seen = set()
    for path in sorted(HISTORY_DIR.glob("vision-fallback-*.json")):
        item = json.loads(path.read_text())
        started = item.get("started_at_utc")
        if not started or started in seen or not item.get("finished_at_utc"):
            continue
        seen.add(started)
        elapsed = float(item.get("elapsed_seconds", 0))
        attempts = int(item.get("completed", 0)) + int(item.get("failed", 0))
        if not attempts or elapsed <= 0:
            continue
        night = datetime.fromisoformat(started).astimezone(ZONE).date().isoformat()
        row = nights.setdefault(night, {"night_local": night, "runs": 0, "completed": 0,
                                      "failed": 0, "elapsed_seconds": 0.0,
                                      "sensitive_skipped": 0, "backfill_completed": 0})
        row["runs"] += 1
        for key in ("completed", "failed", "sensitive_skipped", "backfill_completed"):
            row[key] += int(item.get(key, 0))
        row["elapsed_seconds"] += elapsed
    for row in nights.values():
        attempts = row["completed"] + row["failed"]
        row["elapsed_seconds"] = round(row["elapsed_seconds"], 1)
        row["seconds_per_attempt_including_overhead"] = round(row["elapsed_seconds"] / attempts, 2)
        safe_attempts = frame_capacity(row["elapsed_seconds"] / attempts)
        row["estimated_attempts_at_80_percent_window"] = safe_attempts
        row["estimated_completed_at_80_percent_window"] = math.floor(safe_attempts * row["completed"] / attempts)
    return sorted(nights.values(), key=lambda row: row["night_local"])


def analysis_for_day(day) -> dict:
    results = json.loads(RESULTS_PATH.read_text())
    benchmarks = {}
    for model in results["models"]:
        runs = [image.get("models", {}).get(model, {}) for image in results["images"]]
        seconds = [run["elapsed_seconds"] for run in runs if run.get("status") == "complete"]
        if not seconds:
            continue
        p90 = nearest_rank(seconds, 0.90)
        capacity = frame_capacity(p90)
        benchmarks[model] = {
            "sample_size": len(seconds),
            "median_seconds": round(statistics.median(seconds), 2),
            "p90_seconds_nearest_rank": round(p90, 2),
            "pilot_capacity_at_80_percent_window": capacity,
            "all_described_interval_seconds_at_8h_active": math.ceil(8 * 3600 / capacity) if capacity else None,
            "all_described_interval_seconds_at_10h_active": math.ceil(10 * 3600 / capacity) if capacity else None,
            "all_described_interval_seconds_at_12h_active": math.ceil(12 * 3600 / capacity) if capacity else None,
            "eligible_for_current_overnight_memory_floor": model != "27b_q3",
        }
    selected = {str(seconds): len(select_images(day, seconds, None)) for seconds in (10, 20, 30, 60)}
    start = datetime.combine(day, time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=ZONE).astimezone(timezone.utc)
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        capture_counts = dict(connection.execute(
            "SELECT ocr_status,count(*) FROM screenshots WHERE timestamp_utc >= ? "
            "AND timestamp_utc < ? AND path LIKE ? GROUP BY ocr_status",
            (start.isoformat(), end.isoformat(), str(STATE_DIR / "screenshots") + "/%"),
        ).fetchall())
    finally:
        connection.close()
    raw_count = sum(capture_counts.values())
    selected_current = selected[str(CONFIGURED_INTERVAL_SECONDS)]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "day_local": day.isoformat(),
        "window_local": "00:30-07:00 America/Chicago",
        "total_window_local": "00:30-08:00 America/Chicago",
        "vision_window_local": "00:30-07:00 America/Chicago",
        "total_overnight_seconds": TOTAL_SECONDS,
        "text_reserved_seconds": TEXT_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "utilization_for_provisional_capacity": UTILIZATION,
        "configured_model": "Qwen3.5-9B-Q8_0",
        "configured_safety_ceiling": CONFIGURED_LIMIT,
        "configured_selection_interval_seconds": CONFIGURED_INTERVAL_SECONDS,
        "ocr_complete_candidates_by_interval_seconds": selected,
        "secure_capture_counts_by_ocr_status": capture_counts,
        "current_day_coverage": {
            "raw_secure_screenshots": raw_count,
            "selected_at_configured_interval": selected_current,
            "selected_fraction_of_raw": round(selected_current / raw_count, 3) if raw_count else None,
            "selected_exceeds_safety_ceiling": selected_current > CONFIGURED_LIMIT,
        },
        "benchmarks": benchmarks,
        "important_distinction": "Raw screenshots remain in the private archive; these limits apply to model-selected descriptions, not capture or OCR.",
        "provisional": True,
    }
    report["sustained_9b_nights"] = sustained_primary_nights()
    recent = [row for row in report["sustained_9b_nights"] if row["completed"] + row["failed"] >= 50][-3:]
    if recent:
        report["recent_conservative_9b_completed_capacity"] = min(
            row["estimated_completed_at_80_percent_window"] for row in recent)
        report["capacity_uncertainty"] = "Measured nights differ in contention, image complexity, retries and power; this is a provisional planning bound."
    if NIGHTLY_RECEIPT.exists() and not PRIMARY_RECEIPT.exists():
        night = json.loads(NIGHTLY_RECEIPT.read_text())
        day_receipts = list(night.get("day_receipts", []))
        if not night.get("finished_at_utc") and LATEST_RECEIPT.exists():
            latest = json.loads(LATEST_RECEIPT.read_text())
            if (latest.get("started_at_utc", "") >= night.get("started_at_utc", "")
                    and latest.get("day_local") not in {item.get("day_local") for item in day_receipts}):
                day_receipts.append(latest)
        completed = sum(item.get("completed", 0) for item in day_receipts)
        failed = sum(item.get("failed", 0) for item in day_receipts)
        if completed or failed:
            started = datetime.fromisoformat(night["started_at_utc"])
            finished = datetime.fromisoformat(night.get("finished_at_utc") or report["generated_at_utc"])
            elapsed = (finished - started).total_seconds()
            connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            try:
                timings = [row[0] for row in connection.execute(
                    "SELECT elapsed_seconds FROM vision_descriptions WHERE updated_at_utc >= ? "
                    "AND updated_at_utc <= ? AND status IN ('complete','incomplete','timeout','model_error') "
                    "AND elapsed_seconds IS NOT NULL",
                    (night["started_at_utc"], finished.isoformat()),
                )]
            finally:
                connection.close()
            p90 = nearest_rank(timings, 0.90) if timings else None
            report["actual_latest_night"] = {
                "started_at_utc": night["started_at_utc"],
                "finished_at_utc": night.get("finished_at_utc"),
                "completed": completed,
                "failed": failed,
                "sensitive_skipped": sum(item.get("sensitive_skipped", 0) for item in day_receipts),
                "selected": sum(item.get("selected", 0) for item in day_receipts),
                "elapsed_seconds": round(elapsed, 1),
                "seconds_per_completed_including_overhead": round(elapsed / completed, 2) if completed else None,
                "seconds_per_model_attempt_including_overhead": round(elapsed / (completed + failed), 2),
                "observed_p90_model_attempt_seconds": round(p90, 2) if p90 else None,
                "p90_capacity_at_80_percent_window": frame_capacity(p90) if p90 else None,
                "stop_reason": night.get("stop_reason"),
            }
            report["provisional"] = not bool(night.get("finished_at_utc")) or completed < 50
    if PRIMARY_RECEIPT.exists():
        primary = json.loads(PRIMARY_RECEIPT.read_text())
        report["latest_primary_launch"] = {key: primary.get(key) for key in (
            "started_at_utc", "finished_at_utc", "day_local", "completed", "failed", "stop_reason")}
        if primary.get("day_local") != day.isoformat():
            primary = {}
        completed = primary.get("completed", 0)
        failed = primary.get("failed", 0)
        attempts = completed + failed
        if attempts:
            started = datetime.fromisoformat(primary["started_at_utc"])
            finished = datetime.fromisoformat(primary.get("finished_at_utc") or report["generated_at_utc"])
            elapsed = (finished - started).total_seconds()
            per_attempt = elapsed / attempts
            safe_attempts = frame_capacity(per_attempt)
            success_rate = completed / attempts
            report["actual_latest_night"] = {
                "model": "9b_q8",
                "day_local": primary.get("day_local"),
                "started_at_utc": primary["started_at_utc"],
                "finished_at_utc": primary.get("finished_at_utc"),
                "planned_candidates": primary.get("selected", 0),
                "attempted": attempts,
                "completed": completed,
                "failed": failed,
                "sensitive_skipped": primary.get("sensitive_skipped", 0),
                "direct_completed": primary.get("direct_completed"),
                "quality_completed": primary.get("quality_completed"),
                "elapsed_seconds": round(elapsed, 1),
                "seconds_per_attempt_including_overhead": round(per_attempt, 2),
                "observed_completion_rate": round(success_rate, 3),
                "estimated_attempts_at_80_percent_window": safe_attempts,
                "estimated_completed_at_80_percent_window_if_all_direct": math.floor(safe_attempts * success_rate),
                "stop_reason": primary.get("stop_reason"),
            }
            report["provisional"] = not bool(primary.get("finished_at_utc")) or attempts < 50
    historical = []
    if HISTORY_DIR.is_dir():
        for path in sorted(HISTORY_DIR.glob("vision-*.json")):
            item = json.loads(path.read_text())
            matching = ([entry for entry in item.get("day_receipts", [])
                         if entry.get("day_local") == day.isoformat()]
                        if path.name.startswith("vision-nightly-") else
                        [item] if item.get("day_local") == day.isoformat() else [])
            if matching and item.get("finished_at_utc"):
                historical.append({
                    "kind": "9b" if path.name.startswith("vision-fallback-") else "2b",
                    "started_at_utc": item["started_at_utc"],
                    "elapsed_seconds": round(sum(entry.get("elapsed_seconds", 0) for entry in matching)
                                             if item.get("day_receipts") else item.get("elapsed_seconds", 0), 1),
                    "completed": sum(entry.get("completed", 0) for entry in matching),
                    "failed": sum(entry.get("failed", 0) for entry in matching),
                    "selection_interval_seconds": matching[0].get("interval_seconds")
                    if item.get("day_receipts") else None,
                    "provenance": item.get("receipt_provenance", "original private run receipt"),
                })
    if historical:
        connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            status_by_path = {}
            for path, status in connection.execute(
                "SELECT path,status FROM vision_descriptions WHERE timestamp_utc>=? AND timestamp_utc<?",
                (start.isoformat(), end.isoformat()),
            ):
                status_by_path.setdefault(path, set()).add(status)
        finally:
            connection.close()
        actual_interval = next((item["selection_interval_seconds"] for item in historical
                                if item["selection_interval_seconds"]), CONFIGURED_INTERVAL_SECONDS)
        selected_paths = [row[0] for row in select_images(day, actual_interval, None)]
        complete = sum("complete" in status_by_path.get(path, ()) for path in selected_paths)
        sensitive = sum("complete" not in status_by_path.get(path, ())
                        and "sensitive_skipped" in status_by_path.get(path, ()) for path in selected_paths)
        unresolved = len(selected_paths) - complete - sensitive
        total_elapsed = sum(item["elapsed_seconds"] for item in historical)
        model_attempts = sum(item["completed"] + item["failed"] for item in historical)
        seconds_per_attempt = total_elapsed / model_attempts if model_attempts else None
        report["combined_night"] = {
            "runs": historical,
            "selection_interval_seconds": actual_interval,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "model_attempts_across_runs": model_attempts,
            "selected_unique_screenshots": len(selected_paths),
            "complete_unique_screenshots": complete,
            "sensitive_skipped_unique_screenshots": sensitive,
            "unresolved_unique_screenshots": unresolved,
            "complete_fraction_of_nonsensitive": round(complete / (len(selected_paths) - sensitive), 3)
            if len(selected_paths) > sensitive else None,
            "effective_seconds_per_attempt_including_retries": round(seconds_per_attempt, 2)
            if seconds_per_attempt else None,
            "sustainable_model_attempts_per_night_at_80_percent_window": frame_capacity(seconds_per_attempt)
            if seconds_per_attempt else None,
            "full_window_model_attempts_without_reserve": math.floor(WINDOW_SECONDS / seconds_per_attempt)
            if seconds_per_attempt else None,
            "provisional_reason": "one completed night; future activity mix and thermal load may differ",
        }
    return report


def private_write(value: dict) -> None:
    write_json(CAPACITY_RECEIPT, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-ago", type=int, default=1)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.days_ago < 0:
        parser.error("--days-ago must be nonnegative")
    day = datetime.now(ZONE).date() - timedelta(days=args.days_ago)
    report = analysis_for_day(day)
    if args.write:
        private_write(report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
