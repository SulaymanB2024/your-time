"""Build a private, self-contained activity dashboard without a web server."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from behavior_analysis import aggregate as aggregate_behavior
from behavior_analysis import build as build_behavior
from chronicle_rollup import (
    DAYS,
    MONTHS,
    YEARS,
    calendar_summary,
    work_artifacts,
)
from chronicle_rollup import read_json as read_rollup
from chronicle_rollup import refresh as refresh_chronicle
from daily_analysis import ZONE, analyze
from daily_focus import build as focus_build
from dashboard_metrics import PHONE_NAMES as PHONE_NAMES
from dashboard_metrics import activity_score, app_name, mac_windows, time_of_day
from dashboard_metrics import mac_behavior as mac_behavior
from dashboard_timeline import build_timeline
from data_quality import check_analysis
from local_synthesis import block_projection, fingerprint, summary_is_current
from private_io import atomic_write, open_private_file, prepare_directory
from screen_context_tagging import allocate_visible_context, read_supported_tags
from secure_store import STATE_DIR
from task_corrections import active_outcomes
from task_corrections import review as correction_review
from task_corrections import summary as correction_summary
from topic_allocation import allocate, broad_context
from window_topic_tagging import window_id

DASHBOARD_DIR = STATE_DIR / "dashboard"
INDEX = DASHBOARD_DIR / "index.html"
MANIFEST = DASHBOARD_DIR / "manifest.json"
LOCK = STATE_DIR / "dashboard-refresh.lock"
ANALYSIS_DIR = STATE_DIR / "analyses"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def private_write(path: Path, payload: bytes) -> None:
    atomic_write(path, payload)


def runtime_snapshot(day=None) -> dict:
    day = day or datetime.now(ZONE).date()
    capture = read_json(STATE_DIR / "screen-capture-status.json")
    mac_collector = read_json(STATE_DIR / "mac-activity-status.json")
    ocr = read_json(STATE_DIR / "ocr-latest-receipt.json")
    features = read_json(STATE_DIR / "screen-feature-latest-receipt.json")
    project_files = read_json(STATE_DIR / "project-file-latest-receipt.json")
    project_git = read_json(STATE_DIR / "project-git-latest-receipt.json")
    screen_topics = read_json(ANALYSIS_DIR / f"screen-context-{day.isoformat()}.json")
    storage = read_json(STATE_DIR / "screen-storage-status.json")
    phone = read_json(STATE_DIR / "phone-quality-latest.json")
    vision = read_json(STATE_DIR / "vision-fallback-latest-receipt.json")
    disk = shutil.disk_usage(STATE_DIR)
    try:
        battery = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True,
                                 text=True, timeout=5, check=True).stdout
        power = "AC" if battery.splitlines() and "AC Power" in battery.splitlines()[0] else "Battery"
    except (OSError, subprocess.SubprocessError):
        power = "Unknown"
    return {
        "capture_checked_at_utc": capture.get("checked_at_utc"),
        "mac_collector_checked_at_utc": mac_collector.get("checked_at_utc"),
        "mac_collector_status": mac_collector.get("status", "not_checked"),
        "mac_consecutive_untitled_samples": mac_collector.get("consecutive_untitled_samples"),
        "mac_window_identity_status": mac_collector.get("window_identity_status", "not_checked"),
        "mac_accessibility_access": mac_collector.get("accessibility_access"),
        "mac_screen_capture_access": mac_collector.get("screen_capture_access"),
        "ocr_checked_at_utc": ocr.get("checked_at_utc"),
        "ocr_pending_after_run": ocr.get("pending_after_run"),
        "ocr_failed_last_run": ocr.get("failed"),
        "ocr_geometry_pending": ocr.get("geometry_pending"),
        "screen_feature_pending": features.get("pending_after"),
        "project_file_checked_at_utc": project_files.get("checked_at_utc"),
        "project_file_status": project_files.get("status", "not_checked"),
        "project_git_checked_at_utc": project_git.get("checked_at_utc"),
        "project_git_status": project_git.get("status", "not_checked"),
        "screen_context_selected": screen_topics.get("selected"),
        "screen_context_processed": screen_topics.get("selected_processed",
                                                      len(screen_topics.get("tags", []))),
        "screen_context_stop_reason": screen_topics.get("stop_reason"),
        "capture_locked": capture.get("screen_locked"),
        "capture_blocked_reason": capture.get("capture_blocked_reason"),
        "screenshots_saved_last_check": capture.get("screenshots_saved_this_check"),
        "storage_checked_at_utc": storage.get("checked_at_utc"),
        "storage_below_threshold": storage.get("below_threshold"),
        "disk_free_gib": round(disk.free / 1024**3, 1),
        "disk_floor_gib": round(storage.get("stop_below_bytes", 10 * 1024**3) / 1024**3, 1),
        "power": power,
        "phone_checked_at_utc": phone.get("checked_at_utc"),
        "phone_quality_status": phone.get("quality_status", "not_checked"),
        "phone_source_age_minutes": (
            round(float(phone["source_age_minutes"]) + max(0.0, (
                datetime.now(timezone.utc) - datetime.fromisoformat(phone["checked_at_utc"])
            ).total_seconds() / 60), 1)
            if phone.get("source_age_minutes") is not None and phone.get("checked_at_utc")
            else phone.get("source_age_minutes")),
        "phone_source_behind_ledger_minutes": phone.get("source_behind_ledger_minutes"),
        "overnight_started_at_utc": vision.get("started_at_utc"),
        "overnight_finished_at_utc": vision.get("finished_at_utc"),
        "overnight_stop_reason": vision.get("stop_reason"),
        "overnight_completed": vision.get("completed"),
    }


def daily_snapshot(day, now: datetime) -> dict:
    analysis = analyze(day, now=now)
    focus = focus_build(day, now=now)
    synthesis = read_json(ANALYSIS_DIR / f"synthesis-{day.isoformat()}.json")
    window_topics = read_json(ANALYSIS_DIR / f"window-topics-{day.isoformat()}.json")
    screen_context = read_json(ANALYSIS_DIR / f"screen-context-{day.isoformat()}.json")
    screen_similarity = read_json(ANALYSIS_DIR / f"screen-similarity-{day.isoformat()}.json")
    screen_tags = read_supported_tags(day)
    topic_by_id = {item["id"]: item["topic"] for item in window_topics.get("tags", [])
                   if item.get("status") == "model_inference"}
    window_rows = mac_windows(analysis)
    for item in window_rows:
        item["topic"] = topic_by_id.get(item["id"])
    workstreams = allocate(analysis["mac"]["segments"], window_topics.get("tags", []))
    screen_workstreams = allocate_visible_context(analysis["mac"]["segments"], screen_tags)
    current_inputs = {block["id"]: fingerprint(block_projection(block)) for block in focus["blocks"]}
    inferred = {row["block_id"]: row["result"] for row in synthesis.get("blocks", [])
                if row.get("status") == "complete" and "result" in row
                and row.get("block_id") in current_inputs
                and row.get("input_sha256") == current_inputs[row["block_id"]]}
    synthesis_current = (set(current_inputs) == {row.get("block_id") for row in synthesis.get("blocks", [])}
        and all(row.get("input_sha256") == current_inputs.get(row.get("block_id"))
                for row in synthesis.get("blocks", [])))
    summary_current = summary_is_current(synthesis, current_inputs)
    summary_usable = summary_current or ("summary_status" not in synthesis
        and synthesis.get("status") == "complete" and synthesis_current)
    synthesis_coverage = dict(synthesis.get("coverage", {}))
    if synthesis:
        synthesis_coverage.update(total_blocks=len(focus["blocks"]), complete_blocks=len(inferred),
            sampled_seconds=round(sum(block["sampled_seconds"] for block in focus["blocks"]), 1),
            complete_seconds=round(sum(block["sampled_seconds"] for block in focus["blocks"]
                                       if block["id"] in inferred), 1))
        supported_ids = set(synthesis.get("summary_evidence_ids", [])) if summary_current else set()
        synthesis_coverage.update(summary_evidence_blocks=len(supported_ids),
            summary_evidence_seconds=round(sum(block["sampled_seconds"] for block in focus["blocks"]
                                               if block["id"] in supported_ids), 1))
    blocks = []
    categories = Counter()
    for block in focus["blocks"]:
        kind = (inferred.get(block["id"]) or {}).get("activity_kind") or "unclear"
        categories[kind] += block["sampled_seconds"]
        dominant = block["top_windows"][0] if block["top_windows"] else None
        topic = None
        if dominant and dominant["sampled_seconds"] >= block["sampled_seconds"] * 0.5:
            topic = (topic_by_id.get(window_id(block["app"], dominant["title"]))
                     or broad_context(block["app"], dominant["title"]))
        blocks.append({"id": block["id"], "start_utc": block["start_utc"],
                       "end_utc": block["end_utc"], "app": app_name(block["app"]),
                       "sampled_seconds": block["sampled_seconds"],
                       "window_count": block["distinct_window_count"],
                       "screenshots": block["screenshot_count"],
                       "top_windows": block["top_windows"][:3],
                       "inference": inferred.get(block["id"]),
                       "workstream": topic})
    visual = analysis["visual"]
    behavior = build_behavior(analysis["mac"]["segments"], window_topics.get("tags", []))
    return {
        "day": day.isoformat(), "complete_day": analysis["complete_day"],
        "data_quality": check_analysis(analysis, workstreams=workstreams,
            screen_workstreams=screen_workstreams, focus_blocks=focus["blocks"], behavior=behavior),
        "mac_active_seconds": analysis["mac"]["state_sampled_seconds"].get("active", 0),
        "mac_unattributed_seconds": analysis["mac"]["state_sampled_seconds"].get("unattributed", 0),
        "mac_app_only_seconds": analysis["mac"].get("app_only_seconds", 0),
        "mac_foreground_seconds": (analysis["mac"]["state_sampled_seconds"].get("active", 0)
                                   + analysis["mac"]["state_sampled_seconds"].get("unattributed", 0)),
        "mac_idle_seconds": analysis["mac"]["state_sampled_seconds"].get("idle", 0),
        "mac_locked_seconds": analysis["mac"]["state_sampled_seconds"].get("locked", 0),
        "mac_unobserved_seconds": analysis["mac"]["unobserved_seconds"],
        "mac_apps": [{**item, "name": app_name(item["app"])}
                     for item in analysis["mac"]["app_sampled_seconds"]],
        "mac_windows": window_rows,
        "iphone_focus_seconds": analysis["iphone"]["device_union_seconds"],
        "iphone_apps": [{**item, "name": app_name(item["app"])}
                        for item in analysis["iphone"]["app_totals"]],
        "iphone_intervals": analysis["iphone"]["interval_count"],
        "browser_domains": analysis.get("browser", {}).get("domains", []),
        "browser_tracked_seconds": analysis.get("browser", {}).get("tracked_seconds", 0),
        "screenshots": visual["indexed_screenshots"],
        "ocr_complete": visual["ocr_status_counts"].get("complete", 0),
        "ocr_pending": visual["new_capture_ocr_pending"],
        "vision_statuses": visual["vision_status_counts"],
        "hourly": time_of_day(analysis),
        "score": activity_score(analysis, topic_by_id, now, screen_tags),
        "timeline": build_timeline(analysis, behavior, app_name),
        "behavior": behavior,
        "work_artifacts": work_artifacts(day),
        "calendar": calendar_summary(day),
        "activity_categories": [{"kind": kind, "seconds": round(seconds, 1)}
                                for kind, seconds in categories.most_common()],
        "workstreams": workstreams,
        "screen_workstreams": screen_workstreams,
        "user_task_labels": correction_summary(day, analysis),
        "task_outcomes": active_outcomes(day),
        "review_candidates": correction_review(day, analysis)[:12],
        "screen_context_status": screen_context.get("status", "not_run"),
        "screen_context_selected": screen_context.get("selected"),
        "screen_context_processed": screen_context.get("selected_processed",
                                                       len(screen_context.get("tags", []))),
        "screen_similarity_matches": len(screen_similarity.get("propagated", [])),
        "workstream_status": window_topics.get("status", "not_run"),
        "blocks": blocks,
        "synthesis_status": ("partial" if summary_current and not synthesis_current else
                             synthesis.get("status", "not_run") if synthesis_current or not synthesis
                             else "stale_evidence"),
        "synthesis_coverage": synthesis_coverage,
        "summary_scope": synthesis.get("summary_scope") if summary_current else None,
        "summary_evidence_blocks": len(synthesis.get("summary_evidence_ids", [])) if summary_current else 0,
        "themes": synthesis.get("themes", []) if summary_usable else [],
        "candidate_outcomes": synthesis.get("candidate_outcomes", []) if summary_usable else [],
        "verified_accomplishments": [],
    }


def snapshot(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    refresh_chronicle(now=now)
    local_today = now.astimezone(ZONE).date()
    days = [daily_snapshot(local_today - timedelta(days=offset), now)
            for offset in range(6, -1, -1)]
    quality = read_json(STATE_DIR / "vision-quality-eval/assistant-visual-review.json")
    def readable_period(path):
        item = read_rollup(path)
        for app in item.get("mac_apps", []):
            app["name"] = app_name(app.get("app"))
        for app in item.get("iphone_apps", []):
            app["name"] = app_name(app.get("app"))
        return item
    months = [readable_period(path) for path in sorted(MONTHS.glob("*.json"))]
    years = [readable_period(path) for path in sorted(YEARS.glob("*.json"))]
    history = []
    for path in sorted(DAYS.glob("*.json")):
        item = read_rollup(path)
        mac, phone = item["mac"], item["iphone"]
        history.append({
            "day": item["day_local"], "complete_day": item["complete_day"], "compact_only": True,
            "mac_active_seconds": mac["active_seconds"], "mac_unattributed_seconds": mac["unattributed_seconds"],
            "mac_foreground_seconds": mac["active_seconds"] + mac["unattributed_seconds"],
            "mac_app_only_seconds": mac.get("app_only_seconds", 0), "mac_idle_seconds": mac["idle_seconds"],
            "mac_locked_seconds": mac["locked_seconds"], "mac_unobserved_seconds": mac["unknown_seconds"],
            "mac_apps": [{**app, "name": app_name(app["app"])} for app in mac["app_totals"]],
            "iphone_focus_seconds": phone["focus_union_seconds"], "iphone_intervals": phone["interval_count"],
            "iphone_apps": [{**app, "name": app_name(app["app"])} for app in phone["app_totals"]],
            "workstreams": item["workstreams"], "screen_workstreams": item.get("screen_workstreams", []),
            "user_task_labels": item.get("user_task_labels", []), "task_outcomes": item.get("task_outcomes", []),
            "behavior": item.get("behavior", {}), "work_artifacts": item.get("work_artifacts", {}),
            "calendar": item.get("calendar", {}), "browser_domains": item.get("browser", {}).get("domains", []),
            "blocks": [], "score": [], "hourly": {}, "screenshots": item["visual"]["screenshots"],
            "ocr_complete": item["visual"]["ocr_complete"], "ocr_pending": item["visual"]["ocr_pending"],
        })
    return {
        "schema_version": 1, "generated_at_utc": now.isoformat(),
        "timezone": str(ZONE), "days": days, "history": history, "months": months, "years": years,
        "week_behavior": aggregate_behavior(days),
        "runtime": runtime_snapshot(local_today),
        "caption_eval": quality.get("summary", {}),
        "definitions": {
            "mac_active": "Mac samples with recent input and a window title. Three minutes without input is classified as idle, including possible reading or calls. This does not prove attention or task completion.",
            "mac_app_only": "Foreground app is known, but the window or task is not. This time also appears within window-unknown time.",
            "mac_unobserved": "Time without usable Mac samples. It is unknown, not idle time.",
            "iphone_focus": "Union of locally synced iPhone app focus intervals; it may lag and can overlap Mac time.",
            "progress": "Local model suggestions are unverified. Confirmed accomplishments require an independent receipt or your confirmation.",
        },
    }


from dashboard_ui import render


def main() -> None:
    os.umask(0o077)
    prepare_directory(STATE_DIR)
    fd = open_private_file(LOCK)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "another_dashboard_refresh_is_running"}))
            return
        data = snapshot()
        payload = render(data)
        receipt = {"generated_at_utc": data["generated_at_utc"], "days": len(data["days"]),
                   "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                   "status": "private_dashboard_written"}
        private_write(INDEX, payload)
        private_write(MANIFEST, (json.dumps(receipt, indent=2) + "\n").encode())
        print(json.dumps(receipt, sort_keys=True))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
