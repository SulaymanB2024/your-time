"""Audit coverage and performance locally, emitting aggregates only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import subprocess

from chronicle_rollup import private_write_if_changed, work_artifacts
from daily_analysis import ZONE, analyze
from data_quality import check_analysis
from behavior_analysis import build as build_behavior
from screen_context_tagging import allocate_visible_context, read_supported_tags
from topic_allocation import allocate
from secure_store import DB_PATH, STATE_DIR
from vision_capacity import nearest_rank, sustained_primary_nights
from vision_quality_eval import proxies

JOBS = (
    "secure-window-reader", "secure-mac-activity", "secure-screen-record",
    "secure-offline-ocr", "secure-features", "secure-phone-import",
    "project-file-watch", "project-git", "daily-analysis", "secure-text-analysis",
    "overnight-vision", "screen-storage-guard", "activity-dashboard-refresh",
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def job_status(label: str) -> dict:
    result = subprocess.run(["/bin/launchctl", "print", f"gui/{os.getuid()}/com.sulayman.{label}"],
                            capture_output=True, text=True, timeout=10)
    state = re.search(r"^\s*state = (.+)$", result.stdout, re.M)
    exit_code = re.search(r"^\s*last exit code = (\d+)$", result.stdout, re.M)
    return {"loaded": result.returncode == 0,
            "state": state.group(1) if state else None,
            "last_exit_code": int(exit_code.group(1)) if exit_code else None}


def day_metrics(report: dict) -> dict:
    mac = report["mac"]
    states = mac["state_sampled_seconds"]
    titled, untitled = states.get("active", 0), states.get("unattributed", 0)
    foreground = titled + untitled
    span = (datetime.fromisoformat(report["analyzed_through_utc"]) -
            datetime.fromisoformat(report["start_utc"])).total_seconds()
    return {
        "day_local": report["day_local"], "complete_day": report["complete_day"],
        "foreground_seconds": round(foreground, 3),
        "window_identified_seconds": titled, "window_unidentified_seconds": untitled,
        "window_identified_fraction": round(titled / foreground, 4) if foreground else None,
        "mac_observed_fraction": round(mac["observed_sampled_seconds"] / span, 4) if span else None,
        "mac_unknown_seconds": mac["unobserved_seconds"],
        "idle_seconds": states.get("idle", 0), "locked_seconds": states.get("locked", 0),
        "gaps_over_one_minute": len(mac["gaps_over_one_minute"]),
        "iphone_union_seconds": report["iphone"]["device_union_seconds"],
        "iphone_overlapping_session_seconds": report["iphone"]["overlap_seconds"],
        "raw_screenshots": report["visual"]["indexed_screenshots"],
        "ocr_status_counts": report["visual"]["ocr_status_counts"],
    }


def audit(days: int = 7) -> dict:
    now = datetime.now(timezone.utc)
    today = now.astimezone(ZONE).date()
    rows = []
    for age in reversed(range(days)):
        day = today - timedelta(days=age)
        analysis = analyze(day, now=now)
        row = day_metrics(analysis)
        topics = read_json(STATE_DIR / f"analyses/window-topics-{day.isoformat()}.json")
        row["data_quality"] = check_analysis(analysis,
            workstreams=allocate(analysis["mac"]["segments"], topics.get("tags", [])),
            screen_workstreams=allocate_visible_context(analysis["mac"]["segments"], read_supported_tags(day)),
            behavior=build_behavior(analysis["mac"]["segments"], topics.get("tags", [])))
        artifact = work_artifacts(day)
        row["work_artifacts"] = {key: artifact.get(key, 0) for key in (
            "local_commits", "file_events", "changed_files", "work_file_events",
            "generated_file_events_excluded")}
        synthesis = read_json(STATE_DIR / f"analyses/synthesis-{day.isoformat()}.json")
        row["synthesis"] = {"status": synthesis.get("status", "not_run"),
                            "block_status_counts": dict(Counter(block.get("status", "unknown")
                                                                 for block in synthesis.get("blocks", [])))}
        rows.append(row)
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        integrity = db.execute("PRAGMA quick_check").fetchone()[0]
        table_counts = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        for table in ("events", "screenshots", "vision_descriptions", "git_receipts",
                                      "project_file_events", "task_outcomes", "task_corrections", "calendar_items")}
        source_counts = dict(db.execute("SELECT source,COUNT(*) FROM events GROUP BY source"))
        screenshots = db.execute("SELECT path,timestamp_utc FROM screenshots").fetchall()
        missing = sum(not Path(path).is_file() for path, _ in screenshots)
        nonprivate = sum((Path(path).stat().st_mode & 0o077) != 0
                         for path, _ in screenshots if Path(path).is_file())
        orphan_descriptions = db.execute("SELECT COUNT(*) FROM vision_descriptions v LEFT JOIN "
                                        "screenshots s ON s.path=v.path WHERE s.path IS NULL").fetchone()[0]
        orphan_geometry = db.execute("SELECT COUNT(*) FROM ocr_geometry g LEFT JOIN "
                                    "screenshots s ON s.path=g.path WHERE s.path IS NULL").fetchone()[0]
        orphan_features = db.execute("SELECT COUNT(*) FROM screen_features f LEFT JOIN "
                                    "screenshots s ON s.path=f.path WHERE s.path IS NULL").fetchone()[0]
        future_events = db.execute("SELECT COUNT(*) FROM events WHERE julianday(timestamp_utc)>julianday(?)",
                                   ((now + timedelta(minutes=5)).isoformat(),)).fetchone()[0]
        invalid_event_durations = db.execute("SELECT COUNT(*) FROM events WHERE source IN "
            "('mac_window_sample','mac_idle','mac_locked','mac_window_legacy','mac_idle_legacy','iphone_app') "
            "AND (duration_seconds IS NULL OR duration_seconds<=0 OR duration_seconds>21600)").fetchone()[0]
        context_candidates = db.execute(
            "SELECT s.timestamp_utc,s.active_app,s.active_window,e.timestamp_utc,e.data_json "
            "FROM screenshots s LEFT JOIN events e ON e.rowid=(SELECT p.rowid FROM events p "
            "WHERE p.source='mac_window_sample' AND p.timestamp_utc<=s.timestamp_utc "
            "ORDER BY p.timestamp_utc DESC LIMIT 1) "
            "WHERE s.path LIKE ? AND s.active_window IS NOT NULL",
            (str(STATE_DIR / "screenshots") + "/%",)).fetchall()
        potential_title_backfills = 0
        for at, app, title, prior_at, prior_data in context_candidates:
            prior = json.loads(prior_data) if prior_data else {}
            if (prior_at and 0 <= (datetime.fromisoformat(at) - datetime.fromisoformat(prior_at)).total_seconds() <= 15
                    and prior.get("app") and prior["app"] != app and prior.get("title") == title):
                potential_title_backfills += 1
        model_rows = {}
        for model in [item[0] for item in db.execute("SELECT DISTINCT model_sha256 FROM vision_descriptions")]:
            stats, times = Counter(), []
            status_counts = dict(db.execute("SELECT status,COUNT(*) FROM vision_descriptions "
                                            "WHERE model_sha256=? GROUP BY status", (model,)))
            for caption, ocr, elapsed in db.execute(
                    "SELECT v.description,s.ocr_text,v.elapsed_seconds FROM vision_descriptions v "
                    "JOIN screenshots s ON s.path=v.path WHERE v.model_sha256=? AND v.status='complete'",
                    (model,)):
                flags = proxies(caption, ocr or "")
                for key in ("unclear", "completed_action_claim", "identifier_leak"):
                    stats[key] += int(flags[key])
                if elapsed and elapsed > 0:
                    times.append(elapsed)
            model_rows[model[:12]] = {
                "status_counts": status_counts,
                "median_complete_seconds": round(statistics.median(times), 1) if times else None,
                "p90_complete_seconds": round(nearest_rank(times, .9), 1) if times else None,
                "text_quality_proxies": dict(stats),
                "proxy_limit": "Flags are lexical checks, not accuracy scores or confirmed leaks/actions.",
            }
        for row in rows:
            day = datetime.fromisoformat(row["day_local"]).date()
            start = datetime.combine(day, datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
            end = start.astimezone(ZONE) + timedelta(days=1)
            until = end.astimezone(timezone.utc)
            row["unique_captioned_frames"] = db.execute(
                "SELECT COUNT(DISTINCT path) FROM vision_descriptions WHERE status='complete' "
                "AND timestamp_utc>=? AND timestamp_utc<?", (start.isoformat(), until.isoformat())).fetchone()[0]
            row["captioned_fraction_of_raw"] = round(row["unique_captioned_frames"] / row["raw_screenshots"], 4) if row["raw_screenshots"] else None
    finally:
        db.close()
    phone = read_json(STATE_DIR / "phone-quality-latest.json")
    mac = read_json(STATE_DIR / "mac-activity-status.json")
    reader = read_json(STATE_DIR / "window-reader-latest.json")
    capture = read_json(STATE_DIR / "screen-capture-status.json")
    file_watch = read_json(STATE_DIR / "project-file-latest-receipt.json")
    latest = read_json(STATE_DIR / "vision-fallback-latest-receipt.json")
    power = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True,
                           text=True, timeout=10).stdout
    memory = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True,
                            text=True, timeout=10).stdout
    free_match = re.search(r"System-wide memory free percentage:\s*(\d+)%", memory)
    nights = sustained_primary_nights()
    recent = [n for n in nights if n["completed"] + n["failed"] >= 50][-3:]
    return {
        "schema_version": 1, "audited_at_utc": now.isoformat(), "timezone": str(ZONE),
        "goal": "Account for daily behavior, focus and task time with explicit evidence limits; record confirmed outcomes separately from model guesses.",
        "days": rows, "integrity": {"sqlite_quick_check": integrity,
            "missing_indexed_images": missing, "nonprivate_indexed_images": nonprivate,
            "orphan_descriptions": orphan_descriptions, "orphan_ocr_geometry": orphan_geometry,
            "orphan_screen_features": orphan_features, "future_events_over_five_minutes": future_events,
            "invalid_event_durations": invalid_event_durations},
        "data_quality": {"days_checked": len(rows),
            "days_passed": sum(row["data_quality"]["status"] == "passed" for row in rows),
            "checks_run": sum(row["data_quality"]["checks_run"] for row in rows),
            "scope": "Accounting, bounds and joins; passing is not a semantic accuracy score."},
        "context_quality": {"potential_cross_app_title_backfills": potential_title_backfills,
            "native_titled_images_checked": len(context_candidates),
            "scope": "Potential older OCR metadata contamination; title coincidence is possible. Original missing titles cannot be reconstructed from this check."},
        "table_counts": table_counts, "event_source_counts": source_counts,
        "vision_models_by_hash_prefix": model_rows, "sustained_9b_nights": nights,
        "conservative_recent_9b_completed_capacity": min(n["estimated_completed_at_80_percent_window"] for n in recent) if recent else None,
        "runtime": {"jobs": {label: job_status(label) for label in JOBS},
            "disk_free_gib": round(shutil.disk_usage(STATE_DIR).free / 1024**3, 1),
            "load_average": [round(x, 2) for x in os.getloadavg()],
            "memory_free_percent": int(free_match.group(1)) if free_match else None,
            "ac_power": "AC Power" in power.splitlines()[0] if power.splitlines() else None,
            "window_reader_trusted": reader.get("trusted"),
            "window_reader_has_title": bool(reader.get("title")),
            "mac_sampler_identity_status": mac.get("window_identity_status"),
            "capture_checked_at_utc": capture.get("checked_at_utc"),
            "capture_blocked_reason": capture.get("capture_blocked_reason"),
            "file_watcher": {k: file_watch.get(k) for k in ("checked_at_utc", "status", "dropped", "coalesced_since_start", "ignored_generated_since_start")},
            "phone": {k: phone.get(k) for k in ("checked_at_utc", "quality_status", "source_age_minutes", "source_behind_ledger_minutes")},
            "latest_vision_launch": {k: latest.get(k) for k in ("started_at_utc", "finished_at_utc", "completed", "failed", "stop_reason")}},
        "evidence_limits": [
            "Device times can overlap; never add Mac and iPhone time as a single total.",
            "A missing interval is unknown, not idle; lack of a title is not lack of activity.",
            "Foreground samples require input within three minutes; long reading or calls can appear idle. They do not capture clicks, keystrokes, thought, off-device work or background audio.",
            "iPhone sync records provide app intervals, not phone screen contents, actions or confirmed Screen Time totals.",
            "Model text and similarity propagation infer context; even a complete caption can be wrong.",
            "Files and local commits may be produced by automation; neither proves the user personally completed or delivered a task.",
            "Calendar, reminders, explicit outcomes and browser-extension evidence remain absent until actually collected.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.days <= 31:
        parser.error("--days must be 1-31")
    report = audit(args.days)
    if args.write:
        private_write_if_changed(STATE_DIR / "chronicle-audit-latest.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
