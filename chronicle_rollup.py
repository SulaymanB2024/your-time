"""Persist compact day, month, and year views of the private activity ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from daily_analysis import ZONE, analyze, union_seconds
from private_io import atomic_write, prepare_directory, validate_file
from screen_context_tagging import allocate_visible_context, read_supported_tags
from secure_store import DB_PATH, STATE_DIR
from task_corrections import active_outcomes
from task_corrections import summary as correction_summary
from topic_allocation import allocate

ROOT = STATE_DIR / "chronicle"
DAYS = ROOT / "days"
MONTHS = ROOT / "months"
YEARS = ROOT / "years"
VERSION = 3


def private_write_if_changed(path: Path, value: dict) -> bool:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    prepare_directory(path.parent)
    validate_file(path)
    if path.is_file() and path.read_bytes() == payload:
        return False
    atomic_write(path, payload)
    return True


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def local_boundaries(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
    return start, end


def unique_caption_count(start: datetime, end: datetime) -> int:
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return db.execute("SELECT COUNT(DISTINCT path) FROM vision_descriptions "
                          "WHERE status='complete' AND timestamp_utc>=? AND timestamp_utc<?",
                          (start.isoformat(), end.isoformat())).fetchone()[0]
    finally:
        db.close()


def workstream_id(label: str, status: str = "") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
    digest = hashlib.sha256(json.dumps([label, status], ensure_ascii=False).encode()).hexdigest()[:12]
    return f"{normalized[:60] or 'topic'}-{digest}"


def work_artifacts(day: date) -> dict:
    start, end = local_boundaries(day)
    if not DB_PATH.is_file():
        return {"local_commits": 0, "file_events": 0, "changed_files": 0, "repositories": []}
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        commits = db.execute(
            "SELECT repo_path,COUNT(*) FROM git_receipts "
            "WHERE committed_at_utc>=? AND committed_at_utc<? GROUP BY repo_path ORDER BY COUNT(*) DESC",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        file_rows = db.execute(
            "SELECT project_root,relative_path,COUNT(*) FROM project_file_events "
            "WHERE timestamp_utc>=? AND timestamp_utc<? GROUP BY project_root,relative_path",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    except sqlite3.OperationalError:
        return {"local_commits": 0, "file_events": 0, "changed_files": 0, "repositories": []}
    finally:
        db.close()
    from project_paths import is_work_path
    work_rows = [(root, path, count) for root, path, count in file_rows if is_work_path(path)]
    return {"local_commits": sum(count for _, count in commits),
            "file_events": sum(count for _, _, count in file_rows),
            "changed_files": len(work_rows),
            "work_file_events": sum(count for _, _, count in work_rows),
            "generated_file_events_excluded": sum(count for _, path, count in file_rows
                                                   if not is_work_path(path)),
            "repositories": [{"name": Path(path).name, "commits": count}
                             for path, count in commits[:8]],
            "interpretation": "Local commits and file changes are artifacts, not proof of delivery."}


def calendar_summary(day: date) -> dict:
    scope = read_json(STATE_DIR / "calendar-scope.json")
    enabled = bool(scope.get("events") or scope.get("reminders"))
    if not enabled or not DB_PATH.is_file():
        return {"enabled": enabled, "scheduled_seconds": 0,
                "event_count": 0, "reminders_marked_complete": 0}
    start, end = local_boundaries(day)
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        events = db.execute(
            "SELECT start_utc,end_utc,all_day FROM calendar_items WHERE source='event' "
            "AND start_utc<? AND end_utc>?",
            (end.isoformat(), start.isoformat()),
        ).fetchall()
        reminders = db.execute(
            "SELECT COUNT(*) FROM calendar_items WHERE source='reminder' AND completed=1 "
            "AND start_utc>=? AND start_utc<?",
            (start.isoformat(), end.isoformat()),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        events, reminders = [], 0
    finally:
        db.close()
    intervals = []
    for at, until, all_day in events:
        if all_day:
            continue
        lower = max(start, datetime.fromisoformat(at))
        upper = min(end, datetime.fromisoformat(until))
        if lower < upper and (datetime.fromisoformat(until) - datetime.fromisoformat(at)).total_seconds() < 24 * 3600:
            intervals.append((lower, upper))
    return {"enabled": enabled, "scheduled_seconds": round(union_seconds(intervals), 3),
            "event_count": len(events), "reminders_marked_complete": reminders,
            "interpretation": "Scheduled time is context, not observed attendance."}


def day_summary(day: date, *, now: datetime | None = None) -> dict:
    from behavior_analysis import build as build_behavior
    from behavior_analysis import compact
    report = analyze(day, now=now)
    start, full_end = local_boundaries(day)
    analyzed_end = datetime.fromisoformat(report["analyzed_through_utc"])
    elapsed = max(0.0, (analyzed_end - start).total_seconds())
    mac = report["mac"]["state_sampled_seconds"]
    observed = report["mac"]["observed_sampled_seconds"]
    window_topics = read_json(STATE_DIR / f"analyses/window-topics-{day.isoformat()}.json")
    workstreams = [{"id": workstream_id(item["label"], item["status"]), **item}
                   for item in allocate(report["mac"]["segments"], window_topics.get("tags", []))]
    screen_context = read_json(STATE_DIR / f"analyses/screen-context-{day.isoformat()}.json")
    screen_similarity = read_json(STATE_DIR / f"analyses/screen-similarity-{day.isoformat()}.json")
    screen_workstreams = [{"id": workstream_id(item["label"], item["status"]), **item}
                          for item in allocate_visible_context(report["mac"]["segments"],
                                                                read_supported_tags(day))]
    visual = report["visual"]
    return {
        "schema_version": VERSION, "day_local": day.isoformat(), "timezone": str(ZONE),
        "behavior": compact(build_behavior(report["mac"]["segments"], window_topics.get("tags", []))),
        "start_utc": start.isoformat(), "end_utc": full_end.isoformat(),
        "analyzed_through_utc": analyzed_end.isoformat(),
        "day_length_seconds": (full_end - start).total_seconds(),
        "analyzed_seconds": round(elapsed, 3),
        "complete_day": report["complete_day"],
        "mac": {"active_seconds": mac.get("active", 0),
                "unattributed_seconds": mac.get("unattributed", 0),
                "app_only_seconds": report["mac"].get("app_only_seconds", 0),
                "idle_seconds": mac.get("idle", 0),
                "locked_seconds": mac.get("locked", 0),
                "observed_seconds": observed,
                "unknown_seconds": report["mac"]["unobserved_seconds"],
                "coverage_fraction": round(observed / elapsed, 4) if elapsed else None,
                "app_totals": report["mac"]["app_sampled_seconds"]},
        "iphone": {"focus_union_seconds": report["iphone"]["device_union_seconds"],
                   "overlap_seconds": report["iphone"]["overlap_seconds"],
                   "interval_count": report["iphone"]["interval_count"],
                   "app_totals": report["iphone"]["app_totals"]},
        "browser": report.get("browser", {"domains": [], "tracked_seconds": 0, "event_count": 0}),
        "visual": {"screenshots": visual["indexed_screenshots"],
                   "ocr_complete": visual["ocr_status_counts"].get("complete", 0),
                   "ocr_pending": visual["new_capture_ocr_pending"],
                   "distinct_captioned_screenshots": unique_caption_count(start, full_end)},
        "workstreams": workstreams,
        "work_artifacts": work_artifacts(day),
        "calendar": calendar_summary(day),
        "screen_workstreams": screen_workstreams,
        "user_task_labels": correction_summary(day, report),
        "task_outcomes": active_outcomes(day),
        "screen_context_status": screen_context.get("status", "not_run"),
        "screen_similarity_matches": len(screen_similarity.get("propagated", [])),
        "workstream_tagging_status": window_topics.get("status", "not_run"),
        "coverage_status": ("partial_day" if not report["complete_day"] else
                            "mac_observed" if observed else "no_mac_samples"),
        "interpretation": "Mac and iPhone time are separate; workstream labels are local model inferences.",
    }


def earliest_day() -> date | None:
    if not DB_PATH.is_file():
        return None
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        values = [db.execute("SELECT MIN(timestamp_utc) FROM events").fetchone()[0],
                  db.execute("SELECT MIN(timestamp_utc) FROM screenshots").fetchone()[0]]
    finally:
        db.close()
    timestamps = [datetime.fromisoformat(value).astimezone(ZONE).date()
                  for value in values if value]
    return min(timestamps) if timestamps else None


def aggregate(days: list[dict], *, period: str, key: str) -> dict:
    from behavior_analysis import aggregate as aggregate_behavior
    mac_apps = Counter()
    phone_apps = Counter()
    browser_domains = Counter()
    workstreams = Counter()
    screen_workstreams = Counter()
    user_labels = Counter()
    confirmed_results = Counter()
    for item in days:
        for app in item["mac"]["app_totals"]:
            mac_apps[app["app"]] += app["seconds"]
        for app in item["iphone"]["app_totals"]:
            phone_apps[app["app"]] += app["focus_seconds"]
        for site in item.get("browser", {}).get("domains", []):
            browser_domains[site["site_host"]] += site["seconds"]
        for tag in item["workstreams"]:
            workstreams[(tag["id"], tag["label"], tag.get("status", "specific_model"))] += tag["sampled_seconds"]
        for tag in item.get("screen_workstreams", []):
            screen_workstreams[(tag["id"], tag["label"])] += tag["sampled_seconds"]
        for tag in item.get("user_task_labels", []):
            user_labels[tag["label"]] += tag["sampled_seconds"]
        for outcome in item.get("task_outcomes", []):
            confirmed_results[outcome["label"]] += 1
    return {
        "schema_version": VERSION, "period": period, "key": key,
        "behavior": aggregate_behavior(days),
        "timezone": str(ZONE), "days_recorded": len(days),
        "complete_days": sum(item["complete_day"] for item in days),
        "mac_days_observed": sum(item["mac"]["observed_seconds"] > 0 for item in days),
        "iphone_days_with_intervals": sum(item["iphone"]["interval_count"] > 0 for item in days),
        "mac_active_seconds": round(sum(item["mac"]["active_seconds"] for item in days), 3),
        "mac_unattributed_seconds": round(sum(item["mac"].get("unattributed_seconds", 0) for item in days), 3),
        "mac_foreground_seconds": round(sum(item["mac"]["active_seconds"]
                                        + item["mac"].get("unattributed_seconds", 0)
                                        for item in days), 3),
        "mac_app_only_seconds": round(sum(item["mac"].get("app_only_seconds", 0) for item in days), 3),
        "mac_observed_seconds": round(sum(item["mac"]["observed_seconds"] for item in days), 3),
        "mac_unknown_seconds": round(sum(item["mac"]["unknown_seconds"] for item in days), 3),
        "iphone_focus_seconds": round(sum(item["iphone"]["focus_union_seconds"] for item in days), 3),
        "browser_tracked_seconds": round(sum(browser_domains.values()), 3),
        "browser_domains": [{"site_host": host, "seconds": round(seconds, 3)}
                            for host, seconds in browser_domains.most_common()],
        "local_commits": sum(item.get("work_artifacts", {}).get("local_commits", 0) for item in days),
        "file_events": sum(item.get("work_artifacts", {}).get("file_events", 0) for item in days),
        "sum_daily_changed_files": sum(item.get("work_artifacts", {}).get("changed_files", 0) for item in days),
        "scheduled_seconds": round(sum(item.get("calendar", {}).get("scheduled_seconds", 0) for item in days), 3),
        "reminders_marked_complete": sum(item.get("calendar", {}).get("reminders_marked_complete", 0) for item in days),
        "calendar_enabled": any(item.get("calendar", {}).get("enabled", False) for item in days),
        "screenshots": sum(item["visual"]["screenshots"] for item in days),
        "captioned_screenshots": sum(item["visual"]["distinct_captioned_screenshots"] for item in days),
        "mac_apps": [{"app": app, "seconds": round(seconds, 3)}
                     for app, seconds in mac_apps.most_common()],
        "iphone_apps": [{"app": app, "seconds": round(seconds, 3)}
                        for app, seconds in phone_apps.most_common()],
        "workstreams": [{"id": identity, "label": label,
                         "sampled_seconds": round(seconds, 3),
                         "status": status}
                        for (identity, label, status), seconds in workstreams.most_common()],
        "screen_workstreams": [{"id": identity, "label": label,
                                "sampled_seconds": round(seconds, 3),
                                "status": "screen_suggested"}
                               for (identity, label), seconds in screen_workstreams.most_common()],
        "user_task_labels": [{"label": label, "sampled_seconds": round(seconds, 3),
                              "status": "user_confirmed_label"}
                             for label, seconds in user_labels.most_common()],
        "user_labeled_seconds": round(sum(user_labels.values()), 3),
        "confirmed_results_count": sum(confirmed_results.values()),
        "confirmed_results": [{"label": label, "count": count}
                              for label, count in confirmed_results.most_common()],
        "mac_screen_suggested_seconds": round(sum(screen_workstreams.values()), 3),
        "days": [{"day": item["day_local"], "coverage_status": item["coverage_status"],
                  "behavior": item.get("behavior", {}),
                  "workstreams": item["workstreams"],
                  "mac_active_seconds": item["mac"]["active_seconds"],
                  "mac_unattributed_seconds": item["mac"].get("unattributed_seconds", 0),
                  "mac_foreground_seconds": (item["mac"]["active_seconds"]
                                             + item["mac"].get("unattributed_seconds", 0)),
                  "mac_app_only_seconds": item["mac"].get("app_only_seconds", 0),
                  "mac_observed_seconds": item["mac"]["observed_seconds"],
                  "iphone_focus_seconds": item["iphone"]["focus_union_seconds"],
                  "local_commits": item.get("work_artifacts", {}).get("local_commits", 0),
                  "file_events": item.get("work_artifacts", {}).get("file_events", 0)}
                 for item in days],
        "workstream_labels_are_inferred": True,
    }


def refresh(*, now: datetime | None = None, recent_days: int = 8,
            force_backfill: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZONE).date()
    first = earliest_day()
    if first is None:
        return {"status": "no_activity_data", "days_written": 0}
    os.umask(0o077)
    prepare_directory(ROOT)
    changed_months = set()
    days_written = 0
    cursor = first
    while cursor <= today:
        path = DAYS / f"{cursor.isoformat()}.json"
        if (force_backfill or not path.is_file() or cursor >= today - timedelta(days=recent_days)
                or read_json(path).get("schema_version") != VERSION):
            if private_write_if_changed(path, day_summary(cursor, now=now)):
                days_written += 1
                changed_months.add(cursor.strftime("%Y-%m"))
        cursor += timedelta(days=1)
    for path in DAYS.glob("*.json"):
        month = path.stem[:7]
        if not (MONTHS / f"{month}.json").is_file():
            changed_months.add(month)
    month_written = year_written = 0
    for month in sorted(changed_months):
        day_files = sorted(DAYS.glob(month + "-*.json"))
        values = [read_json(path) for path in day_files]
        if private_write_if_changed(MONTHS / f"{month}.json", aggregate(values, period="month", key=month)):
            month_written += 1
    changed_years = {month[:4] for month in changed_months}
    for path in DAYS.glob("*.json"):
        year = path.stem[:4]
        if not (YEARS / f"{year}.json").is_file():
            changed_years.add(year)
    for year in sorted(changed_years):
        day_files = sorted(DAYS.glob(year + "-*.json"))
        values = [read_json(path) for path in day_files]
        if private_write_if_changed(YEARS / f"{year}.json", aggregate(values, period="year", key=year)):
            year_written += 1
    return {"status": "refreshed", "days_written": days_written,
            "months_written": month_written, "years_written": year_written,
            "earliest_day": first.isoformat(), "latest_day": today.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-backfill", action="store_true")
    args = parser.parse_args()
    print(json.dumps(refresh(force_backfill=args.force_backfill), sort_keys=True))


if __name__ == "__main__":
    main()
