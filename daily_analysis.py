"""Build an evidence-labeled local activity analysis without sending private data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from secure_store import DB_PATH, STATE_DIR


ZONE = ZoneInfo("America/Chicago")
ANALYSIS_DIR = STATE_DIR / "analyses"
SCREENSHOT_DIR = STATE_DIR / "screenshots"
PHONE_QUALITY = STATE_DIR / "phone-quality-latest.json"
MAX_PHONE_SESSION = timedelta(hours=6)
MAC_SAMPLE_GAP = timedelta(seconds=2)
MAX_LEGACY_INTERVAL = timedelta(hours=6)
OVERLAY_ONLY_BUNDLES = frozenset({"com.FluidApp.app"})


def seconds_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def union_seconds(intervals: list[tuple[datetime, datetime]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += seconds_between(start, end)
            start, end = next_start, next_end
    return total + seconds_between(start, end)


def clip_interval(start: datetime, end: datetime, lower: datetime, upper: datetime):
    clipped = max(start, lower), min(end, upper)
    return clipped if clipped[0] < clipped[1] else None


def mac_segments(rows: list[dict], start: datetime, end: datetime) -> tuple[list[dict], list[dict]]:
    samples = []
    for row in sorted(rows, key=lambda item: item["timestamp_utc"]):
        if row["source"] not in {"mac_window_sample", "mac_idle", "mac_locked",
                                 "mac_window_legacy", "mac_idle_legacy"}:
            continue
        at = datetime.fromisoformat(row["timestamp_utc"])
        duration = row.get("duration_seconds") or 0
        maximum = (MAX_LEGACY_INTERVAL.total_seconds()
                   if row["source"].endswith("_legacy") else 30)
        if not 0 < duration <= maximum:
            continue
        interval = clip_interval(at, at + timedelta(seconds=duration), start, end)
        if interval:
            samples.append((interval[0], interval[1], row))

    # Polling jitter must not create overlapping ownership of a second.
    trimmed = []
    for index, (at, until, row) in enumerate(samples):
        if index + 1 < len(samples):
            until = min(until, samples[index + 1][0])
        if at < until:
            trimmed.append((at, until, row))

    segments = []
    for at, until, row in trimmed:
        state = {"mac_window_sample": "active", "mac_idle": "idle", "mac_locked": "locked",
                 "mac_window_legacy": "active", "mac_idle_legacy": "idle"}[row["source"]]
        evidence = ("activitywatch_legacy_interval" if row["source"].endswith("_legacy")
                    else "mac_five_second_sample")
        data = row.get("data") or {}
        if state == "active" and not data.get("title"):
            state = "unattributed"
        app = data.get("app") if state in {"active", "unattributed"} else None
        title = data.get("title") if state == "active" else None
        site_host = data.get("site_host") if state in {"active", "unattributed"} else None
        if (segments and segments[-1]["state"] == state and segments[-1]["evidence"] == evidence
                and segments[-1]["app"] == app
                and segments[-1]["window"] == title
                and segments[-1].get("site_host") == site_host
                and at - datetime.fromisoformat(segments[-1]["end_utc"]) <= MAC_SAMPLE_GAP):
            item = segments[-1]
            item["end_utc"] = until.isoformat()
            item["sample_count"] += 1
            item["sampled_seconds"] += seconds_between(at, until)
        else:
            segments.append({
                "start_utc": at.isoformat(), "end_utc": until.isoformat(),
                "state": state, "app": app, "window": title,
                "site_host": site_host,
                "sample_count": 1, "sampled_seconds": seconds_between(at, until),
                "evidence": evidence,
            })

    gaps = []
    cursor = start
    for at, until, _ in trimmed:
        if seconds_between(cursor, at) >= 60:
            gaps.append({"start_utc": cursor.isoformat(), "end_utc": at.isoformat(),
                         "seconds": seconds_between(cursor, at)})
        cursor = max(cursor, until)
    if seconds_between(cursor, end) >= 60:
        gaps.append({"start_utc": cursor.isoformat(), "end_utc": end.isoformat(),
                     "seconds": seconds_between(cursor, end)})
    return segments, gaps


def phone_analysis(rows: list[dict], start: datetime, end: datetime) -> dict:
    sessions = []
    by_app = defaultdict(list)
    for row in rows:
        if row["source"] != "iphone_app":
            continue
        at = datetime.fromisoformat(row["timestamp_utc"])
        duration = row.get("duration_seconds") or 0
        if not 0 < duration <= MAX_PHONE_SESSION.total_seconds():
            continue
        interval = clip_interval(at, at + timedelta(seconds=duration), start, end)
        if not interval:
            continue
        app = (row.get("data") or {}).get("app") or "unknown_app"
        by_app[app].append(interval)
        sessions.append({"start_utc": interval[0].isoformat(), "end_utc": interval[1].isoformat(),
                         "seconds": seconds_between(*interval), "app": app,
                         "evidence": "apple_screen_time_synced_focus_interval"})
    sessions.sort(key=lambda item: item["start_utc"])
    all_intervals = [interval for intervals in by_app.values() for interval in intervals]
    raw_seconds = sum(seconds_between(*interval) for interval in all_intervals)
    return {
        "sessions": sessions,
        "app_totals": [
            {"app": app, "focus_seconds": round(union_seconds(intervals), 3),
             "interval_count": len(intervals)}
            for app, intervals in sorted(by_app.items(), key=lambda pair: -union_seconds(pair[1]))
        ],
        "device_union_seconds": round(union_seconds(all_intervals), 3),
        "overlap_seconds": round(max(0, raw_seconds - union_seconds(all_intervals)), 3),
        "interval_count": len(sessions),
        "evidence": "synced_iphone_intervals_may_lag_or_be_incomplete",
    }


def browser_analysis(rows: list[dict], mac: list[dict], start: datetime, end: datetime) -> dict:
    events = sorted((row for row in rows if row["source"] == "browser_tab"),
                    key=lambda row: row["timestamp_utc"])
    totals = Counter()
    for index, row in enumerate(events):
        data = row.get("data") or {}
        if data.get("event_type") == "browser_blur" or not data.get("site_host"):
            continue
        at = max(start, datetime.fromisoformat(row["timestamp_utc"]))
        next_at = (datetime.fromisoformat(events[index + 1]["timestamp_utc"])
                   if index + 1 < len(events) else end)
        until = min(end, next_at)
        if at >= until:
            continue
        for segment in mac:
            app = (segment.get("app") or "").casefold()
            if segment["state"] not in {"active", "unattributed"} or "chrome" not in app:
                continue
            segment_start = datetime.fromisoformat(segment["start_utc"])
            segment_end = datetime.fromisoformat(segment["end_utc"])
            overlap = seconds_between(max(at, segment_start), min(until, segment_end))
            duration = seconds_between(segment_start, segment_end)
            if overlap and duration:
                totals[data["site_host"]] += overlap * segment["sampled_seconds"] / duration
    return {"domains": [{"site_host": domain, "seconds": round(seconds, 3)}
                        for domain, seconds in totals.most_common()],
            "tracked_seconds": round(sum(totals.values()), 3),
            "event_count": len(events),
            "evidence": "active_tab_events_clipped_to_observed_chrome_foreground"}


def load_rows(start: datetime, end: datetime) -> list[dict]:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        lower = start - MAX_PHONE_SESSION
        return [
            {"timestamp_utc": at, "source": source, "duration_seconds": duration,
             "data": json.loads(data)}
            for at, source, duration, data in connection.execute(
                "SELECT timestamp_utc,source,duration_seconds,data_json FROM events "
                "WHERE timestamp_utc >= ? AND timestamp_utc < ? ORDER BY timestamp_utc",
                (lower.isoformat(), end.isoformat()),
            )
        ]
    finally:
        connection.close()


def visual_counts(start: datetime, end: datetime) -> dict:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        screenshots = connection.execute(
            "SELECT path,ocr_status FROM screenshots WHERE timestamp_utc >= ? AND timestamp_utc < ?",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        vision = dict(connection.execute(
            "SELECT status,count(*) FROM vision_descriptions "
            "WHERE timestamp_utc >= ? AND timestamp_utc < ? GROUP BY status",
            (start.isoformat(), end.isoformat()),
        ).fetchall())
    finally:
        connection.close()
    indexed = {path for path, _ in screenshots}
    pending = sum(1 for path, status in screenshots
                  if status != "complete" and str(path).startswith(str(SCREENSHOT_DIR) + "/"))
    day = start.date()
    while day <= end.date():
        folder = SCREENSHOT_DIR / day.strftime("%Y%m%d")
        if folder.is_dir():
            for path in folder.glob("*.webp"):
                try:
                    at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except FileNotFoundError:
                    continue
                if start <= at < end and str(path) not in indexed:
                    pending += 1
        day += timedelta(days=1)
    return {
        "indexed_screenshots": len(screenshots),
        "ocr_status_counts": dict(Counter(status for _, status in screenshots)),
        "new_capture_ocr_pending": pending,
        "vision_status_counts": vision,
        "vision_descriptions_are_inferences": True,
    }


def phone_source_quality() -> dict:
    if not PHONE_QUALITY.is_file():
        return {"quality_status": "not_checked"}
    source = json.loads(PHONE_QUALITY.read_text(encoding="utf-8"))
    return {key: source.get(key) for key in (
        "checked_at_utc", "quality_status", "source_age_minutes",
        "source_behind_ledger_minutes", "interpretation")}


def analyze(day: date, *, now: datetime | None = None) -> dict:
    start = datetime.combine(day, time.min, tzinfo=ZONE).astimezone(timezone.utc)
    day_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=ZONE).astimezone(timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    now = now.astimezone(timezone.utc)
    end = min(day_end, now)
    if end <= start:
        raise ValueError("Cannot analyze a future day")
    rows = load_rows(start, end)
    segments, gaps = mac_segments(rows, start, end)
    mac_totals = Counter()
    app_totals = Counter()
    app_only_seconds = 0.0
    for segment in segments:
        mac_totals[segment["state"]] += segment["sampled_seconds"]
        app = segment.get("app")
        if segment["state"] in {"active", "unattributed"} and app and app not in OVERLAY_ONLY_BUNDLES:
            app_totals[app] += segment["sampled_seconds"]
            if segment["state"] == "unattributed":
                app_only_seconds += segment["sampled_seconds"]
    observed = sum(mac_totals.values())
    return {
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "day_local": day.isoformat(), "timezone": str(ZONE),
        "start_utc": start.isoformat(), "analyzed_through_utc": end.isoformat(),
        "complete_day": end == day_end,
        "mac": {
            "state_sampled_seconds": {key: round(value, 3) for key, value in mac_totals.items()},
            "app_sampled_seconds": [
                {"app": app, "seconds": round(seconds, 3)}
                for app, seconds in app_totals.most_common()
            ],
            "app_only_seconds": round(app_only_seconds, 3),
            "observed_sampled_seconds": round(observed, 3),
            "unobserved_seconds": round(max(0, seconds_between(start, end) - observed), 3),
            "gaps_over_one_minute": gaps,
            "segments": segments,
            "evidence": "five_second_samples_estimate_foreground_state_not_clicks_or_completions",
        },
        "iphone": phone_analysis(rows, start, end),
        "browser": browser_analysis(rows, segments, start, end),
        "visual": visual_counts(start, end),
        "source_quality": {"iphone_sync": phone_source_quality()},
        "interpretation_rules": [
            "Do not add Mac and iPhone durations; concurrent use is possible.",
            "Unobserved Mac time is unknown, not idle or zero activity.",
            "A recorded frontmost app without a window title supports app-level time, not task-level attribution.",
            "Overlay-only apps are excluded from app totals because they can obscure the underlying work.",
            "Window titles and OCR are visible text, not proof of a completed action.",
            "Vision descriptions are untrusted model inferences.",
        ],
    }


def private_write(path: Path, content: bytes) -> None:
    ANALYSIS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(ANALYSIS_DIR, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=".analysis-", dir=ANALYSIS_DIR)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-ago", type=int, action="append")
    parser.add_argument("--write", action="store_true", help="Save a private analysis and SHA-256 manifest")
    args = parser.parse_args()
    days_ago = args.days_ago or [1]
    if any(value < 0 for value in days_ago):
        parser.error("--days-ago must be zero or greater")
    for value in dict.fromkeys(days_ago):
        report = analyze(datetime.now(ZONE).date() - timedelta(days=value))
        payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        result = {
            "day_local": report["day_local"], "complete_day": report["complete_day"],
            "mac_segments": len(report["mac"]["segments"]),
            "mac_unobserved_seconds": report["mac"]["unobserved_seconds"],
            "iphone_intervals": report["iphone"]["interval_count"],
            "iphone_overlap_seconds": report["iphone"]["overlap_seconds"],
            "visual": report["visual"], "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": "local_analysis_written" if args.write else "preview_only",
        }
        if args.write:
            name = f"analysis-{report['day_local']}"
            private_write(ANALYSIS_DIR / f"{name}.json", payload)
            private_write(ANALYSIS_DIR / f"{name}.manifest.json",
                          (json.dumps(result, indent=2) + "\n").encode())
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
