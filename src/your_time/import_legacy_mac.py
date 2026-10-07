"""Import only pre-collector Mac window/AFK intervals from preserved ActivityWatch."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from daily_analysis import ANALYSIS_DIR, private_write
from secure_store import DB_PATH, insert_events

SOURCE = Path.home() / "Library/Application Support/activitywatch/aw-server/peewee-sqlite.v2.db"
RECEIPT = ANALYSIS_DIR / "legacy-activitywatch-import.json"
MAX_DURATION = 6 * 3600


def merge(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    result = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersections(start: datetime, end: datetime,
                  intervals: list[tuple[datetime, datetime]]):
    for left, right in intervals:
        at, until = max(start, left), min(end, right)
        if at < until:
            yield at, until


def subtract(interval: tuple[datetime, datetime],
             exclusions: list[tuple[datetime, datetime]]):
    pieces = [interval]
    for left, right in exclusions:
        next_pieces = []
        for start, end in pieces:
            if right <= start or left >= end:
                next_pieces.append((start, end))
            else:
                if start < left:
                    next_pieces.append((start, left))
                if right < end:
                    next_pieces.append((right, end))
        pieces = next_pieces
    return pieces


def first_new_collector_event() -> datetime:
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        value = db.execute("SELECT MIN(timestamp_utc) FROM events "
                           "WHERE source IN ('mac_window_sample','mac_idle','mac_locked')").fetchone()[0]
    finally:
        db.close()
    return datetime.fromisoformat(value) if value else datetime.now(timezone.utc)


def read_activitywatch(source: Path) -> list[tuple]:
    db = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        return db.execute(
            "SELECT b.type,e.timestamp,e.duration,e.datastr FROM eventmodel e "
            "JOIN bucketmodel b ON e.bucket_id=b.key "
            "WHERE b.type IN ('currentwindow','afkstatus') ORDER BY e.timestamp"
        ).fetchall()
    finally:
        db.close()


def build_rows(source_rows: list[tuple], cutoff: datetime) -> list[dict]:
    windows = []
    active = []
    idle = []
    for kind, timestamp, duration, raw in source_rows:
        seconds = float(duration or 0)
        if not 0 < seconds <= MAX_DURATION:
            continue
        start = datetime.fromisoformat(timestamp).astimezone(timezone.utc)
        end = min(start + timedelta(seconds=seconds), cutoff)
        if start >= end:
            continue
        data = json.loads(raw or "{}")
        if kind == "currentwindow":
            windows.append((start, end, data))
        elif kind == "afkstatus" and data.get("status") == "not-afk":
            active.append((start, end))
        elif kind == "afkstatus" and data.get("status") == "afk":
            idle.append((start, end))
    active = merge(active)
    idle = merge(idle)
    rows = []
    windows.sort(key=lambda item: item[0])
    for index, (start, end, data) in enumerate(windows):
        if index + 1 < len(windows):
            end = min(end, windows[index + 1][0])
        for at, until in intersections(start, end, active):
            rows.append({"timestamp_utc": at.isoformat(),
                         "duration_seconds": round((until - at).total_seconds(), 3),
                         "source": "mac_window_legacy",
                         "data": {"app": data.get("app"), "title": data.get("title"),
                                  "origin": "activitywatch_currentwindow"}})
    for interval in idle:
        for at, until in subtract(interval, active):
            rows.append({"timestamp_utc": at.isoformat(),
                         "duration_seconds": round((until - at).total_seconds(), 3),
                         "source": "mac_idle_legacy",
                         "data": {"origin": "activitywatch_afkstatus"}})
    return sorted(rows, key=lambda item: item["timestamp_utc"])


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError("Preserved ActivityWatch source is unavailable")
    cutoff = first_new_collector_event()
    rows = build_rows(read_activitywatch(SOURCE), cutoff)
    added = insert_events(rows)
    receipt = {"status": "legacy_mac_imported", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "cutoff_utc": cutoff.isoformat(), "eligible_intervals": len(rows),
               "inserted_intervals": added,
               "window_intervals": sum(row["source"] == "mac_window_legacy" for row in rows),
               "idle_intervals": sum(row["source"] == "mac_idle_legacy" for row in rows)}
    private_write(RECEIPT, (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({key: receipt[key] for key in ("status", "eligible_intervals",
                                                   "inserted_intervals", "window_intervals",
                                                   "idle_intervals")}, sort_keys=True))


if __name__ == "__main__":
    main()
