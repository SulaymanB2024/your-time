"""Safely import synced iPhone foreground intervals into private SQLite.

Run this with the Python environment from ActivityWatch/aw-import-screentime.
The upstream watcher can turn an iCloud sync gap into a multi-day interval;
this importer excludes implausible intervals and records the exclusion. It
does not start a local HTTP server or perform app-title network lookups.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from private_io import write_json
from secure_store import insert_events

STATE_DIR = Path.home() / "Library/Application Support/personal-activity-ledger"
CONFIG_PATH = STATE_DIR / "phone.json"
RECEIPT_PATH = STATE_DIR / "phone-latest-receipt.json"
MAX_SESSION = timedelta(hours=6)
MAX_SOURCE_AGE = timedelta(hours=36)
LOOKBACK = timedelta(days=3)


def read_intervals(device_id: str):
    from aw_import_screentime.__main__ import (
        iter_app_in_focus_events,
        iter_device_files,
        stitch_intervals_with_state,
    )
    files = list(iter_device_files(device_id))
    raw = [event for path in files for event in iter_app_in_focus_events(path)]
    raw.sort(key=lambda event: event.cf_absolute_time)
    sessions, _ = stitch_intervals_with_state(raw, tzinfo=timezone.utc)
    return files, raw, sessions


def choose_device(now: datetime, pinned: str | None):
    from aw_import_screentime.__main__ import SYNC_DB_PATH, get_device_ids
    candidates = get_device_ids(SYNC_DB_PATH, platform=2)
    if pinned:
        if pinned not in candidates:
            raise RuntimeError("Pinned phone device is absent from Apple sync metadata")
        files, raw, sessions = read_intervals(pinned)
        return pinned, files, raw, sessions

    recent = []
    for candidate in candidates:
        files, raw, sessions = read_intervals(candidate)
        if sessions and now - sessions[-1].timestamp <= MAX_SOURCE_AGE:
            recent.append((candidate, files, raw, sessions))
    if len(recent) != 1:
        raise RuntimeError(
            f"Expected one recently synced iOS device; found {len(recent)}. "
            "Select the intended iPhone before importing."
        )
    return recent[0]


def select_sessions(sessions, now: datetime, *, lookback: timedelta | None = LOOKBACK):
    accepted = []
    suspect = 0
    old = 0
    cutoff = now - lookback if lookback is not None else None
    for event in sessions:
        duration = event.duration or timedelta(0)
        end = event.timestamp + duration
        if cutoff is not None and end < cutoff:
            old += 1
            continue
        if (
            duration <= timedelta(0)
            or duration > MAX_SESSION
            or event.timestamp > now + timedelta(minutes=5)
            or end > now + timedelta(minutes=5)
        ):
            suspect += 1
            continue
        accepted.append(event)
    return accepted, suspect, old


def write_private_json(path: Path, value: dict):
    write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backfill-all", action="store_true",
                        help="Import all plausible synced iPhone sessions, not only recent days")
    args = parser.parse_args()
    logging.getLogger("aw_import_screentime").setLevel(logging.ERROR)
    now = datetime.now(timezone.utc)
    pinned = None
    if CONFIG_PATH.exists():
        pinned = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["device_id"]
    device_id, files, raw, sessions = choose_device(now, pinned)
    accepted, suspect, old = select_sessions(sessions, now,
                                             lookback=None if args.backfill_all else LOOKBACK)
    latest = sessions[-1].timestamp if sessions else None
    age = now - latest if latest else None
    if age is None or age > MAX_SOURCE_AGE:
        raise RuntimeError("Phone activity has not synced in the last 36 hours")
    if not accepted:
        raise RuntimeError("No plausible phone sessions were found in the last 3 days")

    result = {
        "checked_at_utc": now.isoformat(),
        "source_files": len(files),
        "raw_focus_events": len(raw),
        "plausible_sessions": len(accepted),
        "suspect_sessions_excluded": suspect,
        "older_sessions_skipped": old,
        "backfill_all": args.backfill_all,
        "earliest_accepted_start_utc": min((event.timestamp for event in accepted), default=None).isoformat() if accepted else None,
        "latest_closed_session_age_minutes": round(age.total_seconds() / 60, 1),
        "inserted_sessions": 0,
        "status": "dry_run" if args.dry_run else "imported",
    }
    if not args.dry_run:
        rows = [
            {
                "timestamp_utc": event.timestamp.astimezone(timezone.utc).isoformat(),
                "source": "iphone_app",
                "duration_seconds": event.duration.total_seconds(),
                "data": {"app": event.data["app"]},
            }
            for event in accepted
        ]
        result["inserted_sessions"] = insert_events(rows)
        if pinned is None:
            write_private_json(CONFIG_PATH, {"device_id": device_id})
        write_private_json(RECEIPT_PATH, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
