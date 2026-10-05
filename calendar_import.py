"""Import only explicitly selected EventKit calendar/reminder metadata locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_synthesis import safe_input_text
from secure_store import STATE_DIR, connect, prepare_private_dir

SCOPE_PATH = STATE_DIR / "calendar-scope.json"
EXPORT_PATH = STATE_DIR / "calendar-eventkit-export.json"
STATUS_PATH = STATE_DIR / "calendar-latest-receipt.json"
HEALTH_RE = re.compile(r"\b(doctor|dentist|therapy|hospital|medical|medication|pharmacy|clinic|healthcare)\b", re.I)


def read_private_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError("Private input file unavailable")
    if path.stat().st_mode & 0o077 or path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Private input permissions or size invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid private input")
    return value


def allowed(scope: dict, kind: str) -> dict[str, bool]:
    rows = scope.get(kind)
    if not isinstance(rows, list):
        raise ValueError("Calendar scope must list selected sources")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Invalid calendar scope item")
        result[row["id"]] = row.get("includeTitle") is True
    return result


def parse_time(value: str) -> datetime:
    at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError("Calendar time lacks timezone")
    return at.astimezone(timezone.utc)


def rows_from_export(scope: dict, payload: dict, now: datetime) -> tuple[list[tuple], dict]:
    event_scope = allowed(scope, "events")
    reminder_scope = allowed(scope, "reminders")
    if not event_scope and not reminder_scope:
        raise ValueError("No calendars or reminder lists selected")
    result = []
    skipped_health = skipped_unscoped = 0
    for kind, values in (("event", payload.get("events", [])),
                         ("reminder", payload.get("reminders", []))):
        if not isinstance(values, list):
            raise ValueError("Invalid EventKit export")
        selected = event_scope if kind == "event" else reminder_scope
        for item in values:
            if not isinstance(item, dict) or item.get("calendarId") not in selected:
                skipped_unscoped += 1
                continue
            source_id = item.get("id")
            if not isinstance(source_id, str) or not source_id:
                continue
            raw_title = item.get("title") or ""
            if not isinstance(raw_title, str) or HEALTH_RE.search(raw_title):
                skipped_health += 1
                continue
            title = safe_input_text(raw_title, 160) if selected[item["calendarId"]] else ""
            at = parse_time(item["start"] if kind == "event" else item["at"])
            if at < now - timedelta(days=366) or at > now + timedelta(days=31):
                continue
            end = parse_time(item["end"]).isoformat() if kind == "event" else None
            identity = hashlib.sha256(f"{kind}:{item['calendarId']}:{source_id}".encode()).hexdigest()
            result.append((identity, kind, item["calendarId"], at.isoformat(), end,
                           title or None, int(bool(item.get("completed"))) if kind == "reminder" else None,
                           now.isoformat(), int(bool(item.get("allDay"))) if kind == "event" else None))
    return result, {"skipped_health_titles": skipped_health,
                    "skipped_unscoped": skipped_unscoped}


def write_receipt(value: dict) -> None:
    prepare_private_dir()
    fd, temporary = tempfile.mkstemp(prefix=".calendar-receipt-", dir=STATE_DIR)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump({"checked_at_utc": datetime.now(timezone.utc).isoformat(), **value},
                      output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, STATUS_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run() -> dict:
    now = datetime.now(timezone.utc)
    scope = read_private_json(SCOPE_PATH)
    payload = read_private_json(EXPORT_PATH)
    rows, skipped = rows_from_export(scope, payload, now)
    inserted = 0
    with connect() as database:
        for row in rows:
            cursor = database.execute(
                "INSERT OR IGNORE INTO calendar_items "
                "(id,source,calendar_id,start_utc,end_utc,title,completed,observed_at_utc,all_day) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
            inserted += cursor.rowcount
    result = {"status": "imported", "selected_items": len(rows),
              "new_items": inserted, **skipped}
    write_receipt(result)
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.umask(0o077)
    print(json.dumps(run(), sort_keys=True))


if __name__ == "__main__":
    main()
