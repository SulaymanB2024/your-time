"""Compare Apple's current synced iPhone source with retained local events."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from private_io import open_private_file, prepare_directory, write_json
from secure_store import DB_PATH, STATE_DIR

PHONE_RECEIPT = STATE_DIR / "phone-latest-receipt.json"
QUALITY_RECEIPT = STATE_DIR / "phone-quality-latest.json"
HISTORY_DIR = STATE_DIR / "phone-quality-history"
REGRESSION_TOLERANCE = timedelta(minutes=5)
STALE_AFTER = timedelta(hours=36)


def latest_retained_start() -> datetime | None:
    if not DB_PATH.is_file():
        return None
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        value = connection.execute(
            "SELECT MAX(timestamp_utc) FROM events WHERE source='iphone_app'"
        ).fetchone()[0]
    finally:
        connection.close()
    return datetime.fromisoformat(value) if value else None


def evaluate(receipt: dict | None, ledger_latest: datetime | None,
             now: datetime, *, reader_exit: int = 0) -> dict:
    source_start = None
    checked = None
    if receipt and receipt.get("checked_at_utc") and receipt.get("latest_closed_session_age_minutes") is not None:
        checked = datetime.fromisoformat(receipt["checked_at_utc"])
        # The packaged importer reports the latest session START as a rounded
        # age, despite its older "closed_session" field name.
        source_start = checked - timedelta(minutes=receipt["latest_closed_session_age_minutes"])
    behind = ((ledger_latest - source_start).total_seconds() / 60
              if source_start and ledger_latest else None)
    source_age = (now - source_start).total_seconds() / 60 if source_start else None
    if reader_exit:
        status = "reader_failed"
    elif source_start is None:
        status = "source_timestamp_unavailable"
    elif source_age is not None and source_age > STALE_AFTER.total_seconds() / 60:
        status = "source_stale"
    elif behind is not None and behind > REGRESSION_TOLERANCE.total_seconds() / 60:
        status = "source_behind_retained_ledger"
    elif ledger_latest is None:
        status = "no_retained_iphone_events"
    else:
        status = "source_current_relative_to_ledger"
    return {
        "checked_at_utc": now.isoformat(),
        "reader_exit_code": reader_exit,
        "quality_status": status,
        "source_receipt_checked_at_utc": checked.isoformat() if checked else None,
        "latest_source_session_start_estimated_utc": source_start.isoformat() if source_start else None,
        "latest_retained_iphone_start_utc": ledger_latest.isoformat() if ledger_latest else None,
        "source_age_minutes": round(source_age, 1) if source_age is not None else None,
        "source_behind_ledger_minutes": round(max(0, behind), 1) if behind is not None else None,
        "source_receipt_status": receipt.get("status") if receipt else None,
        "source_inserted_sessions": receipt.get("inserted_sessions") if receipt else None,
        "interpretation": "Apple's readable sync files can lag or rewind; this compares timestamps, not iPhone Screen Time totals or proof of missing use.",
    }


def private_write(path: Path, value: dict) -> None:
    write_json(path, value)


def append_history(value: dict) -> None:
    prepare_directory(HISTORY_DIR)
    path = HISTORY_DIR / (datetime.fromisoformat(value["checked_at_utc"]).date().isoformat() + ".jsonl")
    fd = open_private_file(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-exit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    receipt = json.loads(PHONE_RECEIPT.read_text()) if PHONE_RECEIPT.is_file() else None
    result = evaluate(receipt, latest_retained_start(), datetime.now(timezone.utc),
                      reader_exit=args.reader_exit)
    if not args.dry_run:
        private_write(QUALITY_RECEIPT, result)
        append_history(result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
