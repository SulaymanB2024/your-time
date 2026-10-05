"""Write a private, model-ready daily timeline from the local SQLite ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from secure_store import DB_PATH, STATE_DIR


ZONE = ZoneInfo("America/Chicago")
SCREENSHOT_DIRS = [STATE_DIR / "pensieve/screenshots", STATE_DIR / "screenshots"]
EXPORT_DIR = STATE_DIR / "exports"
PHONE_QUALITY_PATH = STATE_DIR / "phone-quality-latest.json"
MIN_FREE_BYTES = 10 * 1024**3


def ledger_events(start_utc: datetime, end_utc: datetime):
    if not DB_PATH.exists():
        return
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        for event_id, timestamp, source, duration, data_json in connection.execute(
            "SELECT id,timestamp_utc,source,duration_seconds,data_json FROM events "
            "WHERE timestamp_utc >= ? AND timestamp_utc < ? ORDER BY timestamp_utc",
            (start_utc.isoformat(), end_utc.isoformat()),
        ):
            yield {
                "timestamp_utc": timestamp,
                "source": source,
                "duration_seconds": duration,
                "data": json.loads(data_json),
                "evidence": {"system": "private_sqlite", "event_id": event_id},
            }

        indexed_paths = set()
        for path, timestamp, app, window, url, text, status in connection.execute(
            "SELECT path,timestamp_utc,active_app,active_window,url,ocr_text,ocr_status "
            "FROM screenshots WHERE timestamp_utc >= ? AND timestamp_utc < ? "
            "ORDER BY timestamp_utc",
            (start_utc.isoformat(), end_utc.isoformat()),
        ):
            indexed_paths.add(Path(path))
            yield {
                "timestamp_utc": timestamp,
                "source": "mac_screen" if status == "complete" else "mac_screen_pending",
                "data": {
                    "active_app": app,
                    "active_window": window,
                    "url": url,
                    "visible_text": text,
                },
                "evidence": {
                    "system": "offline_apple_vision",
                    "screenshot_path": str(path),
                    "ocr_status": status,
                    "timestamp_basis": (
                        "capture_time" if str(path).startswith(str(STATE_DIR / "screenshots") + "/")
                        else "filesystem_mtime"
                    ),
                },
            }

        for path, timestamp, description, model_sha, prompt_version, screenshot_sha, elapsed in connection.execute(
            "SELECT path,timestamp_utc,description,model_sha256,prompt_version,"
            "screenshot_sha256,elapsed_seconds FROM vision_descriptions "
            "WHERE status='complete' AND timestamp_utc >= ? AND timestamp_utc < ? "
            "ORDER BY timestamp_utc",
            (start_utc.isoformat(), end_utc.isoformat()),
        ):
            yield {
                "timestamp_utc": timestamp,
                "source": "mac_vision_inference",
                "data": {"description": description, "untrusted_inference": True},
                "evidence": {
                    "system": "local_vision_model",
                    "screenshot_path": path,
                    "screenshot_sha256": screenshot_sha,
                    "model_sha256": model_sha,
                    "prompt_version": prompt_version,
                    "elapsed_seconds": elapsed,
                },
            }

        # Keep raw screenshots in the timeline while the offline OCR queue catches up.
        for screenshot_dir in SCREENSHOT_DIRS:
            if not screenshot_dir.exists():
                continue
            for path in screenshot_dir.rglob("*.webp"):
                if path in indexed_paths or path.name.startswith("temp_"):
                    continue
                try:
                    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except FileNotFoundError:
                    continue
                if start_utc <= timestamp < end_utc:
                    yield {
                        "timestamp_utc": timestamp.isoformat(),
                        "source": "mac_screen_pending",
                        "data": {"visible_text": None},
                        "evidence": {
                            "system": "private_screenshot_capture",
                            "screenshot_path": str(path),
                            "index_status": "pending",
                            "timestamp_basis": "filesystem_mtime",
                        },
                    }
    finally:
        connection.close()


def private_write(path: Path, content: bytes):
    EXPORT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(EXPORT_DIR, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(content)
    os.replace(temporary, path)


def export_day(day, *, write: bool = False):
    if shutil.disk_usage(STATE_DIR).free < MIN_FREE_BYTES:
        raise RuntimeError("Free space is below the 10 GiB export safety floor")
    start = datetime.combine(day, time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=ZONE).astimezone(timezone.utc)
    events = list(ledger_events(start, end))
    events.sort(key=lambda event: event["timestamp_utc"])
    payload = b"".join(
        (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for event in events
    )
    name = f"timeline-{day.isoformat()}"
    path = EXPORT_DIR / f"{name}.jsonl"
    manifest = {
        "day_local": day.isoformat(),
        "timezone": str(ZONE),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(events),
        "source_counts": dict(Counter(event["source"] for event in events)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": "local_export_only" if write else "preview_only",
    }
    if PHONE_QUALITY_PATH.is_file():
        quality = json.loads(PHONE_QUALITY_PATH.read_text(encoding="utf-8"))
        manifest["iphone_sync_quality"] = {
            "checked_at_utc": quality.get("checked_at_utc"),
            "quality_status": quality.get("quality_status"),
            "source_behind_ledger_minutes": quality.get("source_behind_ledger_minutes"),
        }
    if write:
        private_write(path, payload)
        private_write(EXPORT_DIR / f"{name}.manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-ago", type=int, action="append")
    parser.add_argument("--write", action="store_true", help="Create a private JSONL copy for a deliberate model handoff")
    args = parser.parse_args()
    today = datetime.now(ZONE).date()
    for days_ago in args.days_ago or [1]:
        if days_ago < 0:
            parser.error("--days-ago must be zero or greater")
        print(json.dumps(export_day(today - timedelta(days=days_ago), write=args.write), sort_keys=True))


if __name__ == "__main__":
    main()
