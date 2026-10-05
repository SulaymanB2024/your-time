"""Index private screenshots with Apple's on-device OCR, without an HTTP server."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ocrmac import ocrmac
from PIL import Image

from secure_store import STATE_DIR, connect, prepare_private_dir
from vision_batch import in_overnight_window


SCREENSHOT_DIRS = [STATE_DIR / "pensieve/screenshots", STATE_DIR / "screenshots"]
OCR_RECEIPT = STATE_DIR / "ocr-latest-receipt.json"
OCR_LOCK = STATE_DIR / "ocr-index.lock"


def private_screenshots():
    result = []
    for screenshot_dir in SCREENSHOT_DIRS:
        if not screenshot_dir.exists():
            continue
        root = screenshot_dir.resolve()
        for path in screenshot_dir.rglob("*.webp"):
            if path.is_symlink() or path.name.startswith("temp_"):
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_uid != os.getuid() or not path.resolve().is_relative_to(root):
                continue
            result.append((stat.st_mtime, path))
    result.sort()
    return result


def index_one(path: Path, timestamp: datetime):
    with Image.open(path) as image:
        description = image.getexif().get(270)
        image_size = image.size
    try:
        metadata = json.loads(description) if isinstance(description, str) and len(description) <= 8192 else {}
    except json.JSONDecodeError:
        metadata = {}
    recognized = ocrmac.OCR(str(path), language_preference=["en-US"]).recognize(px=True)
    text = "\n".join(str(item[0]) for item in recognized)
    boxes = [[round(float(value), 1) for value in item[2]]
             + [round(float(item[1]), 3)] for item in recognized
             if len(item) >= 3 and len(item[2]) == 4]
    return {
        "path": str(path),
        "timestamp_utc": timestamp.isoformat(),
        "active_app": metadata.get("active_app"),
        "active_window": metadata.get("active_window"),
        "url": metadata.get("url"),
        "ocr_text": text,
        "ocr_geometry": {"width": image_size[0], "height": image_size[1], "boxes": boxes},
        "ocr_status": "complete",
    }


def nearby_mac_window(connection, timestamp: datetime):
    row = connection.execute(
        "SELECT timestamp_utc,data_json FROM events WHERE source='mac_window_sample' "
        "AND timestamp_utc <= ? ORDER BY timestamp_utc DESC LIMIT 1",
        (timestamp.isoformat(),),
    ).fetchone()
    if row is None or timestamp - datetime.fromisoformat(row[0]) > timedelta(seconds=15):
        return None, None
    data = json.loads(row[1])
    return data.get("app"), data.get("title")


def save_geometry(connection, path: str, geometry: dict) -> None:
    connection.execute(
        "INSERT INTO ocr_geometry VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET image_width=excluded.image_width, "
        "image_height=excluded.image_height, boxes_json=excluded.boxes_json, "
        "updated_at_utc=excluded.updated_at_utc",
        (path, geometry["width"], geometry["height"],
         json.dumps(geometry["boxes"], separators=(",", ":")),
         datetime.now(timezone.utc).isoformat()),
    )


def backfill_geometry(limit: int, available: set[str]) -> dict:
    if limit < 0:
        raise ValueError("geometry backfill limit must not be negative")
    with connect() as connection:
        candidates = connection.execute(
            "SELECT s.path,s.timestamp_utc FROM screenshots s "
            "LEFT JOIN ocr_geometry g ON g.path=s.path "
            "WHERE s.ocr_status='complete' AND g.path IS NULL "
            "ORDER BY s.timestamp_utc DESC LIMIT ?", (max(1, limit * 3),)
        ).fetchall()
    added = failed = 0
    for path_text, timestamp in candidates:
        if added + failed >= limit:
            break
        if path_text not in available:
            failed += 1
            continue
        try:
            geometry = index_one(Path(path_text), datetime.fromisoformat(timestamp))["ocr_geometry"]
            with connect() as connection:
                save_geometry(connection, path_text, geometry)
            added += 1
        except (OSError, ValueError, KeyError):
            failed += 1
    with connect() as connection:
        pending = connection.execute(
            "SELECT COUNT(*) FROM screenshots s LEFT JOIN ocr_geometry g ON g.path=s.path "
            "WHERE s.ocr_status='complete' AND g.path IS NULL"
        ).fetchone()[0]
    return {"geometry_backfilled": added, "geometry_failed": failed,
            "geometry_pending": pending}


def geometry_pending_count() -> int:
    with connect() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM screenshots s LEFT JOIN ocr_geometry g ON g.path=s.path "
            "WHERE s.ocr_status='complete' AND g.path IS NULL"
        ).fetchone()[0]


def run(limit: int, geometry_backfill_limit: int = 0) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    discovered = private_screenshots()
    with connect() as connection:
        complete = {row[0] for row in connection.execute(
            "SELECT path FROM screenshots WHERE ocr_status='complete'"
        )}
    candidates = [(mtime, path) for mtime, path in discovered if str(path) not in complete]
    indexed = 0
    failures = 0
    for mtime, path in candidates[:limit]:
        timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc)
        try:
            item = index_one(path, timestamp)
        except Exception:
            failures += 1
            continue
        with connect() as connection:
            captured = connection.execute(
                "SELECT timestamp_utc,active_app,active_window,url FROM screenshots WHERE path=?",
                (item["path"],)).fetchone()
            if captured:
                # A missing title at capture is an observation, not permission
                # to borrow another app's older title when OCR runs later.
                item["timestamp_utc"], item["active_app"], item["active_window"], item["url"] = captured
            elif not item["active_app"]:
                item["active_app"], item["active_window"] = nearby_mac_window(connection, timestamp)
            connection.execute(
                "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "active_app=COALESCE(screenshots.active_app,excluded.active_app), "
                "active_window=COALESCE(screenshots.active_window,excluded.active_window), "
                "url=COALESCE(screenshots.url,excluded.url), "
                "ocr_text=excluded.ocr_text,ocr_status=excluded.ocr_status, "
                "indexed_at_utc=excluded.indexed_at_utc",
                (
                    item["path"],
                    item["timestamp_utc"],
                    item["active_app"],
                    item["active_window"],
                    item["url"],
                    item["ocr_text"],
                    item["ocr_status"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            geometry = item.get("ocr_geometry")
            if geometry:
                save_geometry(connection, item["path"], geometry)
        indexed += 1
    result = {
        "discovered": len(discovered),
        "already_indexed": len(complete),
        "newly_indexed": indexed,
        "failed": failures,
        "pending_after_run": max(0, len(candidates) - indexed),
    }
    if geometry_backfill_limit:
        if in_overnight_window(datetime.now(timezone.utc)):
            result["geometry_pending"] = geometry_pending_count()
            result["geometry_deferred_reason"] = "overnight_vision_window"
        else:
            result.update(backfill_geometry(geometry_backfill_limit,
                                            {str(path) for _, path in discovered}))
    return result


def write_receipt(result: dict) -> None:
    prepare_private_dir()
    value = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "status": "partial" if result["pending_after_run"] or result["failed"] else "complete",
             **result}
    fd, temporary = tempfile.mkstemp(prefix=".ocr-receipt-", dir=STATE_DIR)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, OCR_RECEIPT)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--geometry-backfill-limit", type=int, default=0)
    args = parser.parse_args()
    prepare_private_dir()
    fd = os.open(OCR_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "another_ocr_run_active"}))
            return
        started = time.monotonic()
        result = run(args.limit, args.geometry_backfill_limit)
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_receipt(result)
        print(json.dumps(result, sort_keys=True))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
