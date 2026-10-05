import os
import sqlite3
from datetime import datetime, timedelta, timezone

import daily_export


def test_export_keeps_raw_screenshots_pending_ocr(tmp_path, monkeypatch):
    root = tmp_path
    images = root / "screenshots" / "20260928"
    images.mkdir(parents=True)
    indexed = images / "indexed.webp"
    pending = images / "pending.webp"
    pending_registered = images / "pending-registered.webp"
    indexed.write_bytes(b"image")
    pending.write_bytes(b"image")
    pending_registered.write_bytes(b"image")
    instant = datetime(2026, 9, 28, 20, tzinfo=timezone.utc)
    for path in (indexed, pending, pending_registered):
        os.utime(path, (instant.timestamp(), instant.timestamp()))

    db = root / "activity-ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE events(id TEXT,timestamp_utc TEXT,source TEXT,duration_seconds REAL,data_json TEXT);"
        "CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,active_app TEXT,active_window TEXT,url TEXT,ocr_text TEXT,ocr_status TEXT);"
        "CREATE TABLE vision_descriptions(path TEXT,timestamp_utc TEXT,description TEXT,model_sha256 TEXT,prompt_version TEXT,screenshot_sha256 TEXT,elapsed_seconds REAL,status TEXT);"
    )
    connection.execute(
        "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(indexed), instant.isoformat(), "Example", "Example window", None, "visible words", "complete"),
    )
    connection.execute(
        "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(pending_registered), instant.isoformat(), "Example", "Example window", None, None, "pending"),
    )
    connection.execute(
        "INSERT INTO vision_descriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (str(indexed), instant.isoformat(), "Visible task.", "model-sha", "prompt-v1", "image-sha", 12.0, "complete"),
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr(daily_export, "DB_PATH", db)
    monkeypatch.setattr(daily_export, "SCREENSHOT_DIRS", [images.parent])
    rows = list(daily_export.ledger_events(instant - timedelta(hours=1), instant + timedelta(hours=1)))
    assert {row["source"] for row in rows} == {"mac_screen", "mac_screen_pending", "mac_vision_inference"}
    assert sum(row["source"] == "mac_screen_pending" for row in rows) == 2
    assert next(row for row in rows if row["source"] == "mac_screen")["data"]["visible_text"] == "visible words"
    assert next(row for row in rows if row["source"] == "mac_vision_inference")["data"]["untrusted_inference"] is True
    assert all(row["timestamp_utc"] == instant.isoformat() for row in rows)
