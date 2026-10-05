"""Private local SQLite store for activity records; no HTTP listener."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from private_io import open_private_file, prepare_directory, validate_file

STATE_DIR = Path.home() / "Library/Application Support/personal-activity-ledger"
DB_PATH = STATE_DIR / "activity-ledger.sqlite3"


def prepare_private_dir() -> None:
    prepare_directory(STATE_DIR)


@contextmanager
def connect():
    prepare_private_dir()
    # Create mode 0600 before SQLite opens it. SQLite uses the database mode
    # for its WAL/SHM files; do not mutate the process-wide umask in a context
    # manager that can be used by more than one thread.
    os.close(open_private_file(DB_PATH))
    for suffix in ("-wal", "-shm", "-journal"):
        validate_file(DB_PATH.with_name(DB_PATH.name + suffix))
    connection = None
    try:
        connection = sqlite3.connect(DB_PATH, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY,
              timestamp_utc TEXT NOT NULL,
              source TEXT NOT NULL,
              duration_seconds REAL,
              data_json TEXT NOT NULL,
              inserted_at_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_time ON events(timestamp_utc);
            CREATE TABLE IF NOT EXISTS screenshots (
              path TEXT PRIMARY KEY,
              timestamp_utc TEXT NOT NULL,
              active_app TEXT,
              active_window TEXT,
              url TEXT,
              ocr_text TEXT,
              ocr_status TEXT NOT NULL,
              indexed_at_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS screenshots_time ON screenshots(timestamp_utc);
            CREATE TABLE IF NOT EXISTS ocr_geometry (
              path TEXT PRIMARY KEY,
              image_width INTEGER NOT NULL,
              image_height INTEGER NOT NULL,
              boxes_json TEXT NOT NULL,
              updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS screen_features (
              path TEXT PRIMARY KEY,
              screenshot_sha256 TEXT NOT NULL,
              feature_revision INTEGER NOT NULL,
              element_count INTEGER NOT NULL,
              vector BLOB NOT NULL,
              indexed_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vision_descriptions (
              path TEXT NOT NULL,
              model_sha256 TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              screenshot_sha256 TEXT NOT NULL,
              timestamp_utc TEXT NOT NULL,
              description TEXT,
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL,
              elapsed_seconds REAL,
              updated_at_utc TEXT NOT NULL,
              PRIMARY KEY(path, model_sha256, prompt_version)
            );
            CREATE INDEX IF NOT EXISTS vision_descriptions_time ON vision_descriptions(timestamp_utc);
            CREATE TABLE IF NOT EXISTS git_receipts (
              id TEXT PRIMARY KEY,
              repo_common_key TEXT NOT NULL,
              repo_path TEXT NOT NULL,
              commit_sha TEXT NOT NULL,
              committed_at_utc TEXT NOT NULL,
              observed_at_utc TEXT NOT NULL,
              evidence_tier TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS git_receipts_time ON git_receipts(committed_at_utc);
            CREATE TABLE IF NOT EXISTS project_file_events (
              id TEXT PRIMARY KEY,
              timestamp_utc TEXT NOT NULL,
              project_root TEXT NOT NULL,
              relative_path TEXT NOT NULL,
              event_type TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS project_file_events_time ON project_file_events(timestamp_utc);
            CREATE TABLE IF NOT EXISTS task_corrections (
              id TEXT PRIMARY KEY,
              day_local TEXT NOT NULL,
              start_utc TEXT NOT NULL,
              end_utc TEXT NOT NULL,
              label TEXT NOT NULL,
              created_at_utc TEXT NOT NULL,
              evidence_tier TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS task_corrections_day ON task_corrections(day_local);
            CREATE TABLE IF NOT EXISTS task_outcomes (
              id TEXT PRIMARY KEY,
              day_local TEXT NOT NULL,
              label TEXT NOT NULL,
              source_correction_id TEXT NOT NULL,
              evidence_tier TEXT NOT NULL,
              created_at_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS task_outcomes_day ON task_outcomes(day_local);
            CREATE TABLE IF NOT EXISTS calendar_items (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              calendar_id TEXT NOT NULL,
              start_utc TEXT NOT NULL,
              end_utc TEXT,
              title TEXT,
              completed INTEGER,
              observed_at_utc TEXT NOT NULL,
              all_day INTEGER
            );
            CREATE INDEX IF NOT EXISTS calendar_items_time ON calendar_items(start_utc);
            """
        )
        if "all_day" not in {row[1] for row in connection.execute("PRAGMA table_info(calendar_items)")}:
            connection.execute("ALTER TABLE calendar_items ADD COLUMN all_day INTEGER")
        yield connection
        connection.commit()
    except BaseException:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            connection.close()


def insert_events(rows: list[dict]) -> int:
    added = 0
    now = datetime.now(timezone.utc).isoformat()
    with connect() as connection:
        for row in rows:
            data_json = json.dumps(row["data"], sort_keys=True, ensure_ascii=False)
            identity = json.dumps(
                [row["source"], row["timestamp_utc"], row.get("duration_seconds"), data_json],
                separators=(",", ":"),
            )
            event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    row["timestamp_utc"],
                    row["source"],
                    row.get("duration_seconds"),
                    data_json,
                    now,
                ),
            )
            added += cursor.rowcount
    return added
