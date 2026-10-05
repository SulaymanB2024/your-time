"""Watch approved work folders for source-file changes without reading contents."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.fsevents import FSEventsObserver

from project_activity import SCAN_ROOT
from project_paths import is_work_path
from secure_store import STATE_DIR, connect, prepare_private_dir

ROOT = SCAN_ROOT
STATUS_PATH = STATE_DIR / "project-file-latest-receipt.json"
LOCK_PATH = STATE_DIR / "project-file-watch.lock"
MIN_FREE_BYTES = 10 * 1024**3
MAX_QUEUE = 10_000
FLUSH_SECONDS = 30
COALESCE_SECONDS = 120


def project_for(relative: Path) -> str | None:
    parts = relative.parts
    if not parts:
        return None
    if parts[0] == "CodexWork":
        return str(ROOT / "CodexWork" / parts[1]) if len(parts) >= 2 else None
    return str(ROOT / parts[0])


def allowed_file(path_text: str) -> tuple[str, str] | None:
    path = Path(path_text)
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return None
    if not relative.parts or len(str(relative)) > 512:
        return None
    if not is_work_path(relative):
        return None
    if path.is_symlink() or path.name.endswith((".tmp", ".swp", "~")):
        return None
    project = project_for(relative)
    return (project, str(relative)) if project else None


class QueuedEvents(FileSystemEventHandler):
    def __init__(self):
        self.queue = deque(maxlen=MAX_QUEUE)
        self.lock = threading.Lock()
        self.recent = {}
        self.dropped = 0
        self.coalesced = 0
        self.ignored = 0

    def on_any_event(self, event):
        if event.is_directory:
            return
        target = event.dest_path if event.event_type == "moved" else event.src_path
        scoped = allowed_file(target)
        if not scoped:
            self.ignored += 1
            return
        now = time.monotonic()
        key = (scoped[1], event.event_type)
        with self.lock:
            if now - self.recent.get(key, -COALESCE_SECONDS) < COALESCE_SECONDS:
                self.coalesced += 1
                return
            self.recent[key] = now
            if len(self.recent) > 20_000:
                self.recent = {item: at for item, at in self.recent.items()
                               if now - at < COALESCE_SECONDS}
            if len(self.queue) == MAX_QUEUE:
                self.dropped += 1
            self.queue.append((datetime.now(timezone.utc).isoformat(), scoped[0],
                               scoped[1], event.event_type))

    def drain(self) -> tuple[list[tuple], int]:
        with self.lock:
            rows = list(self.queue)
            self.queue.clear()
            dropped = self.dropped
            self.dropped = 0
        return rows, dropped


def write_receipt(value: dict) -> None:
    prepare_private_dir()
    fd, temporary = tempfile.mkstemp(prefix=".file-watch-receipt-", dir=STATE_DIR)
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


def flush(handler: QueuedEvents) -> dict:
    if shutil.disk_usage(STATE_DIR).free < MIN_FREE_BYTES:
        with handler.lock:
            queued, dropped = len(handler.queue), handler.dropped
        return {"status": "paused_low_disk", "inserted": 0,
                "not_written": queued, "dropped": dropped}
    rows, dropped = handler.drain()
    inserted = 0
    with connect() as database:
        for timestamp, project, relative, kind in rows:
            bucket = int(datetime.fromisoformat(timestamp).timestamp()) // COALESCE_SECONDS
            event_id = hashlib.sha256(json.dumps([bucket, relative, kind]).encode()).hexdigest()
            cursor = database.execute(
                "INSERT OR IGNORE INTO project_file_events VALUES (?, ?, ?, ?, ?)",
                (event_id, timestamp, project, relative, kind))
            inserted += cursor.rowcount
    return {"status": "watching", "inserted": inserted, "not_written": 0,
            "dropped": dropped, "last_event_utc": rows[-1][0] if rows else None,
            "coalesced_since_start": handler.coalesced,
            "ignored_generated_since_start": handler.ignored,
            "coalesce_seconds": COALESCE_SECONDS}


def main() -> None:
    os.umask(0o077)
    prepare_private_dir()
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        handler = QueuedEvents()
        observer = FSEventsObserver()
        observer.schedule(handler, str(ROOT), recursive=True)
        running = True

        def stop(signum, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        observer.start()
        write_receipt({"status": "watching", "root_count": 2, "inserted": 0,
                       "not_written": 0, "dropped": 0})
        try:
            while running:
                time.sleep(FLUSH_SECONDS)
                write_receipt({"root_count": 2, **flush(handler)})
        finally:
            observer.stop()
            observer.join(timeout=10)
            write_receipt({"root_count": 2, **flush(handler), "status": "stopped"})
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
