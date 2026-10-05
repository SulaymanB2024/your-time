"""Chrome Native Messaging bridge to the private local activity ledger."""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

from local_synthesis import safe_input_text
from private_io import write_json
from secure_store import STATE_DIR, insert_events, prepare_private_dir
from task_corrections import (
    clear_label,
    mark_outcome,
    retract_outcome,
    review,
    set_label,
)

STATUS_PATH = STATE_DIR / "browser-bridge-latest-receipt.json"
DASHBOARD_REFRESH = "/Users/sulaymanbowles/Projects/personal-activity-ledger/secure_dashboard.zsh"
MAX_MESSAGE = 64 * 1024
EVENT_TYPES = {"tab_active", "tab_updated", "window_focused", "browser_blur"}
DOMAIN_RE = re.compile(r"[a-z0-9.-]{1,253}\Z")


def write_status(*, stored: bool, reason: str | None = None) -> None:
    prepare_private_dir()
    prior = {}
    try:
        prior = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    value = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "received_since_start": int(prior.get("received_since_start", 0)) + 1,
             "stored_since_start": int(prior.get("stored_since_start", 0)) + int(stored),
             "last_status": "stored" if stored else reason or "ignored"}
    write_json(STATUS_PATH, value)


def tab_event(message: dict) -> dict:
    if message.get("incognito"):
        write_status(stored=False, reason="incognito_excluded")
        return {"ok": True, "stored": False}
    kind = message.get("event_type")
    if kind not in EVENT_TYPES:
        raise ValueError("invalid event type")
    at = datetime.fromisoformat(message.get("at", ""))
    if at.tzinfo is None or abs((datetime.now(timezone.utc) - at).total_seconds()) > 300:
        raise ValueError("event timestamp outside local window")
    domain = message.get("domain")
    if domain is not None and (not isinstance(domain, str) or not DOMAIN_RE.fullmatch(domain.casefold())):
        raise ValueError("invalid domain")
    if kind != "browser_blur" and not domain:
        raise ValueError("active tab needs a domain")
    raw_title = message.get("title")
    if raw_title is not None and not isinstance(raw_title, str):
        raise ValueError("invalid title")
    title = safe_input_text(raw_title, 180)
    tab_id, window_id = int(message.get("tab_id", 0)), int(message.get("window_id", 0))
    if not (0 <= tab_id < 2**31 and 0 <= window_id < 2**31):
        raise ValueError("invalid tab identity")
    row = {"timestamp_utc": at.astimezone(timezone.utc).isoformat(),
           "source": "browser_tab", "duration_seconds": None,
           "data": {"browser": "Chrome", "event_type": kind,
                    "site_host": domain.casefold() if domain else None,
                    "title": title or None,
                    "audible": bool(message.get("audible", False)),
                    "tab_id": tab_id,
                    "window_id": window_id}}
    inserted = insert_events([row])
    write_status(stored=bool(inserted))
    return {"ok": True, "stored": bool(inserted)}


def requested_day(message: dict) -> date:
    value = message.get("day")
    if not isinstance(value, str):
        raise ValueError("day required")
    day = date.fromisoformat(value)
    if day < date(2026, 9, 1) or day > datetime.now(timezone.utc).date() + timedelta(days=1):
        raise ValueError("day outside review range")
    return day


def refresh_dashboard() -> None:
    subprocess.run(["/bin/zsh", DASHBOARD_REFRESH], capture_output=True,
                   timeout=20, check=False)


def handle(message: dict) -> dict:
    if not isinstance(message, dict):
        raise ValueError("invalid message")
    kind = message.get("kind")
    if kind == "tab_event":
        return tab_event(message)
    if kind == "list_review":
        day = requested_day(message)
        offset = int(message.get("offset", 0))
        if offset < 0 or offset > 200:
            raise ValueError("invalid offset")
        rows = review(day)
        return {"ok": True, "day": day.isoformat(), "total": len(rows),
                "intervals": rows[offset:offset + 20]}
    if kind == "set_label":
        day = requested_day(message)
        result = set_label(day, str(message.get("id", "")), message.get("label", ""))
        refresh_dashboard()
        return {"ok": True, "correction": result}
    if kind == "clear_label":
        day = requested_day(message)
        cleared = clear_label(day, str(message.get("id", "")))
        if cleared:
            refresh_dashboard()
        return {"ok": True, "cleared": cleared}
    if kind == "mark_outcome":
        day = requested_day(message)
        result = mark_outcome(day, str(message.get("id", "")))
        refresh_dashboard()
        return {"ok": True, "outcome": result}
    if kind == "retract_outcome":
        day = requested_day(message)
        changed = retract_outcome(day, str(message.get("id", "")))
        if changed:
            refresh_dashboard()
        return {"ok": True, "retracted": changed}
    raise ValueError("unsupported message")


def read_message(stream) -> dict:
    header = stream.read(4)
    if len(header) != 4:
        raise ValueError("missing native message")
    size = struct.unpack("<I", header)[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("native message too large")
    body = stream.read(size)
    if len(body) != size:
        raise ValueError("truncated native message")
    return json.loads(body)


def write_message(stream, value: dict) -> None:
    body = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    stream.write(struct.pack("<I", len(body)))
    stream.write(body)
    stream.flush()


def main() -> None:
    os.umask(0o077)
    try:
        result = handle(read_message(sys.stdin.buffer))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        result = {"ok": False, "error": "invalid_request"}
    except Exception:
        result = {"ok": False, "error": "local_error"}
    write_message(sys.stdout.buffer, result)


if __name__ == "__main__":
    main()
