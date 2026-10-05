"""Sample frontmost Mac activity into private SQLite without a local server."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from AppKit import NSRunningApplication, NSWorkspace
from Foundation import NSDate, NSRunLoop
from Quartz import (
    CGPreflightScreenCaptureAccess,
    CGEventSourceSecondsSinceLastEventType,
    CGSessionCopyCurrentDictionary,
    CGWindowListCopyWindowInfo,
    kCGAnyInputEventType,
    kCGEventSourceStateCombinedSessionState,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
)

from secure_store import STATE_DIR, insert_events, prepare_private_dir


POLL_SECONDS = 5
IDLE_SECONDS = 180
MIN_FREE_BYTES = 10 * 1024**3
STATUS_PATH = STATE_DIR / "mac-activity-status.json"
LOCK_PATH = STATE_DIR / "mac-activity.lock"
OVERLAY_ONLY_BUNDLES = frozenset({"com.FluidApp.app"})
MIN_SUBSTANTIVE_WIDTH = 350
MIN_SUBSTANTIVE_HEIGHT = 250
WINDOW_READER_STATUS = STATE_DIR / "window-reader-latest.json"
HOST_RE = re.compile(r"^[a-z0-9.-]{1,253}$")


def recent_reader_sample() -> dict:
    """Read only a fresh, private sample from the signed launchd reader."""
    try:
        fd = os.open(WINDOW_READER_STATUS, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            age = time.time() - info.st_mtime
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 4096
                    or age < -5 or age > 12):
                return {}
            with os.fdopen(fd, "r", encoding="utf-8") as source:
                fd = -1
                payload = json.load(source)
                return payload if isinstance(payload, dict) else {}
        finally:
            if fd >= 0:
                os.close(fd)
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        return {}


def accessibility_trusted() -> bool:
    return recent_reader_sample().get("trusted") is True


def classify(locked: bool, idle_seconds: float, app: str | None, title: str | None):
    if locked:
        return "mac_locked", {}
    if idle_seconds >= IDLE_SECONDS:
        return "mac_idle", {}
    return "mac_window_sample", {"app": app, "title": title}


def substantive_window(window):
    bounds = window.get("kCGWindowBounds") or {}
    return (window.get("kCGWindowLayer") == 0
            and float(window.get("kCGWindowAlpha", 1)) > 0
            and float(bounds.get("Width", 0)) >= MIN_SUBSTANTIVE_WIDTH
            and float(bounds.get("Height", 0)) >= MIN_SUBSTANTIVE_HEIGHT)


def accessibility_window_details(pid: int) -> dict:
    if not isinstance(pid, int) or pid <= 0:
        return {"title": None, "site_host": None}
    result = recent_reader_sample()
    if result.get("trusted") is not True or result.get("pid") != pid:
        return {"title": None, "site_host": None}
    title = result.get("title")
    host = result.get("site_host")
    return {
        "title": title[:500] if isinstance(title, str) and title else None,
        "site_host": host if isinstance(host, str) and HOST_RE.fullmatch(host) else None,
    }


def accessibility_window_title(pid: int) -> str | None:
    return accessibility_window_details(pid)["title"]


def accessibility_document_domain(pid: int) -> str | None:
    return accessibility_window_details(pid)["site_host"]


def select_window(front_pid, front_app, windows, bundle_for_pid,
                  title_for_pid=lambda pid: None):
    own = next((item for item in windows
                if item.get("kCGWindowOwnerPID") == front_pid and substantive_window(item)), None)
    if own:
        return front_app, own.get("kCGWindowName") or title_for_pid(front_pid)

    # FluidVoice's small floating control can become macOS's frontmost app
    # while a different, full-size window remains the actual work surface.
    # Restrict the fallback to this known overlay so a sensitive app or modal
    # cannot silently be replaced with the window underneath it.
    if front_app in OVERLAY_ONLY_BUNDLES:
        visible = next((item for item in windows if substantive_window(item)), None)
        if visible:
            app = bundle_for_pid(visible.get("kCGWindowOwnerPID"))
            if app:
                return app, (visible.get("kCGWindowName")
                             or title_for_pid(visible.get("kCGWindowOwnerPID")))

    title = next((item.get("kCGWindowName") for item in windows
                  if item.get("kCGWindowOwnerPID") == front_pid
                  and item.get("kCGWindowLayer") == 0
                  and item.get("kCGWindowName")), None)
    return front_app, title or title_for_pid(front_pid)


def frontmost_window(*, include_domain: bool = False):
    # Both the sampler and screen recorder are long-lived Foundation clients.
    # Service pending app notifications before reading NSWorkspace's cache.
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.01))
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return (None, None, None) if include_domain else (None, None)
    pid = app.processIdentifier()
    app_id = app.bundleIdentifier() or app.localizedName()
    front_app = str(app_id) if app_id else None
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) or []
    def bundle_for_pid(owner_pid):
        if not isinstance(owner_pid, int):
            return None
        owner = NSRunningApplication.runningApplicationWithProcessIdentifier_(owner_pid)
        value = owner.bundleIdentifier() or owner.localizedName() if owner else None
        return str(value) if value else None

    details_by_pid = {}
    def details(owner_pid):
        if owner_pid not in details_by_pid:
            details_by_pid[owner_pid] = accessibility_window_details(owner_pid)
        return details_by_pid[owner_pid]

    chosen_app, title = select_window(pid, front_app,
                                      windows, bundle_for_pid,
                                      lambda owner_pid: details(owner_pid)["title"])
    if not include_domain:
        return chosen_app, str(title) if title else None
    selected_pid = pid if chosen_app == front_app else next(
        (window.get("kCGWindowOwnerPID") for window in windows
         if substantive_window(window)
         and bundle_for_pid(window.get("kCGWindowOwnerPID")) == chosen_app), None)
    browser = chosen_app and any(name in chosen_app.casefold()
                                 for name in ("chrome", "safari", "firefox", "edge", "arc"))
    host = details(selected_pid)["site_host"] if browser and selected_pid else None
    return chosen_app, str(title) if title else None, host


def collect_one():
    now = datetime.now(timezone.utc)
    session = CGSessionCopyCurrentDictionary()
    locked = not session or bool(session.get("CGSSessionScreenIsLocked"))
    idle = CGEventSourceSecondsSinceLastEventType(
        kCGEventSourceStateCombinedSessionState, kCGAnyInputEventType
    )
    app, title, host = ((None, None, None) if locked or idle >= IDLE_SECONDS
                        else frontmost_window(include_domain=True))
    source, data = classify(locked, idle, app, title)
    if source == "mac_window_sample" and host:
        data["site_host"] = host
    row = {
        "timestamp_utc": now.isoformat(),
        "source": source,
        "duration_seconds": POLL_SECONDS,
        "data": data,
    }
    return insert_events([row]), source, bool(data.get("title"))


def write_status(counts: Counter, *, source: str | None, untitled_streak: int,
                 stop_reason: str | None = None, recent_titles: list | None = None) -> None:
    prepare_private_dir()
    screen_access = bool(CGPreflightScreenCaptureAccess())
    ax_access = accessibility_trusted()
    window_count = counts["mac_window_sample"]
    untitled_fraction = counts["untitled"] / window_count if window_count else 0.0
    recent_windows = [item for item in recent_titles if item is not None] if recent_titles is not None else None
    recent_count = len(recent_windows) if recent_windows is not None else window_count
    recent_fraction = (sum(not item for item in recent_windows) / recent_count
                       if recent_windows else 0.0) if recent_windows is not None else untitled_fraction
    identity_status = ("permission_blocked" if window_count and not screen_access else
                       "needs_review" if untitled_streak >= 60 or
                       (recent_count >= 24 and recent_fraction >= 0.8) else
                       "permission_limited" if not ax_access else "normal")
    status = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "stopped" if stop_reason else "sampling",
              "last_source": source, "stop_reason": stop_reason,
              "samples_since_start": counts["samples"],
              "inserted_since_start": counts["inserted"],
              "window_samples_since_start": counts["mac_window_sample"],
              "window_samples_without_title_since_start": counts["untitled"],
              "window_samples_without_title_fraction": round(untitled_fraction, 3),
              "recent_window_samples": recent_count,
              "recent_window_samples_without_title_fraction": round(recent_fraction, 3),
              "consecutive_untitled_samples": untitled_streak,
              "window_identity_status": identity_status,
              "screen_capture_access": screen_access,
              "accessibility_access": ax_access,
              "idle_samples_since_start": counts["mac_idle"],
              "locked_samples_since_start": counts["mac_locked"]}
    fd, temporary = tempfile.mkstemp(prefix=".mac-status-", dir=STATE_DIR)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(status, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, STATUS_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    prepare_private_dir()
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        counts = Counter()
        streak = 0
        last_source = None
        recent_titles = deque(maxlen=60)
        while True:
            tick_started = time.monotonic()
            if shutil.disk_usage(STATE_DIR).free < MIN_FREE_BYTES:
                # The separate storage guard unloads both capture agents and
                # requires an explicit resume after storage is reclaimed.
                write_status(counts, source=last_source, untitled_streak=streak,
                             stop_reason="low_disk")
                return
            added, source, has_title = collect_one()
            recent_titles.append(has_title if source == "mac_window_sample" else None)
            counts["samples"] += 1
            counts["inserted"] += added
            counts[source] += 1
            streak = streak + 1 if source == "mac_window_sample" and not has_title else 0
            if source == "mac_window_sample" and not has_title:
                counts["untitled"] += 1
            last_source = source
            if args.once or counts["samples"] == 1 or counts["samples"] % 12 == 0:
                write_status(counts, source=source, untitled_streak=streak,
                             recent_titles=list(recent_titles))
            if args.once:
                print(json.dumps({"inserted": added, "source": source}))
                return
            # Keep the nominal five-second cadence. Sleeping five seconds
            # after each database write inflated small apparent data gaps.
            time.sleep(max(0.0, POLL_SECONDS - (time.monotonic() - tick_started)))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
