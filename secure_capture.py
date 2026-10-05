"""Capture changed screens into private files without Pensieve's web server."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from mss import mss
from PIL import Image, ImageChops, ImageStat
from Quartz import CGSessionCopyCurrentDictionary

from mac_activity import frontmost_window
from private_io import atomic_write, open_private_file, prepare_directory, write_json
from secure_store import STATE_DIR, connect, prepare_private_dir

SCREENSHOT_DIR = STATE_DIR / "screenshots"
LOCK_PATH = STATE_DIR / "screen-capture.lock"
STATUS_PATH = STATE_DIR / "screen-capture-status.json"
INTERVAL_SECONDS = 10
MIN_FREE_BYTES = 10 * 1024**3
MIN_FRAME_GAP_SECONDS = 20
MAX_FRAME_GAP_SECONDS = 120
MIN_VISUAL_CHANGE = 1.0
SENSITIVE_APPS = frozenset({
    "com.1password.1password", "com.agilebits.onepassword7",
    "com.bitwarden.desktop", "com.apple.passwords", "com.apple.keychainaccess",
    "com.apple.securityagent", "com.apple.loginwindow",
    "com.apple.systempreferences", "keepassxc", "dashlane",
})
SENSITIVE_TITLE_WORDS = (
    "password", "passcode", "verification code", "recovery key",
    "one-time code", "security code", "credit card", "bank account",
)


@dataclass
class FrameState:
    digest: bytes
    preview: Image.Image
    app: str
    window: str | None
    saved_at: float


@dataclass
class CaptureResult:
    saved: int
    skipped_redundant: int = 0
    blocked_reason: str | None = None


def screen_locked() -> bool:
    session = CGSessionCopyCurrentDictionary()
    # macOS omits the lock key in an unlocked session. An unavailable session
    # dictionary must never authorize a screenshot.
    return not session or bool(session.get("CGSSessionScreenIsLocked", False))


def sensitive_context(app: str | None, window: str | None) -> str | None:
    if not app:
        return "unknown_frontmost_app"
    app_name = app.casefold()
    if app_name in SENSITIVE_APPS or any(name in app_name for name in ("1password", "bitwarden", "keepassxc", "dashlane")):
        return "sensitive_app"
    if window and any(word in window.casefold() for word in SENSITIVE_TITLE_WORDS):
        return "sensitive_window_title"
    return None


def should_save_frame(previous: FrameState | None, digest: bytes, preview: Image.Image,
                      app: str, window: str | None, now: float) -> bool:
    if previous is None or (app, window) != (previous.app, previous.window):
        return True
    age = now - previous.saved_at
    if age < MIN_FRAME_GAP_SECONDS:
        return False
    if age >= MAX_FRAME_GAP_SECONDS:
        return True
    if digest == previous.digest:
        return False
    difference = ImageChops.difference(previous.preview, preview)
    return ImageStat.Stat(difference).mean[0] >= MIN_VISUAL_CHANGE


def private_day_dir(now: datetime) -> Path:
    prepare_private_dir()
    prepare_directory(SCREENSHOT_DIR)
    day_dir = SCREENSHOT_DIR / now.strftime("%Y%m%d")
    prepare_directory(day_dir)
    return day_dir


def save_private_image(image: Image.Image, path: Path) -> None:
    payload = BytesIO()
    image.save(payload, format="WEBP", quality=75, method=3)
    atomic_write(path, payload.getvalue())


def capture_once(previous_frames: dict[int, FrameState]) -> CaptureResult:
    if screen_locked():
        return CaptureResult(0, blocked_reason="screen_locked_or_unknown")
    if shutil.disk_usage(STATE_DIR).free < MIN_FREE_BYTES:
        return CaptureResult(0, blocked_reason="low_disk")
    app, window = frontmost_window()
    blocked = sensitive_context(app, window)
    if blocked:
        return CaptureResult(0, blocked_reason=blocked)
    now = datetime.now(timezone.utc)
    monotonic_now = time.monotonic()
    target_dir = private_day_dir(now)
    result = CaptureResult(0)
    with mss() as screen:
        for display_index, monitor in enumerate(screen.monitors[1:], start=1):
            previous = previous_frames.get(display_index)
            if previous and (app, window) == (previous.app, previous.window) and monotonic_now - previous.saved_at < MIN_FRAME_GAP_SECONDS:
                result.skipped_redundant += 1
                continue
            shot = screen.grab(monitor)
            digest = hashlib.blake2b(shot.bgra, digest_size=16).digest()
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            preview = image.convert("L")
            preview.thumbnail((128, 72), Image.Resampling.BILINEAR)
            if not should_save_frame(previous, digest, preview, app, window, monotonic_now):
                result.skipped_redundant += 1
                continue
            # The user may lock the Mac or switch to a sensitive app while the
            # display is being read. Discard that frame before writing it.
            if screen_locked() or frontmost_window() != (app, window):
                result.blocked_reason = "context_changed_during_capture"
                break
            filename = f"capture-{now.strftime('%Y%m%dT%H%M%S%fZ')}-display-{display_index}.webp"
            path = target_dir / filename
            save_private_image(image, path)
            with connect() as database:
                database.execute(
                    "INSERT OR IGNORE INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(path), now.isoformat(), app, window, None, None,
                     "pending", datetime.now(timezone.utc).isoformat()),
                )
            previous_frames[display_index] = FrameState(digest, preview, app, window, monotonic_now)
            result.saved += 1
    return result


def write_status(result: CaptureResult) -> None:
    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "screen_locked": screen_locked(),
        "screenshots_saved_this_check": result.saved,
        "screenshots_skipped_redundant_this_check": result.skipped_redundant,
        "capture_blocked_reason": result.blocked_reason,
        "free_bytes": shutil.disk_usage(STATE_DIR).free,
    }
    write_json(STATUS_PATH, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    prepare_private_dir()
    fd = open_private_file(LOCK_PATH)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return
    previous_frames: dict[int, FrameState] = {}
    try:
        while True:
            if shutil.disk_usage(STATE_DIR).free < MIN_FREE_BYTES:
                return
            started = time.monotonic()
            result = capture_once(previous_frames)
            write_status(result)
            if args.once:
                print(json.dumps({"screenshots_saved": result.saved, "blocked_reason": result.blocked_reason}))
                return
            time.sleep(max(0, INTERVAL_SECONDS - (time.monotonic() - started)))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
