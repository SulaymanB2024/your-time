import os

from PIL import Image

import secure_capture
import secure_store
from secure_capture import FrameState, save_private_image, sensitive_context, should_save_frame


def test_screenshot_write_is_private_and_atomic(tmp_path):
    path = tmp_path / "sample.webp"
    save_private_image(Image.new("RGB", (20, 20), "white"), path)
    assert path.is_file()
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert not path.with_name("." + path.name + ".tmp").exists()


def test_unknown_session_fails_closed(monkeypatch):
    monkeypatch.setattr(secure_capture, "CGSessionCopyCurrentDictionary", lambda: None)
    assert secure_capture.screen_locked()
    monkeypatch.setattr(secure_capture, "CGSessionCopyCurrentDictionary", lambda: {})
    assert secure_capture.screen_locked()
    monkeypatch.setattr(secure_capture, "CGSessionCopyCurrentDictionary", lambda: {"kCGSSessionOnConsoleKey": True})
    assert not secure_capture.screen_locked()


def test_sensitive_frontmost_context_is_not_captured():
    assert sensitive_context("com.1password.1password", None) == "sensitive_app"
    assert sensitive_context("com.apple.Safari", "Enter verification code") == "sensitive_window_title"
    assert sensitive_context(None, None) == "unknown_frontmost_app"
    assert sensitive_context("com.apple.Safari", "Research notes") is None


def test_capture_keeps_switches_and_periodic_context_but_skips_noise():
    base = Image.new("L", (128, 72), 100)
    same = base.copy()
    changed = Image.new("L", (128, 72), 180)
    previous = FrameState(b"old", base, "com.apple.Safari", "Page A", 100)
    assert not should_save_frame(previous, b"new", changed, "com.apple.Safari", "Page A", 110)
    assert not should_save_frame(previous, b"old", same, "com.apple.Safari", "Page A", 130)
    assert should_save_frame(previous, b"new", changed, "com.apple.Safari", "Page A", 130)
    assert should_save_frame(previous, b"new", changed, "com.apple.Safari", "Page B", 110)
    assert should_save_frame(previous, b"old", same, "com.apple.Safari", "Page A", 221)


def test_saved_frame_registers_exact_app_context_before_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(secure_capture, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_capture, "SCREENSHOT_DIR", tmp_path / "screenshots")
    monkeypatch.setattr(secure_capture, "screen_locked", lambda: False)
    monkeypatch.setattr(secure_capture, "frontmost_window", lambda: ("com.example.editor", "Draft"))

    class FakeShot:
        size = (20, 20)
        rgb = bytes([100, 120, 140]) * 400
        bgra = bytes([140, 120, 100, 255]) * 400

    class FakeScreen:
        monitors = [None, {"width": 20, "height": 20}]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def grab(self, _):
            return FakeShot()

    monkeypatch.setattr(secure_capture, "mss", FakeScreen)
    result = secure_capture.capture_once({})
    assert result.saved == 1
    with secure_store.connect() as database:
        assert database.execute(
            "SELECT active_app,active_window,ocr_status FROM screenshots"
        ).fetchone() == ("com.example.editor", "Draft", "pending")


def test_sensitive_app_blocks_capture_before_screen_read(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_capture, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_capture, "screen_locked", lambda: False)
    monkeypatch.setattr(secure_capture, "frontmost_window", lambda: ("com.1password.1password", "Vault"))
    monkeypatch.setattr(secure_capture, "mss", lambda: (_ for _ in ()).throw(AssertionError("screen read")))
    result = secure_capture.capture_once({})
    assert result.saved == 0
    assert result.blocked_reason == "sensitive_app"
