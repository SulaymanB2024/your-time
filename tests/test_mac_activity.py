import json
import os
import time
from collections import Counter

import mac_activity
from mac_activity import classify, select_window


def test_locked_and_idle_samples_do_not_keep_window_titles():
    assert classify(True, 0, "browser", "private title") == ("mac_locked", {})
    assert classify(False, 181, "browser", "private title") == ("mac_idle", {})
    assert classify(False, 2, "browser", "page") == (
        "mac_window_sample",
        {"app": "browser", "title": "page"},
    )


def test_small_fluid_overlay_uses_top_substantive_window():
    windows = [
        {"kCGWindowOwnerPID": 1, "kCGWindowLayer": 3,
         "kCGWindowBounds": {"Width": 600, "Height": 288}},
        {"kCGWindowOwnerPID": 2, "kCGWindowLayer": 0,
         "kCGWindowBounds": {"Width": 700, "Height": 40}},
        {"kCGWindowOwnerPID": 2, "kCGWindowLayer": 0,
         "kCGWindowBounds": {"Width": 700, "Height": 900},
         "kCGWindowName": "Project notes"},
    ]
    assert select_window(1, "com.FluidApp.app", windows,
                         lambda pid: {2: "com.example.editor"}.get(pid)) == (
        "com.example.editor", "Project notes")


def test_unknown_overlay_does_not_bypass_frontmost_sensitive_app():
    windows = [{"kCGWindowOwnerPID": 2, "kCGWindowLayer": 0,
                "kCGWindowBounds": {"Width": 700, "Height": 900},
                "kCGWindowName": "Work"}]
    assert select_window(1, "com.1password.1password", windows,
                         lambda pid: "com.example.editor") == (
        "com.1password.1password", None)


def test_accessibility_title_fills_only_a_missing_window_server_title():
    windows = [{"kCGWindowOwnerPID": 2, "kCGWindowLayer": 0,
                "kCGWindowBounds": {"Width": 700, "Height": 900}}]
    assert select_window(2, "com.example.editor", windows,
                         lambda pid: None, lambda pid: "Focused project") == (
        "com.example.editor", "Focused project")


def test_mac_health_receipt_is_private_and_contains_no_window_text(tmp_path, monkeypatch):
    monkeypatch.setattr(mac_activity, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mac_activity, "STATUS_PATH", tmp_path / "mac-activity-status.json")
    monkeypatch.setattr(mac_activity, "prepare_private_dir", lambda: None)
    monkeypatch.setattr(mac_activity, "CGPreflightScreenCaptureAccess", lambda: True)
    monkeypatch.setattr(mac_activity, "accessibility_trusted", lambda: True)
    mac_activity.write_status(Counter(samples=12, inserted=12, mac_window_sample=8,
                                      untitled=3), source="mac_window_sample",
                              untitled_streak=3)
    path = tmp_path / "mac-activity-status.json"
    item = json.loads(path.read_text())
    assert path.stat().st_mode & 0o777 == 0o600
    assert item["window_samples_without_title_since_start"] == 3
    assert set(item).isdisjoint({"app", "title", "window", "ocr_text"})
    mac_activity.write_status(Counter(samples=60, inserted=60, mac_window_sample=60,
                                      untitled=60), source="mac_window_sample",
                              untitled_streak=60)
    assert json.loads(path.read_text())["window_identity_status"] == "needs_review"
    mac_activity.write_status(Counter(samples=36, inserted=36, mac_window_sample=30,
                                      untitled=27), source="mac_window_sample",
                              untitled_streak=3)
    assert json.loads(path.read_text())["window_identity_status"] == "needs_review"
    monkeypatch.setattr(mac_activity, "accessibility_trusted", lambda: False)
    mac_activity.write_status(Counter(samples=12, mac_window_sample=10, untitled=2),
                              source="mac_window_sample", untitled_streak=0)
    assert json.loads(path.read_text())["window_identity_status"] == "permission_limited"


def test_accessibility_helper_accepts_only_bounded_title_and_hostname(monkeypatch):
    monkeypatch.setattr(mac_activity, "recent_reader_sample",
                        lambda: {"trusted": True, "pid": 1, "title": "x" * 700,
                                 "site_host": "example.com"})
    assert mac_activity.accessibility_document_domain(1) == "example.com"
    assert len(mac_activity.accessibility_window_title(1)) == 500
    monkeypatch.setattr(mac_activity, "recent_reader_sample",
                        lambda: {"trusted": True, "pid": 1, "title": None,
                                 "site_host": "example.com/private/token"})
    assert mac_activity.accessibility_document_domain(1) is None
    assert mac_activity.accessibility_window_details(2)["title"] is None


def test_window_reader_sample_requires_private_recent_regular_file(tmp_path, monkeypatch):
    sample = tmp_path / "window-reader-latest.json"
    sample.write_text('{"trusted":true,"pid":7,"title":"Draft","site_host":null}')
    monkeypatch.setattr(mac_activity, "WINDOW_READER_STATUS", sample)
    os.chmod(sample, 0o644)
    assert mac_activity.recent_reader_sample() == {}
    os.chmod(sample, 0o600)
    assert mac_activity.accessibility_window_details(7)["title"] == "Draft"
    old = time.time() - 30
    os.utime(sample, (old, old))
    assert mac_activity.recent_reader_sample() == {}
    sample.unlink()
    target = tmp_path / "target.json"
    target.write_text('{"trusted":true}')
    sample.symlink_to(target)
    assert mac_activity.recent_reader_sample() == {}


def test_mac_sample_persists_browser_host_without_full_url(monkeypatch):
    captured = []
    monkeypatch.setattr(mac_activity, "CGSessionCopyCurrentDictionary", lambda: {"kCGSSessionOnConsoleKey": True})
    monkeypatch.setattr(mac_activity, "CGEventSourceSecondsSinceLastEventType", lambda *args: 0)
    monkeypatch.setattr(mac_activity, "frontmost_window",
                        lambda include_domain=False: ("com.google.Chrome", "Draft", "example.com"))
    monkeypatch.setattr(mac_activity, "insert_events", lambda rows: captured.extend(rows) or 1)
    assert mac_activity.collect_one() == (1, "mac_window_sample", True)
    assert captured[0]["data"] == {"app": "com.google.Chrome", "title": "Draft",
                                    "site_host": "example.com"}


def test_recovered_sampler_health_uses_recent_samples_not_old_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(mac_activity, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mac_activity, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(mac_activity, "prepare_private_dir", lambda: None)
    monkeypatch.setattr(mac_activity, "CGPreflightScreenCaptureAccess", lambda: True)
    monkeypatch.setattr(mac_activity, "accessibility_trusted", lambda: True)
    mac_activity.write_status(Counter(samples=1000, mac_window_sample=1000, untitled=950),
                              source="mac_window_sample", untitled_streak=0,
                              recent_titles=[True] * 60)
    assert json.loads((tmp_path / "status.json").read_text())["window_identity_status"] == "normal"


def test_unknown_session_does_not_collect_window_identity(monkeypatch):
    rows = []
    monkeypatch.setattr(mac_activity, "CGSessionCopyCurrentDictionary", lambda: None)
    monkeypatch.setattr(mac_activity, "CGEventSourceSecondsSinceLastEventType", lambda *args: 0)
    monkeypatch.setattr(mac_activity, "frontmost_window", lambda **_: (_ for _ in ()).throw(AssertionError("window read")))
    monkeypatch.setattr(mac_activity, "insert_events", lambda data: rows.extend(data) or 1)
    assert mac_activity.collect_one() == (1, "mac_locked", False)
    assert rows[0]["data"] == {}
