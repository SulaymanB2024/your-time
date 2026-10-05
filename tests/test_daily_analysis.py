import os
import sqlite3
from datetime import date, datetime, timedelta, timezone

import daily_analysis


def test_phone_intervals_clip_at_day_boundary_and_report_overlap():
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    rows = [
        {"source": "iphone_app", "timestamp_utc": (start - timedelta(minutes=5)).isoformat(),
         "duration_seconds": 600, "data": {"app": "A"}},
        {"source": "iphone_app", "timestamp_utc": (start + timedelta(minutes=3)).isoformat(),
         "duration_seconds": 600, "data": {"app": "B"}},
    ]
    result = daily_analysis.phone_analysis(rows, start, end)
    assert result["interval_count"] == 2
    assert result["device_union_seconds"] == 780
    assert result["overlap_seconds"] == 120
    assert result["sessions"][0]["start_utc"] == start.isoformat()


def test_mac_samples_keep_unknown_gaps_separate_from_idle():
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    end = start + timedelta(minutes=3)
    rows = [
        {"source": "mac_window_sample", "timestamp_utc": start.isoformat(),
         "duration_seconds": 5, "data": {"app": "Editor", "title": "Draft"}},
        {"source": "mac_window_sample", "timestamp_utc": (start + timedelta(seconds=5.1)).isoformat(),
         "duration_seconds": 5, "data": {"app": "Editor", "title": "Draft"}},
        {"source": "mac_idle", "timestamp_utc": (start + timedelta(minutes=2)).isoformat(),
         "duration_seconds": 5, "data": {}},
    ]
    segments, gaps = daily_analysis.mac_segments(rows, start, end)
    assert len(segments) == 2
    assert segments[0]["sample_count"] == 2
    assert segments[0]["sampled_seconds"] == 10
    assert segments[1]["state"] == "idle"
    assert len(gaps) == 1
    assert gaps[0]["seconds"] > 100


def test_frontmost_app_without_window_is_unattributed_not_app_use():
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    rows = [{"source": "mac_window_sample", "timestamp_utc": start.isoformat(),
             "duration_seconds": 5, "data": {"app": "com.FluidApp.app", "title": None}}]
    segments, _ = daily_analysis.mac_segments(rows, start, start + timedelta(seconds=10))
    assert segments[0]["state"] == "unattributed"
    assert segments[0]["app"] == "com.FluidApp.app"
    assert segments[0]["window"] is None


def test_browser_domain_transition_splits_segments_without_losing_samples():
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    rows = [
        {"source": "mac_window_sample", "timestamp_utc": start.isoformat(),
         "duration_seconds": 5,
         "data": {"app": "com.google.Chrome", "title": "New Tab", "site_host": "one.example"}},
        {"source": "mac_window_sample", "timestamp_utc": (start + timedelta(seconds=5)).isoformat(),
         "duration_seconds": 5,
         "data": {"app": "com.google.Chrome", "title": "New Tab", "site_host": "two.example"}},
    ]
    segments, _ = daily_analysis.mac_segments(rows, start, start + timedelta(seconds=10))
    assert [item["site_host"] for item in segments] == ["one.example", "two.example"]
    assert sum(item["sampled_seconds"] for item in segments) == 10


def test_browser_tab_time_is_clipped_to_actual_chrome_foreground():
    start = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    rows = [
        {"source": "browser_tab", "timestamp_utc": start.isoformat(),
         "data": {"event_type": "tab_active", "site_host": "one.example"}},
        {"source": "browser_tab", "timestamp_utc": (start + timedelta(minutes=5)).isoformat(),
         "data": {"event_type": "tab_active", "site_host": "two.example"}},
        {"source": "browser_tab", "timestamp_utc": (start + timedelta(minutes=8)).isoformat(),
         "data": {"event_type": "browser_blur", "site_host": None}},
    ]
    mac = [{"state": "active", "app": "com.google.Chrome", "window": "Work",
            "start_utc": start.isoformat(),
            "end_utc": (start + timedelta(minutes=10)).isoformat(),
            "sampled_seconds": 600}]
    result = daily_analysis.browser_analysis(rows, mac, start, start + timedelta(minutes=15))
    assert result["tracked_seconds"] == 480
    assert result["domains"] == [
        {"site_host": "one.example", "seconds": 300},
        {"site_host": "two.example", "seconds": 180}]


def test_analysis_reads_private_sqlite_and_marks_partial_day(tmp_path, monkeypatch):
    db = tmp_path / "activity-ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE events(timestamp_utc TEXT,source TEXT,duration_seconds REAL,data_json TEXT);"
        "CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,ocr_status TEXT);"
        "CREATE TABLE vision_descriptions(timestamp_utc TEXT,status TEXT);"
    )
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    connection.execute("INSERT INTO events VALUES (?,?,?,?)",
                       (start.isoformat(), "mac_window_sample", 5, '{"app":"Editor","title":"Draft"}'))
    connection.execute("INSERT INTO events VALUES (?,?,?,?)",
                       ((start + timedelta(seconds=5)).isoformat(), "mac_window_sample", 5,
                        '{"app":"com.google.Chrome","title":null}'))
    connection.execute("INSERT INTO events VALUES (?,?,?,?)",
                       ((start + timedelta(seconds=10)).isoformat(), "mac_window_sample", 5,
                        '{"app":"com.FluidApp.app","title":null}'))
    connection.execute("INSERT INTO events VALUES (?,?,?,?)",
                       ((start - timedelta(minutes=1)).isoformat(), "iphone_app", 120, '{"app":"Phone A"}'))
    connection.commit()
    connection.close()
    monkeypatch.setattr(daily_analysis, "DB_PATH", db)
    monkeypatch.setattr(daily_analysis, "SCREENSHOT_DIR", tmp_path / "screenshots")
    quality = tmp_path / "phone-quality.json"
    quality.write_text('{"checked_at_utc":"2026-09-28T05:00:00+00:00","quality_status":"source_behind_retained_ledger","source_behind_ledger_minutes":120}')
    monkeypatch.setattr(daily_analysis, "PHONE_QUALITY", quality)
    report = daily_analysis.analyze(date(2026, 9, 28), now=start + timedelta(minutes=2))
    assert not report["complete_day"]
    assert report["mac"]["observed_sampled_seconds"] == 15
    assert report["mac"]["unobserved_seconds"] == 105
    assert report["mac"]["app_only_seconds"] == 5
    assert report["mac"]["app_sampled_seconds"] == [
        {"app": "Editor", "seconds": 5}, {"app": "com.google.Chrome", "seconds": 5}]
    assert report["iphone"]["device_union_seconds"] == 60
    assert report["interpretation_rules"]
    assert report["source_quality"]["iphone_sync"]["quality_status"] == "source_behind_retained_ledger"
    local_now = (start + timedelta(minutes=2)).astimezone(timezone(timedelta(hours=-5)))
    local_report = daily_analysis.analyze(date(2026, 9, 28), now=local_now)
    assert local_report["analyzed_through_utc"] == report["analyzed_through_utc"]
    assert local_report["mac"]["observed_sampled_seconds"] == 15
    assert local_report["iphone"]["device_union_seconds"] == 60
    out = tmp_path / "private.json"
    monkeypatch.setattr(daily_analysis, "ANALYSIS_DIR", tmp_path)
    daily_analysis.private_write(out, b"{}\n")
    assert os.stat(out).st_mode & 0o777 == 0o600
