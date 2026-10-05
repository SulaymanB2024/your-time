import json
from datetime import datetime, timedelta, timezone

import daily_analysis
import import_legacy_mac


def test_legacy_import_intersects_non_afk_and_stops_before_new_collector():
    cutoff = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    rows = [
        ("currentwindow", "2026-09-28T11:00:00+00:00", 3600,
         json.dumps({"app": "Browser", "title": "First", "url": "https://private.example"})),
        ("currentwindow", "2026-09-28T11:20:00+00:00", 1200,
         json.dumps({"app": "Browser", "title": "Second"})),
        ("afkstatus", "2026-09-28T11:10:00+00:00", 1200,
         json.dumps({"status": "not-afk"})),
        ("afkstatus", "2026-09-28T11:30:00+00:00", 2400,
         json.dumps({"status": "afk"})),
        ("app", "2026-09-28T11:00:00+00:00", 2000,
         json.dumps({"app": "Phone"})),
    ]
    result = import_legacy_mac.build_rows(rows, cutoff)
    assert sum(row["duration_seconds"] for row in result
               if row["source"] == "mac_window_legacy") == 1200
    assert sum(row["duration_seconds"] for row in result
               if row["source"] == "mac_idle_legacy") == 1800
    assert all("url" not in row["data"] for row in result)
    assert all(row["timestamp_utc"] < cutoff.isoformat() for row in result)


def test_legacy_intervals_are_observed_with_distinct_provenance():
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    rows = [{"source": "mac_window_legacy", "timestamp_utc": start.isoformat(),
             "duration_seconds": 600, "data": {"app": "Browser", "title": "Draft"}},
            {"source": "mac_idle_legacy",
             "timestamp_utc": (start + timedelta(minutes=10)).isoformat(),
             "duration_seconds": 600, "data": {}}]
    segments, gaps = daily_analysis.mac_segments(rows, start, start + timedelta(minutes=20))
    assert [item["state"] for item in segments] == ["active", "idle"]
    assert segments[0]["evidence"] == "activitywatch_legacy_interval"
    assert sum(item["sampled_seconds"] for item in segments) == 1200
    assert gaps == []
