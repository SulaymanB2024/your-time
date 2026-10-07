from datetime import date, datetime, timedelta, timezone

import pytest

from chronicle_rollup import local_boundaries
from dashboard_assets import read_asset
from dashboard_metrics import activity_score, time_of_day


def segment(start, end, seconds, state="active"):
    return {"start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "sampled_seconds": seconds, "state": state, "app": "Editor", "window": "Draft"}


def session(start, end):
    return {"start_utc": start.isoformat(), "end_utc": end.isoformat(), "app": "Phone App"}


@pytest.mark.parametrize("day,hours", [(date(2026, 3, 8), 23),
                                      (date(2026, 9, 29), 24), (date(2026, 11, 1), 25)])
def test_day_length_and_device_totals_survive_clock_changes(day, hours):
    start, end = local_boundaries(day)
    analysis = {"day_local": day.isoformat(),
                "mac": {"segments": [segment(start, end, hours * 1800)]},
                "iphone": {"sessions": [session(start, end), session(start, end)]}}
    score = activity_score(analysis, {}, end)
    hourly = time_of_day(analysis)
    assert len(score) == hours * 12
    assert score[0]["start_utc"] == start.isoformat()
    assert score[-1]["end_utc"] == end.isoformat()
    assert sum(row["mac_seconds"] for row in score) == hours * 1800
    assert sum(row["iphone_seconds"] for row in score) == hours * 3600
    assert sum(hourly["mac"]) == hours * 1800
    assert sum(hourly["iphone"]) == hours * 3600


def test_clipping_exclusive_ends_gaps_and_weighted_samples():
    start, end = local_boundaries(date(2026, 9, 29))
    analysis = {"day_local": "2026-09-29", "mac": {"segments": [
        segment(start - timedelta(minutes=5), start + timedelta(minutes=5), 300),
        segment(start + timedelta(minutes=10), start + timedelta(minutes=15), 300, "unattributed"),
        segment(end, end + timedelta(minutes=5), 300),
        segment(start, start, 5)]}, "iphone": {"sessions": [
            session(start, start + timedelta(minutes=3)),
            session(start + timedelta(minutes=2), start + timedelta(minutes=5))]}}
    score = activity_score(analysis, {}, start + timedelta(minutes=20))
    assert score[0]["mac_seconds"] == 150
    assert score[0]["iphone_seconds"] == 300
    assert score[1]["mac_state"] == "unknown"
    assert score[2]["mac_state"] == "unattributed"
    assert score[2]["mac_unattributed_seconds"] == 300
    assert score[3]["mac_state"] == "unknown"
    assert score[4]["mac_state"] == "future"
    assert sum(row["mac_seconds"] for row in score) == 150
    hourly = time_of_day(analysis)
    assert sum(hourly["mac"]) == 150
    assert sum(hourly["unattributed"]) == 300
    assert sum(hourly["iphone"]) == 300


def test_screen_suggestions_stay_in_their_exact_supported_slot():
    start, end = local_boundaries(date(2026, 9, 29))
    analysis = {"day_local": "2026-09-29", "mac": {"segments": [
        segment(start, start + timedelta(minutes=10), 600, "unattributed")]},
        "iphone": {"sessions": []}}
    tags = [{"status": "model_inference", "timestamp_utc": start.isoformat(), "topic": "First"},
            {"status": "featureprint_match", "timestamp_utc": (start + timedelta(minutes=5)).isoformat(), "topic": "Second"},
            {"status": "wrong", "timestamp_utc": start.isoformat(), "topic": "Unsupported"},
            {"status": "model_inference", "timestamp_utc": end.isoformat(), "topic": "Outside"}]
    score = activity_score(analysis, {}, end, tags)
    assert score[0]["mac_screen_topic"] == "First"
    assert score[1]["mac_screen_topic"] == "Second"
    assert score[2]["mac_screen_topic"] is None


def test_offset_timestamps_reach_the_same_utc_slots():
    start, end = local_boundaries(date(2026, 9, 29))
    row = segment(start, start + timedelta(minutes=5), 300)
    analysis = {"day_local": "2026-09-29", "mac": {"segments": [row]}, "iphone": {"sessions": []}}
    expected = activity_score(analysis, {}, end)
    for key in ("start_utc", "end_utc"):
        row[key] = datetime.fromisoformat(row[key]).astimezone(timezone(timedelta(hours=2))).isoformat()
    assert activity_score(analysis, {}, end) == expected


@pytest.mark.parametrize("name", ["../private.json", "/tmp/secret", "unreviewed.js"])
def test_asset_loader_rejects_unregistered_paths(name):
    with pytest.raises(ValueError, match="Unknown dashboard source asset"):
        read_asset(name)
