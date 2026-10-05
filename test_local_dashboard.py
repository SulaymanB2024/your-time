import hashlib
import json
from datetime import datetime, timezone

import local_dashboard


def test_private_dashboard_escapes_untrusted_text_and_blocks_network(tmp_path, monkeypatch):
    data = {"schema_version": 1, "generated_at_utc": "2026-09-29T18:00:00+00:00",
            "timezone": "America/Chicago", "runtime": {}, "caption_eval": {},
            "days": [{"day": "2026-09-29", "complete_day": False,
                      "mac_active_seconds": 5, "mac_idle_seconds": 0,
                      "mac_locked_seconds": 0, "mac_unobserved_seconds": 0,
                      "iphone_focus_seconds": 0, "mac_apps": [], "iphone_apps": [],
                      "iphone_intervals": 0, "screenshots": 0, "ocr_complete": 0,
                      "ocr_pending": 0, "vision_statuses": {},
                      "synthesis_status": "not_run", "synthesis_coverage": {},
                      "themes": [], "candidate_outcomes": [],
                      "activity_categories": [],
                      "verified_accomplishments": [],
                      "blocks": [{"id": "b1", "app": "</script><script>alert(1)</script>"}]}]}
    payload = local_dashboard.render(data)
    assert b"connect-src &#x27;none&#x27;" in payload
    assert b"<script>alert(1)</script>" not in payload
    assert b"\\u003c/script\\u003e" in payload
    assert b'src="http' not in payload and b'href="http' not in payload
    monkeypatch.setattr(local_dashboard, "DASHBOARD_DIR", tmp_path)
    path = tmp_path / "index.html"
    local_dashboard.private_write(path, payload)
    assert path.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_snapshot_reuses_per_device_totals_without_adding_them(monkeypatch):
    def fake_analysis(day, now=None):
        return {"day_local": day.isoformat(), "complete_day": False,
                "start_utc": "2026-09-29T05:00:00+00:00",
                "analyzed_through_utc": "2026-09-29T05:02:00+00:00",
                "mac": {"state_sampled_seconds": {"active": 60},
                "observed_sampled_seconds": 60,
                "unobserved_seconds": 60, "app_sampled_seconds": [], "segments": []},
                "iphone": {"device_union_seconds": 120, "app_totals": [], "interval_count": 1,
                           "sessions": [{"start_utc": "2026-09-29T05:00:00+00:00",
                                         "end_utc": "2026-09-29T05:02:00+00:00", "app": "A"}]},
                "visual": {"indexed_screenshots": 1, "ocr_status_counts": {"complete": 1},
                           "new_capture_ocr_pending": 0, "vision_status_counts": {}}}
    monkeypatch.setattr(local_dashboard, "analyze", fake_analysis)
    monkeypatch.setattr(local_dashboard, "focus_build", lambda day, now=None: {"blocks": []})
    monkeypatch.setattr(local_dashboard, "read_json", lambda path: {})
    monkeypatch.setattr(local_dashboard, "read_supported_tags", lambda day: [])
    item = local_dashboard.daily_snapshot(datetime(2026, 9, 29).date(),
                                          datetime(2026, 9, 29, tzinfo=timezone.utc))
    assert item["mac_active_seconds"] == 60
    assert item["iphone_focus_seconds"] == 120
    assert "combined_seconds" not in item


def test_known_iphone_bundle_ids_have_readable_names():
    assert local_dashboard.app_name("com.atebits.Tweetie2") == "X"
    assert local_dashboard.app_name("com.facebook.hatch") == "Instagram"
    assert local_dashboard.app_name("com.google.ios.youtubemusic") == "YouTube Music"


def test_hourly_phone_intervals_use_union_and_mac_time_splits():
    analysis = {"day_local": "2026-09-29",
                "iphone": {"sessions": [
                    {"start_utc": "2026-09-29T14:00:00+00:00", "end_utc": "2026-09-29T14:30:00+00:00"},
                    {"start_utc": "2026-09-29T14:15:00+00:00", "end_utc": "2026-09-29T14:45:00+00:00"}]},
                "mac": {"segments": [{"state": "active", "start_utc": "2026-09-29T14:30:00+00:00",
                                      "end_utc": "2026-09-29T15:30:00+00:00", "sampled_seconds": 3600}]}}
    hourly = local_dashboard.time_of_day(analysis)
    assert hourly["iphone"][9] == 2700
    assert hourly["mac"][9] == 1800
    assert hourly["mac"][10] == 1800


def test_repeated_fall_back_hour_keeps_both_hours():
    analysis = {"day_local": "2026-11-01", "mac": {"segments": []},
                "iphone": {"sessions": [
                    {"start_utc": "2026-11-01T06:00:00+00:00", "end_utc": "2026-11-01T07:00:00+00:00"},
                    {"start_utc": "2026-11-01T07:00:00+00:00", "end_utc": "2026-11-01T08:00:00+00:00"}]}}
    assert local_dashboard.time_of_day(analysis)["iphone"][1] == 7200


def test_day_score_aligns_device_lanes_without_combining_time():
    analysis = {"day_local": "2026-09-29",
                "mac": {"segments": [{"state": "active", "app": "Editor", "window": "Draft",
                                      "start_utc": "2026-09-29T14:00:00+00:00",
                                      "end_utc": "2026-09-29T14:05:00+00:00",
                                      "sampled_seconds": 300}]},
                "iphone": {"sessions": [{"app": "Phone App",
                    "start_utc": "2026-09-29T14:02:30+00:00",
                    "end_utc": "2026-09-29T14:05:00+00:00"}]}}
    score = local_dashboard.activity_score(analysis, {},
        datetime(2026, 9, 29, 15, tzinfo=timezone.utc))
    target = next(x for x in score if x["start_utc"] == "2026-09-29T14:00:00+00:00")
    assert target["mac_seconds"] == 300
    assert target["mac_unattributed_seconds"] == 0
    assert target["iphone_seconds"] == 150
    assert target["mac_topic"] == "Editor (task unknown)"
    assert any(x["mac_state"] == "future" for x in score)


def test_day_score_keeps_titled_and_untitled_time_visible_together():
    analysis = {"day_local": "2026-09-29", "iphone": {"sessions": []},
                "mac": {"segments": [
                    {"state": "active", "app": "Editor", "window": "Draft",
                     "start_utc": "2026-09-29T14:00:00+00:00",
                     "end_utc": "2026-09-29T14:02:00+00:00", "sampled_seconds": 120},
                    {"state": "unattributed", "app": "Overlay", "window": None,
                     "start_utc": "2026-09-29T14:02:00+00:00",
                     "end_utc": "2026-09-29T14:05:00+00:00", "sampled_seconds": 180}]}}
    tags = [{"status": "model_inference", "topic": "Project planning",
             "timestamp_utc": "2026-09-29T14:03:00+00:00"}]
    score = local_dashboard.activity_score(analysis, {},
        datetime(2026, 9, 29, 15, tzinfo=timezone.utc), tags)
    target = next(x for x in score if x["start_utc"] == "2026-09-29T14:00:00+00:00")
    assert target["mac_seconds"] == 120
    assert target["mac_unattributed_seconds"] == 180
    assert target["mac_screen_topic"] == "Project planning"


def test_stale_synthesis_is_hidden_when_source_evidence_changes(monkeypatch):
    import copy
    from test_data_quality import valid_report
    from local_synthesis import block_projection, fingerprint
    analysis = valid_report()
    analysis.update(day_local="2026-10-03", complete_day=True)
    analysis["visual"].update(new_capture_ocr_pending=0, vision_status_counts={})
    analysis["mac"]["gaps_over_one_minute"] = []
    block = {"id": "b", "app": "Editor", "start_utc": analysis["start_utc"],
             "end_utc": analysis["mac"]["segments"][0]["end_utc"], "sampled_seconds": 5,
             "top_windows": [], "distinct_window_count": 0, "screenshot_count": 0, "visual_evidence": []}
    old_hash = fingerprint(block_projection(block))
    block["top_windows"] = [{"title": "Changed context", "sampled_seconds": 5}]
    synthesis = {"status": "complete", "blocks": [{"block_id": "b", "status": "complete",
                 "input_sha256": old_hash, "result": {"activity_kind": "writing", "focus_label": "Old task"}}],
                 "themes": [{"label": "Old theme"}], "candidate_outcomes": []}
    monkeypatch.setattr(local_dashboard, "analyze", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(local_dashboard, "focus_build", lambda *_args, **_kwargs: {"blocks": [block]})
    monkeypatch.setattr(local_dashboard, "read_json", lambda path: synthesis if path.name.startswith("synthesis-") else {})
    monkeypatch.setattr(local_dashboard, "read_supported_tags", lambda _: [])
    monkeypatch.setattr(local_dashboard, "correction_summary", lambda *_: [])
    monkeypatch.setattr(local_dashboard, "correction_review", lambda *_: [])
    monkeypatch.setattr(local_dashboard, "active_outcomes", lambda *_: [])
    monkeypatch.setattr(local_dashboard, "calendar_summary", lambda *_: {})
    monkeypatch.setattr(local_dashboard, "work_artifacts", lambda *_: {})
    item = local_dashboard.daily_snapshot(datetime(2026, 10, 3).date(), datetime(2026, 10, 4, tzinfo=timezone.utc))
    assert item["blocks"][0]["inference"] is None
    assert item["themes"] == []
    assert item["synthesis_status"] == "stale_evidence"
    assert item["data_quality"]["status"] == "passed"
