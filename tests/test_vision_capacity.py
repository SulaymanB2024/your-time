import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import vision_capacity


def test_capacity_uses_conservative_nearest_rank_and_actual_window():
    timings = [10.58, 11.39, 11.59, 11.86, 12.08, 15.94, 16.29, 20.1, 20.24, 29.04]
    assert vision_capacity.nearest_rank(timings, 0.90) == 20.24
    assert vision_capacity.frame_capacity(20.24) == 924
    assert vision_capacity.frame_capacity(100_000) == 0


def test_sustained_capacity_survives_later_skips_and_dedupes_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_capacity, "HISTORY_DIR", tmp_path)
    at = datetime(2026, 10, 3, 6, tzinfo=timezone.utc).isoformat()
    receipt = {"started_at_utc": at, "finished_at_utc": at,
               "elapsed_seconds": 6000, "completed": 50, "failed": 10}
    for name in ("one", "duplicate"):
        (tmp_path / f"vision-fallback-{name}.json").write_text(json.dumps(receipt))
    (tmp_path / "vision-fallback-battery.json").write_text(json.dumps({
        "started_at_utc": datetime(2026, 10, 4, 6, tzinfo=timezone.utc).isoformat(),
        "finished_at_utc": at, "completed": 0, "failed": 0, "stop_reason": "battery_power"}))
    nights = vision_capacity.sustained_primary_nights()
    assert len(nights) == 1
    assert nights[0]["runs"] == 1
    assert nights[0]["completed"] == 50
    assert nights[0]["estimated_completed_at_80_percent_window"] == 155


def test_capacity_keeps_capture_and_model_selection_separate(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,ocr_status TEXT)")
    connection.executemany("INSERT INTO screenshots VALUES (?,?,?)", [
        (str(tmp_path / "screenshots/a.webp"), datetime(2026, 9, 28, 20, tzinfo=timezone.utc).isoformat(), "complete"),
        (str(tmp_path / "screenshots/b.webp"), datetime(2026, 9, 28, 20, 1, tzinfo=timezone.utc).isoformat(), "pending"),
    ])
    connection.commit()
    connection.close()
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"models": {"2b_q4": "sha"},
                                   "images": [{"models": {"2b_q4": {"status": "complete", "elapsed_seconds": 20.24}}}]}))
    monkeypatch.setattr(vision_capacity, "DB_PATH", db)
    monkeypatch.setattr(vision_capacity, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_capacity, "RESULTS_PATH", results)
    monkeypatch.setattr(vision_capacity, "NIGHTLY_RECEIPT", tmp_path / "none.json")
    monkeypatch.setattr(vision_capacity, "PRIMARY_RECEIPT", tmp_path / "none-primary.json")
    monkeypatch.setattr(vision_capacity, "HISTORY_DIR", tmp_path / "none-history")
    monkeypatch.setattr(vision_capacity, "select_images", lambda day, seconds, limit: [None] * {10: 1, 20: 1, 30: 1, 60: 1}[seconds])
    report = vision_capacity.analysis_for_day(date(2026, 9, 28))
    assert report["secure_capture_counts_by_ocr_status"] == {"complete": 1, "pending": 1}
    assert report["ocr_complete_candidates_by_interval_seconds"]["20"] == 1
    assert report["current_day_coverage"]["selected_fraction_of_raw"] == 0.5
    assert report["benchmarks"]["2b_q4"]["pilot_capacity_at_80_percent_window"] == 924
    assert report["total_overnight_seconds"] == 27000
    assert report["text_reserved_seconds"] == 3600
    assert report["window_seconds"] == 23400
    assert report["provisional"]


def test_running_night_uses_latest_receipt_and_attempt_times(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,ocr_status TEXT)")
    connection.execute("CREATE TABLE vision_descriptions(status TEXT,elapsed_seconds REAL,updated_at_utc TEXT)")
    started = datetime.now(timezone.utc) - timedelta(minutes=2)
    for seconds in (10, 20, 30):
        connection.execute("INSERT INTO vision_descriptions VALUES (?,?,?)",
                           ("complete" if seconds < 30 else "incomplete", seconds,
                            (started + timedelta(seconds=seconds)).isoformat()))
    connection.commit()
    connection.close()
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"models": {"2b_q4": "sha"}, "images": [
        {"models": {"2b_q4": {"status": "complete", "elapsed_seconds": 20}}}]}))
    night = tmp_path / "night.json"
    night.write_text(json.dumps({"started_at_utc": started.isoformat(), "day_receipts": []}))
    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"started_at_utc": (started + timedelta(seconds=1)).isoformat(),
                                  "day_local": "2026-09-28", "selected": 3, "completed": 2,
                                  "failed": 1, "sensitive_skipped": 0}))
    monkeypatch.setattr(vision_capacity, "DB_PATH", db)
    monkeypatch.setattr(vision_capacity, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_capacity, "RESULTS_PATH", results)
    monkeypatch.setattr(vision_capacity, "NIGHTLY_RECEIPT", night)
    monkeypatch.setattr(vision_capacity, "LATEST_RECEIPT", latest)
    monkeypatch.setattr(vision_capacity, "PRIMARY_RECEIPT", tmp_path / "none-primary.json")
    monkeypatch.setattr(vision_capacity, "HISTORY_DIR", tmp_path / "none-history")
    monkeypatch.setattr(vision_capacity, "select_images", lambda *_: [])
    report = vision_capacity.analysis_for_day(date(2026, 9, 28))
    actual = report["actual_latest_night"]
    assert actual["selected"] == 3 and actual["completed"] == 2 and actual["failed"] == 1
    assert actual["observed_p90_model_attempt_seconds"] == 30
    assert actual["p90_capacity_at_80_percent_window"] == 624
    assert report["provisional"]


def test_primary_9b_capacity_uses_attempts_not_planned_candidates(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,ocr_status TEXT)")
    connection.commit()
    connection.close()
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"models": {}, "images": []}))
    started = datetime(2026, 9, 30, 6, tzinfo=timezone.utc)
    primary = tmp_path / "primary.json"
    primary.write_text(json.dumps({"started_at_utc": started.isoformat(),
                                   "finished_at_utc": (started + timedelta(seconds=1000)).isoformat(),
                                   "day_local": "2026-09-29", "selected": 500,
                                   "completed": 8, "failed": 2, "direct_completed": 6,
                                   "quality_completed": 2, "stop_reason": "overnight_window_ended"}))
    monkeypatch.setattr(vision_capacity, "DB_PATH", db)
    monkeypatch.setattr(vision_capacity, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_capacity, "RESULTS_PATH", results)
    monkeypatch.setattr(vision_capacity, "NIGHTLY_RECEIPT", tmp_path / "none.json")
    monkeypatch.setattr(vision_capacity, "PRIMARY_RECEIPT", primary)
    monkeypatch.setattr(vision_capacity, "HISTORY_DIR", tmp_path / "none-history")
    monkeypatch.setattr(vision_capacity, "select_images", lambda *_: [])
    actual = vision_capacity.analysis_for_day(date(2026, 9, 29))["actual_latest_night"]
    assert actual["planned_candidates"] == 500
    assert actual["attempted"] == 10
    assert actual["estimated_attempts_at_80_percent_window"] == 187
    assert actual["estimated_completed_at_80_percent_window_if_all_direct"] == 149


def test_combined_capacity_counts_unique_images_across_runs(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,ocr_status TEXT)")
    connection.execute("CREATE TABLE vision_descriptions(path TEXT,status TEXT,timestamp_utc TEXT)")
    at = datetime(2026, 9, 28, 20, tzinfo=timezone.utc).isoformat()
    connection.executemany("INSERT INTO screenshots VALUES (?,?,?)",
                           [(str(tmp_path / f"screenshots/{name}.webp"), at, "complete") for name in ("a", "b")])
    connection.executemany("INSERT INTO vision_descriptions VALUES (?,?,?)", [
        (str(tmp_path / "screenshots/a.webp"), "complete", at),
        (str(tmp_path / "screenshots/b.webp"), "sensitive_skipped", at),
    ])
    connection.commit()
    connection.close()
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"models": {"2b_q4": "sha"}, "images": [
        {"models": {"2b_q4": {"status": "complete", "elapsed_seconds": 10}}}]}))
    history = tmp_path / "history"
    history.mkdir()
    (history / "vision-nightly-first.json").write_text(json.dumps({
        "started_at_utc": at, "finished_at_utc": at,
        "day_receipts": [{"day_local": "2026-09-28", "elapsed_seconds": 20,
                          "completed": 1, "failed": 1}]}))
    (history / "vision-fallback-second.json").write_text(json.dumps({
        "started_at_utc": at, "finished_at_utc": at, "day_local": "2026-09-28",
        "elapsed_seconds": 10, "completed": 0, "failed": 0}))
    monkeypatch.setattr(vision_capacity, "DB_PATH", db)
    monkeypatch.setattr(vision_capacity, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_capacity, "RESULTS_PATH", results)
    monkeypatch.setattr(vision_capacity, "NIGHTLY_RECEIPT", tmp_path / "none.json")
    monkeypatch.setattr(vision_capacity, "PRIMARY_RECEIPT", tmp_path / "none-primary.json")
    monkeypatch.setattr(vision_capacity, "HISTORY_DIR", history)
    monkeypatch.setattr(vision_capacity, "select_images", lambda *_: [
        (str(tmp_path / "screenshots/a.webp"),), (str(tmp_path / "screenshots/b.webp"),)])
    report = vision_capacity.analysis_for_day(date(2026, 9, 28))
    combined = report["combined_night"]
    assert combined["total_elapsed_seconds"] == 30
    assert combined["selected_unique_screenshots"] == 2
    assert combined["complete_unique_screenshots"] == 1
    assert combined["sensitive_skipped_unique_screenshots"] == 1
    assert combined["model_attempts_across_runs"] == 2
    assert combined["sustainable_model_attempts_per_night_at_80_percent_window"] == 1248
