import json
import os
from datetime import datetime, timedelta, timezone

import phone_quality


def test_source_regression_is_not_reported_as_fresh_import():
    now = datetime(2026, 9, 29, 18, tzinfo=timezone.utc)
    receipt = {"checked_at_utc": now.isoformat(), "status": "imported",
               "latest_closed_session_age_minutes": 180.0, "inserted_sessions": 0}
    result = phone_quality.evaluate(receipt, now - timedelta(minutes=60), now)
    assert result["quality_status"] == "source_behind_retained_ledger"
    assert result["source_behind_ledger_minutes"] == 120.0
    assert result["source_age_minutes"] == 180.0
    assert result["source_receipt_status"] == "imported"


def test_reader_failure_has_priority_over_stale_receipt():
    now = datetime(2026, 9, 29, 18, tzinfo=timezone.utc)
    receipt = {"checked_at_utc": (now - timedelta(days=2)).isoformat(),
               "latest_closed_session_age_minutes": 120.0}
    assert phone_quality.evaluate(receipt, None, now, reader_exit=1)["quality_status"] == "reader_failed"


def test_quality_receipt_and_history_stay_private(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_quality, "STATE_DIR", tmp_path)
    monkeypatch.setattr(phone_quality, "HISTORY_DIR", tmp_path / "history")
    value = {"checked_at_utc": "2026-09-29T18:00:00+00:00", "quality_status": "source_current_relative_to_ledger"}
    latest = tmp_path / "latest.json"
    phone_quality.private_write(latest, value)
    phone_quality.append_history(value)
    history = tmp_path / "history/2026-09-29.jsonl"
    assert os.stat(latest).st_mode & 0o777 == 0o600
    assert os.stat(history).st_mode & 0o777 == 0o600
    assert json.loads(history.read_text().splitlines()[0]) == value
