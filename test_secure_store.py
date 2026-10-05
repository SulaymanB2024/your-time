import os

import secure_store


def test_local_events_are_private_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "activity-ledger.sqlite3")
    row = {
        "timestamp_utc": "2026-09-28T20:00:00+00:00",
        "source": "iphone_app",
        "duration_seconds": 15.0,
        "data": {"app": "example"},
    }
    assert secure_store.insert_events([row]) == 1
    assert secure_store.insert_events([row]) == 0
    assert os.stat(secure_store.DB_PATH).st_mode & 0o777 == 0o600
    assert os.stat(tmp_path).st_mode & 0o777 == 0o700
