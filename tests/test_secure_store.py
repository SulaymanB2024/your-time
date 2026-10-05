import os

import pytest

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


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_database_and_sidecars_reject_symlinks(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "activity-ledger.sqlite3")
    victim = tmp_path / "unrelated"
    victim.write_bytes(b"preserve")
    victim.chmod(0o600)
    (tmp_path / (secure_store.DB_PATH.name + suffix)).symlink_to(victim)
    with pytest.raises(OSError):
        with secure_store.connect():
            pytest.fail("must reject before database access")
    assert victim.read_bytes() == b"preserve"


def test_failed_transaction_rolls_back_and_sidecars_stay_private(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "activity-ledger.sqlite3")
    previous = os.umask(0o022)
    try:
        with pytest.raises(ValueError, match="abort"):
            with secure_store.connect() as connection:
                connection.execute("INSERT INTO events VALUES ('synthetic', 'now', 'test', 1, '{}', 'now')")
                for suffix in ("", "-wal", "-shm"):
                    assert (tmp_path / (secure_store.DB_PATH.name + suffix)).stat().st_mode & 0o777 == 0o600
                assert os.umask(0o022) == 0o022
                raise ValueError("abort")
        with secure_store.connect() as connection:
            assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    finally:
        os.umask(previous)
