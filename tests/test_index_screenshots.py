from datetime import datetime, timedelta, timezone

import index_screenshots
import secure_store
from index_screenshots import nearby_mac_window


def test_screen_uses_only_nearby_foreground_window(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    now = datetime(2026, 9, 28, 20, tzinfo=timezone.utc)
    secure_store.insert_events(
        [{
            "timestamp_utc": now.isoformat(),
            "source": "mac_window_sample",
            "duration_seconds": 5,
            "data": {"app": "com.example.editor", "title": "Draft"},
        }]
    )
    with secure_store.connect() as database:
        assert nearby_mac_window(database, now + timedelta(seconds=10)) == ("com.example.editor", "Draft")
        assert nearby_mac_window(database, now + timedelta(seconds=30)) == (None, None)


def test_ocr_updates_pending_capture_without_losing_exact_context(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    image = tmp_path / "capture.webp"
    image.write_bytes(b"test")
    captured = datetime(2026, 9, 28, 20, tzinfo=timezone.utc)
    with secure_store.connect() as database:
        database.execute(
            "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(image), captured.isoformat(), "com.example.exact", "Exact window",
             None, None, "pending", captured.isoformat()),
        )
    monkeypatch.setattr(index_screenshots, "private_screenshots", lambda: [(captured.timestamp() + 2, image)])
    monkeypatch.setattr(index_screenshots, "index_one", lambda path, timestamp: {
        "path": str(path), "timestamp_utc": timestamp.isoformat(),
        "active_app": None, "active_window": None, "url": None,
        "ocr_text": "visible words", "ocr_status": "complete",
        "ocr_geometry": {"width": 500, "height": 300, "boxes": [[5, 6, 40, 8, 0.9]]},
    })
    result = index_screenshots.run(10)
    assert result["newly_indexed"] == 1
    with secure_store.connect() as database:
        assert database.execute(
            "SELECT timestamp_utc,active_app,active_window,ocr_text,ocr_status "
            "FROM screenshots WHERE path=?", (str(image),),
        ).fetchone() == (captured.isoformat(), "com.example.exact", "Exact window", "visible words", "complete")
        geometry = database.execute("SELECT image_width,image_height,boxes_json FROM ocr_geometry WHERE path=?",
                                    (str(image),)).fetchone()
        assert geometry == (500, 300, "[[5,6,40,8,0.9]]")


def test_ocr_receipt_is_private_and_contains_counts_only(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(index_screenshots, "STATE_DIR", tmp_path)
    monkeypatch.setattr(index_screenshots, "OCR_RECEIPT", tmp_path / "ocr-latest-receipt.json")
    result = {"discovered": 10, "already_indexed": 7, "newly_indexed": 2,
              "failed": 0, "pending_after_run": 1, "elapsed_seconds": 1.2}
    index_screenshots.write_receipt(result)
    path = tmp_path / "ocr-latest-receipt.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert '"status": "partial"' in path.read_text()
    assert "path" not in path.read_text()


def test_ocr_preserves_missing_title_during_app_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    captured = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    path = tmp_path / "capture.webp"
    path.write_bytes(b"synthetic")
    secure_store.insert_events([{"source": "mac_window_sample", "timestamp_utc": captured.isoformat(),
        "duration_seconds": 5, "data": {"app": "Editor", "title": "Previous draft"}}])
    with secure_store.connect() as db:
        db.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)", (str(path), captured.isoformat(),
                    "Browser", None, None, None, "pending", captured.isoformat()))
    monkeypatch.setattr(index_screenshots, "private_screenshots", lambda: [(captured.timestamp() + 2, path)])
    monkeypatch.setattr(index_screenshots, "index_one", lambda path, timestamp: {
        "path": str(path), "timestamp_utc": timestamp.isoformat(), "active_app": None,
        "active_window": None, "url": None, "ocr_text": "Recognized", "ocr_status": "complete"})
    assert index_screenshots.run(1)["newly_indexed"] == 1
    with secure_store.connect() as db:
        assert db.execute("SELECT timestamp_utc,active_app,active_window FROM screenshots").fetchone() == (
            captured.isoformat(), "Browser", None)


def test_overnight_ocr_keeps_new_capture_work_but_defers_layout_backfill(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    path = tmp_path / "old.webp"
    path.write_bytes(b"synthetic")
    now = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    with secure_store.connect() as database:
        database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                         (str(path), now.isoformat(), "Editor", "Draft", None,
                          "recognized", "complete", now.isoformat()))
    monkeypatch.setattr(index_screenshots, "private_screenshots",
                        lambda: [(now.timestamp(), path)])
    monkeypatch.setattr(index_screenshots, "in_overnight_window", lambda _: True)
    monkeypatch.setattr(index_screenshots, "backfill_geometry",
                        lambda *args: (_ for _ in ()).throw(AssertionError("backfill ran")))
    result = index_screenshots.run(10, geometry_backfill_limit=80)
    assert result["geometry_deferred_reason"] == "overnight_vision_window"
    assert result["geometry_pending"] == 1
