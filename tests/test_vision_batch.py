import os
import struct
from datetime import date, datetime, timedelta, timezone

import secure_store
import vision_batch


def test_description_requires_final_answer_and_redacts_identifiers():
    assert vision_batch.clean_description(b"unfinished thinking") is None
    text = b"<think>internal</think> A code 123456 was shown at https://example.com and a@example.com."
    result = vision_batch.clean_description(text)
    assert "internal" not in result
    assert "123456" not in result
    assert "https://" not in result
    assert "a@example.com" not in result


def test_selection_keeps_latest_image_in_each_interval(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    # Local midnight on September 28 is 05:00 UTC in Chicago.
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    with secure_store.connect() as database:
        for offset in (5, 25, 70):
            database.execute(
                "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"/private/example-{offset}.webp",
                    (start + timedelta(seconds=offset)).isoformat(),
                    None, None, None, "", "complete", start.isoformat(),
                ),
            )
        database.execute(
            "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("/private/pending.webp", (start + timedelta(seconds=95)).isoformat(),
             "example", "window", None, None, "pending", start.isoformat()),
        )
    rows = vision_batch.select_images(date(2026, 9, 28), 60, 10)
    assert [row[0] for row in rows] == ["/private/example-25.webp", "/private/example-70.webp"]


def test_spread_selection_covers_day_instead_of_first_frames_only():
    assert vision_batch.spread_selection(list(range(10)), 3) == [0, 4, 9]
    assert vision_batch.spread_selection(list(range(10)), 1) == [5]


def test_featureprint_deduplication_keeps_distinct_screen_change():
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    rows = [(f"/private/capture-{index}-display-1.webp",
             (start + timedelta(seconds=40 * index)).isoformat(), "Editor", "Draft", "")
            for index in range(3)]
    vector = lambda value: struct.pack("<768f", *([value] * 768))
    features = {rows[0][0]: vector(0), rows[1][0]: vector(.001),
                rows[2][0]: vector(.1)}
    kept = vision_batch.dedupe_by_featureprint(rows, features)
    assert [row[0] for row in kept] == [rows[0][0], rows[2][0]]


def test_overnight_window_remaining_time_is_bounded():
    at = datetime(2026, 9, 29, 11, 58, tzinfo=timezone.utc)  # 06:58 in Chicago
    assert vision_batch.seconds_until_window_end(at) == 120
    assert vision_batch.seconds_until_window_end(at + timedelta(minutes=2)) == 0


def test_run_skips_existing_and_spreads_pending_under_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(vision_batch, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_batch, "RECEIPT_PATH", tmp_path / "receipt.json")
    start = datetime(2026, 9, 28, 5, tzinfo=timezone.utc)
    paths = []
    with secure_store.connect() as database:
        for offset in range(0, 600, 60):
            path = tmp_path / f"image-{offset}.webp"
            path.write_bytes(b"synthetic")
            paths.append(path)
            at = (start + timedelta(seconds=offset)).isoformat()
            database.execute("INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (str(path), at, "Editor", "Draft", None, "ordinary", "complete", at))
        for path in (paths[0], paths[-1]):
            at = start.isoformat()
            database.execute("INSERT INTO vision_descriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (str(path), "model-sha", vision_batch.PROMPT_VERSION,
                              "image-sha", at, "Done", "complete", 1, 1.0, at))
    monkeypatch.setattr(vision_batch, "resource_gate", lambda **_: None)
    monkeypatch.setattr(vision_batch, "verify_model", lambda: (paths[0], paths[0], "model-sha"))
    seen = []
    monkeypatch.setattr(vision_batch, "describe_image", lambda path, *_: (seen.append(path) or "Visible task.", "complete", 0.1))
    result = vision_batch.run(date(2026, 9, 28), 60, 3, benchmark_now=True)
    assert result["selected"] == 10
    assert result["skipped_existing"] == 2
    assert result["selected_for_inference"] == 3
    assert result["deferred_capacity"] == 5
    assert seen[0] == paths[1] and seen[-1] == paths[-2]


def test_resume_skips_completed_description(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(vision_batch, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_batch, "RECEIPT_PATH", tmp_path / "receipt.json")
    image = tmp_path / "image.webp"
    image.write_bytes(b"synthetic image bytes")
    timestamp = datetime(2026, 9, 28, 5, 1, tzinfo=timezone.utc).isoformat()
    with secure_store.connect() as database:
        database.execute(
            "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(image), timestamp, "example", "window", "", "ordinary text", "complete", timestamp),
        )
    monkeypatch.setattr(vision_batch, "resource_gate", lambda **_: None)
    monkeypatch.setattr(vision_batch, "verify_model", lambda: (image, image, "model-sha"))
    monkeypatch.setattr(vision_batch, "describe_image", lambda *_: ("Visible task.", "complete", 0.1))
    first = vision_batch.run(date(2026, 9, 28), 60, 1, benchmark_now=True)
    second = vision_batch.run(date(2026, 9, 28), 60, 1, benchmark_now=True)
    assert first["completed"] == 1
    assert second["completed"] == 0
    assert second["skipped_existing"] == 1
    assert (tmp_path / "receipt.json").is_file()


def test_sensitive_ocr_terms_are_detected():
    assert vision_batch.SENSITIVE_RE.search("Enter the verification code")


def test_memory_report_parsing_uses_free_percent_not_vm_counter():
    assert vision_batch.memory_free_percent("System-wide memory free percentage: 67%") == 67
    assert vision_batch.memory_free_percent("vm.memory_pressure: 554") is None


def test_historical_receipt_name_is_unique_and_private(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_batch, "HISTORICAL_RECEIPT_DIR", tmp_path)
    first = vision_batch.historical_receipt_path("2026-09-29T06:00:04.312690+00:00")
    second = vision_batch.historical_receipt_path("2026-09-29T09:16:09.719390+00:00")
    assert first != second
    vision_batch.write_receipt({"started_at_utc": "2026-09-29T06:00:04.312690+00:00"}, first)
    assert os.stat(first).st_mode & 0o777 == 0o600


def test_sensitive_screen_skips_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(vision_batch, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_batch, "RECEIPT_PATH", tmp_path / "receipt.json")
    image = tmp_path / "sensitive.webp"
    image.write_bytes(b"synthetic image bytes")
    timestamp = datetime(2026, 9, 28, 5, 1, tzinfo=timezone.utc).isoformat()
    with secure_store.connect() as database:
        database.execute(
            "INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(image), timestamp, "com.example", "Login", "", "Enter password", "complete", timestamp),
        )
    monkeypatch.setattr(vision_batch, "resource_gate", lambda **_: None)
    monkeypatch.setattr(vision_batch, "verify_model", lambda: (image, image, "model-sha"))
    monkeypatch.setattr(vision_batch, "describe_image", lambda *_: (_ for _ in ()).throw(AssertionError("model called")))
    result = vision_batch.run(date(2026, 9, 28), 60, 1, benchmark_now=True)
    assert result["sensitive_skipped"] == 1
    assert result["completed"] == 0
