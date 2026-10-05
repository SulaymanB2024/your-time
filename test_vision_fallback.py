from datetime import date, datetime, timezone

import secure_store
import vision_fallback


def test_fallback_selects_only_ocr_complete_2b_incompletes(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    at = datetime(2026, 9, 28, 20, tzinfo=timezone.utc).isoformat()
    with secure_store.connect() as database:
        for path, ocr_status in [("/private/one.webp", "complete"),
                                 ("/private/two.webp", "pending"),
                                 ("/private/three.webp", "complete")]:
            database.execute("INSERT INTO screenshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (path, at, "Editor", "Draft", None, "words", ocr_status, at))
        for path in ("/private/one.webp", "/private/two.webp", "/private/three.webp"):
            database.execute("INSERT INTO vision_descriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (path, "2b-sha", vision_fallback.PROMPT_VERSION, "image-sha", at,
                              None, "incomplete", 1, 20.0, at))
        database.execute("INSERT INTO vision_descriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         ("/private/three.webp", "9b-sha", vision_fallback.MODEL_PROMPT_VERSION,
                          "image-sha", at, "Done", "complete", 1, 90.0, at))
    rows = vision_fallback.select_incomplete(date(2026, 9, 28), "2b-sha", "9b-sha", 60)
    assert [row[0] for row in rows] == ["/private/one.webp"]
    assert rows[0][5] == 0
    assert rows[0][6] == "image-sha"


def test_second_fallback_attempt_uses_larger_generation_budget():
    assert vision_fallback.generation_budget(0) == (1536, 240)
    assert vision_fallback.generation_budget(1) == (2048, 300)


def test_gated_attempt_is_preserved_in_history_without_loading_model(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(vision_fallback, 'RECEIPT', tmp_path/'latest.json')
    monkeypatch.setattr(vision_fallback, 'HISTORY_DIR', tmp_path/'history')
    monkeypatch.setattr(vision_fallback, 'resource_gate', lambda **_: 'battery_power')
    def unexpected(*args, **kwargs):
        raise AssertionError('Gated attempt must not load models')
    monkeypatch.setattr(vision_fallback, 'verify_models', unexpected)
    result = vision_fallback.run(date(2026,10,4), 20, mode='mixed')
    assert result['stop_reason'] == 'battery_power' and result['completed'] == 0
    history = list((tmp_path/'history').glob('*.json'))
    assert len(history) == 1 and json.loads(history[0].read_text()) == result
    assert history[0].stat().st_mode & 0o777 == 0o600


def test_fallback_result_has_distinct_model_provenance_and_bounded_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    at = datetime(2026, 9, 28, 20, tzinfo=timezone.utc).isoformat()
    secure_store.prepare_private_dir()
    vision_fallback.save_description("/private/one.webp", at, "image-sha", "9b-sha",
                                     {"status": "incomplete", "description": None, "elapsed_seconds": 90.0})
    vision_fallback.save_description("/private/one.webp", at, "image-sha", "9b-sha",
                                     {"status": "complete", "description": "Visible task.", "elapsed_seconds": 80.0})
    with secure_store.connect() as database:
        assert database.execute(
            "SELECT model_sha256,prompt_version,status,attempts FROM vision_descriptions"
        ).fetchone() == ("9b-sha", vision_fallback.MODEL_PROMPT_VERSION, "complete", 2)


def test_only_private_secure_or_preserved_legacy_images_are_allowed(tmp_path, monkeypatch):
    secure = tmp_path / "screenshots"
    legacy = tmp_path / "pensieve/screenshots"
    secure.mkdir(parents=True)
    legacy.mkdir(parents=True)
    first = secure / "one.webp"
    second = legacy / "two.webp"
    outside = tmp_path / "outside.webp"
    for path in (first, second, outside):
        path.write_bytes(b"synthetic")
    link = secure / "link.webp"
    link.symlink_to(outside)
    monkeypatch.setattr(vision_fallback, "ALLOWED_SCREENSHOT_ROOTS", (secure, legacy))
    assert vision_fallback.is_private_screenshot(first)
    assert vision_fallback.is_private_screenshot(second)
    assert not vision_fallback.is_private_screenshot(outside)
    assert not vision_fallback.is_private_screenshot(link)


def test_quality_sample_uses_completed_2b_across_app_contexts(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    with secure_store.connect() as database:
        for index, app in enumerate(("Editor", "Editor", "Browser", "Browser", "Mail")):
            at = datetime(2026, 9, 28, 15 + index, tzinfo=timezone.utc).isoformat()
            path = f"/private/{index}.webp"
            database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                             (path, at, app, f"Window {index}", None, "ordinary", "complete", at))
            database.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (path, "2b", vision_fallback.PROMPT_VERSION, f"hash{index}", at,
                              "Context", "complete", 1, 12, at))
        at = datetime(2026, 9, 28, 17, tzinfo=timezone.utc).isoformat()
        database.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("/private/2.webp", "9b", vision_fallback.MODEL_PROMPT_VERSION,
                          "hash2", at, "Better context", "complete", 1, 40, at))
    rows = vision_fallback.select_quality(date(2026, 9, 28), "2b", "9b", 3)
    assert len(rows) == 3
    assert len({row[2] for row in rows}) >= 2
    assert all(row[0] != "/private/2.webp" for row in rows)
    assert all(row[6].startswith("hash") for row in rows)


def test_direct_9b_uses_uncovered_frames_and_retries_once(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    at = datetime(2026, 9, 28, 20, tzinfo=timezone.utc)
    with secure_store.connect() as database:
        for index in range(5):
            timestamp = at.replace(minute=index).isoformat()
            path = f"/private/direct-{index}.webp"
            database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                             (path, timestamp, "Editor", "Draft", None, "ordinary",
                              "complete", timestamp))
        for index, model, status, attempts in (
                (1, "2b", "complete", 1),
                (2, "9b", "complete", 1),
                (3, "9b", "incomplete", 1),
                (4, "9b", "incomplete", 2)):
            timestamp = at.replace(minute=index).isoformat()
            database.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (f"/private/direct-{index}.webp", model,
                              vision_fallback.MODEL_PROMPT_VERSION if model == "9b" else vision_fallback.PROMPT_VERSION,
                              "hash", timestamp, None, status, attempts, 1.0, timestamp))
    rows = vision_fallback.select_direct(date(2026, 9, 28), "2b", "9b", 20)
    assert {row[0]: row[5] for row in rows} == {
        "/private/direct-0.webp": 0, "/private/direct-3.webp": 1}
    assert all(row[6] is None for row in rows)


def test_9b_balances_new_coverage_with_quality_upgrades():
    rows = vision_fallback.interleave_new_and_upgrades([("new1",), ("new2",)],
                                                        [("old1",), ("old2",)], 3)
    assert rows == [("direct", ("new1",)), ("quality", ("old1",)),
                    ("direct", ("new2",))]


def test_primary_priority_spreads_partial_night_across_day():
    ordered = vision_fallback.spread_priority([(i,) for i in range(16)], 16)
    assert sorted(item[0] for item in ordered) == list(range(16))
    assert {item[0] // 4 for item in ordered[:7]} == {0, 1, 2, 3}


def test_partial_night_prioritizes_distinct_contexts_before_repeats():
    rows = [(f"frame-{i}", datetime(2026, 9, 28, 20, i, tzinfo=timezone.utc).isoformat(),
             "Editor", "Long task" if i < 18 else f"Other task {i}")
            for i in range(20)]
    chosen = vision_fallback.context_priority(rows, 20)
    assert len({row[3] for row in chosen[:3]}) == 3
    assert len({row[0] for row in chosen}) == 20


def test_backfill_has_bounded_share_and_retains_yesterday_priority():
    current = [("direct", (i,)) for i in range(10)]
    backlog = [(f"past-{i}",) for i in range(10)]
    chosen = vision_fallback.with_backfill(current, backlog, 10)
    assert [lane for lane, _ in chosen] == ["direct"] * 4 + ["backfill"] + ["direct"] * 4 + ["backfill"]


def test_backfill_covers_old_unseen_days_and_ignores_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    with secure_store.connect() as database:
        for index, day in enumerate((24, 25, 27)):
            at = datetime(2026, 9, day, 20, tzinfo=timezone.utc).isoformat()
            database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                             (f"/private/backfill-{index}.webp", at, "Editor", "Draft", None,
                              "ordinary", "complete", at))
        database.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("/private/backfill-2.webp", "9b", vision_fallback.MODEL_PROMPT_VERSION,
                          "hash", at, "Visible work", "complete", 1, 80, at))
    rows = vision_fallback.select_backfill(date(2026, 9, 28), "2b", "9b", 10)
    assert {row[0] for row in rows} == {"/private/backfill-0.webp", "/private/backfill-1.webp"}


def test_prior_9b_failures_get_one_recent_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    with secure_store.connect() as database:
        for index, day, status, attempts in (
                (0, 27, "incomplete", 1),
                (1, 26, "incomplete", 2),
                (2, 25, "complete", 1),
                (3, 20, "incomplete", 1)):
            at = datetime(2026, 9, day, 20, tzinfo=timezone.utc).isoformat()
            path = f"/private/prior-{index}.webp"
            database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                             (path, at, "Editor", "Draft", None, "ordinary", "complete", at))
            database.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (path, "9b", vision_fallback.MODEL_PROMPT_VERSION,
                              "hash", at, None, status, attempts, 90.0, at))
    rows = vision_fallback.select_prior_retries(date(2026, 9, 28), "9b", 15)
    assert [row[0] for row in rows] == ["/private/prior-0.webp"]
    assert rows[0][5:] == (1, "hash")


def test_primary_9b_describes_direct_frame_and_persists_sensitive_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(vision_fallback, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vision_fallback, "RECEIPT", tmp_path / "primary.json")
    monkeypatch.setattr(vision_fallback, "HISTORY_DIR", tmp_path / "history")
    root = tmp_path / "screenshots"
    root.mkdir()
    monkeypatch.setattr(vision_fallback, "ALLOWED_SCREENSHOT_ROOTS", (root,))
    at = datetime(2026, 9, 28, 20, tzinfo=timezone.utc)
    with secure_store.connect() as database:
        for index, ocr in enumerate(("ordinary", "password")):
            path = root / f"frame-{index}.webp"
            path.write_bytes(b"synthetic private frame")
            timestamp = at.replace(minute=index).isoformat()
            database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                             (str(path), timestamp, "Editor", "Draft", None, ocr,
                              "complete", timestamp))
        prior_path = root / "prior.webp"
        prior_path.write_bytes(b"synthetic private frame")
        prior_at = at.replace(day=27).isoformat()
        database.execute("INSERT INTO screenshots VALUES (?,?,?,?,?,?,?,?)",
                         (str(prior_path), prior_at, "Editor", "Draft", None,
                          "ordinary", "complete", prior_at))
        database.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (str(prior_path), "strong-sha", vision_fallback.MODEL_PROMPT_VERSION,
                          vision_fallback.sha256_file(prior_path), prior_at, None, "incomplete", 1, 90, prior_at))
    monkeypatch.setattr(vision_fallback, "resource_gate", lambda **_: None)
    monkeypatch.setattr(vision_fallback, "in_overnight_window", lambda _: True)
    monkeypatch.setattr(vision_fallback, "seconds_until_window_end", lambda _: 10000)
    monkeypatch.setattr(vision_fallback, "verify_models", lambda _: {
        "9b_q8": {"weights_sha256": "strong-sha"}})
    seen = []
    monkeypatch.setattr(vision_fallback, "run_image", lambda path, *_args, **_kwargs:
                        (seen.append(path) or {"status": "complete", "description": "Visible work.",
                                               "elapsed_seconds": 1.0}))
    result = vision_fallback.run(date(2026, 9, 28), 10, mode="mixed", hard_limit=0)
    assert result["direct_selected"] == 2
    assert result["direct_completed"] == 1
    assert result["sensitive_skipped"] == 1
    assert result["retry_completed"] == 1
    assert result["backfill_selected"] == 0
    assert result["selected"] == 3
    assert len(seen) == 2
    with secure_store.connect() as database:
        assert dict(database.execute("SELECT path,status FROM vision_descriptions")) == {
            str(root / "frame-0.webp"): "complete",
            str(root / "frame-1.webp"): "sensitive_skipped",
            str(prior_path): "complete"}
