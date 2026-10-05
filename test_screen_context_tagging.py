from datetime import date

import json

import screen_context_tagging
from screen_context_tagging import allocate_visible_context, make_report, validate_batch


def test_model_topic_needs_an_exact_supporting_word_and_topic_token():
    rows = [{"id": "one", "ocr": "SEO dashboard ranking report", "strong_caption": ""},
            {"id": "two", "ocr": "Calendar and weather", "strong_caption": ""}]
    result = validate_batch({"tags": [
        {"id": "one", "topic": "SEO dashboard", "evidence_word": "ranking"},
        {"id": "two", "topic": "SEO dashboard", "evidence_word": "weather"},
    ]}, rows)
    assert result[0]["status"] == "model_inference"
    assert result[1]["status"] == "unclassified"


def test_screen_coverage_is_bounded_and_never_relabels_whole_unknown_segment():
    segments = [{"state": "unattributed", "start_utc": "2026-09-29T12:00:00+00:00",
                 "end_utc": "2026-09-29T13:00:00+00:00", "sampled_seconds": 3600}]
    tags = [{"timestamp_utc": "2026-09-29T12:10:00+00:00", "topic": "SEO dashboard",
             "status": "model_inference"},
            {"timestamp_utc": "2026-09-29T12:10:10+00:00", "topic": "SEO dashboard",
             "status": "model_inference"},
            {"timestamp_utc": "2026-09-29T12:20:00+00:00", "topic": "Social feed",
             "status": "unclassified"}]
    rows = allocate_visible_context(segments, tags)
    assert rows[0]["label"] == "SEO dashboard"
    assert 50 <= rows[0]["sampled_seconds"] <= 70
    assert sum(item["sampled_seconds"] for item in rows) < 100


def test_private_report_keeps_model_labels_but_not_ocr_or_caption_input():
    report = make_report(date(2026, 9, 29),
                         [{"id": "one", "ocr": "private source text",
                           "strong_caption": "private description"}],
                         [{"id": "one", "topic": "Project planning",
                           "status": "model_inference", "timestamp_utc": "2026-09-29T12:00:00+00:00"}],
                         0, "model", "complete")
    assert "ocr" not in str(report)
    assert "private source text" not in str(report)
    assert "private description" not in str(report)


def test_selection_change_retains_prior_supported_frame(monkeypatch):
    day = date(2026, 9, 29)
    previous = {"day_local": day.isoformat(), "version": "screen_context_v1",
                "model_sha256": "model", "tags": [
                    {"id": "old", "topic": "Project planning", "status": "model_inference",
                     "timestamp_utc": "2026-09-29T12:00:00+00:00", "input_sha256": "oldhash"}]}
    selected = [{"id": "new", "timestamp_utc": "2026-09-29T12:02:00+00:00",
                 "ocr": "SEO planning dashboard", "strong_caption": ""}]
    saved = []
    monkeypatch.setattr(screen_context_tagging, "source_rows", lambda _: (selected, 0))
    monkeypatch.setattr(screen_context_tagging, "read_json_file", lambda _: previous)
    monkeypatch.setattr(screen_context_tagging, "private_write", lambda path, body: saved.append(json.loads(body)))
    monkeypatch.setattr(screen_context_tagging, "resource_gate", lambda benchmark_now: None)
    monkeypatch.setattr(screen_context_tagging, "vision_busy", lambda: False)
    monkeypatch.setattr(screen_context_tagging, "model_call", lambda model, prompt, schema: (
        {"tags": [{"id": "new", "topic": "SEO planning", "evidence_word": "planning"}]}, 1))
    result = screen_context_tagging.run_day(day, None, "model")
    assert result["status"] == "complete" and result["tagged"] == 1
    assert {item["id"] for item in saved[-1]["tags"]} == {"old", "new"}
    assert saved[-1]["selected_processed"] == 1


def test_missing_model_ids_are_retryable_and_schema_only_allows_real_ids():
    rows = [{"id": "one", "ocr": "Planning report", "strong_caption": ""}]
    assert validate_batch({"tags": []}, rows)[0]["status"] == "model_failed"
    schema = screen_context_tagging.batch_schema(rows)
    assert schema["properties"]["tags"]["items"]["properties"]["id"]["enum"] == ["one"]
    assert "enum" not in screen_context_tagging.SCHEMA["properties"]["tags"]["items"]["properties"]["id"]


def test_screen_suggestion_cannot_cross_to_another_foreground_app():
    segments = [{"state": "unattributed", "app": "Editor", "start_utc": "2026-10-03T12:00:00+00:00",
                 "end_utc": "2026-10-03T12:00:30+00:00", "sampled_seconds": 30}]
    tags = [{"timestamp_utc": "2026-10-03T12:00:10+00:00", "topic": "Social feed",
             "status": "model_inference", "app_hint": "Browser"}]
    assert allocate_visible_context(segments, tags) == []


def test_supported_tags_require_matching_source_and_new_propagation_version(tmp_path, monkeypatch):
    import hashlib
    import sqlite3
    path = str(tmp_path / "frame.webp")
    at = "2026-10-03T12:00:00+00:00"
    dbpath = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,active_app TEXT)")
        db.execute("INSERT INTO screenshots VALUES (?,?,?)", (path, at, "Editor"))
    identity = hashlib.sha256(path.encode()).hexdigest()[:20]
    good = {"id": identity, "timestamp_utc": at, "topic": "Planning", "status": "model_inference"}
    monkeypatch.setattr(screen_context_tagging, "DB_PATH", dbpath)
    def read(path):
        if path.name.startswith("screen-context"):
            return {"tags": [good, {**good, "id": "missing"},
                             {**good, "timestamp_utc": "2026-10-03T12:10:00+00:00"}]}
        return {"version": "screen_similarity_v1", "propagated": [{**good, "status": "featureprint_match"}]}
    monkeypatch.setattr(screen_context_tagging, "read_json_file", read)
    accepted = screen_context_tagging.read_supported_tags(date(2026, 10, 3))
    assert accepted == [{**good, "app_hint": "Editor"}]


def test_old_unclear_cache_is_revisited_with_new_id_constraints(monkeypatch):
    import hashlib
    day = date(2026, 10, 3)
    row = {"id": "one", "timestamp_utc": "2026-10-03T12:00:00+00:00", "ocr": "SEO report", "strong_caption": ""}
    digest = hashlib.sha256(json.dumps({k: row[k] for k in ("ocr", "strong_caption")}, sort_keys=True).encode()).hexdigest()
    previous = {"day_local": day.isoformat(), "version": "screen_context_v1", "model_sha256": "model",
                "tags": [{"id": "one", "topic": "Unclear", "status": "unclassified", "timestamp_utc": row["timestamp_utc"], "input_sha256": digest}]}
    monkeypatch.setattr(screen_context_tagging, "source_rows", lambda _: ([row], 0))
    monkeypatch.setattr(screen_context_tagging, "read_json_file", lambda _: previous)
    monkeypatch.setattr(screen_context_tagging, "private_write", lambda *_: None)
    monkeypatch.setattr(screen_context_tagging, "resource_gate", lambda **_: None)
    monkeypatch.setattr(screen_context_tagging, "vision_busy", lambda: False)
    monkeypatch.setattr(screen_context_tagging, "model_call", lambda *_: (
        {"tags": [{"id": "one", "topic": "SEO report", "evidence_word": "report"}]}, 1))
    result = screen_context_tagging.run_day(day, None, "model")
    assert result["processed_this_run"] == 1
    assert result["specific"] == 1
