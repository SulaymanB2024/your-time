import pytest

import window_topic_tagging


def test_window_topic_requires_specific_source_support():
    row = {"id": "w1", "title": "Analytics Engineer posting", "strong_captions": [], "seconds": 120}
    assert not window_topic_tagging.supported_topic("Data Brokerage Research", "Analytics", row)
    assert window_topic_tagging.supported_topic("Analytics Engineer role", "Analytics", row)
    result = window_topic_tagging.validate_batch({"tags": [{"id": "w1",
        "topic": "Data Brokerage Research", "evidence_word": "Analytics"}]}, [row])
    assert result[0]["topic"] == "Unclear"


def test_window_batch_never_adds_invented_ids():
    row = {"id": "w1", "title": "Chess tutorial video", "strong_captions": [], "seconds": 90}
    value = {"tags": [{"id": "invented", "topic": "Chess", "evidence_word": "Chess"},
                      {"id": "w1", "topic": "Chess tutorial", "evidence_word": "Chess"}]}
    result = window_topic_tagging.validate_batch(value, [row])
    assert len(result) == 1
    assert result[0]["id"] == "w1"
    assert result[0]["topic"] == "Chess tutorial"
    assert not window_topic_tagging.topic_is_specific("YouTube Browsing")
    assert not window_topic_tagging.topic_is_specific("ChatGPT")
    assert window_topic_tagging.topic_is_specific("Minecraft tutorial")


def test_missing_model_id_is_retryable_instead_of_cached_as_unclear():
    rows = [{"id": "w1", "title": "Chess tutorial", "strong_captions": []}]
    assert window_topic_tagging.validate_batch({"tags": []}, rows)[0]["status"] == "model_failed"
    schema = window_topic_tagging.batch_schema(rows)
    assert schema["properties"]["tags"]["items"]["properties"]["id"]["enum"] == ["w1"]


def test_window_worker_shares_one_time_budget_across_days(tmp_path, monkeypatch):
    import sys
    clock = [0.0]
    calls = []
    monkeypatch.setattr(sys, "argv", ["window_topic_tagging.py"])
    monkeypatch.setattr(window_topic_tagging, "STATE_DIR", tmp_path)
    monkeypatch.setattr(window_topic_tagging, "LOCK", tmp_path / "text.lock")
    monkeypatch.setattr(window_topic_tagging, "resource_gate", lambda **_: None)
    monkeypatch.setattr(window_topic_tagging, "vision_busy", lambda: False)
    monkeypatch.setattr(window_topic_tagging, "verify_model", lambda: (tmp_path / "fake.gguf", "sha"))
    monkeypatch.setattr(window_topic_tagging.time, "monotonic", lambda: clock[0])
    def run(*_args, **kwargs):
        calls.append(kwargs["max_seconds"])
        clock[0] += 1300
        return {"stop_reason": None if len(calls) == 1 else "run_time_limit"}
    monkeypatch.setattr(window_topic_tagging, "run_day", run)
    window_topic_tagging.main()
    assert calls == [1500, 200]


def test_short_supported_subjects_are_not_forced_to_unclear():
    row = {"id": "seo", "title": "SEO campaign overview", "strong_captions": []}
    assert window_topic_tagging.supported_topic("SEO", "SEO", row)
    assert window_topic_tagging.topic_is_specific("SEO")
    assert not window_topic_tagging.supported_topic("SQL", "SEO", row)


@pytest.mark.parametrize("module_name", ["window_topic_tagging", "screen_context_tagging"])
@pytest.mark.parametrize("case,expected", [("supported", "complete"),
                                          ("abstention", "complete"),
                                          ("unsupported", "filtered_batch"),
                                          ("sensitive", "filtered_batch"),
                                          ("malformed_extra", "filtered_batch"),
                                          ("missing_id", "partial_batch"),
                                          ("invalid_batch", "schema_error")])
def test_tagging_disposition_follows_validator_and_keeps_unknown_ids_retryable(monkeypatch, module_name, case, expected):
    import importlib
    from datetime import date

    from local_synthesis import TextModelOutput

    module = importlib.import_module(module_name)
    row = {"id": "one", "title": "Chess tutorial", "strong_captions": [], "seconds": 120,
           "timestamp_utc": "2026-10-03T12:00:00+00:00", "ocr": "Chess tutorial", "strong_caption": ""}
    values = {"supported": {"tags": [{"id": "one", "topic": "Chess tutorial", "evidence_word": "Chess"}]},
              "abstention": {"tags": [{"id": "one", "topic": "Unclear", "evidence_word": ""}]},
              "unsupported": {"tags": [{"id": "one", "topic": "Finance report", "evidence_word": "Chess"}]},
              "sensitive": {"tags": [{"id": "one", "topic": "alice@example.com", "evidence_word": "Chess"}]},
              "malformed_extra": {"tags": [{"id": "one", "topic": "Chess tutorial", "evidence_word": "Chess"},
                                             {"private_invalid_field": "discarded"}]},
              "missing_id": {"tags": []}, "invalid_batch": {"invalid": []}}
    output = TextModelOutput(values[case], "a" * 32)
    observed = []
    monkeypatch.setattr(module, "source_rows", lambda _: ([row], 120 if module_name == "window_topic_tagging" else 0))
    monkeypatch.setattr(module, "read_json_file", lambda _: {})
    monkeypatch.setattr(module, "resource_gate", lambda **kwargs: None)
    monkeypatch.setattr(module, "vision_busy", lambda: False)
    monkeypatch.setattr(module, "private_write", lambda *args: None)
    def synthetic_call(*args, **kwargs):
        assert kwargs == {"telemetry_variant": "window_topics" if module_name == "window_topic_tagging" else "screen_context",
                          "telemetry_prompt_version": module.VERSION}
        return output, 1
    monkeypatch.setattr(module, "model_call", synthetic_call)
    monkeypatch.setattr(module, "mark_output", lambda value, status: observed.append((value.telemetry_attempt_id, status)))
    result = module.run_day(date(2026, 10, 3), None, "synthetic-model")
    assert observed == [("a" * 32, expected)]
    assert result["status"] == ("partial" if expected in {"partial_batch", "schema_error"} else "complete")
