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
