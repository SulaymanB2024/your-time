import json
import time
from datetime import date

import pytest

import local_synthesis


def test_json_parser_and_evidence_validation_reject_untrusted_claims():
    schema = local_synthesis.block_schema("mac-real")
    assert schema["properties"]["evidence_ids"]["items"]["enum"] == ["mac-real"]
    assert schema["properties"]["evidence_ids"]["maxItems"] == 1
    assert "enum" not in local_synthesis.BLOCK_SCHEMA["properties"]["evidence_ids"]["items"]
    day_schema = local_synthesis.day_schema({"mac-real", "mac-other"})
    for key in ("themes", "candidate_outcomes"):
        assert day_schema["properties"][key]["items"]["properties"]["evidence_ids"]["items"]["enum"] == ["mac-other", "mac-real"]
    parsed = local_synthesis.parse_model_json(b'{"focus_label":"Report drafting"}\n> EOF by user')
    assert parsed["focus_label"] == "Report drafting"
    valid = {"focus_label": "Report drafting", "activity_kind": "writing",
             "observed_work": "A draft is visible", "candidate_progress": "",
             "evidence_ids": ["mac-block-0001"], "uncertainty": "Only one screenshot"}
    assert local_synthesis.validate_block(valid, "mac-block-0001")["activity_kind"] == "writing"
    invalid = dict(valid, candidate_progress="Submitted the report")
    try:
        local_synthesis.validate_block(invalid, "mac-block-0001")
        assert False, "completed action must be rejected"
    except ValueError:
        pass
    try:
        local_synthesis.validate_day({"themes": [{"label": "Writing", "summary": "Draft visible",
                                                    "evidence_ids": ["invented"]}],
                                      "candidate_outcomes": [], "uncertainty": ""},
                                     {"mac-block-0001"})
        assert False, "invented citation must be rejected"
    except ValueError:
        pass
    day = local_synthesis.validate_day({"themes": [],
        "candidate_outcomes": [{"description": "A draft may be in progress",
                                "evidence_ids": ["invented"]}], "uncertainty": ""},
        {"mac-block-0001"})
    assert day["candidate_outcomes"] == []
    assert day["rejected_candidate_count"] == 1


def test_private_synthesis_caches_stable_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    block = {"id": "mac-block-0001", "start_utc": "2026-09-28T15:00:00+00:00",
             "end_utc": "2026-09-28T15:10:00+00:00", "sampled_seconds": 600,
             "app": "Editor", "top_windows": [{"title": "Draft"}],
             "visual_evidence": [], "screenshot_count": 0}
    focus = {"blocks": [block]}
    data = (json.dumps(focus) + "\n").encode()
    (tmp_path / "focus-2026-09-28.json").write_bytes(data)
    (tmp_path / "focus-2026-09-28.manifest.json").write_text(json.dumps({
        "sha256": __import__("hashlib").sha256(data).hexdigest()}))
    calls = []
    def fake_call(model, prompt, schema):
        calls.append(schema)
        if "focus_label" in schema["properties"]:
            return ({"focus_label": "Report drafting", "activity_kind": "writing",
                     "observed_work": "A draft is visible", "candidate_progress": "",
                     "evidence_ids": ["mac-block-0001"], "uncertainty": "Short sample"}, 1.0)
        return ({"themes": [{"label": "Writing", "summary": "A draft was visible",
                             "evidence_ids": ["mac-block-0001"]}],
                 "candidate_outcomes": [], "uncertainty": "One block"}, 1.0)
    monkeypatch.setattr(local_synthesis, "model_call", fake_call)
    first = local_synthesis.run_day(date(2026, 9, 28), tmp_path / "fake.gguf", "sha",
                                    deadline=time.monotonic() + 1000, allow_battery=True)
    second = local_synthesis.run_day(date(2026, 9, 28), tmp_path / "fake.gguf", "sha",
                                     deadline=time.monotonic() + 1000, allow_battery=True)
    assert first["status"] == second["status"] == "complete"
    assert len(calls) == 2
    saved = json.loads((tmp_path / "synthesis-2026-09-28.json").read_text())
    assert saved["verified_accomplishments"] == []
    assert saved["coverage"]["reused"] == 1
    assert (tmp_path / "synthesis-2026-09-28.json").stat().st_mode & 0o777 == 0o600
    assert not local_synthesis.needs_synthesis(date(2026, 9, 28), "sha")


def test_model_prompt_is_private_file_not_process_argument(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "STATE_DIR", tmp_path)
    observed = {}
    def fake_run(command, **kwargs):
        observed["command"] = command
        path = command[command.index("-f", 4) + 1]
        observed["prompt_mode"] = path and ( __import__("os").stat(path).st_mode & 0o777)
        observed["prompt_text"] = __import__("pathlib").Path(path).read_text()
        return __import__("types").SimpleNamespace(returncode=0, stdout=b'{"ok":true}')
    monkeypatch.setattr(local_synthesis, "run_model", fake_run)
    result, _ = local_synthesis.model_call(tmp_path / "fake.gguf", "private window title",
                                           {"type": "object"})
    assert result == {"ok": True}
    assert "private window title" not in observed["command"]
    assert observed["prompt_mode"] == 0o600
    assert observed["prompt_text"] == "private window title"
    assert not __import__("pathlib").Path(observed["command"][observed["command"].index("-f", 4) + 1]).exists()


@pytest.mark.parametrize("variant,version", [("window_topics", "window_topics_v3"),
                                            ("screen_context", "screen_context_v2")])
def test_shared_text_call_records_caller_identity_without_changing_decoding(tmp_path, monkeypatch, variant, version):
    from types import SimpleNamespace

    from inference_telemetry import Attempt

    monkeypatch.setattr(local_synthesis, "STATE_DIR", tmp_path)
    identities = []
    def synthetic_run(command, **kwargs):
        attempt = Attempt(tmp_path, command, kwargs["telemetry"])
        attempt.finish("complete", returncode=0)
        identities.append(attempt.identity)
        return SimpleNamespace(returncode=0, stdout=b'{"tags":[]}', telemetry_attempt_id=attempt.identity)
    monkeypatch.setattr(local_synthesis, "run_model", synthetic_run)
    local_synthesis.model_call(tmp_path / "fake.gguf", "Synthetic tagging input", {"type": "object"},
                               telemetry_variant=variant, telemetry_prompt_version=version)
    saved = json.loads((tmp_path / "inference-attempts" / (identities[0] + ".json")).read_text())
    assert saved["context"]["variant"] == variant
    assert saved["context"]["prompt_version"] == version
    assert saved["configuration"]["-n"] == 768
    assert "result_status" not in saved  # The caller's validator supplies this.
    assert "Synthetic tagging input" not in json.dumps(saved)


@pytest.mark.parametrize("variant,version", [("private arbitrary label", "v1"),
                                            ("chapter", "private window title")])
def test_unbounded_telemetry_identity_refuses_before_model_execution(tmp_path, monkeypatch, variant, version):
    monkeypatch.setattr(local_synthesis, "run_model", lambda *args, **kwargs: pytest.fail("model must not run"))
    with pytest.raises(ValueError, match="Invalid text telemetry identity"):
        local_synthesis.model_call(tmp_path / "fake.gguf", "Synthetic input", {},
                                   telemetry_variant=variant, telemetry_prompt_version=version)


def test_short_legacy_chapter_is_retained_without_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    blocks = [{"id": "short", "start_utc": "2026-09-28T14:00:00+00:00",
               "end_utc": "2026-09-28T14:00:20+00:00", "sampled_seconds": 20,
               "app": "Browser", "top_windows": [], "visual_evidence": [], "screenshot_count": 0},
              {"id": "long", "start_utc": "2026-09-28T15:00:00+00:00",
               "end_utc": "2026-09-28T15:02:00+00:00", "sampled_seconds": 120,
               "app": "Editor", "top_windows": [], "visual_evidence": [], "screenshot_count": 0}]
    payload=(json.dumps({"blocks": blocks})+"\n").encode()
    (tmp_path/"focus-2026-09-28.json").write_bytes(payload)
    (tmp_path/"focus-2026-09-28.manifest.json").write_text(json.dumps({
        "sha256": __import__("hashlib").sha256(payload).hexdigest()}))
    calls=[]
    def fake_call(model,prompt,schema):
        calls.append(schema)
        if "focus_label" in schema["properties"]:
            return ({"focus_label":"Draft","activity_kind":"writing",
                     "observed_work":"Draft visible","candidate_progress":"",
                     "evidence_ids":["long"],"uncertainty":""},1)
        return ({"themes":[],"candidate_outcomes":[],"uncertainty":""},1)
    monkeypatch.setattr(local_synthesis,"model_call",fake_call)
    result=local_synthesis.run_day(date(2026,9,28),tmp_path/"fake.gguf","sha",
                                   deadline=time.monotonic()+1000,allow_battery=True)
    assert result["status"]=="complete"
    assert result["insufficient_context_blocks"]==1
    assert len(calls)==2


def test_bounded_backfill_keeps_later_cached_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    blocks=[]
    for identity,hour in [("new",14),("cached",15)]:
        blocks.append({"id":identity,"start_utc":f"2026-09-28T{hour}:00:00+00:00",
                       "end_utc":f"2026-09-28T{hour}:02:00+00:00",
                       "sampled_seconds":120,"app":"Editor","top_windows":[],
                       "visual_evidence":[],"screenshot_count":0})
    payload=(json.dumps({"blocks":blocks})+"\n").encode()
    (tmp_path/"focus-2026-09-28.json").write_bytes(payload)
    (tmp_path/"focus-2026-09-28.manifest.json").write_text(json.dumps({
        "sha256":__import__("hashlib").sha256(payload).hexdigest()}))
    cached={"block_id":"cached","sampled_seconds":120,
            "input_sha256":local_synthesis.fingerprint(local_synthesis.block_projection(blocks[1])),
            "model_sha256":"sha","prompt_version":local_synthesis.PROMPT_VERSION,
            "status":"complete","result":{"focus_label":"Draft","activity_kind":"writing",
            "observed_work":"Draft visible","candidate_progress":"",
            "evidence_ids":["cached"],"uncertainty":""},"elapsed_seconds":1}
    (tmp_path/"synthesis-2026-09-28.json").write_text(json.dumps({"blocks":[cached]}))
    def fake_call(model,prompt,schema):
        if "focus_label" in schema["properties"]:
            return ({**cached["result"],"evidence_ids":["new"]},1)
        return ({"themes":[],"candidate_outcomes":[],"uncertainty":""},1)
    monkeypatch.setattr(local_synthesis,"model_call",fake_call)
    result=local_synthesis.run_day(date(2026,9,28),tmp_path/"fake.gguf","sha",
                                   deadline=time.monotonic()+1000,allow_battery=True,limit=1)
    saved=json.loads((tmp_path/"synthesis-2026-09-28.json").read_text())
    assert result["status"]=="complete"
    assert {row["block_id"] for row in saved["blocks"]}=={"new","cached"}
    assert result["reused"]==1


def test_block_projection_uses_model_identity_for_9b_reliability(monkeypatch):
    import daily_focus
    monkeypatch.setattr(daily_focus, "stronger_model_sha", lambda: "pinned9b")
    block = {"id": "b", "start_utc": "2026-10-03T12:00:00+00:00",
             "end_utc": "2026-10-03T12:10:00+00:00", "sampled_seconds": 600,
             "visual_evidence": [{"model_inference": {
                 "text": "A draft", "model_sha256": "pinned9b", "prompt_version": "primary_9b_v1"}},
                 {"model_inference": {"text": "A page", "model_sha256": "2b", "prompt_version": "fallback_v1"}}]}
    captions = local_synthesis.block_projection(block)["captions"]
    assert captions[0]["reliability"] == "stronger_9b"
    assert captions[1]["reliability"] == "weaker_2b"


def test_resource_stop_preserves_later_cached_results(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: "system_load_high")
    blocks = [{"id": identity, "start_utc": f"2026-10-03T{hour}:00:00+00:00",
               "end_utc": f"2026-10-03T{hour}:02:00+00:00", "sampled_seconds": 120,
               "app": "Editor", "top_windows": [], "visual_evidence": [], "screenshot_count": 0}
              for identity, hour in [("pending", 12), ("cached", 13)]]
    payload = json.dumps({"blocks": blocks}).encode()
    (tmp_path / "focus-2026-10-03.json").write_bytes(payload)
    (tmp_path / "focus-2026-10-03.manifest.json").write_text(json.dumps({
        "sha256": __import__("hashlib").sha256(payload).hexdigest()}))
    cached = {"block_id": "cached", "input_sha256": local_synthesis.fingerprint(local_synthesis.block_projection(blocks[1])),
              "model_sha256": "sha", "prompt_version": local_synthesis.PROMPT_VERSION,
              "status": "complete", "result": {}, "sampled_seconds": 120}
    (tmp_path / "synthesis-2026-10-03.json").write_text(json.dumps({"blocks": [cached]}))
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake.gguf", "sha",
        deadline=time.monotonic() + 1000)
    saved = json.loads((tmp_path / "synthesis-2026-10-03.json").read_text())
    assert result["status"] == "partial"
    assert result["attempted_this_run"] == 0
    assert result["reused"] == 1
    assert saved["blocks"] == [cached]
    from model_execution import ModelBusy
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    def busy(*_args, **_kwargs):
        raise ModelBusy("local_model_busy")
    monkeypatch.setattr(local_synthesis, "model_call", busy)
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake.gguf", "sha",
        deadline=time.monotonic() + 1000)
    assert result["stop_reason"] == "local_model_busy"
    assert result["failed_this_run"] == result["attempted_this_run"] == 0
    assert result["reused"] == 1


def write_focus(tmp_path, blocks):
    payload = json.dumps({"blocks": blocks}).encode()
    (tmp_path / "focus-2026-10-03.json").write_bytes(payload)
    (tmp_path / "focus-2026-10-03.manifest.json").write_text(json.dumps({
        "sha256": __import__("hashlib").sha256(payload).hexdigest()}))


def sample_block(identity, hour, seconds=120):
    return {"id": identity, "start_utc": f"2026-10-03T{hour:02d}:00:00+00:00",
            "end_utc": f"2026-10-03T{hour:02d}:10:00+00:00", "sampled_seconds": seconds,
            "app": "Editor", "top_windows": [], "visual_evidence": [], "screenshot_count": 0}


def test_partial_day_gets_a_cited_summary_before_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    blocks = [sample_block("morning", 12, 120), sample_block("afternoon", 18, 600),
              sample_block("evening", 23, 240)]
    write_focus(tmp_path, blocks)
    clock = {"now": 100}
    monkeypatch.setattr(local_synthesis.time, "monotonic", lambda: clock["now"])
    calls = []
    def fake_call(model, prompt, schema):
        clock["now"] += 110
        if "focus_label" in schema["properties"]:
            identity = schema["properties"]["evidence_ids"]["items"]["enum"][0]
            calls.append(identity)
            return ({"focus_label": "Drafting", "activity_kind": "writing",
                     "observed_work": "A draft is visible", "candidate_progress": "",
                     "evidence_ids": [identity], "uncertainty": ""}, 110)
        calls.append("summary")
        ids = schema["properties"]["themes"]["items"]["properties"]["evidence_ids"]["items"]["enum"]
        return ({"themes": [{"label": "Writing", "summary": "Drafting appears in selected chapters",
                             "evidence_ids": ids}], "candidate_outcomes": [], "uncertainty": "Subset"}, 110)
    monkeypatch.setattr(local_synthesis, "model_call", fake_call)
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=700)
    saved = json.loads((tmp_path / "synthesis-2026-10-03.json").read_text())
    assert result["status"] == "partial" and result["summary_status"] == "complete"
    assert calls == ["afternoon", "morning", "summary"]
    assert clock["now"] <= 700 and result["stop_reason"] == "summary_budget_reserved"
    assert result["complete_seconds"] == result["summary_evidence_seconds"] == 720
    assert result["sampled_seconds"] == 960 and saved["verified_accomplishments"] == []
    current = {block["id"]: local_synthesis.fingerprint(local_synthesis.block_projection(block)) for block in blocks}
    assert local_synthesis.summary_is_current(saved, current)
    current["morning"] = "changed"
    assert not local_synthesis.summary_is_current(saved, current)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: "battery_power")
    calls.clear()
    reused = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=1000)
    assert not calls and reused["summary_status"] == "complete" and reused["status"] == "partial"


def test_chapter_selection_covers_hours_and_prioritizes_duration():
    blocks = [sample_block(f"early-{i}", 12, 60 + i) for i in range(20)]
    blocks += [sample_block("midday", 18, 300), sample_block("late", 23, 200)]
    ordered = local_synthesis.prioritize_blocks(blocks)
    assert {block["id"] for block in ordered[:3]} == {"early-19", "midday", "late"}
    assert len({block["id"] for block in ordered}) == len(blocks)


def test_failed_summary_refresh_preserves_previous_valid_partial_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    blocks = [sample_block("morning", 12, 120), sample_block("afternoon", 18, 600),
              sample_block("evening", 23, 240)]
    write_focus(tmp_path, blocks)
    fail_summary = {"enabled": False}
    def fake_call(model, prompt, schema):
        if "focus_label" in schema["properties"]:
            identity = schema["properties"]["evidence_ids"]["items"]["enum"][0]
            return ({"focus_label": "Drafting", "activity_kind": "writing",
                     "observed_work": "Draft visible", "candidate_progress": "",
                     "evidence_ids": [identity], "uncertainty": ""}, 1)
        if fail_summary["enabled"]:
            raise RuntimeError("synthetic summary failure")
        ids = schema["properties"]["themes"]["items"]["properties"]["evidence_ids"]["items"]["enum"]
        return ({"themes": [{"label": "Writing", "summary": "Drafting appears in a selected chapter",
                             "evidence_ids": ids}], "candidate_outcomes": [], "uncertainty": ""}, 1)
    monkeypatch.setattr(local_synthesis, "model_call", fake_call)
    path = tmp_path / "synthesis-2026-10-03.json"
    local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=time.monotonic()+1000, limit=1)
    first = json.loads(path.read_text())
    fail_summary["enabled"] = True
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=time.monotonic()+1000, limit=1)
    second = json.loads(path.read_text())
    assert result["complete_blocks"] == 2 and result["status"] == "partial"
    assert second["themes"] == first["themes"] and second["summary_evidence_ids"] == first["summary_evidence_ids"]
    assert result["summary_evidence_blocks"] == 1 and second["summary_refresh_status"] == "failed"
    current = {block["id"]: local_synthesis.fingerprint(local_synthesis.block_projection(block)) for block in blocks}
    assert local_synthesis.summary_is_current(second, current)


def test_summary_has_bounded_input_without_invalidating_cached_chapters(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    blocks = [sample_block(f"block-{i}", 12 + i % 12, 120 + i) for i in range(45)]
    write_focus(tmp_path, blocks)
    cached = [{"block_id": block["id"], "sampled_seconds": block["sampled_seconds"],
        "input_sha256": local_synthesis.fingerprint(local_synthesis.block_projection(block)),
        "model_sha256": "sha", "prompt_version": local_synthesis.PROMPT_VERSION,
        "status": "complete", "result": {"focus_label": "Draft", "activity_kind": "writing"}}
        for block in blocks]
    (tmp_path / "synthesis-2026-10-03.json").write_text(json.dumps({"blocks": cached}))
    calls = []
    def fake_call(model, prompt, schema):
        assert "focus_label" not in schema["properties"]
        ids = schema["properties"]["themes"]["items"]["properties"]["evidence_ids"]["items"]["enum"]
        calls.append(ids)
        return ({"themes": [], "candidate_outcomes": [], "uncertainty": ""}, 1)
    monkeypatch.setattr(local_synthesis, "model_call", fake_call)
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=time.monotonic() + 1000)
    assert result["status"] == "complete" and result["reused"] == 45
    assert result["summary_evidence_blocks"] == len(calls[0]) == 20 and len(calls) == 1
    assert result["summary_input_blocks"] == 20
    assert result["summary_cited_blocks"] == result["summary_cited_seconds"] == 0
    assert result["summary_evidence_seconds"] < result["complete_seconds"]
    assert not local_synthesis.needs_synthesis(date(2026, 10, 3), "sha")


def test_truncated_summary_records_decode_failure_and_links_the_next_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from inference_telemetry import Attempt

    monkeypatch.setattr(local_synthesis, "STATE_DIR", tmp_path)
    identities, commands = [], []
    def truncated(command, **kwargs):
        commands.append(command)
        attempt = Attempt(tmp_path, command, kwargs["telemetry"])
        attempt.finish("complete", returncode=0)
        identities.append(attempt.identity)
        return SimpleNamespace(returncode=0, stdout=b'{"themes":[', telemetry_attempt_id=attempt.identity)
    monkeypatch.setattr(local_synthesis, "run_model", truncated)
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            local_synthesis.model_call(tmp_path / "fake.gguf", "Synthetic day", local_synthesis.day_schema({"block"}))
    first, second = [json.loads((tmp_path / "inference-attempts" / (identity + ".json")).read_text())
                     for identity in identities]
    assert first["status"] == "complete" and first["result_status"] == "decoding_error"
    assert second["retry_of"] == identities[0] and second["result_status"] == "decoding_error"
    assert first["context"]["variant"] == "day_summary"
    assert first["context"]["prompt_version"] == local_synthesis.DAY_PROMPT_VERSION
    assert commands[0][commands[0].index("-n") + 1] == "1024"


def test_summary_safety_rejection_marks_attempt_without_serializing_its_identity(tmp_path, monkeypatch):
    from inference_telemetry import Attempt

    monkeypatch.setattr(local_synthesis, "STATE_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    attempt = Attempt(tmp_path, [], {"stage": "text", "variant": "day_summary"})
    attempt.finish("complete", returncode=0)
    output = local_synthesis.TextModelOutput({"themes": [{"label": "Writing",
        "summary": "Published the draft", "evidence_ids": ["block"]}],
        "candidate_outcomes": [], "uncertainty": ""}, attempt.identity)
    assert attempt.identity not in json.dumps(output)
    monkeypatch.setattr(local_synthesis, "model_call", lambda *args: (output, 1))
    report = {"summary_status": "not_run"}
    local_synthesis._summarize_batch(report, [{"block_id": "block", "sampled_seconds": 120,
        "result": {"activity_kind": "writing", "focus_label": "Draft"}}], tmp_path / "fake", False)
    saved = json.loads(attempt.path.read_text())
    assert report["summary_status"] == "failed"
    assert saved["result_status"] == report["day_failure_code"] == "unsupported_completion_rejected"
    assert "Published" not in json.dumps(report)
    assert local_synthesis.failure_code(ValueError("SECRET PRIVATE WINDOW")) == "schema_error"


def test_day_output_bounds_are_independent_of_engine_grammar():
    ids = {"a", "b", "c"}
    theme = {"label": "Writing", "summary": "Draft visible", "evidence_ids": ["a"]}
    invalid = [
        {"themes": [theme] * 5, "candidate_outcomes": [], "uncertainty": ""},
        {"themes": [dict(theme, summary="x" * 121)], "candidate_outcomes": [], "uncertainty": ""},
        {"themes": [dict(theme, evidence_ids=["a", "b", "c"])], "candidate_outcomes": [], "uncertainty": ""},
        {"themes": [dict(theme, evidence_ids=["a", "a"])], "candidate_outcomes": [], "uncertainty": ""},
        {"themes": [None], "candidate_outcomes": [], "uncertainty": ""},
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            local_synthesis.validate_day(value, ids)


def test_new_day_version_reuses_chapters_but_refreshes_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    block = sample_block("chapter", 12, 120)
    write_focus(tmp_path, [block])
    cached = {"block_id": block["id"], "sampled_seconds": 120,
        "input_sha256": local_synthesis.fingerprint(local_synthesis.block_projection(block)),
        "model_sha256": "sha", "prompt_version": local_synthesis.PROMPT_VERSION,
        "status": "complete", "result": {"focus_label": "Draft", "activity_kind": "writing"}}
    path = tmp_path / "synthesis-2026-10-03.json"
    path.write_text(json.dumps({"blocks": [cached], "model_sha256": "sha",
        "day_prompt_version": "focus_day_sample_v1", "summary_status": "complete"}))
    calls = []
    def summary_only(model, prompt, schema):
        assert "themes" in schema["properties"]
        calls.append(schema)
        return ({"themes": [{"label": "Writing", "summary": "Draft visible", "evidence_ids": ["chapter"]}],
                 "candidate_outcomes": [], "uncertainty": ""}, 1)
    monkeypatch.setattr(local_synthesis, "model_call", summary_only)
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=time.monotonic()+1000)
    saved = json.loads(path.read_text())
    assert result["reused"] == 1 and result["completed_this_run"] == 0
    assert result["summary_status"] == "complete" and len(calls) == 1
    assert saved["blocks"][0] == cached
    assert saved["day_prompt_version"] == "focus_day_sample_v2"


def test_legacy_valid_summary_keeps_provenance_when_new_generation_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(local_synthesis, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(local_synthesis, "resource_gate", lambda **_: None)
    block = sample_block("chapter", 12, 120)
    write_focus(tmp_path, [block])
    cached = {"block_id": "chapter", "sampled_seconds": 120,
        "input_sha256": local_synthesis.fingerprint(local_synthesis.block_projection(block)),
        "model_sha256": "sha", "prompt_version": local_synthesis.PROMPT_VERSION,
        "status": "complete", "result": {"focus_label": "Draft", "activity_kind": "writing"}}
    legacy = {"blocks": [cached], "model_sha256": "sha", "prompt_version": local_synthesis.PROMPT_VERSION,
        "status": "complete", "day_prompt_version": "focus_day_sample_v1", "summary_status": "complete",
        "summary_evidence_ids": ["chapter"], "day_input_sha256": local_synthesis.summary_fingerprint([cached]),
        "themes": [{"label": "Writing", "summary": "Draft visible", "evidence_ids": ["chapter"]}],
        "candidate_outcomes": [], "uncertainty": "Selected chapters"}
    path = tmp_path / "synthesis-2026-10-03.json"
    path.write_text(json.dumps(legacy))
    current = {"chapter": cached["input_sha256"]}
    assert local_synthesis.summary_is_current(legacy, current)
    assert local_synthesis.needs_synthesis(date(2026, 10, 3), "sha")
    def fail(*args):
        raise ValueError("SECRET PRIVATE FAILURE")
    monkeypatch.setattr(local_synthesis, "model_call", fail)
    result = local_synthesis.run_day(date(2026, 10, 3), tmp_path / "fake", "sha", deadline=time.monotonic()+1000)
    saved = json.loads(path.read_text())
    assert saved["themes"] == legacy["themes"] and saved["blocks"] == legacy["blocks"]
    assert saved["day_prompt_version"] == "focus_day_sample_v1"
    assert saved["summary_status"] == "complete" and saved["summary_refresh_status"] == "failed"
    assert result["status"] == "partial" and result["day_failure_code"] == "schema_error"
    assert result["summary_input_blocks"] == result["summary_cited_blocks"] == 1
    assert result["summary_cited_seconds"] == 120
    assert "SECRET" not in json.dumps(saved)
    assert local_synthesis.summary_is_current(saved, current)
    saved = json.loads((tmp_path / "synthesis-2026-10-03.json").read_text())
    saved["day_input_sha256"] = "corrupt"
    (tmp_path / "synthesis-2026-10-03.json").write_text(json.dumps(saved))
    assert local_synthesis.needs_synthesis(date(2026, 10, 3), "sha")
