import json
import time
from datetime import date

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
    assert result["summary_evidence_seconds"] < result["complete_seconds"]
    assert not local_synthesis.needs_synthesis(date(2026, 10, 3), "sha")
    saved = json.loads((tmp_path / "synthesis-2026-10-03.json").read_text())
    saved["day_input_sha256"] = "corrupt"
    (tmp_path / "synthesis-2026-10-03.json").write_text(json.dumps(saved))
    assert local_synthesis.needs_synthesis(date(2026, 10, 3), "sha")
