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
    monkeypatch.setattr(local_synthesis.subprocess, "run", fake_run)
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
