import json
from datetime import date

import workstream_tagging


def test_grouping_keeps_each_block_once_and_unknown_explicit():
    rows = [{"id": "mac-block-0001", "minutes": 5, "sampled_seconds": 302,
             "window_context": ["Report draft"], "strong_captions": []},
            {"id": "mac-block-0002", "minutes": 8, "sampled_seconds": 481,
             "window_context": ["Editing report"], "strong_captions": []},
            {"id": "mac-block-0003", "minutes": 2, "sampled_seconds": 121,
             "window_context": ["Chess video"], "strong_captions": []}]
    value = {"workstreams": [
        {"label": "Report writing", "block_ids": ["mac-block-0001", "mac-block-0002"]},
        {"label": "Duplicate", "block_ids": ["mac-block-0001"]},
        {"label": "Invented", "block_ids": ["absent"]}], "uncertainty": "Some context unclear"}
    groups, rejected = workstream_tagging.validated_groups(value, rows)
    assert rejected == 2
    assert {x["label"] for x in groups} == {"Report writing", "Unclear"}
    assert sum(x["sampled_seconds"] for x in groups) == 904
    assert sorted(identity for group in groups for identity in group["block_ids"]) == [
        "mac-block-0001", "mac-block-0002", "mac-block-0003"]
    assert not workstream_tagging.label_supported("Chess and strategy", rows[0])


def test_tagging_cache_uses_model_and_input_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(workstream_tagging, "ANALYSIS_DIR", tmp_path)
    focus = {"blocks": [{"id": "mac-block-0001", "sampled_seconds": 300,
                          "top_windows": [{"title": "Draft"}]}]}
    synthesis = {"status": "complete", "blocks": [{"block_id": "mac-block-0001",
        "status": "complete", "result": {"activity_kind": "writing",
                                           "focus_label": "Report draft"}}]}
    (tmp_path / "focus-2026-09-28.json").write_text(json.dumps(focus))
    (tmp_path / "synthesis-2026-09-28.json").write_text(json.dumps(synthesis))
    day = date(2026, 9, 28)
    assert workstream_tagging.needs_tagging(day, "sha")
    rows = workstream_tagging.input_rows(focus, synthesis)
    import hashlib
    input_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    (tmp_path / "workstreams-2026-09-28.json").write_text(json.dumps({
        "status": "complete", "input_sha256": input_sha,
        "model_sha256": "sha", "version": workstream_tagging.VERSION}))
    assert not workstream_tagging.needs_tagging(day, "sha")
    tags = json.loads((tmp_path / "workstreams-2026-09-28.json").read_text())
    assert workstream_tagging.tags_match_inputs(focus, synthesis, tags)
    assert not workstream_tagging.tags_match_inputs(focus, {**synthesis, "status": "partial"}, tags)
