import json
import sqlite3

import pytest

from activity_context import (
    CLAIMS,
    RESULT_VERSION,
    pack,
    parse_result,
    raw_safety_checks,
    validate,
)


def result(context):
    identity = context["evidence"][0]["id"]
    answer = dict(version=RESULT_VERSION, activity_kind="coding", project_candidate="Atlas",
                  task_candidate="Editing code", visible_work="An editor displays code.",
                  evidence_ids=[identity], uncertainty="supported")
    answer["claim_evidence"] = {key: [identity] for key in CLAIMS}
    return answer


def test_nonthinking_json_and_closed_thinking_use_the_same_schema():
    context = {"evidence": [{"id": "s1", "source": "screen_context"}]}
    answer = result(context)
    output = json.dumps(answer)
    assert parse_result(output, context) == answer
    assert parse_result("<think>private reasoning</think>" + output, context) == answer
    with pytest.raises(ValueError, match="incomplete_thinking"):
        parse_result("<think>" + output, context)
    with pytest.raises(json.JSONDecodeError):
        parse_result(output + "trailing commentary", context)


def test_citations_must_cover_individual_claims_and_current_observation():
    context = {"evidence": [{"id": "s1", "source": "screen_context"}, {"id": "other", "source": "nearby_observation"}]}
    answer = result(context)
    answer["claim_evidence"]["project_candidate"] = ["other"]
    answer["evidence_ids"].append("other")
    with pytest.raises(ValueError, match="current observation"):
        validate(answer, context)
    answer = result(context)
    answer["project_candidate"] = None
    with pytest.raises(ValueError, match="Null candidate"):
        validate(answer, context)
    answer["claim_evidence"]["project_candidate"] = []
    assert validate(answer, context) == answer


def test_completion_synonyms_are_not_accepted_as_visible_work():
    context = {"evidence": [{"id": "s1", "source": "screen_context"}]}
    answer = result(context)
    answer["visible_work"] = "The project was deployed."
    with pytest.raises(ValueError):
        validate(answer, context)


def test_raw_safety_flags_precede_redaction_and_never_copy_content():
    marker = "private@example.invalid"
    result = raw_safety_checks("<think>reasoning</think>" + marker + " was sent")
    assert result["email"] and result["unsupported_completion"]
    assert marker not in json.dumps(result)
    assert raw_safety_checks("<think>unfinished")["available"] is False


def test_context_rejects_cross_window_prior_day_and_unrelated_file_changes(tmp_path):
    path = tmp_path / "db.sqlite"
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE screenshots(path,timestamp_utc,active_app,active_window,ocr_text,ocr_status);"
                     "CREATE TABLE project_file_events(id,timestamp_utc,project_root,relative_path);"
                     "CREATE TABLE task_corrections(id,start_utc,end_utc,label,evidence_tier);")
    at = "2026-10-04T05:00:30+00:00"
    db.executemany("INSERT INTO screenshots VALUES(?,?,?,?,?,?)", [
        ("current", at, "Editor", "Atlas editor", "code", "complete"),
        ("old", "2026-10-04T04:59:50+00:00", "Editor", "Atlas editor", "prior day", "complete"),
        ("other", "2026-10-04T05:00:10+00:00", "Browser", "Other", "text", "complete")])
    db.execute("INSERT INTO project_file_events VALUES(?,?,?,?)", ("f", at, "/projects/Beacon", "main.py"))
    db.execute("INSERT INTO task_corrections VALUES(?,?,?,?,?)", ("c", "2026-10-04T04:00:00+00:00", "2026-10-04T06:00:00+00:00", "Unrelated", "user_confirmed_label"))
    db.commit()
    db.close()
    context = pack("current", db_path=path, include_corrections=True)
    assert len(context["evidence"]) == 1
    assert context["evidence"][0]["metadata_reliability"].startswith("recorded_hint")
