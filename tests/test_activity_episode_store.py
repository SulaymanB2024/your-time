"""Synthetic episode, freshness and attribution regressions; no private inputs."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import activity_episode_store as store
from private_io import write_json

BASE = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def at(seconds):
    return (BASE + timedelta(seconds=seconds)).isoformat()


def sample(start=0, end=120, *, identity="sample-a", state="active", **kwargs):
    return {"id": identity, "start_utc": at(start), "end_utc": at(end),
            "sampled_seconds": end - start, "state": state,
            "app": "Synthetic editor", "window": "Shared title", **kwargs}


def record(episode, *, timestamp=30, identity="screen-a", task="Draft outline", project="Project Alpha",
           project_bindings=(), task_bindings=(), extra_evidence=()):
    context = {"version": "activity_context_v1", "input_sha256": store.fingerprint("context input"),
               "evidence": [{"id": identity, "source": "synthetic_screen", "timestamp_utc": at(timestamp),
                             "ocr": "Synthetic visible work"}, *extra_evidence]}
    result = {"version": "activity_result_v1", "activity_kind": "writing", "project_candidate": project,
              "task_candidate": task, "visible_work": "An outline is visible", "evidence_ids": [identity],
              "claim_evidence": {"activity_kind": [identity], "project_candidate": [identity] if project else [],
                                 "task_candidate": [identity] if task else [], "visible_work": [identity]},
              "uncertainty": "supported"}
    fingerprints = store.make_fingerprints(context, screenshot_sha256=store.fingerprint("image"),
        input_sha256=store.fingerprint("exact request"), engine_sha256=store.fingerprint("engine"),
        prompt_sha256=store.fingerprint("prompt"), model_sha256=store.fingerprint("model"))
    args = {"episode": episode, "observation_id": identity, "timestamp_utc": at(timestamp), "context": context,
            "result": result, "fingerprints": fingerprints,
            "project_bindings": project_bindings, "task_bindings": task_bindings}
    return store.make_inference(**args), args


def evidence(records):
    return {r["observation_id"]: {"timestamp_utc": r["timestamp_utc"],
            **{key: r["fingerprints"][key] for key in store.SOURCE_KEYS}} for r in records}


def runtime(records):
    return {key: records[0]["fingerprints"][key] for key in store.RUNTIME_KEYS}


def promotion():
    return {"enabled": True, "semantic_review_id": "synthetic-review",
            "trial_nights": ["2026-09-28", "2026-09-29", "2026-09-30"],
            "trial_receipt_sha256": store.fingerprint("synthetic three-night receipt")}


def allocate(episodes, records=(), corrections=(), **kwargs):
    return store.allocate(episodes, records, corrections, current_evidence=evidence(records),
                          runtime_fingerprints=runtime(records) if records else {}, **kwargs)


def correction(start=0, end=120, label="User task", identity="correction-a", **kwargs):
    value = {"id": identity, "start_utc": at(start), "end_utc": at(end), "label": label,
             "evidence_tier": "user_confirmed_label", **kwargs}
    value["revision_sha256"] = store.correction_fingerprint(value)
    return value


def test_same_title_task_change_splits_context_and_never_propagates_whole_day():
    episodes = store.build_episodes([sample(0, 300, context_key="visible-task-a"),
                                     sample(300, 600, identity="sample-b", context_key="visible-task-b")])
    assert len(episodes) == 2
    first, _ = record(episodes[0], timestamp=30)
    second, _ = record(episodes[1], timestamp=330, identity="screen-b", task="Review charts")
    result = allocate(episodes, [first, second], reviewed_record_ids=[first["id"], second["id"]], promotion=promotion())
    assert result["totals"]["allocated_foreground_seconds"] == 120
    assert result["totals"]["unknown_foreground_seconds"] == 480
    assert {r["attribution"]["label"] for r in result["intervals"]} == {None, "Draft outline", "Review charts"}


@pytest.mark.parametrize("change", [{"window": "New title"}, {"app": "Other app"}, {"context_key": "new"},
                                    {"evidence": "legacy"}, {"support_barrier": True}, {"state": "idle"}, {"state": "locked"}])
def test_episode_support_changes_are_barriers(change):
    episodes = store.build_episodes([sample(0, 5), sample(5, 10, identity="sample-b", **change),
                                     sample(10, 15, identity="sample-c")])
    assert len(episodes) == 3
    assert sum(ep["sampled_seconds"] for ep in episodes) == 15


def test_explicit_task_boundary_splits_reused_title_without_carrying_label():
    episodes = store.build_episodes([sample()], context_boundaries=[at(60)])
    assert [(ep["start_utc"], ep["end_utc"]) for ep in episodes] == [(at(0), at(60)), (at(60), at(120))]


def test_missing_gap_is_never_allocated_and_sparse_samples_need_exact_runs():
    sparse = sample(0, 30, sampled_seconds=10)
    with pytest.raises(ValueError, match="exact support runs"):
        store.build_episodes([sparse])
    sparse["support_runs"] = [{"start_utc": at(0), "end_utc": at(5)}, {"start_utc": at(25), "end_utc": at(30)}]
    episodes = store.build_episodes([sparse])
    assert len(episodes) == 2
    result = allocate(episodes, corrections=[correction(state="unused")])
    assert result["totals"]["observed_sampled_seconds"] == 10
    assert all(r["end_utc"] <= at(5) or r["start_utc"] >= at(25) for r in result["intervals"])


def test_overlapping_source_seconds_fail_closed():
    with pytest.raises(ValueError, match="Overlapping"):
        store.build_episodes([sample(0, 10), sample(5, 15, identity="sample-b")])


def test_polling_jitter_groups_context_but_never_allocates_missing_time():
    episodes = store.build_episodes([sample(0, 30), sample(31.5, 60, identity="sample-b")], max_gap_seconds=2)
    assert len(episodes) == 1
    value, _ = record(episodes[0], timestamp=29)
    result = allocate(episodes, [value], reviewed_record_ids=[value["id"]], promotion=promotion())
    assert result["totals"]["observed_sampled_seconds"] == 58.5
    assert result["totals"]["allocated_foreground_seconds"] == 57.5  # support ends at t=59
    assert all(r["end_utc"] <= at(30) or r["start_utc"] >= at(31.5) for r in result["intervals"])
    with pytest.raises(ValueError, match="cannot support"):
        record(episodes[0], timestamp=31)


@pytest.mark.parametrize("change", [{"window": "Other title"}, {"context_key": "other"},
                                    {"evidence": "other source"}, {"state": "idle"}, {"support_barrier": True}])
def test_jitter_does_not_bridge_context_or_support_changes(change):
    assert len(store.build_episodes([sample(0, 5), sample(5.5, 10, identity="b", **change)],
                                   max_gap_seconds=2)) == 2


def test_jitter_respects_long_gaps_explicit_boundaries_and_maximum_span():
    samples = [sample(0, 5), sample(6, 12, identity="b")]
    assert len(store.build_episodes(samples, max_gap_seconds=2, context_boundaries=[at(5.5)])) == 2
    assert len(store.build_episodes(samples, max_gap_seconds=2, max_seconds=6)) == 2
    assert len(store.build_episodes([sample(0, 5), sample(7.5, 10, identity="b")], max_gap_seconds=2)) == 2
    for invalid in [True, -1, 2.1, float("nan"), "2"]:
        with pytest.raises(ValueError, match="jitter"):
            store.build_episodes(samples, max_gap_seconds=invalid)


def test_corrected_slot_applies_only_to_unattributed_support_and_exact_ids():
    episodes = store.build_episodes([sample(0, 60, state="unattributed"), sample(60, 120, identity="sample-b")])
    result = allocate(episodes, corrections=[correction()])
    assert result["totals"]["allocated_foreground_seconds"] == 60
    assert result["intervals"][1]["attribution"]["label"] is None
    restricted = allocate(episodes, corrections=[correction(sample_ids=["different-sample"])])
    assert restricted["totals"]["allocated_foreground_seconds"] == 0
    assert allocate(episodes, corrections=[correction(sample_ids=[])])["totals"]["allocated_foreground_seconds"] == 0


def test_background_file_change_cannot_bind_visible_project():
    episodes = store.build_episodes([sample()])
    value, _ = record(episodes[0], project_bindings=[{"id": "project-b", "aliases": ["Project Alpha"],
                "evidence_ids": ["file-background"]}],
                extra_evidence=[{"id": "file-background", "source": "file_change_metadata", "project": "Project Alpha"}])
    assert value["project_identity"] is None
    assert value["task_identity"] is None
    assert value["evidence_tier"] == "model_hypothesis"


def test_supported_aliases_preserve_project_identity_and_separate_task_identity():
    episodes = store.build_episodes([sample()])
    bindings = [{"id": "repo-common-alpha", "aliases": ["Project Alpha", "Alpha"], "evidence_ids": ["screen-a"]}]
    tasks = [{"id": "task-outline-1", "project_id": "repo-common-alpha", "aliases": ["Draft outline"], "evidence_ids": ["screen-a"]},
             {"id": "task-other", "project_id": "repo-common-other", "aliases": ["Draft outline"], "evidence_ids": ["screen-a"]}]
    value, _ = record(episodes[0], project_bindings=bindings, task_bindings=tasks)
    assert value["project_identity"]["id"] == "repo-common-alpha"
    assert value["project_identity"]["aliases"] == ["Alpha", "Project Alpha"]
    assert value["task_identity"]["id"] == "task-outline-1"
    collision, _ = record(episodes[0], project_bindings=[*bindings, {**bindings[0], "id": "different-project"}])
    assert collision["project_identity"] is None


def test_same_display_label_with_conflicting_canonical_identity_abstains():
    episodes = store.build_episodes([sample()])
    records = [record(episodes[0], project_bindings=[{"id": project_id, "aliases": ["Project Alpha"],
                                                    "evidence_ids": ["screen-a"]}])[0]
               for project_id in ("repo-one", "repo-two")]
    result = allocate(episodes, records, reviewed_record_ids=[r["id"] for r in records], promotion=promotion())
    assert result["totals"]["allocated_foreground_seconds"] == 0
    assert any(r["attribution"]["status"] == "model_conflict" for r in result["intervals"])


def test_verified_aliases_of_same_canonical_identity_agree():
    episodes = store.build_episodes([sample()])
    bindings = [{"id": "repo-one", "aliases": ["Project Alpha", "Alpha"], "evidence_ids": ["screen-a"]}]
    tasks = [{"id": "task-one", "project_id": "repo-one", "aliases": ["Draft outline", "Outline"],
              "evidence_ids": ["screen-a"]}]
    records = [record(episodes[0], project=project, task=task,
                      project_bindings=bindings, task_bindings=tasks)[0]
               for project, task in [("Project Alpha", "Draft outline"), ("Alpha", "Outline")]]
    result = allocate(episodes, records, reviewed_record_ids=[r["id"] for r in records], promotion=promotion())
    assert result["totals"]["allocated_foreground_seconds"] == 60
    assert all(r["attribution"].get("project_id") == "repo-one" for r in result["intervals"] if r["attribution"]["label"])


@pytest.mark.parametrize("key", store.SOURCE_KEYS + store.RUNTIME_KEYS)
def test_changed_source_or_runtime_fingerprint_rejects_inference(key):
    episodes = store.build_episodes([sample()])
    value, _ = record(episodes[0])
    sources, engine = evidence([value]), runtime([value])
    assert store.is_fresh(value, episodes=episodes, current_evidence=sources, runtime_fingerprints=engine)
    if key in store.SOURCE_KEYS:
        sources["screen-a"][key] = store.fingerprint("changed OCR, image, or context")
    else:
        engine[key] = store.fingerprint("changed engine, prompt, model or adapter")
    assert not store.is_fresh(value, episodes=episodes, current_evidence=sources, runtime_fingerprints=engine)


def test_changed_episode_or_missing_evidence_rejects_cache():
    episodes = store.build_episodes([sample()])
    value, _ = record(episodes[0])
    assert not store.is_fresh(value, episodes=episodes, current_evidence={}, runtime_fingerprints=runtime([value]))
    changed = store.build_episodes([sample(window="Corrected title")])
    assert not store.is_fresh(value, episodes=changed, current_evidence=evidence([value]), runtime_fingerprints=runtime([value]))


def test_stale_ocr_is_excluded_from_allocation_and_retracted_correction_is_ignored():
    episodes = store.build_episodes([sample(state="unattributed")])
    value, _ = record(episodes[0])
    changed = evidence([value])
    changed["screen-a"]["context_sha256"] = store.fingerprint("new OCR context")
    result = store.allocate(episodes, [value], [{**correction(), "evidence_tier": "retracted"}],
        current_evidence=changed, runtime_fingerprints=runtime([value]),
        promotion=promotion(), reviewed_record_ids=[value["id"]])
    assert result["rejected_inferences"] == 1
    assert result["totals"]["allocated_foreground_seconds"] == 0


def test_partial_hypothesis_is_stored_without_specific_time_credit():
    episodes = store.build_episodes([sample()])
    _, args = record(episodes[0])
    args["result"]["uncertainty"] = "partial"
    value = store.make_inference(**args)
    result = allocate(episodes, [value], promotion=promotion(), reviewed_record_ids=[value["id"]])
    assert result["fresh_inferences"] == 1
    assert result["totals"]["allocated_foreground_seconds"] == 0


def test_unclear_activity_does_not_count_as_allocated_task_time():
    episodes = store.build_episodes([sample()])
    _, args = record(episodes[0], project=None, task=None)
    args["result"]["activity_kind"] = "unclear"
    value = store.make_inference(**args)
    assert allocate(episodes, [value], promotion=promotion(), reviewed_record_ids=[value["id"]])["totals"]["allocated_foreground_seconds"] == 0


def test_persistence_is_immutable_private_and_filters_stale_records(tmp_path):
    episodes = store.build_episodes([sample()])
    expected, args = record(episodes[0])
    actual = store.persist_inference(tmp_path / "episodes", **args)
    assert actual == expected == store.persist_inference(tmp_path / "episodes", **args)
    path = tmp_path / "episodes" / (actual["id"] + ".json")
    assert path.stat().st_mode & 0o077 == 0
    assert "Synthetic visible work" not in path.read_text()  # No raw source OCR is duplicated.
    records = store.read_fresh_inferences(path.parent, episodes=episodes,
        current_evidence=evidence([actual]), runtime_fingerprints=runtime([actual]))
    assert records == [actual]
    assert store.read_fresh_inferences(path.parent, episodes=episodes,
        current_evidence={}, runtime_fingerprints=runtime([actual])) == []


def test_tampering_cannot_promote_outcomes_or_widen_support(tmp_path):
    episodes = store.build_episodes([sample()])
    value, _ = record(episodes[0])
    for key, changed in [("confirmed_outcomes", ["delivery"]), ("scope_end_utc", at(120))]:
        bad = {**value, key: changed}
        bad["id"] = "inference-" + store.fingerprint({k: v for k, v in bad.items() if k != "id"})
        assert not store.is_fresh(bad, episodes=episodes, current_evidence=evidence([value]), runtime_fingerprints=runtime([value]))
    bad = copy.deepcopy(value)
    bad["result"]["task_candidate"] = "Different candidate"
    write_json(tmp_path / (value["id"] + ".json"), bad)
    assert store.read_fresh_inferences(tmp_path, episodes=episodes,
        current_evidence=evidence([value]), runtime_fingerprints=runtime([value])) == []


def test_model_allocation_defaults_off_and_needs_review_plus_three_nights():
    episodes = store.build_episodes([sample()])
    value, _ = record(episodes[0])
    result = allocate(episodes, [value])
    assert result["totals"]["allocated_foreground_seconds"] == 0
    assert not result["model_allocation_enabled"]
    with pytest.raises(ValueError, match="three-night"):
        allocate(episodes, [value], promotion={**promotion(), "trial_nights": ["2026-09-28"]})
    assert allocate(episodes, [value], promotion=promotion())["totals"]["allocated_foreground_seconds"] == 0
    reviewed = allocate(episodes, [value], promotion=promotion(), reviewed_record_ids=[value["id"]])
    assert reviewed["totals"]["allocated_foreground_seconds"] == 60
    assert reviewed["confirmed_outcomes"] == []
    assert reviewed["intervals"][0]["attribution"]["evidence_tier"] == "supported_inference"


def test_overlapping_user_and_model_allocations_count_each_second_once():
    episodes = store.build_episodes([sample(state="unattributed")])
    value, _ = record(episodes[0])
    result = allocate(episodes, [value], [correction(20, 40), correction(20, 40, identity="same-label-copy")],
                      promotion=promotion(), reviewed_record_ids=[value["id"]])
    assert result["totals"] == {"observed_sampled_seconds": 120, "foreground_seconds": 120,
        "allocated_foreground_seconds": 60, "unknown_foreground_seconds": 60, "nonforeground_seconds": 0}
    assert sum(row["observed_seconds"] for row in result["intervals"]) == 120
    assert next(row for row in result["intervals"] if row["start_utc"] == at(20))["attribution"]["label"] == "User task"


def test_conflicting_user_labels_remain_unknown_instead_of_double_counting():
    episodes = store.build_episodes([sample(state="unattributed")])
    result = allocate(episodes, corrections=[correction(), correction(label="Other task", identity="correction-b")])
    assert result["totals"]["unknown_foreground_seconds"] == 120
    assert result["intervals"][0]["attribution"]["status"] == "user_correction_conflict"


def test_changed_correction_scope_or_label_needs_current_revision():
    episodes = store.build_episodes([sample(state="unattributed")])
    stale = correction()
    stale["label"] = "Changed task"
    with pytest.raises(ValueError, match="revision fingerprint"):
        allocate(episodes, corrections=[stale])
    stale["revision_sha256"] = store.correction_fingerprint(stale)
    assert allocate(episodes, corrections=[stale])["totals"]["allocated_foreground_seconds"] == 120


def test_model_overlap_uses_nearest_anchor_and_ties_abstain():
    episodes = store.build_episodes([sample()])
    first, _ = record(episodes[0], timestamp=30)
    second, _ = record(episodes[0], timestamp=60, identity="screen-b", task="Review charts")
    result = allocate(episodes, [first, second], promotion=promotion(), reviewed_record_ids=[first["id"], second["id"]])
    assert result["totals"]["allocated_foreground_seconds"] == 90
    assert any(row["start_utc"] == at(45) for row in result["intervals"])
    assert sum(row["observed_seconds"] for row in result["intervals"]) == 120
    second, _ = record(episodes[0], timestamp=30, identity="screen-b", task="Review charts")
    conflict = allocate(episodes, [first, second], promotion=promotion(), reviewed_record_ids=[first["id"], second["id"]])
    assert conflict["totals"]["allocated_foreground_seconds"] == 0


def test_idle_locked_and_support_barriers_never_gain_task_time():
    episodes = store.build_episodes([sample(0, 30, state="idle"), sample(30, 60, identity="locked", state="locked"),
                                     sample(60, 90, identity="barrier", state="unattributed", support_barrier=True)])
    result = allocate(episodes, corrections=[correction()])
    assert result["totals"]["observed_sampled_seconds"] == 90
    assert result["totals"]["nonforeground_seconds"] == 60
    assert result["totals"]["allocated_foreground_seconds"] == 0
    with pytest.raises(ValueError, match="cannot support"):
        record(episodes[2], timestamp=70)


def test_dst_repeated_hour_has_exact_elapsed_support():
    zone = ZoneInfo("America/Chicago")
    first = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0)
    second = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    episodes = store.build_episodes([{**sample(), "start_utc": first.isoformat(), "end_utc": second.isoformat(), "sampled_seconds": 3600}])
    assert len(episodes) == 4
    assert allocate(episodes)["totals"]["observed_sampled_seconds"] == 3600


def test_fractional_samples_and_correction_boundaries_preserve_exact_seconds():
    episodes = store.build_episodes([sample(0, 1.25, state="unattributed")])
    result = allocate(episodes, corrections=[correction(0.1, 0.35)])
    assert result["totals"]["observed_sampled_seconds"] == 1.25
    assert result["totals"]["allocated_foreground_seconds"] == 0.25
    assert result["totals"]["unknown_foreground_seconds"] == 1


def test_status_cli_is_metadata_only(tmp_path, monkeypatch, capsys):
    write_json(tmp_path / "inference-synthetic.json", {"private_result": "synthetic content"})
    monkeypatch.setattr(store, "_read_record", lambda _: pytest.fail("status must not deserialize records"))
    monkeypatch.setattr("sys.argv", ["activity_episode_store", "status", "--store-dir", str(tmp_path)])
    store.main()
    output = capsys.readouterr().out
    assert json.loads(output)["records"] == 1
    assert "synthetic content" not in output
    assert str(tmp_path) not in output


def test_reads_do_not_create_missing_store_and_reject_linked_directory(tmp_path):
    missing = tmp_path / "missing"
    assert store.status(missing)["records"] == 0
    assert store.read_fresh_inferences(missing, episodes=[], current_evidence={}, runtime_fingerprints={}) == []
    assert not missing.exists()
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical path"):
        store.status(linked)


def test_malformed_cache_objects_are_skipped_without_breaking_reader(tmp_path):
    write_json(tmp_path / "inference-array.json", [])
    assert store.read_fresh_inferences(tmp_path, episodes=[], current_evidence={}, runtime_fingerprints={}) == []


def test_invalid_claim_mapping_and_context_hash_fail_before_persistence(tmp_path):
    episodes = store.build_episodes([sample()])
    _, args = record(episodes[0])
    args["result"]["claim_evidence"]["task_candidate"] = []
    with pytest.raises(ValueError, match="current observation"):
        store.persist_inference(tmp_path / "never-created", **args)
    assert not (tmp_path / "never-created").exists()
    _, args = record(episodes[0])
    args["context"]["evidence"][0]["ocr"] = "Changed synthetic OCR"
    with pytest.raises(ValueError, match="Context fingerprint"):
        store.make_inference(**args)
