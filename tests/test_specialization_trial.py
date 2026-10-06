"""Synthetic trial gates, full-day denominators and exact observed support."""

import json
from datetime import date, datetime, timedelta, timezone

import pytest

import specialization_trial as trial
from activity_context import metadata_identity
from activity_episode_store import build_episodes
from private_io import write_json
from specialization_benchmark import GRADES, VARIANTS, BenchmarkError, fingerprint


def cohort(day=date(2026, 10, 4), count=10, limit=2):
    base = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=6)
    samples, rows, contexts = [], [], {}
    for i in range(count):
        instant = base + timedelta(minutes=30 * i)
        path = f"/synthetic/frame-{day}-{i}.webp"
        timestamp = (instant + timedelta(seconds=2)).isoformat()
        samples.append(dict(start_utc=instant.isoformat(), end_utc=(instant + timedelta(seconds=10)).isoformat(),
                            state="active", sampled_seconds=10, app="Editor", window="Example project"))
        rows.append((path, timestamp, "Editor", "Example project", "Outline", "complete"))
        contexts[path] = {"version": "activity_context_v1", "evidence": [{"id": f"screen-{i}", "source": "screen_context",
            "timestamp_utc": timestamp, "metadata_sha256": metadata_identity("Editor", "Example project")}]}
    return trial.build_cohort(day, rows, build_episodes(samples), limit,
                              context_builder=lambda path: contexts[path], image_digest=lambda path: "a" * 64)


def output(row):
    identity = row["context"]["evidence"][0]["id"]
    return {"version": "activity_result_v1", "activity_kind": "writing", "project_candidate": None,
            "task_candidate": None, "visible_work": "An outline is visible", "evidence_ids": [identity],
            "claim_evidence": {"activity_kind": [identity], "project_candidate": [], "task_candidate": [],
                               "visible_work": [identity]}, "uncertainty": "supported"}


def test_full_day_denominator_does_not_shrink_to_selected_frames():
    selected = cohort()
    data = {"results": {r["id"]: {"status": "complete", "activity": output(r)} for r in selected["selected"]},
            "sessions": []}
    result = trial.coverage(selected, data)
    assert result["raw_captures"] == result["eligible_captures"] == 10
    assert result["selected"] == result["completed"] == 2
    assert result["eligible_bins"] == result["occupied_foreground_bins"] == 10
    assert result["eligible_bin_coverage"] == result["observed_bin_coverage"] == .2
    assert result["semantic_accuracy"] is result["accurately_described_episodes"] is None
    assert result["production_changed"] is False
    assert "Example project" not in json.dumps(result)


def test_overlapping_frame_support_unions_exact_runs_and_rejects_gap_anchor():
    base = datetime(2026, 10, 4, 6, tzinfo=timezone.utc)
    def instant(seconds):
        return (base + timedelta(seconds=seconds)).isoformat()
    samples = [dict(start_utc=instant(a), end_utc=instant(b), state="active", sampled_seconds=b-a,
                    app="Editor", window="Example") for a, b in [(0, 5), (6, 11)]]
    episodes = build_episodes(samples, max_gap_seconds=2)
    rows = [(f"/synthetic/{i}.webp", instant(sec), "Editor", "Example", "Outline", "complete")
            for i, sec in enumerate((2, 8, 5.5))]
    def context(path):
        timestamp = next(r[1] for r in rows if r[0] == path)
        return {"evidence": [{"id": path, "source": "screen_context", "timestamp_utc": timestamp,
                             "metadata_sha256": metadata_identity("Editor", "Example")}]}
    selected = trial.build_cohort(date(2026, 10, 4), rows + [rows[0]], episodes, 10,
                                  context_builder=context, image_digest=lambda p: "a" * 64)
    assert selected["excluded"] == {"no_foreground_anchor": 1, "duplicate_observation": 1}
    assert selected["eligible_captures"] == len(selected["selected"]) == 2
    data = {"results": {r["id"]: {"status": "complete"} for r in selected["selected"]}}
    assert trial.coverage(selected, data)["valid_capture_support_seconds"] == 10
    assert trial.coverage(selected, data)["model_claimed_supported_seconds"] == 0


def test_two_fall_back_hour_occurrences_remain_distinct_bins():
    first = datetime.fromisoformat("2026-11-01T01:10:00-05:00")
    second = datetime.fromisoformat("2026-11-01T01:10:00-06:00")
    assert len(trial.bins([(first, first + timedelta(seconds=1)), (second, second + timedelta(seconds=1))])) == 2


def test_three_nights_require_three_nonempty_distinct_cohorts_and_success_nights():
    summaries = [dict(source_day=f"2026-10-0{i}", cohort_complete=True, completed=1,
                      execution_nights=["2026-10-05"]) for i in (1, 2, 3)]
    assert not trial.trial_ready(summaries)
    for i, row in enumerate(summaries, 1):
        row["execution_nights"] = [f"2026-10-0{i+1}"]
    assert trial.trial_ready(summaries)
    summaries[2]["completed"] = 0
    assert not trial.trial_ready(summaries)
    assert not trial.trial_ready([summaries[0]] * 3)


def test_daytime_trial_never_reads_private_inputs_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "Guard", lambda budget: type("Guard", (), {
        "check": lambda self, reserve: (_ for _ in ()).throw(trial.BudgetEnded("outside_vision_window"))})())
    monkeypatch.setattr(trial, "verify_trial", lambda *a: pytest.fail("private trial read"))
    result = trial.run_trial({}, tmp_path / "study", 1000)
    assert result == {"status": "partial", "stop_reason": "outside_vision_window", "private_examples_opened": False}
    assert not (tmp_path / "study").exists()


def reviewed_comparison(root, split, experiment, count=2):
    data = {"experiment_sha256": fingerprint(experiment), "results": {},
            "sessions": [{"stage": "session", "status": "finished", "elapsed_seconds": 100}]}
    assessment = {"experiment_sha256": fingerprint(experiment), "items": {}}
    for i in range(count):
        identity = f"example-{i}"
        data["results"][identity] = {}
        for variant in VARIANTS:
            key = identity + variant
            record = {"status": "complete", "activity": None, "description": "A synthetic outline", "assessment_key": key,
                      "elapsed_seconds": 10}
            data["results"][identity][variant] = record
            assessment["items"][key] = {"example_id": identity,
                "output": {k: record.get(k) for k in ("activity", "description", "status")},
                "grades": {k: k not in {"privacy_leak", "unsupported_completion"} for k in GRADES},
                "claim_support": {k: "not_asserted" for k in ("activity_kind", "project_candidate", "task_candidate", "visible_work")}}
    folder = root / "benchmark" / split
    write_json(folder / "results.json", data)
    write_json(folder / "blind-assessment.json", assessment)
    return data, assessment


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "study"
    config = {"exports": {"validation": {"count": 2}, "test": {"count": 2}}}
    spec = {"version": "synthetic", "configuration": config, "parameters": {"batch_requests": 2}}
    monkeypatch.setattr(trial, "experiment_spec", lambda config, root: spec)
    reviewed_comparison(root, "validation", spec)
    reviewed_comparison(root, "test", spec)
    candidate = {"candidate": "mlx_base", "specification": spec, "experiment_sha256": fingerprint(spec),
        "validation_results_sha256": trial.file_digest(root / "benchmark/validation/results.json", private=True),
        "reviewed_assessment_sha256": trial.file_digest(root / "benchmark/validation/blind-assessment.json", private=True)}
    candidate["frozen_candidate_sha256"] = fingerprint(candidate)
    write_json(root / "benchmark/frozen-candidate.json", candidate)
    result = trial.freeze_trial(config, root, reviewed_test_sha256=trial.file_digest(root / "benchmark/test/blind-assessment.json", private=True))
    return config, root, result


def test_trial_freeze_cannot_reselect_candidate_or_reuse_changed_grades(frozen):
    config, root, receipt = frozen
    assert receipt["production_changed"] is False
    assert trial.verify_trial(config, root)["candidate"] == "mlx_base"
    path = root / "benchmark/test/blind-assessment.json"
    data = trial.read_state(path)
    first = next(iter(data["items"].values()))
    first["grades"]["project_correct"] = False
    write_json(path, data)
    with pytest.raises(BenchmarkError, match="trial_review_changed"):
        trial.verify_trial(config, root)


def test_missing_claim_review_cannot_start_trial(frozen):
    config, root, receipt = frozen
    path = root / "benchmark/test/blind-assessment.json"
    data = trial.read_state(path)
    next(iter(data["items"].values()))["claim_support"]["visible_work"] = None
    write_json(path, data)
    with pytest.raises(BenchmarkError, match="claim_review_incomplete"):
        trial.freeze_trial(config, root, reviewed_test_sha256=trial.file_digest(path, private=True))


@pytest.fixture
def runtime(frozen, monkeypatch):
    config, root, receipt = frozen
    clock = [datetime(2026, 10, 6, 1, tzinfo=trial.ZONE)]
    calls = []
    class Now(datetime):
        @classmethod
        def now(cls, zone=None):
            return clock[0].astimezone(zone or timezone.utc)
    class Guard:
        def __init__(self, budget):
            pass
        def check(self, reserve):
            pass
    class Runner:
        def __init__(self, spec, guard):
            pass
        def start(self, variant):
            assert variant == "mlx_base"
        def infer(self, row, identity):
            calls.append(identity)
            return {"status": "complete", "activity": output(row), "raw_safety_checks": {
                "available": True, "sensitive_keyword": False, "email": False, "url": False, "unsupported_completion": False}}
        def close(self):
            pass
    monkeypatch.setattr(trial, "datetime", Now)
    monkeypatch.setattr(trial, "Guard", Guard)
    monkeypatch.setattr(trial, "Runner", Runner)
    monkeypatch.setattr(trial, "load_cohort", lambda day, limit, guard: cohort(day, 4, min(limit, 4)))
    return config, root, clock, calls


def test_shadow_runs_three_distinct_days_and_never_changes_production(runtime):
    config, root, clock, calls = runtime
    first = trial.run_trial(config, root, 1000)
    assert first["cohort_complete"] and first["completed"] == 4
    assert first["semantic_accuracy"] is None and first["production_changed"] is False
    assert trial.run_trial(config, root, 1000)["stop_reason"] == "awaiting_next_distinct_night"
    assert len(calls) == 4
    for i in (1, 2):
        clock[0] += timedelta(days=1)
        result = trial.run_trial(config, root, 1000)
    assert result["status"] == "trial_complete" and len(calls) == 12
    recommendation = trial.recommendation(root)
    assert len(recommendation["source_days"]) == len(recommendation["execution_nights"]) == 3
    assert recommendation["status"] == "awaiting_coordinator_recommendation"
    assert recommendation["daily_semantic_accuracy"] is recommendation["accurate_episodes_per_night"] is None
    assert recommendation["production_changed"] is False
    assert "An outline" not in json.dumps(recommendation)


def test_crashed_trial_cost_survives_resume_and_is_not_measured(runtime, monkeypatch):
    config, root, clock, calls = runtime
    original = trial.Runner.infer
    interrupted = False
    def infer(self, row, identity):
        nonlocal interrupted
        if not interrupted and len(calls) == 1:
            interrupted = True
            raise KeyboardInterrupt
        return original(self, row, identity)
    monkeypatch.setattr(trial.Runner, "infer", infer)
    with pytest.raises(KeyboardInterrupt):
        trial.run_trial(config, root, 1000)
    result = trial.run_trial(config, root, 1000)
    assert result["cohort_complete"] and result["failed"] == 1
    assert result["total_session_seconds"] is None
    assert result["conservative_total_seconds"] >= 1030


def test_preparation_crash_is_journaled_before_private_load(runtime, monkeypatch):
    config, root, clock, calls = runtime
    original = trial.load_cohort
    def interrupted(*args):
        data = trial.read_state(root / "trial/2026-10-05/results.json")
        assert data["sessions"][0]["status"] == "running"
        assert data["sessions"][0]["reserved_seconds"] == 1030
        raise KeyboardInterrupt
    monkeypatch.setattr(trial, "load_cohort", interrupted)
    with pytest.raises(KeyboardInterrupt):
        trial.run_trial(config, root, 1000)
    assert not calls
    monkeypatch.setattr(trial, "load_cohort", original)
    result = trial.run_trial(config, root, 1000)
    assert result["completed"] == 4
    assert result["total_session_seconds"] is None
    assert result["conservative_total_seconds"] >= 1030
    assert trial.recommendation(root)["nights"][0]["interrupted_sessions"] == 1


def test_preparation_budget_refusal_preserves_wall_cost(runtime, monkeypatch):
    config, root, clock, calls = runtime
    monkeypatch.setattr(trial, "load_cohort", lambda *a: (_ for _ in ()).throw(trial.BudgetEnded("battery_power")))
    result = trial.run_trial(config, root, 1000)
    assert result["stop_reason"] == "battery_power"
    assert result["preparation_complete"] is False and not calls
    assert result["total_session_seconds"] is not None


def test_crash_after_cohort_write_recovers_on_following_night(runtime, monkeypatch):
    config, root, clock, calls = runtime
    original = trial.write_json
    crashed = False
    def write(path, value):
        nonlocal crashed
        original(path, value)
        if path.name == "cohort.json" and not crashed:
            crashed = True
            raise KeyboardInterrupt
    monkeypatch.setattr(trial, "write_json", write)
    with pytest.raises(KeyboardInterrupt):
        trial.run_trial(config, root, 1000)
    clock[0] += timedelta(days=1)
    result = trial.run_trial(config, root, 1000)
    assert result["source_day"] == "2026-10-05"
    assert result["completed"] == 4 and result["conservative_total_seconds"] >= 1030


def test_indexed_preparation_journal_survives_to_following_night(runtime, monkeypatch):
    config, root, clock, calls = runtime
    original = trial.write_json
    crashed = False
    def write(path, value):
        nonlocal crashed
        original(path, value)
        if path.name == "results.json" and not crashed:
            crashed = True
            raise KeyboardInterrupt
    monkeypatch.setattr(trial, "write_json", write)
    with pytest.raises(KeyboardInterrupt):
        trial.run_trial(config, root, 1000)
    clock[0] += timedelta(days=1)
    result = trial.run_trial(config, root, 1000)
    assert result["source_day"] == "2026-10-05"
    assert result["completed"] == 4 and result["conservative_total_seconds"] >= 1030


def test_candidate_review_hashes_cannot_change_with_old_token(frozen):
    config, root, _ = frozen
    path = root / "benchmark/frozen-candidate.json"
    item = trial.read_state(path)
    item["validation_results_sha256"] = "a" * 64
    write_json(path, item)
    with pytest.raises(BenchmarkError, match="trial_candidate_changed"):
        trial.verify_trial(config, root)


def test_wrong_frame_identity_cannot_resume_or_be_recommended(runtime):
    config, root, clock, calls = runtime
    trial.run_trial(config, root, 1000)
    path = root / "trial/2026-10-05/results.json"
    item = trial.read_state(path)
    next(iter(item["results"].values()))["identity"] = "a" * 64
    write_json(path, item)
    with pytest.raises(BenchmarkError, match="trial_frame_identity_changed"):
        trial.run_trial(config, root, 1000)
    with pytest.raises(BenchmarkError, match="trial_frame_identity_changed"):
        trial.recommendation(root)


def test_cached_summary_from_another_trial_cannot_claim_success(runtime):
    config, root, clock, calls = runtime
    trial.run_trial(config, root, 1000)
    write_json(root / "trial/2026-10-05/summary.json", {
        "trial_sha256": "a" * 64, "source_day": "2026-10-05", "completed": 999,
        "cohort_complete": True, "execution_nights": ["2026-10-06", "2026-10-07", "2026-10-08"]})
    result = trial.recommendation(root)
    assert result["status"] == "trial_incomplete"
    assert result["nights"][0]["completed"] == 4
    assert result["execution_nights"] == ["2026-10-06"]


def test_failed_worker_preserves_boolean_safety_and_failure_code(runtime, monkeypatch):
    config, root, clock, calls = runtime
    def blocked(*args):
        error = trial.WorkerError("raw_safety_failure")
        error.telemetry_attempt_id = "a" * 32
        error.raw_safety_checks = {"available": True, "sensitive_keyword": True, "email": False,
                                   "url": False, "unsupported_completion": False, "private_text": "SECRET"}
        raise error
    monkeypatch.setattr(trial.Runner, "infer", blocked)
    result = trial.run_trial(config, root, 1000)
    assert result["failed"] == result["raw_privacy_flags"] == result["raw_safety_available"] == 1
    assert result["pending"] == 3
    assert result["failure_codes"] == {"raw_safety_failure": 1}
    assert "SECRET" not in json.dumps(trial.read_state(root / "trial/2026-10-05/results.json"))


@pytest.mark.parametrize("at_start,attempted", [(True, False), (False, False), (False, True)])
def test_resource_refusal_is_pending_until_an_attempt_was_admitted(runtime, monkeypatch, at_start, attempted):
    config, root, clock, calls = runtime
    def refused(*args):
        error = trial.WorkerError("battery_power")
        if attempted:
            error.telemetry_attempt_id = "a" * 32
        raise error
    monkeypatch.setattr(trial.Runner, "start" if at_start else "infer", refused)
    result = trial.run_trial(config, root, 1000)
    assert result["failed"] == int(attempted)
    assert result["pending"] == 4 - int(attempted)
    if not attempted:
        assert result["stop_reason"] == "battery_power"


def test_empty_cohort_finishes_without_counting_as_a_successful_night(runtime, monkeypatch):
    config, root, clock, calls = runtime
    monkeypatch.setattr(trial, "load_cohort", lambda day, limit, guard: cohort(day, 0, 1))
    result = trial.run_trial(config, root, 1000)
    assert result["cohort_complete"] and result["completed"] == 0 and not calls
    assert result["execution_nights"] == []
    assert trial.recommendation(root)["status"] == "trial_incomplete"
