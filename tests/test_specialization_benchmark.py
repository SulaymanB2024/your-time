"""Synthetic-only benchmark checks; no GPU, model, ledger or private corpus."""

import copy
import hashlib
import json
import os
from datetime import datetime, timezone

import pytest

import specialization_benchmark as benchmark
from activity_context import RESULT_VERSION, prompt
from private_io import atomic_write, write_json


def test_trained_adapter_is_compared_even_when_literal_proxy_selects_base():
    final = "step-000030-aabbccdd"
    progress = {"epoch": 2, "offset": 0, "checkpoint": final, "best": {"checkpoint": None}}
    assert benchmark.adapter_checkpoint(progress) == (final, "final_epoch_experiment_requires_semantic_review")
    progress["best"]["checkpoint"] = "step-000015-aabbccdd"
    assert benchmark.adapter_checkpoint(progress) == ("step-000015-aabbccdd", "validation_proxy_preselection")
    progress.update(epoch=0, offset=10, best={"checkpoint": None})
    assert benchmark.adapter_checkpoint(progress)[0] is None
    progress["best"]["checkpoint"] = "../adapter"
    with pytest.raises(benchmark.BenchmarkError, match="invalid_best_checkpoint"):
        benchmark.adapter_checkpoint(progress)


def example(identity="screen-a", *, episode="episode-a", project="Example project", task=None):
    context = {"version": "activity_context_v1", "evidence": [{"id": identity,
        "source": "screen_context", "timestamp_utc": "2026-10-01T13:00:00+00:00",
        "app": "Example editor", "window": "Example project", "ocr": "Draft outline"}]}
    result = {"version": RESULT_VERSION, "activity_kind": "writing", "project_candidate": project,
        "task_candidate": task, "visible_work": "A draft outline is visible",
        "evidence_ids": [identity], "claim_evidence": {"activity_kind": [identity],
            "project_candidate": [identity] if project is not None else [],
            "task_candidate": [identity] if task is not None else [], "visible_work": [identity]},
        "uncertainty": "supported"}
    return {"id": identity, "images": ["/synthetic/owned-frame.webp"], "image_sha256": "a" * 64,
        "context": context, "episode_id": episode, "split": "validation", "source_class": "real",
        "messages": [{"role": "user", "content": prompt(context)},
                     {"role": "assistant", "content": json.dumps(result)}], "reference": result}


def private_export(data_root, rows, split="validation"):
    payload = ("\n".join(json.dumps({k: v for k, v in row.items() if k != "reference"}) for row in rows) + "\n").encode()
    path = data_root / ("sealed" if split == "test" else "export") / (split + ".jsonl")
    atomic_write(path, payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "count": len(rows)}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    study = tmp_path / "study"
    rows = [example(), example("screen-b", episode="episode-b", task="Draft outline")]
    config = {"exports": {"validation": private_export(data_root, rows),
                          "test": {"sha256": "b" * 64, "count": 19}}, "seed": 7}
    calls = []

    class Guard:
        def __init__(self, budget):
            self.deadline = 1e30

        def check(self, reserve=0):
            return None

        def remaining(self):
            return 1000

    class Runner:
        def __init__(self, spec, guard):
            self.worker = None

        def start(self, variant):
            self.variant = variant

        def infer(self, row, identity):
            calls.append((self.variant, row["id"]))
            if self.variant == "production_q8":
                return {"status": "complete", "description": "An outline is visible"}
            return {"status": "complete", "activity": row["reference"], "raw_safety_checks": {
                "available": True, "sensitive_keyword": False, "email": False,
                "url": False, "unsupported_completion": False}}

        def close(self):
            pass

    def spec(config, study_root):
        return {"version": benchmark.VERSION, "configuration": config,
            "parameters": benchmark.parameters(config), "pins": {"adapter": {"id": "selected"}},
            "engine_sha256": "c" * 64}

    monkeypatch.setattr(benchmark, "DATA_ROOT", data_root)
    monkeypatch.setattr(benchmark, "Guard", Guard)
    monkeypatch.setattr(benchmark, "Runner", Runner)
    monkeypatch.setattr(benchmark, "experiment_spec", spec)
    return config, study, rows, calls


def test_independent_null_labels_and_stale_context_are_not_semantic_grades():
    row = example(task=None)
    response = {"status": "complete", "activity": copy.deepcopy(row["reference"])}
    response["activity"]["project_candidate"] = "Unrelated project"
    row["context"]["evidence"].append({"id": "nearby", "source": "nearby_observation",
                                       "window": "Unrelated project"})
    scored = benchmark.score_response(response, row)
    assert scored["exact_proxy"]["project_candidate"] is False
    assert scored["exact_proxy"]["task_candidate"] is True
    assert scored["exact_proxy"]["joint"] is False
    assert "evidence_supported" not in scored["exact_proxy"]


def test_raw_safety_blocks_even_when_clean_final_matches_reference():
    row = example()
    scored = benchmark.score_response({"status": "complete", "activity": row["reference"],
        "raw_output": "An account email person@example.invalid was visible"}, row)
    assert scored["status"] == "raw_safety_blocked"
    assert scored["raw_safety"]["privacy"] is True
    assert scored["exact_proxy"] is None
    assert "raw_output" not in scored


def test_aggregate_raw_flags_and_invalid_schema_are_preserved():
    row = example()
    response = {"status": "complete", "activity": {"invalid": "shape"},
        "raw_safety_checks": {"available": True, "sensitive_keyword": False, "email": False,
                              "url": False, "unsupported_completion": True}}
    scored = benchmark.score_response(response, row)
    assert scored["status"] == "raw_safety_blocked"
    assert scored["raw_safety"]["completion"]


def test_failures_stay_denominator_and_unavailable_raw_checks_are_null():
    rows = [example(), example("screen-b", episode="episode-b")]
    good = benchmark.score_response({"status": "complete", "activity": rows[0]["reference"]}, rows[0])
    good["elapsed_seconds"] = 1
    results = {rows[0]["id"]: {"mlx_base": good}, rows[1]["id"]: {"mlx_base": {
        "status": "timeout", "elapsed_seconds": 9, "exact_proxy": None}}}
    summary = benchmark.aggregate(rows, results)
    scores = summary["variants"]["mlx_base"]
    assert scores["selected_denominator"] == 2
    assert scores["complete"] == 1 and scores["failed"] == 1
    assert scores["literal_joint_proxy_correct"] == 1
    assert scores["project_proxy"]["answerable_recall"] == .5
    assert scores["raw_privacy_flags"] is None
    assert scores["total_attempt_seconds"] == 10


def test_cluster_interval_does_not_treat_one_episode_as_independent_frames():
    one = benchmark.paired_interval([("same", False, True)] * 20)
    assert one["pairs"] == 20 and one["groups"] == 1
    assert one["interval_95"] is None
    pairs = [("one", False, True)] * 5 + [("two", True, False)] * 5
    assert benchmark.paired_interval(pairs) == benchmark.paired_interval(pairs)
    assert benchmark.paired_interval(pairs)["interval_95"] == [-1, 1]


def test_temporal_coverage_counts_unique_bins_and_episodes():
    rows = [example(), example("screen-b", episode="episode-a")]
    results = {}
    for row in rows:
        scored = benchmark.score_response({"status": "complete", "activity": row["reference"]}, row)
        scored["elapsed_seconds"] = 1
        results[row["id"]] = {"mlx_base": scored}
    summary = benchmark.aggregate(rows, results)
    assert summary["unique_episodes"] == 1
    assert summary["unique_30_minute_bins"] == 1
    assert summary["variants"]["mlx_base"]["complete_unique_episodes"] == 1
    assert summary["variants"]["mlx_base"]["complete_unique_30_minute_bins"] == 1


def test_guard_refuses_daytime_without_reading_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "remaining_seconds", lambda *a, **k: 0)
    monkeypatch.setattr(benchmark, "load_examples", lambda *a: pytest.fail("export opened"))
    result = benchmark.run_validation({}, tmp_path / "study", 100)
    assert result == {"status": "partial", "stop_reason": "outside_vision_window",
                      "private_examples_opened": False}
    assert not (tmp_path / "study").exists()


def test_validation_resume_full_identity_and_blind_files(harness):
    config, study, rows, calls = harness
    result = benchmark.run_validation(config, study, 1000)
    assert result["status"] == "complete" and len(calls) == 8
    assert "Example project" not in json.dumps(result)
    assert "screen-a" not in json.dumps(result)
    benchmark.run_validation(config, study, 1000)
    assert len(calls) == 8
    assessment = benchmark.read_state(study / "benchmark/validation/blind-assessment.json")
    assert len(assessment["items"]) == 8
    assert all("variant" not in entry for entry in assessment["items"].values())
    assert os.stat(study / "benchmark/validation/results.json").st_mode & 0o777 == 0o600
    changed = {**config, "image_tokens": 1024}
    mismatch = benchmark.run_validation(changed, study, 1000)
    assert mismatch["stop_reason"] == "resume_identity_mismatch"
    assert len(calls) == 8


def test_context_change_cannot_resume_old_results(harness, monkeypatch):
    config, study, rows, calls = harness
    benchmark.run_validation(config, study, 1000)
    changed_rows = copy.deepcopy(rows)
    changed_rows[0]["context"]["evidence"][0]["window"] = "Other visible context"
    # Emulate a dishonest unchanged export identity; the per-example identity still refuses reuse.
    monkeypatch.setattr(benchmark, "load_examples", lambda *a: changed_rows)
    result = benchmark.run_validation(config, study, 1000)
    assert result["stop_reason"] == "resume_example_changed"
    assert len(calls) == 8


def test_failure_is_not_rerun_or_removed_on_resume(harness, monkeypatch):
    config, study, rows, calls = harness
    runner = benchmark.Runner
    original = runner.infer

    def infer(self, row, identity):
        if self.variant == "context_q8" and row["id"] == "screen-a":
            calls.append((self.variant, row["id"]))
            raise TimeoutError("synthetic diagnostics must not escape")
        return original(self, row, identity)

    monkeypatch.setattr(runner, "infer", infer)
    result = benchmark.run_validation(config, study, 1000)
    assert result["status"] == "complete"
    assert result["variants"]["context_q8"]["failed"] == 1
    benchmark.run_validation(config, study, 1000)
    assert len(calls) == 8
    assert "synthetic diagnostics" not in json.dumps(result)


def test_pre_inference_time_refusal_stays_pending_and_resumes(harness, monkeypatch):
    from specialization_worker import WorkerError

    config, study, rows, calls = harness
    original = benchmark.Runner.infer
    refused = False
    def infer(self, row, identity):
        nonlocal refused
        if not refused and self.variant == "mlx_base":
            refused = True
            raise WorkerError("insufficient_window_for_request")
        return original(self, row, identity)
    monkeypatch.setattr(benchmark.Runner, "infer", infer)
    partial = benchmark.run_validation(config, study, 1000)
    assert partial["status"] == "partial" and partial["stop_reason"] == "benchmark_budget"
    data = benchmark.read_state(study / "benchmark/validation/results.json")
    assert sum("mlx_base" in result for result in data["results"].values()) == 0
    assert benchmark.run_validation(config, study, 1000)["status"] == "complete"
    assert len(calls) == 8


def reviewed_freeze(config, study):
    benchmark.run_validation(config, study, 1000)
    path = study / "benchmark/validation/blind-assessment.json"
    assessment = benchmark.read_state(path)
    for entry in assessment["items"].values():
        entry["grades"] = {key: key not in {"privacy_leak", "unsupported_completion"}
                           for key in benchmark.GRADES}
    write_json(path, assessment)
    return benchmark.freeze_candidate(config, study, candidate="mlx_adapter",
        reviewed_assessment_sha256=benchmark.file_digest(path, private=True))


def test_freeze_requires_complete_human_review_and_does_not_open_test(harness, monkeypatch):
    config, study, rows, calls = harness
    benchmark.run_validation(config, study, 1000)
    path = study / "benchmark/validation/blind-assessment.json"
    with pytest.raises(benchmark.BenchmarkError, match="blind_review_incomplete"):
        benchmark.freeze_candidate(config, study, candidate="mlx_adapter",
            reviewed_assessment_sha256=benchmark.file_digest(path, private=True))
    monkeypatch.setattr(benchmark, "load_examples", lambda *a: pytest.fail("test or export opened during freeze"))
    assessment = benchmark.read_state(path)
    for entry in assessment["items"].values():
        entry["grades"] = {key: key not in {"privacy_leak", "unsupported_completion"}
                           for key in benchmark.GRADES}
    write_json(path, assessment)
    result = benchmark.freeze_candidate(config, study, candidate="mlx_adapter",
        reviewed_assessment_sha256=benchmark.file_digest(path, private=True))
    assert result["status"] == "frozen"


def test_locked_test_rejects_changed_configuration_before_any_test_read(harness, monkeypatch):
    config, study, rows, calls = harness
    frozen = reviewed_freeze(config, study)
    monkeypatch.setattr(benchmark, "load_examples", lambda *a: pytest.fail("sealed test opened"))
    result = benchmark.run_locked_test({**config, "image_tokens": 1024}, study, 1000,
        frozen_candidate_sha256=frozen["frozen_candidate_sha256"])
    assert result["stop_reason"] == "frozen_configuration_mismatch"
    assert result["test_access_authorized"] is False


def test_locked_test_opens_only_matching_frozen_candidate(harness):
    config, study, rows, calls = harness
    test_rows = copy.deepcopy(rows)
    for row in test_rows:
        row["split"] = "test"
    config["exports"]["test"] = private_export(benchmark.DATA_ROOT, test_rows, "test")
    frozen = reviewed_freeze(config, study)
    result = benchmark.run_locked_test(config, study, 1000,
        frozen_candidate_sha256=frozen["frozen_candidate_sha256"])
    assert result["status"] == "complete" and result["split"] == "test"
    assert len(calls) == 16
    result = benchmark.run_locked_test(config, study, 1000,
        frozen_candidate_sha256=frozen["frozen_candidate_sha256"])
    assert result["status"] == "complete" and len(calls) == 16


def test_human_pairs_include_caption_baseline_without_inventing_labels():
    rows = [example()]
    results = {"screen-a": {"production_q8": {"status": "complete", "assessment_key": "a",
        "elapsed_seconds": 1}, "context_q8": {"status": "complete", "assessment_key": "b",
        "elapsed_seconds": 1}}}
    grades = {k: k not in {"privacy_leak", "unsupported_completion"} for k in benchmark.GRADES}
    bad = {**grades, "project_correct": False}
    assessment = {"items": {"a": {"grades": bad}, "b": {"grades": grades}}}
    summary = benchmark.aggregate(rows, results, assessment)
    assert summary["paired_literal_proxies"]["production_q8__context_q8"]["pairs"] == 0
    assert summary["paired_human_scores"]["production_q8__context_q8"]["delta"] == 1


def test_dst_bin_uses_local_time_and_wilson_handles_zero_denominator():
    row = example()
    row["context"]["evidence"][0]["timestamp_utc"] = datetime(2026, 11, 1, 7, 20,
                                                            tzinfo=timezone.utc).isoformat()
    assert benchmark.temporal_key(row) == ("2026-11-01", 1, 0, -21600)
    earlier = copy.deepcopy(row)
    earlier["context"]["evidence"][0]["timestamp_utc"] = "2026-11-01T06:20:00+00:00"
    assert benchmark.temporal_key(earlier) != benchmark.temporal_key(row)
    assert benchmark.wilson(0, 0) is None
    assert benchmark.wilson(0, 20)[0] == 0


def test_q8_pair_preserves_first_attempt_budgets_separate_from_mlx(tmp_path, monkeypatch):
    import vision_fallback
    import vision_quality_eval

    monkeypatch.setattr(vision_fallback, "generation_budget", lambda n: (1536, 240) if n == 0 else (2048, 300))
    config = {"image_tokens": 512, "max_tokens": 768, "thinking": False, "context_tokens": 4096}
    q8 = benchmark.q8_parameters(config)
    assert q8["max_tokens"] == 1536 and q8["timeout_seconds"] == 240
    image = tmp_path / "data/frame.webp"
    atomic_write(image, b"synthetic image bytes, no model/PIL decoding")
    row = example()
    row["images"] = [str(image)]
    row["image_sha256"] = benchmark.file_digest(image, private=True)
    monkeypatch.setattr(benchmark, "DATA_ROOT", image.parent)
    captured = []
    monkeypatch.setattr(vision_quality_eval, "run_image", lambda path, model, **kwargs: captured.append(kwargs) or {})

    class Guard:
        def check(self, reserve=0):
            pass

        def remaining(self):
            return 1000

    spec = {"parameters": benchmark.parameters(config), "q8_parameters": q8,
            "pins": {"q8": {"model_dir": str(tmp_path),
                "weights": {"name": "pinned.gguf", "sha256": "a" * 64},
                "projector": {"name": "projector.gguf", "sha256": "b" * 64}}}}
    runner = benchmark.Runner(spec, Guard())
    for variant in ("production_q8", "context_q8"):
        runner.variant = variant
        runner.infer(row, "c" * 64)
    for request in captured:
        assert request["max_tokens"] == 1536 and request["image_tokens"] == 1024
        assert request["image_side"] == 1600 and request["threads"] == 4
        assert request["timeout_seconds"] == 240
    assert captured[0]["context"] is None
    assert captured[1]["context"] == row["context"]


def test_worker_error_raw_flags_survive_failure(harness, monkeypatch):
    from specialization_worker import WorkerError

    config, study, rows, calls = harness
    original = benchmark.Runner.infer

    def infer(self, row, identity):
        if self.variant == "mlx_adapter":
            error = WorkerError("raw_safety_failure")
            error.raw_safety_checks = {"available": True, "sensitive_keyword": False,
                "email": False, "url": False, "unsupported_completion": True}
            raise error
        return original(self, row, identity)

    monkeypatch.setattr(benchmark.Runner, "infer", infer)
    summary = benchmark.run_validation(config, study, 1000)
    assert summary["variants"]["mlx_adapter"]["raw_completion_flags"] >= 1
    assert summary["variants"]["mlx_adapter"]["literal_joint_proxy_correct"] == 0


def test_pending_budget_interruption_does_not_become_a_false_attempt(harness, monkeypatch):
    config, study, rows, calls = harness
    monkeypatch.setattr(benchmark.Runner, "infer", lambda *a: (_ for _ in ()).throw(benchmark.BudgetEnded("benchmark_budget")))
    summary = benchmark.run_validation(config, study, 1000)
    assert summary["status"] == "partial"
    assert sum(v["attempted"] for v in summary["variants"].values()) == 0
    assert sum(v["pending"] for v in summary["variants"].values()) == 8
    data = benchmark.read_state(study / "benchmark/validation/results.json")
    assert all(not variants for variants in data["results"].values())


def test_preparation_crash_retains_unmeasured_reservation(harness, monkeypatch):
    config, study, rows, calls = harness
    original = benchmark.load_examples
    def interrupted(*args):
        data = benchmark.read_state(study / "benchmark/validation/results.json")
        assert data["sessions"][0]["status"] == "running"
        assert data["sessions"][0]["reserved_seconds"] == 1030
        raise KeyboardInterrupt
    monkeypatch.setattr(benchmark, "load_examples", interrupted)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_validation(config, study, 1000)
    monkeypatch.setattr(benchmark, "load_examples", original)
    result = benchmark.run_validation(config, study, 1000)
    assert result["status"] == "complete"
    assert result["total_session_seconds"] is None
    assert result["conservative_total_seconds"] >= 1030
    assert result["interrupted_sessions"] == 1


def test_resource_guard_retains_exact_validated_stop_reason(monkeypatch):
    import vision_batch

    monkeypatch.setattr(benchmark, "remaining_seconds", lambda *a, **k: 1000)
    monkeypatch.setattr(vision_batch, "resource_gate", lambda **k: "battery_power")
    with pytest.raises(benchmark.BudgetEnded, match="^battery_power$"):
        benchmark.Guard(1000).check()


def test_export_hash_mismatch_does_not_echo_private_content(tmp_path):
    root = tmp_path / "data"
    row = example()
    private_export(root, [row])
    with pytest.raises(benchmark.BenchmarkError, match="export_changed") as error:
        benchmark.load_examples(root / "export/validation.jsonl", "validation", {"count": 1, "sha256": "e" * 64})
    assert "Example project" not in str(error.value)


def test_full_spec_changes_for_engine_adapter_prompt_and_configuration(monkeypatch, tmp_path):
    pins = {"engine": "initial", "adapter": "first"}
    monkeypatch.setattr(benchmark, "model_pins", lambda *a: copy.deepcopy(pins))
    config = {"exports": {"validation": {"count": 20, "sha256": "a" * 64}}}
    initial = benchmark.fingerprint(benchmark.experiment_spec(config, tmp_path))
    pins["engine"] = "changed"
    assert benchmark.fingerprint(benchmark.experiment_spec(config, tmp_path)) != initial
    pins["engine"] = "initial"
    pins["adapter"] = "second"
    assert benchmark.fingerprint(benchmark.experiment_spec(config, tmp_path)) != initial
    pins["adapter"] = "first"
    monkeypatch.setattr(benchmark, "prompt", lambda ctx: "Changed template " + json.dumps(ctx))
    assert benchmark.fingerprint(benchmark.experiment_spec(config, tmp_path)) != initial
    assert benchmark.fingerprint(benchmark.experiment_spec({**config, "q8_image_side": 1024}, tmp_path)) != initial


def test_freeze_rejects_correct_grade_for_failure(harness, monkeypatch):
    config, study, rows, calls = harness
    original = benchmark.Runner.infer

    def infer(self, row, identity):
        if self.variant == "context_q8":
            return {"status": "timeout"}
        return original(self, row, identity)

    monkeypatch.setattr(benchmark.Runner, "infer", infer)
    with pytest.raises(benchmark.BenchmarkError, match="failure_cannot_receive_correct_grade"):
        reviewed_freeze(config, study)


def test_startup_is_counted_in_wall_capacity(harness, monkeypatch):
    config, study, rows, calls = harness
    clock = [0.0]
    original = benchmark.Runner.start

    def start(self, variant):
        clock[0] += 10
        return original(self, variant)

    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark.Runner, "start", start)
    summary = benchmark.run_validation(config, study, 1000)
    assert summary["status"] == "complete"
    assert summary["total_session_seconds"] == 40
    assert sum(summary["startup_seconds_by_variant"].values()) == 40
    assert sum(v["total_attempt_seconds"] for v in summary["variants"].values()) == 0


def test_crashed_session_keeps_unknown_cost_and_conservative_reservation(harness, monkeypatch):
    config, study, rows, calls = harness
    original = benchmark.Runner.infer
    clock = [0.0]
    interrupted = False
    def infer(self, row, identity):
        nonlocal interrupted
        clock[0] += 7
        if not interrupted and len(calls) == 1:
            interrupted = True
            raise KeyboardInterrupt
        return original(self, row, identity)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark.Runner, "infer", infer)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_validation(config, study, 1000)
    data = benchmark.read_state(study / "benchmark/validation/results.json")
    assert data["sessions"][0]["status"] == "running"
    assert data["sessions"][0]["elapsed_seconds"] is None
    summary = benchmark.run_validation(config, study, 1000)
    assert summary["status"] == "complete"
    assert summary["total_session_seconds"] is None
    assert summary["interrupted_sessions"] == 1
    assert summary["conservative_total_seconds"] >= 1030


def test_mlx_runner_uses_pinned_worker_identity_and_identical_pair_settings(tmp_path, monkeypatch):
    import specialization_worker

    image = tmp_path / "data/frame.webp"
    atomic_write(image, b"synthetic immutable frame")
    row = example()
    row["images"] = [str(image)]
    row["image_sha256"] = benchmark.file_digest(image, private=True)
    monkeypatch.setattr(benchmark, "DATA_ROOT", image.parent)
    controllers = []

    class Guard:
        def check(self, reserve=0):
            pass

        def remaining(self):
            return 1000

    class Controller:
        def __init__(self, config):
            self.config, self.requests, self.closed = config, [], False
            self.ready = {"model_sha256": "c" * 64,
                "adapter_sha256": "d" * 64 if config.adapter_dir else None}
            controllers.append(self)

        def start(self):
            pass

        def request(self, context, **kwargs):
            self.requests.append((context, kwargs))
            return {"status": "complete", "result": row["reference"]}

        def close(self):
            self.closed = True

    monkeypatch.setattr(specialization_worker, "WorkerController", Controller)
    spec = {"parameters": benchmark.parameters({}), "pins": {
        "mlx": {"model_dir": str(tmp_path / "model"), "model_manifest": str(tmp_path / "manifest.json"),
                "model_sha256": "c" * 64},
        "adapter": {"directory": str(tmp_path / "adapter"), "manifest": str(tmp_path / "adapter-manifest.json"),
                    "snapshot_sha256": "d" * 64}}}
    runner = benchmark.Runner(spec, Guard())
    runner.start("mlx_base")
    runner.infer(row, "e" * 64)
    runner.close()
    runner.start("mlx_adapter")
    runner.infer(row, "f" * 64)
    runner.close()
    base, adapted = controllers
    assert base.closed and adapted.closed
    assert base.config.adapter_dir is None and adapted.config.adapter_dir is not None
    a, b = base.requests[0][1], adapted.requests[0][1]
    assert {k: v for k, v in a.items() if k != "request_id"} == {k: v for k, v in b.items() if k != "request_id"}
    assert a["context_tokens"] == 4096 and a["image_tokens"] == 512
    assert a["max_tokens"] == 768 and a["thinking"] is False


def test_reused_engine_identity_mismatch_stops_before_image_inference(tmp_path, monkeypatch):
    import specialization_worker

    closed = []

    class Guard:
        def check(self, reserve=0):
            pass

        def remaining(self):
            return 1000

    class Controller:
        def __init__(self, config):
            self.ready = {"model_sha256": "changed", "adapter_sha256": None}

        def start(self):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(specialization_worker, "WorkerController", Controller)
    spec = {"parameters": benchmark.parameters({}), "pins": {"mlx": {
        "model_dir": str(tmp_path / "model"), "model_manifest": str(tmp_path / "manifest.json"),
        "model_sha256": "c" * 64}, "adapter": None}}
    with pytest.raises(benchmark.BenchmarkError, match="worker_identity_mismatch"):
        benchmark.Runner(spec, Guard()).start("mlx_base")
    assert closed == [True]
