"""Runtime exploration stays private, paired and separate from release evidence."""

import copy

import pytest
from test_specialization_benchmark import example, private_export
from test_specialization_benchmark import harness as harness

import specialization_benchmark as benchmark
import specialization_tuning as tuning
from private_io import write_json


def test_partition_preserves_transitive_episode_and_near_image_groups():
    rows = []
    for i in range(20):
        row = example(str(i), episode=f"episode-{i // 2}")
        row["image_sha256"] = f"{i:064x}"
        row["near_duplicate_group"] = f"group-{i // 2}"
        rows.append(row)
    split = tuning.partition(rows)
    assert len(split["exploration_ids"]) == 8 and len(split["confirmation_ids"]) == 12
    selected = set(split["exploration_ids"])
    for a in rows:
        for b in rows:
            if a["episode_id"] == b["episode_id"] or a["near_duplicate_group"] == b["near_duplicate_group"]:
                assert (a["id"] in selected) == (b["id"] in selected)
    assert tuning.partition(rows) == split


def test_unpartitionable_validation_does_not_claim_independent_confirmation():
    rows = [example(str(i)) for i in range(20)]
    split = tuning.partition(rows)
    assert len(split["exploration_ids"]) == 20 and not split["confirmation_ids"]
    assert "not_independent_confirmation" in split["scope"]


def test_exploration_cannot_read_test_or_freeze_candidate(harness):
    config, root, rows, calls = harness
    spec = benchmark.experiment_spec(config, root)
    spec.update(purpose="exploratory_tuning", comparison_variants=["mlx_base"], selected_validation_ids=[rows[0]["id"]])
    result = benchmark._run(config, root / "tuning/cell", 5000, "validation", spec,
                            variants=("mlx_base",), row_ids=[rows[0]["id"]])
    assert result["status"] == "complete" and result["exploratory_only"]
    assert calls == [("mlx_base", rows[0]["id"])]
    with pytest.raises(benchmark.BenchmarkError, match="invalid_exploratory_comparison"):
        benchmark._run(config, root, 5000, "test", spec, variants=("mlx_base",), row_ids=[rows[0]["id"]])
    with pytest.raises(benchmark.BenchmarkError, match="exploratory_results_cannot_freeze_candidate"):
        benchmark.freeze_candidate({**config, "exploratory_only": True}, root, candidate="mlx_base", reviewed_assessment_sha256="a" * 64)


def test_selected_subset_is_part_of_resume_identity(harness):
    config, root, rows, calls = harness
    spec = benchmark.experiment_spec(config, root)
    spec.update(purpose="exploratory_tuning", comparison_variants=["mlx_base"], selected_validation_ids=[rows[0]["id"]])
    folder = root / "tuning/cell"
    benchmark._run(config, folder, 5000, "validation", spec, variants=("mlx_base",), row_ids=[rows[0]["id"]])
    changed = {**spec, "selected_validation_ids": [rows[1]["id"]]}
    with pytest.raises(benchmark.BenchmarkError, match="resume_identity_mismatch"):
        benchmark._run(config, folder, 5000, "validation", changed, variants=("mlx_base",), row_ids=[rows[1]["id"]])


def test_daytime_tuning_and_confirmation_open_no_private_reference(tmp_path, monkeypatch):
    class Refuse:
        def __init__(self, budget):
            pass
        def check(self, *a):
            raise tuning.BudgetEnded("outside_vision_window")
    monkeypatch.setattr(tuning, "Guard", Refuse)
    monkeypatch.setattr(tuning, "load_examples", lambda *a: pytest.fail("private reference read"))
    assert tuning.run({}, tmp_path / "study", 1000)["private_examples_opened"] is False
    assert tuning.confirm({}, tmp_path / "study", 1000)["private_examples_opened"] is False
    assert not (tmp_path / "study").exists()


def test_runtime_nomination_cannot_authorize_test_without_confirmation(harness):
    config, root, rows, calls = harness
    with pytest.raises(benchmark.BenchmarkError, match="runtime_confirmation_review_required"):
        benchmark.freeze_candidate({**config, "tuning_required": True}, root, candidate="mlx_base", reviewed_assessment_sha256="a" * 64)


def test_runtime_settings_refuse_changed_choice_or_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "verify_choice", lambda *args: None)
    config = {"configuration_sha256": "a" * 64}
    with pytest.raises(benchmark.BenchmarkError, match="runtime_selection_required"):
        benchmark.runtime_settings({**config, "tuning_required": True}, tmp_path)
    body = {"parent_configuration_sha256": "a" * 64, "overrides": {"image_tokens": 1024}}
    body["runtime_choice_sha256"] = benchmark.fingerprint(body)
    write_json(tmp_path / "tuning/runtime-choice.json", body)
    settings, _ = benchmark.runtime_settings(config, tmp_path)
    assert settings["image_tokens"] == 1024
    body["overrides"]["image_tokens"] = 1536
    write_json(tmp_path / "tuning/runtime-choice.json", body)
    with pytest.raises(benchmark.BenchmarkError, match="runtime_choice_changed"):
        benchmark.runtime_settings(config, tmp_path)


def test_reducing_unclear_or_runtime_cannot_compensate_for_quality_errors():
    control = {"selected": 8, "complete": 8, "joint_correct": 6, "unsupported_claims": 0,
               "uncertainty_errors": 1, "critical": 0}
    faster_wrong = {**control, "joint_correct": 5}
    assert not tuning.quality_not_worse(faster_wrong, control)
    assert not tuning.quality_not_worse({**control, "unsupported_claims": 1}, control)
    assert not tuning.quality_not_worse({**control, "critical": 1}, control)
    assert not tuning.quality_not_worse({**control, "complete": 7}, control)


def test_counterbalanced_plan_reverses_conditions_without_new_semantic_samples():
    rows = [example(str(i), episode=f"episode-{i}") for i in range(20)]
    for i, row in enumerate(rows):
        row["image_sha256"] = f"{i:064x}"
    config = {"seed": 7, "configuration_sha256": "a" * 64, "exports": {"validation": {"sha256": "b" * 64}}}
    backend = {"backend_choice_sha256": "c" * 64}
    plan = tuning.make_plan(config, None, backend, rows)
    assert plan["rounds"][1] == list(reversed(plan["rounds"][0]))
    assert len(plan["exploration_ids"]) == 8
    assert set(plan["exploration_ids"]).isdisjoint(plan["confirmation_ids"])
    assert plan["selection_policy"]["minimum_wall_gain_both_rounds"] == .15
    changed = copy.deepcopy(plan)
    changed["exploration_ids"].reverse()
    assert benchmark.fingerprint({k: v for k, v in changed.items() if k != "plan_sha256"}) != plan["plan_sha256"]


def grade(path):
    assessment = benchmark.read_state(path)
    for item in assessment["items"].values():
        item["grades"] = {k: k not in {"privacy_leak", "unsupported_completion"} for k in benchmark.GRADES}
        item["claim_support"] = {k: "not_asserted" for k in tuning.CLAIMS}
    write_json(path, assessment)
    return benchmark.file_digest(path, private=True)


@pytest.fixture
def tuning_runtime(harness, monkeypatch):
    config, root, rows, calls = harness
    rows = [example(f"example-{i}", episode=f"episode-{i}") for i in range(20)]
    for i, row in enumerate(rows):
        row["image_sha256"] = f"{i:064x}"
    config["exports"]["validation"] = private_export(benchmark.DATA_ROOT, rows)
    config["configuration_sha256"] = "a" * 64
    original = benchmark.experiment_spec
    def spec(c, r, *, exploratory=False, nomination_only=False):
        s = original(c, r)
        settings, choice = benchmark.runtime_settings(c, r, exploratory=exploratory, nomination_only=nomination_only)
        s.update(parameters=benchmark.parameters(settings), q8_parameters=benchmark.q8_parameters(settings),
                 runtime_choice=choice, q8_production_parameters=benchmark.q8_parameters(c))
        return s
    monkeypatch.setattr(benchmark, "experiment_spec", spec)
    monkeypatch.setattr(tuning, "experiment_spec", spec)
    monkeypatch.setattr(tuning, "Guard", benchmark.Guard)
    monkeypatch.setattr(tuning, "DATA_ROOT", benchmark.DATA_ROOT)
    benchmark.run_validation(config, root, 5000)
    path = root / "benchmark/validation/blind-assessment.json"
    tuning.begin(config, root, "mlx_base", grade(path))
    return config, root, rows, calls


def test_sweep_preserves_original_study_and_full_confirmation_is_required(tuning_runtime, monkeypatch):
    config, root, rows, calls = tuning_runtime
    original = (root / "benchmark/validation/results.json").read_bytes()
    result = tuning.run(config, root, 5000)
    assert result["status"] == "complete"
    assert result["unique_exploration_examples"] == 8 and result["confirmation_examples"] == 12
    assert result["production_changed"] is False
    assert (root / "benchmark/validation/results.json").read_bytes() == original
    hashes = {}
    for repeat in range(2):
        for condition in tuning.CONDITIONS:
            folder = root / "tuning" / f"round-{repeat}" / condition / "benchmark/validation"
            hashes[f"{repeat}:{condition}"] = grade(folder / "blind-assessment.json")
    selected = tuning.freeze_choice(config, root, hashes)
    assert selected["status"] == "runtime_nominated" and selected["production_changed"] is False
    assert not (root / "benchmark/frozen-candidate.json").exists()
    confirmed = tuning.confirm(config, root, 5000)
    assert confirmed["status"] == "complete"
    final_review = grade(root / "confirmation/benchmark/validation/blind-assessment.json")
    decision = tuning.finalize_confirmation(config, root, final_review)
    assert decision["status"] == "runtime_confirmed"
    assert (root / "benchmark/validation/results.json").read_bytes() == original
    assert not (root / "benchmark/test").exists()
    frozen = benchmark.freeze_candidate(config, root, candidate="mlx_base", reviewed_assessment_sha256=final_review)
    assert frozen["status"] == "frozen"
    assert benchmark.read_state(root / "benchmark/frozen-candidate.json")["validation_location"] == "confirmation/benchmark/validation"


def test_preparation_crash_retains_tuning_reservation(tuning_runtime, monkeypatch):
    config, root, rows, calls = tuning_runtime
    original = tuning.load_examples
    monkeypatch.setattr(tuning, "load_examples", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        tuning.run(config, root, 5000)
    monkeypatch.setattr(tuning, "load_examples", original)
    result = tuning.run(config, root, 5000)
    assert result["total_session_seconds"] is None
    assert result["conservative_total_seconds"] >= 5030
    assert result["interrupted_sessions"] == 1


def test_completed_cells_keep_exact_costs_on_resume(tuning_runtime):
    config, root, rows, calls = tuning_runtime
    assert tuning.run(config, root, 5000)["status"] == "complete"
    cell = root / "tuning/round-0/mlx_control/benchmark/validation/results.json"
    original, attempts = cell.read_bytes(), len(calls)
    assert tuning.run(config, root, 5000)["status"] == "complete"
    assert len(calls) == attempts and cell.read_bytes() == original
    data = benchmark.read_state(cell)
    first = next(iter(data["results"].values()))["mlx_base"]
    first["identity"] = "f" * 64
    write_json(cell, data)
    with pytest.raises(benchmark.BenchmarkError, match="tuning_identity_changed"):
        tuning.run(config, root, 5000)


def test_split_resident_lifetimes_are_not_clean_trials():
    blocks = [{"stage": "startup", "variant": "mlx_base", "intended_requests": 8,
               "attempted_requests": 7, "completed_requests": 7},
              {"stage": "startup", "variant": "mlx_base", "intended_requests": 1,
               "attempted_requests": 1, "completed_requests": 1}]
    assert not tuning.clean_blocks({"sessions": blocks}, "mlx_base")
    blocks[0].update(attempted_requests=8, completed_requests=8)
    assert tuning.clean_blocks({"sessions": blocks}, "mlx_base")


def test_full_resident_block_reserves_per_request_sampler_margin(harness, monkeypatch):
    config, root, rows, calls = harness
    rows = [example(f"example-{i}", episode=f"episode-{i}") for i in range(8)]
    config["exports"]["validation"] = private_export(benchmark.DATA_ROOT, rows)
    config["batch_requests"] = 8
    class Guard:
        def __init__(self, budget):
            self.seconds = budget
        def check(self, reserve=0):
            if self.seconds <= reserve:
                raise benchmark.BudgetEnded("benchmark_budget")
        def remaining(self):
            return self.seconds
    monkeypatch.setattr(benchmark, "Guard", Guard)
    spec = benchmark.experiment_spec(config, root)
    ids = [row["id"] for row in rows]
    spec.update(purpose="exploratory_tuning", comparison_variants=["mlx_base"], selected_validation_ids=ids)
    old_reserve = 8 * spec["parameters"]["timeout_seconds"] + 336
    result = benchmark._run(config, root / "tuning/cell", old_reserve, "validation", spec,
                            variants=("mlx_base",), row_ids=ids)
    assert result["status"] == "partial" and result["variants"]["mlx_base"]["pending"] == 8
    assert not calls


def test_swapped_errors_cannot_cancel_individual_regressions():
    pair = {"complete": True, **{k: True for k in benchmark.GRADES[:4]}, "critical": False, "unsupported_claims": 0}
    baseline = {"selected": 2, "complete": 2, "joint_correct": 1, "unsupported_claims": 0,
                "uncertainty_errors": 0, "critical": 0,
                "by_example": {"a": {**pair, "task_correct": False}, "b": pair}}
    candidate = {**baseline, "by_example": {"a": pair, "b": {**pair, "task_correct": False}}}
    assert not tuning.quality_not_worse(candidate, baseline)


def nominate_and_confirm(config, root):
    assert tuning.run(config, root, 5000)["status"] == "complete"
    hashes = {f"{repeat}:{condition}": grade(root / "tuning" / f"round-{repeat}" / condition / "benchmark/validation/blind-assessment.json")
              for repeat in range(2) for condition in tuning.CONDITIONS}
    tuning.freeze_choice(config, root, hashes)
    assert tuning.confirm(config, root, 5000)["status"] == "complete"
    return root / "confirmation/benchmark/validation"


def test_safety_regression_on_exploration_row_rejects_confirmation(tuning_runtime):
    config, root, rows, calls = tuning_runtime
    folder = nominate_and_confirm(config, root)
    data = benchmark.read_state(folder / "results.json")
    identity = benchmark.read_state(root / "tuning/plan.json")["exploration_ids"][0]
    data["results"][identity]["mlx_base"]["raw_safety"]["privacy"] = True
    write_json(folder / "results.json", data)
    result = tuning.finalize_confirmation(config, root, grade(folder / "blind-assessment.json"))
    assert result["status"] == "reference_retained"
    assert benchmark.runtime_settings(config, root)[1] is None
    baseline_review = benchmark.file_digest(root / "benchmark/validation/blind-assessment.json", private=True)
    benchmark.freeze_candidate(config, root, candidate="mlx_base", reviewed_assessment_sha256=baseline_review)
    assert benchmark.read_state(root / "benchmark/frozen-candidate.json")["validation_location"] == "benchmark/validation"


def test_changed_tuning_review_invalidates_nomination(tuning_runtime):
    config, root, rows, calls = tuning_runtime
    nominate_and_confirm(config, root)
    path = root / "tuning/round-0/mlx_control/benchmark/validation/blind-assessment.json"
    changed = benchmark.read_state(path)
    next(iter(changed["items"].values()))["grades"]["task_correct"] = False
    write_json(path, changed)
    with pytest.raises(benchmark.BenchmarkError, match="tuning_proof_changed"):
        benchmark.runtime_settings(config, root)
