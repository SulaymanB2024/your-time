import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import specialization_training as training
from activity_context import CLAIMS, RESULT_VERSION, prompt


def private_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if isinstance(payload, bytes) else payload.encode())
    path.chmod(0o600)
    return path


def record(tmp_path, identity="example", synthetic=False):
    source = "synthetic_screen" if synthetic else "screen_context"
    context = {"version": "activity_context_v1", "evidence": [{"id": identity, "source": source}],
               "limits": "Synthetic test evidence only."}
    gold = {"version": RESULT_VERSION, "activity_kind": "coding", "project_candidate": "Example",
            "task_candidate": "Editing code", "visible_work": "Code editor visible",
            "evidence_ids": [identity], "claim_evidence": {k: [identity] for k in CLAIMS},
            "uncertainty": "supported"}
    image = private_file(tmp_path / f"{identity}.webp", b"synthetic bytes " + identity.encode())
    return {"id": identity, "images": [str(image)], "context": context,
            "messages": [{"role": "user", "content": prompt(context)},
                         {"role": "assistant", "content": json.dumps(gold)}],
            "target": gold, "synthetic": synthetic, "image_digest": training.file_digest(image)}


def export(tmp_path, rows, name="train.jsonl"):
    return private_file(tmp_path / name, "".join(json.dumps(row) + "\n" for row in rows))


def model_keys():
    return [f"language_model.model.layers.{layer}.{branch}.{suffix}"
            for layer in range(32)
            for branch, suffixes in [("mlp", ["gate_proj", "up_proj", "down_proj"]),
                                     ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]
                                      if (layer + 1) % 4 == 0 else [])]
            for suffix in suffixes]


def runtime_args(tmp_path, stage="train"):
    return [stage, "--model-path", str(tmp_path / "model"),
            "--model-manifest", str(tmp_path / "manifest.json"),
            "--train-jsonl", str(tmp_path / "train.jsonl"),
            "--validation-jsonl", str(tmp_path / "validation.jsonl"),
            "--run-dir", str(tmp_path / "specialization" / "run"), "--state-dir", str(tmp_path)]


def test_dry_run_does_not_read_inputs_or_import_mlx(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run touched runtime data or model")

    monkeypatch.setattr(training, "read_private", forbidden)
    monkeypatch.setattr(training, "MLXEngine", forbidden)
    assert training.main(["dry-run", "--model-path", "/does/not/exist"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recipe"]["image_tokens"] == 512
    assert result["recipe"]["learning_rate"] == 1e-5
    assert result["expected_projection_modules"] == 128
    assert not result["loads_model"] and not result["reads_dataset"]
    assert "mlx.core" not in sys.modules


def test_recipe_rejects_unreviewed_hyperparameters():
    with pytest.raises(training.TrainingError, match="recipe_changes"):
        training.Recipe(rank=16)
    with pytest.raises(training.TrainingError):
        training.Recipe(epochs=3)


def test_all_hybrid_targets_are_exact_and_no_recurrent_gates():
    keys = model_keys()
    assert len(keys) == 128
    training.assert_adapter_keys(keys)
    keys[0] = "language_model.model.layers.0.linear_attn.in_proj_a"
    with pytest.raises(training.TrainingError, match="unsafe_adapter_target"):
        training.assert_adapter_keys(keys)


def test_missing_or_duplicate_projection_is_rejected():
    keys = model_keys()
    with pytest.raises(training.TrainingError, match="target_count"):
        training.assert_adapter_keys(keys[:-1])
    keys[-1] = keys[0]
    with pytest.raises(training.TrainingError, match="target_count"):
        training.assert_adapter_keys(keys)


def test_same_projection_counts_in_wrong_decoder_layers_are_rejected():
    keys = model_keys()
    keys = [key.replace("layers.3.self_attn", "layers.2.self_attn") for key in keys]
    with pytest.raises(training.TrainingError, match="target_layout"):
        training.assert_adapter_keys(keys)


def test_completion_mask_covers_answer_eos_and_excludes_prompt_image_padding():
    ids, mask = training.completion_layout([10, 99, 99, 11], [12, 13], 14, 99)
    assert ids == [10, 99, 99, 11, 12, 13, 14]
    assert mask == [0, 0, 0, 0, 1, 1, 1]
    assert training.shifted_loss_mask([1] * len(ids) + [0, 0], mask + [0, 0]) == [0, 0, 0, 1, 1, 1, 0, 0]


@pytest.mark.parametrize("prefix,answer,limit", [([1, 2], [3], 10), ([99], [], 10),
                                                 ([99], [99], 10), ([99, 1], [2, 3], 4)])
def test_missing_image_empty_answer_or_truncation_fails(prefix, answer, limit):
    with pytest.raises(training.TrainingError):
        training.completion_layout(prefix, answer, 9, 99, limit)


def test_empty_mask_is_rejected():
    with pytest.raises(training.TrainingError, match="empty_assistant_loss"):
        training.shifted_loss_mask([1, 1, 1], [0, 0, 0])


def test_private_exports_validate_claim_mapping_and_prompt(tmp_path):
    row = record(tmp_path, synthetic=True)
    rows, digest = training.read_export(export(tmp_path, [row]), "train")
    assert len(digest) == 64 and rows[0]["target"] == row["target"]
    assert rows[0]["synthetic"]
    row["messages"][0]["content"] = "stale prompt"
    with pytest.raises(training.TrainingError, match="stale_context"):
        training.read_export(export(tmp_path, [row]), "train")


def test_validation_rejects_synthetic_or_test_split(tmp_path):
    row = record(tmp_path, synthetic=True)
    with pytest.raises(training.TrainingError, match="synthetic_validation"):
        training.read_export(export(tmp_path, [row], "validation.jsonl"), "validation")
    row["split"] = "test"
    with pytest.raises(training.TrainingError, match="wrong_export_split"):
        training.read_export(export(tmp_path, [row]), "train")
    with pytest.raises(training.TrainingError, match="sealed_test"):
        training.read_export(Path("/never/read"), "test")


def test_symlink_or_nonprivate_input_rejected(tmp_path):
    path = private_file(tmp_path / "input.jsonl", "")
    link = tmp_path / "link.jsonl"
    link.symlink_to(path)
    with pytest.raises(training.TrainingError):
        training.read_private(link)
    path.chmod(0o644)
    with pytest.raises(training.TrainingError, match="not_private"):
        training.read_private(path)


@pytest.mark.parametrize("field", ["id", "image_digest", "episode_id", "near_duplicate_group"])
def test_partition_overlap_is_rejected(field):
    left = {"id": "a", "image_digest": "ha", "synthetic": True}
    right = {"id": "b", "image_digest": "hb", "synthetic": False}
    left[field] = right[field] = "same"
    with pytest.raises(training.TrainingError, match="cross_split"):
        training.check_partitions([left, {"id": "c", "synthetic": True}], [right] * 20)


def test_partition_population_and_controls_are_required():
    validation = [{"id": f"v{i}", "synthetic": False} for i in range(20)]
    with pytest.raises(training.TrainingError, match="two_synthetic"):
        training.check_partitions([{"id": "train", "synthetic": False}], validation)
    with pytest.raises(training.TrainingError, match="20_real"):
        training.check_partitions([], validation[:19])


def test_generation_scores_include_schema_abstention_evidence_and_completion(tmp_path):
    row = record(tmp_path)
    rows = [row] * 3
    bad_mapping = {**row["target"], "claim_evidence": {k: [] for k in CLAIMS}}
    outputs = [json.dumps(row["target"]), json.dumps(bad_mapping), "I completed and saved the work."]
    result = training.score_generations(rows, outputs)
    assert result["schema_valid"] == result["joint_candidates_correct"] == 1
    assert result["claim_evidence_exact"] == 1
    assert result["completion_claim_flags"] == 1
    assert result["correct_affirmative_candidates"] == 2


def test_checkpoint_selection_keeps_base_on_tie_and_rejects_safety_regression(tmp_path):
    row = record(tmp_path)
    base = training.score_generations([row], [json.dumps(row["target"])])
    assert not training.select_candidate(base, base, base)
    candidate = {**base, "completion_claim_flags": 1, "activity_correct": 2}
    assert not training.select_candidate(base, base, candidate)
    invalid = {**base, "schema_valid": 0, "joint_candidates_correct": 2}
    assert not training.select_candidate(base, base, invalid)
    with pytest.raises(training.TrainingError, match="population_changed"):
        training.select_candidate(base, base, {**base, "examples": 2})


@pytest.mark.parametrize("local,expected", [("2026-10-05T00:29:59", 0),
                                            ("2026-10-05T00:30:00", 2700),
                                            ("2026-10-05T06:58:00", 60),
                                            ("2026-10-05T07:00:00", 0),
                                            ("2026-10-05T17:00:00", 0)])
def test_clock_gate_preserves_synthesis_hour(local, expected):
    now = datetime.fromisoformat(local).replace(tzinfo=ZoneInfo("America/Chicago"))
    assert training.overnight_budget(2700, now) == expected


def test_daytime_stage_never_reads_data_or_loads_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(training, "overnight_budget", lambda *args: 0)
    monkeypatch.setattr(training, "worker", lambda *args: pytest.fail("daytime worker started"))
    assert training.main(runtime_args(tmp_path)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "outside_training_window"


def test_unlocked_direct_worker_cannot_load_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(training, "overnight_budget", lambda *args: 300)
    monkeypatch.setenv("YOUR_TIME_ML_WORKER", "1")
    monkeypatch.setattr(training, "worker", lambda *args: pytest.fail("unlocked worker started"))
    assert training.main(runtime_args(tmp_path) + ["--worker"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "worker_requires_shared_runner_lock"


def test_worker_requires_inherited_actual_lock(tmp_path):
    path = private_file(tmp_path / "local-model-execution.lock", "")
    fd = os.open(path, os.O_RDWR)
    try:
        assert not training.inherited_lock_present(tmp_path)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert training.inherited_lock_present(tmp_path)
    finally:
        os.close(fd)
    assert not training.inherited_lock_present(tmp_path)


def test_parent_routes_sandbox_offline_and_shared_runner(tmp_path, monkeypatch, capsys):
    import model_execution
    import vision_batch

    seen = {}

    def runner(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, b'{"status":"verified"}', b"")

    monkeypatch.setattr(training, "overnight_budget", lambda *args: 300)
    monkeypatch.setattr(vision_batch, "resource_gate", lambda **kwargs: None)
    monkeypatch.setattr(model_execution, "run_model", runner)
    assert training.main(runtime_args(tmp_path, "verify")) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert seen["command"][0] == "/usr/bin/sandbox-exec"
    assert seen["command"][-1] == "--worker"
    assert seen["state_dir"] == tmp_path and seen["timeout"] == 300
    assert all(seen["env"][k] == v for k, v in training.OFFLINE.items())


def test_crash_output_and_exception_details_do_not_escape(tmp_path, monkeypatch, capsys):
    import model_execution
    import vision_batch

    monkeypatch.setattr(training, "overnight_budget", lambda *args: 300)
    monkeypatch.setattr(vision_batch, "resource_gate", lambda **kwargs: None)
    monkeypatch.setattr(model_execution, "run_model", lambda command, **kwargs:
        subprocess.CompletedProcess(command, 1, b"PRIVATE_CANARY", b"PRIVATE_CANARY"))
    assert training.main(runtime_args(tmp_path)) == 1
    assert "PRIVATE_CANARY" not in capsys.readouterr().out
    monkeypatch.setattr(training, "overnight_budget", lambda *args: (_ for _ in ()).throw(ValueError("PRIVATE_CANARY")))
    assert training.main(runtime_args(tmp_path)) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "ValueError"


class FakeEngine:
    frozen_digest = "frozen"

    def __init__(self, scores):
        self.scores = scores
        self.gradients = []
        self.update_counts = []
        self.weight = 0

    def gradient(self, row):
        self.gradients.append(row["id"])
        return 1, 1.0

    def add_gradients(self, previous, value):
        return (previous or 0) + value

    def update(self, value, count):
        assert value == count
        self.update_counts.append(count)
        self.weight += 1

    def save(self, folder):
        training.write_state(folder / "mock.json", {"weight": self.weight})

    def restore(self, folder):
        self.weight = training.read_state(folder / "mock.json")["weight"]

    def validate(self, rows):
        return {**self.scores, "joint_candidates_correct": self.weight}

    def frozen_hash(self):
        return self.frozen_digest


class Allow:
    def check(self, *args):
        pass


def initial_progress(scores):
    return {"experiment_sha256": "experiment", "epoch": 0, "offset": 0,
            "microsteps": 0, "updates": 0, "frozen_sha256": "frozen",
            "baseline": scores, "epoch_scores": {}, "best": {"checkpoint": None, "scores": scores}}


def test_custom_loop_accumulates_eight_flushes_tail_and_selects_epoch_checkpoints(tmp_path):
    scores = {"examples": 20, "schema_valid": 20, "completion_claim_flags": 0,
              "joint_candidates_correct": 0, "activity_correct": 0,
              "project_correct": 0, "task_correct": 0, "claim_evidence_exact": 0}
    engine, recipe = FakeEngine(scores), training.Recipe()
    progress = training.save_checkpoint(engine, tmp_path, initial_progress(scores))
    result = training.train_epochs(engine, [{"id": str(i)} for i in range(9)],
                                   [{}] * 20, tmp_path, progress, recipe, Allow())
    assert result["status"] == "complete" and result["microsteps"] == 18
    assert result["epoch"] == 2 and result["updates"] == 4
    assert engine.update_counts == [8, 1, 8, 1]
    assert len(result["epoch_scores"]) == 2
    assert result["best"]["checkpoint"] == result["checkpoint"]
    assert result["best"]["scores"]["joint_candidates_correct"] == 4
    for epoch in range(2):
        assert sorted(engine.gradients[epoch * 9:(epoch + 1) * 9]) == [str(i) for i in range(9)]
    resumed = FakeEngine(scores)
    committed = training.read_state(tmp_path / "progress.json")
    training.restore_checkpoint(resumed, tmp_path, committed, "experiment")
    assert resumed.weight == 4


def test_generated_optimizer_history_keeps_resume_and_best_states(tmp_path):
    class Engine:
        def save(self, folder):
            from private_io import atomic_write

            atomic_write(folder / "optimizer.safetensors", b"generated optimizer")
            atomic_write(folder / "adapter.safetensors", b"adapter history")
    progress = {"updates": 0, "best": {"checkpoint": None}}
    first = training.save_checkpoint(Engine(), tmp_path, progress)
    progress = {**first, "best": {"checkpoint": first["checkpoint"]}}
    for index in range(1, 5):
        progress = training.save_checkpoint(Engine(), tmp_path, {**progress, "updates": index})
    snapshots = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(snapshots) == 5
    assert sum((p / "optimizer.safetensors").exists() for p in snapshots) == 3
    assert all((p / "adapter.safetensors").exists() for p in snapshots)
    assert (tmp_path / first["checkpoint"] / "optimizer.safetensors").exists()
    assert (tmp_path / progress["checkpoint"] / "optimizer.safetensors").exists()


def test_interrupted_accumulation_resumes_only_the_committed_boundary(tmp_path):
    scores = {"examples": 20, "schema_valid": 20, "completion_claim_flags": 0,
              "joint_candidates_correct": 0, "activity_correct": 0,
              "project_correct": 0, "task_correct": 0, "claim_evidence_exact": 0}
    engine = FakeEngine(scores)
    progress = training.save_checkpoint(engine, tmp_path, initial_progress(scores))

    class StopAfterTen:
        calls = 0

        def check(self, *args):
            self.calls += 1
            if self.calls > 10:
                raise training.BudgetEnded("test_stop")

    rows = [{"id": str(i)} for i in range(12)]
    with pytest.raises(training.BudgetEnded):
        training.train_epochs(engine, rows, [{}] * 20, tmp_path, progress,
                              training.Recipe(), StopAfterTen())
    committed = training.read_state(tmp_path / "progress.json")
    assert committed["microsteps"] == 8 and committed["offset"] == 8
    resumed = FakeEngine(scores)
    training.restore_checkpoint(resumed, tmp_path, committed, "experiment")
    finished = training.train_epochs(resumed, rows, [{}] * 20, tmp_path, committed,
                                      training.Recipe(), Allow())
    assert finished["microsteps"] == 24 and finished["updates"] == 4
    assert resumed.gradients[:4] == [rows[i]["id"] for i in training.epoch_order(12, 20261005, 0)[8:]]


def test_resume_rejects_changed_experiment_or_pointer_metadata(tmp_path):
    engine = FakeEngine({})
    progress = training.save_checkpoint(engine, tmp_path, initial_progress({}))
    with pytest.raises(training.TrainingError, match="experiment_mismatch"):
        training.restore_checkpoint(engine, tmp_path, progress, "changed")
    altered = {**progress, "offset": 1}
    with pytest.raises(training.TrainingError, match="progress_mismatch"):
        training.restore_checkpoint(engine, tmp_path, altered, "experiment")


def test_failed_checkpoint_does_not_advance_pointer(tmp_path):
    engine = FakeEngine({})
    original = training.save_checkpoint(engine, tmp_path, initial_progress({}))

    def fail(folder):
        raise OSError("synthetic failure")

    engine.save = fail
    with pytest.raises(OSError):
        training.save_checkpoint(engine, tmp_path, {**original, "updates": 1})
    assert training.read_state(tmp_path / "progress.json") == original


def test_adapter_closure_uses_external_manifest_and_detects_changes(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir(mode=0o700)
    private_file(adapter / "adapter_config.json", "{}")
    private_file(adapter / "adapters.safetensors", "synthetic weights")
    manifest = {"version": "mlx_adapter_assets_v1", "model_revision": training.MODEL_REVISION,
                "mlx_vlm": training.MLX_VLM_VERSION, "base_model_sha256": "a" * 64,
                "files": [{"name": p.name, "size": p.stat().st_size, "sha256": training.file_digest(p)}
                          for p in sorted(adapter.iterdir())]}
    manifest["manifest_sha256"] = training.fingerprint(manifest)
    path = private_file(tmp_path / "adapter-manifest.json", json.dumps(manifest))
    training.verify_adapter_assets(adapter, path)
    training.verify_adapter_assets(adapter, path, "a" * 64)
    with pytest.raises(training.TrainingError, match="base_model_mismatch"):
        training.verify_adapter_assets(adapter, path, "b" * 64)
    private_file(adapter / "adapters.safetensors", "tampered weights")
    with pytest.raises(training.TrainingError, match="asset_hash_mismatch"):
        training.verify_adapter_assets(adapter, path)
    with pytest.raises(training.TrainingError, match="manifest_must_be_external"):
        training.verify_adapter_assets(adapter, adapter / "manifest.json")


def test_model_manifest_is_external_and_complete(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    config = {"model_type": "qwen3_5", "quantization": {"bits": 4},
              "text_config": {"num_hidden_layers": 32, "hidden_size": 4096,
                              "full_attention_interval": 4,
                              "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8}}
    names = {"config.json", "tokenizer.json", "chat_template.jinja", "weights.safetensors"}
    monkeypatch.setattr(training, "MODEL_FILES", names)
    monkeypatch.setattr(training, "PINNED_WEIGHTS", {})
    for name in names:
        private_file(model / name, json.dumps(config) if name == "config.json" else "synthetic")
    manifest = {"version": "mlx_assets_v1", "repository": training.MODEL_REPOSITORY,
                "revision": training.MODEL_REVISION,
                "files": [{"name": p.name, "size": p.stat().st_size, "sha256": training.file_digest(p)}
                          for p in sorted(model.iterdir())]}
    manifest["manifest_sha256"] = __import__("hashlib").sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = private_file(tmp_path / "model-manifest.json", json.dumps(manifest))
    assert training.verify_assets(model, path, Allow()) == manifest["manifest_sha256"]
    private_file(model / "unexpected.py", "synthetic")
    with pytest.raises(training.TrainingError, match="closure_mismatch"):
        training.verify_assets(model, path, Allow())
    with pytest.raises(training.TrainingError, match="manifest_must_be_external"):
        training.verify_assets(model, model / "manifest.json", Allow())


def test_epoch_shuffle_is_reproducible_and_changes_between_epochs():
    first = training.epoch_order(120, 20261005, 0)
    assert first == training.epoch_order(120, 20261005, 0)
    assert first != training.epoch_order(120, 20261005, 1)
    assert sorted(first) == list(range(120))


def test_kernel_deadline_terminates_a_stalled_native_call():
    source = ("import ctypes\nfrom specialization_training import kernel_deadline\n"
              "with kernel_deadline(.15):\n    ctypes.CDLL(None).sleep(3)\n")
    result = subprocess.run([sys.executable, "-B", "-c", source], capture_output=True,
                            timeout=5, cwd=Path(training.__file__).parent)
    assert result.returncode == -signal.SIGALRM


def test_adapter_base_identity_matches_resident_worker():
    from specialization_worker import snapshot_identity

    entries = [{"name": "b", "size": 2, "sha256": "a" * 64},
               {"name": "a", "size": 1, "sha256": "b" * 64}]
    assert training.snapshot_identity(entries) == snapshot_identity(entries)


def test_resume_cursor_rejects_uncommitted_or_overrun_steps(tmp_path):
    scores = {"examples": 20, "schema_valid": 20, "completion_claim_flags": 0,
              "joint_candidates_correct": 0, "activity_correct": 0,
              "project_correct": 0, "task_correct": 0, "claim_evidence_exact": 0}
    engine = FakeEngine(scores)
    progress = {**initial_progress(scores), "offset": 1, "microsteps": 1}
    with pytest.raises(training.TrainingError, match="invalid_committed"):
        training.train_epochs(engine, [{"id": str(i)} for i in range(9)], [{}] * 20,
                              tmp_path, progress, training.Recipe(), Allow())


def test_direct_loss_uses_shifted_assistant_and_padding_mask():
    np = pytest.importorskip("numpy")
    engine = training.MLXEngine.__new__(training.MLXEngine)

    class Core:
        float32 = np.float32
        int32 = np.int32
        array = staticmethod(np.array)
        take = staticmethod(np.take)

    class NN:
        class losses:
            @staticmethod
            def cross_entropy(logits, labels):
                # A controlled tokenwise loss: prompt losses must disappear.
                assert logits.shape == (1, 2, 10)
                assert labels.tolist() == [[3, 4]]
                return np.asarray([[2, 4]], dtype=np.float32)

    class Model:
        def __call__(self, ids, pixels, mask, **kwargs):
            assert ids.shape[0] == 1 and kwargs["image_grid_thw"].shape == (1, 3)
            assert pixels is not None
            return type("Output", (), {"logits": np.zeros((1, 5, 10), dtype=np.float32)})()

    engine.mx, engine.nn, engine.np = Core(), NN(), np
    batch = {"input_ids": np.asarray([[1, 9, 2, 3, 4, 0]]),
             "attention_mask": np.asarray([[1, 1, 1, 1, 1, 0]]),
             "completion_mask": np.asarray([[0, 0, 0, 1, 1, 1]]),
             "pixel_values": np.ones((4, 4)), "image_grid_thw": np.asarray([[1, 2, 2]])}
    assert float(engine.loss(Model(), batch)) == 3.0
    broken = copy.deepcopy(batch)
    broken["input_ids"] = np.zeros((1, 7), dtype=int)
    with pytest.raises(training.TrainingError, match="alignment_mismatch"):
        engine.loss(Model(), broken)


def test_validation_generations_keep_per_attempt_decode_history(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import inference_telemetry

    monkeypatch.setattr(inference_telemetry, 'read_command', lambda command: None)
    engine = training.MLXEngine.__new__(training.MLXEngine)
    engine.guard = SimpleNamespace(check=lambda *args: None)
    engine.model = SimpleNamespace(eval=lambda: None)
    engine.template = lambda *args, **kwargs: 'Synthetic validation prompt'
    engine.processor, engine.config = None, {}
    engine.mx = SimpleNamespace(clear_cache=lambda: None, get_active_memory=lambda: 1000,
                                get_cache_memory=lambda: 100, get_peak_memory=lambda: 1200)
    engine.recipe = training.Recipe()
    engine.telemetry_state_dir = tmp_path
    engine.telemetry_experiment = 'e'*64
    engine.base_model_sha256 = 'a'*64
    engine.validation_adapter_sha256 = 'b'*64
    engine.generate_fn = lambda *args, **kwargs: SimpleNamespace(
        text='invalid structured response', prompt_tokens=100, generation_tokens=20, generation_tps=5)
    row = dict(messages=[dict(role='user', content='synthetic'), dict(role='assistant', content='{}')],
               images=['synthetic.png'], image_digest='c'*64, context={'evidence': []})
    assert engine.generate(row) == 'invalid structured response'
    engine.generate(row)
    records = [json.loads(p.read_text()) for p in (tmp_path/'inference-attempts').glob('*.json')]
    assert len(records) == 2
    assert all(r['status'] == 'complete' and r['result_status'] == 'schema_error' for r in records)
    assert all(r['context']['adapter_sha256'] == 'b'*64 for r in records)
    assert 'Synthetic validation prompt' not in json.dumps(records)
    assert 'invalid structured response' not in json.dumps(records)
