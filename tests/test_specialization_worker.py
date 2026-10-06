"""Synthetic protocol/state tests. No model import or private evidence access."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import specialization_worker as worker
from activity_context import RESULT_VERSION


def context():
    return {"version": "activity_context_v1", "evidence": [
        {"id": "synthetic-a", "source": "synthetic_screen", "app": "Editor"}]}


def result():
    return {"version": RESULT_VERSION, "activity_kind": "coding",
            "project_candidate": None, "task_candidate": None,
            "visible_work": "An editor shows source code.",
            "evidence_ids": ["synthetic-a"], "uncertainty": "partial",
            "claim_evidence": {"activity_kind": ["synthetic-a"],
                               "project_candidate": [], "task_candidate": [],
                               "visible_work": ["synthetic-a"]}}


def request(**updates):
    value = {"version": worker.PROTOCOL, "id": "a", "context": context(),
             "image_path": None, "image_sha256": None, "thinking": False,
             "thinking_budget": 256, "max_tokens": 768, "timeout_seconds": 5,
             "image_side": 1600, "image_tokens": 1024, "seed": 3, "context_tokens": 4096}
    return dict(value, **updates)


@pytest.fixture
def config(tmp_path):
    root = tmp_path.resolve()
    return worker.WorkerConfig(str(root / "model"), str(root / "manifest.json"),
                               str(root / "state"), (str(root / "images"),))


class Channel:
    def __init__(self, incoming):
        self.incoming = iter(incoming)
        self.sent = []

    def receive(self, deadline):
        return next(self.incoming, None)

    def send(self, value, deadline):
        self.sent.append(value)


class FakeBackend:
    model_sha = "a" * 64
    adapter_sha = None
    load_seconds = 0.01

    def __init__(self, *args):
        self.calls = []

    def generate(self, value, *args):
        self.calls.append(value)
        output = json.dumps(result())
        if value["thinking"]:
            output = "<think>synthetic reasoning</think>" + output
        return output, {"prompt_tokens": 100, "generated_tokens": 30}


def test_gates_precede_backend_initialization(config):
    def forbidden(*args):
        raise AssertionError("Model must not initialize")
    with pytest.raises(worker.WorkerError, match="outside_vision_window"):
        worker.serve(config, Channel([]), time.monotonic() + 5,
                     backend_factory=forbidden, admission=lambda: "outside_vision_window")
    with pytest.raises(worker.WorkerError, match="deadline"):
        worker.serve(config, Channel([]), time.monotonic() - 1,
                     backend_factory=forbidden, admission=lambda: None)


@pytest.mark.parametrize("thinking", [False, True])
def test_direct_and_thinking_results_are_strictly_validated(config, thinking):
    backend = FakeBackend()
    response = worker.handle_request(backend, request(thinking=thinking), config,
                                     time.monotonic() + 5, lambda: None)
    assert response["result"] == result()
    assert "synthetic reasoning" not in json.dumps(response)


def test_rejects_wrong_citations_and_incomplete_thinking(config):
    backend = FakeBackend()
    backend.generate = lambda *args: ("<think>unfinished", {})
    with pytest.raises(ValueError, match="incomplete_thinking"):
        worker.handle_request(backend, request(), config, time.monotonic() + 5, lambda: None)
    bad = result()
    bad["evidence_ids"] = ["unavailable"]
    backend.generate = lambda *args: (json.dumps(bad), {})
    with pytest.raises(ValueError):
        worker.handle_request(backend, request(), config, time.monotonic() + 5, lambda: None)


@pytest.mark.parametrize("change", [
    {"timeout_seconds": 301}, {"max_tokens": 99999}, {"thinking": 1},
    {"thinking_budget": 768}, {"image_side": 4096}, {"image_tokens": 8192},
    {"image_path": "https://example.invalid/image", "image_sha256": "a" * 64},
    {"seed": float("nan")}, {"context": {"evidence": []}},
])
def test_request_bounds_fail_closed(change):
    with pytest.raises(worker.WorkerError):
        worker.validate_request(request(**change))


def test_context_byte_limit():
    value = request()
    value["context"]["evidence"][0]["app"] = "x" * worker.MAX_FRAME
    with pytest.raises(worker.WorkerError, match="frame_too_large"):
        worker.validate_request(value)


def test_service_is_serial_bounded_and_rejects_duplicate_ids(config):
    channel = Channel([request(), request()])
    with pytest.raises(worker.WorkerError, match="duplicate_request"):
        worker.serve(config, channel, time.monotonic() + 5, FakeBackend, lambda: None)
    assert [r["status"] for r in channel.sent] == ["ready", "complete"]
    channel = Channel([request(), request(id="b")])
    worker.serve(replace(config, max_requests=1), channel, time.monotonic() + 5,
                 FakeBackend, lambda: None)
    assert len(channel.sent) == 2


def test_model_failure_terminates_service_without_diagnostics(config):
    class Failure(FakeBackend):
        def generate(self, *args):
            raise RuntimeError("private prompt or model output")
    channel = Channel([request(), request(id="b")])
    worker.serve(config, channel, time.monotonic() + 5, Failure, lambda: None)
    assert channel.sent[-1]["code"] == "inference_error"
    assert "private" not in json.dumps(channel.sent)


def test_gate_rechecked_after_loading_and_before_each_request(config):
    checks = iter([None, None, "battery_power"])
    channel = Channel([request()])
    with pytest.raises(worker.WorkerError, match="battery_power"):
        worker.serve(config, channel, time.monotonic() + 5, FakeBackend, lambda: next(checks))
    assert [r["status"] for r in channel.sent] == ["ready"]


def test_snapshot_hashes_and_complete_runtime_closure(tmp_path):
    root = tmp_path.resolve()
    model = root / "model"
    model.mkdir(mode=0o700)
    path = model / "config.json"
    path.write_bytes(b"{}")
    path.chmod(0o600)
    entries = [{"name": path.name, "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]
    manifest = root / "pin.json"
    manifest.write_text(json.dumps({"files": entries}))
    manifest.chmod(0o600)
    assert worker.verify_snapshot(model, manifest, time.monotonic() + 5, lambda: None) == worker.snapshot_identity(entries)
    extra = model / "chat_template.jinja"
    extra.write_text("unverified")
    extra.chmod(0o600)
    with pytest.raises(worker.WorkerError, match="unpinned_runtime_file"):
        worker.verify_snapshot(model, manifest, time.monotonic() + 5, lambda: None)
    extra.unlink()
    path.write_bytes(b"[]")
    with pytest.raises(worker.WorkerError, match="snapshot_hash_mismatch"):
        worker.verify_snapshot(model, manifest, time.monotonic() + 5, lambda: None)


def test_private_image_paths_reject_symlinks_and_nonprivate_files(tmp_path):
    path = tmp_path.resolve() / "source.png"
    path.write_bytes(b"synthetic")
    path.chmod(0o600)
    link = path.with_name("link.png")
    link.symlink_to(path)
    with pytest.raises(worker.WorkerError, match="invalid_path"):
        worker.private_path(link)
    path.chmod(0o644)
    with pytest.raises(worker.WorkerError, match="nonprivate_path"):
        worker.private_path(path)


def test_exact_adapter_keys_rank_shapes_and_targets():
    key = "language_model.model.layers.0.self_attn.q_proj"
    config = {"fine_tune_type": "lora", "lora_parameters": {
        "keys": [key], "rank": 8, "scale": 2.0, "dropout": 0.0}}
    weights = {key + ".lora_a": SimpleNamespace(shape=(64, 8)),
               key + ".lora_b": SimpleNamespace(shape=(8, 32))}
    worker.validate_adapter(config, weights, {key: (64, 32)})
    with pytest.raises(worker.WorkerError, match="adapter_key_mismatch"):
        worker.validate_adapter(config, dict(weights, ignored=SimpleNamespace(shape=(1,))), {key: (64, 32)})
    with pytest.raises(worker.WorkerError, match="adapter_shape_mismatch"):
        worker.validate_adapter(config, {k: SimpleNamespace(shape=(1, 1)) for k in weights}, {key: (64, 32)})
    with pytest.raises(worker.WorkerError, match="adapter_target_missing"):
        worker.validate_adapter(config, weights, {})


def mocked_mlx_backend():
    backend = worker.MlxBackend.__new__(worker.MlxBackend)
    lm = SimpleNamespace(_position_ids="stale", _rope_deltas="stale")
    model = SimpleNamespace(language_model=lm, config=SimpleNamespace(eos_token_id=[1]))
    stopping = SimpleNamespace(reset=lambda ids: None)
    tokenizer = SimpleNamespace(stopping_criteria=stopping, thinking_budget_criteria="stale")
    backend.processor = SimpleNamespace(tokenizer=tokenizer,
                                        image_processor=SimpleNamespace(patch_size=16, merge_size=2))
    seeds, caches, calls = [], [], []
    backend.model = model
    backend.config = {"model_type": "qwen3_5", "image_token_id": 2}
    backend.mx = SimpleNamespace(synchronize=lambda: None, clear_cache=lambda: None,
                                 reset_peak_memory=lambda: None, random=SimpleNamespace(seed=seeds.append),
                                 get_active_memory=lambda: 1000, get_cache_memory=lambda: 500,
                                 get_peak_memory=lambda: 2000)
    def make_cache(model):
        cache = [SimpleNamespace(state=None)]
        caches.append(cache)
        return cache
    backend.make_cache = make_cache
    backend.template = lambda *args, **kwargs: json.dumps(kwargs)
    backend.prepare_inputs = lambda *args, **kwargs: {"input_ids": SimpleNamespace(size=100), "attention_mask": None}
    def stream(*args, **kwargs):
        assert lm._position_ids is None and lm._rope_deltas is None
        assert tokenizer.thinking_budget_criteria is None
        assert kwargs["prompt_cache"][0].state is None
        kwargs["prompt_cache"][0].state = "synthetic recurrent state"
        lm._position_ids, lm._rope_deltas = "changed", "changed"
        tokenizer.thinking_budget_criteria = "changed"
        calls.append(kwargs)
        yield SimpleNamespace(text=json.dumps(result()), finish_reason="stop", prompt_tokens=100, generation_tokens=30)
    backend.dispatch = SimpleNamespace(stream_generate=stream)
    return backend, calls, caches, seeds


def test_independent_requests_reset_recurrent_position_sampling_and_caches(config):
    backend, calls, caches, seeds = mocked_mlx_backend()
    for thinking in (True, False, True):
        worker.handle_request(backend, request(thinking=thinking), config,
                              time.monotonic() + 5, lambda: None)
    assert [c["enable_thinking"] for c in calls] == [True, False, True]
    assert [c["thinking_budget"] for c in calls] == [256, None, 256]
    assert len({id(c["prompt_cache"]) for c in calls}) == 3
    assert seeds == [3] * 6
    assert backend.model.language_model._rope_deltas is None
    assert backend.processor.tokenizer.thinking_budget_criteria is None
    assert all(not c["prompt_cache"] for c in calls)
    assert len(caches) == 6


def test_engine_wall_metrics_include_both_state_resets(config, monkeypatch):
    backend, _, _, _ = mocked_mlx_backend()
    clock = [0.0]
    original = backend.reset
    def reset(seed):
        clock[0] += 2
        return original(seed)
    def now():
        clock[0] += .01
        return clock[0]
    backend.reset = reset
    monkeypatch.setattr(worker.time, "monotonic", now)
    _, metrics = backend.generate(request(), config, 100, lambda: None)
    assert metrics["reset_prepare_seconds"] >= 2
    assert metrics["cleanup_seconds"] >= 2
    assert metrics["request_total_seconds"] >= 4
    assert metrics["time_to_first_token_seconds"] >= 2


def test_crop_preserves_original_evidence_and_records_actual_preprocessor_geometry(config):
    import numpy as np
    from PIL import Image

    original = context()
    viewed = worker.context_for_view(original, "center_80")
    assert "image_view" not in original["evidence"][0]
    assert "full_observation" in viewed["evidence"][0]["image_view"]["scope"]
    path = Path(config.image_roots[0]) / "synthetic.png"
    path.parent.mkdir(mode=0o700, parents=True)
    with Image.new("RGB", (1000, 600), "white") as image:
        image.save(path)
    path.chmod(0o600)
    backend, _, _, _ = mocked_mlx_backend()
    sizes = []
    def prepare(*args, **kwargs):
        sizes.append(kwargs["images"][0].size)
        return {"input_ids": np.array([[2] * 8]), "attention_mask": None, "image_grid_thw": np.array([[1, 4, 8]])}
    backend.prepare_inputs = prepare
    _, metrics = backend.generate(request(image_path=str(path), image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                          image_crop="center_80"), config, time.monotonic() + 5, lambda: None)
    assert sizes == [(800, 480)]
    assert metrics["vision_tokens"] == 8
    assert metrics["image_prepared_width"] == 128 and metrics["image_prepared_height"] == 64
    with Image.open(path) as source:
        assert source.size == (1000, 600)  # Original private image was never edited.


def test_allocator_metrics_are_separate_from_engine_timings(config):
    backend, _, _, _ = mocked_mlx_backend()
    response = worker.handle_request(backend, request(), config, time.monotonic() + 5, lambda: None)
    assert response["memory_metrics"] == {
        "scope": "mlx_allocator", "active_bytes": 1000, "cache_bytes": 500, "peak_bytes": 2000}
    assert not any(key.startswith("mlx_") for key in response["engine_metrics"])


def test_truncation_resets_before_reuse(config):
    backend, _, _, _ = mocked_mlx_backend()
    def truncated(*args, **kwargs):
        backend.model.language_model._position_ids = "changed"
        yield SimpleNamespace(text="<think>", finish_reason="length")
    backend.dispatch.stream_generate = truncated
    with pytest.raises(worker.WorkerError, match="output_truncated"):
        backend.generate(request(), config, time.monotonic() + 5, lambda: None)
    assert backend.model.language_model._position_ids is None


def test_expanded_prompt_and_output_budget_checked_before_model_step(config):
    backend, calls, _, _ = mocked_mlx_backend()
    backend.prepare_inputs = lambda *args, **kwargs: {"input_ids": SimpleNamespace(size=1400)}
    with pytest.raises(worker.WorkerError, match="context_budget"):
        backend.generate(request(context_tokens=2048, image_tokens=512), config,
                         time.monotonic() + 5, lambda: None)
    assert calls == []


def test_json_pipe_fragmentation_eof_invalid_json_and_timeout():
    read_fd, write_fd = os.pipe()
    reply_r, reply_w = os.pipe()
    channel = worker.JsonPipe(read_fd, reply_w)
    try:
        os.write(write_fd, b'{"x":')
        os.write(write_fd, b'1}\n{"x":2}\n')
        assert channel.receive(time.monotonic() + 1) == {"x": 1}
        assert channel.receive(time.monotonic() + 1) == {"x": 2}
        with pytest.raises(worker.WorkerError, match="deadline"):
            channel.receive(time.monotonic() + 0.01)
        os.write(write_fd, b'{"x":NaN}\n')
        with pytest.raises(worker.WorkerError, match="invalid_json"):
            channel.receive(time.monotonic() + 1)
        os.close(write_fd)
        write_fd = None
        assert channel.receive(time.monotonic() + 1) is None
    finally:
        for fd in (read_fd, write_fd, reply_r, reply_w):
            if fd is not None:
                os.close(fd)


@pytest.fixture
def fake_child_controller(config, monkeypatch):
    monkeypatch.setattr(worker, "gate", lambda: None)
    monkeypatch.setattr(worker, "remaining_seconds", lambda *args, **kwargs: 600)
    code = '''import sys,json,time
print(json.dumps({"version":"specialization_worker_v1","status":"ready","model_sha256":"a"*64,"adapter_sha256":None,"load_seconds":0.1}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r.get("op")=="shutdown": break
 if r["id"]=="hang": time.sleep(60)
 elif r["id"]=="crash": sys.exit(3)
 else: print(json.dumps({"version":r["version"],"id":r["id"],"status":"complete","result":RESULT,"engine_metrics":{"prompt_tokens":100,"generated_tokens":30}}),flush=True)
'''.replace("RESULT", repr(result()))
    controller = worker.WorkerController(config, python=Path(sys.executable))
    monkeypatch.setattr(controller, "_command", lambda: [sys.executable, "-u", "-c", code])
    return controller


def test_controller_holds_one_lock_entire_lifetime_and_private_telemetry(fake_child_controller):
    controller = fake_child_controller.start()
    fd = os.open(Path(controller.config.state_dir) / "local-model-execution.lock", os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = controller.request(context(), timeout_seconds=5)
        assert response["result"] == result()
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = Path(controller.config.state_dir) / "inference-attempts" / (response["telemetry_attempt_id"] + ".json")
        text = receipt.read_text()
        assert "An editor shows" not in text and "synthetic-a" not in text
        assert receipt.stat().st_mode & 0o777 == 0o600
        assert json.loads(text)["result_status"] == "complete"
    finally:
        controller.close()
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)


def test_controller_receipts_distinguish_effective_decoding_settings(fake_child_controller):
    from telemetry_summary import summarize

    controller = fake_child_controller.start()
    try:
        for thinking, side in ((False, 1024), (True, 2048)):
            controller.request(context(), thinking=thinking, image_side=side, timeout_seconds=5)
        receipts = [json.loads(p.read_text()) for p in (Path(controller.config.state_dir) / "inference-attempts").glob("*.json")]
        requests = [r for r in receipts if r["context"]["stage"] == "vision"]
        assert len({r["configuration_sha256"] for r in requests}) == 2
        groups = summarize(Path(controller.config.state_dir))["groups"]
        assert len([g for g in groups if g.startswith("vision:" )]) == 2
    finally:
        controller.close()


def test_worker_inherits_lock_even_when_parent_descriptor_is_lost(fake_child_controller):
    controller = fake_child_controller.start()
    fd = os.open(Path(controller.config.state_dir) / "local-model-execution.lock", os.O_RDWR)
    try:
        os.close(controller.lock_fd)
        controller.lock_fd = None
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        controller.close()
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)


@pytest.mark.parametrize("request_id,code", [("hang", "deadline"), ("crash", "invalid_worker_response")])
def test_controller_timeout_and_crash_reap_only_its_worker_and_release_lock(fake_child_controller, request_id, code):
    controller = fake_child_controller.start()
    child = controller.child
    with pytest.raises(worker.WorkerError, match=code):
        controller.request(context(), request_id=request_id, timeout_seconds=1)
    assert child.poll() is not None
    fd = os.open(Path(controller.config.state_dir) / "local-model-execution.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)


def test_controller_gate_prevents_launch_and_second_request(fake_child_controller, monkeypatch):
    controller = fake_child_controller
    monkeypatch.setattr(worker, "gate", lambda: "battery_power")
    with pytest.raises(worker.WorkerError, match="battery_power"):
        controller.start()
    assert controller.child is None
    monkeypatch.setattr(worker, "gate", lambda: None)
    controller.start()
    child = controller.child
    monkeypatch.setattr(worker, "gate", lambda: "outside_vision_window")
    with pytest.raises(worker.WorkerError, match="outside_vision_window"):
        controller.request(context())
    assert child.poll() is not None


def test_lifetime_watchdog_kills_hung_owned_process_without_loading_a_model():
    code = "import time; from specialization_worker import lifetime_watchdog; lifetime_watchdog(time.monotonic()+.05); time.sleep(60)"
    child = subprocess.run([sys.executable, "-c", code], timeout=5, capture_output=True)
    assert child.returncode == 124


def test_kernel_deadline_terminates_native_call_holding_gil():
    code = """import ctypes,time
from specialization_worker import KernelDeadline
KernelDeadline(time.monotonic()+.05)
ctypes.PyDLL(None).sleep(60)
"""
    child = subprocess.run([sys.executable, "-c", code], timeout=5, capture_output=True)
    assert child.returncode == -signal.SIGALRM


def test_nested_kernel_deadline_restores_whole_worker_limit():
    code = """import time
from specialization_worker import KernelDeadline
outer=KernelDeadline(time.monotonic()+.1)
inner=KernelDeadline(time.monotonic()+1)
inner.cancel()
time.sleep(60)
"""
    child = subprocess.run([sys.executable, "-c", code], timeout=5, capture_output=True)
    assert child.returncode == -signal.SIGALRM
