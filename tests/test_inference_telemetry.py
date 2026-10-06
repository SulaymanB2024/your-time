import json
import plistlib
import subprocess
import sys

import pytest

from inference_telemetry import (
    Attempt,
    engine_timings,
    gpu_statistics,
    record_result,
    safe_context,
)
from model_execution import run_model


def test_prompt_timing_is_not_misreported_as_generation():
    result = engine_timings("llama_perf_context_print: prompt eval time = 250 ms / 10 tokens\n")
    assert result["prompt_seconds"] == .25
    assert result["generation_seconds"] is None
    assert result["generated_tokens"] is None
    result = engine_timings("x: load time = 1500 ms\nx: prompt eval time = 250 ms / 10 tokens\nx: eval time = 500 ms / 20 runs\n")
    assert result["load_seconds"] == 1.5
    assert result["tokens_per_second"] == 40


def test_telemetry_never_retains_prompt_output_paths_or_error_bodies(tmp_path, monkeypatch):
    monkeypatch.setattr("inference_telemetry.read_command", lambda _: None)
    marker = "PRIVATE-PROMPT-AND-CAPTION"
    result = run_model([sys.executable, "-c", f"print({marker!r}); import sys; print({marker!r},file=sys.stderr)"],
                       state_dir=tmp_path, capture_output=True, timeout=5,
                       telemetry={"prompt": marker, "caption": marker, "stage": "vision",
                                  "input_sha256": "a" * 64, "variant": "/private/window/name"})
    paths = list((tmp_path / "inference-attempts").glob("*.json"))
    assert len(paths) == 1
    assert marker not in paths[0].read_text()
    data = json.loads(paths[0].read_text())
    assert data["context"] == {"stage": "vision", "input_sha256": "a" * 64}
    assert data["attempt_id"] == result.telemetry_attempt_id
    assert data["engine"]["time_to_first_token_seconds"] is None
    assert paths[0].stat().st_mode & 0o777 == 0o600


def test_timeout_preserves_attempt_and_releases_only_its_own_child(tmp_path, monkeypatch):
    monkeypatch.setattr("inference_telemetry.read_command", lambda _: None)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_model([sys.executable, "-c", "import time;time.sleep(10)"], state_dir=tmp_path,
                  timeout=.05, capture_output=True)
    identity = caught.value.telemetry_attempt_id
    data = json.loads((tmp_path / "inference-attempts" / (identity + ".json")).read_text())
    assert data["status"] == "timeout"
    assert run_model([sys.executable, "-c", "pass"], state_dir=tmp_path, timeout=5).returncode == 0


def test_retry_history_survives_caption_replacement(tmp_path):
    context = {"stage": "vision", "input_sha256": "a" * 64}
    first = Attempt(tmp_path, ["model", "-n", "100"], context)
    first.finish("complete")
    record_result(tmp_path, first.identity, "schema_error")
    second = Attempt(tmp_path, ["model", "-n", "100"], context)
    second.finish("complete", engine_metrics={"time_to_first_token_seconds": .4})
    assert second.data["retry_of"] == first.identity
    assert first.path.exists() and second.path.exists()
    assert second.data["engine"]["time_to_first_token_seconds"] == .4
    different = Attempt(tmp_path, ["model", "-n", "101"], context)
    assert "retry_of" not in different.data


def test_device_gpu_statistics_are_never_model_attributed():
    payload = plistlib.dumps([{"PerformanceStatistics": {"Device Utilization %": 80, "Renderer Utilization %": 70}}])
    values = gpu_statistics(payload)
    assert values["scope"] == "whole_device"
    assert values["device_percent"] == 80
    assert values["in_use_bytes"] is None
    assert gpu_statistics(b"invalid")["device_percent"] is None
    assert safe_context({"image_width": float("nan"), "prompt_version": "unsafe label"}) == {}


def test_variable_preparation_time_does_not_break_retry_identity(tmp_path):
    first = Attempt(tmp_path, ["model", "-n", "100"], {"input_sha256": "a" * 64, "image_prepare_seconds": .1})
    first.finish("process_error")
    second = Attempt(tmp_path, ["model", "-n", "100"], {"input_sha256": "a" * 64, "image_prepare_seconds": .2})
    assert second.data["retry_of"] == first.identity
    assert second.data["context"]["image_prepare_seconds"] == .2


def test_invalid_subprocess_arguments_do_not_leave_a_running_attempt(tmp_path):
    with pytest.raises(ValueError):
        run_model([sys.executable], state_dir=tmp_path, capture_output=True, stdout=subprocess.PIPE)
    assert not (tmp_path / "inference-attempts").exists()


def test_native_deadline_survives_coordinator_crash_and_releases_lock(tmp_path):
    import time
    from pathlib import Path

    from model_execution import ModelBusy

    repo = Path(__file__).resolve().parent.parent
    signal_path = tmp_path/'child-started'
    child_code = "from pathlib import Path; import time; Path("+repr(str(signal_path))+").write_text('ready'); time.sleep(20)"
    coordinator_code = "from pathlib import Path; from model_execution import run_model; run_model("+repr([sys.executable, '-c', child_code])+", state_dir=Path("+repr(str(tmp_path))+"), timeout=1, capture_output=True)"
    coordinator = subprocess.Popen([sys.executable, '-c', coordinator_code], cwd=repo,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until = time.monotonic()+3
        while not signal_path.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert signal_path.exists()
        coordinator.kill()
        coordinator.wait(timeout=3)
        with pytest.raises(ModelBusy):
            run_model([sys.executable, '-c', 'pass'], state_dir=tmp_path, timeout=5)
        time.sleep(1.1)
        assert run_model([sys.executable, '-c', 'pass'], state_dir=tmp_path, timeout=5).returncode == 0
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait()
