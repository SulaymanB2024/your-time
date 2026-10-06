from private_io import write_json
from telemetry_summary import summarize


def attempt(root, name, **fields):
    write_json(root / "inference-attempts" / (name + ".json"), {
        "version": "inference_telemetry_v1", "context": {"stage": "vision", "variant": "mlx_base"},
        "status": "complete", "result_status": "complete", "process_seconds": 10, "elapsed_seconds": 10, **fields})


def test_attempt_timings_cannot_be_presented_as_sustainable_capacity(tmp_path):
    attempt(tmp_path, "request")
    attempt(tmp_path, "startup", process_seconds=20, elapsed_seconds=20)
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["elapsed_attempt_seconds"] == 30
    assert row["estimated_frames_per_6_5h_20_percent_reserve"] is None
    assert "full_session_wall_time" in row["capacity_scope"]


def test_unknown_cost_is_not_zero_and_counters_remain_scoped(tmp_path):
    attempt(tmp_path, "request", logical_cpu_count=8, samples=[{
        "cpu_percent_one_core": 400, "gpu": {"device_percent": 80}}])
    attempt(tmp_path, "interrupted", status="running", elapsed_seconds=None)
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["elapsed_attempt_seconds"] is None
    assert row["measured_attempt_seconds"] == 10
    assert row["attempts_with_unknown_elapsed"] == 1
    assert row["sampled_process_cpu_percent_all_logical_cores"] == 50
    assert row["sampled_whole_device_gpu_percent"] == 80
    assert row["gpu_scope"] == "whole_device_not_attributable_to_model"


def test_model_adapter_and_configuration_identities_do_not_blend(tmp_path):
    for i in range(3):
        attempt(tmp_path, str(i), context={"stage": "vision", "variant": "mlx_adapter",
            "model_sha256": "a" * 64, "adapter_sha256": str(i) * 64}, configuration_sha256="b" * 64)
    attempt(tmp_path, "configuration", context={"stage": "vision", "variant": "mlx_adapter",
        "model_sha256": "a" * 64, "adapter_sha256": "0" * 64}, configuration_sha256="c" * 64)
    assert len(summarize(tmp_path)["groups"]) == 4


def test_successful_process_without_decode_disposition_is_unknown(tmp_path):
    attempt(tmp_path, "unknown", result_status=None)
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["complete_process_and_decode"] == 0
    assert row["complete_process"] == row["decoding_disposition_unknown"] == 1


def test_content_fields_and_untrusted_files_cannot_enter_summary(tmp_path):
    marker = "PRIVATE WINDOW / SECRET"
    attempt(tmp_path, "safe", context={"stage": "vision", "variant": marker}, caption=marker,
            configuration_sha256=marker, samples=[{"cpu_percent_one_core": float("nan")}])
    target = tmp_path / "outside.json"
    write_json(target, {"version": "inference_telemetry_v1"})
    (tmp_path / "inference-attempts/linked.json").symlink_to(target)
    import json
    result = summarize(tmp_path)
    assert marker not in json.dumps(result)
    assert sum(r["attempts"] for r in result["groups"].values()) == 1
