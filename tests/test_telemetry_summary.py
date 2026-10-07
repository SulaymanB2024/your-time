import pytest

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
    assert row["failed_or_pending"] == row["decode_failed"] == row["process_failed"] == 0


def test_process_and_decode_failures_do_not_double_count_or_absorb_unknowns(tmp_path):
    attempt(tmp_path, "success")
    attempt(tmp_path, "training", result_status=None)
    attempt(tmp_path, "decode", result_status="schema_error")
    attempt(tmp_path, "crash", status="process_error", result_status="schema_error")
    attempt(tmp_path, "pending", status="running", result_status=None)
    attempt(tmp_path, "unknown", status=None, result_status=None)
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["attempts"] == 6
    assert row["complete_process"] == 3
    assert row["complete_process_and_decode"] == 1
    assert row["process_failed"] == row["process_pending"] == row["decode_failed"] == 1
    assert row["failed_or_pending"] == 3
    assert row["process_disposition_unknown"] == row["decoding_disposition_unknown"] == 1


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


def test_successful_startup_has_no_decode_and_cannot_count_as_a_frame(tmp_path):
    attempt(tmp_path, "startup", context={"stage": "vision_startup", "variant": "mlx_resident"},
            result_status="not_applicable")
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["complete_process"] == row["decoding_not_applicable"] == 1
    assert row["complete_process_and_decode"] == row["decode_failed"] == 0
    assert row["decoding_disposition_unknown"] == row["failed_or_pending"] == 0
    assert row["mean_request_seconds"] is None


def test_not_applicable_does_not_hide_generation_or_startup_process_errors(tmp_path):
    attempt(tmp_path, "generation", result_status="not_applicable")
    attempt(tmp_path, "invalid_context", context=None, result_status="not_applicable")
    attempt(tmp_path, "startup_error", context={"stage": "vision_startup", "variant": "mlx_resident"},
            status="worker_start_failed", result_status="not_applicable")
    groups = summarize(tmp_path)["groups"].values()
    assert sum(row["decoding_not_applicable"] for row in groups) == 0
    assert sum(row["decode_failed"] for row in groups) == 2
    assert sum(row["process_failed"] for row in groups) == 1
    assert sum(row["failed_or_pending"] for row in groups) == 3


@pytest.mark.parametrize("count", [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(0.5, id="fractional"),
    pytest.param(8.0, id="float"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(None, id="missing"),
    pytest.param("8", id="string"),
    pytest.param([], id="list"),
    pytest.param({}, id="object"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinite"),
    pytest.param(10**400, id="oversized_integer"),
])
def test_invalid_cpu_counts_keep_attempts_and_other_measurements(tmp_path, count):
    attempt(tmp_path, "invalid_count", logical_cpu_count=count, samples=[{
        "cpu_percent_one_core": 400, "gpu": {"device_percent": 80}}])
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["sampled_process_cpu_percent_all_logical_cores"] is None
    assert row["attempts"] == row["complete_process_and_decode"] == 1
    assert row["elapsed_attempt_seconds"] == 10
    assert row["sampled_whole_device_gpu_percent"] == 80

    attempt(tmp_path, "valid_zero", logical_cpu_count=8, samples=[{
        "cpu_percent_one_core": 0, "gpu": {"device_percent": 0}}])
    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["sampled_process_cpu_percent_all_logical_cores"] == 0
    assert row["attempts"] == row["complete_process_and_decode"] == 2
    assert row["elapsed_attempt_seconds"] == 20
    assert row["sampled_whole_device_gpu_percent"] == 40


def test_malformed_dispositions_remain_unknown_without_losing_attempts(tmp_path):
    malformed = [None, False, 1, [], {}]
    for index, value in enumerate(malformed):
        attempt(tmp_path, f"decode_{index}", result_status=value)
        attempt(tmp_path, f"process_{index}", status=value)
    attempt(tmp_path, "success")
    attempt(tmp_path, "decode_failure", result_status="schema_error")
    attempt(tmp_path, "process_failure", status="process_error", result_status="schema_error")
    attempt(tmp_path, "pending", status="running", elapsed_seconds=None)

    row = next(iter(summarize(tmp_path)["groups"].values()))
    assert row["attempts"] == 14
    assert row["complete_process"] == 7
    assert row["complete_process_and_decode"] == row["decode_failed"] == 1
    assert row["decoding_disposition_unknown"] == row["process_disposition_unknown"] == 5
    assert row["process_failed"] == row["process_pending"] == 1
    assert row["failed_or_pending"] == 3
    assert row["elapsed_attempt_seconds"] is None
    assert row["measured_attempt_seconds"] == 130
    assert row["attempts_with_unknown_elapsed"] == 1
    assert row["complete_process"] == sum(row[key] for key in (
        "complete_process_and_decode", "decoding_disposition_unknown",
        "decoding_not_applicable", "decode_failed"))
    assert row["attempts"] == sum(row[key] for key in (
        "complete_process", "process_failed", "process_pending", "process_disposition_unknown"))


def test_oversized_measurements_are_unavailable_without_dropping_attempts(tmp_path):
    import json

    oversized = 10**400
    attempt(tmp_path, "oversized", process_seconds=oversized, elapsed_seconds=oversized,
            logical_cpu_count=8, samples=[{
                "cpu_percent_one_core": oversized, "gpu": {"device_percent": oversized}}])
    attempt(tmp_path, "measured", logical_cpu_count=8, samples=[{
        "cpu_percent_one_core": 400, "gpu": {"device_percent": 80}}])
    result = summarize(tmp_path)
    row = next(iter(result["groups"].values()))
    assert row["attempts"] == row["complete_process_and_decode"] == 2
    assert row["mean_request_seconds"] == row["median_request_seconds"] == 10
    assert row["elapsed_attempt_seconds"] is None
    assert row["measured_attempt_seconds"] == 10
    assert row["attempts_with_unknown_elapsed"] == 1
    assert row["sampled_process_cpu_percent_all_logical_cores"] == 50
    assert row["sampled_whole_device_gpu_percent"] == 80
    json.dumps(result, allow_nan=False)
