"""Aggregate local performance counters without opening activity contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from inference_telemetry import safe_context
from private_io import open_private_file
from secure_store import STATE_DIR


def measured(value):
    if type(value) not in (int, float):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def startup_without_decoding(item):
    context = item.get("context")
    return (isinstance(context, dict) and context.get("stage") == "vision_startup"
            and item.get("result_status") == "not_applicable")


def summarize(root: Path = STATE_DIR) -> dict:
    attempts = []
    for path in (root / "inference-attempts").glob("*.json"):
        if path.is_symlink():
            continue
        try:
            with os.fdopen(open_private_file(path, os.O_RDONLY)) as stream:
                body = stream.read(4 * 1024**2 + 1)
            if len(body) > 4 * 1024**2:
                continue
            item = json.loads(body)
        except (ValueError, OSError, RuntimeError):
            continue
        if isinstance(item, dict) and item.get("version") == "inference_telemetry_v1":
            attempts.append(item)
    groups = {}
    for item in attempts:
        context = safe_context(item.get("context") if isinstance(item.get("context"), dict) else {})
        identity = {k: context.get(k) for k in ("model_sha256", "projector_sha256", "adapter_sha256",
                    "engine_sha256", "prompt_version", "template_sha256", "experiment_sha256")}
        config = item.get("configuration_sha256")
        identity["configuration_sha256"] = config if isinstance(config, str) and len(config) == 64 and all(c in "0123456789abcdef" for c in config) else None
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        key = ":".join(context.get(k, "unknown") for k in ("stage", "engine_version", "variant")) + ":" + digest
        groups.setdefault(key, []).append(item)
    result = {}
    for key, values in groups.items():
        process_done = [r for r in values if r.get("status") == "complete"]
        done = [r for r in process_done if r.get("result_status") == "complete"]
        process_pending = sum(r.get("status") == "running" for r in values)
        process_failed = sum(isinstance(r.get("status"), str)
                             and r["status"] not in {"complete", "running"}
                             for r in values)
        # Resident startup has no caption to decode. Only the explicitly
        # declared startup stage may use this disposition; a generation cannot
        # bypass failure accounting by declaring its result not applicable.
        no_decode = [r for r in process_done if startup_without_decoding(r)]
        decode_failed = sum(isinstance(r.get("result_status"), str)
                            and r["result_status"] != "complete"
                            and not startup_without_decoding(r)
                            for r in process_done)
        elapsed = [r["process_seconds"] for r in done if measured(r.get("process_seconds")) is not None]
        samples = [(r, s) for r in values if isinstance(r.get("samples"), list) for s in r["samples"] if isinstance(s, dict)]
        cpu = [s["cpu_percent_one_core"] / r["logical_cpu_count"] for r, s in samples
               if measured(s.get("cpu_percent_one_core")) is not None
               and type(r.get("logical_cpu_count")) is int
               and r["logical_cpu_count"] > 0
               and measured(r["logical_cpu_count"]) is not None]
        gpu = [s["gpu"]["device_percent"] for _, s in samples if isinstance(s.get("gpu"), dict)
               and measured(s["gpu"].get("device_percent")) is not None]
        times = [measured(r.get("elapsed_seconds")) for r in values]
        known = sum(v for v in times if v is not None)
        total = known if all(v is not None for v in times) else None
        result[key] = {"attempts": len(values), "complete_process_and_decode": len(done),
                       "complete_process": len(process_done),
                       "decoding_disposition_unknown": sum(not isinstance(r.get("result_status"), str) for r in process_done),
                       "decoding_not_applicable": len(no_decode),
                       "process_pending": process_pending,
                       "process_failed": process_failed,
                       "decode_failed": decode_failed,
                       "process_disposition_unknown": sum(not isinstance(r.get("status"), str) for r in values),
                       "failed_or_pending": process_pending + process_failed + decode_failed,
                       "median_request_seconds": statistics.median(elapsed) if elapsed else None,
                       "mean_request_seconds": statistics.mean(elapsed) if elapsed else None,
                       "sampled_process_cpu_percent_all_logical_cores": statistics.mean(cpu) if cpu else None,
                       "sampled_whole_device_gpu_percent": statistics.mean(gpu) if gpu else None,
                       "elapsed_attempt_seconds": total,
                       "measured_attempt_seconds": known,
                       "attempts_with_unknown_elapsed": sum(v is None for v in times),
                       "estimated_frames_per_6_5h_20_percent_reserve": None,
                       "capacity_scope": "requires_full_session_wall_time_from_benchmark_or_trial;attempts_exclude_startup_cleanup_and_preparation_gaps",
                       "gpu_scope": "whole_device_not_attributable_to_model"}
    return {"version": "telemetry_summary_v4", "groups": result,
            "failure_scope": "process_failure_or_explicit_decode_failure;missing_or_nonstring_decode_disposition_is_unknown;successful_declared_startup_has_no_decode",
            "missing_measurements": "null; CPU/RSS exclude some Metal memory; GPU counters cover all apps"}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(summarize()))
