"""Aggregate local performance counters without opening activity contents."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from secure_store import STATE_DIR


def summarize(root: Path = STATE_DIR) -> dict:
    attempts = []
    for path in (root / "inference-attempts").glob("*.json"):
        if path.is_symlink():
            continue
        try:
            item = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if item.get("version") == "inference_telemetry_v1":
            attempts.append(item)
    groups = {}
    for item in attempts:
        context = item.get("context", {})
        key = ":".join(str(context.get(k, "unknown")) for k in ("stage", "engine_version", "variant"))
        groups.setdefault(key, []).append(item)
    result = {}
    for key, values in groups.items():
        done = [r for r in values if r.get("status") == "complete" and r.get("result_status", "complete") == "complete"]
        elapsed = [r["process_seconds"] for r in done if isinstance(r.get("process_seconds"), (int, float))]
        cpu = [s["cpu_percent_one_core"] / r["logical_cpu_count"] for r in values
               for s in r.get("samples", []) if s.get("cpu_percent_one_core") is not None and r.get("logical_cpu_count")]
        gpu = [s["gpu"]["device_percent"] for r in values for s in r.get("samples", [])
               if s.get("gpu", {}).get("device_percent") is not None]
        total = sum(r.get("elapsed_seconds", 0) for r in values)
        result[key] = {"attempts": len(values), "complete_process_and_decode": len(done),
                       "failed_or_pending": len(values) - len(done),
                       "median_request_seconds": statistics.median(elapsed) if elapsed else None,
                       "mean_request_seconds": statistics.mean(elapsed) if elapsed else None,
                       "sampled_process_cpu_percent_all_logical_cores": statistics.mean(cpu) if cpu else None,
                       "sampled_whole_device_gpu_percent": statistics.mean(gpu) if gpu else None,
                       "elapsed_attempt_seconds": total,
                       "estimated_frames_per_6_5h_20_percent_reserve": math.floor(.8 * 23400 * len(done) / total) if total else None,
                       "capacity_scope": "frames_not_accuracy_episodes; includes_failed_attempt_cost; partial_measurement",
                       "gpu_scope": "whole_device_not_attributable_to_model"}
    return {"version": "telemetry_summary_v1", "groups": result,
            "missing_measurements": "null; CPU/RSS exclude some Metal memory; GPU counters cover all apps"}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(summarize()))
