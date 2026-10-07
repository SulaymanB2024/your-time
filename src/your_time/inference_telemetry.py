"""Content-free private measurements for each local inference attempt.

Device counters describe the whole Mac, never attributable model utilization.
Missing counters are null. No command, prompt, output or raw stderr is retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import plistlib
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from private_io import open_private_file, write_json

VERSION = "inference_telemetry_v1"
SAFE_LABEL = re.compile(r"[a-zA-Z0-9_.-]{1,80}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")
NUMERIC_FLAGS = {"-c", "-n", "-t", "-tb", "-ngl", "--image-max-tokens", "--temp",
                 "--thinking", "--thinking-budget", "--image-side", "--center-crop", "--seed", "--request-timeout"}


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def configuration(command: list[str]) -> dict:
    values = {}
    for i, flag in enumerate(command[:-1]):
        if flag in NUMERIC_FLAGS and number(command[i + 1]) is not None:
            values[flag] = number(command[i + 1])
    return values


def safe_context(value: dict | None) -> dict:
    result = {}
    for key, item in (value or {}).items():
        if key in {"stage", "prompt_version", "engine_version", "variant", "cache_state", "retry_of"}:
            if isinstance(item, str) and SAFE_LABEL.fullmatch(item):
                result[key] = item
        elif key in {"model_sha256", "input_sha256", "prompt_sha256", "projector_sha256", "adapter_sha256", "template_sha256", "engine_sha256", "experiment_sha256"}:
            if isinstance(item, str) and HASH.fullmatch(item):
                result[key] = item
        elif key in {"image_prepare_seconds", "image_width", "image_height"}:
            if number(item) is not None:
                result[key] = number(item)
    return result


def read_command(command: list[str]) -> bytes | None:
    try:
        result = subprocess.run(command, capture_output=True, timeout=3)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_statistics(payload: bytes | None) -> dict:
    result = {"device_percent": None, "renderer_percent": None,
              "tiler_percent": None, "in_use_bytes": None,
              "scope": "whole_device"}
    try:
        nodes = plistlib.loads(payload or b"")
    except (ValueError, plistlib.InvalidFileException):
        return result
    def walk(node):
        if isinstance(node, list):
            for child in node:
                yield from walk(child)
        elif isinstance(node, dict):
            if isinstance(node.get("PerformanceStatistics"), dict):
                yield node["PerformanceStatistics"]
            for key, child in node.items():
                if key != "PerformanceStatistics" and isinstance(child, (list, dict)):
                    yield from walk(child)
    stats = next(walk(nodes), {})
    for key, source in (("device_percent", "Device Utilization %"),
                        ("renderer_percent", "Renderer Utilization %"),
                        ("tiler_percent", "Tiler Utilization %"),
                        ("in_use_bytes", "In use system memory")):
        result[key] = number(stats.get(source))
    return result


def engine_timings(stderr: bytes | str | None) -> dict:
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr or ""
    result = {"load_seconds": None, "prompt_seconds": None, "generation_seconds": None,
              "prompt_tokens": None, "generated_tokens": None,
              "vision_encode_seconds": None, "tokens_per_second": None,
              "image_prepare_seconds": None,
              "reset_prepare_seconds": None, "cleanup_seconds": None,
              "request_total_seconds": None,
              "vision_tokens": None, "image_prepared_width": None, "image_prepared_height": None,
              "time_to_first_token_seconds": None}
    for label, key in (("load time", "load_seconds"),
                       ("prompt eval time", "prompt_seconds"),
                       ("eval time", "generation_seconds")):
        matches = re.findall(r"(?:^|\n)[^\n=]*?(?<!prompt )\b" + label +
                             r"\s*=\s*([\d.]+)\s*ms(?:\s*/\s*(\d+)\s*(?:tokens|runs))?", text)
        if matches:
            elapsed, count = matches[-1]
            result[key] = (number(elapsed) or 0) / 1000
            if count and key != "load_seconds":
                result["prompt_tokens" if key == "prompt_seconds" else "generated_tokens"] = int(count)
    match = re.search(r"(?:encode_image|image encoding|image encoded)[^\n]*?([\d.]+)\s*ms", text, re.I)
    if match:
        result["vision_encode_seconds"] = (number(match[1]) or 0) / 1000
    seconds, tokens = result["generation_seconds"], result["generated_tokens"]
    if seconds and tokens is not None:
        result["tokens_per_second"] = tokens / seconds
    return result


def record_result(state_dir: Path, attempt_id: str | None, status: str) -> None:
    """Attach content-free decoding disposition without removing process history."""
    if not attempt_id or not re.fullmatch(r"[a-f0-9]{32}", attempt_id) or not SAFE_LABEL.fullmatch(status):
        return
    path = state_dir / "inference-attempts" / (attempt_id + ".json")
    try:
        with os.fdopen(open_private_file(path, os.O_RDONLY)) as stream:
            data = json.load(stream)
        data["result_status"] = status
        write_json(path, data)
        if status != "complete" and HASH.fullmatch(data.get("request_sha256", "")):
            index = state_dir / "inference-attempt-index" / (data["request_sha256"] + ".json")
            with os.fdopen(open_private_file(index, os.O_RDONLY)) as stream:
                prior = json.load(stream)
            if prior.get("attempt_id") == attempt_id:
                write_json(index, {"attempt_id": attempt_id, "status": status})
    except (OSError, RuntimeError, ValueError):
        return


class Attempt:
    def __init__(self, state_dir: Path, command: list[str], context: dict | None):
        self.identity = uuid.uuid4().hex
        self.path = state_dir / "inference-attempts" / (self.identity + ".json")
        config = configuration(command)
        self.data = {"version": VERSION, "attempt_id": self.identity,
                     "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "status": "running", "context": safe_context(context),
                     "configuration": config,
                     "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                     "unavailable_identity_fields": sorted({"model_sha256", "projector_sha256", "engine_sha256", "prompt_sha256", "template_sha256", "adapter_sha256"} - set(safe_context(context))),
                     "measurement_notes": {"llama_generated_tokens": "eval_runs_when_reported_not_final_caption_length",
                                           "gpu": "whole_device_not_model_specific", "rss": "process_resident_memory_not_total_Metal_allocation"},
                     "logical_cpu_count": os.cpu_count(), "samples": [],
                     "cpu_scope": "process_percent_of_one_core", "device_scope": "whole_device"}
        identity_context = {key: value for key, value in self.data["context"].items()
                            if key not in {"image_prepare_seconds", "retry_of"}}
        request_hash = hashlib.sha256(json.dumps({"context": identity_context, "configuration": config}, sort_keys=True).encode()).hexdigest()
        self.data["request_sha256"] = request_hash
        self.index_path = state_dir / "inference-attempt-index" / (request_hash + ".json")
        if self.index_path.exists():
            try:
                with os.fdopen(open_private_file(self.index_path, os.O_RDONLY)) as stream:
                    prior = json.load(stream)
                if prior.get("status") != "complete" and re.fullmatch(r"[a-f0-9]{32}", prior.get("attempt_id", "")):
                    self.data["retry_of"] = prior["attempt_id"]
            except (OSError, RuntimeError, ValueError):
                pass
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = None
        self.error = False
        self._write()
        try:
            write_json(self.index_path, {"attempt_id": self.identity, "status": "running"})
        except (OSError, RuntimeError):
            self.error = True

    def _write(self):
        try:
            write_json(self.path, self.data)
        except (OSError, RuntimeError):
            self.error = True

    def start_sampling(self, pid: int):
        def sample():
            count = 0
            while not self.stop.is_set():
                item = {"elapsed_seconds": round(time.monotonic() - self.started, 3),
                        "cpu_percent_one_core": None, "rss_bytes": None}
                payload = read_command(["/bin/ps", "-p", str(pid), "-o", "%cpu=,rss="])
                fields = (payload or b"").split()
                if len(fields) == 2:
                    item["cpu_percent_one_core"] = number(fields[0])
                    rss = number(fields[1])
                    item["rss_bytes"] = rss * 1024 if rss is not None else None
                if count % 5 == 0:
                    item["gpu"] = gpu_statistics(read_command(["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"]))
                    memory = (read_command(["/usr/bin/memory_pressure", "-Q"]) or b"").decode()
                    match = re.search(r"memory free percentage:\s*(\d+)%", memory)
                    item["memory_free_percent"] = int(match[1]) if match else None
                    vm = (read_command(["/usr/bin/vm_stat"]) or b"").decode()
                    page = re.search(r"page size of (\d+) bytes", vm)
                    item["system_vm"] = {}
                    for source, key in (("Pages occupied by compressor", "compressed_bytes"),
                                        ("Swapins", "swap_in_bytes_total"), ("Swapouts", "swap_out_bytes_total")):
                        match = re.search(re.escape(source) + r":\s*(\d+)", vm)
                        item["system_vm"][key] = int(match[1]) * int(page[1]) if match and page else None
                    therm = (read_command(["/usr/bin/pmset", "-g", "therm"]) or b"").decode()
                    item["thermal_limits"] = {}
                    for source in ("CPU_Speed_Limit", "CPU_Scheduler_Limit"):
                        match = re.search(source + r"\s*=\s*(\d+)", therm)
                        item["thermal_limits"][source] = int(match[1]) if match else None
                self.data["samples"].append(item)
                count += 1
                if count >= 14400 or self.stop.wait(2):
                    return
        self.thread = threading.Thread(target=sample, name="inference-telemetry", daemon=True)
        self.thread.start()

    def finish(self, status: str, *, stderr=None, returncode=None, engine_metrics=None):
        process_seconds = time.monotonic() - self.started
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=20)
        metrics = engine_timings(stderr)
        for key, value in (engine_metrics or {}).items():
            if key in metrics and number(value) is not None:
                metrics[key] = number(value)
        vm_samples = [s["system_vm"] for s in self.data["samples"] if "system_vm" in s]
        deltas = {}
        for key in ("swap_in_bytes_total", "swap_out_bytes_total"):
            values = [s[key] for s in vm_samples if s.get(key) is not None]
            deltas[key.removesuffix("_total") + "_delta"] = max(0, values[-1] - values[0]) if len(values) >= 2 else None
        self.data.update(status=status, returncode=returncode,
                         finished_at_utc=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=round(time.monotonic() - self.started, 3),
                         process_seconds=round(process_seconds, 3),
                         sampler_shutdown_seconds=round(time.monotonic() - self.started - process_seconds, 3),
                         system_swap= deltas, engine=metrics)
        self._write()
        try:
            write_json(self.index_path, {"attempt_id": self.identity, "status": status})
        except (OSError, RuntimeError):
            self.error = True
