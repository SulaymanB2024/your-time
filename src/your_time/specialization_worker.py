"""Bounded, offline, serial MLX-VLM worker; importing this module loads no model."""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import select
import signal
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from activity_context import parse_result, prompt, raw_safety_checks
from inference_telemetry import Attempt
from model_execution import ModelBusy
from overnight_schedule import VISION_END, in_window, remaining_seconds
from private_io import open_private_file
from vision_batch import resource_gate

PROTOCOL = "specialization_worker_v1"
PACKAGES = {"mlx-vlm": "0.7.6", "mlx": "0.32.3", "mlx-metal": "0.32.3",
            "transformers": "5.18.0", "numpy": "2.4.6", "torch": "2.12.1",
            "torchvision": "0.27.1", "pillow": "12.3.0", "tokenizers": "0.23.2"}
PROJECT = Path(__file__).resolve().parent
MAX_FRAME = 64 * 1024
MAX_OUTPUT = 32 * 1024
IDENTITY = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class WorkerError(RuntimeError):
    """Contains only a public failure code, never model diagnostics."""

    def __init__(self, code):
        super().__init__(code if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,64}", code) else "worker_error")


def gate() -> str | None:
    if not in_window(datetime.now(timezone.utc), end=VISION_END):
        return "outside_vision_window"
    try:
        return resource_gate(benchmark_now=False)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return "resource_status_unavailable"


def check_admission(deadline: float, admission=None) -> None:
    if time.monotonic() >= deadline:
        raise WorkerError("deadline")
    reason = (admission or gate)()
    if reason:
        raise WorkerError(reason)


def private_path(path: Path, *, directory: bool = False) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise WorkerError("invalid_path")
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise WorkerError("nonprivate_path")
    if not directory and info.st_nlink != 1:
        raise WorkerError("linked_file")
    return path


def read_json(path: Path, maximum=MAX_FRAME) -> dict:
    path = private_path(path)
    with os.fdopen(open_private_file(path, os.O_RDONLY), "rb") as source:
        data = source.read(maximum + 1)
    if len(data) > maximum:
        raise WorkerError("metadata_too_large")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise WorkerError("invalid_metadata")
    return value


def snapshot_identity(entries: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sorted(entries, key=lambda entry: entry["name"]),
                                    sort_keys=True).encode()).hexdigest()


def verify_snapshot(folder: Path, manifest: Path, deadline: float, admission=gate) -> str:
    """Verify the caller's pinned manifest and the complete runtime file closure."""
    folder = private_path(folder, directory=True)
    spec = read_json(manifest)
    entries = spec.get("files")
    if not isinstance(entries, list) or not entries:
        raise WorkerError("invalid_manifest")
    names = set()
    next_gate = 0.0
    for entry in entries:
        check_admission(deadline, admission)
        if not isinstance(entry, dict):
            raise WorkerError("invalid_manifest")
        name, digest = entry.get("name"), entry.get("sha256")
        if (not isinstance(name, str) or Path(name).name != name or name in names
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise WorkerError("invalid_manifest")
        names.add(name)
        path = private_path(folder / name)
        if "size" in entry and path.stat().st_size != entry["size"]:
            raise WorkerError("snapshot_size_mismatch")
        hasher = hashlib.sha256()
        with os.fdopen(open_private_file(path, os.O_RDONLY), "rb") as source:
            while chunk := source.read(1024 * 1024):
                if time.monotonic() >= deadline:
                    raise WorkerError("deadline")
                if time.monotonic() >= next_gate:
                    check_admission(deadline, admission)
                    next_gate = time.monotonic() + 5
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise WorkerError("snapshot_hash_mismatch")
        check_admission(deadline, admission)
        next_gate = time.monotonic() + 5
    runtime_suffixes = {".json", ".jinja", ".safetensors", ".model", ".txt", ".py"}
    actual = {p.name for p in folder.iterdir() if p.suffix in runtime_suffixes}
    if manifest.parent == folder:
        actual.discard(manifest.name)
    if not actual <= names:
        raise WorkerError("unpinned_runtime_file")
    return snapshot_identity(entries)


@dataclass(frozen=True)
class WorkerConfig:
    model_dir: str
    model_manifest: str
    state_dir: str
    image_roots: tuple[str, ...]
    adapter_dir: str | None = None
    adapter_manifest: str | None = None
    max_seconds: int = 1800
    max_requests: int = 100
    startup_seconds: int = 300

    def __post_init__(self):
        paths = [self.model_dir, self.model_manifest, self.state_dir, *self.image_roots]
        paths += [p for p in (self.adapter_dir, self.adapter_manifest) if p is not None]
        if (not self.image_roots or any(not isinstance(p, str) or not Path(p).is_absolute()
                                     or ".." in Path(p).parts for p in paths)
                or bool(self.adapter_dir) != bool(self.adapter_manifest)
                or type(self.max_seconds) is not int or not 1 <= self.max_seconds <= 23400
                or type(self.max_requests) is not int or not 1 <= self.max_requests <= 2000
                or type(self.startup_seconds) is not int or not 1 <= self.startup_seconds <= 300):
            raise WorkerError("invalid_worker_config")


def validate_request(value: dict) -> dict:
    fields = {"version", "id", "context", "image_path", "image_sha256", "thinking",
              "thinking_budget", "max_tokens", "timeout_seconds", "image_side",
              "image_tokens", "seed", "context_tokens"}
    if not isinstance(value, dict) or not fields <= set(value) or set(value) - fields - {"image_crop"}:
        raise WorkerError("invalid_request")
    value = {"image_crop": "none", **value}
    if value["image_crop"] not in {"none", "center_80"}:
        raise WorkerError("invalid_image_crop")
    if (value["version"] != PROTOCOL or not isinstance(value["id"], str)
            or not IDENTITY.fullmatch(value["id"]) or type(value["thinking"]) is not bool
            or not isinstance(value["context"], dict)
            or not isinstance(value["context"].get("evidence"), list)
            or not value["context"]["evidence"]):
        raise WorkerError("invalid_request")
    for item in value["context"]["evidence"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise WorkerError("invalid_context")
    limits = {"thinking_budget": (0, 512), "max_tokens": (32, 2048),
              "timeout_seconds": (1, 300), "seed": (0, 2**32 - 1)}
    for key, (low, high) in limits.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise WorkerError("invalid_request_budget")
    if (type(value["image_side"]) is not int or value["image_side"] not in {1024, 1600, 2048}
            or type(value["image_tokens"]) is not int or value["image_tokens"] not in {512, 1024, 1536}
            or type(value["context_tokens"]) is not int or value["context_tokens"] not in {2048, 4096}
            or value["thinking_budget"] >= value["max_tokens"]):
        raise WorkerError("invalid_request_budget")
    if value["image_path"] is not None:
        if (not isinstance(value["image_path"], str) or not Path(value["image_path"]).is_absolute()
                or not isinstance(value["image_sha256"], str)
                or not SHA256.fullmatch(value["image_sha256"])):
            raise WorkerError("invalid_image")
    elif value["image_sha256"] is not None:
        raise WorkerError("invalid_image")
    encode_frame(value)  # Also rejects non-finite values and oversize contexts.
    return value


def context_for_view(context, crop):
    if crop == "none":
        return context
    if crop != "center_80":
        raise WorkerError("invalid_image_crop")
    result = copy.deepcopy(context)
    for item in result.get("evidence", []):
        if item.get("source") in {"screen_context", "synthetic_screen"}:
            item["image_view"] = {"mode": crop, "scope": "center_80_percent_each_axis;OCR_and_metadata_cover_full_observation"}
    return result


def crop_image(image, mode):
    if mode == "none":
        return image
    if mode != "center_80":
        raise WorkerError("invalid_image_crop")
    x, y = image.width // 10, image.height // 10
    cropped = image.crop((x, y, image.width - x, image.height - y))
    image.close()
    return cropped


def encode_frame(value: dict) -> bytes:
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(data) > MAX_FRAME:
        raise WorkerError("frame_too_large")
    return data


class JsonPipe:
    def __init__(self, read_fd: int, write_fd: int):
        self.read_fd, self.write_fd = read_fd, write_fd
        self.buffer = bytearray()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)

    def receive(self, deadline: float) -> dict | None:
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.read_fd], [], [], remaining)[0]:
                raise WorkerError("deadline")
            chunk = os.read(self.read_fd, min(4096, MAX_FRAME + 1 - len(self.buffer)))
            if not chunk:
                if self.buffer:
                    raise WorkerError("truncated_frame")
                return None
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_FRAME:
                raise WorkerError("frame_too_large")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, UnicodeError) as error:
            raise WorkerError("invalid_json") from error
        if not isinstance(value, dict):
            raise WorkerError("invalid_json")
        return value

    def send(self, value: dict, deadline: float):
        data = memoryview(encode_frame(value))
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.write_fd], [], remaining)[1]:
                raise WorkerError("deadline")
            written = os.write(self.write_fd, data)
            data = data[written:]


def validate_adapter(config: dict, weights: dict, modules: dict) -> None:
    """Reject ignored keys and wrong ranks/shapes before attaching any layers."""
    params = config.get("lora_parameters", {})
    keys, rank = params.get("keys"), params.get("rank")
    if (config.get("fine_tune_type", "lora") != "lora" or not isinstance(keys, list)
            or not keys or any(not isinstance(k, str) or not k.startswith("language_model.") for k in keys)
            or len(set(keys)) != len(keys) or type(rank) is not int or not 1 <= rank <= 64):
        raise WorkerError("invalid_adapter_config")
    for key in ("scale", "dropout"):
        number = params.get(key)
        if type(number) not in {int, float} or not math.isfinite(number):
            raise WorkerError("invalid_adapter_config")
    if params["scale"] <= 0 or not 0 <= params["dropout"] < 1:
        raise WorkerError("invalid_adapter_config")
    expected = {}
    for key in keys:
        if key not in modules:
            raise WorkerError("adapter_target_missing")
        inputs, outputs = modules[key]
        expected[key + ".lora_a"] = (inputs, rank)
        expected[key + ".lora_b"] = (rank, outputs)
    if set(weights) != set(expected):
        raise WorkerError("adapter_key_mismatch")
    if any(tuple(weights[key].shape) != shape for key, shape in expected.items()):
        raise WorkerError("adapter_shape_mismatch")


def configure_pixels(processor, tokens: int) -> None:
    image_processor = processor.image_processor
    factor = image_processor.patch_size * image_processor.merge_size
    if factor != 32 or tokens not in {512, 1024, 1536}:
        raise WorkerError("unexpected_image_processor_budget")
    pixels = tokens * factor**2
    image_processor.min_pixels, image_processor.max_pixels = min(65536, pixels), pixels
    image_processor.size = {"shortest_edge": image_processor.min_pixels, "longest_edge": pixels}


class MlxBackend:
    def __init__(self, config: WorkerConfig, deadline: float, admission=gate):
        check_admission(deadline, admission)
        if any(importlib.metadata.version(name) != version for name, version in PACKAGES.items()):
            raise WorkerError("runtime_version_mismatch")
        self.model_sha = verify_snapshot(Path(config.model_dir), Path(config.model_manifest), deadline, admission)
        metadata = read_json(Path(config.model_manifest))
        self.template_sha = next((entry["sha256"] for entry in metadata["files"]
                                  if entry["name"] == "chat_template.jinja"), None)
        self.config = read_json(Path(config.model_dir) / "config.json")
        text = self.config.get("text_config", {})
        if (self.config.get("model_type") != "qwen3_5" or text.get("hidden_size") != 4096
                or text.get("num_hidden_layers") != 32):
            raise WorkerError("requires_qwen3_5_9b")
        self.adapter_sha = None
        if config.adapter_dir:
            self.adapter_sha = verify_snapshot(Path(config.adapter_dir), Path(config.adapter_manifest), deadline, admission)
            if read_json(Path(config.adapter_manifest)).get("base_model_sha256") != self.model_sha:
                raise WorkerError("adapter_base_mismatch")
        check_admission(deadline, admission)
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.generate import dispatch
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        self.mx, self.dispatch = mx, dispatch
        self.make_cache, self.template, self.prepare_inputs = make_prompt_cache, apply_chat_template, prepare_inputs
        # Override only this worker's imported helper; never raise the wired limit
        # to consume the device-wide recommended maximum with other apps open.
        dispatch.wired_limit = lambda *args, **kwargs: contextlib.nullcontext()
        mx.set_cache_limit(256 * 1024**2)
        started = time.monotonic()
        self.model, self.processor = load(config.model_dir, lazy=False, strict=True,
                                          trust_remote_code=False, local_files_only=True)
        if config.adapter_dir:
            self._load_adapter(Path(config.adapter_dir))
        self.model.eval()
        mx.synchronize()
        self.load_seconds = time.monotonic() - started
        check_admission(deadline, admission)

    def _load_adapter(self, folder: Path):
        import mlx.nn as nn
        from mlx_vlm.trainer.utils import _apply_lora_layers

        config = read_json(folder / "adapter_config.json")
        weights = self.mx.load(str(folder / "adapters.safetensors"))
        modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                outputs, inputs = module.weight.shape
                if isinstance(module, nn.QuantizedLinear):
                    inputs = inputs * 32 // module.bits
                modules[name] = (inputs, outputs)
        validate_adapter(config, weights, modules)
        self.model.freeze()
        _apply_lora_layers(self.model, config)
        from mlx.utils import tree_flatten

        if {k for k, _ in tree_flatten(self.model.trainable_parameters())} != set(weights):
            raise WorkerError("adapter_attached_key_mismatch")
        self.model.load_weights(list(weights.items()), strict=False)
        self.mx.eval(self.model.parameters())

    def reset(self, seed: int):
        self.mx.synchronize()
        lm = self.model.language_model
        lm._position_ids = None
        lm._rope_deltas = None
        self.processor.tokenizer.stopping_criteria.reset(self.model.config.eos_token_id)
        self.processor.tokenizer.thinking_budget_criteria = None
        self.mx.random.seed(seed)
        self.mx.clear_cache()
        self.mx.reset_peak_memory()
        return self.make_cache(lm)

    def generate(self, request: dict, config: WorkerConfig, deadline: float, admission=gate):
        from PIL import Image

        started = time.monotonic()
        cache = self.reset(request["seed"])
        reset_seconds = time.monotonic() - started
        image = None
        iterator = None
        preparation_started = time.monotonic()
        metrics = {}
        try:
            if request["image_path"] is not None:
                path = private_path(Path(request["image_path"]))
                if not any(path.is_relative_to(Path(root)) for root in config.image_roots):
                    raise WorkerError("image_outside_roots")
                if path.stat().st_size > 32 * 1024**2:
                    raise WorkerError("image_too_large")
                with os.fdopen(open_private_file(path, os.O_RDONLY), "rb") as source:
                    image_bytes = source.read(32 * 1024**2 + 1)
                if len(image_bytes) > 32 * 1024**2:
                    raise WorkerError("image_too_large")
                if hashlib.sha256(image_bytes).hexdigest() != request["image_sha256"]:
                    raise WorkerError("image_hash_mismatch")
                with Image.open(io.BytesIO(image_bytes)) as source:
                    if source.width * source.height > 32_000_000:
                        raise WorkerError("image_too_large")
                    image = source.convert("RGB")
                    image = crop_image(image, request["image_crop"])
                    image.thumbnail((request["image_side"], request["image_side"]), Image.Resampling.LANCZOS)
            check_admission(deadline, admission)
            formatted = self.template(self.processor, self.config, prompt(request["context"]),
                                      num_images=int(image is not None), enable_thinking=request["thinking"])
            processor = self.processor.image_processor
            # MLX-VLM 0.7.6's native-input wrapper does not forward pixel-budget
            # keyword arguments. Set the pinned Qwen processor's actual config.
            configure_pixels(self.processor, request["image_tokens"])
            factor = processor.patch_size * processor.merge_size
            inputs = self.prepare_inputs(self.processor, images=[image] if image else None,
                                         prompts=formatted, add_special_tokens=False,
                                         min_pixels=min(65536, request["image_tokens"] * factor**2),
                                         max_pixels=request["image_tokens"] * factor**2)
            if inputs["input_ids"].size + request["max_tokens"] > request["context_tokens"]:
                raise WorkerError("context_budget")
            if image is not None:
                actual = int((inputs["input_ids"] == self.config["image_token_id"]).sum().item())
                if actual > request["image_tokens"]:
                    raise WorkerError("image_token_budget")
            else:
                actual = 0
            grid = inputs.get("image_grid_thw")
            prepared_width = prepared_height = None
            if grid is not None and getattr(grid, "shape", None) == (1, 3):
                prepared_height = int(grid[0][1].item()) * processor.patch_size
                prepared_width = int(grid[0][2].item()) * processor.patch_size
            inputs["mask"] = inputs.pop("attention_mask", None)
            self.mx.synchronize()
            image_prepare_seconds = time.monotonic() - preparation_started
            check_admission(deadline, admission)
            iterator = self.dispatch.stream_generate(
                self.model, self.processor, formatted, prompt_cache=cache, **inputs,
                max_tokens=request["max_tokens"], temperature=0.0, top_k=0,
                top_p=1.0, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                enable_thinking=request["thinking"],
                thinking_budget=request["thinking_budget"] if request["thinking"] else None,
                prefill_step_size=256, seed=request["seed"], verbose=False,
            )
            output, first_token, last = "", None, None
            next_gate = 0.0
            for part in iterator:
                if first_token is None:
                    first_token = time.monotonic()
                if time.monotonic() >= deadline:
                    raise WorkerError("deadline")
                if time.monotonic() >= next_gate:
                    check_admission(deadline, admission)
                    next_gate = time.monotonic() + 5
                output += part.text
                if len(output.encode()) > MAX_OUTPUT:
                    raise WorkerError("output_too_large")
                last = part
            self.mx.synchronize()
            if last is None or last.finish_reason != "stop":
                raise WorkerError("output_truncated")
            prompt_rate = getattr(last, "prompt_tps", 0)
            metrics.update({"prompt_tokens": last.prompt_tokens,
                            "generated_tokens": last.generation_tokens,
                            "prompt_seconds": last.prompt_tokens / prompt_rate if prompt_rate > 0 else None,
                            "tokens_per_second": getattr(last, "generation_tps", None),
                            "image_prepare_seconds": image_prepare_seconds,
                            "vision_tokens": actual,
                            "image_prepared_width": prepared_width, "image_prepared_height": prepared_height,
                            "generation_seconds": time.monotonic() - first_token,
                            "time_to_first_token_seconds": first_token - started,
                            "mlx_active_bytes": self.mx.get_active_memory(),
                            "mlx_cache_bytes": self.mx.get_cache_memory(),
                            "mlx_peak_bytes": self.mx.get_peak_memory(),
                            "reset_prepare_seconds": reset_seconds})
            return output, metrics
        finally:
            cleanup_started = time.monotonic()
            if iterator is not None:
                iterator.close()
            cache.clear()
            if image is not None:
                image.close()
            self.reset(request["seed"])
            metrics.update(cleanup_seconds=time.monotonic() - cleanup_started,
                           request_total_seconds=time.monotonic() - started)


def handle_request(backend, request: dict, config: WorkerConfig, deadline: float, admission=gate):
    request = validate_request(request)
    check_admission(deadline, admission)
    output, metrics = backend.generate(request, config, deadline, admission)
    safety = raw_safety_checks(output)
    if safety.get("available") and any(safety.get(key) for key in ("sensitive_keyword", "email", "url", "unsupported_completion")):
        return {"version": PROTOCOL, "id": request["id"], "status": "error", "code": "raw_safety_failure", "raw_safety_checks": safety}
    try:
        result = parse_result(output, request["context"])
    except (TypeError, ValueError) as error:
        error.raw_safety_checks = safety
        raise
    memory = {"scope": "mlx_allocator"}
    for name in ("active_bytes", "cache_bytes", "peak_bytes"):
        memory[name] = metrics.pop("mlx_" + name, None)
    return {"version": PROTOCOL, "id": request["id"], "status": "complete",
            "result": result, "engine_metrics": metrics, "memory_metrics": memory,
            "raw_safety_checks": safety}


def serve(config: WorkerConfig, channel, deadline: float, backend_factory=MlxBackend,
          admission=gate, watchdog_factory=None):
    check_admission(deadline, admission)
    with contextlib.redirect_stdout(__import__("sys").stderr):
        backend = backend_factory(config, deadline, admission)
    check_admission(deadline, admission)
    channel.send({"version": PROTOCOL, "status": "ready", "model_sha256": backend.model_sha,
                  "adapter_sha256": backend.adapter_sha, "load_seconds": backend.load_seconds,
                  "template_sha256": getattr(backend, "template_sha", None)}, deadline)
    seen = set()
    for _ in range(config.max_requests):
        while True:
            check_admission(deadline, admission)
            try:
                value = channel.receive(min(deadline, time.monotonic() + 5))
                break
            except WorkerError as error:
                if str(error) != "deadline":
                    raise
        if value is None or value == {"version": PROTOCOL, "op": "shutdown"}:
            return
        request = validate_request(value)
        if request["id"] in seen:
            raise WorkerError("duplicate_request")
        seen.add(request["id"])
        request_deadline = min(deadline, time.monotonic() + request["timeout_seconds"])
        request_timer = (watchdog_factory or lifetime_watchdog)(request_deadline)
        try:
            with contextlib.redirect_stdout(__import__("sys").stderr):
                response = handle_request(backend, request, config, request_deadline, admission)
        except Exception as error:
            code = str(error) if isinstance(error, WorkerError) else "invalid_output" if isinstance(error, ValueError) else "inference_error"
            failure = {"version": PROTOCOL, "id": request["id"], "status": "error", "code": code}
            if isinstance(getattr(error, "raw_safety_checks", None), dict):
                failure["raw_safety_checks"] = error.raw_safety_checks
            channel.send(failure, deadline)
            # Failed/cancelled generations are not reused, even if reset failed.
            return
        finally:
            request_timer.cancel()
        channel.send(response, deadline)


class WorkerController:
    def __init__(self, config: WorkerConfig, *, python: Path = PROJECT / ".ml-venv/bin/python"):
        self.config, self.python = config, Path(python)
        self.child = None
        self.lock_fd = None
        self.channel = None
        self.serial = threading.Lock()
        self.count = 0
        self.seen = set()

    def _command(self):
        return ["/usr/bin/sandbox-exec", "-f", str(PROJECT / "network-off.sb"),
                str(self.python), str(Path(__file__).resolve()), "--worker-config",
                json.dumps(asdict(self.config)), "--lock-fd", str(self.lock_fd),
                "--deadline", str(self.deadline)]

    def start(self):
        if self.child is not None:
            raise WorkerError("already_started")
        available = remaining_seconds(datetime.now(timezone.utc), end=VISION_END)
        self.deadline = time.monotonic() + min(self.config.max_seconds, available)
        check_admission(self.deadline)
        self.lock_fd = open_private_file(Path(self.config.state_dir) / "local-model-execution.lock")
        attempt = None
        try:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ModelBusy("local_model_busy") from error
            env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                   "LANG": "en_US.UTF-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
            attempt = Attempt(Path(self.config.state_dir), [],
                              {"stage": "vision_startup", "engine_version": "mlx-vlm-0.7.6",
                               "variant": "mlx_resident"})
            self.child = subprocess.Popen(self._command(), stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          pass_fds=(self.lock_fd,), env=env, start_new_session=True)
            attempt.start_sampling(self.child.pid)
            self.channel = JsonPipe(self.child.stdout.fileno(), self.child.stdin.fileno())
            ready = self.channel.receive(min(self.deadline, time.monotonic() + self.config.startup_seconds))
            if not ready or ready.get("version") != PROTOCOL or ready.get("status") != "ready":
                raise WorkerError("worker_start_failed")
            self.ready = ready
            attempt.data["result_status"] = "not_applicable"
            attempt.finish("complete", returncode=0, engine_metrics={"load_seconds": ready.get("load_seconds")})
            self.ready["startup_telemetry_attempt_id"] = attempt.identity
            return self
        except BaseException:
            self.close()
            if attempt is not None:
                attempt.finish("worker_start_failed")
            raise

    def request(self, context: dict, *, image_path: str | None = None,
                image_sha256: str | None = None, thinking=False, thinking_budget=256,
                max_tokens=768, timeout_seconds=180, image_side=1600,
                image_tokens=1024, context_tokens=4096, seed=0, request_id=None, image_crop="none",
                experiment_sha256=None):
        if not self.serial.acquire(blocking=False):
            raise ModelBusy("worker_request_busy")
        attempt = None
        try:
            if self.child is None or self.child.poll() is not None:
                raise WorkerError("worker_not_running")
            check_admission(self.deadline)
            context = context_for_view(context, image_crop)
            request = validate_request({"version": PROTOCOL, "id": request_id or uuid.uuid4().hex,
                                        "context": context, "image_path": image_path,
                                        "image_sha256": image_sha256, "thinking": thinking,
                                        "thinking_budget": thinking_budget, "max_tokens": max_tokens,
                                        "timeout_seconds": timeout_seconds, "image_side": image_side,
                                        "image_tokens": image_tokens, "seed": seed,
                                        "context_tokens": context_tokens, "image_crop": image_crop})
            if self.count >= self.config.max_requests or request["id"] in self.seen:
                raise WorkerError("request_limit_or_duplicate")
            if self.deadline - time.monotonic() < timeout_seconds + 5:
                raise WorkerError("insufficient_window_for_request")
            self.seen.add(request["id"])
            self.count += 1
            attempt = Attempt(Path(self.config.state_dir),
                              ["-c", str(context_tokens), "-n", str(max_tokens), "--image-max-tokens", str(image_tokens),
                               "--thinking", str(int(thinking)), "--thinking-budget", str(thinking_budget),
                               "--image-side", str(image_side), "--center-crop", str(int(image_crop == "center_80")),
                               "--seed", str(seed), "--request-timeout", str(timeout_seconds)],
                              {"stage": "vision", "variant": "mlx_resident", "cache_state": "fresh",
                               "engine_version": "mlx-vlm-0.7.6", "prompt_version": "activity_context_v1",
                               "experiment_sha256": experiment_sha256,
                               "model_sha256": self.ready["model_sha256"],
                               "adapter_sha256": self.ready.get("adapter_sha256"),
                               "template_sha256": self.ready.get("template_sha256"),
                               "engine_sha256": fingerprint({"worker": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "packages": PACKAGES}),
                               "prompt_sha256": hashlib.sha256(prompt(context).encode()).hexdigest(),
                               "input_sha256": fingerprint({"context": context, "image_sha256": image_sha256,
                                   "parameters": {key: request[key] for key in ("thinking", "thinking_budget", "max_tokens",
                                       "timeout_seconds", "image_side", "image_tokens", "context_tokens", "seed", "image_crop")}})})
            attempt.start_sampling(self.child.pid)
            deadline = min(self.deadline, time.monotonic() + timeout_seconds)
            self.channel.send(request, deadline)
            response = self.channel.receive(deadline)
            if (not response or response.get("version") != PROTOCOL or response.get("id") != request["id"]):
                raise WorkerError("invalid_worker_response")
            if response.get("status") != "complete":
                error = WorkerError(response.get("code", "inference_error"))
                flags = response.get("raw_safety_checks", {})
                error.raw_safety_checks = {key: flags[key] for key in ("available", "sensitive_keyword", "email", "url", "unsupported_completion") if type(flags.get(key)) is bool}
                raise error
            response["result"] = parse_result(json.dumps(response["result"]), context)
            metrics = response.get("engine_metrics", {})
            if not isinstance(metrics, dict):
                raise WorkerError("invalid_worker_metrics")
            memory = response.get("memory_metrics", {})
            if not isinstance(memory, dict):
                raise WorkerError("invalid_worker_metrics")
            bounded = {"scope": "mlx_allocator"}
            for key in ("active_bytes", "cache_bytes", "peak_bytes"):
                value = memory.get(key)
                if value is not None and (type(value) not in {int, float}
                                          or not math.isfinite(value) or value < 0):
                    raise WorkerError("invalid_worker_metrics")
                bounded[key] = value
            attempt.data["worker_memory"] = bounded
            response["memory_metrics"] = bounded
            metrics["load_seconds"] = 0  # Loading belongs to the separate startup attempt.
            attempt.data["startup_attempt_id"] = self.ready.get("startup_telemetry_attempt_id")
            attempt.data["result_status"] = "complete"  # Both protocol and result schema were validated.
            attempt.finish("complete", returncode=0, engine_metrics=metrics)
            response["telemetry_attempt_id"] = attempt.identity
            response["telemetry_available"] = not attempt.error
            return response
        except BaseException as error:
            child = self.child
            self.close()
            if attempt is not None:
                attempt.finish(str(error) if isinstance(error, WorkerError) else "worker_error",
                               returncode=child.returncode if child else None)
                error.telemetry_attempt_id = attempt.identity
            raise
        finally:
            self.serial.release()

    def close(self):
        child = self.child
        try:
            if child is not None:
                if child.poll() is None:
                    try:
                        self.channel.send({"version": PROTOCOL, "op": "shutdown"}, time.monotonic() + 0.2)
                        child.wait(timeout=0.5)
                    except (OSError, WorkerError, subprocess.TimeoutExpired, AttributeError):
                        child.terminate()
                        try:
                            child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                for stream in (child.stdin, child.stdout):
                    if stream is not None:
                        stream.close()
        finally:
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
            self.child = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


class KernelDeadline:
    """Nested process deadlines whose default signal action needs no Python/GIL."""

    def __init__(self, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            os._exit(124)
        self.previous_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        previous, _ = signal.setitimer(signal.ITIMER_REAL, remaining)
        self.previous_deadline = time.monotonic() + previous if previous else None

    def cancel(self):
        if self.previous_deadline is None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            remaining = self.previous_deadline - time.monotonic()
            if remaining <= 0:
                os._exit(124)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        signal.signal(signal.SIGALRM, self.previous_handler)


def lifetime_watchdog(deadline: float):
    """Test/embed safeguard; the owned production process uses KernelDeadline."""
    timer = threading.Timer(max(0, deadline - time.monotonic()), os._exit, args=(124,))
    timer.daemon = True
    timer.start()
    return timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", required=True, help="WorkerConfig JSON; requests use stdin")
    parser.add_argument("--lock-fd", required=True, type=int)
    parser.add_argument("--deadline", required=True, type=float)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if not math.isfinite(args.deadline):
            raise WorkerError("invalid_deadline")
        config = WorkerConfig(**json.loads(args.worker_config))
        info = os.fstat(args.lock_fd)
        path = Path(config.state_dir) / "local-model-execution.lock"
        expected = private_path(path).stat()
        if (info.st_ino, info.st_dev) != (expected.st_ino, expected.st_dev):
            raise WorkerError("invalid_lock_descriptor")
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = min(args.deadline, time.monotonic() + config.max_seconds,
                       time.monotonic() + remaining_seconds(datetime.now(timezone.utc), end=VISION_END))
        channel = JsonPipe(0, 1)
        timer = KernelDeadline(deadline)
        try:
            serve(config, channel, deadline, watchdog_factory=KernelDeadline)
        finally:
            timer.cancel()
    except Exception:
        # No private request, path, prompt, output or exception body in stderr.
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
