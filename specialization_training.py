"""Bounded local Qwen3.5 QLoRA; dry-run never reads data or loads MLX.

Heavy stages are launched by the parent under network-off.sb and the shared
model_execution lock. Only the root's explicit train/validation exports are
accepted. Private text, paths, predictions and exceptions never reach stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

VERSION = "specialization_training_v1"
MLX_VLM_VERSION = "0.7.6"
MODEL_REPOSITORY = "mlx-community/Qwen3.5-9B-MLX-4bit"
MODEL_REVISION = "938d8919941c6e7efd3c7150eff7fe9d12afa631"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODEL_FILES = {"config.json", "chat_template.jinja", "model.safetensors.index.json",
               "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
               "tokenizer.json", "video_preprocessor_config.json", "vocab.json",
               "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
PINNED_WEIGHTS = {
    "model-00001-of-00002.safetensors": (5349771222, "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd"),
    "model-00002-of-00002.safetensors": (600449850, "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d"),
}
CHECKPOINT_NAME = re.compile(r"step-\d{6}-[a-f0-9]{8}\Z")
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
RUNTIME_PACKAGES = {"mlx-vlm": "0.7.6", "mlx": "0.32.3", "mlx-metal": "0.32.3",
                    "transformers": "5.18.0", "numpy": "2.4.6",
                    "torch": "2.12.1", "torchvision": "0.27.1", "pillow": "12.3.0", "tokenizers": "0.23.2"}


class TrainingError(RuntimeError):
    """Messages are fixed safe codes, never underlying dataset/package errors."""


class BudgetEnded(TrainingError):
    pass


@dataclass(frozen=True)
class Recipe:
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-5
    batch_size: int = 1
    accumulation: int = 8
    sequence_length: int = 2048
    epochs: int = 2
    seed: int = 20261005
    image_tokens: int = 512
    generation_tokens: int = 512
    memory_limit_gib: int = 10

    def __post_init__(self):
        if not 1 <= self.epochs <= 2 or self.image_tokens not in {512, 1024}:
            raise TrainingError("invalid_recipe")
        if not 8 <= self.memory_limit_gib <= 12 or not 0 <= self.seed < 2**32:
            raise TrainingError("invalid_resource_or_seed")
        if (self.rank, self.alpha, self.learning_rate, self.batch_size,
                self.accumulation, self.sequence_length, self.generation_tokens) != (
                    8, 16, 1e-5, 1, 8, 2048, 512):
            raise TrainingError("recipe_changes_require_new_experiment")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def private_path(path: Path, *, directory=False) -> Path:
    """Read-only checks: no implicit chmod/create of supplied inputs."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise TrainingError("input_path_not_local_regular")
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise TrainingError("input_not_private_owned")
    if not directory and info.st_nlink != 1:
        raise TrainingError("input_has_multiple_links")
    return path


def read_private(path: Path, limit=16 * 1024**2) -> bytes:
    private_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            raise TrainingError("input_not_private_owned")
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise TrainingError("input_too_large")
    return payload


def file_digest(path: Path) -> str:
    private_path(path)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_export(path: Path, split: str) -> tuple[list[dict], str]:
    from activity_context import prompt, validate

    if split not in {"train", "validation"}:
        raise TrainingError("sealed_test_not_supported")
    payload = read_private(path)
    rows = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise TrainingError("invalid_export_identity")
        if row.get("split", split) != split:
            raise TrainingError("wrong_export_split")
        messages, images, context = row.get("messages"), row.get("images"), row.get("context")
        if (not isinstance(messages, list) or len(messages) != 2
                or not all(isinstance(m, dict) for m in messages)
                or [m.get("role") for m in messages] != ["user", "assistant"]
                or not all(isinstance(m.get("content"), str) for m in messages)
                or not isinstance(images, list) or len(images) != 1
                or not isinstance(images[0], str) or not isinstance(context, dict)):
            raise TrainingError("invalid_export_record")
        if messages[0]["content"] != prompt(context):
            raise TrainingError("stale_context_prompt")
        row["target"] = validate(json.loads(messages[1]["content"]), context)
        row["synthetic"] = any(e.get("source") == "synthetic_screen"
                               for e in context.get("evidence", []))
        if split == "validation" and row["synthetic"]:
            raise TrainingError("synthetic_validation_forbidden")
        private_path(Path(images[0]))
        row["image_digest"] = file_digest(Path(images[0]))
        if row.get("image_sha256", row["image_digest"]) != row["image_digest"]:
            raise TrainingError("export_image_hash_mismatch")
        if row.get("source_class", "synthetic" if row["synthetic"] else "real") != (
                "synthetic" if row["synthetic"] else "real"):
            raise TrainingError("export_source_class_mismatch")
        rows.append(row)
    if not rows or len(rows) > 240 or len({r["id"] for r in rows}) != len(rows):
        raise TrainingError("invalid_export_count_or_duplicate_identity")
    return rows, hashlib.sha256(payload).hexdigest()


def check_partitions(train: list[dict], validation: list[dict]) -> None:
    for key in ("id", "image_digest", "episode_id", "near_duplicate_group"):
        left = {r[key] for r in train if r.get(key)}
        right = {r[key] for r in validation if r.get(key)}
        if left & right:
            raise TrainingError("cross_split_leakage")
    if len(validation) != 20:
        raise TrainingError("validation_requires_20_real_examples")
    if sum(r["synthetic"] for r in train) < 2:
        raise TrainingError("two_synthetic_controls_required")


def completion_layout(prefix_ids, answer_ids, eos_id: int, image_id: int,
                      max_length=2048) -> tuple[list[int], list[int]]:
    """Append separately tokenized targets to the exact generation prefix."""
    prefix, answer = list(prefix_ids), list(answer_ids)
    if not prefix or not answer or image_id not in prefix or image_id in answer:
        raise TrainingError("invalid_image_or_completion_tokens")
    target = answer + [eos_id]
    if len(prefix) + len(target) > max_length:
        raise TrainingError("example_exceeds_sequence_limit")
    return prefix + target, [0] * len(prefix) + [1] * len(target)


def shifted_loss_mask(attention, completion) -> list[int]:
    if len(attention) != len(completion) or len(completion) < 2:
        raise TrainingError("invalid_completion_mask")
    result = [int(bool(a) and bool(c)) for a, c in zip(attention[1:], completion[1:])]
    if not any(result):
        raise TrainingError("empty_assistant_loss")
    return result


def assert_adapter_keys(keys: list[str]) -> None:
    if not keys or len(keys) != 128 or len(set(keys)) != len(keys):
        raise TrainingError("unexpected_adapter_target_count")
    for key in keys:
        suffix = key.rsplit(".", 1)[-1]
        if (not key.startswith("language_model.") or suffix not in TARGETS
                or ".linear_attn." in key
                or not (".mlp." in key or ".self_attn." in key)):
            raise TrainingError("unsafe_adapter_target")
    counts = {suffix: sum(k.endswith("." + suffix) for k in keys) for suffix in TARGETS}
    if counts != {s: 32 if s in {"gate_proj", "up_proj", "down_proj"} else 8 for s in TARGETS}:
        raise TrainingError("unexpected_hybrid_target_coverage")
    expected = {f"language_model.model.layers.{layer}.{branch}.{suffix}"
                for layer in range(32)
                for branch, suffixes in (("mlp", ("gate_proj", "up_proj", "down_proj")),
                                         ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")
                                          if (layer + 1) % 4 == 0 else ()))
                for suffix in suffixes}
    if set(keys) != expected:
        raise TrainingError("unexpected_hybrid_target_layout")


def score_generations(rows: list[dict], outputs: list[str]) -> dict:
    from activity_context import COMPLETION, parse_result

    if len(rows) != len(outputs) or not rows:
        raise TrainingError("incomplete_validation_generations")
    scores = {"examples": len(rows), "schema_valid": 0, "completion_claim_flags": 0,
              "activity_correct": 0, "project_correct": 0, "task_correct": 0,
              "joint_candidates_correct": 0, "claim_evidence_exact": 0,
              "affirmative_candidates": 0, "correct_affirmative_candidates": 0,
              "abstentions": 0, "correct_abstentions": 0}
    for row, output in zip(rows, outputs):
        scores["completion_claim_flags"] += bool(COMPLETION.search(output))
        try:
            value = parse_result(output, row["context"])
        except (ValueError, TypeError, KeyError):
            continue
        gold = row["target"]
        scores["schema_valid"] += 1
        for field, counter in (("activity_kind", "activity_correct"),
                               ("project_candidate", "project_correct"),
                               ("task_candidate", "task_correct")):
            scores[counter] += value[field] == gold[field]
        scores["joint_candidates_correct"] += all(value[k] == gold[k]
            for k in ("project_candidate", "task_candidate"))
        scores["claim_evidence_exact"] += all(set(value["claim_evidence"][k]) == set(v)
                                              for k, v in gold["claim_evidence"].items())
        for field in ("project_candidate", "task_candidate"):
            affirmative = value[field] is not None
            scores["affirmative_candidates"] += affirmative
            scores["correct_affirmative_candidates"] += affirmative and value[field] == gold[field]
            scores["abstentions"] += not affirmative
            scores["correct_abstentions"] += not affirmative and gold[field] is None
    return scores


def selection_key(scores: dict) -> tuple:
    return (-scores["completion_claim_flags"], scores["schema_valid"],
            scores["joint_candidates_correct"], scores["activity_correct"],
            scores["project_correct"] + scores["task_correct"], scores["claim_evidence_exact"])


def select_candidate(baseline: dict, best: dict, candidate: dict) -> bool:
    if candidate["examples"] != baseline["examples"] or best["examples"] != baseline["examples"]:
        raise TrainingError("validation_population_changed")
    if (candidate["completion_claim_flags"] > baseline["completion_claim_flags"]
            or candidate["schema_valid"] < baseline["schema_valid"]):
        return False
    return selection_key(candidate) > selection_key(best)


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def overnight_budget(max_seconds: int, now: datetime | None = None) -> float:
    from overnight_schedule import VISION_END, remaining_seconds

    if not 210 <= max_seconds <= 6 * 3600:
        raise TrainingError("invalid_time_budget")
    return max(0, min(max_seconds, remaining_seconds(now or datetime.now(timezone.utc),
                                                   end=VISION_END) - 60))


def inherited_lock_present(state_dir: Path) -> bool:
    """The sandbox child must actually inherit the existing runner's locked fd."""
    path = state_dir / "local-model-execution.lock"
    try:
        identity = path.stat()
        found = False
        for name in os.listdir("/dev/fd"):
            try:
                info = os.fstat(int(name))
                found |= (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino)
            except (ValueError, OSError):
                continue
        if not found:
            return False
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
            except BlockingIOError:
                return True
        finally:
            os.close(fd)
    except OSError:
        return False


@contextlib.contextmanager
def kernel_deadline(seconds):
    """Default SIGALRM kills native stalled calls even if the parent disappears."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise TrainingError("invalid_kernel_deadline")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class Guard:
    def __init__(self, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            raise TrainingError("invalid_time_budget")
        self.deadline = time.monotonic() + seconds
        self.last_resource_check = -math.inf

    def check(self, reserve=30):
        from vision_batch import resource_gate

        if overnight_budget(6 * 3600) < reserve or time.monotonic() + reserve >= self.deadline:
            raise BudgetEnded("training_budget_ended")
        if time.monotonic() - self.last_resource_check >= 15:
            gate = resource_gate(benchmark_now=False)
            if gate:
                raise BudgetEnded(gate)
            self.last_resource_check = time.monotonic()


def verify_assets(model_path: Path, manifest_path: Path, guard) -> str:
    private_path(model_path, directory=True)
    if manifest_path.is_relative_to(model_path):
        raise TrainingError("asset_manifest_must_be_external")
    manifest = json.loads(read_private(manifest_path))
    if (manifest.get("repository") != MODEL_REPOSITORY or manifest.get("revision") != MODEL_REVISION
            or manifest.get("version") != "mlx_assets_v1"):
        raise TrainingError("unpinned_model_assets")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    # Match the root downloader's existing JSON hash convention.
    sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if sha != manifest.get("manifest_sha256"):
        raise TrainingError("asset_manifest_mismatch")
    entries = manifest.get("files", [])
    names = [entry.get("name") for entry in entries]
    if len(names) != len(set(names)) or set(names) != MODEL_FILES:
        raise TrainingError("incomplete_asset_manifest")
    if {p.name for p in model_path.iterdir()} != MODEL_FILES:
        raise TrainingError("model_file_closure_mismatch")
    for entry in entries:
        guard.check()
        name = entry["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise TrainingError("unsafe_asset_name")
        path = model_path / name
        if name in PINNED_WEIGHTS and (entry["size"], entry["sha256"]) != PINNED_WEIGHTS[name]:
            raise TrainingError("unpinned_model_weights")
        if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
            raise TrainingError("asset_hash_mismatch")
    config = json.loads(read_private(model_path / "config.json"))
    text = config.get("text_config", {})
    if (config.get("model_type") != "qwen3_5" or text.get("num_hidden_layers") != 32
            or text.get("hidden_size") != 4096 or text.get("full_attention_interval") != 4
            or text.get("layer_types") != (["linear_attention"] * 3 + ["full_attention"]) * 8
            or config.get("quantization", {}).get("bits") != 4):
        raise TrainingError("unexpected_model_configuration")
    return sha


def snapshot_identity(entries):
    """Match specialization_worker's base-model identity, not the manifest body hash."""
    return hashlib.sha256(json.dumps(sorted(entries, key=lambda entry: entry["name"]),
                                    sort_keys=True).encode()).hexdigest()


def restore_rng(mx, keys):
    """Restore the native MLX key through seed; assigning random.state is inert."""
    if len(keys) != 1 or keys[0].shape != (2,) or keys[0].dtype != mx.uint32:
        raise TrainingError("unsupported_rng_state")
    hi, lo = keys[0].tolist()
    mx.random.seed((int(hi) << 32) | int(lo))
    if not bool(mx.array_equal(mx.random.state[0], keys[0]).item()):
        raise TrainingError("rng_restore_mismatch")


class MLXEngine:
    """Imported/constructed only inside the guarded sandbox worker."""

    def __init__(self, model_path: Path, recipe: Recipe, guard, base_model_sha256: str):
        if any(importlib.metadata.version(name) != version for name, version in RUNTIME_PACKAGES.items()):
            raise TrainingError("wrong_mlx_vlm_version")
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten, tree_map, tree_unflatten
        from mlx_vlm import generate, load
        from mlx_vlm.generate import dispatch
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.trainer.utils import get_peft_model, grad_checkpoint, save_adapter
        from mlx_vlm.utils import prepare_inputs

        self.mx, self.nn, self.optim, self.np = mx, nn, optim, np
        self.flatten, self.map, self.unflatten = tree_flatten, tree_map, tree_unflatten
        self.generate_fn, self.template, self.prepare_inputs = generate, apply_chat_template, prepare_inputs
        self.peft, self.checkpoint, self.save_adapter_fn = get_peft_model, grad_checkpoint, save_adapter
        self.recipe, self.guard = recipe, guard
        self.base_model_sha256 = base_model_sha256
        # Keep generation from temporarily raising this process's wired limit
        # to the entire device recommendation while the user's apps stay open.
        dispatch.wired_limit = lambda *args, **kwargs: contextlib.nullcontext()
        limit = recipe.memory_limit_gib * 1024**3
        mx.set_memory_limit(limit)
        mx.set_cache_limit(256 * 1024**2)
        mx.set_wired_limit(min(limit, mx.device_info()["max_recommended_working_set_size"]))
        mx.random.seed(recipe.seed)
        guard.check(180)
        self.model, self.processor = load(str(model_path), trust_remote_code=False, local_files_only=True)
        self.config = self.model.config.__dict__
        self.image_id = self.config.get("image_token_index") or self.config.get("image_token_id")
        self.tokenizer = self.processor.tokenizer
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(self.eos_id, int) or self.eos_id == self.tokenizer.unk_token_id:
            raise TrainingError("missing_end_of_turn_token")
        image_processor = self.processor.image_processor
        pixels = recipe.image_tokens * 32**2
        image_processor.max_pixels, image_processor.min_pixels = pixels, 32**2
        image_processor.size = {"longest_edge": pixels, "shortest_edge": 32**2}
        if (image_processor.patch_size, image_processor.merge_size) != (16, 2):
            raise TrainingError("unexpected_vision_patch_geometry")
        self.optimizer = optim.Adam(learning_rate=recipe.learning_rate)
        self.frozen_digest = None

    def prepare(self, row):
        self.guard.check()
        if file_digest(Path(row["images"][0])) != row["image_digest"]:
            raise TrainingError("image_source_changed")
        prompt = self.template(self.processor, self.config, row["messages"][:-1],
                               num_images=1, enable_thinking=False, add_generation_prompt=True)
        inputs = self.prepare_inputs(self.processor, images=row["images"], prompts=[prompt],
                                     image_token_index=self.image_id, add_special_tokens=False)
        mx, np = self.mx, self.np
        prefix = np.asarray(inputs["input_ids"]).reshape(-1).tolist()
        if not (np.asarray(inputs.get("attention_mask", [1] * len(prefix))) == 1).all():
            raise TrainingError("unexpected_prefix_padding")
        answer = self.tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
        ids, mask = completion_layout(prefix, answer, self.eos_id, self.image_id,
                                      self.recipe.sequence_length)
        pixels, grid = inputs.get("pixel_values"), inputs.get("image_grid_thw")
        if pixels is None or grid is None or pixels.size == 0:
            raise TrainingError("missing_pixels_or_image_grid")
        shape = np.asarray(grid).reshape(-1, 3)
        if shape.shape != (1, 3) or not (shape > 0).all():
            raise TrainingError("invalid_image_grid")
        image_count = sum(t == self.image_id for t in prefix)
        expected = int(np.prod(shape[0])) // 4
        if image_count != expected or image_count > self.recipe.image_tokens:
            raise TrainingError("image_feature_token_mismatch")
        shifted_loss_mask([1] * len(ids), mask)
        batch = {**inputs, "input_ids": mx.array([ids], dtype=mx.int32),
                 "attention_mask": mx.ones((1, len(ids)), dtype=mx.int32),
                 "completion_mask": mx.array([mask], dtype=mx.int32),
                 "image_grid_thw": mx.array(shape, dtype=mx.int32)}
        return batch

    def forward(self, batch):
        extras = {k: v for k, v in batch.items()
                  if k not in {"input_ids", "attention_mask", "pixel_values", "completion_mask"}}
        return self.model(batch["input_ids"][:, :-1], batch["pixel_values"],
                          batch["attention_mask"][:, :-1], **extras).logits

    def loss(self, model, batch):
        mx = self.mx
        extras = {k: v for k, v in batch.items()
                  if k not in {"input_ids", "attention_mask", "pixel_values", "completion_mask"}}
        labels = batch["input_ids"][:, 1:]
        logits = model(batch["input_ids"][:, :-1], batch["pixel_values"],
                       batch["attention_mask"][:, :-1], **extras).logits
        if logits.shape[:2] != labels.shape:
            raise TrainingError("logit_label_alignment_mismatch")
        mask = batch["attention_mask"][:, 1:] * batch["completion_mask"][:, 1:]
        if labels.shape[0] != 1:
            raise TrainingError("loss_requires_batch_size_one")
        positions = self.np.flatnonzero(self.np.asarray(mask[0]))
        if not positions.size:
            raise TrainingError("loss_requires_assistant_tokens")
        # The frozen vocabulary head still creates full logits. Gather before
        # fp32/CE so prompt positions do not inflate the supervised loss graph.
        indices = mx.array(positions, dtype=mx.int32)
        selected = mx.take(logits, indices, axis=1).astype(mx.float32)
        targets = mx.take(labels, indices, axis=1)
        return self.nn.losses.cross_entropy(selected, targets).mean()

    def frozen_hash(self):
        self.guard.check()
        h = hashlib.sha256()
        for name, value in sorted(self.flatten(self.model.parameters())):
            if name.endswith((".lora_a", ".lora_b")):
                continue
            self.guard.check()
            # Attaching LoRALinear nests the unchanged base under .linear.
            h.update(name.replace(".linear.", ".").encode())
            h.update(str((value.shape, value.dtype)).encode())
            raw = value.reshape(-1).view(self.mx.uint8).reshape(-1)
            for offset in range(0, raw.size, 8 * 1024**2):
                h.update(self.np.asarray(raw[offset:offset + 8 * 1024**2]).tobytes())
        return h.hexdigest()

    def attach(self):
        self.model.freeze()
        self.model = self.peft(self.model, list(TARGETS), rank=self.recipe.rank,
                               alpha=self.recipe.alpha, dropout=0.0, verbose=False)
        keys = self.model.config.lora["lora_parameters"]["keys"]
        assert_adapter_keys(keys)
        if self.model.config.lora["lora_parameters"]["scale"] != 2:
            raise TrainingError("incorrect_lora_scale")
        names = {name for name, _ in self.flatten(self.model.trainable_parameters())}
        expected = {k + suffix for k in keys for suffix in (".lora_a", ".lora_b")}
        if names != expected:
            raise TrainingError("base_or_vision_not_frozen")
        layers = self.model.language_model.model.layers
        self.checkpoint(layers[0])  # All 32 hybrid decoder instances share this class.
        self.grad_fn = self.nn.value_and_grad(self.model, self.loss)

    def gradient(self, row):
        self.guard.check(90)
        self.model.train()  # DeltaNet uses its differentiable path in training mode.
        batch = self.prepare(row)
        loss, gradient = self.grad_fn(self.model, batch)
        self.mx.eval(loss, gradient)
        leaves = [g for _, g in self.flatten(gradient)]
        if not math.isfinite(float(loss.item())) or not all(bool(self.mx.all(self.mx.isfinite(g)).item()) for g in leaves):
            raise TrainingError("nonfinite_loss_or_gradient")
        if not any(bool(self.mx.any(g != 0).item()) for g in leaves):
            raise TrainingError("zero_adapter_gradient")
        return gradient, float(loss.item())

    def add_gradients(self, previous, gradient):
        return gradient if previous is None else self.map(lambda a, b: a + b, previous, gradient)

    def update(self, gradient, count):
        if not 1 <= count <= self.recipe.accumulation:
            raise TrainingError("invalid_accumulation_count")
        self.optimizer.update(self.model, self.map(lambda g: g / count, gradient))
        self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        if not all(bool(self.mx.all(self.mx.isfinite(v)).item())
                   for _, v in self.flatten(self.model.trainable_parameters())):
            raise TrainingError("nonfinite_adapter_update")
        self.mx.clear_cache()

    def generate(self, row):
        from activity_context import parse_result
        from inference_telemetry import Attempt, record_result

        self.guard.check(90)
        self.model.eval()
        prompt = self.template(self.processor, self.config, row["messages"][:-1],
                               num_images=1, enable_thinking=False, add_generation_prompt=True)
        context = {"stage": "training_validation", "variant": "adapter_checkpoint",
                   "engine_version": "mlx-vlm-0.7.6", "prompt_version": VERSION,
                   "model_sha256": self.base_model_sha256,
                   "engine_sha256": fingerprint(RUNTIME_PACKAGES),
                   "input_sha256": fingerprint({"image": row["image_digest"], "prompt": prompt,
                                                 "recipe": asdict(self.recipe)}),
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "adapter_sha256": self.validation_adapter_sha256,
                   "experiment_sha256": self.telemetry_experiment}
        attempt = Attempt(self.telemetry_state_dir,
                          ["--image-max-tokens", str(self.recipe.image_tokens),
                           "-n", str(self.recipe.generation_tokens)], context)
        attempt.start_sampling(os.getpid())
        try:
            result = self.generate_fn(self.model, self.processor, prompt=prompt, image=row["images"],
                                      temperature=0, max_tokens=self.recipe.generation_tokens,
                                      enable_thinking=False, verbose=False, prefill_step_size=256)
            self.mx.clear_cache()
            prompt_tokens, prompt_rate = getattr(result, "prompt_tokens", None), getattr(result, "prompt_tps", 0)
            generation_tokens, generation_rate = getattr(result, "generation_tokens", None), getattr(result, "generation_tps", 0)
            attempt.data["worker_memory"] = {"scope": "mlx_allocator", "active_bytes": self.mx.get_active_memory(),
                                             "cache_bytes": self.mx.get_cache_memory(), "peak_bytes": self.mx.get_peak_memory()}
            attempt.finish("complete", engine_metrics={
                "prompt_tokens": prompt_tokens,
                "prompt_seconds": prompt_tokens / prompt_rate if prompt_tokens and prompt_rate > 0 else None,
                "generation_seconds": generation_tokens / generation_rate if generation_tokens and generation_rate > 0 else None,
                "generated_tokens": getattr(result, "generation_tokens", None),
                "tokens_per_second": getattr(result, "generation_tps", None)})
            try:
                parse_result(result.text, row["context"])
                disposition = "complete"
            except (TypeError, ValueError):
                disposition = "schema_error"
            record_result(self.telemetry_state_dir, attempt.identity, disposition)
            return result.text
        except BaseException:
            attempt.finish("interrupted_or_generation_error")
            raise

    def validate(self, rows):
        # Bind actual adapter arrays once per checkpoint, without retaining them
        # in telemetry. Each generation remains a distinct historical attempt.
        digest = hashlib.sha256()
        for name, value in sorted(self.flatten(self.model.trainable_parameters())):
            digest.update(name.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            raw = value.reshape(-1).view(self.mx.uint8).reshape(-1)
            for offset in range(0, raw.size, 8 * 1024**2):
                digest.update(self.np.asarray(raw[offset:offset + 8 * 1024**2]).tobytes())
        self.validation_adapter_sha256 = digest.hexdigest()
        return score_generations(rows, [self.generate(row) for row in rows])

    def probe(self, batch):
        self.model.eval()
        value = self.forward(batch)[:, -1, :].astype(self.mx.float32)
        self.mx.eval(value)
        return self.mx.array(self.np.asarray(value).copy())

    def assert_close(self, left, right):
        if not bool(self.mx.all(self.mx.abs(left - right) <= 1e-5).item()):
            raise TrainingError("adapter_probe_parity_failed")

    def verify(self, train, validation, run_dir):
        maximum = 0
        for row in train + validation:
            maximum = max(maximum, self.prepare(row)["input_ids"].shape[1])
            self.mx.clear_cache()
        controls = [r for r in train if r["synthetic"]][:2]
        batch = self.prepare(controls[0])
        base = self.probe(batch)
        changed_image = self.probe({**batch, "pixel_values": self.mx.zeros_like(batch["pixel_values"])})
        if not bool(self.mx.any(self.mx.abs(base - changed_image) > 1e-7).item()):
            raise TrainingError("image_does_not_affect_logits")
        self.frozen_digest = self.frozen_hash()
        self.attach()
        self.assert_close(base, self.probe(batch))
        initial = {k: self.mx.array(self.np.asarray(v).copy())
                   for k, v in self.flatten(self.model.trainable_parameters())}
        rng = [self.mx.array(x) for x in self.mx.random.state]
        gradient = None
        for i in range(self.recipe.accumulation):
            value, _ = self.gradient(controls[i % 2])
            gradient = self.add_gradients(gradient, value)
        self.update(gradient, self.recipe.accumulation)
        if not any(bool(self.mx.any(v != initial[k]).item())
                   for k, v in self.flatten(self.model.trainable_parameters())):
            raise TrainingError("adapter_update_had_no_effect")
        trained_probe = self.probe(batch)
        with tempfile.TemporaryDirectory(prefix=".verify-", dir=run_dir) as folder:
            path = Path(folder)
            self.save_adapter_fn(self.model, path / "adapters.safetensors")
            self.model.update(self.unflatten([(k, self.mx.zeros_like(v))
                              for k, v in self.flatten(self.model.trainable_parameters())]))
            self.restore_adapters(path)
            self.assert_close(trained_probe, self.probe(batch))
        if self.frozen_hash() != self.frozen_digest:
            raise TrainingError("frozen_weights_changed")
        self.model.update(self.unflatten(list(initial.items())))
        self.optimizer = self.optim.Adam(learning_rate=self.recipe.learning_rate)
        restore_rng(self.mx, rng)
        return {"examples_checked": len(train) + len(validation), "max_sequence_tokens": maximum,
                "image_alignment": True, "assistant_only_mask": True, "image_changes_logits": True,
                "finite_nonzero_gradients": True, "update_after_microbatches": self.recipe.accumulation,
                "zero_adapter_parity": True, "frozen_weights_unchanged": True, "adapter_reload_parity": True}

    def restore_adapters(self, folder):
        config = json.loads(read_private(folder / "adapter_config.json"))
        if config != self.model.config.lora:
            raise TrainingError("adapter_configuration_mismatch")
        values = self.mx.load(str(folder / "adapters.safetensors"))
        expected = dict(self.flatten(self.model.trainable_parameters()))
        if set(values) != set(expected):
            raise TrainingError("adapter_weight_keys_mismatch")
        if any(value.shape != expected[key].shape
               or not bool(self.mx.all(self.mx.isfinite(value)).item()) for key, value in values.items()):
            raise TrainingError("adapter_weight_shape_or_value_mismatch")
        self.model.load_weights(list(values.items()), strict=False)

    def save(self, folder):
        adapter = folder / "adapter"
        from private_io import prepare_directory

        prepare_directory(adapter)
        self.save_adapter_fn(self.model, adapter / "adapters.safetensors")
        manifest = {"version": "mlx_adapter_assets_v1", "model_revision": MODEL_REVISION,
                    "base_model_sha256": self.base_model_sha256,
                    "mlx_vlm": MLX_VLM_VERSION, "recipe": asdict(self.recipe),
                    "files": [{"name": p.name, "size": p.stat().st_size,
                               "sha256": file_digest(p)} for p in sorted(adapter.iterdir())]}
        manifest["manifest_sha256"] = fingerprint(manifest)
        write_state(folder / "adapter-manifest.json", manifest)
        arrays = {"optimizer." + k: v for k, v in self.flatten(self.optimizer.state)}
        if not all(isinstance(v, self.mx.array) for v in arrays.values()):
            raise TrainingError("unsupported_optimizer_state")
        arrays.update({f"rng.{i}": v for i, v in enumerate(self.mx.random.state)})
        self.mx.save_safetensors(str(folder / "optimizer.safetensors"), arrays)

    def restore(self, folder):
        verify_adapter_assets(folder / "adapter", folder / "adapter-manifest.json", self.base_model_sha256)
        self.restore_adapters(folder / "adapter")
        arrays = self.mx.load(str(folder / "optimizer.safetensors"))
        self.optimizer.state = self.unflatten([(k[10:], v) for k, v in arrays.items()
                                               if k.startswith("optimizer.")])
        restore_rng(self.mx, [arrays[f"rng.{i}"] for i in range(len(self.mx.random.state))])


def write_state(path, value):
    from private_io import write_json

    write_json(path, value)


def verify_adapter_assets(adapter_path, manifest_path, base_model_sha256=None):
    private_path(adapter_path, directory=True)
    if manifest_path.is_relative_to(adapter_path):
        raise TrainingError("adapter_manifest_must_be_external")
    manifest = json.loads(read_private(manifest_path))
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if (manifest.get("version") != "mlx_adapter_assets_v1"
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("mlx_vlm") != MLX_VLM_VERSION
            or not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("base_model_sha256", "")))
            or manifest.get("manifest_sha256") != fingerprint(body)):
        raise TrainingError("adapter_manifest_mismatch")
    if base_model_sha256 is not None and manifest["base_model_sha256"] != base_model_sha256:
        raise TrainingError("adapter_base_model_mismatch")
    expected = {"adapter_config.json", "adapters.safetensors"}
    if {p.name for p in adapter_path.iterdir()} != expected:
        raise TrainingError("adapter_file_closure_mismatch")
    records = manifest["files"]
    if len(records) != 2 or {r["name"] for r in records} != expected:
        raise TrainingError("incomplete_adapter_manifest")
    for entry in records:
        path = adapter_path / entry["name"]
        if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
            raise TrainingError("adapter_asset_hash_mismatch")


def read_state(path):
    return json.loads(read_private(path)) if path.exists() else None


def save_checkpoint(engine, run_dir, progress):
    from private_io import prepare_directory

    name = f"step-{progress['updates']:06d}-{uuid.uuid4().hex[:8]}"
    folder = run_dir / name
    prepare_directory(folder)
    engine.save(folder)
    metadata = {**progress, "checkpoint": name}
    write_state(folder / "progress.json", metadata)
    # Commit pointer only after every checkpoint file is closed and durable.
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    write_state(run_dir / "progress.json", metadata)
    # Adapters retain experiment history. Optimizer arrays are rolling resume
    # state, so only current, previous and best need this large generated file.
    keep = {name, progress.get("checkpoint"), progress.get("best", {}).get("checkpoint")}
    for candidate in run_dir.iterdir():
        if not CHECKPOINT_NAME.fullmatch(candidate.name) or candidate.name in keep:
            continue
        optimizer = candidate / "optimizer.safetensors"
        if not optimizer.exists():
            continue
        private_path(candidate, directory=True)
        private_path(optimizer)
        folder_fd = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.unlink("optimizer.safetensors", dir_fd=folder_fd)
        finally:
            os.close(folder_fd)
    return metadata


def restore_checkpoint(engine, run_dir, progress, expected_fingerprint):
    if progress["experiment_sha256"] != expected_fingerprint:
        raise TrainingError("resume_experiment_mismatch")
    if not CHECKPOINT_NAME.fullmatch(progress.get("checkpoint", "")):
        raise TrainingError("invalid_checkpoint_reference")
    folder = run_dir / progress["checkpoint"]
    stored = read_state(folder / "progress.json")
    immutable = ("experiment_sha256", "epoch", "offset", "microsteps", "updates",
                 "checkpoint", "baseline", "frozen_sha256")
    if any(stored.get(key) != progress.get(key) for key in immutable):
        raise TrainingError("checkpoint_progress_mismatch")
    engine.restore(folder)


def train_epochs(engine, train, validation, run_dir, progress, recipe, guard):
    """Resume only committed updates; interrupted accumulation is replayed."""
    baseline = progress["baseline"]
    epoch, offset = progress["epoch"], progress["offset"]
    if (type(epoch) is not int or type(offset) is not int or not 0 <= epoch <= recipe.epochs
            or not 0 <= offset < len(train) or epoch == recipe.epochs and offset
            or progress["microsteps"] != epoch * len(train) + offset
            or progress["updates"] != epoch * math.ceil(len(train) / recipe.accumulation)
            + offset // recipe.accumulation or offset % recipe.accumulation):
        raise TrainingError("invalid_committed_training_cursor")
    gradient, count = None, 0
    while True:
        guard.check(90)
        epoch, offset = progress["epoch"], progress["offset"]
        if offset == 0 and epoch and str(epoch) not in progress["epoch_scores"]:
            candidate = engine.validate(validation)
            if engine.frozen_hash() != engine.frozen_digest:
                raise TrainingError("frozen_weights_changed")
            scores = {**progress["epoch_scores"], str(epoch): candidate}
            best = progress["best"]
            if select_candidate(baseline, best["scores"], candidate):
                best = {"checkpoint": progress["checkpoint"], "scores": candidate}
            progress = {**progress, "epoch_scores": scores, "best": best}
            write_state(run_dir / "progress.json", progress)
        if epoch >= recipe.epochs:
            return {**progress, "status": "complete"}
        index = epoch_order(len(train), recipe.seed, epoch)[offset]
        value, _ = engine.gradient(train[index])
        gradient = engine.add_gradients(gradient, value)
        count += 1
        upper = offset + 1
        completed_epoch = upper == len(train)
        progress = {**progress, "epoch": epoch + int(completed_epoch),
                    "offset": 0 if completed_epoch else upper,
                    "microsteps": progress["microsteps"] + 1}
        if count == recipe.accumulation or completed_epoch:
            engine.update(gradient, count)  # Divide a final short batch by its actual count.
            gradient, count = None, 0
            progress = save_checkpoint(engine, run_dir, {**progress, "updates": progress["updates"] + 1})


def worker(args, recipe, seconds):
    import logging

    from private_io import prepare_directory

    logging.disable(logging.CRITICAL)
    guard = Guard(seconds)
    guard.check(180)
    run_dir, state_dir = args.run_dir, args.state_dir
    if not run_dir.is_relative_to(state_dir / "specialization") or run_dir == args.model_path:
        raise TrainingError("run_directory_outside_specialization")
    prepare_directory(run_dir)
    experiment = {"version": VERSION, "mlx_vlm": MLX_VLM_VERSION, "recipe": asdict(recipe),
                  "targets": list(TARGETS), "model_revision": MODEL_REVISION,
                  "runtime_packages": RUNTIME_PACKAGES,
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "requirements_sha256": hashlib.sha256(Path(__file__).with_name("requirements-ml.txt").read_bytes()).hexdigest()}
    try:
        experiment["model_assets_sha256"] = verify_assets(args.model_path, args.asset_manifest, guard)
        experiment["base_model_sha256"] = snapshot_identity(json.loads(read_private(args.asset_manifest))["files"])
        train, experiment["train_sha256"] = read_export(args.train_jsonl, "train")
        validation, experiment["validation_sha256"] = read_export(args.validation_jsonl, "validation")
        check_partitions(train, validation)
        experiment["image_content_sha256"] = fingerprint([r["image_digest"] for r in train + validation])
        identity = fingerprint(experiment)
        existing = read_state(run_dir / "experiment.json")
        if existing and existing != experiment:
            raise TrainingError("run_directory_has_different_experiment")
        write_state(run_dir / "experiment.json", experiment)
        engine = MLXEngine(args.model_path, recipe, guard, experiment["base_model_sha256"])
        engine.telemetry_state_dir, engine.telemetry_experiment = state_dir, identity
        checks = engine.verify(train, validation, run_dir)
        write_state(run_dir / "verification.json", {"experiment_sha256": identity, **checks})
        if args.stage == "verify":
            return {"status": "verified", "checks": checks, "experiment_sha256": identity}
        progress = read_state(run_dir / "progress.json")
        if progress:
            restore_checkpoint(engine, run_dir, progress, identity)
            if progress.get("frozen_sha256") != engine.frozen_digest:
                raise TrainingError("resumed_base_weights_mismatch")
        elif args.stage == "validate":
            raise TrainingError("no_checkpoint_to_validate")
        else:
            baseline = engine.validate(validation)
            progress = {"experiment_sha256": identity, "epoch": 0, "offset": 0, "microsteps": 0,
                        "updates": 0, "baseline": baseline, "epoch_scores": {},
                        "frozen_sha256": engine.frozen_digest,
                        "best": {"checkpoint": None, "scores": baseline}}
            progress = save_checkpoint(engine, run_dir, progress)
        if args.stage == "validate":
            selected = progress["best"]["checkpoint"]
            if selected:
                verify_adapter_assets(run_dir / selected / "adapter", run_dir / selected / "adapter-manifest.json",
                                      experiment["base_model_sha256"])
                engine.restore_adapters(run_dir / selected / "adapter")
            else:
                # Verify restored the exact initialization; zero B returns to the base.
                engine.model.update(engine.unflatten([(k, engine.mx.zeros_like(v))
                    for k, v in engine.flatten(engine.model.trainable_parameters()) if k.endswith(".lora_b")]))
            scores = engine.validate(validation)
            if scores != progress["best"]["scores"]:
                raise TrainingError("selected_generation_reload_mismatch")
            if engine.frozen_hash() != engine.frozen_digest:
                raise TrainingError("frozen_weights_changed")
            experimental = selected or progress["checkpoint"]
            if selected is None:
                # Verify the final trained adapter as well as the proxy-selected
                # base, so the independent benchmark can assess semantic gains.
                verify_adapter_assets(run_dir / experimental / "adapter", run_dir / experimental / "adapter-manifest.json",
                                      experiment["base_model_sha256"])
                engine.restore_adapters(run_dir / experimental / "adapter")
                experimental_scores = engine.validate(validation)
                if experimental_scores != progress["epoch_scores"].get(str(progress["epoch"])):
                    raise TrainingError("experimental_generation_reload_mismatch")
                if engine.frozen_hash() != engine.frozen_digest:
                    raise TrainingError("frozen_weights_changed")
            result = {"status": "validated", "scores": scores, "selected_adapter": bool(selected),
                      "experimental_adapter_checkpoint": experimental, "experimental_adapter_reload_parity": True,
                      "experiment_sha256": identity}
        else:
            progress = train_epochs(engine, train, validation, run_dir, progress, recipe, guard)
            result = {"status": "complete", "epochs": progress["epoch"], "updates": progress["updates"],
                      "microsteps": progress["microsteps"], "baseline": progress["baseline"],
                      "best": progress["best"], "experiment_sha256": identity}
        write_state(run_dir / "receipt.json", result)
        return result
    except BudgetEnded as error:
        result = {"status": "partial", "stop_reason": str(error), "resume": "train_same_arguments"}
        write_state(run_dir / "receipt.json", result)
        return result


def dry_run(recipe):
    return {"status": "dry_run", "version": VERSION, "recipe": asdict(recipe),
            "targets": list(TARGETS), "expected_projection_modules": 128,
            "frozen": ["base", "vision", "projector", "recurrent_projections"],
            "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION,
            "mlx_vlm_required": MLX_VLM_VERSION, "stages": ["dry-run", "verify", "train", "validate"],
            "heavy_window": "00:30-07:00 America/Chicago", "sealed_test": "never_read",
            "reads_dataset": False, "loads_model": False,
            "export_fields": ["id", "images", "messages", "context"],
            "validation_examples": 20, "synthetic_controls_minimum": 2}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("dry-run", "verify", "train", "validate"))
    for name in ("model-path", "train-jsonl", "validation-jsonl", "run-dir", "state-dir"):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--model-manifest", "--asset-manifest", dest="asset_manifest", type=Path)
    result.add_argument("--epochs", type=int, default=2, choices=(1, 2))
    result.add_argument("--seed", type=int, default=20261005)
    result.add_argument("--image-tokens", type=int, default=512, choices=(512, 1024))
    result.add_argument("--memory-limit-gib", type=int, default=10, choices=range(8, 13))
    result.add_argument("--max-seconds", type=int, default=2700)
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        recipe = Recipe(epochs=args.epochs, seed=args.seed, image_tokens=args.image_tokens,
                        memory_limit_gib=args.memory_limit_gib)
        if args.stage == "dry-run":
            result = dry_run(recipe)
        else:
            paths = (args.model_path, args.asset_manifest, args.train_jsonl, args.validation_jsonl,
                     args.run_dir, args.state_dir)
            if any(p is None or not p.is_absolute() or ".." in p.parts for p in paths):
                raise TrainingError("explicit_absolute_runtime_paths_required")
            seconds = overnight_budget(args.max_seconds)
            if seconds < 180:
                result = {"status": "outside_training_window"}
            elif args.worker:
                if os.environ.get("YOUR_TIME_ML_WORKER") != "1" or not inherited_lock_present(args.state_dir):
                    raise TrainingError("worker_requires_shared_runner_lock")
                if any(os.environ.get(k) != v for k, v in OFFLINE.items()):
                    raise TrainingError("worker_requires_offline_environment")
                os.umask(0o077)
                # Package warnings can include private paths; keep all internal output suppressed.
                with kernel_deadline(seconds), open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    result = worker(args, recipe, seconds)
            else:
                from model_execution import run_model
                from vision_batch import resource_gate

                gate = resource_gate(benchmark_now=False)
                if gate:
                    result = {"status": "resource_gate", "reason": gate}
                else:
                    arguments = list(sys.argv[1:] if argv is None else argv)
                    command = ["/usr/bin/sandbox-exec", "-f", str(Path(__file__).with_name("network-off.sb")),
                               sys.executable, "-B", str(Path(__file__).resolve()), *arguments, "--worker"]
                    completed = run_model(command, state_dir=args.state_dir, timeout=seconds,
                        capture_output=True, env={**os.environ, **OFFLINE, "YOUR_TIME_ML_WORKER": "1"},
                        telemetry={"stage": "training", "variant": args.stage,
                                   "engine_version": "mlx-vlm-0.7.6", "prompt_version": VERSION})
                    if completed.returncode == -signal.SIGALRM:
                        result = {"status": "partial", "stop_reason": "kernel_deadline",
                                  "resume": "train_same_arguments"}
                    elif completed.returncode:
                        # Never echo raw stdout/stderr, even on package crashes.
                        result = {"status": "worker_failed", "exit_code": completed.returncode}
                        try:
                            failure = json.loads(completed.stdout)
                            code = failure.get("error")
                            if failure.get("status") == "failed" and isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", code):
                                result["error"] = code
                        except (ValueError, TypeError, AttributeError):
                            pass
                    else:
                        result = json.loads(completed.stdout)
        print(json.dumps(result, sort_keys=True))
        return 1 if result["status"] in {"failed", "worker_failed"} else 0
    except subprocess.TimeoutExpired:
        print(json.dumps({"status": "partial", "stop_reason": "hard_deadline", "resume": "train_same_arguments"}))
        return 0
    except Exception as error:
        if type(error).__name__ == "ModelBusy":
            print(json.dumps({"status": "resource_gate", "reason": "local_model_busy"}))
            return 0
        code = str(error) if isinstance(error, TrainingError) else type(error).__name__
        print(json.dumps({"status": "failed", "error": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
