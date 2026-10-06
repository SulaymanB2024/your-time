"""Private paired evaluation; stdout/API receipts contain aggregates only.

Importing this module reads no private examples and loads no model. Test access
is a separate operation requiring an immutable, reviewed candidate freeze.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from activity_context import parse_result, prompt
from model_execution import ModelBusy
from overnight_schedule import VISION_END, ZONE, remaining_seconds
from private_io import open_private_file, write_json
from secure_store import STATE_DIR
from specialization_dataset import ROOT as DATA_ROOT
from vision_batch import EMAIL_RE, LONG_NUMBER_RE, PROMPT, SENSITIVE_RE, URL_RE

VERSION = "specialization_benchmark_v1"
VARIANTS = ("production_q8", "context_q8", "mlx_base", "mlx_adapter")
PROJECT = Path(__file__).resolve().parent
MAX_EXPORT = 16 * 1024**2
CODE = ("specialization_benchmark.py", "activity_context.py", "vision_quality_eval.py",
        "vision_batch.py", "vision_fallback.py", "specialization_worker.py", "model_execution.py",
        "inference_telemetry.py", "engine_identity.py", "overnight_schedule.py", "model_deadline.py",
        "private_io.py", "secure_store.py", "specialization_assets.py", "network-off.sb")
GRADES = ("project_correct", "task_correct", "evidence_supported",
          "uncertainty_appropriate", "privacy_leak", "unsupported_completion")
COMPLETION = re.compile(r"\b(completed|finished|submitted|published|delivered|sent|saved|"
                        r"finalized|achieved|deployed|resolved|launched|released|merged|"
                        r"shipped|purchased|deleted|emailed|posted)\b", re.I)
SHA = re.compile(r"[a-f0-9]{64}\Z")
CHECKPOINT = re.compile(r"step-\d{6}-[a-f0-9]{8}\Z")


class BenchmarkError(RuntimeError):
    """Public code only; private paths, values and exception bodies never escape."""

    def __init__(self, code):
        super().__init__(code if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,80}", code)
                         else "benchmark_error")


class BudgetEnded(BenchmarkError):
    pass


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path, guard=None, *, private=False) -> str:
    hasher = hashlib.sha256()
    fd = open_private_file(path, os.O_RDONLY) if private else os.open(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if guard:
                guard.check()
            hasher.update(chunk)
    return hasher.hexdigest()


def read_private(path: Path, *, maximum=MAX_EXPORT) -> bytes:
    with os.fdopen(open_private_file(path, os.O_RDONLY), "rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise BenchmarkError("private_file_too_large")
    return payload


def decode(payload: bytes | str):
    return json.loads(payload, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = decode(read_private(path))
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("invalid_private_metadata") from None


class Guard:
    def __init__(self, budget: int):
        if type(budget) is not int or not 1 <= budget <= 5400:
            raise BenchmarkError("invalid_nightly_budget")
        self.deadline = time.monotonic() + min(budget, remaining_seconds(
            datetime.now(timezone.utc), end=VISION_END))
        self.next_resource_check = 0.0

    def check(self, reserve=0):
        if remaining_seconds(datetime.now(timezone.utc), end=VISION_END) <= reserve:
            raise BudgetEnded("outside_vision_window")
        if self.deadline - time.monotonic() <= reserve:
            raise BudgetEnded("benchmark_budget")
        if time.monotonic() >= self.next_resource_check:
            from vision_batch import resource_gate

            try:
                reason = resource_gate(benchmark_now=False)
            except (OSError, RuntimeError, subprocess.SubprocessError):
                reason = "resource_status_unavailable"
            if reason:
                raise BudgetEnded("resource_gate")
            self.next_resource_check = time.monotonic() + 5

    def remaining(self):
        return max(0, min(self.deadline - time.monotonic(), remaining_seconds(
            datetime.now(timezone.utc), end=VISION_END)))


def load_examples(path: Path, split: str, expected: dict) -> list[dict]:
    """Only called inside the admitted run; never returns bodies to its caller UI."""
    try:
        payload = read_private(path)
        if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise BenchmarkError("export_changed")
        rows = [decode(line) for line in payload.splitlines() if line.strip()]
        if not rows or len(rows) > 100 or len(rows) != expected["count"]:
            raise BenchmarkError("invalid_export_count")
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or row.get("split") != split
                    or row.get("source_class") != "real" or not isinstance(row.get("id"), str)
                    or row["id"] in seen or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"])
                    or not isinstance(row.get("episode_id"), str) or not row["episode_id"]
                    or not isinstance(row.get("image_sha256"), str)
                    or not SHA.fullmatch(row["image_sha256"])
                    or not isinstance(row.get("images"), list) or len(row["images"]) != 1
                    or not isinstance(row["images"][0], str)
                    or not Path(row["images"][0]).is_absolute()):
                raise BenchmarkError("invalid_export_record")
            messages = row["messages"]
            if (len(messages) != 2 or messages[0]["role"] != "user"
                    or messages[1]["role"] != "assistant"):
                raise BenchmarkError("invalid_reference_messages")
            # References are reviewed visible evidence, never self-generated labels.
            row["reference"] = parse_result(messages[1]["content"], row["context"])
            seen.add(row["id"])
        return rows
    except BenchmarkError:
        raise
    except Exception:
        raise BenchmarkError("invalid_export") from None


def parameters(config: dict) -> dict:
    values = {"image_side": config.get("image_side", 1600),
              "image_tokens": config.get("image_tokens", 512),
              "context_tokens": config.get("context_tokens", 4096),
              "thinking": config.get("thinking", False),
              "thinking_budget": config.get("thinking_budget", 0),
              "max_tokens": config.get("max_tokens", 768),
              "timeout_seconds": config.get("timeout_seconds", 180),
              "threads": config.get("threads", 4), "seed": config.get("seed", 20261005),
              "batch_requests": config.get("batch_requests", 4)}
    choices = {"image_side": {1024, 1600, 2048}, "image_tokens": {512, 1024, 1536},
               "context_tokens": {2048, 4096}, "threads": {2, 4, 8}}
    if (any(type(values[k]) is not int or values[k] not in opts for k, opts in choices.items())
            or type(values["thinking"]) is not bool
            or any(type(values[k]) is not int for k in ("thinking_budget", "max_tokens",
                                                       "timeout_seconds", "seed", "batch_requests"))
            or not 0 <= values["thinking_budget"] <= 512 or not values["thinking_budget"] < values["max_tokens"] <= 2048
            or values["max_tokens"] < 32 or not 1 <= values["timeout_seconds"] <= 300
            or not 0 <= values["seed"] < 2**32 or not 1 <= values["batch_requests"] <= 20):
        raise BenchmarkError("invalid_configuration")
    if config.get("variants", list(VARIANTS)) != list(VARIANTS):
        raise BenchmarkError("requires_four_frozen_variants")
    return values


def q8_parameters(config: dict) -> dict:
    from vision_fallback import generation_budget

    tokens, timeout = generation_budget(0)
    values = {"max_tokens": config.get("q8_max_tokens", tokens),
              "timeout_seconds": config.get("q8_timeout_seconds", timeout),
              "image_side": config.get("q8_image_side", 1600),
              "image_tokens": config.get("q8_image_tokens", 1024),
              "threads": config.get("q8_threads", 4), "context_tokens": 4096,
              "thinking": "existing_enabled_template", "seed": "existing_runner_default",
              "baseline": "generation_budget_first_attempt_no_automatic_retries"}
    if (type(values["max_tokens"]) is not int or not 32 <= values["max_tokens"] <= 2048
            or type(values["timeout_seconds"]) is not int or not 1 <= values["timeout_seconds"] <= 300
            or any(type(values[k]) is not int or values[k] not in choices for k, choices in
                   (("image_side", {1024, 1600, 2048}), ("image_tokens", {512, 1024, 1536}),
                    ("threads", {2, 4, 8})))):
        raise BenchmarkError("invalid_q8_configuration")
    return values


def adapter_checkpoint(progress: dict) -> tuple[str | None, str]:
    """Preselect a trained experiment without mistaking proxies for acceptance."""
    checkpoint = progress.get("best", {}).get("checkpoint")
    selection = "validation_proxy_preselection"
    if checkpoint is None and progress.get("epoch", 0) >= 1 and progress.get("offset") == 0:
        checkpoint = progress.get("checkpoint")
        selection = "final_epoch_experiment_requires_semantic_review"
    if checkpoint is not None and (not isinstance(checkpoint, str) or not CHECKPOINT.fullmatch(checkpoint)):
        raise BenchmarkError("invalid_best_checkpoint")
    return checkpoint, selection


def model_pins(config: dict, study_root: Path) -> dict:
    """Read pinned metadata only here; execution re-verifies actual weight files."""
    from engine_identity import llama_identity
    from specialization_assets import ASSET_MANIFEST, MODEL_DIR
    from specialization_worker import PACKAGES, PROTOCOL, snapshot_identity
    from vision_quality_eval import LLAMA_CLI, MODEL_SPECS

    manifest_path, folder = MODEL_SPECS["9b_q8"]
    q8 = decode(manifest_path.read_bytes())
    weights = next(item for item in q8["files"] if item["name"].startswith("Qwen"))
    projector = next(item for item in q8["files"] if item["name"].startswith("mmproj-F16"))
    assets = read_state(ASSET_MANIFEST)
    progress = read_state(study_root / "training" / "progress.json")
    best, selection = adapter_checkpoint(progress)
    checkpoint = study_root / "training" / best if best else None
    adapter = read_state(checkpoint / "adapter-manifest.json") if checkpoint else None
    if best and not adapter:
        raise BenchmarkError("missing_adapter_manifest")
    if not assets.get("files"):
        raise BenchmarkError("missing_model_manifest")
    pins = {"q8": {"model_dir": str(folder), "manifest_sha256": fingerprint(q8),
                   "files": q8["files"], "weights": weights, "projector": projector,
                   "engine_sha256": llama_identity(LLAMA_CLI)},
            "mlx": {"model_dir": str(MODEL_DIR), "model_manifest": str(ASSET_MANIFEST),
                    "manifest_sha256": fingerprint(assets),
                    "model_sha256": snapshot_identity(assets["files"]),
                    "packages": PACKAGES, "protocol": PROTOCOL,
                    "python_sha256": file_digest(PROJECT / ".ml-venv/bin/python")},
            "adapter": {"checkpoint": best, "selection": selection, "manifest_sha256": fingerprint(adapter),
                        "snapshot_sha256": snapshot_identity(adapter["files"]),
                        "directory": str(checkpoint / "adapter"),
                        "manifest": str(checkpoint / "adapter-manifest.json")}
            if adapter else None}
    return pins


def experiment_spec(config: dict, study_root: Path) -> dict:
    for name, expected in config.get("source_sha256", {}).items():
        if Path(name).name != name or file_digest(PROJECT / name) != expected:
            raise BenchmarkError("frozen_source_changed")
    spec = {"version": VERSION, "configuration": config, "parameters": parameters(config),
            "q8_parameters": q8_parameters(config),
            "pins": model_pins(config, study_root),
            "source_sha256": {name: file_digest(PROJECT / name) for name in CODE},
            "preprocessing": {"colour": "RGB", "resize": "thumbnail_LANCZOS",
                              "mlxbudget": "processor_patch_merge_max_pixels",
                              "q8_context_tokens": 4096,
                              "q8_thinking": "existing_chat_template",
                              "q8_seed": "existing_runner_default"},
            "production_prompt_sha256": fingerprint(PROMPT),
            "context_prompt_template_sha256": fingerprint(prompt({"evidence": []})),
            "scoring": "literal_exact_proxy_plus_independent_blind_review_v1"}
    return spec


def example_identity(row: dict, variant: str, spec: dict) -> str:
    return fingerprint({"experiment": fingerprint(spec), "variant": variant,
                        "id": row["id"], "image_sha256": row["image_sha256"],
                        "context": row["context"], "episode_id": row["episode_id"],
                        "reference": row["reference"],
                        "prompt": PROMPT if variant == "production_q8" else prompt(row["context"])})


def safety_flags(text: str) -> dict:
    """Keyword flags are screening proxies, including negation false positives."""
    return {"privacy": bool(SENSITIVE_RE.search(text) or EMAIL_RE.search(text)
                             or URL_RE.search(text) or LONG_NUMBER_RE.search(text)),
            "completion": bool(COMPLETION.search(text))}


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()) if isinstance(value, str) else value


def score_response(response: dict, row: dict) -> dict:
    if not isinstance(response, dict):
        raise BenchmarkError("invalid_response")
    raw = response.get("raw_output")
    raw_flags = safety_flags(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw) if isinstance(raw, (str, bytes)) else None
    if raw_flags is None and isinstance(response.get("raw_safety"), dict):
        supplied = response["raw_safety"]
        if all(type(supplied.get(k)) is bool for k in ("privacy", "completion")):
            raw_flags = {k: supplied[k] for k in ("privacy", "completion")}
    checks = response.get("raw_safety_checks", {})
    if not isinstance(checks, dict):
        checks = {}
    if checks.get("available") is True and all(type(checks.get(k)) is bool for k in
        ("sensitive_keyword", "email", "url", "unsupported_completion")):
        raw_flags = {"privacy": any(checks[k] for k in ("sensitive_keyword", "email", "url")),
                     "completion": checks["unsupported_completion"]}
    activity = response.get("activity", response.get("result"))
    status = response.get("status", "invalid_response")
    if response.get("status") == "complete" and activity is not None:
        try:
            activity = parse_result(json.dumps(activity), row["context"])
        except (TypeError, ValueError):
            activity, status = None, "schema_error"
    else:
        activity = None
    if raw_flags and any(raw_flags.values()):
        activity, status = None, "raw_safety_blocked"
    visible = " ".join(str(activity.get(key) or "") for key in
                       ("project_candidate", "task_candidate", "visible_work")) if activity else response.get("description", "") or ""
    result = {"raw_safety": raw_flags, "raw_safety_available": raw_flags is not None,
              "final_safety_proxy": safety_flags(visible), "exact_proxy": None,
              "activity": activity, "description": response.get("description"),
              "status": status,
              "telemetry_attempt_id": response.get("telemetry_attempt_id")}
    if activity:
        reference = row["reference"]
        exact = {key: normalize(activity[key]) == normalize(reference[key])
                 for key in ("project_candidate", "task_candidate", "activity_kind", "uncertainty")}
        exact["joint"] = exact["project_candidate"] and exact["task_candidate"]
        exact["project_answerable"] = reference["project_candidate"] is not None
        exact["task_answerable"] = reference["task_candidate"] is not None
        exact["project_abstained"] = activity["project_candidate"] is None
        exact["task_abstained"] = activity["task_candidate"] is None
        # Citation validation is structural; this does NOT score entailment.
        result["exact_proxy"] = exact
    return result


class Runner:
    """One backend lifetime at a time; model APIs own the shared inference lock."""

    def __init__(self, spec: dict, guard: Guard):
        self.spec, self.guard, self.worker = spec, guard, None
        self.variant = None
        self.q8_verified = False

    def start(self, variant: str):
        from specialization_worker import WorkerConfig, WorkerController

        self.variant = variant
        self.guard.check(35)
        if variant.endswith("q8"):
            pins = self.spec["pins"]["q8"]
            if not self.q8_verified:
                for entry in pins["files"]:
                    if Path(entry["name"]).name != entry["name"]:
                        raise BenchmarkError("invalid_weight_manifest")
                    path = Path(pins["model_dir"]) / entry["name"]
                    if path.stat().st_size != entry["size"] or file_digest(path, self.guard, private=True) != entry["sha256"]:
                        raise BenchmarkError("weight_hash_mismatch")
                self.q8_verified = True
            return
        adapter = self.spec["pins"]["adapter"] if variant == "mlx_adapter" else None
        if variant == "mlx_adapter" and adapter is None:
            raise BenchmarkError("no_selected_adapter")
        self.guard.check(335)
        mlx = self.spec["pins"]["mlx"]
        config = WorkerConfig(model_dir=mlx["model_dir"], model_manifest=mlx["model_manifest"],
                              state_dir=str(STATE_DIR),
                              image_roots=(str(DATA_ROOT), str(STATE_DIR / "screenshots"),
                                           str(STATE_DIR / "pensieve/screenshots")),
                              adapter_dir=adapter["directory"] if adapter else None,
                              adapter_manifest=adapter["manifest"] if adapter else None,
                              max_seconds=max(1, int(self.guard.remaining() - 30)),
                              max_requests=self.spec["parameters"]["batch_requests"])
        self.worker = WorkerController(config)
        self.worker.start()
        if (self.worker.ready["model_sha256"] != mlx["model_sha256"]
                or self.worker.ready.get("adapter_sha256") != (adapter["snapshot_sha256"] if adapter else None)):
            self.close()
            raise BenchmarkError("worker_identity_mismatch")

    def infer(self, row: dict, identity: str) -> dict:
        from vision_quality_eval import run_image

        self.guard.check(35)
        options = self.spec["parameters"]
        settings = options if self.worker else self.spec["q8_parameters"]
        # The resident worker reserves five seconds for request control/cleanup
        # and itself ends 30 seconds before the study deadline.
        timeout = min(settings["timeout_seconds"], int(self.guard.remaining() - 40))
        if timeout < 1:
            raise BudgetEnded("benchmark_budget")
        path = Path(row["images"][0])
        if not any(path.is_relative_to(root) for root in
                   (DATA_ROOT, STATE_DIR / "screenshots", STATE_DIR / "pensieve/screenshots")):
            raise BenchmarkError("image_outside_roots")
        if file_digest(path, self.guard, private=True) != row["image_sha256"]:
            raise BenchmarkError("image_changed")
        if self.worker:
            return self.worker.request(row["context"], image_path=str(path),
                image_sha256=row["image_sha256"], thinking=options["thinking"],
                thinking_budget=options["thinking_budget"], max_tokens=options["max_tokens"],
                timeout_seconds=timeout, image_side=options["image_side"],
                image_tokens=options["image_tokens"], context_tokens=options["context_tokens"],
                seed=options["seed"], request_id=identity)
        pins = self.spec["pins"]["q8"]
        model = {"weights": Path(pins["model_dir"]) / pins["weights"]["name"],
                 "projector": Path(pins["model_dir"]) / pins["projector"]["name"],
                 "weights_sha256": pins["weights"]["sha256"],
                 "projector_sha256": pins["projector"]["sha256"]}
        context = row["context"] if self.variant == "context_q8" else None
        return run_image(path, model, timeout_seconds=timeout, max_tokens=settings["max_tokens"],
            prompt=prompt(context) if context else PROMPT, context=context,
            prompt_version="activity_context_v1" if context else "visible_task_v1",
            variant=self.variant, image_side=settings["image_side"],
            image_tokens=settings["image_tokens"], threads=settings["threads"])

    def close(self):
        if self.worker:
            self.worker.close()
        self.worker = None


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - width), min(1, center + width)]


def paired_interval(pairs: list[tuple[str, bool, bool]], *, seed=0, draws=2000) -> dict:
    groups = defaultdict(list)
    for episode, left, right in pairs:
        groups[episode].append(int(right) - int(left))
    if not pairs:
        return {"pairs": 0, "groups": 0, "delta": None, "interval_95": None}
    delta = sum(sum(v) for v in groups.values()) / len(pairs)
    interval = None
    if len(groups) >= 2:
        values, generator, samples = list(groups.values()), random.Random(seed), []
        for _ in range(draws):
            selection = [generator.choice(values) for _ in values]
            samples.append(sum(sum(v) for v in selection) / sum(len(v) for v in selection))
        samples.sort()
        interval = [samples[int(draws * .025)], samples[min(draws - 1, int(draws * .975))]]
    return {"pairs": len(pairs), "groups": len(groups), "delta": delta,
            "interval_95": interval, "method": "paired_episode_cluster_percentile_bootstrap",
            "wins": sum(not a and b for _, a, b in pairs),
            "losses": sum(a and not b for _, a, b in pairs),
            "ties": sum(a == b for _, a, b in pairs),
            "caveat": "Episode IDs are supplied grouping metadata; few groups cannot establish general accuracy."}


def temporal_key(row: dict):
    try:
        item = next(e for e in row["context"]["evidence"] if e.get("source") in {"screen_context", "synthetic_screen"})
        instant = datetime.fromisoformat(item["timestamp_utc"])
        if instant.tzinfo is None:
            return None
        local = instant.astimezone(ZONE)
        # The repeated fall-back hour contains two distinct occupied bins.
        return local.date().isoformat(), local.hour, local.minute // 30, int(local.utcoffset().total_seconds())
    except (KeyError, StopIteration, ValueError, TypeError):
        return None


def aggregate(rows: list[dict], results: dict, assessments: dict | None = None) -> dict:
    variants, paired = {}, {}
    episodes = {r["episode_id"] for r in rows}
    known_bins = {temporal_key(r) for r in rows} - {None}
    assessment_rows = (assessments or {}).get("items", {})
    for variant in VARIANTS:
        runs = [results.get(r["id"], {}).get(variant) for r in rows]
        terminal = [r for r in runs if r]
        attempted = [r for r in terminal if r.get("attempted", True)]
        complete = [r for r in terminal if r["status"] == "complete"]
        scores = [r["exact_proxy"] for r in complete if r.get("exact_proxy")]
        metrics = {"selected_denominator": len(rows), "attempted": len(attempted),
                   "complete": len(complete), "failed": len(terminal) - len(complete),
                   "pending": len(rows) - len(terminal), "structured_proxy_scored": len(scores),
                   "literal_joint_proxy_correct": sum(r["joint"] for r in scores),
                   "literal_joint_proxy_interval_95": wilson(sum(r["joint"] for r in scores), len(rows)) if scores else None,
                   "raw_safety_available": sum(r.get("raw_safety_available", False) for r in attempted),
                   "raw_privacy_flags": sum(bool((r.get("raw_safety") or {}).get("privacy")) for r in attempted),
                   "raw_completion_flags": sum(bool((r.get("raw_safety") or {}).get("completion")) for r in attempted),
                   "final_privacy_proxy_flags": sum(r.get("final_safety_proxy", {}).get("privacy", False) for r in attempted),
                   "final_completion_proxy_flags": sum(r.get("final_safety_proxy", {}).get("completion", False) for r in attempted),
                   "failure_codes": dict(Counter(r["status"] for r in terminal if r["status"] != "complete")),
                   "total_attempt_seconds": round(sum(r.get("elapsed_seconds", 0) for r in attempted), 3),
                   "median_complete_seconds": statistics.median(r["elapsed_seconds"] for r in complete) if complete else None,
                   "complete_unique_episodes": len({row["episode_id"] for row, run in zip(rows, runs) if run and run["status"] == "complete"}),
                   "complete_unique_30_minute_bins": len({temporal_key(row) for row, run in zip(rows, runs)
                                                         if run and run["status"] == "complete"} - {None})}
        for field in ("project", "task"):
            correct = sum(r[field + "_candidate"] for r in scores)
            asserted = [r for r in scores if not r[field + "_abstained"]]
            answerable = sum(r["reference"][field + "_candidate"] is not None for r in rows)
            metrics[field + "_proxy"] = {"correct": correct, "all_selected": len(rows),
                "precision": sum(r[field + "_candidate"] for r in asserted) / len(asserted) if asserted else None,
                "answerable": answerable,
                "unanswerable": len(rows) - answerable,
                "answerable_recall": sum(r[field + "_candidate"] and r[field + "_answerable"] for r in scores) / answerable if answerable else None,
                "correct_abstentions": sum(r[field + "_candidate"] and r[field + "_abstained"] for r in scores),
                "unanswerable_abstention_recall": sum(r[field + "_candidate"] and r[field + "_abstained"] for r in scores)
                    / (len(rows) - answerable) if len(rows) > answerable else None}
        grades = [assessment_rows.get(run.get("assessment_key"), {}).get("grades", {}) for run in terminal]
        graded = [g for run, g in zip(terminal, grades) if all(type(g.get(k)) is bool for k in GRADES)
                  and (run["status"] == "complete" or not any(g[k] for k in GRADES[:4]))]
        metrics["invalid_failure_grades"] = sum(all(type(g.get(k)) is bool for k in GRADES)
            and run["status"] != "complete" and any(g[k] for k in GRADES[:4]) for run, g in zip(terminal, grades))
        metrics["human_reviewed"] = len(graded)
        metrics["human_joint_supported_correct"] = sum(g["project_correct"] and g["task_correct"]
                                                      and g["evidence_supported"] for g in graded)
        metrics["human_privacy_leaks"] = sum(g["privacy_leak"] for g in graded)
        metrics["human_unsupported_completion"] = sum(g["unsupported_completion"] for g in graded)
        if not metrics["raw_safety_available"]:
            metrics["raw_privacy_flags"] = metrics["raw_completion_flags"] = None
        metrics["probability_calibration"] = "not_measured_no_numeric_confidence"
        metrics["ordinal_uncertainty_review"] = {}
        for category in ("supported", "partial", "unclear"):
            checked = []
            for run in terminal:
                if (run.get("activity") or {}).get("uncertainty") != category:
                    continue
                grade = assessment_rows.get(run.get("assessment_key"), {}).get("grades", {})
                if all(type(grade.get(k)) is bool for k in GRADES):
                    checked.append(grade["project_correct"] and grade["task_correct"] and grade["evidence_supported"])
            metrics["ordinal_uncertainty_review"][category] = {"reviewed": len(checked),
                "joint_supported_correct": sum(checked), "fraction": sum(checked) / len(checked) if checked else None}
        variants[variant] = metrics
    human_paired = {}
    for left, right in (("production_q8", "context_q8"), ("context_q8", "mlx_base"), ("mlx_base", "mlx_adapter")):
        pairs = []
        human_pairs = []
        for row in rows:
            runs = results.get(row["id"], {})
            a, b = runs.get(left), runs.get(right)
            # Caption baseline requires human grades; do not invent null labels.
            if left != "production_q8":
                pairs.append((row["episode_id"], bool(a and (a.get("exact_proxy") or {}).get("joint")),
                              bool(b and (b.get("exact_proxy") or {}).get("joint"))))
            ga = assessment_rows.get((a or {}).get("assessment_key"), {}).get("grades", {})
            gb = assessment_rows.get((b or {}).get("assessment_key"), {}).get("grades", {})
            if all(type(g.get(k)) is bool for g in (ga, gb) for k in GRADES):
                correct = lambda run, grade: bool(run and run["status"] == "complete" and
                    grade["project_correct"] and grade["task_correct"] and grade["evidence_supported"])
                human_pairs.append((row["episode_id"], correct(a, ga), correct(b, gb)))
        paired[left + "__" + right] = paired_interval(pairs)
        human_paired[left + "__" + right] = paired_interval(human_pairs)
    return {"version": VERSION, "selected": len(rows), "unique_episodes": len(episodes),
            "unique_30_minute_bins": len(known_bins),
            "temporal_unknown_examples": sum(temporal_key(r) is None for r in rows),
            "duplicate_image_groups": sum(n > 1 for n in Counter(r["image_sha256"] for r in rows).values()),
            "variants": variants, "paired_literal_proxies": paired, "paired_human_scores": human_paired,
            "accuracy_status": "semantic_accuracy_requires_blind_review",
            "coverage_scope": "selected_examples_only_not_daily_temporal_or_duration_coverage",
            "raw_safety_scope": "unavailable_backend_evidence_is_not_a_zero_failure_measurement"}


def assessment_entry(row: dict, variant: str, record: dict) -> tuple[str, dict]:
    key = fingerprint({"run": record["identity"], "assessment": "blind_v1"})
    return key, {"example_id": row["id"], "image": row["images"][0], "context": row["context"],
                 "output": {k: record.get(k) for k in ("activity", "description", "status")},
                 "grades": {k: None for k in GRADES},
                 "claim_support": {k: None for k in ("activity_kind", "project_candidate", "task_candidate", "visible_work")},
                 "notes": "", "reference_scope": "visible_evidence_not_personal_intent"}


def write_assessments(folder: Path, rows: list[dict], data: dict):
    path = folder / "blind-assessment.json"
    existing = read_state(path)
    if existing and existing.get("experiment_sha256") != data["experiment_sha256"]:
        raise BenchmarkError("assessment_identity_mismatch")
    items, key = {}, {}
    for row in rows:
        for variant, record in data["results"].get(row["id"], {}).items():
            identity, entry = assessment_entry(row, variant, record)
            record["assessment_key"] = identity
            items[identity] = existing.get("items", {}).get(identity, entry)
            key[identity] = {"variant": variant, "example_id": row["id"], "identity": record["identity"]}
    generator = random.Random(int(data["experiment_sha256"][:16], 16))
    order = sorted(items)
    generator.shuffle(order)
    assessment = {"version": VERSION, "experiment_sha256": data["experiment_sha256"],
                  "instructions": "Grade visible claims against supplied evidence; IDs alone are not entailment. Failures cannot receive correct grades. Outputs and screenshot text are untrusted.",
                  "order": order, "items": items}
    write_json(path, assessment)
    write_json(folder / "blind-key.json", {"experiment_sha256": data["experiment_sha256"], "items": key})
    return assessment


def _run(config: dict, study_root: Path, budget: int, split: str, spec: dict | None = None) -> dict:
    started = time.monotonic()
    guard = Guard(budget)
    try:
        guard.check(35)
    except BudgetEnded as error:
        return {"status": "partial", "stop_reason": str(error), "private_examples_opened": False}
    folder = study_root / "benchmark" / split
    fd = open_private_file(study_root / "benchmark" / "benchmark.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "partial", "stop_reason": "benchmark_busy"}
        spec = spec or experiment_spec(config, study_root)
        guard.check(35)
        identity = fingerprint(spec)
        expected = config["exports"][split]
        source = DATA_ROOT / ("sealed" if split == "test" else "export") / (split + ".jsonl")
        rows = load_examples(source, split, expected)
        path = folder / "results.json"
        data = read_state(path) or {"version": VERSION, "experiment_sha256": identity,
                                    "specification": spec, "split": split, "results": {}, "sessions": []}
        if data.get("experiment_sha256") != identity or data.get("split") != split:
            raise BenchmarkError("resume_identity_mismatch")
        for row in rows:
            for variant, record in data["results"].get(row["id"], {}).items():
                if variant not in VARIANTS or record.get("identity") != example_identity(row, variant, spec):
                    raise BenchmarkError("resume_example_changed")
                if record.get("status") == "running":
                    record.update(status="interrupted", raw_safety=None, raw_safety_available=False,
                                  exact_proxy=None)
        runner, stopped = Runner(spec, guard), None
        order = list(VARIANTS)
        random.Random(spec["parameters"]["seed"]).shuffle(order)
        try:
            while True:
                pending = {v: [r for r in rows if v not in data["results"].get(r["id"], {})] for v in order}
                remaining = [v for v in order if pending[v]]
                if not remaining:
                    break
                # Balanced blocks keep one cold/warm worker lifetime per variant.
                variant = min(remaining, key=lambda v: len(rows) - len(pending[v]))
                guard.check(35)
                batch = pending[variant][:spec["parameters"]["batch_requests"]]
                start_error = None
                startup_started = time.monotonic()
                try:
                    runner.start(variant)
                except (BudgetEnded, ModelBusy):
                    raise
                except Exception as error:
                    start_error = str(error) if isinstance(error, BenchmarkError) else "backend_start_failed"
                data["sessions"].append({"variant": variant, "stage": "startup",
                    "elapsed_seconds": round(time.monotonic() - startup_started, 3),
                    "startup_telemetry_attempt_id": (getattr(runner.worker, "ready", {}) or {}).get("startup_telemetry_attempt_id")})
                if start_error and start_error != "no_selected_adapter":
                    batch = batch[:1]
                for row in batch:
                    guard.check(35)
                    attempt_started = time.monotonic()
                    run_identity = example_identity(row, variant, spec)
                    data["results"].setdefault(row["id"], {})[variant] = {
                        "identity": run_identity, "status": "running", "attempted": True}
                    write_json(path, data)
                    try:
                        if start_error:
                            raise BenchmarkError(start_error)
                        response = runner.infer(row, run_identity)
                        record = score_response(response, row)
                    except (BudgetEnded, ModelBusy):
                        data["results"][row["id"]].pop(variant)
                        write_json(path, data)
                        raise
                    except Exception as error:
                        from specialization_worker import WorkerError

                        if (isinstance(error, WorkerError) and str(error) == "insufficient_window_for_request"
                                and getattr(error, "telemetry_attempt_id", None) is None):
                            # Image hashing/admission can consume the control
                            # margin. Nothing was inferred: leave this pair pending.
                            data["results"][row["id"]].pop(variant)
                            write_json(path, data)
                            raise BudgetEnded("benchmark_budget") from None
                        record = {"status": str(error) if isinstance(error, BenchmarkError) else "inference_or_schema_error",
                                  "raw_safety": None, "raw_safety_available": False,
                                  "exact_proxy": None, "error_type": type(error).__name__,
                                  "telemetry_attempt_id": getattr(error, "telemetry_attempt_id", None)}
                        if isinstance(getattr(error, "raw_safety_checks", None), dict):
                            flags = score_response({"status": record["status"],
                                "raw_safety_checks": error.raw_safety_checks,
                                "telemetry_attempt_id": getattr(error, "telemetry_attempt_id", None)}, row)
                            record.update(flags)
                        # A failed generation invalidates the resident worker.
                        runner.close()
                        if variant.startswith("mlx") and start_error != "no_selected_adapter":
                            start_error = "worker_unavailable_after_failure"
                    record.update(identity=run_identity, elapsed_seconds=round(time.monotonic() - attempt_started, 3),
                                  attempted=start_error != "no_selected_adapter",
                                  attempt_started_at_utc=datetime.now(timezone.utc).isoformat())
                    data["results"].setdefault(row["id"], {})[variant] = record
                    write_json(path, data)
                    if start_error and start_error != "no_selected_adapter":
                        break
                runner.close()
                if start_error and start_error != "no_selected_adapter":
                    stopped = "backend_failed_resume_remaining_examples"
                    break
        except BudgetEnded as error:
            stopped = str(error)
        except ModelBusy:
            stopped = "local_model_busy"
        finally:
            runner.close()
        data["sessions"].append({"stage": "session", "elapsed_seconds": 0, "stop_reason": stopped,
                                  "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        assessments = write_assessments(folder, rows, data)
        write_json(path, data)
        summary = aggregate(rows, data["results"], assessments)
        elapsed = round(time.monotonic() - started, 3)
        data["sessions"][-1]["elapsed_seconds"] = elapsed
        write_json(path, data)
        summary.update(status="complete" if all(len(data["results"].get(r["id"], {})) == len(VARIANTS) for r in rows) else "partial",
                       experiment_sha256=identity, split=split, stop_reason=stopped,
                       elapsed_seconds=elapsed, total_session_seconds=sum(r["elapsed_seconds"] for r in data["sessions"] if r.get("stage") == "session"),
                       startup_seconds_by_variant={v: round(sum(r["elapsed_seconds"] for r in data["sessions"]
                            if r.get("stage") == "startup" and r.get("variant") == v), 3) for v in VARIANTS})
        write_json(folder / "summary.json", summary)
        return summary
    finally:
        os.close(fd)


def run_validation(config: dict, study_root: Path, budget: int) -> dict:
    """Study orchestrator API; never opens sealed test or private reference manifest."""
    try:
        return _run(config, Path(study_root), budget, "validation")
    except BenchmarkError as error:
        return {"status": "partial", "stop_reason": str(error)}
    except Exception as error:
        return {"status": "partial", "stop_reason": "benchmark_error", "error_type": type(error).__name__}


def freeze_candidate(config: dict, study_root: Path, *, candidate: str,
                     reviewed_assessment_sha256: str) -> dict:
    """Explicit coordinator action after blind validation review; reads no test."""
    folder = Path(study_root) / "benchmark"
    fd = open_private_file(folder / "benchmark.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if candidate not in VARIANTS:
            raise BenchmarkError("invalid_candidate")
        spec = experiment_spec(config, Path(study_root))
        data = read_state(folder / "validation" / "results.json")
        assessments = read_state(folder / "validation" / "blind-assessment.json")
        if (data.get("experiment_sha256") != fingerprint(spec)
                or assessments.get("experiment_sha256") != fingerprint(spec)
                or file_digest(folder / "validation" / "blind-assessment.json", private=True) != reviewed_assessment_sha256):
            raise BenchmarkError("review_or_configuration_changed")
        expected_count = config["exports"]["validation"]["count"]
        runs = [r for variants in data.get("results", {}).values() for r in variants.values()]
        if len(runs) != expected_count * len(VARIANTS):
            raise BenchmarkError("validation_incomplete")
        if candidate == "mlx_adapter" and spec["pins"]["adapter"] is None:
            raise BenchmarkError("no_selected_adapter")
        for run in runs:
            grade = assessments.get("items", {}).get(run.get("assessment_key"), {}).get("grades", {})
            if any(type(grade.get(k)) is not bool for k in GRADES):
                raise BenchmarkError("blind_review_incomplete")
            if run["status"] != "complete" and any(grade[k] for k in GRADES[:4]):
                raise BenchmarkError("failure_cannot_receive_correct_grade")
        frozen = {"version": VERSION, "candidate": candidate, "specification": spec,
                  "experiment_sha256": fingerprint(spec),
                  "reviewed_assessment_sha256": reviewed_assessment_sha256,
                  "validation_results_sha256": file_digest(folder / "validation" / "results.json", private=True)}
        frozen["frozen_candidate_sha256"] = fingerprint(frozen)
        existing = read_state(folder / "frozen-candidate.json")
        if existing and existing != frozen:
            raise BenchmarkError("candidate_already_frozen")
        write_json(folder / "frozen-candidate.json", frozen)
        return {"status": "frozen", "candidate": candidate,
                "frozen_candidate_sha256": frozen["frozen_candidate_sha256"]}
    except BlockingIOError:
        raise BenchmarkError("benchmark_busy") from None
    finally:
        os.close(fd)


def run_locked_test(config: dict, study_root: Path, budget: int, *,
                    frozen_candidate_sha256: str) -> dict:
    """No test read until final configuration and reviewed freeze match exactly."""
    authorized = False
    try:
        guard = Guard(budget)
        guard.check(35)
        folder = Path(study_root) / "benchmark"
        frozen = read_state(folder / "frozen-candidate.json")
        body = {k: v for k, v in frozen.items() if k != "frozen_candidate_sha256"}
        if not frozen or fingerprint(body) != frozen_candidate_sha256 or frozen.get("frozen_candidate_sha256") != frozen_candidate_sha256:
            raise BenchmarkError("invalid_candidate_freeze")
        spec = experiment_spec(config, Path(study_root))
        if frozen["experiment_sha256"] != fingerprint(spec) or frozen["specification"] != spec:
            raise BenchmarkError("frozen_configuration_mismatch")
        if (file_digest(folder / "validation" / "results.json", private=True) != frozen["validation_results_sha256"]
                or file_digest(folder / "validation" / "blind-assessment.json", private=True) != frozen["reviewed_assessment_sha256"]):
            raise BenchmarkError("frozen_review_changed")
        authorized = True
        return _run(config, Path(study_root), max(1, int(guard.remaining())), "test", spec)
    except BenchmarkError as error:
        return {"status": "partial", "stop_reason": str(error), "test_access_authorized": authorized}
    except Exception as error:
        return {"status": "partial", "stop_reason": "locked_test_error", "error_type": type(error).__name__,
                "test_access_authorized": authorized}
