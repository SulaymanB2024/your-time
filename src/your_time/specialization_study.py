"""Resume a finite private specialization study before ordinary nightly vision."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from model_execution import ModelBusy
from overnight_schedule import VISION_END, ZONE, remaining_seconds
from private_io import open_private_file, write_json
from secure_store import STATE_DIR
from specialization_assets import ASSET_MANIFEST, MODEL_DIR
from specialization_dataset import ROOT as DATA_ROOT
from vision_batch import resource_gate

PROJECT = Path(__file__).resolve().parent
STUDY_ROOT = DATA_ROOT / "study-v1"
CONFIG = STUDY_ROOT / "configuration.json"
LATEST = STUDY_ROOT / "latest.json"
MAX_NIGHT_SECONDS = 5400
MAX_NIGHTS = 14
CODE = ("activity_context.py", "specialization_training.py", "specialization_worker.py",
        "specialization_study.py", "specialization_benchmark.py", "requirements-ml.txt", "inference_telemetry.py", "model_execution.py", "model_deadline.py",
        "engine_identity.py", "vision_quality_eval.py", "vision_batch.py", "vision_fallback.py",
        "specialization_dataset.py", "specialization_assets.py", "overnight_schedule.py",
        "private_io.py", "secure_store.py", "network-off.sb", "overnight_vision.zsh",
        "specialization_trial.py", "chronicle_activity.py", "activity_episode_store.py",
        "daily_analysis.py", "task_corrections.py", "specialization_tuning.py")


def read_json(path: Path) -> dict:
    with os.fdopen(open_private_file(path, os.O_RDONLY)) as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration_identity(body: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != "configuration_sha256"},
                                     sort_keys=True).encode()).hexdigest()


def configuration_matches(config: dict, state: dict) -> bool:
    return (config.get("configuration_sha256") == configuration_identity(config)
            and state.get("configuration_sha256") == config.get("configuration_sha256"))


def prepare() -> dict:
    export = read_json(DATA_ROOT / "export-receipt.json")
    body = {"version": "specialization_study_v1", "source_sha256": {name: digest(PROJECT / name) for name in CODE},
            "exports": export["splits"], "assets_sha256": digest(ASSET_MANIFEST),
            "night_seconds": MAX_NIGHT_SECONDS, "max_nights": MAX_NIGHTS,
            "image_tokens": 512, "image_side": 1600, "context_tokens": 4096,
            "thinking": False, "seed": 20261005,
            "variants": ["production_q8", "context_q8", "mlx_base", "mlx_adapter"],
            "tuning_required": True,
            "promotion": "requires_coordinator_quality_review_and_three_night_trial"}
    body["configuration_sha256"] = configuration_identity(body)
    if CONFIG.exists() and read_json(CONFIG) != body:
        raise ValueError("study_configuration_changed_requires_new_study")
    write_json(CONFIG, body)
    if not LATEST.exists():
        write_json(LATEST, {"stage": "verify", "status": "prepared", "nights": [],
                            "configuration_sha256": body["configuration_sha256"]})
    return {"status": "prepared", "configuration_sha256": body["configuration_sha256"],
            "night_seconds": MAX_NIGHT_SECONDS, "max_nights": MAX_NIGHTS}


def refreeze_unstarted() -> dict:
    """Amend source and required stages before the first experimental run."""
    fd = open_private_file(STUDY_ROOT / "study.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, state = read_json(CONFIG), read_json(LATEST)
        if (not configuration_matches(config, state) or state.get("stage") != "verify"
                or state.get("status") != "prepared" or state.get("nights") != []):
            raise ValueError("only_unstarted_prepared_study_can_refreeze")
        for name in ("training", "benchmark", "tuning", "confirmation", "trial", "receipts"):
            folder = STUDY_ROOT / name
            if folder.is_symlink() or (folder.exists() and any(folder.iterdir())):
                raise ValueError("experimental_artifacts_prevent_refreeze")
        if digest(ASSET_MANIFEST) != config["assets_sha256"]:
            raise ValueError("assets_changed_requires_new_study")
        for split, item in config["exports"].items():
            folder = "sealed" if split == "test" else "export"
            if digest(DATA_ROOT / folder / (split + ".jsonl")) != item["sha256"]:
                raise ValueError("dataset_changed_requires_new_study")
        changed = {**config, "source_sha256": {name: digest(PROJECT / name) for name in CODE}, "tuning_required": True}
        changed["configuration_sha256"] = configuration_identity(changed)
        if changed == config:
            return {"status": "already_frozen", "configuration_sha256": config["configuration_sha256"]}
        write_json(STUDY_ROOT / "configuration-history" / (config["configuration_sha256"] + ".json"), {
            "configuration": config, "state": state, "reason": "reviewed_source_only_before_first_run",
            "replaced_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(CONFIG, changed)
        state["configuration_sha256"] = changed["configuration_sha256"]
        write_json(LATEST, state)
        return {"status": "refrozen_unstarted", "configuration_sha256": changed["configuration_sha256"],
                "prior_configuration_sha256": config["configuration_sha256"], "nights_started": 0}
    finally:
        os.close(fd)


def training_command(stage: str, budget: int) -> list[str]:
    return [str(PROJECT / ".ml-venv/bin/python"), str(PROJECT / "specialization_training.py"), stage,
            "--model-path", str(MODEL_DIR), "--model-manifest", str(ASSET_MANIFEST),
            "--train-jsonl", str(DATA_ROOT / "export/train.jsonl"),
            "--validation-jsonl", str(DATA_ROOT / "export/validation.jsonl"),
            "--run-dir", str(STUDY_ROOT / "training"), "--state-dir", str(STATE_DIR),
            "--image-tokens", "512", "--max-seconds", str(budget)]


def run() -> dict:
    if not CONFIG.exists():
        return {"status": "not_prepared"}
    config, state = read_json(CONFIG), read_json(LATEST)
    if not configuration_matches(config, state):
        return {"status": "configuration_changed", "action": "review_before_refreezing"}
    if any(digest(PROJECT / name) != expected for name, expected in config["source_sha256"].items()):
        return {"status": "source_changed", "action": "review_before_refreezing"}
    if state["stage"] in {"quality_review", "tuning_quality_review", "confirmation_quality_review", "candidate_quality_review", "test_quality_review", "trial_quality_review", "complete"}:
        return {"status": state["status"], "stage": state["stage"]}
    if digest(ASSET_MANIFEST) != config["assets_sha256"]:
        return {"status": "assets_changed", "action": "review_before_refreezing"}
    for split in ("train", "validation"):
        if digest(DATA_ROOT / "export" / (split + ".jsonl")) != config["exports"][split]["sha256"]:
            return {"status": "dataset_changed", "action": "review_before_refreezing"}
    available = remaining_seconds(datetime.now(timezone.utc), end=VISION_END)
    if available < 600:
        return {"status": "outside_study_window"}
    reason = resource_gate(benchmark_now=False)
    if reason:
        return {"status": "resource_gate", "reason": reason}
    fd = open_private_file(STUDY_ROOT / "study.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "study_busy"}
        # A contender may have observed old state before the preceding owner
        # finished. Never use that snapshot to advance stages or spend budgets.
        config, state = read_json(CONFIG), read_json(LATEST)
        if not configuration_matches(config, state):
            return {"status": "configuration_changed", "action": "review_before_refreezing"}
        if digest(ASSET_MANIFEST) != config["assets_sha256"]:
            return {"status": "assets_changed", "action": "review_before_refreezing"}
        for split in ("train", "validation"):
            if digest(DATA_ROOT / "export" / (split + ".jsonl")) != config["exports"][split]["sha256"]:
                return {"status": "dataset_changed", "action": "review_before_refreezing"}
        if state["stage"] in {"quality_review", "tuning_quality_review", "confirmation_quality_review", "candidate_quality_review", "test_quality_review", "trial_quality_review", "complete"}:
            return {"status": state["status"], "stage": state["stage"]}
        if any(digest(PROJECT / name) != expected for name, expected in config["source_sha256"].items()):
            return {"status": "source_changed", "action": "review_before_refreezing"}
        available = remaining_seconds(datetime.now(timezone.utc), end=VISION_END)
        if available < 600:
            return {"status": "outside_study_window"}
        reason = resource_gate(benchmark_now=False)
        if reason:
            return {"status": "resource_gate", "reason": reason}
        today = datetime.now(ZONE).date().isoformat()
        used = next((r["elapsed_seconds"] + r.get("reserved_seconds", 0)
                     for r in state["nights"] if r["day"] == today), 0)
        if len(state["nights"]) >= MAX_NIGHTS and not any(r["day"] == today for r in state["nights"]):
            return {"status": "night_limit", "action": "review_measured_results"}
        budget = int(min(MAX_NIGHT_SECONDS - used, available - 300))
        if budget < 600:
            return {"status": "night_study_budget_used"}
        entry = next((r for r in state["nights"] if r["day"] == today), None)
        if entry is None:
            entry = {"day": today, "elapsed_seconds": 0}
            state["nights"].append(entry)
        # Reserve before spawning. If the controller crashes, the full reserved
        # allowance remains charged; a child may still hold the model lock.
        entry["reserved_seconds"] = budget
        started = time.monotonic()
        stage = state["stage"]
        state.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(LATEST, state)
        try:
            if stage in {"verify", "train", "validate"}:
                # The harness owns the inherited model lock and network sandbox.
                completed = subprocess.run(training_command(stage, budget - 30), capture_output=True,
                                           timeout=budget)
                try:
                    result = json.loads(completed.stdout)
                except ValueError:
                    result = {"status": "failed", "error": "invalid_training_receipt"}
                if completed.returncode and result.get("status") not in {"failed", "worker_failed"}:
                    result = {"status": "failed", "error": "training_process_error"}
                next_stage = {("verify", "verified"): "train", ("train", "complete"): "validate",
                              ("validate", "validated"): "benchmark"}.get((stage, result.get("status")), stage)
            elif stage in {"benchmark", "test"}:
                # Imported lazily: no private example reads during status or dry-run.
                from specialization_benchmark import run_locked_test, run_validation

                if stage == "benchmark":
                    result = run_validation(config, STUDY_ROOT, budget - 30)
                    next_stage = "quality_review" if result.get("status") == "complete" else stage
                else:
                    frozen = read_json(STUDY_ROOT / "benchmark/frozen-candidate.json")
                    result = run_locked_test(config, STUDY_ROOT, budget - 30,
                                             frozen_candidate_sha256=frozen["frozen_candidate_sha256"])
                    next_stage = "test_quality_review" if result.get("status") == "complete" else stage
            elif stage == "trial":
                from specialization_trial import recommendation, run_trial

                result = run_trial(config, STUDY_ROOT, budget - 30)
                next_stage = "trial_quality_review" if result.get("status") == "trial_complete" else stage
                if next_stage == "trial_quality_review":
                    recommendation(STUDY_ROOT)
            elif stage in {"tune", "confirm"}:
                from specialization_tuning import confirm
                from specialization_tuning import run as tune

                result = (tune if stage == "tune" else confirm)(config, STUDY_ROOT, budget - 30)
                next_stage = ("tuning_quality_review" if stage == "tune" else "confirmation_quality_review") if result.get("status") == "complete" else stage
            else:
                result, next_stage = {"status": "failed", "error": "unknown_study_stage"}, stage
        except subprocess.TimeoutExpired:
            result, next_stage = {"status": "partial", "stop_reason": "study_deadline"}, stage
        except ModelBusy:
            result, next_stage = {"status": "model_busy"}, stage
        except Exception as error:
            # Exception bodies and package output may contain activity; retain codes only.
            result, next_stage = {"status": "failed", "error_type": type(error).__name__}, stage
        elapsed = round(time.monotonic() - started, 3)
        entry["elapsed_seconds"] += elapsed
        entry.pop("reserved_seconds", None)
        history = {"stage": stage, "result": result, "elapsed_seconds": elapsed,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "configuration_sha256": config["configuration_sha256"]}
        write_json(STUDY_ROOT / "receipts" / (str(time.time_ns()) + ".json"), history)
        state.update(stage=next_stage, status="awaiting_" + next_stage if next_stage.endswith("quality_review") else result.get("status"),
                     latest_result=result, finished_at_utc=history["finished_at_utc"])
        write_json(LATEST, state)
        return {"status": state["status"], "stage": state["stage"], "elapsed_seconds": elapsed,
                "result": result, "nights_started": len(state["nights"])}
    finally:
        os.close(fd)


def accept_validation(candidate: str, assessment_sha256: str) -> dict:
    """Explicit root review freezes a candidate before allowing test generation."""
    from specialization_benchmark import freeze_candidate

    fd = open_private_file(STUDY_ROOT / "study.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, state = read_json(CONFIG), read_json(LATEST)
        if not configuration_matches(config, state):
            raise ValueError("study_configuration_changed")
        if state["stage"] != ("candidate_quality_review" if config.get("tuning_required") else "quality_review"):
            raise ValueError("validation_review_stage_required")
        if any(digest(PROJECT / name) != expected for name, expected in config["source_sha256"].items()):
            raise ValueError("study_source_changed")
        result = freeze_candidate(config, STUDY_ROOT, candidate=candidate,
                                  reviewed_assessment_sha256=assessment_sha256)
        state.update(stage="test", status="candidate_frozen", frozen_candidate=result)
        write_json(LATEST, state)
        return result
    finally:
        os.close(fd)


def begin_tuning(backend: str, assessment_sha256: str) -> dict:
    from specialization_tuning import begin

    return tuning_action("quality_review", "tune", lambda config: begin(config, STUDY_ROOT, backend, assessment_sha256))


def accept_tuning(reviewed_hashes: dict) -> dict:
    from specialization_tuning import freeze_choice

    return tuning_action("tuning_quality_review", "confirm", lambda config: freeze_choice(config, STUDY_ROOT, reviewed_hashes))


def accept_confirmation(assessment_sha256: str) -> dict:
    from specialization_tuning import finalize_confirmation

    return tuning_action("confirmation_quality_review", "candidate_quality_review", lambda config: finalize_confirmation(config, STUDY_ROOT, assessment_sha256))


def tuning_action(required_stage, next_stage, action):
    fd = open_private_file(STUDY_ROOT / "study.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, state = read_json(CONFIG), read_json(LATEST)
        if not configuration_matches(config, state) or state["stage"] != required_stage:
            raise ValueError("tuning_review_stage_required")
        if any(digest(PROJECT / name) != expected for name, expected in config["source_sha256"].items()):
            raise ValueError("study_source_changed")
        result = action(config)
        state.update(stage=next_stage, status="awaiting_" + next_stage if next_stage.endswith("quality_review") else result["status"])
        write_json(LATEST, state)
        return result
    finally:
        os.close(fd)


def accept_test(assessment_sha256: str) -> dict:
    """Bind a complete coordinating held-out review to a shadow-only trial."""
    from specialization_trial import freeze_trial

    fd = open_private_file(STUDY_ROOT / "study.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, state = read_json(CONFIG), read_json(LATEST)
        if not configuration_matches(config, state):
            raise ValueError("study_configuration_changed")
        if state["stage"] != "test_quality_review":
            raise ValueError("test_review_stage_required")
        if any(digest(PROJECT / name) != expected for name, expected in config["source_sha256"].items()):
            raise ValueError("study_source_changed")
        result = freeze_trial(config, STUDY_ROOT, reviewed_test_sha256=assessment_sha256)
        state.update(stage="trial", status="trial_frozen", frozen_trial=result)
        write_json(LATEST, state)
        return result
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "refreeze-unstarted", "run", "status", "begin-tuning", "accept-tuning", "accept-confirmation", "accept-validation", "accept-test"))
    parser.add_argument("--candidate", choices=("production_q8", "context_q8", "mlx_base", "mlx_adapter"))
    parser.add_argument("--assessment-sha256")
    parser.add_argument("--review-hashes", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == "prepare":
        value = prepare()
    elif args.action == "refreeze-unstarted":
        value = refreeze_unstarted()
    elif args.action == "accept-validation":
        value = accept_validation(args.candidate, args.assessment_sha256)
    elif args.action == "accept-test":
        value = accept_test(args.assessment_sha256)
    elif args.action == "begin-tuning":
        value = begin_tuning(args.candidate, args.assessment_sha256)
    elif args.action == "accept-tuning":
        value = accept_tuning(read_json(args.review_hashes))
    elif args.action == "accept-confirmation":
        value = accept_confirmation(args.assessment_sha256)
    elif args.action == "status":
        value = read_json(LATEST) if LATEST.exists() else {"status": "not_prepared"}
    else:
        value = run()
        # Advance completed stages in this night's fixed budget before handing
        # compute back to daily vision. Never retry a failed stage in this loop.
        while value.get("status") in {"verified", "complete", "validated"} and value.get("stage") not in {"quality_review", "tuning_quality_review", "confirmation_quality_review", "candidate_quality_review", "test_quality_review", "trial_quality_review", "complete"}:
            value = run()
        write_json(STUDY_ROOT / "admission-latest.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(), **value})
    print(json.dumps(value))
