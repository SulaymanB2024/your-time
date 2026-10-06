"""Private, bounded paired runtime exploration; never opens sealed references."""

from __future__ import annotations

import contextlib
import fcntl
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from private_io import open_private_file, write_json
from specialization_benchmark import (
    DATA_ROOT,
    GRADES,
    VARIANTS,
    BenchmarkError,
    BudgetEnded,
    Guard,
    _run,
    example_identity,
    experiment_spec,
    file_digest,
    fingerprint,
    load_examples,
    read_state,
    session_totals,
)

VERSION = "runtime_tuning_v1"
REPEATS = 2
GAIN = .15
CONDITIONS = {
    "mlx_control": {}, "mlx_cold": {"batch_requests": 1},
    "mlx_resident8": {"batch_requests": 8},
    "mlx_thinking": {"thinking": True, "thinking_budget": 256},
    "mlx_side1024": {"image_side": 1024}, "mlx_side2048": {"image_side": 2048},
    "mlx_tokens1024": {"image_tokens": 1024}, "mlx_tokens1536": {"image_tokens": 1536},
    "mlx_center80": {"image_crop": "center_80"},
    "q8_control": {"q8_threads": 4}, "q8_threads2": {"q8_threads": 2}, "q8_threads8": {"q8_threads": 8},
}
CLAIMS = {"activity_kind", "project_candidate", "task_candidate", "visible_work"}
CLAIM_GRADES = {"fully_supported", "partial", "unsupported", "not_asserted"}


def partition(rows, target=8):
    """Keep transitive episode, identical-image and near-image groups together."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    keys = {}
    for i, row in enumerate(rows):
        for field in ("episode_id", "image_sha256", "near_duplicate_group"):
            if row.get(field) is not None:
                key = (field, row[field])
                if key in keys:
                    parents[find(i)] = find(keys[key])
                keys[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row["id"])
    choices = {0: []}
    for group in sorted(groups.values(), key=fingerprint):
        for count, ids in sorted(list(choices.items()), reverse=True):
            if count + len(group) <= target:
                choices.setdefault(count + len(group), ids + group)
    possible = [n for n in choices if 4 <= n <= target and len(rows) - n >= 8]
    if not possible:
        return {"exploration_ids": [r["id"] for r in rows], "confirmation_ids": [],
                "scope": "all_validation_reused_no_clean_group_partition;not_independent_confirmation"}
    explore = choices[max(possible)]
    return {"exploration_ids": explore, "confirmation_ids": [r["id"] for r in rows if r["id"] not in explore],
            "scope": "confirmation_reserved_from_runtime_selection;validation_already_used_for_checkpoint_quality"}


def verify_review(data, assessment, variant, ids):
    records = data.get("results", {})
    if set(records) != set(ids):
        raise BenchmarkError("tuning_pairs_incomplete")
    quality = {"joint_correct": 0, "unsupported_claims": 0, "uncertainty_errors": 0,
               "complete": 0, "critical": 0, "reviewed": 0, "selected": len(ids), "by_example": {}}
    for identity in ids:
        record = records[identity].get(variant, {})
        item = assessment.get("items", {}).get(record.get("assessment_key"), {})
        grades, claims = item.get("grades", {}), item.get("claim_support", {})
        if (any(type(grades.get(k)) is not bool for k in GRADES) or item.get("example_id") != identity
                or item.get("output") != {k: record.get(k) for k in ("activity", "description", "status")}
                or set(claims) != CLAIMS or any(v not in CLAIM_GRADES for v in claims.values())):
            raise BenchmarkError("tuning_review_incomplete_or_changed")
        if record.get("status") != "complete" and any(grades[k] for k in GRADES[:4]):
            raise BenchmarkError("failure_cannot_receive_correct_grade")
        quality["reviewed"] += 1
        quality["complete"] += record.get("status") == "complete"
        quality["joint_correct"] += (record.get("status") == "complete" and all(grades[k] for k in GRADES[:3]))
        quality["unsupported_claims"] += sum(v == "unsupported" for v in claims.values())
        quality["uncertainty_errors"] += not grades["uncertainty_appropriate"]
        quality["critical"] += (grades["privacy_leak"] or grades["unsupported_completion"]
                                or record.get("raw_safety_available") is not True
                                or any((record.get("raw_safety") or {}).values())
                                or any((record.get("final_safety_proxy") or {}).values()))
        quality["by_example"][identity] = {
            "complete": record.get("status") == "complete",
            **{k: grades[k] for k in GRADES[:4]},
            "unsupported_claims": sum(v == "unsupported" for v in claims.values()),
            "critical": bool(grades["privacy_leak"] or grades["unsupported_completion"]
                             or record.get("raw_safety_available") is not True
                             or any((record.get("raw_safety") or {}).values())
                             or any((record.get("final_safety_proxy") or {}).values()))}
    return quality


@contextlib.contextmanager
def lock(root):
    fd = open_private_file(Path(root) / "tuning/tuning.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BenchmarkError("tuning_busy") from None
        yield
    finally:
        os.close(fd)


def begin(config, root, backend, reviewed_assessment_sha256):
    with lock(root):
        return _begin(config, root, backend, reviewed_assessment_sha256)


def _begin(config, root, backend, reviewed_assessment_sha256):
    """Freeze the reviewed baseline/backend; this is not candidate acceptance."""
    root = Path(root)
    if backend not in {"mlx_base", "mlx_adapter"} or (root / "benchmark/frozen-candidate.json").exists():
        raise BenchmarkError("tuning_backend_or_stage_invalid")
    spec = experiment_spec(config, root, exploratory=True)
    data = read_state(root / "benchmark/validation/results.json")
    path = root / "benchmark/validation/blind-assessment.json"
    grades = read_state(path)
    if (data.get("experiment_sha256") != fingerprint(spec) or grades.get("experiment_sha256") != fingerprint(spec)
            or file_digest(path, private=True) != reviewed_assessment_sha256
            or len(data.get("results", {})) != config["exports"]["validation"]["count"]):
        raise BenchmarkError("baseline_review_changed")
    for variant in VARIANTS:
        verify_review(data, grades, variant, list(data["results"]))
    body = {"version": VERSION, "backend": backend, "baseline_specification": spec,
            "parent_configuration_sha256": config["configuration_sha256"],
            "baseline_results_sha256": file_digest(root / "benchmark/validation/results.json", private=True),
            "baseline_review_sha256": reviewed_assessment_sha256,
            "policy": "provisional_runtime_exploration_no_test_or_promotion"}
    body["backend_choice_sha256"] = fingerprint(body)
    path = root / "tuning/backend-choice.json"
    old = read_state(path)
    if old and old != body:
        raise BenchmarkError("tuning_backend_already_frozen")
    write_json(path, body)
    return {"status": "tuning_prepared", "backend_choice_sha256": body["backend_choice_sha256"], "production_changed": False}


def backend_choice(config, root):
    body = read_state(root / "tuning/backend-choice.json")
    if (not body or fingerprint({k: v for k, v in body.items() if k != "backend_choice_sha256"}) != body.get("backend_choice_sha256")
            or body.get("parent_configuration_sha256") != config["configuration_sha256"]
            or body.get("baseline_specification") != experiment_spec(config, root, exploratory=True)):
        raise BenchmarkError("tuning_backend_changed")
    for relative, key in (("results.json", "baseline_results_sha256"), ("blind-assessment.json", "baseline_review_sha256")):
        if file_digest(root / "benchmark/validation" / relative, private=True) != body[key]:
            raise BenchmarkError("baseline_review_changed")
    return body


def make_plan(config, root, backend, rows):
    split = partition(rows)
    order = list(CONDITIONS)
    random.Random(config["seed"]).shuffle(order)
    body = {"version": VERSION, "backend_choice_sha256": backend["backend_choice_sha256"],
            "parent_configuration_sha256": config["configuration_sha256"],
            "export_sha256": config["exports"]["validation"]["sha256"],
            "conditions": CONDITIONS, "rounds": [order, list(reversed(order))], **split,
            "row_identities": {r["id"]: fingerprint(r) for r in rows},
            "selection_policy": {"minimum_wall_gain_both_rounds": GAIN, "quality_regression": "none_observed",
                                 "critical_flags": 0, "fallback": "reference_only_not_another_confirmation_choice"}}
    body["plan_sha256"] = fingerprint(body)
    return body


def condition_spec(config, root, backend, plan, condition, repeat):
    settings = {**config, **CONDITIONS[condition], "exploratory_only": True}
    spec = experiment_spec(settings, root, exploratory=True)
    variant = "context_q8" if condition.startswith("q8_") else backend["backend"]
    ids = plan["exploration_ids"] if repeat == 0 else list(reversed(plan["exploration_ids"]))
    spec.update(purpose="exploratory_tuning", tuning_plan_sha256=plan["plan_sha256"],
                condition=condition, repeat=repeat, comparison_variants=[variant], selected_validation_ids=ids)
    return settings, spec, variant, ids


def completed_cell(folder, settings, spec, variant, ids, rows):
    """Verify terminal pairs before skipping; never charge completed cells again."""
    data = read_state(folder / "benchmark/validation/results.json")
    if not data:
        return False
    if (data.get("configuration_sha256") != fingerprint(settings)
            or data.get("experiment_sha256") != fingerprint(spec)
            or data.get("specification") != spec or data.get("split") != "validation"
            or set(data.get("results", {})) - set(ids)):
        raise BenchmarkError("tuning_identity_changed")
    by_id = {row["id"]: row for row in rows}
    for identity, records in data["results"].items():
        if set(records) - {variant}:
            raise BenchmarkError("tuning_identity_changed")
        for record in records.values():
            if record.get("identity") != example_identity(by_id[identity], variant, spec):
                raise BenchmarkError("tuning_identity_changed")
    return (set(data["results"]) == set(ids)
            and all(set(records) == {variant} and records[variant].get("status") not in {None, "running"}
                    for records in data["results"].values())
            and not any(s.get("status") == "running" for s in data.get("sessions", []) if s.get("stage") == "session"))


def clean_blocks(data, variant):
    """Partial or failed lifetimes are measured costs, not controlled trials."""
    blocks = [s for s in data.get("sessions", []) if s.get("stage") == "startup"]
    return bool(blocks) and all(s.get("variant") == variant and s.get("intended_requests", 0) > 0
                               and s.get("attempted_requests") == s["intended_requests"]
                               and s.get("completed_requests") == s["intended_requests"] for s in blocks)


def run(config, root, budget):
    root = Path(root)
    guard, started = Guard(budget), time.monotonic()
    try:
        guard.check(35)
    except BudgetEnded as error:
        return {"status": "partial", "stop_reason": str(error), "private_examples_opened": False}
    fd = open_private_file(root / "tuning/tuning.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "tuning/runtime-choice.json").exists():
            raise BenchmarkError("tuning_already_selected")
        journal_path = root / "tuning/sessions.json"
        journal = read_state(journal_path) or {"sessions": []}
        for old in journal["sessions"]:
            if old["status"] == "running":
                old["status"] = "interrupted"
        session = {"stage": "session", "status": "running", "elapsed_seconds": None,
                   "reserved_seconds": budget + 30, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        journal["sessions"].append(session)
        write_json(journal_path, journal)
        backend = backend_choice(config, root)
        rows = load_examples(DATA_ROOT / "export/validation.jsonl", "validation", config["exports"]["validation"])
        plan = make_plan(config, root, backend, rows)
        old = read_state(root / "tuning/plan.json")
        if old and old != plan:
            raise BenchmarkError("tuning_plan_changed")
        write_json(root / "tuning/plan.json", plan)
        completed, stop = 0, None
        try:
            for repeat, order in enumerate(plan["rounds"]):
                for condition in order:
                    guard.check(35)
                    settings, spec, variant, ids = condition_spec(config, root, backend, plan, condition, repeat)
                    folder = root / "tuning" / f"round-{repeat}" / condition
                    if completed_cell(folder, settings, spec, variant, ids, rows):
                        completed += 1
                        continue
                    receipt = _run(settings, folder, max(1, int(guard.remaining() - 30)), "validation", spec,
                                   variants=(variant,), row_ids=ids)
                    if receipt.get("status") != "complete":
                        stop = receipt.get("stop_reason") or "tuning_partial"
                        raise BudgetEnded(stop)
                    completed += 1
        except BudgetEnded as error:
            stop = str(error)
        session.update(status="finished", elapsed_seconds=round(time.monotonic() - started, 3), stop_reason=stop)
        write_json(journal_path, journal)
        result = {"status": "complete" if completed == REPEATS * len(CONDITIONS) else "partial",
                  "completed_condition_rounds": completed, "condition_rounds": REPEATS * len(CONDITIONS),
                  "unique_exploration_examples": len(plan["exploration_ids"]),
                  "confirmation_examples": len(plan["confirmation_ids"]), "partition_scope": plan["scope"],
                  "plan_sha256": plan["plan_sha256"], "stop_reason": stop,
                  "capacity_scope": "controlled_runtime_exploration_not_daily_coverage_or_general_accuracy",
                  "production_changed": False, **session_totals(journal)}
        write_json(root / "tuning/summary.json", result)
        return result
    except BlockingIOError:
        return {"status": "partial", "stop_reason": "tuning_busy"}
    finally:
        os.close(fd)


def quality_not_worse(candidate, control):
    aggregate_ok = (candidate["complete"] == candidate["selected"] and candidate["joint_correct"] >= control["joint_correct"]
            and candidate["unsupported_claims"] == 0
            and candidate["uncertainty_errors"] <= control["uncertainty_errors"] and candidate["critical"] == 0)
    pairs, baseline = candidate.get("by_example", {}), control.get("by_example", {})
    if set(pairs) != set(baseline):
        return False
    return aggregate_ok and all(
        not pairs[key]["critical"] and pairs[key]["unsupported_claims"] == 0
        and all(not baseline[key][grade] or pairs[key][grade] for grade in ("complete", *GRADES[:4]))
        for key in pairs)


def freeze_choice(config, root, reviewed_hashes):
    """Nominate controls from complete independent review, never from test."""
    with lock(root):
        return _freeze_choice(config, root, reviewed_hashes)


def _freeze_choice(config, root, reviewed_hashes):
    root = Path(root)
    backend = backend_choice(config, root)
    plan = read_state(root / "tuning/plan.json")
    if (not plan or fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]
            or plan["backend_choice_sha256"] != backend["backend_choice_sha256"]):
        raise BenchmarkError("tuning_plan_changed")
    measurements, hashes = {}, {}
    for condition in CONDITIONS:
        measurements[condition] = []
        for repeat in range(REPEATS):
            settings, spec, variant, ids = condition_spec(config, root, backend, plan, condition, repeat)
            folder = root / "tuning" / f"round-{repeat}" / condition / "benchmark/validation"
            key = f"{repeat}:{condition}"
            sha = file_digest(folder / "blind-assessment.json", private=True)
            if sha != reviewed_hashes.get(key):
                raise BenchmarkError("tuning_review_changed")
            data, grades = read_state(folder / "results.json"), read_state(folder / "blind-assessment.json")
            if data.get("experiment_sha256") != fingerprint(spec) or grades.get("experiment_sha256") != fingerprint(spec):
                raise BenchmarkError("tuning_identity_changed")
            quality = verify_review(data, grades, variant, ids)
            cost = session_totals(data)
            hashes[key] = {"results_sha256": file_digest(folder / "results.json", private=True), "review_sha256": sha}
            if cost["total_session_seconds"] is None or cost["interrupted_sessions"] or not clean_blocks(data, variant):
                quality["clean_runtime_trial"] = False
            else:
                quality["clean_runtime_trial"] = True
            measurements[condition].append({"quality": quality, "seconds": cost["conservative_total_seconds"]})
    chosen = {}
    for prefix, control in (("mlx_", "mlx_control"), ("q8_", "q8_control")):
        if any(m["quality"]["critical"] for m in measurements[control]):
            raise BenchmarkError("reference_runtime_has_critical_flags")
        eligible = []
        for condition in CONDITIONS:
            if condition == control or not condition.startswith(prefix):
                continue
            pairs = list(zip(measurements[condition], measurements[control]))
            if all(quality_not_worse(a["quality"], b["quality"]) and a["quality"]["clean_runtime_trial"]
                   and b["quality"]["clean_runtime_trial"] and a["seconds"] is not None and b["seconds"]
                   and a["seconds"] <= (1 - GAIN) * b["seconds"] for a, b in pairs):
                eligible.append(condition)
        chosen[prefix] = min(eligible, key=lambda c: sum(m["seconds"] for m in measurements[c])) if eligible else control
    overrides = {**CONDITIONS[chosen["mlx_"]], **CONDITIONS[chosen["q8_"]]}
    body = {"version": VERSION, "parent_configuration_sha256": config["configuration_sha256"],
            "plan_sha256": plan["plan_sha256"], "nominated_conditions": chosen, "overrides": overrides,
            "reviewed_conditions": hashes, "confirmation_ids": plan["confirmation_ids"],
            "partition_scope": plan["scope"], "policy": "nomination_requires_full_validation_confirmation_before_test"}
    body["runtime_choice_sha256"] = fingerprint(body)
    path = root / "tuning/runtime-choice.json"
    old = read_state(path)
    if old and old != body:
        raise BenchmarkError("runtime_choice_already_frozen")
    write_json(path, body)
    return {"status": "runtime_nominated", "runtime_choice_sha256": body["runtime_choice_sha256"],
            "settings_changed": any(chosen[k] != c for k, c in (("mlx_", "mlx_control"), ("q8_", "q8_control"))),
            "confirmation_examples": len(plan["confirmation_ids"]), "production_changed": False}


def verify_choice(config, root, choice):
    """Recheck the reviewed evidence closure before using nominated controls."""
    backend = backend_choice(config, root)
    plan = read_state(root / "tuning/plan.json")
    if (not plan or fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256")
            or plan.get("backend_choice_sha256") != backend["backend_choice_sha256"]
            or plan.get("conditions") != CONDITIONS or plan.get("parent_configuration_sha256") != config["configuration_sha256"]
            or choice.get("plan_sha256") != plan["plan_sha256"]
            or choice.get("confirmation_ids") != plan["confirmation_ids"] or choice.get("partition_scope") != plan["scope"]):
        raise BenchmarkError("tuning_plan_changed")
    nominated = choice.get("nominated_conditions", {})
    if (set(nominated) != {"mlx_", "q8_"}
            or any(nominated[prefix] not in CONDITIONS or not nominated[prefix].startswith(prefix) for prefix in nominated)
            or choice["overrides"] != {**CONDITIONS[nominated["mlx_"]], **CONDITIONS[nominated["q8_"]]}):
        raise BenchmarkError("runtime_choice_controls_changed")
    proofs = choice.get("reviewed_conditions", {})
    if set(proofs) != {f"{repeat}:{condition}" for repeat in range(REPEATS) for condition in CONDITIONS}:
        raise BenchmarkError("tuning_proof_incomplete")
    for repeat in range(REPEATS):
        for condition in CONDITIONS:
            folder = root / "tuning" / f"round-{repeat}" / condition / "benchmark/validation"
            proof = proofs[f"{repeat}:{condition}"]
            for relative, key in (("results.json", "results_sha256"), ("blind-assessment.json", "review_sha256")):
                if file_digest(folder / relative, private=True) != proof.get(key):
                    raise BenchmarkError("tuning_proof_changed")


def confirm(config, root, budget):
    root = Path(root)
    guard = Guard(budget)
    try:
        guard.check(35)
    except BudgetEnded as error:
        return {"status": "partial", "stop_reason": str(error), "private_examples_opened": False}
    if (root / "tuning/runtime-decision.json").exists():
        raise BenchmarkError("runtime_confirmation_already_reviewed")
    spec = experiment_spec(config, root)
    if not spec.get("runtime_choice"):
        raise BenchmarkError("runtime_selection_required")
    return _run(config, root / "confirmation", max(1, int(guard.remaining() - 30)), "validation", spec)


def finalize_confirmation(config, root, reviewed_assessment_sha256):
    """Confirm the sole nomination, or retain the original controls once."""
    root = Path(root)
    with lock(root):
        backend = backend_choice(config, root)
        choice = read_state(root / "tuning/runtime-choice.json")
        spec = experiment_spec(config, root, nomination_only=True)
        folder = root / "confirmation/benchmark/validation"
        data, assessment = read_state(folder / "results.json"), read_state(folder / "blind-assessment.json")
        if (not choice or data.get("experiment_sha256") != fingerprint(spec)
                or assessment.get("experiment_sha256") != fingerprint(spec)
                or file_digest(folder / "blind-assessment.json", private=True) != reviewed_assessment_sha256
                or len(data.get("results", {})) != config["exports"]["validation"]["count"]):
            raise BenchmarkError("runtime_confirmation_incomplete_or_changed")
        full_quality = {variant: verify_review(data, assessment, variant, list(data["results"])) for variant in VARIANTS}
        ids = choice["confirmation_ids"] or list(data["results"])
        baseline = read_state(root / "benchmark/validation/results.json")
        baseline_grades = read_state(root / "benchmark/validation/blind-assessment.json")
        selected, reference, full_reference = {}, {}, {}
        for variant in (backend["backend"], "context_q8"):
            selected[variant] = verify_review({"results": {key: data["results"][key] for key in ids}}, assessment, variant, ids)
            reference[variant] = verify_review({"results": {key: baseline["results"][key] for key in ids}}, baseline_grades, variant, ids)
            full_reference[variant] = verify_review(baseline, baseline_grades, variant, list(baseline["results"]))
        use_nominated = all(quality_not_worse(selected[v], reference[v])
                            and quality_not_worse(full_quality[v], full_reference[v]) for v in selected)
        body = {"version": VERSION, "parent_configuration_sha256": config["configuration_sha256"],
                "runtime_choice_sha256": choice["runtime_choice_sha256"], "use_nominated": use_nominated,
                "confirmation_results_sha256": file_digest(folder / "results.json", private=True),
                "confirmation_review_sha256": reviewed_assessment_sha256,
                "scored_examples": len(ids), "partition_scope": choice["partition_scope"],
                "confirmation_quality": selected, "reference_quality": reference,
                "all_validation_quality": full_quality, "all_validation_reference_quality": full_reference,
                "policy": "single_nomination_or_original_reference;no_alternative_tuning_from_confirmation"}
        body["runtime_decision_sha256"] = fingerprint(body)
        path = root / "tuning/runtime-decision.json"
        old = read_state(path)
        if old and old != body:
            raise BenchmarkError("runtime_decision_already_frozen")
        write_json(path, body)
        return {"status": "runtime_confirmed" if use_nominated else "reference_retained",
                "runtime_decision_sha256": body["runtime_decision_sha256"], "production_changed": False}
