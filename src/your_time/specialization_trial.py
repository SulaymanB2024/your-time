"""Frozen three-night shadow trial; no production writes or automatic promotion."""

from __future__ import annotations

import fcntl
import math
import os
import sqlite3
import time
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from activity_context import metadata_identity, pack
from activity_episode_store import supports_anchor
from chronicle_activity import episodes_for_day
from daily_analysis import ZONE, union_seconds
from inference_telemetry import SAFE_LABEL, record_result
from model_execution import ModelBusy
from private_io import open_private_file, write_json
from secure_store import DB_PATH, STATE_DIR
from specialization_benchmark import (
    GRADES,
    VARIANTS,
    BenchmarkError,
    BudgetEnded,
    Guard,
    Runner,
    experiment_spec,
    file_digest,
    fingerprint,
    pre_inference_refusal,
    read_state,
    response_record,
    session_totals,
    validation_location,
)
from specialization_dataset import eligible_file
from specialization_worker import WorkerError
from vision_batch import SENSITIVE_RE, spread_selection

VERSION = "specialization_shadow_trial_v1"
CLAIM_GRADES = {"fully_supported", "partial", "unsupported", "not_asserted"}
FRAME_CEILING = 250
NIGHTS = 3


def at(value):
    instant = datetime.fromisoformat(value) if isinstance(value, str) else value
    if instant.tzinfo is None:
        raise BenchmarkError("trial_time_requires_timezone")
    return instant.astimezone(timezone.utc)


def bins(intervals):
    """UTC bins preserve the two occurrences of a repeated local hour."""
    result = set()
    for lower, upper in intervals:
        cursor = math.floor(at(lower).timestamp() / 1800)
        while cursor * 1800 < at(upper).timestamp():
            result.add(cursor)
            cursor += 1
    return result


def support(episode, timestamp):
    instant = at(timestamp)
    if (episode["state"] not in {"active", "unattributed"} or not supports_anchor(episode, instant)
            or any(r["support_barrier"] for r in episode["samples"])):
        return []
    lower, upper = instant - timedelta(seconds=30), instant + timedelta(seconds=30)
    return [(max(lower, at(r["start_utc"])), min(upper, at(r["end_utc"])))
            for r in episode["samples"] if max(lower, at(r["start_utc"])) < min(upper, at(r["end_utc"]))]


def build_cohort(day, rows, episodes, limit, *, file_check=lambda path: True,
                 context_builder=pack, image_digest=file_digest):
    """Freeze full-day denominators before choosing bounded inference inputs."""
    if type(limit) is not int or not 1 <= limit <= FRAME_CEILING:
        raise BenchmarkError("invalid_trial_quota")
    foreground = [(at(r["start_utc"]), at(r["end_utc"])) for e in episodes
                  if e["state"] in {"active", "unattributed"} for r in e["samples"]]
    candidates, excluded, seen = [], Counter(), set()
    episode_starts = [at(e["start_utc"]) for e in episodes]
    for path, timestamp, app, window, ocr, ocr_status in rows:
        if path in seen:
            excluded["duplicate_observation"] += 1
            continue
        seen.add(path)
        if at(timestamp).astimezone(ZONE).date() != day:
            raise BenchmarkError("cohort_observation_outside_day")
        if ocr_status != "complete":
            excluded["ocr_unavailable"] += 1
            continue
        if SENSITIVE_RE.search(" ".join(str(v or "") for v in (app, window, ocr))):
            excluded["sensitive_filtered"] += 1
            continue
        if not file_check(Path(path)):
            excluded["missing_or_invalid_image"] += 1
            continue
        index = bisect_right(episode_starts, at(timestamp)) - 1
        episode = episodes[index] if index >= 0 and supports_anchor(episodes[index], timestamp) else None
        if episode is None or episode["state"] not in {"active", "unattributed"}:
            excluded["no_foreground_anchor"] += 1
            continue
        if any(metadata_identity(r.get("app"), r.get("window")) != metadata_identity(app, window)
               for r in episode["samples"]):
            excluded["collector_metadata_conflict"] += 1
            continue
        runs = support(episode, timestamp)
        if not runs:
            excluded["support_barrier"] += 1
            continue
        candidates.append({"id": fingerprint({"path": path, "timestamp": timestamp}), "images": [path],
                           "timestamp_utc": timestamp, "episode_id": episode["id"],
                           "episode_sha256": episode["episode_sha256"],
                           "collector_metadata_sha256": metadata_identity(app, window),
                           "support_runs": [[lo.isoformat(), hi.isoformat()] for lo, hi in runs],
                           "potential_seconds": union_seconds(runs),
                           "bin": int(at(timestamp).timestamp() // 1800)})
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["bin"]].append(row)
    queues = {}
    for key, values in grouped.items():
        # Visit different episodes before taking extra observations of one.
        by_episode = defaultdict(list)
        for row in sorted(values, key=lambda r: (-r["potential_seconds"], r["id"])):
            by_episode[row["episode_id"]].append(row)
        diverse = []
        while any(by_episode.values()):
            for group in by_episode.values():
                if group:
                    diverse.append(group.pop(0))
        queues[key] = deque(diverse)
    keys, selected = sorted(queues), []
    if limit < len(keys):
        keys = spread_selection(keys, limit)
    while len(selected) < limit and any(queues[k] for k in keys):
        for key in keys:
            if queues[key] and len(selected) < limit:
                selected.append(queues[key].popleft())
    for row in selected:
        row["context"] = context_builder(row["images"][0])
        primary = [e for e in row["context"]["evidence"] if e.get("source") == "screen_context"]
        if (len(primary) != 1 or at(primary[0]["timestamp_utc"]) != at(row["timestamp_utc"])
                or primary[0].get("metadata_sha256") != row["collector_metadata_sha256"]):
            raise BenchmarkError("cohort_context_changed")
        row["image_sha256"] = image_digest(Path(row["images"][0]))
    body = {"version": VERSION, "source_day": day.isoformat(),
            "raw_captures": len(seen), "eligible_captures": len(candidates), "excluded": dict(excluded),
            "occupied_foreground_bins": sorted(bins(foreground)), "eligible_bins": sorted(grouped),
            "foreground_seconds": union_seconds(foreground),
            "foreground_episode_count": sum(e["state"] in {"active", "unattributed"} for e in episodes),
            "eligible_episode_count": len({r["episode_id"] for r in candidates}),
            "collector_sha256": fingerprint(episodes), "selected": selected,
            "limits": "Frozen source snapshot; candidate support is not verified attention or task accuracy."}
    body["cohort_sha256"] = fingerprint(body)
    return body


def load_cohort(day, limit, guard):
    start = datetime.combine(day, datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
    guard.check(35)
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT path,timestamp_utc,active_app,active_window,ocr_text,ocr_status "
                                  "FROM screenshots WHERE timestamp_utc>=? AND timestamp_utc<? ORDER BY timestamp_utc",
                                  (start.isoformat(), end.isoformat())).fetchall()
    finally:
        connection.close()
    roots = (STATE_DIR / "screenshots", STATE_DIR / "pensieve/screenshots")
    def check(path):
        guard.check(35)
        return eligible_file(path, roots)
    return build_cohort(day, rows, episodes_for_day(day), limit, file_check=check,
                        image_digest=lambda p: file_digest(p, guard, private=True))


def semantic_totals(data, assessment, count):
    """Aggregate coordinating grades; no model grades or newly invented answers."""
    if len(data.get("results", {})) != count:
        raise BenchmarkError("held_out_comparison_incomplete")
    result = {v: Counter(selected=count) for v in VARIANTS}
    for identity, variants in data["results"].items():
        if set(variants) != set(VARIANTS):
            raise BenchmarkError("held_out_comparison_incomplete")
        for variant, record in variants.items():
            item = assessment.get("items", {}).get(record.get("assessment_key"), {})
            grades = item.get("grades", {})
            if (item.get("example_id") != identity or any(type(grades.get(k)) is not bool for k in GRADES)
                    or item.get("output") != {k: record.get(k) for k in ("activity", "description", "status")}):
                raise BenchmarkError("held_out_review_incomplete_or_changed")
            if record["status"] != "complete" and any(grades[k] for k in GRADES[:4]):
                raise BenchmarkError("failure_cannot_receive_correct_grade")
            claim = item.get("claim_support", {})
            if set(claim) != {"activity_kind", "project_candidate", "task_candidate", "visible_work"} or any(v not in CLAIM_GRADES for v in claim.values()):
                raise BenchmarkError("claim_review_incomplete")
            totals = result[variant]
            totals["complete"] += record["status"] == "complete"
            totals["joint_supported_correct"] += (record["status"] == "complete" and grades["project_correct"]
                                                   and grades["task_correct"] and grades["evidence_supported"])
            totals["evidence_errors"] += not grades["evidence_supported"]
            totals["uncertainty_errors"] += not grades["uncertainty_appropriate"]
            totals["privacy_leaks"] += grades["privacy_leak"]
            totals["unsupported_completion"] += grades["unsupported_completion"]
            totals["unsupported_claims"] += sum(v == "unsupported" for v in claim.values())
    return {v: dict(totals) for v, totals in result.items()}


def freeze_trial(config, root, *, reviewed_test_sha256):
    """Root freezes held-out review and the exact candidate for shadow execution."""
    root = Path(root)
    fd = open_private_file(root / "benchmark/benchmark.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = read_state(root / "benchmark/frozen-candidate.json")
        if not frozen or fingerprint({k: v for k, v in frozen.items() if k != "frozen_candidate_sha256"}) != frozen["frozen_candidate_sha256"]:
            raise BenchmarkError("invalid_candidate_freeze")
        spec = experiment_spec(config, root)
        if read_state(root / "tuning/runtime-decision.json").get("runtime_decision_sha256") != frozen.get("runtime_decision_sha256"):
            raise BenchmarkError("runtime_decision_changed")
        if frozen["specification"] != spec or frozen["experiment_sha256"] != fingerprint(spec):
            raise BenchmarkError("frozen_configuration_mismatch")
        validated = root / validation_location(frozen)
        for path, key in (("results.json", "validation_results_sha256"),
                          ("blind-assessment.json", "reviewed_assessment_sha256")):
            if file_digest(validated / path, private=True) != frozen[key]:
                raise BenchmarkError("frozen_review_changed")
        validation = read_state(validated / "results.json")
        validation_grades = read_state(validated / "blind-assessment.json")
        test = read_state(root / "benchmark/test/results.json")
        test_grades = read_state(root / "benchmark/test/blind-assessment.json")
        if (test.get("experiment_sha256") != fingerprint(spec) or test_grades.get("experiment_sha256") != fingerprint(spec)
                or file_digest(root / "benchmark/test/blind-assessment.json", private=True) != reviewed_test_sha256):
            raise BenchmarkError("held_out_review_changed")
        pilot = {"validation": semantic_totals(validation, validation_grades, config["exports"]["validation"]["count"]),
                 "test": semantic_totals(test, test_grades, config["exports"]["test"]["count"])}
        candidate = frozen["candidate"]
        # Never reinterpret the test to select another model or checkpoint.
        if candidate not in VARIANTS:
            raise BenchmarkError("invalid_frozen_candidate")
        successful = pilot["validation"][candidate]["complete"]
        wall = session_totals(validation)["conservative_total_seconds"]
        if not successful or wall is None or wall <= 0:
            raise BenchmarkError("validation_runtime_unmeasured")
        all_attempts = sum(r.get("elapsed_seconds", 0) for variants in validation["results"].values() for r in variants.values())
        startups = [r for r in validation.get("sessions", []) if r.get("stage") == "startup"]
        startup_total = sum(r["elapsed_seconds"] for r in startups)
        candidate_cost = sum(variants[candidate].get("elapsed_seconds", 0) for variants in validation["results"].values())
        candidate_cost += sum(r["elapsed_seconds"] for r in startups if r.get("variant") == candidate)
        candidate_cost += max(0, wall - all_attempts - startup_total)
        if candidate_cost <= 0:
            raise BenchmarkError("candidate_runtime_unmeasured")
        quota = min(FRAME_CEILING, max(1, math.floor(.8 * 5400 * successful / candidate_cost)))
        body = {"version": VERSION, "candidate": candidate, "specification": spec,
                "frozen_candidate_sha256": frozen["frozen_candidate_sha256"],
                "reviewed_test_sha256": reviewed_test_sha256,
                "test_results_sha256": file_digest(root / "benchmark/test/results.json", private=True),
                "pilot_semantic_scores": pilot, "per_night_frame_limit": quota,
                "quota_scope": "candidate_attempts_startup_and_all_unattributed_overhead;revise_only_in_new_trial",
                "required_nights": NIGHTS, "policy": "shadow_only_no_production_promotion"}
        body["trial_sha256"] = fingerprint(body)
        existing = read_state(root / "trial/frozen-trial.json")
        if existing and existing != body:
            raise BenchmarkError("trial_already_frozen")
        write_json(root / "trial/frozen-trial.json", body)
        return {"status": "trial_frozen", "candidate": candidate, "trial_sha256": body["trial_sha256"],
                "per_night_frame_limit": quota, "production_changed": False}
    finally:
        os.close(fd)


def verify_trial(config, root):
    frozen = read_state(root / "trial/frozen-trial.json")
    body = {k: v for k, v in frozen.items() if k != "trial_sha256"}
    if not frozen or fingerprint(body) != frozen["trial_sha256"] or frozen["specification"] != experiment_spec(config, root):
        raise BenchmarkError("trial_identity_changed")
    candidate = read_state(root / "benchmark/frozen-candidate.json")
    if (fingerprint({k: v for k, v in candidate.items() if k != "frozen_candidate_sha256"}) != frozen["frozen_candidate_sha256"]
            or candidate.get("frozen_candidate_sha256") != frozen["frozen_candidate_sha256"]
            or candidate.get("candidate") != frozen["candidate"]):
        raise BenchmarkError("trial_candidate_changed")
    if read_state(root / "tuning/runtime-decision.json").get("runtime_decision_sha256") != candidate.get("runtime_decision_sha256"):
        raise BenchmarkError("runtime_decision_changed")
    validated = root / validation_location(candidate)
    for path, expected in ((root / "benchmark/test/results.json", frozen["test_results_sha256"]),
                           (root / "benchmark/test/blind-assessment.json", frozen["reviewed_test_sha256"]),
                           (validated / "results.json", candidate["validation_results_sha256"]),
                           (validated / "blind-assessment.json", candidate["reviewed_assessment_sha256"])):
        if file_digest(path, private=True) != expected:
            raise BenchmarkError("trial_review_changed")
    return frozen


def coverage(cohort, data):
    rows, results = cohort["selected"], data.get("results", {})
    complete = [r for r in rows if results.get(r["id"], {}).get("status") == "complete"]
    claimed = [r for r in complete if (results[r["id"]].get("activity") or {}).get("uncertainty") == "supported"
               and (results[r["id"]]["activity"].get("activity_kind") != "unclear"
                    or results[r["id"]]["activity"].get("project_candidate")
                    or results[r["id"]]["activity"].get("task_candidate"))]
    valid_runs = [(at(lo), at(hi)) for r in complete for lo, hi in r["support_runs"]]
    claimed_runs = [(at(lo), at(hi)) for r in claimed for lo, hi in r["support_runs"]]
    eligible_bins, occupied_bins = set(cohort["eligible_bins"]), set(cohort["occupied_foreground_bins"])
    completed_bins = {r["bin"] for r in complete}
    return {"source_day": cohort["source_day"], "raw_captures": cohort["raw_captures"],
            "eligible_captures": cohort["eligible_captures"], "excluded": cohort["excluded"],
            "selected": len(rows), "completed": len(complete),
            "failed": sum(r.get("status") not in {"complete", "running"} for r in results.values()),
            "pending": len(rows) - len(results) + sum(r.get("status") == "running" for r in results.values()),
            "raw_safety_available": sum(r.get("raw_safety_available") is True for r in results.values()),
            "raw_safety_unavailable": sum(r.get("raw_safety_available") is not True for r in results.values()),
            "raw_privacy_flags": sum((r.get("raw_safety") or {}).get("privacy") is True for r in results.values()),
            "raw_completion_flags": sum((r.get("raw_safety") or {}).get("completion") is True for r in results.values()),
            "failure_codes": dict(Counter(r["status"] for r in results.values()
                                           if r.get("status") not in {"complete", "running"})),
            "occupied_foreground_bins": len(occupied_bins),
            "eligible_bins": len(eligible_bins), "completed_bins": len(completed_bins),
            "eligible_bin_coverage": len(completed_bins & eligible_bins) / len(eligible_bins) if eligible_bins else None,
            "observed_bin_coverage": len(completed_bins & occupied_bins) / len(occupied_bins) if occupied_bins else None,
            "foreground_seconds": cohort["foreground_seconds"],
            "valid_capture_support_seconds": round(union_seconds(valid_runs), 6),
            "model_claimed_supported_seconds": round(union_seconds(claimed_runs), 6),
            "completed_unique_episodes": len({r["episode_id"] for r in complete}),
            "foreground_episode_count": cohort["foreground_episode_count"],
            "eligible_episode_count": cohort["eligible_episode_count"],
            "semantic_accuracy": None, "accurately_described_episodes": None,
            "coverage_scope": "structural_validity_and_observed_support;_new_frames_have_no_semantic_reference",
            "production_changed": False, **session_totals(data)}


def trial_ready(summaries):
    qualified = [s for s in summaries if s.get("cohort_complete") and s.get("completed", 0) > 0]
    return (len({s["source_day"] for s in qualified}) >= NIGHTS
            and len({n for s in qualified for n in s.get("execution_nights", [])}) >= NIGHTS)


def verify_cohort(cohort, frozen):
    if (cohort.get("trial_sha256") != frozen["trial_sha256"]
            or fingerprint({k: v for k, v in cohort.items() if k != "cohort_sha256"}) != cohort.get("cohort_sha256")):
        raise BenchmarkError("trial_cohort_changed")


def verified_summary(root, day, frozen):
    """Recompute completion from bound records, never trust cached summaries."""
    folder = root / "trial" / day
    cohort, data = read_state(folder / "cohort.json"), read_state(folder / "results.json")
    if data and data.get("trial_sha256") != frozen["trial_sha256"]:
        raise BenchmarkError("trial_resume_changed")
    if not cohort:
        if data.get("results") or data.get("cohort_sha256"):
            raise BenchmarkError("trial_preparation_changed")
        return {"source_day": day, "cohort_complete": False, "completed": 0, "execution_nights": [], **session_totals(data)}
    verify_cohort(cohort, frozen)
    if (cohort["source_day"] != day or data.get("cohort_sha256") not in {None, cohort["cohort_sha256"]}
            or (data.get("cohort_sha256") is None and data.get("results"))):
        raise BenchmarkError("trial_resume_changed")
    expected = {r["id"]: fingerprint({"cohort": cohort["cohort_sha256"], "frame": r["id"], "trial": frozen["trial_sha256"]})
                for r in cohort["selected"]}
    for key, record in data.get("results", {}).items():
        if key not in expected or record.get("identity") != expected[key]:
            raise BenchmarkError("trial_frame_identity_changed")
        if record.get("status") == "complete":
            try:
                night = date.fromisoformat(record["execution_night"])
            except (KeyError, ValueError, TypeError):
                raise BenchmarkError("trial_execution_night_changed") from None
            if night <= date.fromisoformat(day) or night.isoformat() != record["execution_night"]:
                raise BenchmarkError("trial_execution_night_changed")
    summary = coverage(cohort, data)
    summary.update(trial_sha256=frozen["trial_sha256"], cohort_sha256=cohort["cohort_sha256"],
                   cohort_complete=not summary["pending"], execution_nights=sorted({r["execution_night"]
                       for r in data["results"].values() if r["status"] == "complete"}))
    return summary


def run_trial(config, root, budget):
    """Run one immutable previous-day cohort; serial backends own the model lock."""
    root, started = Path(root), time.monotonic()
    guard = Guard(budget)
    try:
        guard.check(35)  # Do not read activity or write a cohort during daytime.
    except BudgetEnded as error:
        return {"status": "partial", "stop_reason": str(error), "private_examples_opened": False}
    fd = open_private_file(root / "trial/trial.lock")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "partial", "stop_reason": "trial_busy"}
        frozen = verify_trial(config, root)
        today = datetime.now(ZONE).date()
        index = read_state(root / "trial/index.json") or {"trial_sha256": frozen["trial_sha256"], "cohorts": []}
        if index["trial_sha256"] != frozen["trial_sha256"]:
            raise BenchmarkError("trial_index_changed")
        if (not isinstance(index["cohorts"], list) or len(index["cohorts"]) > 14
                or len(set(index["cohorts"])) != len(index["cohorts"])):
            raise BenchmarkError("invalid_trial_cohort_index")
        for value in index["cohorts"]:
            try:
                parsed = date.fromisoformat(value)
            except (ValueError, TypeError):
                raise BenchmarkError("invalid_trial_cohort_day") from None
            if parsed.isoformat() != value or parsed >= today:
                raise BenchmarkError("invalid_trial_cohort_day")
        summaries = [verified_summary(root, d, frozen) for d in index["cohorts"]]
        if trial_ready(summaries):
            return {"status": "trial_complete", "trial_sha256": frozen["trial_sha256"], "cohorts_completed": sum(s.get("cohort_complete", False) for s in summaries),
                    "production_changed": False}
        pending = next((d for d, s in zip(index["cohorts"], summaries) if not s.get("cohort_complete")), None)
        source_day = date.fromisoformat(pending) if pending else today - timedelta(days=1)
        if source_day.isoformat() in index["cohorts"] and not pending:
            return {"status": "partial", "stop_reason": "awaiting_next_distinct_night"}
        folder = root / "trial" / source_day.isoformat()
        cohort = read_state(folder / "cohort.json")
        path = folder / "results.json"
        data = read_state(path) or {"trial_sha256": frozen["trial_sha256"], "cohort_sha256": None,
                                   "results": {}, "sessions": []}
        for old in data["sessions"]:
            if old["status"] == "running":
                old["status"] = "interrupted"
        for record in data["results"].values():
            if record["status"] == "running":
                record.update(status="interrupted", activity=None, description=None)
        session = {"stage": "session", "status": "running", "elapsed_seconds": None,
                   "reserved_seconds": budget + 30, "execution_night": today.isoformat(),
                   "started_at_utc": datetime.now(timezone.utc).isoformat()}
        data["sessions"].append(session)
        if source_day.isoformat() not in index["cohorts"]:
            index["cohorts"].append(source_day.isoformat())
            write_json(root / "trial/index.json", index)
        write_json(path, data)  # Indexed reservation BEFORE opening/preparing activity.
        if not cohort:
            try:
                cohort = load_cohort(source_day, frozen["per_night_frame_limit"], guard)
            except BudgetEnded as error:
                session.update(status="finished", elapsed_seconds=round(time.monotonic() - started, 3), stop_reason=str(error))
                write_json(path, data)
                return {"status": "partial", "stop_reason": str(error), "preparation_complete": False, **session_totals(data)}
            cohort["trial_sha256"] = frozen["trial_sha256"]
            cohort["cohort_sha256"] = fingerprint({k: v for k, v in cohort.items() if k != "cohort_sha256"})
            write_json(folder / "cohort.json", cohort)
        verify_cohort(cohort, frozen)
        if data["cohort_sha256"] not in {None, cohort["cohort_sha256"]}:
            raise BenchmarkError("trial_resume_changed")
        data["cohort_sha256"] = cohort["cohort_sha256"]
        write_json(path, data)
        runner, stop = Runner(frozen["specification"], guard), None
        variant = frozen["candidate"]
        pending_rows = [r for r in cohort["selected"] if r["id"] not in data["results"]]
        batch_size = frozen["specification"]["parameters"]["batch_requests"]
        try:
            for offset in range(0, len(pending_rows), batch_size):
                guard.check(35)
                try:
                    runner.start(variant)
                except Exception as error:
                    if pre_inference_refusal(error):
                        raise BudgetEnded(str(error)) from None
                    raise
                for row in pending_rows[offset:offset + batch_size]:
                    guard.check(35)
                    identity = fingerprint({"cohort": cohort["cohort_sha256"], "frame": row["id"], "trial": frozen["trial_sha256"]})
                    data["results"][row["id"]] = {"status": "running", "identity": identity}
                    write_json(path, data)
                    try:
                        record = response_record(runner.infer(row, identity), row["context"])
                        if record["status"] == "complete" and not record["raw_safety_available"]:
                            record.update(status="raw_safety_unavailable", activity=None, description=None)
                        if any(record["final_safety_proxy"].values()):
                            record.update(status="final_safety_blocked", activity=None, description=None)
                        record_result(STATE_DIR, record.get("telemetry_attempt_id"), record["status"])
                    except (BudgetEnded, ModelBusy):
                        data["results"].pop(row["id"])
                        write_json(path, data)
                        raise
                    except Exception as error:
                        if pre_inference_refusal(error):
                            data["results"].pop(row["id"])
                            write_json(path, data)
                            raise BudgetEnded("trial_budget" if str(error) == "insufficient_window_for_request" else str(error)) from None
                        record = {"status": str(error) if isinstance(error, WorkerError) else "inference_failed", "error_type": type(error).__name__,
                                  "telemetry_attempt_id": getattr(error, "telemetry_attempt_id", None)}
                        if not SAFE_LABEL.fullmatch(record["status"]):
                            record["status"] = "inference_failed"
                        safety = response_record({"status": record["status"], "raw_safety_checks": getattr(error, "raw_safety_checks", {})}, row["context"])
                        record.update({k: safety[k] for k in ("raw_safety", "raw_safety_available", "final_safety_proxy")})
                        runner.close()
                        data["results"][row["id"]] = {**record, "identity": identity}
                        write_json(path, data)
                        raise BudgetEnded("backend_failed_resume_remaining_frames") from None
                    data["results"][row["id"]] = {**record, "identity": identity, "execution_night": today.isoformat()}
                    write_json(path, data)
                runner.close()
        except BudgetEnded as error:
            stop = str(error)
        except ModelBusy:
            stop = "local_model_busy"
        except Exception as error:
            stop = str(error) if isinstance(error, BenchmarkError) else "trial_backend_failed"
        finally:
            runner.close()
        session.update(status="finished", elapsed_seconds=round(time.monotonic() - started, 3), stop_reason=stop)
        write_json(path, data)
        summary = verified_summary(root, source_day.isoformat(), frozen)
        summary.update(status="partial", stop_reason=stop, trial_sha256=frozen["trial_sha256"],
                       elapsed_seconds=session["elapsed_seconds"])
        write_json(folder / "summary.json", summary)
        if trial_ready([verified_summary(root, d, frozen) for d in index["cohorts"]]):
            summary["status"] = "trial_complete"
            write_json(folder / "summary.json", summary)
        return summary
    finally:
        os.close(fd)


def recommendation(root):
    """A review artifact, never a deployment command or inferred promotion."""
    root = Path(root)
    frozen, index = read_state(root / "trial/frozen-trial.json"), read_state(root / "trial/index.json")
    if not frozen or index.get("trial_sha256") != frozen["trial_sha256"]:
        raise BenchmarkError("trial_not_frozen")
    summaries = [verified_summary(root, d, frozen) for d in index["cohorts"]]
    qualified = [s for s in summaries if s.get("cohort_complete") and s.get("completed", 0) > 0]
    source_days = {s["source_day"] for s in qualified}
    execution_nights = {n for s in qualified for n in s.get("execution_nights", [])}
    complete = len(source_days) >= NIGHTS and len(execution_nights) >= NIGHTS
    conservative = [s.get("conservative_total_seconds") for s in summaries]
    wall = sum(conservative) if conservative and all(v is not None for v in conservative) else None
    valid = sum(s.get("completed", 0) for s in summaries)
    capacity = math.floor(.8 * 23400 * valid / wall) if wall and complete else None
    body = {"version": VERSION, "trial_sha256": frozen["trial_sha256"], "candidate": frozen["candidate"],
            "status": "awaiting_coordinator_recommendation" if complete else "trial_incomplete",
            "source_days": sorted(source_days), "execution_nights": sorted(execution_nights), "nights": summaries,
            "pilot_semantic_scores": frozen["pilot_semantic_scores"], "daily_semantic_accuracy": None,
            "accurate_episodes_per_night": None, "estimated_valid_frames_per_6_5h_with_20_percent_reserve": capacity,
            "capacity_scope": "conservative_runtime_estimate;valid_frames_not_verified_activity_episodes;three_day_pilot",
            "decision": "requires_root_quality_coverage_resource_review", "production_changed": False}
    write_json(root / "trial/recommendation.json", body)
    return body
