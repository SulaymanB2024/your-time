"""Bridge exact collector time to candidate activity records, without promotion."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from activity_context import metadata_identity
from activity_episode_store import (
    allocate,
    build_episodes,
    correction_fingerprint,
    fingerprint,
    make_fingerprints,
    persist_inference,
    supports_anchor,
)
from daily_analysis import ZONE, load_rows, observed_mac_runs
from secure_store import STATE_DIR
from task_corrections import current_intervals

STORE = STATE_DIR / "activity-episodes"


def episodes_for_day(day: date, *, end: datetime | None = None, rows: list[dict] | None = None) -> list[dict]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
    finish = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=ZONE).astimezone(timezone.utc)
    if end is not None:
        finish = min(finish, end)
    return build_episodes(observed_mac_runs(load_rows(start, finish) if rows is None else rows, start, finish),
                          max_gap_seconds=2)


def current_corrections(day: date) -> list[dict]:
    """Read current labels; their scope remains unattributed collector samples."""
    result = current_intervals(day)
    for row in result:
        row["revision_sha256"] = correction_fingerprint(row)
    return result


def day_activity(day: date, *, end: datetime | None = None, rows: list[dict] | None = None) -> dict:
    episodes = episodes_for_day(day, end=end, rows=rows)
    corrections = current_corrections(day)
    # Candidate outputs are intentionally not read into production time totals.
    # Root must verify semantic quality and three trial nights before a release.
    allocation = allocate(episodes, corrections=corrections,
                          current_evidence={}, runtime_fingerprints={})
    labels = Counter()
    for row in allocation["intervals"]:
        choice = row["attribution"]
        if choice["label"]:
            labels[(choice["label"], choice["evidence_tier"])] += row["observed_seconds"]
    return {"version": "chronicle_activity_v1", "episode_count": len(episodes),
            "totals": allocation["totals"], "model_allocation_enabled": False,
            "tasks": [{"label": label, "evidence_tier": tier, "sampled_seconds": round(seconds, 6)}
                      for (label, tier), seconds in labels.most_common()],
            "dependencies_sha256": fingerprint({"episodes": episodes, "corrections": corrections}),
            "confirmed_outcomes": [],
            "interpretation": "Exact observed time; model candidates await semantic review and trial."}


def persist_result(*, context: dict, result: dict, screenshot_sha256: str,
                   input_sha256: str, engine_sha256: str, prompt_sha256: str,
                   model_sha256: str, adapter_sha256: str | None = None,
                   store_dir: Path = STORE) -> dict:
    primary = [r for r in context["evidence"] if r["source"] == "screen_context"]
    if len(primary) != 1:
        raise ValueError("One current captured observation is required")
    observation = primary[0]
    at = datetime.fromisoformat(observation["timestamp_utc"])
    episodes = episodes_for_day(at.astimezone(ZONE).date())
    episode = next((e for e in episodes if supports_anchor(e, at)), None)
    if episode is None:
        return {"status": "no_observed_foreground_support"}
    # Recorded screen metadata is only a hint. Conflicting collector evidence
    # cannot acquire task time or continuity from that hint.
    if any(metadata_identity(r.get("app"), r.get("window")) != observation.get("metadata_sha256")
           for r in episode["samples"]):
        return {"status": "context_disagrees_with_collector"}
    pins = make_fingerprints(context, screenshot_sha256=screenshot_sha256,
                            input_sha256=input_sha256, engine_sha256=engine_sha256,
                            prompt_sha256=prompt_sha256, model_sha256=model_sha256,
                            adapter_sha256=adapter_sha256)
    record = persist_inference(store_dir, episode=episode, observation_id=observation["id"],
                               timestamp_utc=observation["timestamp_utc"], context=context,
                               result=result, fingerprints=pins)
    return {"status": "candidate_persisted", "record_id": record["id"],
            "episode_id": record["episode_id"], "model_allocation_enabled": False}
