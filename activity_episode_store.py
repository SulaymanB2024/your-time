"""Fingerprint-bound hypotheses and exact, conservative observed-time allocation.

No capture, database, model, or project discovery occurs here. Callers supply
observed runs, evidence snapshots, identities and explicitly reviewed policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from activity_context import validate
from private_io import write_json

VERSION = "activity_episode_inference_v1"
ALLOCATION_VERSION = "activity_episode_allocation_v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_KEYS = ("screenshot_sha256", "input_sha256", "context_sha256")
RUNTIME_KEYS = ("engine_sha256", "prompt_sha256", "model_sha256", "adapter_sha256")
STATES = {"active", "unattributed", "idle", "locked"}
MAX_SUPPORT_SECONDS = 60
MAX_RECORD_BYTES = 128 * 1024


def fingerprint(value: object) -> str:
    """Hash canonical JSON; the full context includes its existing input hash."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _at(value: str | datetime) -> datetime:
    at = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("Times must be timezone-aware")
    return at.astimezone(timezone.utc)


def _seconds(start: datetime, end: datetime) -> Decimal:
    delta = end - start
    return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1_000_000


def _digest(value, *, nullable=False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Expected a full SHA-256 fingerprint")


def make_fingerprints(context: dict, *, screenshot_sha256: str, input_sha256: str,
                      engine_sha256: str, prompt_sha256: str, model_sha256: str,
                      adapter_sha256: str | None = None) -> dict:
    result = dict(screenshot_sha256=screenshot_sha256, input_sha256=input_sha256,
                  context_sha256=fingerprint(context), engine_sha256=engine_sha256,
                  prompt_sha256=prompt_sha256, model_sha256=model_sha256,
                  adapter_sha256=adapter_sha256)
    _check_fingerprints(result)
    return result


def _check_fingerprints(value: dict) -> None:
    if not isinstance(value, dict) or set(value) != set(SOURCE_KEYS + RUNTIME_KEYS):
        raise ValueError("Missing or extra source/runtime fingerprints")
    for key in SOURCE_KEYS + RUNTIME_KEYS:
        _digest(value[key], nullable=key == "adapter_sha256")


def _observed_runs(samples: list[dict]) -> list[dict]:
    runs = []
    for sample in samples:
        if sample.get("state") not in STATES:
            raise ValueError("Unknown observed state")
        start, end = _at(sample["start_utc"]), _at(sample["end_utc"])
        if start >= end:
            raise ValueError("Observed intervals must have positive duration")
        seconds = sample.get("sampled_seconds", float(_seconds(start, end)))
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("Invalid sampled seconds")
        expected = Decimal(str(seconds))
        supports = sample.get("support_runs", [{"start_utc": start, "end_utc": end}])
        if not isinstance(supports, list) or not supports:
            raise ValueError("Exact support runs required")
        identity = sample.get("id") or "sample-" + fingerprint(sample)
        if not isinstance(identity, str):
            raise ValueError("Invalid sample identity")
        amount = Decimal(0)
        for support in supports:
            lower, upper = _at(support["start_utc"]), _at(support["end_utc"])
            if not start <= lower < upper <= end:
                raise ValueError("Support run falls outside its sample")
            amount += _seconds(lower, upper)
            runs.append({"sample_id": identity, "start_utc": lower.isoformat(),
                         "end_utc": upper.isoformat(), "state": sample["state"],
                         "app": sample.get("app"), "window": sample.get("window"),
                         "context_key": sample.get("context_key"),
                         "source": sample.get("evidence", sample.get("source")),
                         "support_barrier": bool(sample.get("support_barrier"))})
        if amount != expected:
            raise ValueError("Sparse samples require exact support runs; never smear time across gaps")
    runs.sort(key=lambda item: (item["start_utc"], item["end_utc"], item["sample_id"]))
    for previous, current in zip(runs, runs[1:]):
        if _at(current["start_utc"]) < _at(previous["end_utc"]):
            raise ValueError("Overlapping observed runs must be resolved by the deterministic source layer")
    return runs


def _episode_payload(episode: dict) -> dict:
    return {key: episode[key] for key in ("start_utc", "end_utc", "state", "samples")}


def build_episodes(samples: list[dict], *, context_boundaries=(), max_seconds=900,
                   max_gap_seconds=0) -> list[dict]:
    """Group matching context, preserving exact disjoint observed support.

    Explicit context boundaries split reused titles; they do not carry labels.
    Optional polling jitter tolerance groups runs across at most two seconds,
    without counting the missing interval or allowing an anchor inside it.
    Sparse merged segments need exact support_runs, otherwise attribution fails
    closed instead of crediting a fabricated continuous interval.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= 900:
        raise ValueError("Episode bound must be 1-900 seconds")
    if (isinstance(max_gap_seconds, bool) or not isinstance(max_gap_seconds, (int, float))
            or not math.isfinite(max_gap_seconds) or not 0 <= max_gap_seconds <= 2):
        raise ValueError("Polling jitter tolerance must be 0-2 seconds")
    boundaries = sorted({_at(value) for value in context_boundaries})
    episodes = []
    for run in _observed_runs(samples):
        start, end = _at(run["start_utc"]), _at(run["end_utc"])
        cuts = [start] + [at for at in boundaries if start < at < end] + [end]
        for lower, upper in zip(cuts, cuts[1:]):
            cursor = lower
            while cursor < upper:
                previous = episodes[-1] if episodes else None
                key = tuple(run.get(k) for k in ("state", "app", "window", "context_key", "source"))
                prior_end = _at(previous["end_utc"]) if previous else cursor
                can_extend = (previous and previous["key"] == key
                              and 0 <= _seconds(prior_end, cursor) <= Decimal(str(max_gap_seconds))
                              and not any(prior_end <= at <= cursor for at in boundaries)
                              and not run["support_barrier"] and not previous["barrier"]
                              and _seconds(_at(previous["start_utc"]), cursor) < max_seconds)
                if not can_extend:
                    previous = {"key": key, "barrier": run["support_barrier"],
                                "start_utc": cursor.isoformat(), "end_utc": cursor.isoformat(),
                                "state": run["state"], "samples": []}
                    episodes.append(previous)
                finish = min(upper, _at(previous["start_utc"]) + timedelta(seconds=max_seconds))
                previous["samples"].append({**run, "start_utc": cursor.isoformat(), "end_utc": finish.isoformat()})
                previous["end_utc"] = finish.isoformat()
                cursor = finish
    for episode in episodes:
        episode.pop("key")
        episode.pop("barrier")
        episode["episode_sha256"] = fingerprint(_episode_payload(episode))
        episode["id"] = "episode-" + episode["episode_sha256"]
        episode["sampled_seconds"] = float(sum((_seconds(_at(r["start_utc"]), _at(r["end_utc"]))
                                                 for r in episode["samples"]), Decimal(0)))
    return episodes


def _check_episode(episode: dict) -> None:
    digest = fingerprint(_episode_payload(episode))
    if episode.get("episode_sha256") != digest or episode.get("id") != "episode-" + digest:
        raise ValueError("Episode fingerprint mismatch")


def supports_anchor(episode: dict, timestamp: str | datetime) -> bool:
    """Visibility requires an actual observed run, including grouped episodes."""
    at = _at(timestamp)
    return any(_at(run["start_utc"]) <= at < _at(run["end_utc"])
               for run in episode["samples"])


def _resolve_identity(candidate, claim_ids: list[str], observation_id: str, bindings,
                      *, project_id=None, task=False) -> dict | None:
    if candidate is None:
        return None
    matches = []
    normalized = " ".join(candidate.split()).casefold()
    for binding in bindings:
        # A background file event is never sufficient to bind the visible work.
        if (observation_id not in binding.get("evidence_ids", [])
                or observation_id not in claim_ids or not binding.get("id")
                or (task and (not project_id or binding.get("project_id") != project_id))):
            continue
        aliases = binding.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise ValueError("Invalid identity aliases")
        if normalized in {" ".join(alias.split()).casefold() for alias in aliases}:
            matches.append({"id": binding["id"], "aliases": sorted(set(aliases)),
                            "evidence_ids": [observation_id],
                            "project_id": binding.get("project_id") if task else binding["id"]})
    identities = {item["id"] for item in matches}
    if len(identities) != 1:
        return None
    return sorted(matches, key=fingerprint)[0]


def make_inference(*, episode: dict, observation_id: str, timestamp_utc: str,
                   context: dict, result: dict, fingerprints: dict,
                   project_bindings=(), task_bindings=(), radius_seconds=30) -> dict:
    """Prepare an immutable hypothesis; no semantic entailment is implied."""
    _check_episode(episode)
    _check_fingerprints(fingerprints)
    if fingerprints["context_sha256"] != fingerprint(context):
        raise ValueError("Context fingerprint mismatch")
    if type(radius_seconds) is not int or not 1 <= radius_seconds <= MAX_SUPPORT_SECONDS // 2:
        raise ValueError("Observation support radius must be 1-30 seconds")
    at = _at(timestamp_utc)
    if (episode["state"] not in {"active", "unattributed"}
            or not supports_anchor(episode, at)
            or any(run["support_barrier"] for run in episode["samples"])):
        raise ValueError("Observation cannot support this episode")
    value = json.loads(json.dumps(result, allow_nan=False))
    validate(value, context)
    primary = [item for item in context["evidence"]
               if item["source"] in {"screen_context", "synthetic_screen", "withheld_observation"}]
    if (len(primary) != 1 or primary[0]["id"] != observation_id
            or _at(primary[0]["timestamp_utc"]) != at):
        raise ValueError("Inference must anchor to its current observation")
    project = _resolve_identity(value["project_candidate"], value["claim_evidence"]["project_candidate"],
                                observation_id, project_bindings)
    task = _resolve_identity(value["task_candidate"], value["claim_evidence"]["task_candidate"],
                             observation_id, task_bindings,
                             project_id=project["id"] if project else None, task=True)
    record = {"version": VERSION, "evidence_tier": "model_hypothesis",
              "episode_id": episode["id"], "episode_sha256": episode["episode_sha256"],
              "observation_id": observation_id, "timestamp_utc": at.isoformat(),
              "scope_start_utc": max(_at(episode["start_utc"]), at - timedelta(seconds=radius_seconds)).isoformat(),
              "scope_end_utc": min(_at(episode["end_utc"]), at + timedelta(seconds=radius_seconds)).isoformat(),
              "fingerprints": dict(fingerprints), "result": value,
              "context_evidence": [{"id": item["id"], "source": item["source"],
                                    "source_sha256": fingerprint(item)} for item in context["evidence"]],
              "project_identity": project, "task_identity": task,
              "confirmed_outcomes": [],
              "limits": "Citations are not entailment proof; screen visibility does not establish attention or completion."}
    record["id"] = "inference-" + fingerprint(record)
    return record


def persist_inference(store_dir: Path, **kwargs) -> dict:
    """Write one immutable record to an explicitly supplied private directory."""
    record = make_inference(**kwargs)
    if len(json.dumps(record, ensure_ascii=False).encode()) > MAX_RECORD_BYTES:
        raise ValueError("Inference record exceeds the store bound")
    write_json(Path(store_dir) / (record["id"] + ".json"), record)
    return record


def is_fresh(record: dict, *, episodes: list[dict], current_evidence: dict,
             runtime_fingerprints: dict) -> bool:
    """Reject missing/changed support and changed runtime; never trust a cache alone."""
    try:
        if record["version"] != VERSION or record["evidence_tier"] != "model_hypothesis" or record["confirmed_outcomes"]:
            return False
        payload = {key: value for key, value in record.items() if key != "id"}
        if record["id"] != "inference-" + fingerprint(payload):
            return False
        _check_fingerprints(record["fingerprints"])
        episode = next((item for item in episodes if item["id"] == record["episode_id"]), None)
        if not episode:
            return False
        _check_episode(episode)
        if episode["episode_sha256"] != record["episode_sha256"]:
            return False
        source = current_evidence[record["observation_id"]]
        if _at(source["timestamp_utc"]) != _at(record["timestamp_utc"]):
            return False
        if any(record["fingerprints"][key] != source[key] for key in SOURCE_KEYS):
            return False
        if any(record["fingerprints"][key] != runtime_fingerprints[key] for key in RUNTIME_KEYS):
            return False
        lower, at, upper = _at(record["scope_start_utc"]), _at(record["timestamp_utc"]), _at(record["scope_end_utc"])
        if not _at(episode["start_utc"]) <= lower <= at < upper <= _at(episode["end_utc"]):
            return False
        if (episode["state"] not in {"active", "unattributed"} or not supports_anchor(episode, at)
                or any(run["support_barrier"] for run in episode["samples"])):
            return False
        if _seconds(lower, upper) > MAX_SUPPORT_SECONDS:
            return False
        validate(record["result"], {"evidence": record["context_evidence"]})
        return True
    except (KeyError, TypeError, ValueError, StopIteration):
        return False


def _read_record(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > MAX_RECORD_BYTES):
            raise ValueError("Invalid private inference file")
        value = json.loads(source.read(MAX_RECORD_BYTES + 1))
        if not isinstance(value, dict):
            raise ValueError("Inference file must contain an object")
        return value


def _store_directory(path: Path) -> bool:
    """Read-only directory check; do not create/chmod a store during reads."""
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("Inference directory must be an absolute canonical path")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError("Expected a private owned inference directory")
    return True


def read_fresh_inferences(store_dir: Path, *, episodes: list[dict], current_evidence: dict,
                          runtime_fingerprints: dict) -> list[dict]:
    """Read records only from the caller's store; rejected records are not returned."""
    path = Path(store_dir)
    if not _store_directory(path):
        return []
    output = []
    for entry in sorted(path.glob("inference-*.json")):
        try:
            record = _read_record(entry)
            if entry.stem == record.get("id") and is_fresh(record, episodes=episodes,
                                                          current_evidence=current_evidence,
                                                          runtime_fingerprints=runtime_fingerprints):
                output.append(record)
        except (OSError, TypeError, ValueError):
            continue
    return output


def _model_enabled(promotion: dict | None) -> bool:
    if not promotion or promotion.get("enabled") is not True:
        return False
    nights = promotion.get("trial_nights", [])
    if (not isinstance(promotion.get("semantic_review_id"), str) or not promotion["semantic_review_id"]
            or not isinstance(nights, list) or not all(isinstance(night, str) for night in nights)
            or len(set(nights)) < 3):
        raise ValueError("Model allocation requires explicit semantic review and a three-night trial")
    _digest(promotion.get("trial_receipt_sha256"))
    for night in nights:
        if not isinstance(night, str) or datetime.strptime(night, "%Y-%m-%d").date().isoformat() != night:
            raise ValueError("Trial nights must be distinct ISO local dates")
    return True


def correction_fingerprint(value: dict) -> str:
    """Bind a caller's current correction revision to all supplied scope fields."""
    payload = {key: item for key, item in value.items() if key != "revision_sha256"}
    for key in ("start_utc", "end_utc"):
        payload[key] = _at(payload[key]).isoformat()
    return fingerprint(payload)


def _corrections(values: list[dict]) -> list[dict]:
    output = []
    for value in values:
        if value.get("evidence_tier") != "user_confirmed_label":
            continue
        lower, upper = _at(value["start_utc"]), _at(value["end_utc"])
        if lower >= upper or not isinstance(value.get("id"), str) or not value["id"]:
            raise ValueError("Invalid correction interval")
        if not isinstance(value.get("label"), str) or not value["label"].strip():
            raise ValueError("Invalid correction label")
        if "sample_ids" in value and (not isinstance(value["sample_ids"], list)
                or not all(isinstance(identity, str) and identity for identity in value["sample_ids"])):
            raise ValueError("Invalid correction sample scope")
        _digest(value.get("revision_sha256"))
        if value["revision_sha256"] != correction_fingerprint(value):
            raise ValueError("Correction revision fingerprint mismatch")
        output.append({**value, "start_utc": lower.isoformat(), "end_utc": upper.isoformat()})
    return output


def _choice(users: list[dict], models: list[dict], midpoint: datetime) -> dict:
    if users:
        labels = {(item["label"], item.get("project_id"), item.get("task_id")) for item in users}
        if len(labels) != 1:
            return {"status": "user_correction_conflict", "evidence_tier": "uncertain", "label": None}
        item = min(users, key=lambda value: value["id"])
        return {"status": "user_confirmed_label", "evidence_tier": "user_confirmed_label",
                "label": item["label"], "project_id": item.get("project_id"), "task_id": item.get("task_id"),
                "correction_id": item["id"], "correction_revision_sha256": item["revision_sha256"]}
    if models:
        distance = min(abs((_at(item["timestamp_utc"]) - midpoint).total_seconds()) for item in models)
        nearest = [item for item in models if abs((_at(item["timestamp_utc"]) - midpoint).total_seconds()) == distance]
        def candidate_identity(item, field):
            identity = (item[field + "_identity"] or {}).get("id")
            label = item["result"][field + "_candidate"]
            return ("canonical", identity) if identity else ("unbound", label)
        labels = {(candidate_identity(item, "project"), candidate_identity(item, "task"),
                   item["result"]["activity_kind"]) for item in nearest}
        if len(labels) != 1:
            return {"status": "model_conflict", "evidence_tier": "uncertain", "label": None}
        item = min(nearest, key=lambda value: value["id"])
        return {"status": "model_inference", "evidence_tier": "supported_inference",
                "label": item["result"]["task_candidate"] or item["result"]["project_candidate"] or item["result"]["activity_kind"],
                "project_id": (item["project_identity"] or {}).get("id"),
                "task_id": (item["task_identity"] or {}).get("id"), "inference_id": item["id"],
                "uncertainty": item["result"]["uncertainty"]}
    return {"status": "unclassified", "evidence_tier": "observed_context_only", "label": None}


def allocate(episodes: list[dict], inferences=(), corrections=(), *, current_evidence: dict,
             runtime_fingerprints: dict, reviewed_record_ids=(), promotion=None) -> dict:
    """Allocate each observed foreground second once; never allocate idle or gaps.

    Precedence: current user correction on unattributed support, then fresh
    semantically reviewed model hypothesis under explicit promotion, then unknown.
    Promotions are caller-verified receipts, never inferred from their presence.
    """
    enabled = _model_enabled(promotion)
    inferences = list(inferences)
    reviewed = set(reviewed_record_ids)
    fresh = [item for item in inferences if is_fresh(item, episodes=episodes,
              current_evidence=current_evidence, runtime_fingerprints=runtime_fingerprints)]
    models = [item for item in fresh if enabled and item["id"] in reviewed
              and item["result"]["uncertainty"] == "supported"
              and (item["result"]["task_candidate"] or item["result"]["project_candidate"]
                   or item["result"]["activity_kind"] != "unclear")]
    users = _corrections(list(corrections))
    # Revalidate exact support and disjointness even for caller-modified episodes.
    all_runs = []
    for episode in episodes:
        _check_episode(episode)
        for run in episode["samples"]:
            all_runs.append({**run, "id": run["sample_id"], "episode_id": episode["id"]})
    _observed_runs(all_runs)
    rows = []
    counts = Counter()
    for episode in episodes:
        for run in episode["samples"]:
            start, end = _at(run["start_utc"]), _at(run["end_utc"])
            candidates = [item for item in models if item["episode_id"] == episode["id"]
                          and _at(item["scope_start_utc"]) < end and _at(item["scope_end_utc"]) > start]
            relevant_users = [item for item in users if run["state"] == "unattributed"
                              and ("sample_ids" not in item or run["sample_id"] in item["sample_ids"])
                              and _at(item["start_utc"]) < end and _at(item["end_utc"]) > start]
            cuts = {start, end}
            for item in relevant_users + candidates:
                keys = ("start_utc", "end_utc") if "evidence_tier" in item and item["evidence_tier"] == "user_confirmed_label" else ("scope_start_utc", "scope_end_utc")
                cuts.update(_at(item[key]) for key in keys if start < _at(item[key]) < end)
            # Nearest observation ownership can switch midway through overlapping supports.
            for first in candidates:
                for second in candidates:
                    middle = _at(first["timestamp_utc"]) + (_at(second["timestamp_utc"]) - _at(first["timestamp_utc"])) / 2
                    if start < middle < end:
                        cuts.add(middle)
            ordered = sorted(cuts)
            for lower, upper in zip(ordered, ordered[1:]):
                midpoint = lower + (upper - lower) / 2
                supported_users = [item for item in relevant_users if _at(item["start_utc"]) <= midpoint < _at(item["end_utc"])]
                supported_models = [item for item in candidates if _at(item["scope_start_utc"]) <= midpoint < _at(item["scope_end_utc"])]
                attribution = (_choice(supported_users, supported_models, midpoint)
                               if run["state"] in {"active", "unattributed"} and not run["support_barrier"]
                               else {"status": "support_barrier" if run["support_barrier"] else "nonforeground",
                                     "evidence_tier": "observed_context_only", "label": None})
                seconds = _seconds(lower, upper)
                counts["observed_sampled_seconds"] += seconds
                if run["state"] in {"active", "unattributed"}:
                    counts["foreground_seconds"] += seconds
                    counts["allocated_foreground_seconds" if attribution["label"] else "unknown_foreground_seconds"] += seconds
                else:
                    counts["nonforeground_seconds"] += seconds
                rows.append({"episode_id": episode["id"], "sample_id": run["sample_id"], "state": run["state"],
                             "start_utc": lower.isoformat(), "end_utc": upper.isoformat(),
                             "observed_seconds": float(seconds), "attribution": attribution})
    names = ("observed_sampled_seconds", "foreground_seconds", "allocated_foreground_seconds", "unknown_foreground_seconds", "nonforeground_seconds")
    return {"version": ALLOCATION_VERSION, "intervals": sorted(rows, key=lambda item: item["start_utc"]),
            "totals": {name: float(counts[name]) for name in names},
            "fresh_inferences": len(fresh), "rejected_inferences": len(inferences) - len(fresh),
            "model_allocation_enabled": enabled, "confirmed_outcomes": [],
            "interpretation": "Observed foreground context is not attention; reviewed model labels remain inference."}


def status(store_dir: Path) -> dict:
    """Metadata-only inventory: never deserialize result or source content."""
    path = Path(store_dir)
    exists = _store_directory(path)
    count = size = invalid = 0
    for entry in path.glob("inference-*.json") if exists else ():
        info = entry.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            invalid += 1
            continue
        count += 1
        size += info.st_size
    return {"version": VERSION, "records": count, "bytes": size, "invalid_entries": invalid,
            "freshness": "not_checked", "promotion": "not_assessed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status",))
    parser.add_argument("--store-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(status(args.store_dir), sort_keys=True))


if __name__ == "__main__":
    main()
