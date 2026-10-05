"""Propagate checked screen topics only across near-identical local frames."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, time as clock_time, timezone

from daily_analysis import ANALYSIS_DIR, ZONE, private_write
from index_features import REVISION, distance
from secure_store import DB_PATH
from vision_batch import SENSITIVE_RE

MAX_DISTANCE = 0.10
MAX_GAP_SECONDS = 120


def display_id(path: str) -> str:
    match = re.search(r"-display-(\d+)\.webp$", path)
    return match.group(1) if match else "unknown"


def screenshot_id(path: str) -> str:
    import hashlib
    return hashlib.sha256(path.encode()).hexdigest()[:20]


def feature_rows(day: date) -> list[dict]:
    start = datetime.combine(day, clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT s.path,s.timestamp_utc,s.active_app,s.active_window,s.ocr_text,"
            "s.ocr_status,f.vector,f.feature_revision "
            "FROM screenshots s LEFT JOIN screen_features f ON f.path=s.path "
            "WHERE s.timestamp_utc>=? AND s.timestamp_utc<? "
            "ORDER BY s.timestamp_utc",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        db.close()
    return [{"id": screenshot_id(path), "at": datetime.fromisoformat(timestamp),
             "app": app, "display": display_id(path), "vector": vector,
             "eligible": (not title and status == "complete" and vector is not None
                          and revision == REVISION
                          and not SENSITIVE_RE.search(" ".join((app or "", ocr or ""))))}
            for path, timestamp, app, title, ocr, status, vector, revision in rows]


def propagate(rows: list[dict], checked_tags: list[dict], threshold: float = MAX_DISTANCE) -> dict:
    """Link adjacent frames within two minutes, stopping at app/context changes."""
    checked = {item["id"]: item["topic"] for item in checked_tags
               if item.get("status") == "model_inference" and item.get("topic")}
    groups = defaultdict(list)
    for row in rows:
        groups[row["display"]].append(row)
    clusters = []
    for group in groups.values():
        current = []
        for row in sorted(group, key=lambda item: item["at"]):
            if not row.get("eligible", True):
                if current:
                    clusters.append(current)
                    current = []
                continue
            if current:
                first, previous = current[0], current[-1]
                near = (row["app"] == previous["app"]
                        and (row["at"] - first["at"]).total_seconds() <= MAX_GAP_SECONDS
                        and distance(row["vector"], first["vector"]) <= threshold)
                if not near:
                    clusters.append(current)
                    current = []
            current.append(row)
        if current:
            clusters.append(current)
    propagated = []
    conflicts = 0
    supported_clusters = 0
    for cluster in clusters:
        labels = {checked[item["id"]] for item in cluster if item["id"] in checked}
        if len(labels) > 1:
            conflicts += 1
            continue
        if len(labels) != 1:
            continue
        supported_clusters += 1
        topic = next(iter(labels))
        anchor = next(item["id"] for item in cluster if item["id"] in checked)
        for item in cluster:
            if item["id"] not in checked:
                propagated.append({"id": item["id"], "timestamp_utc": item["at"].isoformat(),
                                   "topic": topic, "status": "featureprint_match",
                                   "app_hint": item["app"],
                                   "anchor_id": anchor})
    return {"candidate_frames": len(rows), "clusters": len(clusters),
            "supported_clusters": supported_clusters, "conflicting_clusters": conflicts,
            "propagated": propagated}


def run_day(day: date) -> dict:
    source = ANALYSIS_DIR / f"screen-context-{day.isoformat()}.json"
    try:
        checked = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        checked = {}
    result = propagate(feature_rows(day), checked.get("tags", []))
    report = {"version": "screen_similarity_v2", "day_local": day.isoformat(),
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "feature_revision": REVISION, "max_distance": MAX_DISTANCE,
              "max_gap_seconds": MAX_GAP_SECONDS, **result}
    path = ANALYSIS_DIR / f"screen-similarity-{day.isoformat()}.json"
    private_write(path, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    return {key: report[key] for key in ("day_local", "candidate_frames", "clusters",
                                          "supported_clusters", "conflicting_clusters")} | {
                                              "propagated_count": len(report["propagated"])}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-ago", type=int, action="append")
    args = parser.parse_args()
    ages = args.days_ago or [0, 1, 2]
    if any(age < 0 or age > 7 for age in ages):
        parser.error("--days-ago must be 0-7")
    for age in ages:
        print(json.dumps(run_day(datetime.now(ZONE).date() - timedelta(days=age)), sort_keys=True))


if __name__ == "__main__":
    main()
