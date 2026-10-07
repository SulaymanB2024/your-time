"""Conservative behavior measures from recorded, identified Mac intervals."""

from datetime import datetime
from statistics import median

from topic_allocation import broad_context
from vision_batch import SENSITIVE_RE
from window_topic_tagging import topic_is_specific, window_id

BUCKETS = (("Under 2m", 120), ("2–5m", 300), ("5–15m", 900),
           ("15–30m", 1800), ("30m+", float("inf")))
GAP_TOLERANCE = 1.0  # Allow polling jitter, never count it as observed time.


def build(segments: list[dict], tags: list[dict]) -> dict:
    """A stretch keeps one supported topic, or one exact observed window.

    Idle/lock/unidentified intervals and collection gaps end a stretch. Model
    topics may join different apps only with the same supported exact label.
    A switch is a change between adjacent identified contexts, not a return
    from inactivity. Duration always sums sampled seconds, not wall time.
    """
    topics = {tag["id"]: tag["topic"] for tag in tags
              if tag.get("status") == "model_inference"
              and topic_is_specific(tag.get("topic", ""))}
    sessions = []
    current = None
    switches = 0
    window_switches = 0
    previous_window = None
    for row in segments:
        if row["state"] != "active" or row["sampled_seconds"] <= 0:
            current = previous_window = None
            continue
        at = datetime.fromisoformat(row["start_utc"])
        until = datetime.fromisoformat(row["end_utc"])
        if until <= at:
            current = previous_window = None
            continue
        app, title = row.get("app"), row.get("window")
        identity = window_id(app, title or "")
        topic = topics.get(identity)
        hidden = bool(title and SENSITIVE_RE.search(title))
        if hidden:
            topic = None
        label = topic or (title[:160] if title and not hidden else
                          broad_context(app, title, row.get("site_host")))
        status = "specific_model" if topic else "observed_window"
        key = (status, topic if topic else identity)
        adjacent = (current is not None and
                    0 <= (at - datetime.fromisoformat(current["end_utc"])).total_seconds() <= GAP_TOLERANCE)
        if adjacent:
            switches += current["key"] != key
            window_switches += previous_window != identity
        if not adjacent or current["key"] != key:
            current = {"key": key, "label": label, "status": status,
                       "start_utc": row["start_utc"], "end_utc": row["end_utc"],
                       "sampled_seconds": 0.0, "apps": []}
            sessions.append(current)
        current["end_utc"] = row["end_utc"]
        current["sampled_seconds"] += row["sampled_seconds"]
        if app and app not in current["apps"]:
            current["apps"].append(app)
        previous_window = identity
    durations = [item["sampled_seconds"] for item in sessions]
    identified = sum(durations)
    histogram = [{"label": label, "sessions": 0, "sampled_seconds": 0.0}
                 for label, _ in BUCKETS]
    for seconds in durations:
        index = next(i for i, (_, upper) in enumerate(BUCKETS) if seconds < upper)
        histogram[index]["sessions"] += 1
        histogram[index]["sampled_seconds"] += seconds
    for item in histogram:
        item["sampled_seconds"] = round(item["sampled_seconds"], 3)
    longest = max(sessions, key=lambda item: item["sampled_seconds"], default=None)
    for item in sessions:
        del item["key"]
        item["sampled_seconds"] = round(item["sampled_seconds"], 3)
    return {"session_count": len(sessions), "context_switches": switches,
            "window_switches": window_switches,
            "identified_seconds": round(identified, 3),
            "context_switches_per_identified_hour": round(switches * 3600 / identified, 1) if identified else None,
            "median_stretch_seconds": round(median(durations), 3) if durations else None,
            "longest_stretch_seconds": round(max(durations), 3) if durations else None,
            "longest_stretch": longest,
            "sustained_seconds": round(sum(seconds for seconds in durations if seconds >= 1200), 3),
            "brief_sessions": sum(seconds < 120 for seconds in durations),
            "distribution": histogram, "sessions": sessions,
            "definition": "Consecutive identified samples in one supported topic or exact window; excludes idle, lock, unidentified intervals and gaps. Not a measure of attention or productivity."}


def compact(result: dict) -> dict:
    """Long-range records need measures, not a copy of raw window titles."""
    return {key: value for key, value in result.items()
            if key not in {"sessions", "longest_stretch"}}


def aggregate(days: list[dict]) -> dict:
    rows = [day["behavior"] for day in days if "behavior" in day]
    identified = sum(row["identified_seconds"] for row in rows)
    switches = sum(row["context_switches"] for row in rows)
    histogram = [{"label": label,
                  "sessions": sum(row["distribution"][i]["sessions"] for row in rows),
                  "sampled_seconds": round(sum(row["distribution"][i]["sampled_seconds"] for row in rows), 3)}
                 for i, (label, _) in enumerate(BUCKETS)]
    return {"days_with_context": sum(row["identified_seconds"] > 0 for row in rows),
            "identified_seconds": round(identified, 3), "context_switches": switches,
            "context_switches_per_identified_hour": round(switches * 3600 / identified, 1) if identified else None,
            "session_count": sum(row["session_count"] for row in rows),
            "sustained_seconds": round(sum(row["sustained_seconds"] for row in rows), 3),
            "longest_stretch_seconds": max((row["longest_stretch_seconds"] or 0 for row in rows), default=0) or None,
            "distribution": histogram}
