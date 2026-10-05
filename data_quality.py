"""Reconcile private time accounting without returning captured text."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import math

from daily_analysis import union_seconds


def check_analysis(report: dict, *, workstreams: list[dict] | None = None,
                   screen_workstreams: list[dict] | None = None,
                   focus_blocks: list[dict] | None = None,
                   behavior: dict | None = None) -> dict:
    """Check accounting, bounds and grain; passing does not establish accuracy."""
    issues = Counter()
    checks = 0

    def require(code: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            issues[code] += 1

    def finite(value) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0

    start = datetime.fromisoformat(report["start_utc"])
    end = datetime.fromisoformat(report["analyzed_through_utc"])
    span = (end - start).total_seconds()
    mac, phone, visual = report["mac"], report["iphone"], report["visual"]
    states = mac["state_sampled_seconds"]
    observed = mac["observed_sampled_seconds"]
    unknown = mac["unobserved_seconds"]
    require("nonnegative_finite_time", all(finite(x) for x in [span, observed, unknown, *states.values()]))
    require("known_mac_states", set(states) <= {"active", "unattributed", "idle", "locked"})
    require("mac_state_sum", abs(sum(states.values()) - observed) <= .02)
    require("mac_elapsed_reconciliation", abs(observed + unknown - span) <= .02)
    previous_end = start
    segment_seconds = 0.0
    for row in mac["segments"]:
        at, until = datetime.fromisoformat(row["start_utc"]), datetime.fromisoformat(row["end_utc"])
        duration = (until - at).total_seconds()
        require("mac_segment_bounds", start <= at < until <= end)
        require("mac_segment_overlap", at >= previous_end)
        require("mac_segment_sampled_time", finite(row["sampled_seconds"])
                and row["sampled_seconds"] <= duration + .001)
        segment_seconds += row["sampled_seconds"]
        previous_end = until
    require("mac_segment_sum", abs(segment_seconds - observed) <= .02)
    foreground = states.get("active", 0) + states.get("unattributed", 0)
    require("mac_app_time_bound", sum(row["seconds"] for row in mac["app_sampled_seconds"]) <= foreground + .02)
    intervals = []
    for row in phone["sessions"]:
        at, until = datetime.fromisoformat(row["start_utc"]), datetime.fromisoformat(row["end_utc"])
        require("phone_session_bounds", start <= at < until <= end)
        intervals.append((at, until))
    require("phone_union_reconciliation", abs(union_seconds(intervals) - phone["device_union_seconds"]) <= .02)
    require("phone_time_bound", finite(phone["device_union_seconds"]) and phone["device_union_seconds"] <= span + .02)
    require("phone_interval_count", phone["interval_count"] == len(intervals))
    require("phone_app_time_bound", all(finite(row["focus_seconds"]) and row["focus_seconds"] <= span + .02
                                       for row in phone["app_totals"]))
    require("screenshot_status_reconciliation", sum(visual["ocr_status_counts"].values()) == visual["indexed_screenshots"])
    if workstreams is not None:
        require("topic_time_reconciliation", abs(sum(row["sampled_seconds"] for row in workstreams)
                                                  - states.get("active", 0)) <= .1)
    if screen_workstreams is not None:
        require("suggested_screen_time_bound", sum(row["sampled_seconds"] for row in screen_workstreams)
                <= states.get("unattributed", 0) + .1)
    if focus_blocks is not None:
        require("focus_time_reconciliation", abs(sum(row["sampled_seconds"] for row in focus_blocks)
                                                  - states.get("active", 0)) <= .1)
    if behavior is not None:
        identified = behavior["identified_seconds"]
        distribution = behavior["distribution"]
        require("context_time_reconciliation", finite(identified)
                and abs(identified - states.get("active", 0)) <= .1)
        require("context_distribution_reconciliation", all(finite(row["sampled_seconds"]) for row in distribution)
                and abs(sum(row["sampled_seconds"] for row in distribution) - identified) <= .1)
        require("context_session_count", sum(row["sessions"] for row in distribution) == behavior["session_count"])
        require("sustained_context_bound", finite(behavior["sustained_seconds"])
                and behavior["sustained_seconds"] <= identified + .1)
        require("longest_context_bound", behavior["longest_stretch_seconds"] is None or
                (finite(behavior["longest_stretch_seconds"]) and behavior["longest_stretch_seconds"] <= identified + .1))
    return {"status": "passed" if not issues else "failed", "checks_run": checks,
            "violations": [{"code": code, "count": count} for code, count in sorted(issues.items())],
            "scope": "Time accounting and source bounds; not attention, label accuracy or outcomes."}
