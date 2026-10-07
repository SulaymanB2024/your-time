"""Presentation-only labels and interval accounting for the offline dashboard."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta

from chronicle_rollup import local_boundaries
from daily_analysis import ZONE, union_seconds
from topic_allocation import broad_context
from vision_batch import SENSITIVE_RE
from window_topic_tagging import window_id

PHONE_NAMES = {
    "com.atebits.Tweetie2": "X",
    "com.chess.iphone": "Chess",
    "com.facebook.hatch": "Instagram",
    "com.google.ios.youtubemusic": "YouTube Music",
    "com.apple.MobileSMS": "Messages",
    "com.google.Gmail": "Gmail",
    "com.openai.chat": "ChatGPT",
    "com.linkedin.LinkedIn": "LinkedIn",
    "com.google.ios.youtube": "YouTube",
    "com.google.chrome.ios": "Chrome",
    "com.apple.control-center": "Control Center",
    "com.apple.InCallService": "Phone calls",
    "com.apple.camera": "Camera",
    "net.whatsapp.WhatsApp": "WhatsApp",
    "com.apple.SleepLockScreen": "Lock Screen",
    "com.google.Maps": "Google Maps",
    "com.apple.mobilenotes": "Notes",
    "com.apple.springboard.app-library-open-pod": "App Library",
    "com.apple.springboard.today-view": "Today View",
    "com.apple.springboard.stand-by": "StandBy",
    "com.industriousoffice.app": "Industrious",
    "com.toyopagroup.picaboo": "Snapchat",
}


def app_name(value: str | None) -> str:
    if not value:
        return "Unknown app"
    if value in PHONE_NAMES:
        return PHONE_NAMES[value]
    if value.startswith("com.") and "." in value:
        return value.rsplit(".", 1)[-1].replace("_", " ").title()
    return value


def time_of_day(analysis: dict) -> dict:
    day = datetime.fromisoformat(analysis["day_local"]).date()
    start, end = local_boundaries(day)
    step = timedelta(hours=1)
    count = (end - start) // step
    phone_intervals = [[] for _ in range(count)]
    mac = [0.0] * 24
    unattributed = [0.0] * 24
    phone = [0.0] * 24
    for row in analysis["iphone"]["sessions"]:
        lower, upper = _interval(row)
        for index, a, b in _overlapping_bins(lower, upper, start, end, step):
            phone_intervals[index].append((a, b))
    for row in analysis["mac"]["segments"]:
        if row["state"] not in {"active", "unattributed"}:
            continue
        lower, upper = _interval(row)
        duration = (upper - lower).total_seconds()
        target = mac if row["state"] == "active" else unattributed
        for index, a, b in _overlapping_bins(lower, upper, start, end, step):
            hour = (start + index * step).astimezone(ZONE).hour
            target[hour] += row["sampled_seconds"] * (b - a).total_seconds() / duration
    for index, intervals in enumerate(phone_intervals):
        hour = (start + index * step).astimezone(ZONE).hour
        phone[hour] += union_seconds(intervals)
    return {"mac": [round(value, 1) for value in mac],
            "unattributed": [round(value, 1) for value in unattributed],
            "iphone": [round(value, 1) for value in phone]}


def mac_behavior(analysis: dict) -> dict:
    longest = 0.0
    window_changes = 0
    current_start = current_end = None
    current_app = previous_window = None
    for row in analysis["mac"]["segments"]:
        if row["state"] != "active":
            current_start = current_end = None
            current_app = previous_window = None
            continue
        at, end = datetime.fromisoformat(row["start_utc"]), datetime.fromisoformat(row["end_utc"])
        linked = (current_end is not None and row["app"] == current_app
                  and 0 <= (at - current_end).total_seconds() <= 20)
        if not linked:
            current_start = at
        elif previous_window != row["window"]:
            window_changes += 1
        current_end, current_app, previous_window = end, row["app"], row["window"]
        longest = max(longest, (end - current_start).total_seconds())
    return {"longest_same_app_stretch_seconds": round(longest, 1),
            "window_changes": window_changes}


def mac_windows(analysis: dict) -> list[dict]:
    totals = Counter()
    for item in analysis["mac"]["segments"]:
        if item["state"] != "active" or not item.get("window"):
            continue
        title = item["window"]
        if SENSITIVE_RE.search(title):
            title = "[Sensitive window]"
        totals[(item.get("app"), title)] += item["sampled_seconds"]
    return [{"id": window_id(app, title), "title": title,
             "seconds": round(seconds, 3)}
            for (app, title), seconds in totals.most_common(20)]


def activity_score(analysis: dict, topic_by_id: dict[str, str], now: datetime,
                   screen_tags: list[dict] | None = None) -> list[dict]:
    day = datetime.fromisoformat(analysis["day_local"]).date()
    start, end = local_boundaries(day)
    step = timedelta(minutes=5)
    count = (end - start) // step
    mac_topics = [Counter() for _ in range(count)]
    mac_states = [Counter() for _ in range(count)]
    phone_intervals = [[] for _ in range(count)]
    phone_apps = [Counter() for _ in range(count)]
    visible_topics = [Counter() for _ in range(count)]
    for item in analysis["mac"]["segments"]:
        lower, upper = _interval(item)
        duration = (upper - lower).total_seconds()
        # Calculate an inferred label once, only when this interval reaches the day.
        label = None
        for index, a, b in _overlapping_bins(lower, upper, start, end, step):
            seconds = item["sampled_seconds"] * (b - a).total_seconds() / duration
            mac_states[index][item["state"]] += seconds
            if item["state"] == "active":
                if label is None:
                    app, title = item.get("app"), item.get("window")
                    label = ((topic_by_id.get(window_id(app, title)) if title else None)
                             or broad_context(app, title, item.get("site_host")))
                mac_topics[index][label] += seconds
    for item in analysis["iphone"]["sessions"]:
        lower, upper = _interval(item)
        for index, a, b in _overlapping_bins(lower, upper, start, end, step):
            phone_intervals[index].append((a, b))
            phone_apps[index][item["app"]] += (b - a).total_seconds()
    for item in screen_tags or []:
        if item.get("status") not in {"model_inference", "featureprint_match"} or not item.get("timestamp_utc"):
            continue
        timestamp = datetime.fromisoformat(item["timestamp_utc"])
        if start <= timestamp < end:
            visible_topics[(timestamp - start) // step][item["topic"]] += 1
    bins = []
    for index in range(count):
        cursor = start + index * step
        upper = min(end, cursor + step)
        states, topics, apps, visible = (mac_states[index], mac_topics[index],
                                       phone_apps[index], visible_topics[index])
        state = ("future" if cursor >= now else "active" if states["active"] > 0
                 else "unattributed" if states["unattributed"] > 0
                 else "locked" if states["locked"] > 0
                 else "idle" if states["idle"] > 0 else "unknown")
        bins.append({"start_utc": cursor.isoformat(), "end_utc": upper.isoformat(),
                     "mac_state": state, "mac_seconds": round(states["active"], 1),
                     "mac_unattributed_seconds": round(states["unattributed"], 1),
                     "mac_topic": topics.most_common(1)[0][0] if topics else None,
                     "mac_screen_topic": (visible.most_common(1)[0][0]
                                          if visible and states["unattributed"] > 0 else None),
                     "iphone_seconds": round(union_seconds(phone_intervals[index]), 1),
                     "iphone_app": app_name(apps.most_common(1)[0][0]) if apps else None})
    return bins


def _interval(row: dict) -> tuple[datetime, datetime]:
    return datetime.fromisoformat(row["start_utc"]), datetime.fromisoformat(row["end_utc"])


def _overlapping_bins(lower: datetime, upper: datetime, start: datetime,
                      end: datetime, step: timedelta) -> Iterator[tuple[int, datetime, datetime]]:
    """Visit only intersected UTC slots, with exclusive ends and exact clipping."""
    lower, upper = max(start, lower), min(end, upper)
    if lower >= upper:
        return
    first = (lower - start) // step
    last = (upper - start - timedelta(microseconds=1)) // step
    for index in range(first, last + 1):
        at = start + index * step
        yield index, max(at, lower), min(at + step, upper)
