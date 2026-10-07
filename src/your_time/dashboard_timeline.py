"""Exact, local-only intervals for the dashboard's separate device lanes."""

from collections import Counter, defaultdict
from datetime import datetime


def build_timeline(analysis: dict, behavior: dict, app_name) -> dict:
    """Keep foreground gaps; partition overlapping phone intervals by app set.

    Geometry is wall time. Mac labels and sampled durations come from the
    existing conservative behavior analysis; phone duration is interval union.
    No inactive interval becomes foreground, and no model label is confirmed.
    """
    mac = [{key: row[key] for key in
            ("start_utc", "end_utc", "sampled_seconds", "label", "status")}
           for row in behavior.get("sessions", [])]
    for row in analysis["mac"]["segments"]:
        if row["state"] != "unattributed" or row["sampled_seconds"] <= 0:
            continue
        if datetime.fromisoformat(row["end_utc"]) <= datetime.fromisoformat(row["start_utc"]):
            continue
        mac.append({"start_utc": row["start_utc"], "end_utc": row["end_utc"],
                    "sampled_seconds": row["sampled_seconds"],
                    "label": "Window unknown", "status": "unknown"})
    mac.sort(key=lambda row: datetime.fromisoformat(row["start_utc"]))

    events = defaultdict(Counter)
    for row in analysis["iphone"]["sessions"]:
        start, end = map(datetime.fromisoformat, (row["start_utc"], row["end_utc"]))
        if end <= start:
            continue
        app = app_name(row.get("app") or "Unknown app")
        events[start][app] += 1
        events[end][app] -= 1
    points = sorted(events)
    active = Counter()
    phone = []
    for index, start in enumerate(points[:-1]):
        active.update(events[start])
        apps = sorted(app for app, count in active.items() if count > 0)
        if not apps:
            continue
        end = points[index + 1]
        seconds = (end - start).total_seconds()
        if phone and phone[-1]["apps"] == apps and phone[-1]["end_utc"] == start.isoformat():
            phone[-1]["end_utc"] = end.isoformat()
            phone[-1]["sampled_seconds"] += seconds
        else:
            phone.append({"start_utc": start.isoformat(), "end_utc": end.isoformat(),
                          "sampled_seconds": seconds, "apps": apps,
                          "label": apps[0] if len(apps) == 1 else "Overlapping app records",
                          "status": "recorded_app"})
    return {"mac": mac, "iphone": phone}
