"""Review and store user-confirmed task labels for bounded uncertain Mac time."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from daily_analysis import ZONE, analyze
from local_synthesis import validate_text
from screen_context_tagging import read_supported_tags
from secure_store import DB_PATH, connect

SLOT_SECONDS = 15 * 60
MIN_REVIEW_SECONDS = 2 * 60


def slot_id(day: date, start: datetime) -> str:
    return hashlib.sha256(f"review-v1:{day.isoformat()}:{start.isoformat()}".encode()).hexdigest()[:24]


def unknown_seconds(report: dict, start: datetime, end: datetime) -> float:
    """Intersect exact support; absent sparse support cannot justify task time."""
    total = 0.0
    for segment in report["mac"]["segments"]:
        if segment["state"] != "unattributed":
            continue
        lower, upper = (datetime.fromisoformat(segment[key]) for key in ("start_utc", "end_utc"))
        supports = segment.get("support_runs")
        if supports is None:
            if abs((upper-lower).total_seconds() - segment["sampled_seconds"]) > 1e-6:
                continue
            supports = [{"start_utc": lower.isoformat(), "end_utc": upper.isoformat()}]
        for support in supports:
            at, until = (datetime.fromisoformat(support[key]) for key in ("start_utc", "end_utc"))
            total += max(0.0, (min(end, until) - max(start, at)).total_seconds())
    return total


def candidates(day: date, report: dict | None = None) -> list[dict]:
    report = report or analyze(day)
    start = datetime.fromisoformat(report["start_utc"])
    end = datetime.fromisoformat(report["analyzed_through_utc"])
    slots = []
    cursor = start
    while cursor < end:
        upper = min(end, cursor + timedelta(seconds=SLOT_SECONDS))
        total = unknown_seconds(report, cursor, upper)
        if total >= MIN_REVIEW_SECONDS:
            slots.append({"id": slot_id(day, cursor), "day_local": day.isoformat(),
                          "start_utc": cursor.isoformat(), "end_utc": upper.isoformat(),
                          "unknown_seconds": round(total, 3)})
        cursor = upper
    return slots


def existing_labels(day: date) -> dict[str, str]:
    if not DB_PATH.exists():
        return {}
    database = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = database.execute(
            "SELECT id,label FROM task_corrections WHERE day_local=? "
            "AND evidence_tier='user_confirmed_label'",
            (day.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        database.close()
    return dict(rows)


def current_intervals(day: date) -> list[dict]:
    """Current confirmed scope, including its stored cutoff and revision."""
    if not DB_PATH.exists():
        return []
    database = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = database.execute(
            "SELECT id,start_utc,end_utc,label,created_at_utc FROM task_corrections "
            "WHERE day_local=? AND evidence_tier='user_confirmed_label'", (day.isoformat(),)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        database.close()
    return [{"id": identity, "start_utc": start, "end_utc": end, "label": label,
             "created_at_utc": revision, "evidence_tier": "user_confirmed_label"}
            for identity, start, end, label, revision in rows]


def active_outcomes(day: date) -> list[dict]:
    if not DB_PATH.exists():
        return []
    database = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = database.execute(
            "SELECT id,label,source_correction_id FROM task_outcomes "
            "WHERE day_local=? AND evidence_tier='user_confirmed_result' "
            "ORDER BY created_at_utc", (day.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        database.close()
    return [{"id": identity, "label": label, "source_correction_id": source}
            for identity, label, source in rows]


def review(day: date, report: dict | None = None) -> list[dict]:
    report = report or analyze(day)
    labels = {item["id"]: item for item in current_intervals(day)}
    outcomes = {item["source_correction_id"] for item in active_outcomes(day)}
    topics = [(datetime.fromisoformat(item["timestamp_utc"]), item["topic"])
              for item in read_supported_tags(day)]
    result = []
    for row in candidates(day, report):
        start = datetime.fromisoformat(row["start_utc"])
        end = datetime.fromisoformat(row["end_utc"])
        suggested = Counter(topic for at, topic in topics if start <= at < end)
        scope = labels.get(row["id"])
        confirmed = (unknown_seconds(report, datetime.fromisoformat(scope["start_utc"]),
                                     datetime.fromisoformat(scope["end_utc"])) if scope else 0)
        confirmed = min(confirmed, row["unknown_seconds"])
        result.append({**row, "confirmed_label": scope["label"] if scope and confirmed >= row["unknown_seconds"] else None,
                       "confirmed_scope": ({"label": scope["label"], "start_utc": scope["start_utc"],
                                             "end_utc": scope["end_utc"], "sampled_seconds": round(confirmed, 3)} if scope else None),
                       "unconfirmed_seconds": round(max(0, row["unknown_seconds"] - confirmed), 3),
                       "suggested_label": suggested.most_common(1)[0][0] if suggested else None,
                       "outcome_marked": row["id"] in outcomes})
    return result


def set_label(day: date, identity: str, label: str) -> dict:
    label = validate_text(label, 65)
    if not label or label.casefold() == "unclear":
        raise ValueError("Choose a specific, non-sensitive task label")
    report = analyze(day)
    row = next((item for item in candidates(day, report) if item["id"] == identity), None)
    if row is None:
        raise ValueError("Review interval is no longer available")
    now = datetime.now(timezone.utc).isoformat()
    with connect() as database:
        prior = database.execute("SELECT label FROM task_corrections WHERE id=?", (identity,)).fetchone()
        if prior and prior[0] != label:
            database.execute("UPDATE task_outcomes SET evidence_tier='retracted' "
                             "WHERE source_correction_id=?", (identity,))
        database.execute(
            "INSERT INTO task_corrections VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET label=excluded.label, "
            "created_at_utc=excluded.created_at_utc, "
            "evidence_tier=excluded.evidence_tier",
            (identity, day.isoformat(), row["start_utc"], row["end_utc"],
             label, now, "user_confirmed_label"),
        )
    scope = next(item for item in current_intervals(day) if item["id"] == identity)
    seconds = unknown_seconds(report, datetime.fromisoformat(scope["start_utc"]), datetime.fromisoformat(scope["end_utc"]))
    return {"id": identity, "day_local": day.isoformat(), "label": label,
            "sampled_seconds": round(seconds, 3), "start_utc": scope["start_utc"],
            "end_utc": scope["end_utc"], "evidence_tier": "user_confirmed_label"}


def clear_label(day: date, identity: str) -> bool:
    with connect() as database:
        changed = bool(database.execute(
            "UPDATE task_corrections SET evidence_tier='retracted' "
            "WHERE id=? AND day_local=? AND evidence_tier='user_confirmed_label'",
            (identity, day.isoformat()),
        ).rowcount)
        if changed:
            database.execute("UPDATE task_outcomes SET evidence_tier='retracted' "
                             "WHERE source_correction_id=?", (identity,))
        return changed


def mark_outcome(day: date, correction_id: str) -> dict:
    with connect() as database:
        row = database.execute(
            "SELECT label FROM task_corrections WHERE id=? AND day_local=? "
            "AND evidence_tier='user_confirmed_label'", (correction_id, day.isoformat()),
        ).fetchone()
        if not row:
            raise ValueError("Confirm a task label before marking a result")
        identity = hashlib.sha256(f"result-v1:{day}:{correction_id}".encode()).hexdigest()[:24]
        database.execute(
            "INSERT INTO task_outcomes VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET label=excluded.label, "
            "evidence_tier=excluded.evidence_tier, created_at_utc=excluded.created_at_utc",
            (identity, day.isoformat(), row[0], correction_id,
             "user_confirmed_result", datetime.now(timezone.utc).isoformat()),
        )
    return {"id": identity, "day_local": day.isoformat(), "label": row[0],
            "evidence_tier": "user_confirmed_result"}


def retract_outcome(day: date, correction_id: str) -> bool:
    with connect() as database:
        return bool(database.execute(
            "UPDATE task_outcomes SET evidence_tier='retracted' "
            "WHERE day_local=? AND source_correction_id=? "
            "AND evidence_tier='user_confirmed_result'",
            (day.isoformat(), correction_id),
        ).rowcount)


def summary(day: date, report: dict | None = None) -> list[dict]:
    report = report or analyze(day)
    totals = Counter()
    for row in current_intervals(day):
        totals[row["label"]] += unknown_seconds(report, datetime.fromisoformat(row["start_utc"]),
                                               datetime.fromisoformat(row["end_utc"]))
    return [{"label": label, "sampled_seconds": round(seconds, 3),
             "status": "user_confirmed_label"}
            for label, seconds in totals.most_common()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(ZONE).date())
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set-id")
    parser.add_argument("--label")
    parser.add_argument("--clear-id")
    args = parser.parse_args()
    if args.set_id and args.label:
        print(json.dumps(set_label(args.day, args.set_id, args.label)))
    elif args.clear_id:
        print(json.dumps({"cleared": clear_label(args.day, args.clear_id)}))
    elif args.list or not (args.set_id or args.clear_id):
        print(json.dumps({"day_local": args.day.isoformat(), "intervals": review(args.day)},
                         sort_keys=True))
    else:
        parser.error("--set-id requires --label")


if __name__ == "__main__":
    main()
