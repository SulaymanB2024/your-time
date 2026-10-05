"""Use pinned local 9B vision through the available overnight window."""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import os
import time
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path

from secure_store import STATE_DIR, connect
from model_execution import ModelBusy
from vision_batch import (PROMPT_VERSION, SENSITIVE_RE, ZONE, in_overnight_window,
                          resource_gate, seconds_until_window_end, sha256_file,
                          spread_selection, dedupe_by_featureprint, select_images, write_receipt)
from vision_quality_eval import run_image, verify_models


MODEL_PROMPT_VERSION = "fallback_visible_task_v1"
RECEIPT = STATE_DIR / "vision-fallback-latest-receipt.json"
HISTORY_DIR = STATE_DIR / "vision-run-receipts"
LOCK = STATE_DIR / "vision-fallback.lock"
MAX_PER_IMAGE_SECONDS = 240
ALLOWED_SCREENSHOT_ROOTS = (STATE_DIR / "screenshots", STATE_DIR / "pensieve/screenshots")


def is_private_screenshot(path: Path) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid():
        return False
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in ALLOWED_SCREENSHOT_ROOTS)


def select_incomplete(day: date, source_sha: str, fallback_sha: str, limit: int) -> list[tuple]:
    start = datetime.combine(day, clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    with connect() as database:
        rows = database.execute(
            "SELECT s.path,s.timestamp_utc,s.active_app,s.active_window,s.ocr_text, "
            "COALESCE((SELECT fallback.attempts FROM vision_descriptions AS fallback "
            "WHERE fallback.path=s.path AND fallback.model_sha256=? "
            "AND fallback.prompt_version=?),0),source.screenshot_sha256 "
            "FROM vision_descriptions AS source JOIN screenshots AS s ON s.path=source.path "
            "WHERE source.model_sha256=? AND source.prompt_version=? "
            "AND source.status='incomplete' AND s.ocr_status='complete' "
            "AND s.timestamp_utc>=? AND s.timestamp_utc<? "
            "AND NOT EXISTS (SELECT 1 FROM vision_descriptions AS fallback "
            "WHERE fallback.path=s.path AND fallback.model_sha256=? "
            "AND fallback.prompt_version=? AND (fallback.status IN ('complete','sensitive_skipped') OR fallback.attempts>=2)) "
            "ORDER BY s.timestamp_utc",
            (fallback_sha, MODEL_PROMPT_VERSION, source_sha, PROMPT_VERSION,
             start.isoformat(), end.isoformat(),
             fallback_sha, MODEL_PROMPT_VERSION),
        ).fetchall()
    return spread_selection(rows, limit)


def select_quality(day: date, source_sha: str, fallback_sha: str, limit: int) -> list[tuple]:
    """Spread 9B checks across the day's completed 2B frames and app contexts."""
    start = datetime.combine(day, clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    with connect() as database:
        rows = database.execute(
            "SELECT s.path,s.timestamp_utc,s.active_app,s.active_window,s.ocr_text,"
            "COALESCE(fallback.attempts,0),source.screenshot_sha256 "
            "FROM vision_descriptions source JOIN screenshots s ON s.path=source.path "
            "LEFT JOIN vision_descriptions fallback ON fallback.path=s.path "
            "AND fallback.model_sha256=? AND fallback.prompt_version=? "
            "WHERE source.model_sha256=? AND source.prompt_version=? "
            "AND source.status='complete' AND s.ocr_status='complete' "
            "AND s.timestamp_utc>=? AND s.timestamp_utc<? "
            "AND (fallback.status IS NULL OR (fallback.status NOT IN ('complete','sensitive_skipped') AND fallback.attempts<2)) "
            "ORDER BY s.timestamp_utc",
            (fallback_sha, MODEL_PROMPT_VERSION, source_sha, PROMPT_VERSION,
             start.isoformat(), end.isoformat()),
        ).fetchall()
        features = dict(database.execute(
            "SELECT f.path,f.vector FROM screen_features f "
            "JOIN screenshots s ON s.path=f.path WHERE s.timestamp_utc>=? AND s.timestamp_utc<? "
            "AND f.feature_revision=2",
            (start.isoformat(), end.isoformat()),
        ).fetchall())
    rows = dedupe_by_featureprint(rows, features)
    groups: dict[tuple[int, str], list[tuple]] = {}
    for row in rows:
        local = datetime.fromisoformat(row[1]).astimezone(ZONE)
        groups.setdefault((local.hour, row[2] or ""), []).append(row)
    chosen = []
    seen_windows = set()
    while groups and len(chosen) < limit:
        for key in sorted(list(groups)):
            group = groups[key]
            index = next((i for i, row in enumerate(group)
                          if (key, row[3]) not in seen_windows), len(group) // 2)
            row = group.pop(index)
            chosen.append(row)
            seen_windows.add((key, row[3]))
            if not group:
                del groups[key]
            if len(chosen) >= limit:
                break
    return sorted(chosen, key=lambda row: row[1])


def select_direct(day: date, source_sha: str, fallback_sha: str, limit: int) -> list[tuple]:
    """Find spread-out OCR-complete frames that neither model has described."""
    candidates = select_images(day, 10, None)
    start = datetime.combine(day, clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    with connect() as database:
        prior = {(path, model): (status, attempts) for path, model, status, attempts in
                 database.execute(
                     "SELECT path,model_sha256,status,attempts FROM vision_descriptions "
                     "WHERE model_sha256 IN (?,?) AND timestamp_utc>=? AND timestamp_utc<?",
                     (source_sha, fallback_sha, start.isoformat(), end.isoformat()))}
    rows = []
    for path, timestamp, app, window, ocr in candidates:
        # Existing completed 2B rows enter quality upgrades. Sensitive rows
        # remain excluded; other eligible frames belong to the primary 9B queue.
        if prior.get((path, source_sha), (None, 0))[0] in {"complete", "sensitive_skipped"}:
            continue
        status, attempts = prior.get((path, fallback_sha), (None, 0))
        if status in {"complete", "sensitive_skipped"} or attempts >= 2:
            continue
        rows.append((path, timestamp, app, window, ocr, attempts, None))
    return context_priority(rows, limit)


def context_priority(rows: list[tuple], limit: int) -> list[tuple]:
    """Cover hours and distinct foreground contexts before dense repeats."""
    groups = {}
    for row in rows:
        hour = datetime.fromisoformat(row[1]).astimezone(ZONE).hour
        groups.setdefault((hour, row[2] or "", row[3] or ""), []).append(row)
    representatives = sorted((group[len(group) // 2] for group in groups.values()),
                             key=lambda row: row[1])
    chosen = spread_priority(representatives, limit)
    seen = {row[0] for row in chosen}
    remaining = [row for row in rows if row[0] not in seen]
    return chosen + spread_priority(remaining, limit - len(chosen))


def select_backfill(day: date, source_sha: str, model_sha: str, limit: int) -> list[tuple]:
    """Give uncovered retained days a bounded share without displacing yesterday."""
    start = datetime.combine(day - timedelta(days=30), clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day, clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    with connect() as database:
        timestamps = database.execute(
            "SELECT timestamp_utc FROM screenshots WHERE ocr_status='complete' "
            "AND timestamp_utc>=? AND timestamp_utc<? ORDER BY timestamp_utc",
            (start.isoformat(), end.isoformat())).fetchall()
    days = sorted({datetime.fromisoformat(row[0]).astimezone(ZONE).date()
                   for row in timestamps})
    groups = [deque(select_direct(prior, source_sha, model_sha, limit)) for prior in days]
    rows = []
    while any(groups) and len(rows) < limit:
        for group in groups:
            if group:
                rows.append(group.popleft())
                if len(rows) >= limit:
                    break
    return rows


def with_backfill(current: list[tuple[str, tuple]], backlog: list[tuple],
                  limit: int) -> list[tuple[str, tuple]]:
    return with_history(current, [("backfill", row) for row in backlog], limit)


def with_history(current: list[tuple[str, tuple]], history: list[tuple[str, tuple]],
                 limit: int) -> list[tuple[str, tuple]]:
    """Reserve at most one in ten prefix slots for older work while current remains."""
    current, history = deque(current), deque(history)
    result = []
    while len(result) < limit and (current or history):
        for _ in range(9):
            if current and len(result) < limit:
                result.append(current.popleft())
        if history and len(result) < limit:
            result.append(history.popleft())
    return result


def spread_priority(rows: list[tuple], limit: int) -> list[tuple]:
    """Order candidates so partial nights cover the full day first."""
    result = []
    intervals = deque([(0, len(rows) - 1)]) if rows else deque()
    while intervals and len(result) < limit:
        start, end = intervals.popleft()
        middle = (start + end) // 2
        result.append(rows[middle])
        if start < middle:
            intervals.append((start, middle - 1))
        if middle < end:
            intervals.append((middle + 1, end))
    return result


def select_prior_retries(day: date, fallback_sha: str, limit: int) -> list[tuple]:
    """Retry incomplete 9B work from the preceding six complete days once."""
    start = datetime.combine(day - timedelta(days=6), clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    end = datetime.combine(day, clock_time.min, tzinfo=ZONE).astimezone(timezone.utc)
    with connect() as database:
        rows = database.execute(
            "SELECT s.path,s.timestamp_utc,s.active_app,s.active_window,s.ocr_text,"
            "v.attempts,v.screenshot_sha256 FROM vision_descriptions v "
            "JOIN screenshots s ON s.path=v.path "
            "WHERE v.model_sha256=? AND v.prompt_version=? AND v.status='incomplete' "
            "AND v.attempts=1 AND s.ocr_status='complete' "
            "AND s.timestamp_utc>=? AND s.timestamp_utc<? ORDER BY s.timestamp_utc DESC",
            (fallback_sha, MODEL_PROMPT_VERSION, start.isoformat(), end.isoformat()),
        ).fetchall()
    return rows[:limit]


def interleave_new_and_upgrades(direct: list[tuple], quality: list[tuple],
                                limit: int) -> list[tuple[str, tuple]]:
    """Give both new coverage and stronger checks a share of 9B time."""
    result = []
    while len(result) < limit and (direct or quality):
        if direct:
            result.append(("direct", direct.pop(0)))
        if len(result) < limit and quality:
            result.append(("quality", quality.pop(0)))
    return result


def generation_budget(prior_attempts: int) -> tuple[int, int]:
    # Last night's 1024-token ceiling left 16 of 75 9B attempts without a
    # final answer. Give the model more room while bounding each frame.
    return (2048, 300) if prior_attempts else (1536, MAX_PER_IMAGE_SECONDS)


def save_description(path_text: str, timestamp: str, screenshot_sha: str,
                     model_sha: str, result: dict) -> None:
    with connect() as database:
        database.execute(
            "INSERT INTO vision_descriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path,model_sha256,prompt_version) DO UPDATE SET "
            "description=excluded.description,status=excluded.status,"
            "attempts=vision_descriptions.attempts+1,elapsed_seconds=excluded.elapsed_seconds,"
            "updated_at_utc=excluded.updated_at_utc",
            (path_text, model_sha, MODEL_PROMPT_VERSION, screenshot_sha, timestamp,
             result.get("description"), result["status"], 1,
             result["elapsed_seconds"], datetime.now(timezone.utc).isoformat()),
        )


def finish_receipt(receipt: dict, started: float) -> None:
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_receipt(receipt, RECEIPT)
    HISTORY_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(HISTORY_DIR, 0o700)
    history = HISTORY_DIR / ("vision-fallback-" + receipt["started_at_utc"].replace(":", "-").replace("+", "_") + ".json")
    if history.exists():
        raise RuntimeError("Historical fallback receipt already exists")
    write_receipt(receipt, history)


def run(day: date, limit: int, *, mode: str = "hard", hard_limit: int | None = None) -> dict:
    started = time.monotonic()
    receipt = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
               "day_local": day.isoformat(), "max_images": limit,
               "selection_policy": "current_day_9_to_history_1" if mode == "mixed" else "hard_only",
               "selected": 0, "hard_selected": 0, "retry_selected": 0,
               "direct_selected": 0, "backfill_selected": 0,
               "quality_selected": 0, "hard_completed": 0,
               "retry_completed": 0, "direct_completed": 0, "backfill_completed": 0,
               "quality_completed": 0,
               "completed": 0, "sensitive_skipped": 0,
               "failed": 0, "stop_reason": None}
    gate = resource_gate(benchmark_now=False)
    if gate or not in_overnight_window(datetime.now(timezone.utc)):
        receipt["stop_reason"] = gate or "outside_overnight_window"
        finish_receipt(receipt, started)
        return receipt
    models = verify_models(["9b_q8"])
    model = models["9b_q8"]
    source_sha = json.loads(Path(__file__).with_name("vision_model_manifest.json").read_text())["files"][0]["sha256"]
    hard_count = limit if mode == "hard" else min(limit, hard_limit if hard_limit is not None else 30)
    hard_rows = select_incomplete(day, source_sha, model["weights_sha256"], hard_count)
    retry_rows = (select_prior_retries(day, model["weights_sha256"],
                                      min(15, limit - len(hard_rows)))
                  if mode == "mixed" else [])
    remaining = limit - len(hard_rows)
    direct_rows = (select_direct(day, source_sha, model["weights_sha256"], remaining)
                   if mode == "mixed" else [])
    quality_rows = (select_quality(day, source_sha, model["weights_sha256"], remaining)
                    if mode == "mixed" else [])
    backlog = (select_backfill(day, source_sha, model["weights_sha256"], min(100, limit))
               if mode == "mixed" else [])
    reserved_paths = {row[0] for row in hard_rows + retry_rows}
    backlog = [row for row in backlog if row[0] not in reserved_paths]
    current = interleave_new_and_upgrades(direct_rows, quality_rows, remaining)
    history = [("retry", row) for row in retry_rows] + [("backfill", row) for row in backlog]
    rows = [("hard", row) for row in hard_rows] + with_history(current, history, remaining)
    receipt["selected"] = len(rows)
    for lane, _ in rows:
        receipt[f"{lane}_selected"] += 1
    for lane, row in rows:
        path_text, timestamp, app, window, ocr, prior_attempts = row[:6]
        gate = resource_gate(benchmark_now=False)
        if gate or not in_overnight_window(datetime.now(timezone.utc)):
            receipt["stop_reason"] = gate or "overnight_window_ended"
            break
        _, timeout_seconds = generation_budget(prior_attempts)
        if seconds_until_window_end(datetime.now(timezone.utc)) < timeout_seconds + 60:
            receipt["stop_reason"] = "insufficient_window_for_next_image"
            break
        path = Path(path_text)
        if not is_private_screenshot(path):
            receipt["failed"] += 1
            continue
        if SENSITIVE_RE.search(" ".join(value or "" for value in (app, window, ocr))):
            save_description(path_text, timestamp, row[6] or "",
                             model["weights_sha256"],
                             {"status": "sensitive_skipped", "description": None,
                              "elapsed_seconds": 0.0})
            receipt["sensitive_skipped"] += 1
            write_receipt(receipt, RECEIPT)
            continue
        screenshot_sha = sha256_file(path)
        if row[6] is not None and screenshot_sha != row[6]:
            receipt["failed"] += 1
            continue
        max_tokens, timeout_seconds = generation_budget(prior_attempts)
        try:
            result = run_image(path, model, timeout_seconds=timeout_seconds,
                               max_tokens=max_tokens)
        except ModelBusy:
            receipt["stop_reason"] = "local_model_busy"
            break
        save_description(path_text, timestamp, screenshot_sha, model["weights_sha256"], result)
        receipt["completed" if result["status"] == "complete" else "failed"] += 1
        if result["status"] == "complete":
            receipt[f"{lane}_completed"] += 1
        write_receipt(receipt, RECEIPT)
    receipt["stop_reason"] = receipt["stop_reason"] or "selected_images_processed"
    finish_receipt(receipt, started)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", help="Local YYYY-MM-DD; defaults to yesterday")
    parser.add_argument("--days-ago", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=("hard", "mixed"), default="hard")
    parser.add_argument("--hard-limit", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.limit <= 2000:
        parser.error("--limit must be 1-2000")
    if args.mode == "mixed" and not 0 <= args.hard_limit <= args.limit:
        parser.error("--hard-limit must be between zero and --limit")
    if args.days_ago < 1 or args.days_ago > 7:
        parser.error("--days-ago must be 1-7")
    os.umask(0o077)
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        print(json.dumps({"status": "another_fallback_is_running"}))
        return
    try:
        day = date.fromisoformat(args.day) if args.day else datetime.now(ZONE).date() - timedelta(days=args.days_ago)
        result = run(day, args.limit, mode=args.mode, hard_limit=args.hard_limit)
        print(json.dumps({k: result[k] for k in ["selected", "hard_selected", "retry_selected", "direct_selected", "quality_selected", "backfill_selected", "completed", "hard_completed", "retry_completed", "direct_completed", "quality_completed", "backfill_completed", "sensitive_skipped", "failed", "stop_reason"]}, sort_keys=True))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
