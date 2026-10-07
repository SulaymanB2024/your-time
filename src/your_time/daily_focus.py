"""Build bounded, evidence-linked daily focus blocks from the private ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from daily_analysis import ANALYSIS_DIR, ZONE, analyze, private_write
from secure_store import DB_PATH
from vision_batch import SENSITIVE_RE, URL_RE

MAX_BLOCK_SECONDS = 15 * 60
MAX_GAP_SECONDS = 20
MAX_CONTEXT_ITEMS = 4
QUALITY_MANIFEST = Path(__file__).with_name("vision_quality_model_manifest.json")


def stronger_model_sha() -> str:
    manifest = json.loads(QUALITY_MANIFEST.read_text(encoding="utf-8"))
    return next(item["sha256"] for item in manifest["files"]
                if item["name"].endswith(".gguf") and not item["name"].startswith("mmproj"))


def safe_markdown(value: str) -> str:
    value = URL_RE.sub("[URL]", " ".join(value.split())[:240])
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>])", r"\\\1", value)


def visual_evidence(start: datetime, end: datetime, app: str | None = None) -> tuple[int, list[dict]]:
    """Return a small sample; descriptions remain explicitly untrusted."""
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT s.path,s.timestamp_utc,s.active_window,s.ocr_status,"
            "v.description,v.model_sha256,v.screenshot_sha256,v.prompt_version "
            "FROM screenshots s LEFT JOIN vision_descriptions v ON v.path=s.path "
            "AND v.status='complete' WHERE s.timestamp_utc>=? AND s.timestamp_utc<? "
            "AND (? IS NULL OR s.active_app IS NULL OR s.active_app=?) "
            "ORDER BY s.timestamp_utc",
            (start.isoformat(), end.isoformat(), app, app),
        ).fetchall()
    finally:
        db.close()
    grouped: dict[str, dict] = {}
    for path, at, title, ocr_status, caption, model_sha, image_sha, prompt in rows:
        item = grouped.setdefault(path, {"path": path, "timestamp_utc": at,
                                        "window": title, "ocr_status": ocr_status,
                                        "captions": []})
        if caption and not SENSITIVE_RE.search(caption):
            item["captions"].append({"text": caption, "model_sha256": model_sha,
                                      "screenshot_sha256": image_sha, "prompt_version": prompt})
    strong_sha = stronger_model_sha()
    candidates = list(grouped.values())
    strong = [item for item in candidates
              if any(caption["model_sha256"] == strong_sha for caption in item["captions"])]

    def spread(items: list[dict], count: int) -> list[dict]:
        if count <= 0:
            return []
        if len(items) <= count:
            return items
        if count == 1:
            return [items[len(items) // 2]]
        return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]

    preferred = spread(strong, min(2, MAX_CONTEXT_ITEMS))
    preferred_paths = {item["path"] for item in preferred}
    remaining = [item for item in candidates if item["path"] not in preferred_paths]
    candidates = sorted(preferred + spread(remaining, MAX_CONTEXT_ITEMS - len(preferred)),
                        key=lambda item: item["timestamp_utc"])
    result = []
    for item in candidates:
        # Preserve a pointer to the source, without duplicating raw OCR or image bytes.
        captions = sorted(item.pop("captions"), key=lambda c: (
            0 if c["model_sha256"] == strong_sha else 1,
            0 if "quality_eval" in c["prompt_version"] else 1,
            c["prompt_version"]))
        item["model_inference"] = captions[0] if captions else None
        if item["window"] and SENSITIVE_RE.search(item["window"]):
            item["window"] = "[sensitive window]"
        result.append(item)
    return len(grouped), result


def focus_blocks(segments: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    previous_active = False
    for segment in segments:
        if segment["state"] != "active":
            previous_active = False
            continue
        start = datetime.fromisoformat(segment["start_utc"])
        end = datetime.fromisoformat(segment["end_utc"])
        app = segment["app"] or "unknown_app"
        duration = (end - start).total_seconds()
        if duration <= 0:
            previous_active = False
            continue
        cursor = start
        while cursor < end:
            previous = blocks[-1] if blocks and previous_active else None
            can_extend = (previous is not None and previous["app"] == app
                          and 0 <= (cursor - datetime.fromisoformat(previous["end_utc"])).total_seconds()
                          <= MAX_GAP_SECONDS
                          and (cursor - datetime.fromisoformat(previous["start_utc"])).total_seconds()
                          < MAX_BLOCK_SECONDS)
            if not can_extend:
                previous = {"start_utc": cursor.isoformat(), "end_utc": cursor.isoformat(),
                            "app": app, "sampled_seconds": 0.0, "windows": Counter()}
                blocks.append(previous)
            block_end = datetime.fromisoformat(previous["start_utc"]) + timedelta(seconds=MAX_BLOCK_SECONDS)
            slice_end = min(end, block_end)
            sampled = segment["sampled_seconds"] * (slice_end - cursor).total_seconds() / duration
            previous["end_utc"] = slice_end.isoformat()
            previous["sampled_seconds"] += sampled
            if segment["window"]:
                title = segment["window"]
                previous["windows"]["[sensitive window]" if SENSITIVE_RE.search(title) else title] += sampled
            cursor = slice_end
            previous_active = True
    for index, block in enumerate(blocks, 1):
        block["id"] = f"mac-block-{index:04d}"
        block["sampled_seconds"] = round(block["sampled_seconds"], 3)
        windows: Counter = block.pop("windows")
        block["distinct_window_count"] = len(windows)
        block["top_windows"] = [{"title": title, "sampled_seconds": round(seconds, 3)}
                                for title, seconds in windows.most_common(5)]
        count, evidence = visual_evidence(datetime.fromisoformat(block["start_utc"]),
                                          datetime.fromisoformat(block["end_utc"]), block["app"])
        block["screenshot_count"] = count
        block["visual_evidence"] = evidence
        block["focus_interpretation"] = "foreground_app_and_window_context_only"
        block["verified_accomplishments"] = []
    return blocks


def build(day: date, *, now: datetime | None = None) -> dict:
    base = analyze(day, now=now)
    blocks = focus_blocks(base["mac"]["segments"])
    prior_path = ANALYSIS_DIR / f"focus-{day.isoformat()}.json"
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        prior = {}
    remembered = {(item["start_utc"], item["end_utc"], item["app"]): item["id"]
                  for item in prior.get("blocks", []) if all(key in item for key in
                  ("start_utc", "end_utc", "app", "id"))}
    for block in blocks:
        identity = (block["start_utc"], block["end_utc"], block["app"])
        block["id"] = remembered.get(identity) or "mac-" + hashlib.sha256(
            json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:16]
    return {
        "schema_version": 1, "day_local": base["day_local"],
        "generated_at_utc": base["generated_at_utc"],
        "complete_day": base["complete_day"],
        "mac_observed_seconds": base["mac"]["observed_sampled_seconds"],
        "mac_unobserved_seconds": base["mac"]["unobserved_seconds"],
        "iphone_focus_union_seconds": base["iphone"]["device_union_seconds"],
        "iphone_top_apps": base["iphone"]["app_totals"][:10],
        "iphone_sync_quality": base["source_quality"]["iphone_sync"],
        "blocks": blocks,
        "verified_accomplishments": [],
        "limitations": [
            "A foreground app or window indicates visible context, not sustained attention.",
            "Screenshot captions are untrusted model inference and may invent actions.",
            "No completed outcome is verified without an independent receipt or user confirmation.",
            "iPhone synced intervals can lag; phone and Mac durations may overlap.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [f"# Activity review — {report['day_local']}", "",
             "This private report describes recorded foreground context. It does not prove attention or completed actions.", "",
             f"Mac observed: {report['mac_observed_seconds']/3600:.2f} hours; "
             f"Mac unobserved: {report['mac_unobserved_seconds']/3600:.2f} hours. "
             f"iPhone focus union: {report['iphone_focus_union_seconds']/3600:.2f} hours "
             "(may overlap Mac time).", "",
             f"Latest iPhone sync check ({report['iphone_sync_quality'].get('checked_at_utc') or 'time unknown'}): "
             f"{report['iphone_sync_quality'].get('quality_status', 'unknown')}. "
             "This check describes the current readable source, not a historical daily total.", "",
             "## Mac focus blocks", ""]
    for block in report["blocks"]:
        start = datetime.fromisoformat(block["start_utc"]).astimezone(ZONE).strftime("%H:%M")
        end = datetime.fromisoformat(block["end_utc"]).astimezone(ZONE).strftime("%H:%M")
        lines.append(f"### {start}–{end} · {safe_markdown(block['app'])}")
        lines.append("")
        lines.append(f"Observed foreground: {block['sampled_seconds']/60:.1f} min; "
                     f"{block['distinct_window_count']} window titles; "
                     f"{block['screenshot_count']} screenshots.")
        if block["top_windows"]:
            lines.append("Visible context: " + "; ".join(
                f"{safe_markdown(item['title'])} ({item['sampled_seconds']/60:.1f} min)"
                for item in block["top_windows"][:3]))
        captions = [item["model_inference"]["text"] for item in block["visual_evidence"]
                    if item["model_inference"]]
        if captions:
            lines.append("Model descriptions (unverified): " + " / ".join(
                safe_markdown(item) for item in captions[:2]))
        lines.extend(("", "Completed outcomes: unverified.", ""))
    if not report["blocks"]:
        lines.extend(("No Mac foreground blocks were observed for this day.", ""))
    lines.extend(("## Evidence limits", "", *[f"- {item}" for item in report["limitations"]], ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--days-ago", type=int, action="append")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.day and args.days_ago:
        parser.error("choose --day or --days-ago")
    ages = args.days_ago or [1]
    if any(age < 0 for age in ages):
        parser.error("--days-ago must be zero or greater")
    days = [args.day] if args.day else [datetime.now(ZONE).date() - timedelta(days=age)
                                       for age in dict.fromkeys(ages)]
    for day in days:
        report = build(day)
        payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
        markdown = render_markdown(report).encode("utf-8")
        receipt = {"day_local": day.isoformat(), "blocks": len(report["blocks"]),
                   "screenshots_referenced": sum(len(b["visual_evidence"]) for b in report["blocks"]),
                   "verified_accomplishments": 0, "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest(),
                   "markdown_bytes": len(markdown),
                   "markdown_sha256": hashlib.sha256(markdown).hexdigest(),
                   "status": "private_focus_written" if args.write else "preview_only"}
        if args.write:
            name = f"focus-{day.isoformat()}"
            private_write(ANALYSIS_DIR / f"{name}.json", payload)
            private_write(ANALYSIS_DIR / f"{name}.md", markdown)
            private_write(ANALYSIS_DIR / f"{name}.manifest.json",
                          (json.dumps(receipt, indent=2) + "\n").encode())
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
