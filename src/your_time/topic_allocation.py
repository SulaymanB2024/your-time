"""Allocate observed Mac seconds to specific topics or honest broad context."""

from __future__ import annotations

from collections import Counter

from vision_batch import SENSITIVE_RE
from window_topic_tagging import topic_is_specific, window_id


def broad_context(app: str | None, title: str | None, site_host: str | None = None) -> str:
    text = (title or "").casefold()
    if title and SENSITIVE_RE.search(title):
        return "Sensitive window (topic hidden)"
    if "youtube music" in text:
        return "Music site (subject unknown)"
    if "youtube" in text or "video" in text:
        return "Video site (subject unknown)"
    if "chatgpt" in text or "claude" in text:
        return "AI chat (subject unknown)"
    if "gmail" in text or "mail" in text or "messages" in text:
        return "Messages (subject unknown)"
    if "sheets" in text or "spreadsheet" in text:
        return "Spreadsheet (subject unknown)"
    if "docs" in text or "document" in text:
        return "Document (subject unknown)"
    if "github" in text or "codex" in text or "terminal" in text:
        return "Development tool (task unknown)"
    if "chess" in text:
        return "Chess context (activity unknown)"
    if "instagram" in text or "twitter" in text or "reddit" in text:
        return "Social feed (subject unknown)"
    if app and any(word in app.casefold() for word in ("chrome", "safari", "firefox", "edge", "arc")):
        if site_host:
            return f"{site_host[:80]} (topic unknown)"
        return "Browser page (topic unknown)"
    if app:
        return f"{app[:60]} (task unknown)"
    return "Unclear"


def allocate(segments: list[dict], tags: list[dict]) -> list[dict]:
    by_id = {item["id"]: item["topic"] for item in tags
             if item.get("status") == "model_inference" and topic_is_specific(item.get("topic", ""))}
    totals = Counter()
    for segment in segments:
        if segment["state"] != "active":
            continue
        app, title = segment.get("app"), segment.get("window")
        specific = by_id.get(window_id(app, title)) if title else None
        if specific:
            label, status = specific, "specific_model"
        else:
            label, status = broad_context(app, title, segment.get("site_host")), "broad_context"
            if label == "Unclear":
                status = "unclassified"
        totals[(label, status)] += segment["sampled_seconds"]
    return [{"label": label, "status": status, "sampled_seconds": round(seconds, 3)}
            for (label, status), seconds in totals.most_common()]
