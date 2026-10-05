import os
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import daily_focus


def test_focus_blocks_split_at_idle_and_bounded_context(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,active_window TEXT,ocr_status TEXT,active_app TEXT);"
        "CREATE TABLE vision_descriptions(path TEXT,status TEXT,description TEXT,model_sha256 TEXT,screenshot_sha256 TEXT,prompt_version TEXT);"
    )
    start = datetime(2026, 9, 28, 14, tzinfo=timezone.utc)
    for index in range(7):
        at = start + timedelta(minutes=index)
        path = str(tmp_path / f"{index}.webp")
        connection.execute("INSERT INTO screenshots(path,timestamp_utc,active_window,ocr_status) VALUES (?,?,?,?)",
                           (path, at.isoformat(), "Draft", "complete"))
        connection.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?)",
                           (path, "complete", "A draft is visible.", "model", "image", "visible_task_v1"))
    connection.commit()
    connection.close()
    monkeypatch.setattr(daily_focus, "DB_PATH", db)
    def segment(at, seconds, state="active", app="Editor", title="Draft"):
        return {"start_utc": at.isoformat(),
                "end_utc": (at + timedelta(seconds=seconds)).isoformat(),
                "state": state, "app": app, "window": title,
                "sampled_seconds": seconds, "sample_count": seconds // 5}
    blocks = daily_focus.focus_blocks([
        segment(start, 300),
        segment(start + timedelta(minutes=5), 300, title="Notes"),
        segment(start + timedelta(minutes=10), 5, state="idle", app=None, title=None),
        segment(start + timedelta(minutes=11), 300),
    ])
    assert len(blocks) == 2
    assert blocks[0]["sampled_seconds"] == 600
    assert blocks[0]["distinct_window_count"] == 2
    assert blocks[0]["screenshot_count"] == 7
    assert len(blocks[0]["visual_evidence"]) == 4
    assert blocks[0]["verified_accomplishments"] == []
    assert blocks[1]["screenshot_count"] == 0


def test_focus_report_keeps_outcomes_unverified_and_private(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_focus, "analyze", lambda day, now=None: {
        "day_local": day.isoformat(), "generated_at_utc": "2026-09-29T12:00:00+00:00",
        "complete_day": True,
        "mac": {"segments": [], "observed_sampled_seconds": 10, "unobserved_seconds": 20},
        "iphone": {"device_union_seconds": 30, "app_totals": []},
        "source_quality": {"iphone_sync": {"quality_status": "source_behind_retained_ledger"}},
    })
    report = daily_focus.build(datetime(2026, 9, 28).date())
    assert report["verified_accomplishments"] == []
    assert report["iphone_sync_quality"]["quality_status"] == "source_behind_retained_ledger"
    from daily_analysis import private_write
    output = tmp_path / "focus.json"
    private_write(output, b"{}\n")
    assert os.stat(output).st_mode & 0o777 == 0o600
    markdown = daily_focus.render_markdown(report)
    assert "Completed outcomes: unverified" not in markdown  # No Mac blocks were recorded.
    assert "No Mac foreground blocks" in markdown


def test_markdown_escapes_untrusted_window_text():
    value = daily_focus.safe_markdown("![image](https://example.com/a.png)\n# heading")
    assert "![image]" not in value
    assert "https://" not in value
    assert "\n" not in value


def test_visual_sample_includes_stronger_caption(tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.executescript(
        "CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,active_window TEXT,ocr_status TEXT,active_app TEXT);"
        "CREATE TABLE vision_descriptions(path TEXT,status TEXT,description TEXT,model_sha256 TEXT,screenshot_sha256 TEXT,prompt_version TEXT);"
    )
    start = datetime(2026, 9, 28, 14, tzinfo=timezone.utc)
    for i in range(10):
        path = str(tmp_path / f"{i}.webp")
        at = (start + timedelta(minutes=i)).isoformat()
        connection.execute("INSERT INTO screenshots(path,timestamp_utc,active_window,ocr_status) VALUES (?,?,?,?)", (path, at, "Draft", "complete"))
        connection.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?)",
                           (path, "complete", "Context", "2b", "image", "visible_task_v1"))
        if i == 3:
            connection.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?)",
                               (path, "complete", "Stronger context", "9b", "image", "quality_eval_v1"))
    connection.commit()
    connection.close()
    monkeypatch.setattr(daily_focus, "DB_PATH", db)
    monkeypatch.setattr(daily_focus, "stronger_model_sha", lambda: "9b")
    count, sample = daily_focus.visual_evidence(start, start + timedelta(minutes=10))
    assert count == 10
    assert len(sample) == 4
    assert any(item["model_inference"] and item["model_inference"]["model_sha256"] == "9b"
               for item in sample)


def test_backfilled_earlier_block_does_not_shift_existing_evidence_id(tmp_path, monkeypatch):
    day = datetime(2026, 9, 28).date()
    old = {"id": "mac-block-0001", "start_utc": "2026-09-28T15:00:00+00:00",
           "end_utc": "2026-09-28T15:05:00+00:00", "app": "Editor"}
    (tmp_path / "focus-2026-09-28.json").write_text(json.dumps({"blocks": [old]}))
    monkeypatch.setattr(daily_focus, "ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(daily_focus, "focus_blocks", lambda _: [
        {"start_utc": "2026-09-28T14:00:00+00:00",
         "end_utc": "2026-09-28T14:05:00+00:00", "app": "Browser"},
        {key: old[key] for key in ("start_utc", "end_utc", "app")}])
    monkeypatch.setattr(daily_focus, "analyze", lambda day, now=None: {
        "day_local": day.isoformat(), "generated_at_utc": "2026-09-29T00:00:00+00:00",
        "complete_day": True, "mac": {"segments": [], "observed_sampled_seconds": 0,
                                      "unobserved_seconds": 0},
        "iphone": {"device_union_seconds": 0, "app_totals": []},
        "source_quality": {"iphone_sync": {}},
    })
    blocks = daily_focus.build(day)["blocks"]
    assert blocks[1]["id"] == "mac-block-0001"
    assert blocks[0]["id"].startswith("mac-") and blocks[0]["id"] != blocks[1]["id"]


def test_short_inactive_interval_breaks_focus_even_within_gap_allowance(monkeypatch):
    monkeypatch.setattr(daily_focus, "visual_evidence", lambda *_: (0, []))
    start = datetime(2026, 10, 3, 14, tzinfo=timezone.utc)
    for state in ("idle", "locked", "unattributed"):
        rows = [{"state": s, "app": "Editor", "window": "Draft",
                 "start_utc": (start + timedelta(seconds=at)).isoformat(),
                 "end_utc": (start + timedelta(seconds=at + duration)).isoformat(),
                 "sampled_seconds": duration}
                for at, duration, s in [(0, 100, "active"), (100, 5, state), (105, 100, "active")]]
        blocks = daily_focus.focus_blocks(rows)
        assert len(blocks) == 2
        assert sum(b["sampled_seconds"] for b in blocks) == 200


def test_visual_evidence_excludes_another_apps_screenshots(tmp_path, monkeypatch):
    dbpath = tmp_path / "ledger.sqlite3"
    start = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    with sqlite3.connect(dbpath) as db:
        db.executescript("CREATE TABLE screenshots(path TEXT,timestamp_utc TEXT,active_window TEXT,ocr_status TEXT,active_app TEXT);"
                        "CREATE TABLE vision_descriptions(path TEXT,status TEXT,description TEXT,model_sha256 TEXT,screenshot_sha256 TEXT,prompt_version TEXT);")
        for identity, app in (("editor", "Editor"), ("browser", "Browser")):
            db.execute("INSERT INTO screenshots VALUES (?,?,?,?,?)", (identity, start.isoformat(), "Window", "complete", app))
            db.execute("INSERT INTO vision_descriptions VALUES (?,?,?,?,?,?)", (identity, "complete", "Context", "9b", "sha", "primary"))
    monkeypatch.setattr(daily_focus, "DB_PATH", dbpath)
    monkeypatch.setattr(daily_focus, "stronger_model_sha", lambda: "9b")
    count, evidence = daily_focus.visual_evidence(start, start + timedelta(seconds=30), "Editor")
    assert count == 1
    assert evidence[0]["path"] == "editor"
