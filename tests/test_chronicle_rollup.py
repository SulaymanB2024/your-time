import json
from datetime import date

import chronicle_rollup
import secure_store


def test_work_files_exclude_generated_churn_without_deleting_events(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(chronicle_rollup, "DB_PATH", tmp_path / "ledger.sqlite3")
    at = "2026-10-03T15:00:00+00:00"
    with secure_store.connect() as db:
        db.executemany("INSERT INTO project_file_events VALUES (?,?,?,?,?)", [
            ("a", at, "/example", "example/app.py", "modified"),
            ("b", at, "/example", "example/app.py", "modified"),
            ("c", at, "/example", "example/out/index.html", "created"),
            ("d", at, "/example", "example/state/run.sqlite3-wal", "modified"),
        ])
    result = chronicle_rollup.work_artifacts(date(2026, 10, 3))
    assert result["file_events"] == 4
    assert result["changed_files"] == 1
    assert result["work_file_events"] == 2
    assert result["generated_file_events_excluded"] == 2


def test_local_day_lengths_include_dst_transitions():
    spring = chronicle_rollup.local_boundaries(date(2026, 3, 8))
    fall = chronicle_rollup.local_boundaries(date(2026, 11, 1))
    assert (spring[1] - spring[0]).total_seconds() == 23 * 3600
    assert (fall[1] - fall[0]).total_seconds() == 25 * 3600


def test_month_aggregation_keeps_devices_separate_and_unknown_visible():
    def day(label, mac, phone, observed):
        return {"day_local": label, "complete_day": True,
                "coverage_status": "mac_observed" if observed else "no_mac_samples",
                "mac": {"active_seconds": mac, "observed_seconds": observed,
                        "unknown_seconds": 86400 - observed,
                        "unattributed_seconds": max(0, observed - mac),
                        "app_only_seconds": min(100, max(0, observed - mac)),
                        "app_totals": ([{"app": "Editor", "seconds": mac},
                                        {"app": "Browser", "seconds": min(100, max(0, observed - mac))}]
                                       if observed else [])},
                "iphone": {"focus_union_seconds": phone,
                           "interval_count": 1 if phone else 0,
                           "app_totals": [{"app": "Phone", "focus_seconds": phone}] if phone else []},
                "visual": {"screenshots": 2, "distinct_captioned_screenshots": 1},
                "screen_workstreams": [{"id": "planning", "label": "Planning",
                                         "sampled_seconds": 90}] if observed else [],
                "workstreams": [{"id": "draft", "label": "Draft", "sampled_seconds": mac}]
                if observed else []}
    result = chronicle_rollup.aggregate([
        day("2026-09-28", 3600, 7200, 4000),
        day("2026-09-29", 0, 1800, 0)], period="month", key="2026-09")
    assert result["mac_active_seconds"] == 3600
    assert result["mac_unattributed_seconds"] == 400
    assert result["mac_app_only_seconds"] == 100
    assert result["mac_foreground_seconds"] == 4000
    assert result["mac_screen_suggested_seconds"] == 90
    assert result["screen_workstreams"][0]["sampled_seconds"] == 90
    assert result["iphone_focus_seconds"] == 9000
    assert result["mac_days_observed"] == 1
    assert result["mac_unknown_seconds"] > 0
    assert "combined_seconds" not in result


def test_private_rollup_write_is_idempotent(tmp_path):
    target = tmp_path / "days" / "2026-09-29.json"
    assert chronicle_rollup.private_write_if_changed(target, {"day": "2026-09-29"})
    assert not chronicle_rollup.private_write_if_changed(target, {"day": "2026-09-29"})
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700


def test_calendar_summary_keeps_planned_time_separate_and_skips_all_day(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(chronicle_rollup, "STATE_DIR", tmp_path)
    monkeypatch.setattr(chronicle_rollup, "DB_PATH", tmp_path / "ledger.sqlite3")
    (tmp_path / "calendar-scope.json").write_text(json.dumps({
        "events": [{"id": "work", "includeTitle": False}], "reminders": []}))
    with secure_store.connect() as database:
        database.executemany(
            "INSERT INTO calendar_items "
            "(id,source,calendar_id,start_utc,end_utc,title,completed,observed_at_utc,all_day) "
            "VALUES (?,?,?,?,?,?,?,?,?)", [
                ("a", "event", "work", "2026-09-29T15:00:00+00:00",
                 "2026-09-29T16:00:00+00:00", None, None, "2026-09-29T17:00:00+00:00", 0),
                ("b", "event", "work", "2026-09-29T17:00:00+00:00",
                 "2026-09-29T18:00:00+00:00", None, None, "2026-09-29T18:00:00+00:00", 1),
            ])
    result = chronicle_rollup.calendar_summary(date(2026, 9, 29))
    assert result["enabled"]
    assert result["scheduled_seconds"] == 3600
    assert result["event_count"] == 2


def test_file_identity_includes_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(chronicle_rollup, "DB_PATH", tmp_path / "ledger.sqlite3")
    with secure_store.connect() as db:
        db.executemany("INSERT INTO project_file_events VALUES (?,?,?,?,?)", [
            ("a", "2026-10-03T15:00:00+00:00", "/project-one", "app.py", "modified"),
            ("b", "2026-10-03T15:00:00+00:00", "/project-two", "app.py", "modified")])
    result = chronicle_rollup.work_artifacts(date(2026, 10, 3))
    assert result["changed_files"] == 2
    assert result["work_file_events"] == 2


def test_topic_identity_preserves_punctuation_case_nonascii_and_evidence_tier():
    labels = ["C++ development", "C# development", "研究", "Other", "other", "x" * 80, "x" * 80 + "y"]
    assert len({chronicle_rollup.workstream_id(label) for label in labels}) == len(labels)
    assert chronicle_rollup.workstream_id("Planning", "specific_model") != chronicle_rollup.workstream_id("Planning", "broad_context")
    assert chronicle_rollup.workstream_id("Planning") == chronicle_rollup.workstream_id("Planning")


def test_historical_dependency_identity_changes_on_correction_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'DB_PATH', tmp_path / 'ledger.sqlite3')
    monkeypatch.setattr(chronicle_rollup, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(chronicle_rollup, 'DB_PATH', tmp_path / 'ledger.sqlite3')
    with secure_store.connect():
        pass
    day = date(2026, 9, 1)
    first = chronicle_rollup.source_dependencies(day)
    with secure_store.connect() as db:
        db.execute('INSERT INTO task_corrections VALUES (?,?,?,?,?,?,?)',
                   ('label', day.isoformat(), '2026-09-01T12:00:00+00:00',
                    '2026-09-01T12:07:00+00:00', 'Drafting', '2026-10-05T00:00:00+00:00',
                    'user_confirmed_label'))
    second = chronicle_rollup.source_dependencies(day)
    assert first != second
    with secure_store.connect() as db:
        db.execute("UPDATE task_corrections SET evidence_tier='retracted' WHERE id='label'")
    assert chronicle_rollup.source_dependencies(day) != second
