from datetime import date, datetime, timedelta, timezone

import secure_store
import task_corrections


def test_user_task_label_is_bounded_to_unknown_samples_and_retractable(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(task_corrections, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(task_corrections, "read_supported_tags", lambda _: [])
    day = date(2026, 9, 29)
    start = datetime(2026, 9, 29, 5, tzinfo=timezone.utc)
    report = {"start_utc": start.isoformat(),
              "analyzed_through_utc": (start + timedelta(minutes=30)).isoformat(),
              "mac": {"segments": [{"state": "unattributed",
                                     "start_utc": start.isoformat(),
                                     "end_utc": (start + timedelta(minutes=20)).isoformat(),
                                     "sampled_seconds": 1200}]}}
    monkeypatch.setattr(task_corrections, "analyze", lambda _: report)
    intervals = task_corrections.review(day)
    assert len(intervals) == 2
    assert [item["unknown_seconds"] for item in intervals] == [900, 300]
    result = task_corrections.set_label(day, intervals[0]["id"], "SEO planning")
    assert result["sampled_seconds"] == 900
    assert task_corrections.summary(day) == [
        {"label": "SEO planning", "sampled_seconds": 900, "status": "user_confirmed_label"}]
    outcome = task_corrections.mark_outcome(day, intervals[0]["id"])
    assert outcome["evidence_tier"] == "user_confirmed_result"
    assert len(task_corrections.active_outcomes(day)) == 1
    assert task_corrections.clear_label(day, intervals[0]["id"])
    assert task_corrections.summary(day) == []
    assert task_corrections.active_outcomes(day) == []
    task_corrections.set_label(day, intervals[0]["id"], "SEO planning")
    assert task_corrections.summary(day)[0]["sampled_seconds"] == 900
    task_corrections.mark_outcome(day, intervals[0]["id"])
    assert task_corrections.retract_outcome(day, intervals[0]["id"])
    assert task_corrections.active_outcomes(day) == []


def test_confirmed_scope_does_not_extend_and_gaps_do_not_receive_time(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'DB_PATH', tmp_path/'ledger.sqlite3')
    monkeypatch.setattr(task_corrections, 'DB_PATH', tmp_path/'ledger.sqlite3')
    day = date(2026, 10, 3)
    start = datetime(2026, 10, 3, 5, tzinfo=timezone.utc)
    segment = {'state': 'unattributed', 'start_utc': start.isoformat(),
               'end_utc': (start+timedelta(minutes=12)).isoformat(), 'sampled_seconds': 600,
               'support_runs': [{'start_utc': start.isoformat(), 'end_utc': (start+timedelta(minutes=6)).isoformat()},
                                {'start_utc': (start+timedelta(minutes=8)).isoformat(), 'end_utc': (start+timedelta(minutes=12)).isoformat()}]}
    report = {'start_utc': start.isoformat(), 'analyzed_through_utc': (start+timedelta(minutes=12)).isoformat(),
              'mac': {'segments': [segment]}}
    monkeypatch.setattr(task_corrections, 'analyze', lambda _: report)
    identity = task_corrections.slot_id(day, start)
    with secure_store.connect() as db:
        db.execute('INSERT INTO task_corrections VALUES (?,?,?,?,?,?,?)',
                   (identity, day.isoformat(), start.isoformat(), (start+timedelta(minutes=7)).isoformat(),
                    'Drafting', start.isoformat(), 'user_confirmed_label'))
    assert task_corrections.summary(day)[0]['sampled_seconds'] == 360
    updated = task_corrections.set_label(day, identity, 'Planning')
    assert updated['sampled_seconds'] == 360
    assert updated['end_utc'] == (start+timedelta(minutes=7)).isoformat()


def test_review_separates_confirmed_cutoff_from_new_unconfirmed_time(tmp_path, monkeypatch):
    monkeypatch.setattr(secure_store, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'DB_PATH', tmp_path/'ledger.sqlite3')
    monkeypatch.setattr(task_corrections, 'DB_PATH', tmp_path/'ledger.sqlite3')
    monkeypatch.setattr(task_corrections, 'read_supported_tags', lambda _: [])
    day=date(2026,10,3); start=datetime(2026,10,3,5,tzinfo=timezone.utc)
    report={'start_utc':start.isoformat(),'analyzed_through_utc':(start+timedelta(minutes=10)).isoformat(),
            'mac':{'segments':[{'state':'unattributed','start_utc':start.isoformat(),
                                'end_utc':(start+timedelta(minutes=10)).isoformat(),'sampled_seconds':600}]}}
    with secure_store.connect() as db:
        db.execute('INSERT INTO task_corrections VALUES (?,?,?,?,?,?,?)',
                   (task_corrections.slot_id(day,start),day.isoformat(),start.isoformat(),
                    (start+timedelta(minutes=7)).isoformat(),'Planning',start.isoformat(),'user_confirmed_label'))
    row=task_corrections.review(day,report)[0]
    assert row['confirmed_label'] is None
    assert row['confirmed_scope']['sampled_seconds'] == 420
    assert row['unconfirmed_seconds'] == 180
