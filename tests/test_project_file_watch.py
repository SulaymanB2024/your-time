import sqlite3
from contextlib import contextmanager

from watchdog.events import FileModifiedEvent

import project_file_watch
import secure_store


def test_scoped_file_events_are_deduped_and_content_free(tmp_path, monkeypatch):
    monkeypatch.setattr(project_file_watch, "ROOT", tmp_path)
    monkeypatch.setattr(project_file_watch, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    project = tmp_path / "CodexWork" / "example"
    project.mkdir(parents=True)
    source = project / "app.py"
    source.write_text("private source body")
    ignored = project / "node_modules" / "dep.js"
    ignored.parent.mkdir()
    ignored.write_text("dependency")
    handler = project_file_watch.QueuedEvents()
    handler.on_any_event(FileModifiedEvent(str(source)))
    handler.on_any_event(FileModifiedEvent(str(source)))
    handler.on_any_event(FileModifiedEvent(str(ignored)))
    for generated in ["out/index.html", "public/static-pages/index.html", "state/runtime.sqlite3-wal"]:
        handler.on_any_event(FileModifiedEvent(str(project / generated)))
    result = project_file_watch.flush(handler)
    assert result["inserted"] == 1
    with secure_store.connect() as database:
        rows = database.execute("SELECT project_root,relative_path,event_type FROM project_file_events").fetchall()
    assert rows == [(str(project), "CodexWork/example/app.py", "modified")]
    assert "private source body" not in repr(rows)
    assert result["coalesced_since_start"] == 1
    assert result["ignored_generated_since_start"] == 4


def test_failed_commit_keeps_batch_and_new_events_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(project_file_watch, "ROOT", tmp_path)
    monkeypatch.setattr(project_file_watch, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(secure_store, "DB_PATH", tmp_path / "ledger.sqlite3")
    handler = project_file_watch.QueuedEvents()
    first, second = tmp_path/'example/first.py', tmp_path/'example/second.py'
    handler.on_any_event(FileModifiedEvent(str(first)))
    real_connect = project_file_watch.connect

    @contextmanager
    def fail_on_commit():
        with real_connect() as db:
            yield db
            handler.on_any_event(FileModifiedEvent(str(second)))
            raise sqlite3.OperationalError('private error details')

    monkeypatch.setattr(project_file_watch, 'connect', fail_on_commit)
    result = project_file_watch.flush(handler)
    assert result == {'status': 'retry_database', 'inserted': 0, 'not_written': 2, 'dropped': 0}
    assert len(handler.queue) == 2
    with real_connect() as db:
        assert db.execute('SELECT COUNT(*) FROM project_file_events').fetchone()[0] == 0
    monkeypatch.setattr(project_file_watch, 'connect', real_connect)
    assert project_file_watch.flush(handler)['inserted'] == 2
    assert not handler.queue
    assert project_file_watch.flush(handler)['inserted'] == 0


def test_successful_flush_keeps_concurrent_new_events(tmp_path, monkeypatch):
    monkeypatch.setattr(project_file_watch, 'ROOT', tmp_path)
    monkeypatch.setattr(project_file_watch, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'DB_PATH', tmp_path/'ledger.sqlite3')
    handler = project_file_watch.QueuedEvents()
    handler.on_any_event(FileModifiedEvent(str(tmp_path/'example/first.py')))
    real_connect = project_file_watch.connect

    @contextmanager
    def append_during_write():
        with real_connect() as db:
            yield db
            handler.on_any_event(FileModifiedEvent(str(tmp_path/'example/second.py')))

    monkeypatch.setattr(project_file_watch, 'connect', append_during_write)
    result = project_file_watch.flush(handler)
    assert result['inserted'] == 1 and result['not_written'] == 1
    assert len(handler.queue) == 1 and handler.queue[0][2] == 'example/second.py'
    monkeypatch.setattr(project_file_watch, 'connect', real_connect)
    assert project_file_watch.flush(handler)['inserted'] == 1


def test_symlinked_parent_cannot_escape_approved_root(tmp_path, monkeypatch):
    root = tmp_path/'approved'
    root.mkdir()
    outside = tmp_path/'outside'
    outside.mkdir()
    (root/'example').symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(project_file_watch, 'ROOT', root)
    assert project_file_watch.allowed_file(str(root/'example/private.py')) is None


def test_empty_flush_does_not_open_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(project_file_watch, 'STATE_DIR', tmp_path)
    def must_not_open():
        raise AssertionError('No queued events require no database transaction')
    monkeypatch.setattr(project_file_watch, 'connect', must_not_open)
    assert project_file_watch.flush(project_file_watch.QueuedEvents())['inserted'] == 0


def test_low_disk_keeps_pending_events_until_writes_resume(tmp_path, monkeypatch):
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(project_file_watch, 'ROOT', tmp_path)
    monkeypatch.setattr(project_file_watch, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'STATE_DIR', tmp_path)
    monkeypatch.setattr(secure_store, 'DB_PATH', tmp_path/'ledger.sqlite3')
    handler = project_file_watch.QueuedEvents()
    handler.on_any_event(FileModifiedEvent(str(tmp_path/'example/file.py')))
    monkeypatch.setattr(project_file_watch.shutil, 'disk_usage', lambda _: usage(100, 99, 1))
    result = project_file_watch.flush(handler)
    assert result['status'] == 'paused_low_disk' and result['not_written'] == 1
    assert len(handler.queue) == 1
    monkeypatch.setattr(project_file_watch.shutil, 'disk_usage', lambda _: usage(10**12, 1, 10**12-1))
    assert project_file_watch.flush(handler)['inserted'] == 1
