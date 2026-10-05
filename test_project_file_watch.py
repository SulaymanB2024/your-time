from pathlib import Path

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
