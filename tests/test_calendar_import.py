from datetime import datetime, timezone

import calendar_import


def test_scope_filters_events_and_health_titles():
    scope = {"events": [{"id": "work", "includeTitle": True}],
             "reminders": [{"id": "tasks", "includeTitle": False}]}
    payload = {"events": [
        {"id": "one", "calendarId": "work", "start": "2026-09-29T10:00:00Z",
         "end": "2026-09-29T11:00:00Z", "title": "Project review"},
        {"id": "two", "calendarId": "work", "start": "2026-09-29T12:00:00Z",
         "end": "2026-09-29T13:00:00Z", "title": "Doctor visit"},
        {"id": "three", "calendarId": "personal", "start": "2026-09-29T12:00:00Z",
         "end": "2026-09-29T13:00:00Z", "title": "Private"}],
        "reminders": [{"id": "four", "calendarId": "tasks", "at": "2026-09-29T14:00:00Z",
                       "completed": True, "title": "Sensitive task"}]}
    rows, counts = calendar_import.rows_from_export(scope, payload,
        datetime(2026, 9, 30, tzinfo=timezone.utc))
    assert len(rows) == 2
    assert rows[0][5] == "Project review"
    assert rows[1][5] is None
    assert rows[1][6] == 1
    assert counts == {"skipped_health_titles": 1, "skipped_unscoped": 1}
