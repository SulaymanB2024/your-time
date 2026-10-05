import io
import struct
from datetime import datetime, timezone

import browser_bridge


def test_native_protocol_and_tab_event_strip_full_url(monkeypatch):
    sink = io.BytesIO()
    browser_bridge.write_message(sink, {"kind": "list_review", "day": "2026-09-29"})
    assert browser_bridge.read_message(io.BytesIO(sink.getvalue())) == {
        "kind": "list_review", "day": "2026-09-29"}
    captured = []
    monkeypatch.setattr(browser_bridge, "insert_events", lambda rows: captured.extend(rows) or 1)
    monkeypatch.setattr(browser_bridge, "write_status", lambda **kwargs: None)
    message = {"kind": "tab_event", "event_type": "tab_active",
               "at": datetime.now(timezone.utc).isoformat(),
               "domain": "example.com", "title": "Example project", "audible": False,
               "incognito": False, "tab_id": 12, "window_id": 2}
    assert browser_bridge.handle(message) == {"ok": True, "stored": True}
    assert captured[0]["data"]["site_host"] == "example.com"
    assert "url" not in captured[0]["data"]
    assert browser_bridge.handle({**message, "incognito": True}) == {"ok": True, "stored": False}
    assert len(captured) == 1


def test_native_protocol_rejects_oversized_input():
    data = struct.pack('<I', browser_bridge.MAX_MESSAGE + 1)
    try:
        browser_bridge.read_message(io.BytesIO(data))
    except ValueError:
        pass
    else:
        raise AssertionError('oversized message accepted')
