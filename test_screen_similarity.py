import struct
from datetime import datetime, timedelta, timezone

from screen_similarity import propagate


def vector(value):
    return struct.pack('<768f', *([value] * 768))


def row(identity, at, value):
    return {"id": identity, "at": at, "app": "com.example.browser",
            "display": "1", "vector": vector(value)}


def test_only_near_identical_neighbor_inherits_checked_topic():
    start = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    rows = [row("a", start, 0), row("b", start + timedelta(seconds=40), .001),
            row("c", start + timedelta(seconds=70), .1)]
    result = propagate(rows, [{"id": "a", "topic": "Project planning",
                               "status": "model_inference"}])
    assert [item["id"] for item in result["propagated"]] == ["b"]
    assert result["propagated"][0]["status"] == "featureprint_match"


def test_conflicting_checked_labels_do_not_propagate():
    start = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    rows = [row("a", start, 0), row("b", start + timedelta(seconds=20), .001),
            row("c", start + timedelta(seconds=40), .001)]
    result = propagate(rows, [{"id": "a", "topic": "Project planning",
                               "status": "model_inference"},
                              {"id": "b", "topic": "Email", "status": "model_inference"}])
    assert result["conflicting_clusters"] == 1
    assert result["propagated"] == []


def test_similar_frames_cannot_chain_a_label_across_the_day():
    start = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    rows = [row(str(i), start + timedelta(seconds=60 * i), 0) for i in range(8)]
    result = propagate(rows, [{"id": "0", "topic": "Planning", "status": "model_inference"}])
    assert {item["id"] for item in result["propagated"]} == {"1", "2"}


def test_app_switch_and_titled_frame_break_similarity_propagation():
    start = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    tags = [{"id": "a", "topic": "Planning", "status": "model_inference"}]
    for barrier in ({"app": "Other app"}, {"eligible": False}):
        middle = {**row("b", start + timedelta(seconds=20), 0), **barrier}
        result = propagate([row("a", start, 0), middle,
                            row("c", start + timedelta(seconds=40), 0)], tags)
        assert result["propagated"] == []
