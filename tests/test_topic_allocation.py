import topic_allocation
from window_topic_tagging import window_id


def test_specific_and_broad_time_reconcile_without_guessing_subject():
    segments = [
        {"state": "active", "app": "Chrome", "window": "Chess tutorial", "sampled_seconds": 120},
        {"state": "active", "app": "Chrome", "window": "YouTube", "sampled_seconds": 80},
        {"state": "active", "app": "Chrome", "window": None, "sampled_seconds": 50},
        {"state": "idle", "app": None, "window": None, "sampled_seconds": 40},
    ]
    tags = [{"id": window_id("Chrome", "Chess tutorial"), "topic": "Chess tutorial",
             "status": "model_inference"}]
    result = topic_allocation.allocate(segments, tags)
    assert sum(item["sampled_seconds"] for item in result) == 250
    assert any(item["label"] == "Chess tutorial" and item["status"] == "specific_model"
               for item in result)
    assert any(item["label"] == "Video site (subject unknown)" for item in result)
    assert any(item["label"] == "Browser page (topic unknown)" for item in result)


def test_observed_browser_domain_names_context_without_inventing_task():
    rows = topic_allocation.allocate([
        {"state": "active", "app": "com.google.Chrome", "window": "New Tab",
         "site_host": "example.org", "sampled_seconds": 45}], [])
    assert rows == [{"label": "example.org (topic unknown)",
                     "status": "broad_context", "sampled_seconds": 45}]
