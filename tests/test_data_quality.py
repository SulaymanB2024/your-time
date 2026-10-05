from datetime import datetime, timedelta, timezone

from daily_analysis import mac_segments, phone_analysis
from data_quality import check_analysis


def valid_report():
    start = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    end = start + timedelta(seconds=10)
    rows = [{"source": "mac_window_sample", "timestamp_utc": start.isoformat(),
             "duration_seconds": 5, "data": {"app": "Editor", "title": "Draft"}},
            {"source": "mac_idle", "timestamp_utc": (start + timedelta(seconds=5)).isoformat(),
             "duration_seconds": 5, "data": {}}]
    segments, _ = mac_segments(rows, start, end)
    return {"start_utc": start.isoformat(), "analyzed_through_utc": end.isoformat(),
            "mac": {"state_sampled_seconds": {"active": 5, "idle": 5},
                    "observed_sampled_seconds": 10, "unobserved_seconds": 0,
                    "app_sampled_seconds": [{"app": "Editor", "seconds": 5}], "segments": segments},
            "iphone": phone_analysis([], start, end),
            "visual": {"indexed_screenshots": 1, "ocr_status_counts": {"complete": 1}}}


def test_reconciles_real_interval_transforms_without_exposing_captured_bodies():
    report = valid_report()
    result = check_analysis(report, workstreams=[{"sampled_seconds": 5}],
                            screen_workstreams=[], focus_blocks=[{"sampled_seconds": 5}])
    assert result["status"] == "passed"
    assert result["checks_run"] >= 20
    assert "Draft" not in str(result) and "Editor" not in str(result)


def test_detects_overlap_accounting_drift_and_impossible_device_time():
    report = valid_report()
    report["mac"]["segments"][1]["start_utc"] = report["start_utc"]
    report["mac"]["observed_sampled_seconds"] = 12
    report["iphone"]["device_union_seconds"] = 11
    result = check_analysis(report)
    codes = {x["code"] for x in result["violations"]}
    assert {"mac_segment_overlap", "mac_elapsed_reconciliation", "phone_time_bound",
            "phone_union_reconciliation"} <= codes


def test_rejects_topic_overallocation_and_nonfinite_values():
    report = valid_report()
    result = check_analysis(report, workstreams=[{"sampled_seconds": 8}],
                            screen_workstreams=[{"sampled_seconds": 1}])
    assert {x["code"] for x in result["violations"]} == {
        "topic_time_reconciliation", "suggested_screen_time_bound"}
    report["mac"]["unobserved_seconds"] = float("nan")
    assert "nonnegative_finite_time" in {x["code"] for x in check_analysis(report)["violations"]}


def test_context_distribution_and_sustained_time_cannot_exceed_observations():
    from behavior_analysis import build
    report = valid_report()
    behavior = build(report["mac"]["segments"], [])
    assert check_analysis(report, behavior=behavior)["status"] == "passed"
    behavior["distribution"][0]["sampled_seconds"] += 10
    behavior["sustained_seconds"] = 15
    violations = check_analysis(report, behavior=behavior)["violations"]
    assert {row["code"] for row in violations} == {
        "context_distribution_reconciliation", "sustained_context_bound"}
