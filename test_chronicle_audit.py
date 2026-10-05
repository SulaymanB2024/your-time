import json

from chronicle_audit import day_metrics


def test_audit_reports_partial_coverage_without_private_bodies_or_device_sum():
    row = day_metrics({
        "day_local": "2026-10-04", "complete_day": False,
        "start_utc": "2026-10-04T05:00:00+00:00",
        "analyzed_through_utc": "2026-10-04T06:00:00+00:00",
        "mac": {"state_sampled_seconds": {"active": 600, "unattributed": 300,
                                             "idle": 900, "locked": 600},
                "observed_sampled_seconds": 2400, "unobserved_seconds": 1200,
                "gaps_over_one_minute": [{"seconds": 1200}],
                "segments": [{"window": "PRIVATE WINDOW BODY"}]},
        "iphone": {"device_union_seconds": 900, "overlap_seconds": 10,
                   "sessions": [{"app": "PRIVATE APP BODY"}]},
        "visual": {"indexed_screenshots": 50, "ocr_status_counts": {"complete": 48}},
    })
    assert row["foreground_seconds"] == 900
    assert row["window_identified_fraction"] == .6667
    assert row["mac_observed_fraction"] == .6667
    assert row["mac_unknown_seconds"] == 1200
    assert "PRIVATE" not in json.dumps(row)
    assert "total_screen_time" not in row
