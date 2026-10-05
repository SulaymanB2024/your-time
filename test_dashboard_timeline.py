from dashboard_timeline import build_timeline


def interval(start, end, **kwargs):
    return dict(start_utc=f"2026-10-04T14:{start}:00+00:00",
                end_utc=f"2026-10-04T14:{end}:00+00:00", **kwargs)


def test_phone_overlap_is_partitioned_and_not_double_counted():
    result = build_timeline({"mac": {"segments": []}, "iphone": {"sessions": [
        interval("00", "30", app="A"), interval("15", "45", app="B"),
        interval("20", "25", app="A")]}}, {}, lambda app: app)
    phone = result["iphone"]
    assert sum(row["sampled_seconds"] for row in phone) == 2700
    assert [row["apps"] for row in phone] == [["A"], ["A", "B"], ["B"]]
    assert phone[1]["label"] == "Overlapping app records"


def test_timeline_retains_idle_gap_unknown_state_and_model_boundary():
    first = interval("00", "05", label="Project", status="specific_model", sampled_seconds=300)
    second = interval("10", "15", label="Project", status="specific_model", sampled_seconds=300)
    analysis = {"mac": {"segments": [
        interval("05", "10", state="idle", sampled_seconds=300),
        interval("15", "18", state="unattributed", sampled_seconds=180)]},
        "iphone": {"sessions": []}}
    mac = build_timeline(analysis, {"sessions": [first, second]}, lambda app: app)["mac"]
    assert len(mac) == 3
    assert mac[0]["end_utc"] != mac[1]["start_utc"]
    assert mac[0]["status"] == "specific_model"
    assert mac[-1]["status"] == "unknown"
    assert mac[-1]["sampled_seconds"] == 180


def test_phone_repeated_clock_hour_uses_real_utc_elapsed_time():
    sessions = [{"start_utc": "2026-11-01T06:00:00+00:00",
                 "end_utc": "2026-11-01T08:00:00+00:00", "app": "A"}]
    phone = build_timeline({"mac": {"segments": []}, "iphone": {"sessions": sessions}}, {}, str)["iphone"]
    assert len(phone) == 1 and phone[0]["sampled_seconds"] == 7200
