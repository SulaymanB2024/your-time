import json

from activity_episode_store import build_episodes
from chronicle_audit import (
    day_metrics,
    observed_frame_reach,
    private_image_digest,
    production_frame_reach,
)


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


def sample(lower, upper, *, window="PRIVATE WINDOW", state="active"):
    from datetime import datetime
    return {"start_utc": lower, "end_utc": upper, "state": state,
            "app": "PRIVATE APP", "window": window,
            "sampled_seconds": (datetime.fromisoformat(upper) - datetime.fromisoformat(lower)).total_seconds()}


def frame(identity, at, **changes):
    return {"observation_id": identity, "timestamp_utc": at, "captured_at_utc": at,
            "app": "PRIVATE APP", "window": "PRIVATE WINDOW",
            "screenshot_sha256": "a" * 64, "current_image_sha256": "a" * 64, **changes}


def test_reach_unions_support_and_deduplicates_observations_not_image_bytes():
    episodes = build_episodes([sample("2026-10-05T15:00:00+00:00", "2026-10-05T15:02:00+00:00")])
    first = frame("first", "2026-10-05T15:00:30+00:00")
    result = observed_frame_reach(episodes, [first, first, frame("second", "2026-10-05T15:01:00+00:00")])
    assert result["completed_unique_capture_observations"] == 2
    assert result["completed_caption_records"] == 3
    assert result["valid_metadata_consistent_anchors"] == 2
    assert result["excluded"] == {"duplicate_observation": 1}
    assert result["covered_collector_episodes"] == 1
    assert result["anchor_support_union_seconds"] == 90
    assert result["foreground_observed_seconds"] == 120
    assert result["semantic_accuracy"] is None
    assert "PRIVATE" not in json.dumps(result)
    assert "first" not in json.dumps(result)


def test_reach_never_bridges_gap_or_credits_conflict_idle_or_changed_image():
    episodes = build_episodes([
        sample("2026-10-05T15:00:00+00:00", "2026-10-05T15:00:05+00:00"),
        sample("2026-10-05T15:00:06+00:00", "2026-10-05T15:00:10+00:00"),
        sample("2026-10-05T15:00:10+00:00", "2026-10-05T15:00:20+00:00", state="idle"),
    ], max_gap_seconds=2)
    result = observed_frame_reach(episodes, [
        frame("valid", "2026-10-05T15:00:04+00:00"),
        frame("gap", "2026-10-05T15:00:05+00:00"),
        frame("idle", "2026-10-05T15:00:10+00:00"),
        frame("conflict", "2026-10-05T15:00:04+00:00", window="PRIVATE DIFFERENT"),
        frame("changed", "2026-10-05T15:00:04+00:00", current_image_sha256="b" * 64),
        frame("missing", "2026-10-05T15:00:04+00:00", current_image_sha256=None),
        frame("timestamp", "2026-10-05T15:00:04+00:00", captured_at_utc="2026-10-05T15:00:03+00:00"),
    ])
    assert result["anchor_support_union_seconds"] == 9
    assert result["valid_metadata_consistent_anchors"] == 1
    assert result["excluded"] == {"no_foreground_anchor": 2, "collector_metadata_conflict": 1,
                                  "image_hash_changed": 1, "image_unavailable": 1,
                                  "observation_timestamp_conflict": 1}


def test_reach_repeated_local_hour_uses_distinct_utc_bins_and_equivalent_timestamps():
    episodes = build_episodes([
        sample("2026-11-01T01:10:00-05:00", "2026-11-01T01:11:00-05:00"),
        sample("2026-11-01T01:10:00-06:00", "2026-11-01T01:11:00-06:00"),
    ])
    result = observed_frame_reach(episodes, [
        frame("first", "2026-11-01T01:10:30-05:00", captured_at_utc="2026-11-01T06:10:30+00:00"),
    ])
    assert result["foreground_half_hour_bins"] == 2
    assert result["covered_half_hour_bins"] == 1
    assert result["occupied_bin_reach_fraction"] == .5
    assert result["excluded"] == {}


def test_anchor_bins_stay_distinct_from_support_reaching_adjacent_bin():
    episodes = build_episodes([sample("2026-10-05T15:29:40+00:00", "2026-10-05T15:30:40+00:00")])
    result = observed_frame_reach(episodes, [frame("edge", "2026-10-05T15:29:50+00:00")])
    assert result["foreground_half_hour_bins"] == result["covered_half_hour_bins"] == 2
    assert result["anchor_half_hour_bins"] == 1
    assert result["occupied_bin_reach_fraction"] == 1
    assert result["occupied_anchor_bin_reach_fraction"] == .5


def test_run_reach_reads_only_selected_identity_source_day_and_completed_run(tmp_path):
    import hashlib
    import sqlite3
    db_path = tmp_path / "private.sqlite"
    image = tmp_path / "private-image.bin"
    image.write_bytes(b"local test bytes")
    image.chmod(0o600)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    db = sqlite3.connect(db_path)
    db.executescript("CREATE TABLE events(timestamp_utc,source,duration_seconds,data_json);"
                    "CREATE TABLE screenshots(path,timestamp_utc,active_app,active_window);"
                    "CREATE TABLE vision_descriptions(path,timestamp_utc,screenshot_sha256,model_sha256,prompt_version,status,updated_at_utc);")
    db.execute("INSERT INTO events VALUES(?,?,?,?)", ("2026-10-05T15:00:00+00:00", "mac_window_sample", 5,
        json.dumps({"app": "PRIVATE APP", "title": "PRIVATE WINDOW"})))
    db.execute("INSERT INTO screenshots VALUES(?,?,?,?)", (str(image), "2026-10-05T15:00:01+00:00", "PRIVATE APP", "PRIVATE WINDOW"))
    for model, prompt, status, completed, captured in [
        ("model", "prompt", "complete", "2026-10-06T07:00:00+00:00", "2026-10-05T15:00:01+00:00"),
        ("other", "prompt", "complete", "2026-10-06T07:00:00+00:00", "2026-10-05T15:00:01+00:00"),
        ("model", "other", "complete", "2026-10-06T07:00:00+00:00", "2026-10-05T15:00:01+00:00"),
        ("model", "prompt", "incomplete", "2026-10-06T07:00:00+00:00", "2026-10-05T15:00:01+00:00"),
        ("model", "prompt", "complete", "2026-10-06T12:00:00+00:00", "2026-10-05T15:00:01+00:00"),
        ("model", "prompt", "complete", "2026-10-06T07:00:00+00:00", "2026-10-04T15:00:01+00:00"),
    ]:
        db.execute("INSERT INTO vision_descriptions VALUES(?,?,?,?,?,?,?)", (str(image), captured, digest, model, prompt, status, completed))
    db.commit()
    db.close()
    receipt = {"day_local": "2026-10-05", "started_at_utc": "2026-10-06T06:00:00+00:00",
               "finished_at_utc": "2026-10-06T11:00:00+00:00"}
    result = production_frame_reach(receipt, model_sha256="model", prompt_version="prompt", db_path=db_path, image_roots=(tmp_path,))
    assert result["completed_unique_capture_observations"] == 1
    assert result["valid_metadata_consistent_anchors"] == 1
    assert result["anchor_support_union_seconds"] == 5
    assert "PRIVATE" not in json.dumps(result)
    assert str(image) not in json.dumps(result)
    assert production_frame_reach({"day_local": "2026-10-05"}, model_sha256="model", prompt_version="prompt", db_path=tmp_path/"absent") == {"status": "pending", "reason": "run_not_finished"}


def test_image_probe_rejects_unapproved_paths_links_nonprivate_and_oversized_files(tmp_path):
    import os
    archive = tmp_path / "archive"
    archive.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    outside.chmod(0o600)
    assert private_image_digest(outside, (archive,)) is None
    image = archive / "capture"
    image.write_bytes(b"private")
    image.chmod(0o600)
    assert private_image_digest(image, (archive,)) is not None
    link = archive / "linked"
    link.symlink_to(image)
    assert private_image_digest(link, (archive,)) is None
    linked_directory = archive / "linked-directory"
    linked_directory.symlink_to(tmp_path, target_is_directory=True)
    assert private_image_digest(linked_directory / outside.name, (archive,)) is None
    hardlink = archive / "hardlinked"
    os.link(image, hardlink)
    assert private_image_digest(image, (archive,)) is None
    hardlink.unlink()
    image.chmod(0o644)
    assert private_image_digest(image, (archive,)) is None
    image.chmod(0o600)
    with image.open("r+b") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    assert private_image_digest(image, (archive,)) is None


def test_image_probe_rejects_fifo_without_waiting_for_writer(tmp_path):
    import os
    import subprocess
    import sys
    pipe = tmp_path / "fifo"
    os.mkfifo(pipe, 0o600)
    subprocess.run([sys.executable, "-B", "-c",
                    "from pathlib import Path; import sys; from chronicle_audit import private_image_digest; "
                    "p=Path(sys.argv[1]); assert private_image_digest(p,(p.parent,)) is None", str(pipe)],
                   check=True, timeout=2, capture_output=True)
