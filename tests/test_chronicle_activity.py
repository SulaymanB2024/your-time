from datetime import date, datetime, timezone

import chronicle_activity as activity
from activity_episode_store import build_episodes, supports_anchor
from daily_analysis import mac_segments, observed_mac_runs


def fixture():
    return [dict(timestamp_utc=f"2026-10-03T12:00:{second:02d}+00:00",
                 source="mac_window_sample", duration_seconds=5,
                 data={"app": "Editor", "title": ""}) for second in (0, 6)]


def test_source_jitter_gaps_are_preserved_and_time_reconciles():
    start = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    end = datetime(2026, 10, 3, 13, tzinfo=timezone.utc)
    runs = observed_mac_runs(fixture(), start, end)
    merged, _ = mac_segments(fixture(), start, end)
    episodes = build_episodes(runs)
    assert len(merged) == 1 and len(episodes) == 2
    assert sum(e['sampled_seconds'] for e in episodes) == merged[0]['sampled_seconds'] == 10
    assert runs[0]['end_utc'] < runs[1]['start_utc']


def test_production_bridge_groups_jitter_without_gap_support():
    episodes = activity.episodes_for_day(date(2026, 10, 3), rows=fixture())
    assert len(episodes) == 1 and episodes[0]['sampled_seconds'] == 10
    assert supports_anchor(episodes[0], '2026-10-03T12:00:06+00:00')
    assert not supports_anchor(episodes[0], '2026-10-03T12:00:05.5+00:00')


def test_episode_summary_allocates_current_user_label_only_once(tmp_path, monkeypatch):
    start = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
    end = datetime(2026, 10, 3, 13, tzinfo=timezone.utc)
    episodes = build_episodes(observed_mac_runs(fixture(), start, end))
    monkeypatch.setattr(activity, 'episodes_for_day', lambda *a, **k: episodes)
    labels = [dict(id='label', start_utc=start.isoformat(), end_utc='2026-10-03T12:00:08+00:00',
                   label='Planning', evidence_tier='user_confirmed_label', created_at_utc='revision')]
    monkeypatch.setattr(activity, 'current_intervals', lambda day: [dict(row) for row in labels])
    first = activity.day_activity(date(2026, 10, 3))
    assert first['totals']['allocated_foreground_seconds'] == 7
    assert first['totals']['unknown_foreground_seconds'] == 3
    assert first['model_allocation_enabled'] is False
    assert first['confirmed_outcomes'] == []
    labels.clear()
    second = activity.day_activity(date(2026, 10, 3))
    assert second['totals']['allocated_foreground_seconds'] == 0
    assert second['totals']['unknown_foreground_seconds'] == 10
    assert first['dependencies_sha256'] != second['dependencies_sha256']


def test_persistence_refuses_conflicting_screen_metadata(monkeypatch):
    sample = dict(start_utc='2026-10-03T12:00:00+00:00', end_utc='2026-10-03T12:00:05+00:00',
                  state='active', sampled_seconds=5, app='Editor', window='Current')
    monkeypatch.setattr(activity, 'episodes_for_day', lambda day: build_episodes([sample]))
    context = {'evidence': [{'id': 'screen', 'source': 'screen_context',
                            'timestamp_utc': '2026-10-03T12:00:02+00:00',
                            'app': 'Editor', 'window': 'Old'}]}
    result = activity.persist_result(context=context, result={}, screenshot_sha256='a'*64,
                                    input_sha256='b'*64, engine_sha256='c'*64,
                                    prompt_sha256='d'*64, model_sha256='e'*64)
    assert result['status'] == 'context_disagrees_with_collector'
