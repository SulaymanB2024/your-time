from datetime import datetime, timedelta, timezone

import behavior_analysis as behavior
from window_topic_tagging import window_id


def segments(rows):
    at = datetime(2026, 10, 4, 14, tzinfo=timezone.utc)
    output = []
    for state, seconds, app, title in rows:
        end = at + timedelta(seconds=seconds)
        output.append(dict(state=state, sampled_seconds=seconds, app=app, window=title,
                           start_utc=at.isoformat(), end_utc=end.isoformat()))
        at = end
    return output


def test_supported_topic_joins_apps_but_exact_windows_do_not():
    rows = segments([('active', 600, 'Editor', 'Orbit notes'),
                     ('active', 700, 'Browser', 'Orbit plan')])
    tags = [dict(id=window_id(row['app'], row['window']), topic='Orbit', status='model_inference')
            for row in rows]
    result = behavior.build(rows, tags)
    assert result['session_count'] == 1
    assert result['sustained_seconds'] == 1300
    assert result['context_switches'] == 0
    assert result['window_switches'] == 1
    assert result['longest_stretch']['apps'] == ['Editor', 'Browser']
    assert behavior.build(rows, [])['session_count'] == 2


def test_every_state_barrier_ends_a_stretch_and_a_return_is_not_a_switch():
    for state in ('idle', 'locked', 'unattributed'):
        rows = segments([('active', 100, 'Editor', 'Draft'), (state, 1, None, None),
                         ('active', 100, 'Editor', 'Draft')])
        result = behavior.build(rows, [])
        assert result['session_count'] == 2
        assert result['context_switches'] == 0
        assert result['identified_seconds'] == 200


def test_jitter_is_not_counted_and_collection_gaps_split_sessions():
    rows = segments([('active', 100, 'Editor', 'Draft'), ('active', 100, 'Editor', 'Draft')])
    second = datetime.fromisoformat(rows[1]['start_utc'])
    rows[1]['start_utc'] = (second + timedelta(seconds=.5)).isoformat()
    rows[1]['end_utc'] = (second + timedelta(seconds=100.5)).isoformat()
    assert behavior.build(rows, [])['longest_stretch_seconds'] == 200
    rows[1]['start_utc'] = (second + timedelta(seconds=2)).isoformat()
    assert behavior.build(rows, [])['session_count'] == 2


def test_distribution_boundaries_reconcile_to_identified_time():
    rows = []
    for index, seconds in enumerate([119, 120, 300, 900, 1800]):
        rows.append(('active', seconds, 'Editor', f'Draft {index}'))
    result = behavior.build(segments(rows), [])
    assert [row['sessions'] for row in result['distribution']] == [1] * 5
    assert sum(row['sampled_seconds'] for row in result['distribution']) == result['identified_seconds']
    assert result['context_switches'] == 4
    assert result['brief_sessions'] == 1
    assert result['sustained_seconds'] == 1800


def test_sensitive_titles_stay_hidden_and_compact_history_excludes_window_bodies():
    result = behavior.build(segments([('active', 90, 'Browser', 'Password private text')]), [])
    assert 'private text' not in str(result)
    compact = behavior.compact(result)
    assert 'sessions' not in compact and 'longest_stretch' not in compact
    assert 'label' not in compact or 'private text' not in str(compact)


def test_period_switch_rate_is_weighted_by_time_and_empty_is_unknown():
    first = behavior.compact(behavior.build(segments([('active', 100, 'Editor', 'One'),
                ('active', 100, 'Editor', 'Two')]), []))
    second = behavior.compact(behavior.build(segments([('active', 1000, 'Editor', 'One')]), []))
    result = behavior.aggregate([{'behavior': first}, {'behavior': second}])
    assert result['context_switches_per_identified_hour'] == 3
    assert result['session_count'] == 3
    assert result['identified_seconds'] == 1200
    assert behavior.aggregate([])['context_switches_per_identified_hour'] is None
