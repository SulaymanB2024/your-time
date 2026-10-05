from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class Event:
    timestamp: datetime
    duration: timedelta
    data: dict

from phone_import import select_sessions


def test_sync_gap_is_excluded_without_inventing_usage():
    now = datetime(2026, 9, 28, 18, tzinfo=timezone.utc)
    real = Event(timestamp=now - timedelta(hours=2), duration=timedelta(minutes=12), data={"app": "example"})
    gap = Event(timestamp=now - timedelta(hours=20), duration=timedelta(hours=8), data={"app": "example"})
    accepted, suspect, old = select_sessions([gap, real], now)
    assert accepted == [real]
    assert suspect == 1
    assert old == 0


def test_future_interval_is_excluded():
    now = datetime(2026, 9, 28, 18, tzinfo=timezone.utc)
    future = Event(timestamp=now + timedelta(minutes=10), duration=timedelta(minutes=1), data={"app": "example"})
    accepted, suspect, _ = select_sessions([future], now)
    assert accepted == []
    assert suspect == 1


def test_explicit_all_history_backfill_keeps_old_plausible_session():
    now = datetime(2026, 9, 29, 18, tzinfo=timezone.utc)
    old = Event(timestamp=now - timedelta(days=30), duration=timedelta(minutes=20),
                data={"app": "example"})
    suspect = Event(timestamp=now - timedelta(days=29), duration=timedelta(hours=8),
                    data={"app": "example"})
    recent, _, skipped = select_sessions([old, suspect], now)
    assert recent == [] and skipped == 2
    accepted, bad, skipped = select_sessions([old, suspect], now, lookback=None)
    assert accepted == [old]
    assert bad == 1 and skipped == 0
