"""One local-clock policy for vision, text and capacity estimates."""

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/Chicago")
START = time(0, 30)
VISION_END = time(7)
END = time(8)
TOTAL_SECONDS = int(7.5 * 3600)
VISION_SECONDS = int(6.5 * 3600)
TEXT_SECONDS = 3600


def in_window(now: datetime, *, start: time = START, end: time = END) -> bool:
    return start <= now.astimezone(ZONE).time() < end


def remaining_seconds(now: datetime, *, start: time = START, end: time = END) -> float:
    if not in_window(now, start=start, end=end):
        return 0.0
    local = now.astimezone(ZONE)
    finish = datetime.combine(local.date(), end, tzinfo=ZONE).astimezone(timezone.utc)
    # Subtract UTC instants: DST makes a local-clock window vary in real duration.
    return max(0.0, (finish - now.astimezone(timezone.utc)).total_seconds())


def text_budget(maximum: float, now: datetime | None = None) -> float:
    return min(maximum, max(0, remaining_seconds(now or datetime.now(timezone.utc),
                                               start=VISION_END) - 20))
