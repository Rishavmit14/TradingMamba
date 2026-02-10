"""1.9 — Session & Kill Zone Awareness (Crypto-Adapted)

Adds time-of-day context for crypto markets (24/7, no traditional forex sessions).

Crypto adaptations:
- US equity open (~13:30 UTC) = highest BTC volatility
- US equity close (~20:00 UTC) = secondary activity spike
- CME BTC futures open/close = institutional participation
- Asian hours (00:00-08:00 UTC) = lower volatility, range building
- Weekly candle close (Sunday midnight UTC) = structural significance
"""

from __future__ import annotations
from datetime import datetime, timezone
from app.models import Session
from app.config import CRYPTO_SESSIONS, KILL_ZONES


def get_current_session(timestamp: int | None = None) -> Session:
    """Determine the current market session and kill zone status.

    Args:
        timestamp: Unix timestamp in milliseconds. If None, uses current time.

    Returns:
        Session with name, kill_zone flag, and volatility expectation.
    """
    if timestamp:
        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    hour = dt.hour

    # Determine session
    session_name = "late"  # default
    for name, hours in CRYPTO_SESSIONS.items():
        if hours["start"] <= hour < hours["end"]:
            session_name = name
            break

    # Check kill zones
    is_kill_zone = False
    for kz_name, kz_hours in KILL_ZONES.items():
        if kz_hours["start"] <= hour < kz_hours["end"]:
            is_kill_zone = True
            break

    # Volatility expectation
    vol_map = {
        "asian": "low",
        "london": "medium",
        "ny": "high",
        "late": "low",
    }
    volatility = vol_map.get(session_name, "medium")

    return Session(
        name=session_name,
        is_kill_zone=is_kill_zone,
        volatility_expectation=volatility,
    )


def is_weekly_close_window(timestamp: int) -> bool:
    """Check if we're near the weekly candle close (Sunday ~23:00-00:00 UTC).

    Weekly close has structural significance — swing points often form here.
    """
    dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    # Sunday = 6 in Python's weekday()
    return dt.weekday() == 6 and dt.hour >= 22
