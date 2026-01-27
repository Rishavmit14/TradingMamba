"""User models for the trading system"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class UserAlertConfig:
    """User's alert configuration preferences"""

    # Notification channels
    whatsapp_enabled: bool = True
    email_enabled: bool = False
    push_enabled: bool = True

    # Timeframes to receive alerts for
    timeframes: Dict[str, bool] = field(default_factory=lambda: {
        "M15": False,
        "H1": True,
        "H4": True,
        "D1": True,
        "W1": True,
        "MN": False,
    })

    # Minimum confidence to trigger alert
    min_confidence: float = 0.65

    # Alert types
    alert_types: Dict[str, bool] = field(default_factory=lambda: {
        "new_signal": True,
        "signal_triggered": True,
        "tp_hit": True,
        "sl_hit": True,
        "daily_summary": True,
        "weekly_report": True,
    })

    # Quiet hours (no notifications)
    quiet_hours_enabled: bool = False
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "07:00"
    timezone: str = "UTC"


@dataclass
class User:
    """User of the trading system"""

    id: str = field(default_factory=lambda: str(uuid4()))
    email: str = ""
    username: str = ""
    password_hash: str = ""

    # Contact info
    whatsapp_number: Optional[str] = None
    phone_number: Optional[str] = None

    # Watchlist
    watchlist: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NAS100"
    ])

    # Alert preferences
    alert_config: UserAlertConfig = field(default_factory=UserAlertConfig)

    # Account status
    is_active: bool = True
    is_verified: bool = False
    subscription_tier: str = "free"  # free, basic, pro

    # Stats
    signals_received: int = 0
    last_active: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def can_receive_alert(self, timeframe: str, confidence: float) -> bool:
        """Check if user should receive an alert"""
        if not self.is_active:
            return False

        if confidence < self.alert_config.min_confidence:
            return False

        if timeframe not in self.alert_config.timeframes:
            return False

        if not self.alert_config.timeframes.get(timeframe, False):
            return False

        # TODO: Check quiet hours

        return True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "watchlist": self.watchlist,
            "subscription_tier": self.subscription_tier,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }
