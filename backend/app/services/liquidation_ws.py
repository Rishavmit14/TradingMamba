"""Binance forceOrder WebSocket — real-time forced liquidation stream.

Connects to wss://fstream.binance.com/ws/btcusdt@forceOrder and buffers
recent liquidation events for the Quant microstructure filter.

Usage:
    manager = LiquidationWSManager()
    await manager.start()        # Start in FastAPI lifespan
    liqs = manager.get_recent(30)  # Last 30 minutes
    await manager.stop()         # Stop on shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

WS_URL = "wss://fstream.binance.com/ws/btcusdt@forceOrder"
MAX_EVENTS = 2000
RECONNECT_DELAY = 5  # seconds


class LiquidationWSManager:
    """Manages a WebSocket connection to Binance forceOrder stream."""

    def __init__(self):
        self._events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the WebSocket listener as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())
        logger.info("Liquidation WebSocket manager started")

    async def stop(self):
        """Stop the WebSocket listener."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Liquidation WebSocket manager stopped")

    async def _listen_loop(self):
        """Reconnecting listen loop."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not installed — liquidation WS disabled")
            self._running = False
            return

        while self._running:
            try:
                async with websockets.connect(WS_URL) as ws:
                    logger.info("Connected to Binance forceOrder WebSocket")
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            event = self._parse_event(data)
                            if event:
                                self._events.append(event)
                        except Exception as e:
                            logger.debug("Error parsing liquidation event: %s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Liquidation WS disconnected: %s. Reconnecting in %ds...", e, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    def _parse_event(self, data: dict) -> dict | None:
        """Parse a forceOrder event into our internal format.

        Binance forceOrder format:
        {
            "e": "forceOrder",
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",          # SELL = long liquidated, BUY = short liquidated
                "o": "LIMIT",
                "f": "IOC",
                "q": "0.014",         # Quantity
                "p": "9910.0",        # Price
                "ap": "9910.0",       # Average Price
                "X": "FILLED",
                "l": "0.014",         # Last filled quantity
                "z": "0.014",         # Accumulated filled quantity
                "T": 1568014460893    # Timestamp
            }
        }
        """
        order = data.get("o", {})
        if not order:
            return None

        symbol = order.get("s", "")
        if symbol != "BTCUSDT":
            return None

        side = order.get("S", "").lower()  # "sell" = long liquidated
        price = float(order.get("ap", 0) or order.get("p", 0))
        qty = float(order.get("z", 0) or order.get("q", 0))
        timestamp = order.get("T", int(time.time() * 1000))

        qty_usd = price * qty

        return {
            "timestamp": timestamp,
            "side": side,
            "price": price,
            "qty": qty,
            "qty_usd": qty_usd,
        }

    def get_recent(self, window_minutes: int = 30) -> list[dict]:
        """Get liquidation events from the last N minutes."""
        cutoff = int(time.time() * 1000) - (window_minutes * 60 * 1000)
        return [e for e in self._events if e["timestamp"] >= cutoff]

    def get_all(self) -> list[dict]:
        """Get all buffered liquidation events."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def is_running(self) -> bool:
        return self._running


# Global singleton
_manager: LiquidationWSManager | None = None


def get_liquidation_manager() -> LiquidationWSManager:
    """Get or create the global liquidation WebSocket manager."""
    global _manager
    if _manager is None:
        _manager = LiquidationWSManager()
    return _manager
