"""L2 Order Book Depth Recorder — continuous capture for OFI backtesting.

Polls Binance L2 depth every N seconds and persists snapshots to SQLite.
Over time this builds a historical L2 dataset enabling OFI algo backtesting.

Usage:
    recorder = get_l2_recorder()
    await recorder.start()           # Start in FastAPI lifespan
    status = recorder.get_status()   # Check recording stats
    await recorder.stop()            # Stop on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class L2Recorder:
    """Polls Binance L2 depth at regular intervals and persists to SQLite."""

    def __init__(self, symbol: str = "BTCUSDT", interval: int = 10):
        self._symbol = symbol
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._snapshot_count = 0
        self._start_time: float = 0
        self._last_error: str | None = None

    async def start(self):
        """Start the background polling task."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("L2 Depth Recorder started (%ds interval)", self._interval)

    async def stop(self):
        """Stop the polling task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            "L2 Depth Recorder stopped (%d snapshots recorded)", self._snapshot_count
        )

    async def _poll_loop(self):
        """Reconnecting poll loop — fetch L2 depth and persist."""
        from app.services.quant_data import fetch_l2_depth
        from app.services.history_db import open_db, upsert_l2_depth

        while self._running:
            try:
                snapshot = await fetch_l2_depth(self._symbol)
                if snapshot and snapshot.get("best_bid"):
                    snapshot["timestamp"] = int(time.time() * 1000)
                    await asyncio.to_thread(self._write_snapshot, snapshot)
                    self._snapshot_count += 1
                    self._last_error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._last_error = str(e)
                logger.debug("L2 Recorder error: %s", e)

            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    @staticmethod
    def _write_snapshot(snapshot: dict) -> None:
        """Write a single L2 snapshot to SQLite (runs in thread)."""
        from app.services.history_db import open_db, upsert_l2_depth

        conn = open_db()
        try:
            upsert_l2_depth(conn, [snapshot])
        finally:
            conn.close()

    def get_status(self) -> dict:
        """Return recorder status for the API."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "symbol": self._symbol,
            "interval_seconds": self._interval,
            "snapshot_count": self._snapshot_count,
            "uptime_hours": round(uptime / 3600, 2),
            "last_error": self._last_error,
        }

    @property
    def is_running(self) -> bool:
        return self._running


# Global singleton
_recorder: L2Recorder | None = None


def get_l2_recorder() -> L2Recorder:
    """Get or create the global L2 depth recorder."""
    global _recorder
    if _recorder is None:
        from app.config import L2_RECORD_INTERVAL
        _recorder = L2Recorder(interval=L2_RECORD_INTERVAL)
    return _recorder
