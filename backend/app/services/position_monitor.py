"""Phase 5: Background position monitor.

Runs every 10 seconds to check open demo positions against live Binance price.
Closes positions when SL or TP is hit, updates account balance, and sends
Telegram notifications.
"""

import asyncio
import logging
from typing import Optional

import httpx

from app.config import SYMBOL
from app.services.database import (
    get_open_trades,
    close_trade,
    get_account,
    insert_equity_snapshot,
)

logger = logging.getLogger("tradingmamba.position_monitor")


async def _fetch_current_price() -> Optional[float]:
    """Fetch current BTCUSDT price from Binance."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}"
            )
            data = resp.json()
            return float(data["price"])
    except Exception as e:
        logger.error(f"Price fetch failed: {e}")
        return None


class PositionMonitor:
    """Background task that monitors open positions every 10 seconds."""

    def __init__(self, bot=None, check_interval: int = 10):
        self.bot = bot
        self.interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._snapshot_counter = 0

    def start(self):
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self):
        """Main position monitoring loop."""
        await asyncio.sleep(10)  # initial delay

        while True:
            try:
                await self._check_positions()

                # Periodic equity snapshot every ~5 minutes (30 iterations × 10s)
                self._snapshot_counter += 1
                if self._snapshot_counter >= 30:
                    self._snapshot_counter = 0
                    account = await get_account()
                    if account:
                        await insert_equity_snapshot(account["balance"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            await asyncio.sleep(self.interval)

    async def _check_positions(self):
        """Check all open positions against current price."""
        open_trades = await get_open_trades()
        if not open_trades:
            return

        price = await _fetch_current_price()
        if price is None:
            return

        for trade in open_trades:
            trade_id = trade["id"]
            direction = trade["direction"]
            entry = trade["entry_price"]
            sl = trade["stop_loss"]
            tp = trade["take_profit"]

            hit_sl = False
            hit_tp = False

            if direction == "bullish":
                if price <= sl:
                    hit_sl = True
                elif price >= tp:
                    hit_tp = True
            else:  # bearish
                if price >= sl:
                    hit_sl = True
                elif price <= tp:
                    hit_tp = True

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp
                result = await close_trade(trade_id, exit_price=price, source="monitor")

                if result:
                    outcome = result.get("outcome", "")
                    pnl_usd = result.get("pnl_usd", 0) or 0
                    pnl_pct = result.get("pnl_pct", 0) or 0
                    dir_label = "LONG" if direction == "bullish" else "SHORT"

                    event = "SL HIT" if hit_sl else "TP HIT"
                    emoji = "\u274C" if hit_sl else "\u2705"

                    logger.info(
                        f"Trade {trade_id} closed: {event} @ ${price:,.0f} "
                        f"(PnL: {'+'if pnl_usd>=0 else ''}{pnl_usd:,.2f})"
                    )

                    # Send Telegram notification
                    if self.bot and self.bot.chat_id:
                        try:
                            account = await get_account()
                            balance = account.get("balance", 0) if account else 0
                            await self.bot.send_trade_update(
                                f"{emoji} <b>{event}</b>\n\n"
                                f"{dir_label} @ ${entry:,.0f} \u2192 ${price:,.0f}\n"
                                f"P&L: <b>{'+'if pnl_usd>=0 else ''}"
                                f"${pnl_usd:,.2f}</b> ({pnl_pct:+.2f}%)\n\n"
                                f"Balance: <b>${balance:,.2f}</b>"
                            )
                        except Exception as e:
                            logger.error(f"Telegram notification failed: {e}")
