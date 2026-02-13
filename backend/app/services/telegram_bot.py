"""Phase 5: Telegram bot for signal alerts and trade confirmation.

Uses python-telegram-bot v21+ with long-polling (works on localhost).
The bot sends formatted signal alerts with inline Take/Skip buttons,
and handles /start, /balance, /positions, /history commands.
"""

import logging
from typing import Optional, Union

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from app.services.database import (
    get_account,
    get_open_trades,
    get_pending_trades,
    get_trades,
    take_trade,
    skip_trade,
    reset_account,
)
from app.config import DEMO_INITIAL_BALANCE, DEMO_RISK_PER_TRADE_PCT

logger = logging.getLogger("tradingmamba.telegram")


class TelegramAlertBot:
    """Telegram bot that sends signal alerts and handles trade actions."""

    def __init__(self, token: str, chat_id: str = ""):
        self.token = token
        self.chat_id = chat_id or None
        self.username: Optional[str] = None
        self.running = False
        self._app: Optional[Application] = None

    async def start(self):
        """Initialize and start the bot with long-polling."""
        self._app = Application.builder().token(self.token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("balance", self._cmd_balance))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("history", self._cmd_history))
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))
        self._app.add_handler(CallbackQueryHandler(self._callback_handler))

        await self._app.initialize()
        await self._app.start()

        # Get bot info
        bot_info = await self._app.bot.get_me()
        self.username = bot_info.username
        logger.info(f"Bot @{self.username} initialized")

        # Start polling in background
        await self._app.updater.start_polling(drop_pending_updates=True)
        self.running = True
        logger.info(f"Bot polling started (chat_id: {self.chat_id or 'awaiting /start'})")

    async def stop(self):
        """Gracefully stop the bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        self.running = False
        logger.info("Bot stopped")

    # ── Public methods for sending messages ──

    async def send_message(self, chat_id: Union[str, int], text: str):
        """Send a plain text message."""
        if self._app:
            await self._app.bot.send_message(
                chat_id=chat_id, text=text, parse_mode="HTML"
            )

    async def send_signal_alert(
        self,
        signal_data: dict,
        trade_id: int,
        rank: int = 0,
        total_pending: int = 0,
    ) -> Optional[int]:
        """Send a formatted signal alert with Take/Skip buttons.

        rank/total_pending: when >1 pending, shows "Signal 1 of 3 — BEST PICK".
        Returns the Telegram message ID if sent, or None.
        """
        if not self.chat_id or not self._app:
            return None

        direction = signal_data["direction"]
        is_long = direction == "bullish"
        arrow = "\u2191" if is_long else "\u2193"
        dir_label = "LONG" if is_long else "SHORT"
        grade = signal_data["grade"]
        entry = signal_data["entry_price"]
        sl = signal_data["stop_loss"]
        tp = signal_data["take_profit"]
        rr = signal_data["risk_reward_ratio"]
        conf = signal_data["confidence_score"]
        method = (signal_data.get("entry_method") or "unknown").upper()
        pattern = signal_data.get("pattern_type", "")
        confluences = signal_data.get("confluences", [])
        priority = signal_data.get("priority_score", 0)

        sl_pct = abs(entry - sl) / entry * 100
        tp_pct = abs(tp - entry) / entry * 100

        # Header with priority ranking
        if total_pending > 1 and rank > 0:
            rank_label = "\U0001F947 BEST PICK" if rank == 1 else f"#{rank} of {total_pending}"
            text = (
                f"<b>\U0001F514 NEW SIGNAL \u2014 Grade {grade}</b>  "
                f"<b>[{rank_label}]</b>\n"
                f"<i>Priority: {priority:.0f}/100</i>\n\n"
            )
        else:
            text = f"<b>\U0001F514 NEW SIGNAL \u2014 Grade {grade}</b>\n\n"

        text += (
            f"<b>{arrow} {dir_label} BTCUSDT @ ${entry:,.0f}</b>\n"
            f"\u251C SL: ${sl:,.0f} (-{sl_pct:.2f}%)\n"
            f"\u251C TP: ${tp:,.0f} (+{tp_pct:.2f}%)\n"
            f"\u251C R:R: {rr:.1f}\n"
            f"\u2514 Confidence: {conf:.0f}%\n\n"
        )

        if confluences:
            text += f"<b>Confluences:</b> {', '.join(confluences)}\n"
        text += f"<b>Entry:</b> {method} | <b>Pattern:</b> {pattern}\n"

        if signal_data.get("climax_warning"):
            text += "\n\u26A0\uFE0F <b>CLIMAX WARNING</b> \u2014 Proceed with caution!"
        if signal_data.get("is_counter_trend"):
            text += "\n\u21BA <b>Counter-trend</b> signal"

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("\u2705 Take Trade", callback_data=f"take_{trade_id}"),
                InlineKeyboardButton("\u274C Skip", callback_data=f"skip_{trade_id}"),
            ]
        ])

        msg = await self._app.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        return msg.message_id

    async def send_trade_update(self, message: str):
        """Send a trade status update (opened, closed, etc.)."""
        if not self.chat_id or not self._app:
            return
        await self._app.bot.send_message(
            chat_id=self.chat_id, text=message, parse_mode="HTML"
        )

    # ── Command handlers ──

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Register chat and show welcome message."""
        self.chat_id = str(update.effective_chat.id)
        logger.info(f"Chat registered: {self.chat_id}")

        account = await get_account()
        balance = account.get("balance", 0)

        await update.message.reply_html(
            f"<b>\U0001F40D TradingMamba Demo Bot</b>\n\n"
            f"You'll receive signal alerts here.\n"
            f"Current balance: <b>${balance:,.2f}</b>\n\n"
            f"<b>Commands:</b>\n"
            f"/balance \u2014 Show account balance\n"
            f"/positions \u2014 Show open positions\n"
            f"/history \u2014 Recent trade history\n"
            f"/reset \u2014 Reset demo account\n\n"
            f"When a signal appears, tap <b>Take</b> or <b>Skip</b>."
        )

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        account = await get_account()
        if not account:
            await update.message.reply_text("Account not initialized.")
            return

        balance = account["balance"]
        initial = account["initial_balance"]
        pnl = balance - initial
        pnl_pct = (pnl / initial * 100) if initial else 0
        wins = account["wins"]
        losses = account["losses"]
        total = account["total_trades"]
        wr = account.get("win_rate", 0)
        emoji = "\U0001F4C8" if pnl >= 0 else "\U0001F4C9"

        await update.message.reply_html(
            f"<b>{emoji} Demo Account</b>\n\n"
            f"Balance: <b>${balance:,.2f}</b>\n"
            f"P&L: <b>{'+'if pnl>=0 else ''}{pnl:,.2f}</b> ({pnl_pct:+.1f}%)\n"
            f"Trades: {total} ({wins}W / {losses}L)\n"
            f"Win Rate: {wr:.1f}%"
        )

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        open_trades = await get_open_trades()
        if not open_trades:
            await update.message.reply_text("No open positions.")
            return

        lines = ["<b>\U0001F4CA Open Positions</b>\n"]
        for t in open_trades:
            arrow = "\u2191" if t["direction"] == "bullish" else "\u2193"
            dir_label = "LONG" if t["direction"] == "bullish" else "SHORT"
            lines.append(
                f"{arrow} {dir_label} @ ${t['entry_price']:,.0f} "
                f"(SL: ${t['stop_loss']:,.0f} / TP: ${t['take_profit']:,.0f})"
            )
        await update.message.reply_html("\n".join(lines))

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = await get_trades(status="closed", limit=10)
        if not trades:
            await update.message.reply_text("No closed trades yet.")
            return

        lines = ["<b>\U0001F4CB Recent Trades</b>\n"]
        for t in trades:
            arrow = "\u2191" if t["direction"] == "bullish" else "\u2193"
            outcome = t.get("outcome", "?")
            pnl = t.get("pnl_usd", 0) or 0
            emoji = "\u2705" if outcome == "win" else "\u274C"
            lines.append(
                f"{emoji} {arrow} {t['grade']} @ ${t['entry_price']:,.0f} "
                f"\u2192 ${t.get('exit_price', 0):,.0f} "
                f"({'+'if pnl>=0 else ''}{pnl:,.2f})"
            )
        await update.message.reply_html("\n".join(lines))

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await reset_account(DEMO_INITIAL_BALANCE, DEMO_RISK_PER_TRADE_PCT)
        await update.message.reply_html(
            f"\U0001F504 Account reset to <b>${DEMO_INITIAL_BALANCE:,.2f}</b>"
        )

    # ── Callback handler for inline buttons ──

    async def _callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        data = query.data
        if data.startswith("take_"):
            trade_id = int(data.split("_", 1)[1])
            result = await take_trade(trade_id, source="telegram")
            if result:
                size_usd = result.get("position_size_usd", 0)
                size_btc = result.get("position_size_btc", 0)
                dir_label = "LONG" if result["direction"] == "bullish" else "SHORT"
                await query.edit_message_reply_markup(reply_markup=None)
                await self.send_trade_update(
                    f"\u2705 <b>Position Opened</b>\n"
                    f"{dir_label} {size_btc:.6f} BTC (${size_usd:,.0f})\n"
                    f"Entry: ${result['entry_price']:,.0f}"
                )
            else:
                await query.edit_message_reply_markup(reply_markup=None)
                await self.send_trade_update("Trade already taken or expired.")

        elif data.startswith("skip_"):
            trade_id = int(data.split("_", 1)[1])
            ok = await skip_trade(trade_id, source="telegram")
            await query.edit_message_reply_markup(reply_markup=None)
            if ok:
                await self.send_trade_update("\u23ED Signal skipped.")
            else:
                await self.send_trade_update("Trade already skipped or expired.")
