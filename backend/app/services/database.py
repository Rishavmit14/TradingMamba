"""Phase 5: SQLite database for demo account, trades, and signal tracking.

Uses aiosqlite with WAL mode for concurrent access from both the
Telegram bot callbacks and the FastAPI REST endpoints.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import aiosqlite

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "demo_account.db"

# ── Schema ──────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_hash TEXT UNIQUE NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    risk_reward_ratio REAL NOT NULL,
    grade TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    confluences TEXT NOT NULL,
    entry_method TEXT,
    pattern_type TEXT,
    timeframe TEXT NOT NULL,
    is_counter_trend INTEGER DEFAULT 0,
    climax_warning INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    telegram_sent INTEGER DEFAULT 0,
    telegram_message_id INTEGER
);

CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL DEFAULT 10000.0,
    initial_balance REAL NOT NULL DEFAULT 10000.0,
    risk_per_trade_pct REAL NOT NULL DEFAULT 1.0,
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    status TEXT NOT NULL DEFAULT 'pending',
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    risk_reward_ratio REAL NOT NULL,
    grade TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    confluences TEXT NOT NULL,
    entry_method TEXT,
    pattern_type TEXT,
    position_size_usd REAL,
    position_size_btc REAL,
    exit_price REAL,
    pnl_usd REAL,
    pnl_pct REAL,
    outcome TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    opened_at TEXT,
    closed_at TEXT,
    action_source TEXT DEFAULT 'web',
    bars_monitored INTEGER DEFAULT 0,
    trade_source TEXT DEFAULT 'signal'
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ── Helpers ─────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def _get_db() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


# ── Init ────────────────────────────────────────────────


async def init_db(initial_balance: float = 10_000.0, risk_pct: float = 1.0):
    """Create tables and seed account row if missing."""
    db = await _get_db()
    try:
        await db.executescript(_SCHEMA)
        # Migration: add trade_source column if missing (existing DBs)
        try:
            await db.execute("ALTER TABLE trades ADD COLUMN trade_source TEXT DEFAULT 'signal'")
            await db.commit()
        except Exception:
            pass  # column already exists
        # Seed account singleton
        await db.execute(
            """INSERT OR IGNORE INTO account (id, balance, initial_balance, risk_per_trade_pct)
               VALUES (1, ?, ?, ?)""",
            (initial_balance, initial_balance, risk_pct),
        )
        # Seed initial equity snapshot
        count = await db.execute("SELECT COUNT(*) FROM equity_snapshots")
        row = await count.fetchone()
        if row[0] == 0:
            await db.execute(
                "INSERT INTO equity_snapshots (balance, timestamp) VALUES (?, ?)",
                (initial_balance, _now()),
            )
        await db.commit()
    finally:
        await db.close()


# ── Account ─────────────────────────────────────────────


async def get_account() -> dict:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM account WHERE id = 1")
        row = await cursor.fetchone()
        if not row:
            return {}
        d = dict(row)
        total = d["total_trades"]
        d["win_rate"] = round(d["wins"] / total * 100, 1) if total > 0 else 0.0
        d["pnl_total"] = round(d["balance"] - d["initial_balance"], 2)
        return d
    finally:
        await db.close()


async def update_account_balance(delta_usd: float, is_win: bool):
    """Adjust balance after a trade closes and increment win/loss counter."""
    db = await _get_db()
    try:
        if is_win:
            await db.execute(
                """UPDATE account SET balance = balance + ?,
                   total_trades = total_trades + 1, wins = wins + 1,
                   updated_at = ? WHERE id = 1""",
                (delta_usd, _now()),
            )
        else:
            await db.execute(
                """UPDATE account SET balance = balance + ?,
                   total_trades = total_trades + 1, losses = losses + 1,
                   updated_at = ? WHERE id = 1""",
                (delta_usd, _now()),
            )
        await db.commit()
    finally:
        await db.close()


async def update_account_settings(risk_pct: Optional[float] = None):
    db = await _get_db()
    try:
        if risk_pct is not None:
            await db.execute(
                "UPDATE account SET risk_per_trade_pct = ?, updated_at = ? WHERE id = 1",
                (risk_pct, _now()),
            )
        await db.commit()
    finally:
        await db.close()


async def reset_account(initial_balance: float = 10_000.0, risk_pct: float = 1.0):
    """Wipe all trades/signals and reset balance."""
    db = await _get_db()
    try:
        await db.execute("DELETE FROM trades")
        await db.execute("DELETE FROM signals")
        await db.execute("DELETE FROM equity_snapshots")
        await db.execute(
            """UPDATE account SET balance = ?, initial_balance = ?,
               risk_per_trade_pct = ?, total_trades = 0, wins = 0, losses = 0,
               updated_at = ? WHERE id = 1""",
            (initial_balance, initial_balance, risk_pct, _now()),
        )
        await db.execute(
            "INSERT INTO equity_snapshots (balance, timestamp) VALUES (?, ?)",
            (initial_balance, _now()),
        )
        await db.commit()
    finally:
        await db.close()


# ── Signals ─────────────────────────────────────────────


async def signal_exists(signal_hash: str) -> bool:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT 1 FROM signals WHERE signal_hash = ?", (signal_hash,)
        )
        return (await cursor.fetchone()) is not None
    finally:
        await db.close()


async def insert_signal(signal_data: dict) -> int:
    """Insert a new signal and return its ID."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO signals
               (signal_hash, direction, entry_price, stop_loss, take_profit,
                risk_reward_ratio, grade, confidence_score, confluences,
                entry_method, pattern_type, timeframe, is_counter_trend,
                climax_warning, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_data["signal_hash"],
                signal_data["direction"],
                signal_data["entry_price"],
                signal_data["stop_loss"],
                signal_data["take_profit"],
                signal_data["risk_reward_ratio"],
                signal_data["grade"],
                signal_data["confidence_score"],
                json.dumps(signal_data.get("confluences", [])),
                signal_data.get("entry_method"),
                signal_data.get("pattern_type"),
                signal_data.get("timeframe", "M15"),
                int(signal_data.get("is_counter_trend", False)),
                int(signal_data.get("climax_warning", False)),
                _now(),
            ),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def mark_signal_sent(signal_id: int, message_id: Optional[int] = None):
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE signals SET telegram_sent = 1, telegram_message_id = ? WHERE id = ?",
            (message_id, signal_id),
        )
        await db.commit()
    finally:
        await db.close()


# ── Trades ──────────────────────────────────────────────


async def insert_trade(signal_id: int, signal_data: dict) -> int:
    """Create a pending trade linked to a signal."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO trades
               (signal_id, status, direction, entry_price, stop_loss, take_profit,
                risk_reward_ratio, grade, confidence_score, confluences,
                entry_method, pattern_type, created_at)
               VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_id,
                signal_data["direction"],
                signal_data["entry_price"],
                signal_data["stop_loss"],
                signal_data["take_profit"],
                signal_data["risk_reward_ratio"],
                signal_data["grade"],
                signal_data["confidence_score"],
                json.dumps(signal_data.get("confluences", [])),
                signal_data.get("entry_method"),
                signal_data.get("pattern_type"),
                _now(),
            ),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def insert_manual_trade(trade_data: dict) -> Optional[dict]:
    """Create a manual trade directly in 'open' status (no signal required)."""
    import uuid

    db = await _get_db()
    try:
        # Create a placeholder signal record for FK constraint
        manual_hash = f"manual_{uuid.uuid4().hex[:12]}"
        sig_cursor = await db.execute(
            """INSERT INTO signals
               (signal_hash, direction, entry_price, stop_loss, take_profit,
                risk_reward_ratio, grade, confidence_score, confluences,
                entry_method, pattern_type, timeframe, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                manual_hash,
                trade_data["direction"],
                trade_data["entry_price"],
                trade_data["stop_loss"],
                trade_data["take_profit"],
                trade_data.get("risk_reward_ratio", 0),
                "M",  # Manual grade marker
                0,
                "[]",
                "manual",
                "Manual trade",
                trade_data.get("timeframe", "M15"),
                _now(),
            ),
        )
        signal_id = sig_cursor.lastrowid

        # Calculate position size
        acct = await db.execute("SELECT * FROM account WHERE id = 1")
        account = await acct.fetchone()
        balance = account["balance"]
        risk_pct = account["risk_per_trade_pct"]

        entry = trade_data["entry_price"]
        sl = trade_data["stop_loss"]
        risk_amount = balance * (risk_pct / 100)
        price_risk = abs(entry - sl)
        if price_risk == 0:
            price_risk = entry * 0.01

        position_size_btc = risk_amount / price_risk
        position_size_usd = position_size_btc * entry

        rr = abs(trade_data["take_profit"] - entry) / price_risk if price_risk > 0 else 0

        cursor = await db.execute(
            """INSERT INTO trades
               (signal_id, status, direction, entry_price, stop_loss, take_profit,
                risk_reward_ratio, grade, confidence_score, confluences,
                entry_method, pattern_type, position_size_usd, position_size_btc,
                opened_at, action_source, trade_source, created_at)
               VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', ?)""",
            (
                signal_id,
                trade_data["direction"],
                entry,
                sl,
                trade_data["take_profit"],
                round(rr, 2),
                "M",
                0,
                "[]",
                "manual",
                "Manual trade",
                round(position_size_usd, 2),
                round(position_size_btc, 6),
                _now(),
                "web",
                _now(),
            ),
        )
        await db.commit()

        trade_cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (cursor.lastrowid,))
        row = await trade_cursor.fetchone()
        return _trade_to_dict(row) if row else None
    finally:
        await db.close()


async def take_trade(trade_id: int, source: str = "web") -> Optional[dict]:
    """Transition a pending trade to open. Returns the trade dict or None."""
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
        if not trade or trade["status"] != "pending":
            return None

        # Calculate position size
        acct = await db.execute("SELECT * FROM account WHERE id = 1")
        account = await acct.fetchone()
        balance = account["balance"]
        risk_pct = account["risk_per_trade_pct"]

        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        risk_amount = balance * (risk_pct / 100)
        price_risk = abs(entry - sl)
        if price_risk == 0:
            price_risk = entry * 0.01  # fallback 1%

        position_size_btc = risk_amount / price_risk
        position_size_usd = position_size_btc * entry

        await db.execute(
            """UPDATE trades SET status = 'open', position_size_usd = ?,
               position_size_btc = ?, opened_at = ?, action_source = ?
               WHERE id = ?""",
            (round(position_size_usd, 2), round(position_size_btc, 6), _now(), source, trade_id),
        )
        await db.commit()

        # Return updated trade
        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = await cursor.fetchone()
        return _trade_to_dict(row) if row else None
    finally:
        await db.close()


async def skip_trade(trade_id: int, source: str = "web") -> bool:
    """Mark a pending trade as skipped. Returns True if successful."""
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT status FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
        if not trade or trade["status"] != "pending":
            return False
        await db.execute(
            "UPDATE trades SET status = 'skipped', action_source = ?, closed_at = ? WHERE id = ?",
            (source, _now(), trade_id),
        )
        await db.commit()
        return True
    finally:
        await db.close()


async def update_trade_sl_tp(trade_id: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[dict]:
    """Update SL and/or TP of an open trade. Recalculates R:R. Returns trade dict or None."""
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
        if not trade or trade["status"] != "open":
            return None

        new_sl = stop_loss if stop_loss is not None else trade["stop_loss"]
        new_tp = take_profit if take_profit is not None else trade["take_profit"]
        entry = trade["entry_price"]

        # Recalculate R:R
        risk = abs(entry - new_sl)
        reward = abs(new_tp - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        await db.execute(
            "UPDATE trades SET stop_loss = ?, take_profit = ?, risk_reward_ratio = ? WHERE id = ?",
            (new_sl, new_tp, rr, trade_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = await cursor.fetchone()
        return _trade_to_dict(row) if row else None
    finally:
        await db.close()


async def close_trade(trade_id: int, exit_price: float, source: str = "web") -> Optional[dict]:
    """Close an open trade at given price. Returns trade dict or None."""
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
        if not trade or trade["status"] != "open":
            return None

        entry = trade["entry_price"]
        direction = trade["direction"]
        size_btc = trade["position_size_btc"] or 0

        # Calculate PnL
        if direction == "bullish":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        pnl_usd = size_btc * abs(exit_price - entry)
        if pnl_pct < 0:
            pnl_usd = -pnl_usd

        outcome = "win" if pnl_pct > 0 else "loss"

        await db.execute(
            """UPDATE trades SET status = 'closed', exit_price = ?,
               pnl_usd = ?, pnl_pct = ?, outcome = ?, closed_at = ?,
               action_source = ? WHERE id = ?""",
            (exit_price, round(pnl_usd, 2), round(pnl_pct, 4), outcome, _now(), source, trade_id),
        )
        await db.commit()

        # Update account balance
        await update_account_balance(round(pnl_usd, 2), outcome == "win")

        # Snapshot equity
        acct = await db.execute("SELECT balance FROM account WHERE id = 1")
        row = await acct.fetchone()
        if row:
            await db.execute(
                "INSERT INTO equity_snapshots (balance, timestamp) VALUES (?, ?)",
                (row["balance"], _now()),
            )
            await db.commit()

        cursor = await db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = await cursor.fetchone()
        return _trade_to_dict(row) if row else None
    finally:
        await db.close()


async def get_open_trades() -> List[dict]:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY opened_at DESC"
        )
        rows = await cursor.fetchall()
        return [_trade_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_pending_trades() -> List[dict]:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM trades WHERE status = 'pending' ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [_trade_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_trades(status: Optional[str] = None, limit: int = 50) -> List[dict]:
    db = await _get_db()
    try:
        if status:
            cursor = await db.execute(
                "SELECT * FROM trades WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [_trade_to_dict(r) for r in rows]
    finally:
        await db.close()


async def timeout_pending_trades(timeout_seconds: int = 300) -> List[int]:
    """Auto-skip pending trades older than timeout. Returns list of skipped IDs."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            """SELECT id, created_at FROM trades WHERE status = 'pending'"""
        )
        rows = await cursor.fetchall()
        skipped = []
        now = datetime.now(timezone.utc)
        for row in rows:
            created = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            if (now - created).total_seconds() > timeout_seconds:
                await db.execute(
                    "UPDATE trades SET status = 'skipped', closed_at = ?, action_source = 'timeout' WHERE id = ?",
                    (_now(), row["id"]),
                )
                skipped.append(row["id"])
        if skipped:
            await db.commit()
        return skipped
    finally:
        await db.close()


# ── Equity ──────────────────────────────────────────────


async def insert_equity_snapshot(balance: float):
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO equity_snapshots (balance, timestamp) VALUES (?, ?)",
            (balance, _now()),
        )
        await db.commit()
    finally:
        await db.close()


async def get_equity_curve() -> List[dict]:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT balance, timestamp FROM equity_snapshots ORDER BY id ASC"
        )
        rows = await cursor.fetchall()
        return [{"balance": r["balance"], "timestamp": r["timestamp"]} for r in rows]
    finally:
        await db.close()


# ── Helpers ─────────────────────────────────────────────


def _trade_to_dict(row) -> dict:
    """Convert a sqlite Row to a clean dict with parsed JSON fields."""
    d = dict(row)
    # Parse confluences from JSON string
    if isinstance(d.get("confluences"), str):
        try:
            d["confluences"] = json.loads(d["confluences"])
        except (json.JSONDecodeError, TypeError):
            d["confluences"] = []
    return d
