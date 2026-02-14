"""SQLite historical data store for candles + futures.

Single file database at data/tradingmamba.db storing:
- 7 candle tables (candles_1M .. candles_M5)
- 5 futures tables (oi_history, funding_rate, taker_volume, top_trader_ratio, global_ratio)
- 1 sync_meta table tracking last synced timestamp per table

query_candles() returns list[Candle], query_futures() returns the same dict
shape as fetch_market_intel() — zero changes needed in signal engine.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.models import Candle

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "tradingmamba.db"

CANDLE_TFS = ["1M", "W1", "D1", "H4", "H1", "M15", "M5"]

_CANDLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles_{tf} (
    timestamp INTEGER PRIMARY KEY,
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL NOT NULL
);
"""

_FUTURES_SCHEMAS = {
    "oi_history": """
        CREATE TABLE IF NOT EXISTS oi_history (
            timestamp INTEGER PRIMARY KEY,
            oi        REAL NOT NULL,
            oi_usd    REAL NOT NULL
        );
    """,
    "funding_rate": """
        CREATE TABLE IF NOT EXISTS funding_rate (
            timestamp  INTEGER PRIMARY KEY,
            rate       REAL NOT NULL,
            mark_price REAL NOT NULL
        );
    """,
    "taker_volume": """
        CREATE TABLE IF NOT EXISTS taker_volume (
            timestamp INTEGER PRIMARY KEY,
            buy_vol   REAL NOT NULL,
            sell_vol  REAL NOT NULL,
            ratio     REAL NOT NULL
        );
    """,
    "top_trader_ratio": """
        CREATE TABLE IF NOT EXISTS top_trader_ratio (
            timestamp INTEGER PRIMARY KEY,
            long_pct  REAL NOT NULL,
            short_pct REAL NOT NULL,
            ratio     REAL NOT NULL
        );
    """,
    "global_ratio": """
        CREATE TABLE IF NOT EXISTS global_ratio (
            timestamp INTEGER PRIMARY KEY,
            long_pct  REAL NOT NULL,
            short_pct REAL NOT NULL,
            ratio     REAL NOT NULL
        );
    """,
}

_SYNC_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_meta (
    table_name     TEXT PRIMARY KEY,
    last_timestamp INTEGER NOT NULL,
    row_count      INTEGER NOT NULL DEFAULT 0
);
"""


def open_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open (or create) the SQLite database with all tables."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    for tf in CANDLE_TFS:
        conn.execute(_CANDLE_SCHEMA.format(tf=tf))
    for schema in _FUTURES_SCHEMAS.values():
        conn.execute(schema)
    conn.execute(_SYNC_SCHEMA)
    conn.commit()


# ── Write operations (used by fetch script) ──


def upsert_candles(conn: sqlite3.Connection, tf: str, rows: list[dict]) -> int:
    """Insert or replace candle rows. Returns number of rows upserted."""
    if not rows:
        return 0
    table = f"candles_{tf}"
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} (timestamp, open, high, low, close, volume) "
        f"VALUES (:timestamp, :open, :high, :low, :close, :volume)",
        rows,
    )
    conn.commit()
    _update_sync_meta(conn, table)
    return len(rows)


def upsert_oi_history(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO oi_history (timestamp, oi, oi_usd) "
        "VALUES (:timestamp, :oi, :oi_usd)",
        rows,
    )
    conn.commit()
    _update_sync_meta(conn, "oi_history")
    return len(rows)


def upsert_funding_rate(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO funding_rate (timestamp, rate, mark_price) "
        "VALUES (:timestamp, :rate, :mark_price)",
        rows,
    )
    conn.commit()
    _update_sync_meta(conn, "funding_rate")
    return len(rows)


def upsert_taker_volume(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO taker_volume (timestamp, buy_vol, sell_vol, ratio) "
        "VALUES (:timestamp, :buy_vol, :sell_vol, :ratio)",
        rows,
    )
    conn.commit()
    _update_sync_meta(conn, "taker_volume")
    return len(rows)


def upsert_top_trader_ratio(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO top_trader_ratio (timestamp, long_pct, short_pct, ratio) "
        "VALUES (:timestamp, :long_pct, :short_pct, :ratio)",
        rows,
    )
    conn.commit()
    _update_sync_meta(conn, "top_trader_ratio")
    return len(rows)


def upsert_global_ratio(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO global_ratio (timestamp, long_pct, short_pct, ratio) "
        "VALUES (:timestamp, :long_pct, :short_pct, :ratio)",
        rows,
    )
    conn.commit()
    _update_sync_meta(conn, "global_ratio")
    return len(rows)


def _update_sync_meta(conn: sqlite3.Connection, table_name: str) -> None:
    """Update sync_meta with the latest timestamp and row count for a table."""
    row = conn.execute(
        f"SELECT MAX(timestamp), COUNT(*) FROM {table_name}"
    ).fetchone()
    if row and row[0] is not None:
        conn.execute(
            "INSERT OR REPLACE INTO sync_meta (table_name, last_timestamp, row_count) "
            "VALUES (?, ?, ?)",
            (table_name, row[0], row[1]),
        )
        conn.commit()


def get_last_timestamp(conn: sqlite3.Connection, table_name: str) -> int | None:
    """Get the last synced timestamp for a table, or None if empty."""
    row = conn.execute(
        "SELECT last_timestamp FROM sync_meta WHERE table_name = ?",
        (table_name,),
    ).fetchone()
    return row[0] if row else None


def get_sync_summary(conn: sqlite3.Connection) -> dict[str, dict]:
    """Get sync status for all tables."""
    rows = conn.execute("SELECT table_name, last_timestamp, row_count FROM sync_meta").fetchall()
    return {r[0]: {"last_timestamp": r[1], "row_count": r[2]} for r in rows}


# ── Read operations (used by backtester) ──


def query_candles(
    conn: sqlite3.Connection,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> list[Candle]:
    """Query candles for a timeframe within a timestamp range.

    Returns list[Candle] with sequential indices, oldest first.
    """
    table = f"candles_{tf}"
    rows = conn.execute(
        f"SELECT timestamp, open, high, low, close, volume FROM {table} "
        f"WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (start_ms, end_ms),
    ).fetchall()
    return [
        Candle(
            timestamp=r[0], open=r[1], high=r[2],
            low=r[3], close=r[4], volume=r[5], index=i,
        )
        for i, r in enumerate(rows)
    ]


def query_futures(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
) -> dict | None:
    """Query all 5 futures tables for a time window.

    Returns the same dict shape as fetch_market_intel() so the signal
    engine can use it with zero changes. Returns None if no data exists.
    """
    oi_rows = conn.execute(
        "SELECT timestamp, oi, oi_usd FROM oi_history "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (start_ms, end_ms),
    ).fetchall()

    funding_rows = conn.execute(
        "SELECT timestamp, rate, mark_price FROM funding_rate "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (start_ms, end_ms),
    ).fetchall()

    taker_rows = conn.execute(
        "SELECT timestamp, buy_vol, sell_vol, ratio FROM taker_volume "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (start_ms, end_ms),
    ).fetchall()

    top_rows = conn.execute(
        "SELECT timestamp, long_pct, short_pct, ratio FROM top_trader_ratio "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (start_ms, end_ms),
    ).fetchall()

    global_rows = conn.execute(
        "SELECT timestamp, long_pct, short_pct, ratio FROM global_ratio "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
        (start_ms, end_ms),
    ).fetchall()

    # If no futures data at all for this window, return None
    if not oi_rows and not funding_rows and not taker_rows:
        return None

    # Build the same shape as fetch_market_intel()
    oi_history = [
        {"timestamp": r[0], "oi": r[1], "oi_usd": r[2]}
        for r in oi_rows
    ]

    current_oi = oi_rows[-1][1] if oi_rows else 0
    current_oi_usd = oi_rows[-1][2] if oi_rows else 0

    funding_history = [
        {"timestamp": r[0], "rate": r[1], "mark_price": r[2]}
        for r in funding_rows
    ]
    current_funding = funding_rows[-1][1] if funding_rows else 0
    mark_price = funding_rows[-1][2] if funding_rows else 0

    taker_volume = [
        {"timestamp": r[0], "buy_vol": r[1], "sell_vol": r[2], "ratio": r[3]}
        for r in taker_rows
    ]

    top_trader_ratio = [
        {"timestamp": r[0], "long_pct": r[1], "short_pct": r[2], "ratio": r[3]}
        for r in top_rows
    ]

    global_ratio = [
        {"timestamp": r[0], "long_pct": r[1], "short_pct": r[2], "ratio": r[3]}
        for r in global_rows
    ]

    return {
        "open_interest": {
            "current": current_oi,
            "current_usd": current_oi_usd,
            "history": oi_history,
        },
        "funding_rate": {
            "current": current_funding,
            "next_funding_time": 0,
            "mark_price": mark_price,
            "index_price": 0,
            "history": funding_history,
        },
        "top_trader_ratio": top_trader_ratio,
        "global_ratio": global_ratio,
        "taker_volume": taker_volume,
        "premium_index": {
            "mark_price": mark_price,
            "index_price": 0,
            "premium_pct": 0,
        },
    }
