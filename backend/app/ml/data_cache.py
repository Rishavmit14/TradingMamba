"""
Data Cache - OHLCV Market Data with Parquet Caching

Fetches market data via yfinance and caches locally as Parquet files
for fast repeated access. Auto-refreshes stale data.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
CACHE_DIR = DATA_DIR / "market_cache"

# yfinance timeframe mapping
TF_MAP = {
    'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
    'H1': '1h', 'H4': '4h',  # Note: yfinance doesn't support 4h natively
    'D1': '1d', 'W1': '1wk', 'MN': '1mo',
}

# Maximum lookback per timeframe (yfinance limits)
TF_MAX_PERIOD = {
    'M1': '7d', 'M5': '60d', 'M15': '60d', 'M30': '60d',
    'H1': '730d', 'H4': '730d',
    'D1': 'max', 'W1': 'max', 'MN': 'max',
}

# Staleness thresholds
TF_STALE_HOURS = {
    'M1': 0.1, 'M5': 0.5, 'M15': 1, 'M30': 1,
    'H1': 1, 'H4': 4,
    'D1': 24, 'W1': 168, 'MN': 720,
}

# Symbol mapping for yfinance
SYMBOL_MAP = {
    'BTCUSDT': 'BTC-USD',
    'ETHUSDT': 'ETH-USD',
    'SOLUSDT': 'SOL-USD',
    'BNBUSDT': 'BNB-USD',
    'XRPUSDT': 'XRP-USD',
    'ADAUSDT': 'ADA-USD',
    'DOGEUSDT': 'DOGE-USD',
    'AVAXUSDT': 'AVAX-USD',
    'DOTUSDT': 'DOT-USD',
    'LINKUSDT': 'LINK-USD',
    # Traditional markets
    'SPY': 'SPY', 'SPX': '^GSPC',
    'DXY': 'DX-Y.NYB',
    'GOLD': 'GC=F', 'XAUUSD': 'GC=F',
    'NQ': 'NQ=F', 'ES': 'ES=F',
}


class DataCache:
    """OHLCV market data cache with Parquet storage."""

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_yf_symbol(self, symbol: str) -> str:
        """Map trading symbol to yfinance symbol."""
        return SYMBOL_MAP.get(symbol.upper(), symbol)

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for a symbol/timeframe."""
        safe_symbol = symbol.upper().replace('-', '_').replace('/', '_')
        return CACHE_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def _is_stale(self, cache_path: Path, timeframe: str) -> bool:
        """Check if cached data is stale."""
        if not cache_path.exists():
            return True
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        stale_hours = TF_STALE_HOURS.get(timeframe, 24)
        return datetime.now() - mtime > timedelta(hours=stale_hours)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = 'D1',
        lookback_days: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol, using cache when possible.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'SPY')
            timeframe: Timeframe code (M1, M5, M15, M30, H1, H4, D1, W1, MN)
            lookback_days: Override default lookback period
            force_refresh: Force re-download even if cache is fresh

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex
        """
        if yf is None:
            raise ImportError("yfinance is required: pip install yfinance")

        cache_path = self._cache_path(symbol, timeframe)

        # Return cached data if fresh
        if not force_refresh and not self._is_stale(cache_path, timeframe):
            try:
                df = pd.read_parquet(cache_path)
                if lookback_days and len(df) > 0:
                    cutoff = datetime.now() - timedelta(days=lookback_days)
                    df = df[df.index >= cutoff]
                logger.info(f"Cache hit: {symbol} {timeframe} ({len(df)} bars)")
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}, re-downloading")

        # Fetch from yfinance
        yf_symbol = self._get_yf_symbol(symbol)
        yf_interval = TF_MAP.get(timeframe, '1d')

        # Handle H4 (not native to yfinance): fetch H1 and resample
        resample_4h = False
        if timeframe == 'H4':
            yf_interval = '1h'
            resample_4h = True

        # Determine period
        if lookback_days:
            period = f"{lookback_days}d"
        else:
            period = TF_MAX_PERIOD.get(timeframe, '365d')

        logger.info(f"Fetching {yf_symbol} {yf_interval} period={period}")

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=yf_interval)
        except Exception as e:
            logger.error(f"yfinance fetch failed for {yf_symbol}: {e}")
            # Try fallback with download
            try:
                df = yf.download(yf_symbol, period=period, interval=yf_interval, progress=False)
            except Exception as e2:
                logger.error(f"yfinance download also failed: {e2}")
                # Return cached data even if stale
                if cache_path.exists():
                    return pd.read_parquet(cache_path)
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            logger.warning(f"No data returned for {yf_symbol}")
            if cache_path.exists():
                return pd.read_parquet(cache_path)
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        # Normalize columns
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Handle multi-level columns from yf.download
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]

        # Keep only OHLCV
        required = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in required if c in df.columns]
        df = df[available]

        # Resample H1 → H4 if needed
        if resample_4h and len(df) > 0:
            df = df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()

        # Drop NaN rows
        df = df.dropna()

        # Save to cache
        try:
            df.to_parquet(cache_path)
            logger.info(f"Cached {len(df)} bars for {symbol} {timeframe}")
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")

        # Apply lookback filter
        if lookback_days and len(df) > 0:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            cutoff = pd.Timestamp(cutoff).tz_localize(df.index.tz) if df.index.tz else pd.Timestamp(cutoff)
            df = df[df.index >= cutoff]

        return df

    def get_multi_symbol(
        self,
        symbols: list,
        timeframe: str = 'D1',
        lookback_days: int = 365,
    ) -> dict:
        """
        Fetch OHLCV for multiple symbols.

        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_ohlcv(symbol, timeframe, lookback_days)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear cached data."""
        if symbol and timeframe:
            path = self._cache_path(symbol, timeframe)
            if path.exists():
                path.unlink()
        elif symbol:
            for p in CACHE_DIR.glob(f"{symbol.upper()}*.parquet"):
                p.unlink()
        else:
            for p in CACHE_DIR.glob("*.parquet"):
                p.unlink()

    def get_cache_info(self) -> list:
        """Get info about all cached files."""
        info = []
        for p in sorted(CACHE_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                info.append({
                    'file': p.name,
                    'rows': len(df),
                    'start': str(df.index[0]) if len(df) > 0 else None,
                    'end': str(df.index[-1]) if len(df) > 0 else None,
                    'size_kb': round(p.stat().st_size / 1024, 1),
                    'modified': datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                })
            except Exception:
                info.append({'file': p.name, 'error': 'unreadable'})
        return info


# Singleton
_cache_instance = None

def get_data_cache() -> DataCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance
