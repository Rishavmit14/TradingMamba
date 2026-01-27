"""
Free Market Data Service using Yahoo Finance

Provides OHLCV data for forex pairs, indices, and stocks - completely FREE!
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    yf = None
    pd = None

logger = logging.getLogger(__name__)


# Symbol mappings for Yahoo Finance
SYMBOL_MAP = {
    # Forex pairs (Yahoo uses =X suffix)
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'NZDUSD': 'NZDUSD=X',
    'EURGBP': 'EURGBP=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',

    # Gold
    'XAUUSD': 'GC=F',  # Gold futures

    # Indices
    'US30': 'YM=F',    # Dow futures
    'NAS100': 'NQ=F',  # Nasdaq futures
    'SPX500': 'ES=F',  # S&P futures
    'DXY': 'DX-Y.NYB', # Dollar index

    # Popular stocks
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'GOOGL': 'GOOGL',
    'TSLA': 'TSLA',
    'NVDA': 'NVDA',
}

# Timeframe mappings
TIMEFRAME_MAP = {
    'M1': '1m',
    'M5': '5m',
    'M15': '15m',
    'M30': '30m',
    'H1': '1h',
    'H4': '4h',  # Not directly supported, we'll resample
    'D1': '1d',
    'W1': '1wk',
    'MN': '1mo',
}

# Period (how far back) for each timeframe
PERIOD_MAP = {
    'M1': '7d',      # 1-min data limited to 7 days
    'M5': '60d',     # 5-min data limited to 60 days
    'M15': '60d',
    'M30': '60d',
    'H1': '730d',    # 2 years
    'H4': '730d',
    'D1': 'max',
    'W1': 'max',
    'MN': 'max',
}


class FreeMarketDataService:
    """
    Free market data service using Yahoo Finance

    Usage:
        service = FreeMarketDataService()
        data = service.get_ohlcv('EURUSD', 'H1', limit=200)
    """

    def __init__(self):
        if yf is None or pd is None:
            raise ImportError("yfinance and pandas required: pip install yfinance pandas")

    def get_yahoo_symbol(self, symbol: str) -> str:
        """Convert our symbol to Yahoo Finance symbol"""
        return SYMBOL_MAP.get(symbol.upper(), symbol)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = 'H1',
        limit: int = 200
    ) -> Optional['pd.DataFrame']:
        """
        Get OHLCV data for a symbol

        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD')
            timeframe: M1, M5, M15, M30, H1, H4, D1, W1, MN
            limit: Number of candles

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        yahoo_symbol = self.get_yahoo_symbol(symbol)
        interval = TIMEFRAME_MAP.get(timeframe, '1h')
        period = PERIOD_MAP.get(timeframe, '60d')

        try:
            ticker = yf.Ticker(yahoo_symbol)

            # Special handling for H4 (not native in Yahoo)
            if timeframe == 'H4':
                # Get H1 data and resample
                df = ticker.history(period=period, interval='1h')
                if df.empty:
                    return None

                # Resample to 4H
                df = df.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            # Standardize column names
            df.columns = [c.lower() for c in df.columns]

            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    df[col] = 0

            # Limit rows
            df = df[required].tail(limit)

            logger.info(f"Got {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        yahoo_symbol = self.get_yahoo_symbol(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('previousClose')
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ['H1', 'H4', 'D1', 'W1']
    ) -> Dict[str, 'pd.DataFrame']:
        """Get data for multiple timeframes"""
        data = {}
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf)
            if df is not None:
                data[tf] = df
        return data

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        return list(SYMBOL_MAP.keys())

    def search_symbol(self, query: str) -> List[dict]:
        """Search for symbols"""
        query_upper = query.upper()
        matches = []

        for symbol, yahoo_symbol in SYMBOL_MAP.items():
            if query_upper in symbol:
                matches.append({
                    'symbol': symbol,
                    'yahoo_symbol': yahoo_symbol,
                    'type': 'forex' if '=X' in yahoo_symbol else 'other'
                })

        return matches


# Quick test function
def test_market_data():
    """Test the market data service"""
    service = FreeMarketDataService()

    print("Testing Free Market Data Service")
    print("=" * 40)

    # Test EURUSD
    print("\n1. EURUSD H1 data:")
    df = service.get_ohlcv('EURUSD', 'H1', limit=5)
    if df is not None:
        print(df.tail())
    else:
        print("  No data available")

    # Test current price
    print("\n2. Current prices:")
    for symbol in ['EURUSD', 'XAUUSD', 'US30']:
        price = service.get_current_price(symbol)
        print(f"  {symbol}: {price}")

    # Test available symbols
    print("\n3. Available symbols:")
    print(f"  {service.get_available_symbols()}")


if __name__ == "__main__":
    test_market_data()
