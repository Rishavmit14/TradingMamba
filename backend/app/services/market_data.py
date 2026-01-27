"""
Market Data Service

Fetches market data from various providers for analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import httpx
except ImportError:
    httpx = None

from ..config import settings

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> Optional['pd.DataFrame']:
        """Get OHLCV data for a symbol"""
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        pass


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage market data provider"""

    BASE_URL = "https://www.alphavantage.co/query"

    TIMEFRAME_MAP = {
        'M1': '1min',
        'M5': '5min',
        'M15': '15min',
        'M30': '30min',
        'H1': '60min',
        'D1': 'daily',
        'W1': 'weekly',
        'MN': 'monthly',
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ALPHA_VANTAGE_KEY
        if httpx is None:
            raise ImportError("httpx is required for AlphaVantageProvider")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> Optional['pd.DataFrame']:
        """Fetch OHLCV data from Alpha Vantage"""
        if pd is None:
            raise ImportError("pandas is required")

        if not self.api_key:
            logger.error("Alpha Vantage API key not configured")
            return None

        av_timeframe = self.TIMEFRAME_MAP.get(timeframe)
        if not av_timeframe:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        # Determine function based on timeframe
        if av_timeframe in ['daily', 'weekly', 'monthly']:
            function = f'TIME_SERIES_{av_timeframe.upper()}'
            series_key = f'Time Series ({av_timeframe.title()})'
        else:
            function = 'TIME_SERIES_INTRADAY'
            series_key = f'Time Series ({av_timeframe})'

        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full' if limit > 100 else 'compact',
        }

        if 'INTRADAY' in function:
            params['interval'] = av_timeframe

        try:
            with httpx.Client() as client:
                response = client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None

            if series_key not in data:
                logger.error(f"No data found for {symbol}")
                return None

            # Parse data into DataFrame
            series_data = data[series_key]
            df = pd.DataFrame.from_dict(series_data, orient='index')

            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']

            # Convert types
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            df['volume'] = pd.to_numeric(df['volume'])

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Limit rows
            return df.tail(limit)

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return None

    def get_available_symbols(self) -> List[str]:
        """Get commonly used forex pairs"""
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP',
            'EURJPY', 'GBPJPY', 'XAUUSD',
        ]


class PolygonProvider(MarketDataProvider):
    """Polygon.io market data provider"""

    BASE_URL = "https://api.polygon.io"

    TIMEFRAME_MAP = {
        'M1': ('minute', 1),
        'M5': ('minute', 5),
        'M15': ('minute', 15),
        'M30': ('minute', 30),
        'H1': ('hour', 1),
        'H4': ('hour', 4),
        'D1': ('day', 1),
        'W1': ('week', 1),
        'MN': ('month', 1),
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.POLYGON_API_KEY
        if httpx is None:
            raise ImportError("httpx is required for PolygonProvider")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500
    ) -> Optional['pd.DataFrame']:
        """Fetch OHLCV data from Polygon"""
        if pd is None:
            raise ImportError("pandas is required")

        if not self.api_key:
            logger.error("Polygon API key not configured")
            return None

        if timeframe not in self.TIMEFRAME_MAP:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        timespan, multiplier = self.TIMEFRAME_MAP[timeframe]

        # Calculate date range
        end_date = datetime.now()
        if timespan == 'minute':
            start_date = end_date - timedelta(days=7)
        elif timespan == 'hour':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365 * 2)

        # Format for forex (C:EURUSD)
        if len(symbol) == 6 and symbol.isalpha():
            ticker = f"C:{symbol}"
        else:
            ticker = symbol

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

        try:
            with httpx.Client() as client:
                response = client.get(
                    url,
                    params={'apiKey': self.api_key, 'limit': limit}
                )
                response.raise_for_status()
                data = response.json()

            if data.get('status') != 'OK' or not data.get('results'):
                logger.error(f"No data from Polygon for {symbol}")
                return None

            # Parse into DataFrame
            df = pd.DataFrame(data['results'])
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp'
            })

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            return df[['open', 'high', 'low', 'close', 'volume']].tail(limit)

        except Exception as e:
            logger.error(f"Error fetching data from Polygon: {e}")
            return None

    def get_available_symbols(self) -> List[str]:
        """Get available symbols"""
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD',
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL',
        ]


class MarketDataService:
    """
    Main market data service

    Provides a unified interface for fetching market data
    from multiple providers.
    """

    def __init__(self):
        self.providers: Dict[str, MarketDataProvider] = {}

        # Initialize available providers
        if settings.ALPHA_VANTAGE_KEY:
            try:
                self.providers['alphavantage'] = AlphaVantageProvider()
                logger.info("Alpha Vantage provider initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Alpha Vantage: {e}")

        if settings.POLYGON_API_KEY:
            try:
                self.providers['polygon'] = PolygonProvider()
                logger.info("Polygon provider initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Polygon: {e}")

        self.default_provider = 'polygon' if 'polygon' in self.providers else 'alphavantage'

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        provider: Optional[str] = None
    ) -> Optional['pd.DataFrame']:
        """
        Fetch OHLCV data

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN)
            limit: Number of candles
            provider: Specific provider to use

        Returns:
            DataFrame with OHLCV data
        """
        provider_name = provider or self.default_provider

        if provider_name not in self.providers:
            available = list(self.providers.keys())
            if not available:
                logger.error("No market data providers configured")
                return None
            provider_name = available[0]

        return self.providers[provider_name].get_ohlcv(symbol, timeframe, limit)

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str] = ['H1', 'H4', 'D1', 'W1'],
        limit: int = 200
    ) -> Dict[str, 'pd.DataFrame']:
        """
        Fetch data for multiple timeframes

        Returns:
            Dict mapping timeframe to DataFrame
        """
        data = {}
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf, limit)
            if df is not None:
                data[tf] = df
        return data
