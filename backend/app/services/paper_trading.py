"""
Paper Trading Engine - Alpaca API Integration + EdgeTracker Integration

This module provides paper trading capabilities using Alpaca's FREE paper trading API.
It bridges the gap between backtesting and live trading.

Key Features:
- Real market prices (15-min delay on free tier)
- Real order matching simulation
- Real slippage modeling
- Same API as live trading (zero code changes needed)
- Automatic trade outcome tracking
- **HEDGE FUND**: EdgeTracker integration for statistical feedback loop

100% FREE - Uses Alpaca paper trading API.

Setup:
1. Create free account at https://alpaca.markets
2. Get API keys from dashboard (Paper Trading)
3. Set environment variables:
   - ALPACA_API_KEY
   - ALPACA_SECRET_KEY
   - ALPACA_BASE_URL (https://paper-api.alpaca.markets for paper)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Try to import alpaca-trade-api
try:
    from alpaca_trade_api import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Alpaca API not installed. Install with: pip install alpaca-trade-api")

# Import EdgeTracker for statistical feedback loop
try:
    from ..ml.hedge_fund_ml import get_edge_tracker
    EDGE_TRACKER_AVAILABLE = True
except ImportError:
    EDGE_TRACKER_AVAILABLE = False
    logger.warning("EdgeTracker not available for paper trading feedback loop")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PaperTrade:
    """Represents a paper trade"""
    # Identification
    trade_id: str
    symbol: str
    pattern_type: str

    # Order details
    side: str
    quantity: int
    order_type: str

    # Prices
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None

    # Status
    status: str = OrderStatus.PENDING.value
    alpaca_order_id: Optional[str] = None

    # Timing
    created_at: str = ""
    filled_at: Optional[str] = None
    closed_at: Optional[str] = None

    # Results
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    slippage: Optional[float] = None

    # ML Context
    ml_confidence: Optional[float] = None
    pattern_grade: Optional[str] = None
    mtf_confluence: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.trade_id:
            self.trade_id = f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_open(self) -> bool:
        return self.status in [OrderStatus.FILLED.value, OrderStatus.PARTIALLY_FILLED.value] and self.exit_price is None

    def is_closed(self) -> bool:
        return self.exit_price is not None


@dataclass
class AccountSummary:
    """Paper trading account summary"""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    daily_pnl: float
    total_pnl: float
    open_positions: int
    total_trades: int
    win_rate: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PaperTradingEngine:
    """
    Paper trading engine using Alpaca API.

    This is the bridge between backtest and live trading:
    - Backtest: "This pattern has 67% win rate historically"
    - Paper Trading: "Let's see if it works in real-time with real prices"
    - Live Trading: "Now we're confident, let's trade real money"
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent.parent / "data"
        self.trades_dir = self.data_dir / "paper_trades"
        self.trades_dir.mkdir(parents=True, exist_ok=True)

        # Alpaca client
        self.api: Optional[REST] = None
        self._connected = False

        # Local trade tracking (works even without Alpaca)
        self.trades: Dict[str, PaperTrade] = {}
        self._load_trades()

        # Simulated account (when Alpaca not available)
        self.simulated_equity = 100000.0
        self.simulated_cash = 100000.0

    def connect(self) -> bool:
        """
        Connect to Alpaca paper trading API.

        Requires environment variables:
        - ALPACA_API_KEY
        - ALPACA_SECRET_KEY
        """
        if not ALPACA_AVAILABLE:
            print("Alpaca not installed. Running in simulation mode.")
            return False

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            print("Alpaca API keys not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
            print("Running in simulation mode.")
            return False

        try:
            self.api = REST(api_key, secret_key, base_url)
            account = self.api.get_account()
            self._connected = True
            print(f"Connected to Alpaca Paper Trading")
            print(f"  Account Status: {account.status}")
            print(f"  Equity: ${float(account.equity):,.2f}")
            print(f"  Cash: ${float(account.cash):,.2f}")
            return True

        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to Alpaca"""
        return self._connected and self.api is not None

    def get_account_summary(self) -> AccountSummary:
        """Get current account summary"""
        if self.is_connected():
            try:
                account = self.api.get_account()
                positions = self.api.list_positions()

                # Calculate stats from our trades
                closed_trades = [t for t in self.trades.values() if t.is_closed()]
                wins = [t for t in closed_trades if t.pnl and t.pnl > 0]
                win_rate = len(wins) / len(closed_trades) if closed_trades else 0

                return AccountSummary(
                    equity=float(account.equity),
                    cash=float(account.cash),
                    buying_power=float(account.buying_power),
                    portfolio_value=float(account.portfolio_value),
                    daily_pnl=float(account.equity) - float(account.last_equity),
                    total_pnl=float(account.equity) - 100000,  # Assuming 100k start
                    open_positions=len(positions),
                    total_trades=len(self.trades),
                    win_rate=win_rate
                )
            except Exception as e:
                print(f"Error getting account: {e}")

        # Simulated account
        closed_trades = [t for t in self.trades.values() if t.is_closed()]
        wins = [t for t in closed_trades if t.pnl and t.pnl > 0]
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)

        return AccountSummary(
            equity=self.simulated_equity + total_pnl,
            cash=self.simulated_cash,
            buying_power=self.simulated_cash * 2,  # 2x margin
            portfolio_value=self.simulated_equity + total_pnl,
            daily_pnl=0,
            total_pnl=total_pnl,
            open_positions=len([t for t in self.trades.values() if t.is_open()]),
            total_trades=len(self.trades),
            win_rate=win_rate
        )

    def submit_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        pattern_type: str,
        stop_loss: float = None,
        take_profit: float = None,
        order_type: str = "market",
        limit_price: float = None,
        ml_confidence: float = None,
        pattern_grade: str = None,
        mtf_confluence: str = None
    ) -> Optional[PaperTrade]:
        """
        Submit a paper trade.

        Args:
            symbol: Trading symbol (e.g., "AAPL", "SPY")
            side: "buy" or "sell"
            quantity: Number of shares
            pattern_type: ICT pattern that triggered this trade
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_type: "market" or "limit"
            limit_price: Price for limit orders
            ml_confidence: ML model's confidence (0-1)
            pattern_grade: Pattern grade (A+, A, B, etc.)
            mtf_confluence: MTF confluence level
        """
        trade = PaperTrade(
            trade_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol.upper(),
            pattern_type=pattern_type,
            side=side.lower(),
            quantity=quantity,
            order_type=order_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ml_confidence=ml_confidence,
            pattern_grade=pattern_grade,
            mtf_confluence=mtf_confluence,
            status=OrderStatus.PENDING.value
        )

        if self.is_connected():
            try:
                # Submit to Alpaca
                order = self.api.submit_order(
                    symbol=symbol.upper(),
                    qty=quantity,
                    side=side.lower(),
                    type=order_type,
                    time_in_force='day',
                    limit_price=limit_price if order_type == 'limit' else None
                )

                trade.alpaca_order_id = order.id
                trade.status = order.status

                # Wait briefly for fill
                time.sleep(1)
                order = self.api.get_order(order.id)

                if order.status == 'filled':
                    trade.status = OrderStatus.FILLED.value
                    trade.entry_price = float(order.filled_avg_price)
                    trade.filled_at = order.filled_at.isoformat() if order.filled_at else datetime.now().isoformat()

                    # Calculate slippage if limit order
                    if limit_price:
                        trade.slippage = abs(trade.entry_price - limit_price) / limit_price

                    # Submit bracket orders for SL/TP
                    if stop_loss or take_profit:
                        self._submit_bracket_orders(trade, stop_loss, take_profit)

                print(f"Trade submitted: {trade.trade_id}")
                print(f"  Status: {trade.status}")
                print(f"  Entry Price: ${trade.entry_price:,.2f}" if trade.entry_price else "  Pending fill")

            except Exception as e:
                print(f"Error submitting trade: {e}")
                trade.status = OrderStatus.REJECTED.value
        else:
            # Simulation mode - assume instant fill at current price
            trade.status = OrderStatus.FILLED.value
            trade.entry_price = self._get_simulated_price(symbol)
            trade.filled_at = datetime.now().isoformat()
            print(f"[SIMULATED] Trade submitted: {trade.trade_id}")
            print(f"  Entry Price: ${trade.entry_price:,.2f}")

        # Store trade
        self.trades[trade.trade_id] = trade
        self._save_trades()

        return trade

    def _submit_bracket_orders(
        self,
        trade: PaperTrade,
        stop_loss: float,
        take_profit: float
    ):
        """Submit bracket orders for stop loss and take profit"""
        if not self.is_connected():
            return

        try:
            # Stop loss order
            if stop_loss:
                exit_side = "sell" if trade.side == "buy" else "buy"
                self.api.submit_order(
                    symbol=trade.symbol,
                    qty=trade.quantity,
                    side=exit_side,
                    type='stop',
                    time_in_force='gtc',
                    stop_price=stop_loss
                )
                print(f"  Stop Loss set at: ${stop_loss:,.2f}")

            # Take profit order
            if take_profit:
                exit_side = "sell" if trade.side == "buy" else "buy"
                self.api.submit_order(
                    symbol=trade.symbol,
                    qty=trade.quantity,
                    side=exit_side,
                    type='limit',
                    time_in_force='gtc',
                    limit_price=take_profit
                )
                print(f"  Take Profit set at: ${take_profit:,.2f}")

        except Exception as e:
            print(f"Error submitting bracket orders: {e}")

    def close_trade(
        self,
        trade_id: str,
        exit_price: float = None,
        reason: str = "manual"
    ) -> Optional[PaperTrade]:
        """
        Close an open trade.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price (auto-fetched if None)
            reason: Reason for closing (manual, stop_loss, take_profit)
        """
        if trade_id not in self.trades:
            print(f"Trade not found: {trade_id}")
            return None

        trade = self.trades[trade_id]
        if not trade.is_open():
            print(f"Trade already closed: {trade_id}")
            return trade

        if self.is_connected():
            try:
                # Close position via Alpaca
                exit_side = "sell" if trade.side == "buy" else "buy"
                order = self.api.submit_order(
                    symbol=trade.symbol,
                    qty=trade.quantity,
                    side=exit_side,
                    type='market',
                    time_in_force='day'
                )

                # Wait for fill
                time.sleep(1)
                order = self.api.get_order(order.id)

                if order.status == 'filled':
                    trade.exit_price = float(order.filled_avg_price)
                    trade.closed_at = datetime.now().isoformat()

            except Exception as e:
                print(f"Error closing trade: {e}")
                return None
        else:
            # Simulation mode
            trade.exit_price = exit_price or self._get_simulated_price(trade.symbol)
            trade.closed_at = datetime.now().isoformat()

        # Calculate P&L
        if trade.entry_price and trade.exit_price:
            if trade.side == "buy":
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                trade.pnl_percent = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                trade.pnl_percent = (trade.entry_price - trade.exit_price) / trade.entry_price

        trade.status = OrderStatus.FILLED.value
        self._save_trades()

        # =====================================================================
        # HEDGE FUND LEVEL: Record trade outcome to EdgeTracker
        # =====================================================================
        self._record_to_edge_tracker(trade)

        print(f"Trade closed: {trade.trade_id}")
        print(f"  Exit Price: ${trade.exit_price:,.2f}")
        print(f"  P&L: ${trade.pnl:,.2f} ({trade.pnl_percent:+.2%})")

        return trade

    def _record_to_edge_tracker(self, trade: PaperTrade) -> None:
        """
        Record trade outcome to EdgeTracker for statistical feedback loop.

        This is the critical feedback mechanism that makes the ML LEARN from real trades:
        - Records win/loss/breakeven outcome
        - Calculates achieved R:R
        - Tracks by pattern type, session, day of week

        The EdgeTracker uses this to:
        - Update win rate per pattern type
        - Calculate expectancy (expected value per trade)
        - Identify which patterns have statistical edge
        - Determine best trading sessions/days
        """
        if not EDGE_TRACKER_AVAILABLE:
            logger.warning("EdgeTracker not available - trade outcome not recorded")
            return

        if not trade.is_closed() or not trade.pnl:
            return

        try:
            edge_tracker = get_edge_tracker()

            # Determine outcome
            if trade.pnl > 0:
                outcome = 'win'
            elif trade.pnl < 0:
                outcome = 'loss'
            else:
                outcome = 'breakeven'

            # Calculate achieved R:R
            rr_achieved = 0.0
            if trade.entry_price and trade.exit_price and trade.stop_loss:
                risk = abs(trade.entry_price - trade.stop_loss)
                if risk > 0:
                    reward = abs(trade.exit_price - trade.entry_price)
                    rr_achieved = reward / risk

            # Get session and day
            trade_time = datetime.fromisoformat(trade.created_at) if trade.created_at else datetime.now()
            day_of_week = trade_time.strftime("%A")

            hour = trade_time.hour
            if 0 <= hour < 4:
                session = "asian"
            elif 7 <= hour < 10:
                session = "london"
            elif 12 <= hour < 16:
                session = "new_york"
            else:
                session = "off_hours"

            # Record to EdgeTracker
            edge_tracker.record_trade(
                pattern_type=trade.pattern_type,
                outcome=outcome,
                rr_achieved=rr_achieved,
                session=session,
                day_of_week=day_of_week
            )

            logger.info(f"EdgeTracker recorded: {trade.pattern_type} | {outcome} | {rr_achieved:.2f}R")
            print(f"  📊 EdgeTracker: Recorded {outcome} for {trade.pattern_type} pattern")

        except Exception as e:
            logger.error(f"Failed to record trade to EdgeTracker: {e}")

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated current price (for when Alpaca not connected)"""
        # Use a simple random walk from a base price
        import random

        base_prices = {
            "SPY": 500.0,
            "AAPL": 180.0,
            "MSFT": 400.0,
            "GOOGL": 140.0,
            "TSLA": 250.0,
            "QQQ": 420.0,
        }

        base = base_prices.get(symbol.upper(), 100.0)
        # Add some noise
        return base * (1 + random.uniform(-0.02, 0.02))

    def get_open_trades(self) -> List[PaperTrade]:
        """Get all open trades"""
        return [t for t in self.trades.values() if t.is_open()]

    def get_closed_trades(self) -> List[PaperTrade]:
        """Get all closed trades"""
        return [t for t in self.trades.values() if t.is_closed()]

    def get_trades_by_pattern(self, pattern_type: str) -> List[PaperTrade]:
        """Get trades filtered by pattern type"""
        return [t for t in self.trades.values() if t.pattern_type == pattern_type]

    def get_pattern_performance(self, pattern_type: str) -> Dict[str, Any]:
        """Get performance statistics for a specific pattern"""
        trades = self.get_trades_by_pattern(pattern_type)
        closed = [t for t in trades if t.is_closed()]

        if not closed:
            return {"pattern_type": pattern_type, "total_trades": 0}

        wins = [t for t in closed if t.pnl and t.pnl > 0]
        losses = [t for t in closed if t.pnl and t.pnl <= 0]

        total_pnl = sum(t.pnl for t in closed if t.pnl)
        avg_pnl = total_pnl / len(closed)
        win_rate = len(wins) / len(closed)

        avg_win = sum(t.pnl for t in wins if t.pnl) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses if t.pnl) / len(losses) if losses else 0

        return {
            "pattern_type": pattern_type,
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss else float('inf'),
            "expectancy": (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        }

    def _save_trades(self):
        """Save trades to disk"""
        trades_file = self.trades_dir / "paper_trades.json"
        trades_data = {tid: t.to_dict() for tid, t in self.trades.items()}

        with open(trades_file, 'w') as f:
            json.dump(trades_data, f, indent=2)

    def _load_trades(self):
        """Load trades from disk"""
        trades_file = self.trades_dir / "paper_trades.json"

        if trades_file.exists():
            with open(trades_file, 'r') as f:
                trades_data = json.load(f)
                self.trades = {tid: PaperTrade(**data) for tid, data in trades_data.items()}


# Singleton instance
_paper_trader: Optional[PaperTradingEngine] = None


def get_paper_trader() -> PaperTradingEngine:
    """Get or create the singleton paper trader instance"""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTradingEngine()
    return _paper_trader


def submit_ml_signal_trade(
    symbol: str,
    direction: str,
    confidence: float,
    pattern_type: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    pattern_grade: str = None,
    mtf_confluence: str = None,
    position_size_usd: float = 1000
) -> Optional[PaperTrade]:
    """
    Submit a trade based on ML signal.

    This is the main integration point with the signal generator.

    Args:
        symbol: Trading symbol
        direction: "BUY" or "SELL"
        confidence: ML confidence (0-1)
        pattern_type: Pattern that generated signal
        entry_price: Expected entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        pattern_grade: Grade from PatternGrader
        mtf_confluence: Confluence from MTFAnalyzer
        position_size_usd: Dollar amount to trade

    Returns:
        PaperTrade if successful
    """
    trader = get_paper_trader()

    # Calculate quantity based on position size
    quantity = int(position_size_usd / entry_price)
    if quantity < 1:
        quantity = 1

    side = "buy" if direction.upper() == "BUY" else "sell"

    return trader.submit_trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        pattern_type=pattern_type,
        stop_loss=stop_loss,
        take_profit=take_profit,
        ml_confidence=confidence,
        pattern_grade=pattern_grade,
        mtf_confluence=mtf_confluence
    )


# Test function
def test_paper_trading():
    """Test the paper trading engine"""
    print("=" * 60)
    print("PAPER TRADING ENGINE TEST")
    print("=" * 60)

    trader = get_paper_trader()

    # Try to connect to Alpaca
    print("\nAttempting to connect to Alpaca...")
    connected = trader.connect()

    if not connected:
        print("Running in simulation mode (no Alpaca connection)")

    # Get account summary
    print("\n📊 ACCOUNT SUMMARY:")
    summary = trader.get_account_summary()
    print(f"  Equity: ${summary.equity:,.2f}")
    print(f"  Cash: ${summary.cash:,.2f}")
    print(f"  Buying Power: ${summary.buying_power:,.2f}")
    print(f"  Open Positions: {summary.open_positions}")
    print(f"  Total Trades: {summary.total_trades}")
    print(f"  Win Rate: {summary.win_rate:.1%}")

    # Submit a test trade
    print("\n📈 SUBMITTING TEST TRADE...")
    trade = trader.submit_trade(
        symbol="SPY",
        side="buy",
        quantity=10,
        pattern_type="fvg",
        stop_loss=490.0,
        take_profit=510.0,
        ml_confidence=0.85,
        pattern_grade="A",
        mtf_confluence="STRONG"
    )

    if trade:
        print(f"\nTrade created: {trade.trade_id}")
        print(f"  Symbol: {trade.symbol}")
        print(f"  Side: {trade.side}")
        print(f"  Quantity: {trade.quantity}")
        print(f"  Entry: ${trade.entry_price:,.2f}" if trade.entry_price else "  Pending")
        print(f"  Stop Loss: ${trade.stop_loss:,.2f}" if trade.stop_loss else "  None")
        print(f"  Take Profit: ${trade.take_profit:,.2f}" if trade.take_profit else "  None")

        # Simulate closing the trade
        print("\n📉 CLOSING TEST TRADE...")
        trade = trader.close_trade(trade.trade_id)

        if trade and trade.is_closed():
            print(f"\nTrade closed successfully!")
            print(f"  Exit Price: ${trade.exit_price:,.2f}")
            print(f"  P&L: ${trade.pnl:,.2f} ({trade.pnl_percent:+.2%})")

    # Get pattern performance
    print("\n📊 PATTERN PERFORMANCE:")
    perf = trader.get_pattern_performance("fvg")
    print(f"  Pattern: {perf['pattern_type']}")
    print(f"  Total Trades: {perf['total_trades']}")
    if perf['total_trades'] > 0:
        print(f"  Win Rate: {perf['win_rate']:.1%}")
        print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
        print(f"  Expectancy: ${perf['expectancy']:,.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_paper_trading()
