"""
Automated Signal Scheduler for TradingMamba
Runs signal generation on schedule and sends Telegram alerts
100% FREE - uses APScheduler
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed. Run: pip install apscheduler")

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.free_market_data import FreeMarketDataService
from app.services.signal_generator import SignalGenerator
from app.services.telegram_notifier import TelegramNotifier
from app.ml.kill_zones import KillZoneAnalyzer
from app.ml.pattern_recognition import ICTPatternRecognizer
from app.ml.price_predictor import ICTPricePredictor
from app.database import db


class SignalScheduler:
    """
    Automated signal generation and notification scheduler

    Schedule:
    - Every hour: Check for signals during Kill Zones
    - Daily at 22:00 UTC: Generate Daily timeframe signals
    - Weekly on Sunday 20:00 UTC: Generate Weekly signals
    """

    def __init__(self):
        self.market_data = FreeMarketDataService()
        self.signal_generator = SignalGenerator()
        self.kill_zone_analyzer = KillZoneAnalyzer()
        self.pattern_recognizer = ICTPatternRecognizer()
        self.price_predictor = ICTPricePredictor()

        # Telegram notifier (optional - works without if not configured)
        self.telegram = None
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if bot_token and chat_id:
            self.telegram = TelegramNotifier(bot_token, chat_id)
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram not configured - signals will be saved but not sent")

        # Default symbols to monitor
        self.symbols = [
            'EURUSD=X',   # EUR/USD
            'GBPUSD=X',   # GBP/USD
            'USDJPY=X',   # USD/JPY
            'XAUUSD=X',   # Gold
            'GC=F',       # Gold Futures
            'ES=F',       # S&P 500 Futures
            'NQ=F',       # Nasdaq Futures
            'BTC-USD',    # Bitcoin
        ]

        # Minimum confidence to send alert
        self.min_confidence = 0.65

        # Scheduler
        self.scheduler = None
        if SCHEDULER_AVAILABLE:
            self.scheduler = AsyncIOScheduler()

    async def check_and_generate_signals(self, timeframe: str = 'H1'):
        """
        Check market conditions and generate signals

        Parameters:
        - timeframe: 'H1', 'H4', 'D1', 'W1'
        """
        logger.info(f"Running signal check for {timeframe} timeframe")

        # Check if we're in a kill zone (optimal trading window)
        session_info = self.kill_zone_analyzer.get_current_session()
        in_kill_zone = self.kill_zone_analyzer.is_in_kill_zone()

        if timeframe == 'H1' and not in_kill_zone:
            logger.info("Not in Kill Zone - skipping H1 signal check")
            return []

        signals_generated = []

        for symbol in self.symbols:
            try:
                # Fetch market data
                data = self.market_data.get_ohlcv(symbol, timeframe, limit=200)

                if data is None or len(data) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                # Generate signal
                signal = self.signal_generator.generate_signal(symbol, data, timeframe)

                if signal is None:
                    continue

                # Only process actionable signals
                if signal.get('direction') == 'WAIT':
                    continue

                confidence = signal.get('confidence', 0)

                # Enhance with pattern recognition
                patterns = self.pattern_recognizer.detect_all_patterns(data)
                pattern_summary = self.pattern_recognizer.get_pattern_summary(patterns)

                # Add pattern info to signal
                signal['patterns_detected'] = len(patterns)
                signal['pattern_bias'] = pattern_summary.get('bias', 'neutral')

                # Get price prediction
                prediction = self.price_predictor.predict(data, timeframe=timeframe)
                signal['prediction'] = prediction.to_dict()

                # Adjust confidence based on Kill Zone
                if in_kill_zone:
                    confidence = self.kill_zone_analyzer.adjust_confidence_for_timing(
                        confidence, symbol
                    )
                    signal['confidence'] = confidence
                    signal['kill_zone'] = session_info.get('kill_zone', {}).get('name', 'None')

                # Check if signal meets threshold
                if confidence >= self.min_confidence:
                    # Save to database
                    signal_id = db.save_signal(signal)
                    signal['id'] = signal_id

                    # Send Telegram notification
                    if self.telegram:
                        await self._send_telegram_alert(signal)

                    signals_generated.append(signal)
                    logger.info(f"Signal generated: {symbol} {signal['direction']} "
                               f"({confidence:.1%} confidence)")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info(f"Generated {len(signals_generated)} signals for {timeframe}")
        return signals_generated

    async def _send_telegram_alert(self, signal: Dict):
        """Send signal alert via Telegram"""
        if not self.telegram:
            return

        try:
            direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"

            # Format take profit levels
            tp_levels = signal.get('take_profit', [])
            if isinstance(tp_levels, list) and len(tp_levels) > 0:
                tp_text = f"TP1: {tp_levels[0]:.5f}"
                if len(tp_levels) > 1 and tp_levels[1]:
                    tp_text += f"\nTP2: {tp_levels[1]:.5f}"
            else:
                tp_text = "See analysis"

            # Format factors
            factors = signal.get('factors', [])
            factors_text = "\n".join([f"• {f}" for f in factors[:5]])

            message = f"""
{direction_emoji} <b>ICT SIGNAL ALERT</b> {direction_emoji}

<b>{signal['symbol']}</b> | {signal['timeframe']}
Direction: <b>{signal['direction']}</b>
Confidence: <b>{signal['confidence']:.0%}</b>

📊 <b>Levels:</b>
Entry: {signal.get('entry_price', 0):.5f}
Stop Loss: {signal.get('stop_loss', 0):.5f}
{tp_text}
R:R: 1:{signal.get('risk_reward', 0):.1f}

📋 <b>Analysis:</b>
{factors_text}

🎯 <b>Pattern Bias:</b> {signal.get('pattern_bias', 'neutral').title()}
⏰ <b>Kill Zone:</b> {signal.get('kill_zone', 'N/A')}

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC
"""

            await self.telegram.send_message(message)
            logger.info(f"Telegram alert sent for {signal['symbol']}")

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    async def run_hourly_check(self):
        """Hourly signal check - runs during Kill Zones"""
        await self.check_and_generate_signals('H1')

    async def run_4h_check(self):
        """4-hour signal check"""
        await self.check_and_generate_signals('H4')

    async def run_daily_check(self):
        """Daily signal check - runs at 22:00 UTC"""
        await self.check_and_generate_signals('D1')

    async def run_weekly_check(self):
        """Weekly signal check - runs Sunday 20:00 UTC"""
        await self.check_and_generate_signals('W1')

    def start(self):
        """Start the scheduler"""
        if not SCHEDULER_AVAILABLE:
            logger.error("APScheduler not available. Install with: pip install apscheduler")
            return False

        # Hourly check (every hour at minute 5)
        self.scheduler.add_job(
            self.run_hourly_check,
            CronTrigger(minute=5),
            id='hourly_signals',
            name='Hourly Signal Check'
        )

        # 4-hour check (at 00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC)
        self.scheduler.add_job(
            self.run_4h_check,
            CronTrigger(hour='0,4,8,12,16,20', minute=5),
            id='4h_signals',
            name='4H Signal Check'
        )

        # Daily check (22:00 UTC - after NY close)
        self.scheduler.add_job(
            self.run_daily_check,
            CronTrigger(hour=22, minute=0),
            id='daily_signals',
            name='Daily Signal Check'
        )

        # Weekly check (Sunday 20:00 UTC)
        self.scheduler.add_job(
            self.run_weekly_check,
            CronTrigger(day_of_week='sun', hour=20, minute=0),
            id='weekly_signals',
            name='Weekly Signal Check'
        )

        self.scheduler.start()
        logger.info("Signal scheduler started!")
        logger.info("Schedule:")
        logger.info("  - Hourly: Every hour at :05")
        logger.info("  - 4H: Every 4 hours at :05")
        logger.info("  - Daily: 22:00 UTC")
        logger.info("  - Weekly: Sunday 20:00 UTC")

        return True

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("Signal scheduler stopped")

    def get_schedule_info(self) -> Dict:
        """Get current schedule information"""
        if not self.scheduler:
            return {'status': 'not_initialized'}

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            })

        return {
            'status': 'running' if self.scheduler.running else 'stopped',
            'jobs': jobs,
            'symbols': self.symbols,
            'min_confidence': self.min_confidence
        }

    def add_symbol(self, symbol: str):
        """Add a symbol to monitor"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"Added symbol: {symbol}")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from monitoring"""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.info(f"Removed symbol: {symbol}")


# Create global scheduler instance
signal_scheduler = SignalScheduler()


async def run_manual_check(timeframe: str = 'H1'):
    """Run a manual signal check (for testing)"""
    scheduler = SignalScheduler()
    signals = await scheduler.check_and_generate_signals(timeframe)
    return signals


if __name__ == "__main__":
    # Test run
    import asyncio

    async def main():
        print("Running manual signal check...")
        signals = await run_manual_check('H1')
        print(f"\nGenerated {len(signals)} signals:")
        for s in signals:
            print(f"  {s['symbol']}: {s['direction']} ({s['confidence']:.1%})")

    asyncio.run(main())
