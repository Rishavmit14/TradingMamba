"""
Telegram Notification Service

Sends trading signals and alerts via Telegram Bot.
100% FREE - Uses Telegram Bot API.

Setup:
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Start a chat with your bot and get your chat_id
4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env

Usage:
    notifier = TelegramNotifier()
    notifier.send_signal(signal_data)
"""

import os
import json
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends notifications via Telegram Bot API.
    Completely FREE to use.
    """

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None

        # Rate limiting
        self.last_message_time = None
        self.min_interval = 1  # Minimum seconds between messages

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message"""
        if not self.is_configured():
            logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            return False

        if aiohttp is None:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False

        # Rate limiting
        if self.last_message_time:
            elapsed = (datetime.utcnow() - self.last_message_time).total_seconds()
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    }
                ) as response:
                    result = await response.json()

                    if result.get("ok"):
                        self.last_message_time = datetime.utcnow()
                        return True
                    else:
                        logger.error(f"Telegram API error: {result}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Synchronous wrapper for send_message"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.send_message(text, parse_mode))

    def format_signal_message(self, signal: Dict) -> str:
        """Format a trading signal as a Telegram message"""
        direction = signal.get('direction', 'neutral').upper()
        symbol = signal.get('symbol', 'UNKNOWN')
        confidence = signal.get('confidence', 0) * 100
        strength = signal.get('strength', 0) * 100

        # Direction emoji
        if direction == 'BULLISH':
            emoji = "🟢"
            direction_text = "📈 BULLISH"
        elif direction == 'BEARISH':
            emoji = "🔴"
            direction_text = "📉 BEARISH"
        else:
            emoji = "⚪"
            direction_text = "⚖️ NEUTRAL"

        # Build message
        message = f"""
{emoji} <b>Smart Money SIGNAL: {symbol}</b>

{direction_text}
━━━━━━━━━━━━━━━

<b>Confidence:</b> {confidence:.0f}%
<b>Strength:</b> {strength:.0f}%
<b>Confluence:</b> {signal.get('confluence_score', 0)} concepts

<b>Entry Zone:</b>
{signal.get('entry_zone', ['N/A', 'N/A'])[0]:.5f} - {signal.get('entry_zone', ['N/A', 'N/A'])[1]:.5f}

<b>Stop Loss:</b> {signal.get('stop_loss', 0):.5f}

<b>Take Profits:</b>
"""
        # Add take profit levels
        for i, tp in enumerate(signal.get('take_profit', []), 1):
            message += f"  TP{i}: {tp:.5f}\n"

        message += f"""
<b>Risk/Reward:</b> 1:{signal.get('risk_reward', 0):.1f}

<b>Smart Money Concepts:</b>
{', '.join(signal.get('concepts', ['None']))}

━━━━━━━━━━━━━━━
<i>🤖 TradingMamba AI</i>
<i>⚠️ Not financial advice</i>
"""
        return message.strip()

    def format_analysis_message(self, analysis: Dict) -> str:
        """Format a market analysis as a Telegram message"""
        symbol = analysis.get('symbol', 'UNKNOWN')
        timeframes = analysis.get('timeframes', [])

        message = f"""
📊 <b>Smart Money ANALYSIS: {symbol}</b>
━━━━━━━━━━━━━━━

<b>Timeframes:</b> {', '.join(timeframes)}

<b>Detected Concepts:</b>
{', '.join(analysis.get('detected_concepts', ['None']))}

"""
        # Add timeframe details
        for tf, data in analysis.get('analyses', {}).items():
            bias = data.get('bias', 'neutral').upper()
            bias_emoji = "🟢" if bias == "BULLISH" else "🔴" if bias == "BEARISH" else "⚪"
            message += f"""
<b>{tf}:</b> {bias_emoji} {bias}
  Zone: {data.get('zone', 'N/A')}
  OBs: {data.get('order_blocks', 0)} | FVGs: {data.get('fvgs', 0)}
"""

        message += f"""
━━━━━━━━━━━━━━━
<i>🤖 TradingMamba AI</i>
"""
        return message.strip()

    def format_alert_message(self, alert_type: str, message: str) -> str:
        """Format a general alert message"""
        type_emojis = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅',
            'signal': '🎯',
        }

        emoji = type_emojis.get(alert_type, '📢')

        return f"""
{emoji} <b>TRADINGMAMBA ALERT</b>
━━━━━━━━━━━━━━━

{message}

━━━━━━━━━━━━━━━
<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
""".strip()

    async def send_signal(self, signal: Dict) -> bool:
        """Send a trading signal notification"""
        message = self.format_signal_message(signal)
        return await self.send_message(message)

    def send_signal_sync(self, signal: Dict) -> bool:
        """Synchronous wrapper for send_signal"""
        message = self.format_signal_message(signal)
        return self.send_message_sync(message)

    async def send_analysis(self, analysis: Dict) -> bool:
        """Send a market analysis notification"""
        message = self.format_analysis_message(analysis)
        return await self.send_message(message)

    async def send_alert(self, alert_type: str, message: str) -> bool:
        """Send a general alert"""
        formatted = self.format_alert_message(alert_type, message)
        return await self.send_message(formatted)

    async def test_connection(self) -> Dict:
        """Test the Telegram connection"""
        if not self.is_configured():
            return {
                'success': False,
                'error': 'Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID'
            }

        if aiohttp is None:
            return {
                'success': False,
                'error': 'aiohttp not installed'
            }

        try:
            async with aiohttp.ClientSession() as session:
                # Get bot info
                async with session.get(f"{self.base_url}/getMe") as response:
                    bot_info = await response.json()

                if not bot_info.get('ok'):
                    return {'success': False, 'error': 'Invalid bot token'}

                # Send test message
                test_sent = await self.send_message("🤖 TradingMamba connected successfully!")

                return {
                    'success': test_sent,
                    'bot_name': bot_info.get('result', {}).get('username'),
                    'message': 'Connection successful' if test_sent else 'Failed to send test message'
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}


class SignalNotificationManager:
    """
    Manages signal notifications with filtering and scheduling.
    """

    def __init__(self, notifier: TelegramNotifier = None):
        self.notifier = notifier or TelegramNotifier()
        self.sent_signals: List[str] = []  # Track sent signal IDs
        self.min_confidence = 0.6  # Minimum confidence to send
        self.notification_cooldown = 3600  # Seconds between signals for same symbol

        # Symbol-specific last notification times
        self.last_notification: Dict[str, datetime] = {}

    def should_notify(self, signal: Dict) -> bool:
        """Determine if a signal should trigger a notification"""
        # Check confidence threshold
        if signal.get('confidence', 0) < self.min_confidence:
            return False

        # Check cooldown for symbol
        symbol = signal.get('symbol', '')
        last_time = self.last_notification.get(symbol)

        if last_time:
            elapsed = (datetime.utcnow() - last_time).total_seconds()
            if elapsed < self.notification_cooldown:
                return False

        # Check if already sent (by signal ID or hash)
        signal_id = f"{symbol}_{signal.get('direction')}_{signal.get('timestamp', '')[:16]}"
        if signal_id in self.sent_signals:
            return False

        return True

    async def process_signal(self, signal: Dict) -> bool:
        """Process a signal and send notification if appropriate"""
        if not self.should_notify(signal):
            return False

        success = await self.notifier.send_signal(signal)

        if success:
            symbol = signal.get('symbol', '')
            self.last_notification[symbol] = datetime.utcnow()

            signal_id = f"{symbol}_{signal.get('direction')}_{signal.get('timestamp', '')[:16]}"
            self.sent_signals.append(signal_id)

            # Keep only last 100 signal IDs
            if len(self.sent_signals) > 100:
                self.sent_signals = self.sent_signals[-100:]

        return success


# Test function
async def test_telegram():
    """Test Telegram notification"""
    print("=" * 50)
    print("TELEGRAM NOTIFICATION TEST")
    print("=" * 50)

    notifier = TelegramNotifier()

    if not notifier.is_configured():
        print("\n⚠️ Telegram not configured!")
        print("\nTo set up Telegram notifications:")
        print("1. Message @BotFather on Telegram to create a bot")
        print("2. Copy the bot token")
        print("3. Start a chat with your bot")
        print("4. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates")
        print("5. Find your chat_id in the response")
        print("6. Set environment variables:")
        print("   export TELEGRAM_BOT_TOKEN='your_token'")
        print("   export TELEGRAM_CHAT_ID='your_chat_id'")
        return

    # Test connection
    print("\nTesting connection...")
    result = await notifier.test_connection()
    print(f"Result: {result}")

    if result['success']:
        # Send sample signal
        sample_signal = {
            'symbol': 'EURUSD',
            'direction': 'bullish',
            'confidence': 0.75,
            'strength': 0.82,
            'confluence_score': 4,
            'entry_zone': (1.08500, 1.08550),
            'stop_loss': 1.08200,
            'take_profit': [1.08800, 1.09100, 1.09500],
            'risk_reward': 2.5,
            'concepts': ['order_block', 'fair_value_gap', 'premium_discount', 'market_structure'],
            'timestamp': datetime.utcnow().isoformat(),
        }

        print("\nSending sample signal...")
        success = await notifier.send_signal(sample_signal)
        print(f"Signal sent: {success}")


if __name__ == "__main__":
    asyncio.run(test_telegram())
