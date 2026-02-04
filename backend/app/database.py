"""
SQLite Database Module for TradingMamba
FREE local database - no cloud costs!
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import threading

# Database file location
DB_PATH = Path(__file__).parent.parent.parent / "data" / "tradingmamba.db"


class Database:
    """SQLite database manager for TradingMamba"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.db_path = DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Get a database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    youtube_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    duration_seconds INTEGER,
                    published_at TEXT,
                    playlist_id TEXT,
                    playlist_name TEXT,
                    status TEXT DEFAULT 'pending',
                    processed_at TEXT,
                    transcript_quality_score REAL DEFAULT 0,
                    concept_extraction_score REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Transcripts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    text TEXT,
                    confidence REAL DEFAULT 0,
                    FOREIGN KEY (video_id) REFERENCES videos(id)
                )
            """)

            # Smart Money Concepts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ict_concepts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    parent_concept_id TEXT,
                    detection_keywords TEXT,
                    trading_rules TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Concept mentions in videos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_mentions (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    context_text TEXT,
                    confidence_score REAL,
                    FOREIGN KEY (video_id) REFERENCES videos(id),
                    FOREIGN KEY (concept_id) REFERENCES ict_concepts(id)
                )
            """)

            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL,
                    entry_price REAL,
                    entry_zone_low REAL,
                    entry_zone_high REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    risk_reward REAL,
                    factors TEXT,
                    analysis_text TEXT,
                    mtf_bias TEXT,
                    status TEXT DEFAULT 'active',
                    actual_result REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    valid_until TEXT,
                    triggered_at TEXT,
                    closed_at TEXT
                )
            """)

            # Performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_pips REAL,
                    pnl_percent REAL,
                    outcome TEXT,
                    concepts_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)

            # Training history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    training_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    concepts_detected TEXT,
                    rules_extracted TEXT,
                    key_points TEXT,
                    model_version TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos(id)
                )
            """)

            # Users table (for future multi-user support)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    telegram_chat_id TEXT UNIQUE,
                    username TEXT,
                    alert_enabled INTEGER DEFAULT 1,
                    min_confidence REAL DEFAULT 0.65,
                    preferred_symbols TEXT,
                    preferred_timeframes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Backtest results table (Tier 1)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    total_signals INTEGER,
                    win_rate REAL,
                    avg_return_pct REAL,
                    profit_factor REAL,
                    avg_rr REAL,
                    max_drawdown_pct REAL,
                    sample_count INTEGER,
                    lookback_days INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Optimized parameters table (Tier 2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimized_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value REAL NOT NULL,
                    validation_win_rate REAL,
                    validation_profit_factor REAL,
                    train_period_start TEXT,
                    train_period_end TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, param_name)
                )
            """)

            # ML model metrics table (Tier 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    feature_importance TEXT,
                    train_samples INTEGER,
                    test_samples INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Price predictions table (Tier 6 - Genuine ML Predictor)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    prob_bullish REAL,
                    prob_neutral REAL,
                    prob_bearish REAL,
                    price_at_prediction REAL NOT NULL,
                    predicted_at TEXT NOT NULL,
                    feature_snapshot TEXT,
                    model_version TEXT,
                    actual_return REAL,
                    actual_direction TEXT,
                    was_correct INTEGER,
                    resolved_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Predictor metrics table (walk-forward validation results)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictor_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    accuracy REAL,
                    f1_macro REAL,
                    directional_accuracy REAL,
                    n_walk_forward_folds INTEGER,
                    n_train_samples INTEGER,
                    class_distribution TEXT,
                    feature_importance TEXT,
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_youtube_id ON videos(youtube_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_signal ON performance(signal_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimized_params ON optimized_params(symbol, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON price_predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON price_predictions(was_correct)")

            conn.commit()
            print(f"Database initialized at {self.db_path}")

    # ==================== Video Operations ====================

    def save_video(self, video: Dict) -> str:
        """Save or update a video"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO videos
                (id, youtube_id, title, description, duration_seconds, published_at,
                 playlist_id, playlist_name, status, processed_at,
                 transcript_quality_score, concept_extraction_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video.get('id', str(hash(video.get('youtube_id', '')))),
                video.get('youtube_id'),
                video.get('title'),
                video.get('description'),
                video.get('duration_seconds'),
                video.get('published_at'),
                video.get('playlist_id'),
                video.get('playlist_name'),
                video.get('status', 'pending'),
                video.get('processed_at'),
                video.get('transcript_quality_score', 0),
                video.get('concept_extraction_score', 0),
                datetime.utcnow().isoformat()
            ))
            return video.get('id')

    def get_video(self, youtube_id: str) -> Optional[Dict]:
        """Get a video by YouTube ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE youtube_id = ?", (youtube_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_videos(self) -> List[Dict]:
        """Get all videos"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Signal Operations ====================

    def save_signal(self, signal: Dict) -> str:
        """Save a trading signal"""
        signal_id = signal.get('id', str(hash(f"{signal.get('symbol')}{datetime.utcnow()}")))
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Handle take profit list
            take_profits = signal.get('take_profit', [0, 0, 0])
            if isinstance(take_profits, list):
                tp1 = take_profits[0] if len(take_profits) > 0 else 0
                tp2 = take_profits[1] if len(take_profits) > 1 else 0
                tp3 = take_profits[2] if len(take_profits) > 2 else 0
            else:
                tp1, tp2, tp3 = take_profits, 0, 0

            # Handle entry zone
            entry_zone = signal.get('entry_zone', (0, 0))
            if isinstance(entry_zone, (list, tuple)) and len(entry_zone) >= 2:
                ez_low, ez_high = entry_zone[0], entry_zone[1]
            else:
                ez_low, ez_high = 0, 0

            cursor.execute("""
                INSERT INTO signals
                (id, symbol, timeframe, direction, confidence, entry_price,
                 entry_zone_low, entry_zone_high, stop_loss,
                 take_profit_1, take_profit_2, take_profit_3,
                 risk_reward, factors, analysis_text, mtf_bias, status, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                signal.get('symbol'),
                signal.get('timeframe'),
                signal.get('direction'),
                signal.get('confidence'),
                signal.get('entry_price'),
                ez_low,
                ez_high,
                signal.get('stop_loss'),
                tp1, tp2, tp3,
                signal.get('risk_reward'),
                json.dumps(signal.get('factors', [])),
                signal.get('analysis_text'),
                signal.get('mtf_bias'),
                signal.get('status', 'active'),
                signal.get('valid_until')
            ))
            return signal_id

    def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get a signal by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
            row = cursor.fetchone()
            if row:
                signal = dict(row)
                signal['factors'] = json.loads(signal.get('factors', '[]'))
                signal['take_profit'] = [
                    signal.pop('take_profit_1', 0),
                    signal.pop('take_profit_2', 0),
                    signal.pop('take_profit_3', 0)
                ]
                signal['entry_zone'] = (
                    signal.pop('entry_zone_low', 0),
                    signal.pop('entry_zone_high', 0)
                )
                return signal
            return None

    def get_signals(self, symbol: str = None, status: str = None,
                    limit: int = 100) -> List[Dict]:
        """Get signals with optional filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM signals WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            signals = []
            for row in cursor.fetchall():
                signal = dict(row)
                signal['factors'] = json.loads(signal.get('factors', '[]'))
                signal['take_profit'] = [
                    signal.pop('take_profit_1', 0),
                    signal.pop('take_profit_2', 0),
                    signal.pop('take_profit_3', 0)
                ]
                signals.append(signal)
            return signals

    def update_signal_outcome(self, signal_id: str, outcome: str,
                              actual_result: float = None):
        """Update signal outcome"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE signals
                SET status = ?, actual_result = ?, closed_at = ?
                WHERE id = ?
            """, (outcome, actual_result, datetime.utcnow().isoformat(), signal_id))

    # ==================== Performance Operations ====================

    def save_performance(self, perf: Dict) -> str:
        """Save performance record"""
        perf_id = perf.get('id', str(hash(f"{perf.get('signal_id')}{datetime.utcnow()}")))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance
                (id, signal_id, symbol, direction, entry_price, exit_price,
                 pnl_pips, pnl_percent, outcome, concepts_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                perf_id,
                perf.get('signal_id'),
                perf.get('symbol'),
                perf.get('direction'),
                perf.get('entry_price'),
                perf.get('exit_price'),
                perf.get('pnl_pips'),
                perf.get('pnl_percent'),
                perf.get('outcome'),
                json.dumps(perf.get('concepts_used', []))
            ))
            return perf_id

    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total signals
            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]

            # Completed signals
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE status IN ('hit_tp', 'hit_sl', 'closed')
            """)
            completed = cursor.fetchone()[0]

            # Winning signals
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE status = 'hit_tp' OR actual_result > 0
            """)
            wins = cursor.fetchone()[0]

            # Total PnL
            cursor.execute("SELECT SUM(actual_result) FROM signals WHERE actual_result IS NOT NULL")
            total_pnl = cursor.fetchone()[0] or 0

            # By direction
            cursor.execute("""
                SELECT direction, COUNT(*) as count,
                       SUM(CASE WHEN actual_result > 0 THEN 1 ELSE 0 END) as wins
                FROM signals
                WHERE actual_result IS NOT NULL
                GROUP BY direction
            """)
            by_direction = {row[0]: {'total': row[1], 'wins': row[2]}
                          for row in cursor.fetchall()}

            win_rate = (wins / completed * 100) if completed > 0 else 0

            return {
                'total_signals': total_signals,
                'completed_signals': completed,
                'wins': wins,
                'losses': completed - wins,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'by_direction': by_direction
            }

    # ==================== Training History ====================

    def save_training_record(self, record: Dict) -> str:
        """Save training history record"""
        record_id = record.get('id', str(hash(f"{record.get('video_id')}{datetime.utcnow()}")))
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_history
                (id, video_id, concepts_detected, rules_extracted, key_points, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                record.get('video_id'),
                json.dumps(record.get('concepts_detected', [])),
                json.dumps(record.get('rules_extracted', [])),
                json.dumps(record.get('key_points', [])),
                record.get('model_version', '1.0')
            ))
            return record_id

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(DISTINCT video_id) FROM training_history")
            videos_trained = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_history")
            total_sessions = cursor.fetchone()[0]

            cursor.execute("""
                SELECT MAX(training_date) FROM training_history
            """)
            last_training = cursor.fetchone()[0]

            return {
                'videos_trained': videos_trained,
                'total_training_sessions': total_sessions,
                'last_training': last_training
            }

    # ==================== Price Prediction Operations ====================

    def save_prediction(self, pred: Dict) -> int:
        """Save a price prediction"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO price_predictions
                (symbol, timeframe, horizon, direction, confidence,
                 prob_bullish, prob_neutral, prob_bearish,
                 price_at_prediction, predicted_at, feature_snapshot, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred['symbol'], pred['timeframe'], pred['horizon'],
                pred['direction'], pred['confidence'],
                pred.get('prob_bullish', 0), pred.get('prob_neutral', 0),
                pred.get('prob_bearish', 0),
                pred['price_at_prediction'], pred['predicted_at'],
                pred.get('feature_snapshot', '{}'), pred.get('model_version', ''),
            ))
            return cursor.lastrowid

    def get_pending_predictions(self) -> List[Dict]:
        """Get all unresolved predictions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM price_predictions
                WHERE was_correct IS NULL
                ORDER BY created_at ASC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def resolve_prediction(self, pred_id: int, actual_return: float,
                           actual_direction: str, was_correct: int):
        """Resolve a prediction with actual outcome"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE price_predictions
                SET actual_return = ?, actual_direction = ?,
                    was_correct = ?, resolved_at = ?
                WHERE id = ?
            """, (actual_return, actual_direction, was_correct,
                  datetime.utcnow().isoformat(), pred_id))

    def get_prediction_performance(self, symbol: str = None,
                                    lookback_days: int = 90) -> Dict:
        """Get prediction performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - __import__('datetime').timedelta(days=lookback_days)).isoformat()

            # Build query
            where = "WHERE was_correct IS NOT NULL AND created_at > ?"
            params = [cutoff]
            if symbol:
                where += " AND symbol = ?"
                params.append(symbol)

            # Overall stats
            cursor.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(was_correct) as correct,
                       AVG(was_correct) as accuracy
                FROM price_predictions {where}
            """, params)
            row = cursor.fetchone()
            overall = {
                'total': row['total'] or 0,
                'correct': row['correct'] or 0,
                'accuracy': round(row['accuracy'] or 0, 4),
            }

            # By symbol
            cursor.execute(f"""
                SELECT symbol,
                       COUNT(*) as total,
                       SUM(was_correct) as correct,
                       AVG(was_correct) as accuracy
                FROM price_predictions {where}
                GROUP BY symbol
            """, params)
            by_symbol = {
                row['symbol']: {
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'accuracy': round(row['accuracy'] or 0, 4),
                }
                for row in cursor.fetchall()
            }

            # By horizon
            cursor.execute(f"""
                SELECT horizon,
                       COUNT(*) as total,
                       SUM(was_correct) as correct,
                       AVG(was_correct) as accuracy
                FROM price_predictions {where}
                GROUP BY horizon
            """, params)
            by_horizon = {
                str(row['horizon']): {
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'accuracy': round(row['accuracy'] or 0, 4),
                }
                for row in cursor.fetchall()
            }

            return {
                'overall': overall,
                'by_symbol': by_symbol,
                'by_horizon': by_horizon,
                'lookback_days': lookback_days,
            }

    def get_recent_predictions(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """Get recent predictions with outcomes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM price_predictions"
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def save_predictor_metrics(self, metrics: Dict):
        """Save predictor training metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictor_metrics
                (symbol, timeframe, horizon, accuracy, f1_macro,
                 directional_accuracy, n_walk_forward_folds, n_train_samples,
                 class_distribution, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics['symbol'], metrics['timeframe'], metrics['horizon'],
                metrics['accuracy'], metrics['f1_macro'],
                metrics['directional_accuracy'],
                metrics['n_walk_forward_folds'], metrics['n_train_samples'],
                metrics.get('class_distribution', '{}'),
                metrics.get('feature_importance', '{}'),
            ))

    def get_predictor_metrics(self, symbol: str = None) -> List[Dict]:
        """Get predictor training metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM predictor_metrics
                    WHERE symbol = ?
                    ORDER BY trained_at DESC
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT * FROM predictor_metrics
                    ORDER BY trained_at DESC
                """)
            return [dict(row) for row in cursor.fetchall()]


# Singleton instance
db = Database()
