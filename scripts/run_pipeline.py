#!/usr/bin/env python3
"""
TradingMamba - Main Pipeline Orchestrator

Runs the complete ICT AI Trading System pipeline:
1. Load transcripts from all playlists
2. Extract ICT concepts
3. Train ML models
4. Generate trading signals
5. Track and evaluate performance

100% FREE - No paid APIs required.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.ml.training_pipeline import ICTKnowledgeBase, run_training_pipeline
from app.ml.signal_fusion import SignalGenerator, ICTSignal
from app.ml.technical_analysis import FullICTAnalysis
from app.ml.model_evaluator import PerformanceDashboard, PredictionTracker

# Add user packages
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')


def print_banner():
    """Print TradingMamba banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗     ║
║   ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝     ║
║      ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗    ║
║      ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║    ║
║      ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝    ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝     ║
║                                                              ║
║   ███╗   ███╗ █████╗ ███╗   ███╗██████╗  █████╗             ║
║   ████╗ ████║██╔══██╗████╗ ████║██╔══██╗██╔══██╗            ║
║   ██╔████╔██║███████║██╔████╔██║██████╔╝███████║            ║
║   ██║╚██╔╝██║██╔══██║██║╚██╔╝██║██╔══██╗██╔══██║            ║
║   ██║ ╚═╝ ██║██║  ██║██║ ╚═╝ ██║██████╔╝██║  ██║            ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝            ║
║                                                              ║
║   ICT AI Trading Signal System - 100% FREE                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


class TradingMambaPipeline:
    """
    Main pipeline orchestrator for TradingMamba.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.playlists_dir = self.data_dir / "playlists"
        self.models_dir = self.data_dir / "ml_models"

        # Components
        self.knowledge_base = None
        self.signal_generator = None
        self.analyzer = None
        self.dashboard = None

    def check_data_status(self) -> dict:
        """Check status of available data"""
        status = {
            'transcripts': 0,
            'playlists': 0,
            'total_words': 0,
            'models_trained': False
        }

        # Count transcripts
        if self.transcripts_dir.exists():
            transcript_files = list(self.transcripts_dir.glob("*.json"))
            status['transcripts'] = len(transcript_files)

            # Count words
            for tf in transcript_files:
                try:
                    with open(tf) as f:
                        data = json.load(f)
                        status['total_words'] += data.get('word_count', 0)
                except:
                    pass

        # Count playlists
        if self.playlists_dir.exists():
            status['playlists'] = len(list(self.playlists_dir.glob("*.json")))

        # Check if models exist
        if self.models_dir.exists():
            status['models_trained'] = (self.models_dir / "knowledge_base.json").exists()

        return status

    def run_training(self, incremental: bool = False) -> dict:
        """Run the ML training pipeline"""
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING ML MODELS")
        print("=" * 60)

        self.knowledge_base = ICTKnowledgeBase(str(self.data_dir))

        if incremental:
            try:
                self.knowledge_base.load()
                print("Loaded existing knowledge base for incremental training")
            except:
                print("No existing knowledge base found, starting fresh")

        results = self.knowledge_base.train(incremental=incremental)
        self.knowledge_base.save()

        return results

    def initialize_components(self):
        """Initialize all pipeline components"""
        print("\nInitializing components...")

        # Knowledge base
        self.knowledge_base = ICTKnowledgeBase(str(self.data_dir))
        try:
            self.knowledge_base.load()
            print("  ✓ Knowledge base loaded")
        except:
            print("  ! Knowledge base not found - run training first")

        # Signal generator
        self.signal_generator = SignalGenerator(str(self.data_dir))
        try:
            self.signal_generator.load()
            print("  ✓ Signal generator loaded")
        except:
            print("  ! Signal generator not found")

        # Technical analyzer
        self.analyzer = FullICTAnalysis()
        print("  ✓ Technical analyzer initialized")

        # Performance dashboard
        self.dashboard = PerformanceDashboard(str(self.data_dir))
        try:
            self.dashboard.load()
            print("  ✓ Performance dashboard loaded")
        except:
            print("  ! No historical performance data")

    def analyze_symbol(self, symbol: str, timeframes: list = None) -> dict:
        """Run ICT analysis on a symbol"""
        if timeframes is None:
            timeframes = ['H1', 'H4', 'D1']

        print(f"\n📊 Analyzing {symbol}...")

        try:
            # Import market data service
            from app.services.free_market_data import FreeMarketDataService
            market_service = FreeMarketDataService()

            # Fetch data for each timeframe
            market_data = {}
            for tf in timeframes:
                df = market_service.get_ohlcv(symbol, tf, limit=200)
                if df is not None and not df.empty:
                    market_data[tf] = df
                    print(f"  ✓ {tf}: {len(df)} candles")
                else:
                    print(f"  ! {tf}: No data")

            if not market_data:
                return {'error': 'No market data available'}

            # Run ICT analysis on each timeframe
            analyses = {}
            detected_concepts = []

            for tf, df in market_data.items():
                analysis = self.analyzer.analyze(df, tf)
                analyses[tf] = analysis
                detected_concepts.extend(analysis.get('detected_concepts', []))

            # Deduplicate concepts
            detected_concepts = list(set(detected_concepts))

            # Generate signal
            if self.signal_generator and self.knowledge_base:
                signal = self.signal_generator.generate_signal(
                    symbol=symbol,
                    market_data=market_data,
                    detected_concepts=detected_concepts,
                    context={'is_fresh': True}
                )
            else:
                signal = None

            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'timeframes_analyzed': list(market_data.keys()),
                'analyses': analyses,
                'detected_concepts': detected_concepts,
                'signal': signal.__dict__ if signal else None,
            }

        except ImportError as e:
            return {'error': f'Missing dependency: {e}'}
        except Exception as e:
            return {'error': str(e)}

    def generate_daily_report(self) -> dict:
        """Generate daily performance and analysis report"""
        print("\n" + "=" * 60)
        print("DAILY REPORT")
        print("=" * 60)

        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'data_status': self.check_data_status(),
            'analyses': [],
        }

        # Analyze key symbols
        symbols = ['EURUSD', 'XAUUSD', 'US30', 'GBPUSD']

        for symbol in symbols:
            try:
                analysis = self.analyze_symbol(symbol)
                if 'error' not in analysis:
                    report['analyses'].append({
                        'symbol': symbol,
                        'bias': analysis['analyses'].get('H4', {}).get('market_structure', {}).get('bias', 'neutral'),
                        'concepts': analysis['detected_concepts'],
                        'signal': analysis.get('signal'),
                    })
            except Exception as e:
                print(f"  ! Error analyzing {symbol}: {e}")

        # Add learning progress
        if self.knowledge_base:
            report['learning_progress'] = self.knowledge_base.get_learning_progress()

        # Add performance metrics
        if self.dashboard:
            report['performance'] = self.dashboard.get_summary_report()

        return report

    def run_full_pipeline(self, train: bool = True, analyze: bool = True):
        """Run the complete pipeline"""
        print_banner()

        print(f"\n📅 Run Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # Check data status
        status = self.check_data_status()
        print(f"\n📊 Data Status:")
        print(f"   Playlists: {status['playlists']}")
        print(f"   Transcripts: {status['transcripts']}")
        print(f"   Total Words: {status['total_words']:,}")
        print(f"   Models Trained: {'Yes' if status['models_trained'] else 'No'}")

        if status['transcripts'] == 0:
            print("\n⚠️  No transcripts found! Run transcript collection first:")
            print("   python scripts/get_transcripts.py --all")
            return

        # Training phase
        if train:
            training_results = self.run_training(incremental=status['models_trained'])
            print(f"\n✓ Training complete. F1 Score: {training_results.get('components', {}).get('classifier', {}).get('ensemble_f1', 'N/A')}")

        # Initialize components
        self.initialize_components()

        # Analysis phase
        if analyze:
            report = self.generate_daily_report()

            print("\n" + "=" * 60)
            print("SIGNAL SUMMARY")
            print("=" * 60)

            for analysis in report.get('analyses', []):
                signal = analysis.get('signal')
                if signal:
                    print(f"\n{analysis['symbol']}:")
                    print(f"   Direction: {signal.get('direction', 'N/A')}")
                    print(f"   Strength: {signal.get('strength', 0):.2f}")
                    print(f"   Confidence: {signal.get('confidence', 0):.2%}")
                    print(f"   Concepts: {', '.join(signal.get('concepts', []))}")
                else:
                    print(f"\n{analysis['symbol']}: No signal (insufficient confluence)")

            # Save report
            report_path = self.data_dir / f"daily_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Report saved: {report_path}")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='TradingMamba - ICT AI Trading Signal System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --train            # Train models only
  python run_pipeline.py --analyze EURUSD   # Analyze specific symbol
  python run_pipeline.py --status           # Check data status
  python run_pipeline.py --report           # Generate daily report
        """
    )

    parser.add_argument('--train', action='store_true', help='Run training only')
    parser.add_argument('--analyze', type=str, help='Analyze specific symbol')
    parser.add_argument('--status', action='store_true', help='Show data status')
    parser.add_argument('--report', action='store_true', help='Generate daily report')
    parser.add_argument('--no-train', action='store_true', help='Skip training in full pipeline')

    args = parser.parse_args()

    pipeline = TradingMambaPipeline()

    if args.status:
        print_banner()
        status = pipeline.check_data_status()
        print("\n📊 Data Status:")
        print(f"   Playlists: {status['playlists']}")
        print(f"   Transcripts: {status['transcripts']}")
        print(f"   Total Words: {status['total_words']:,}")
        print(f"   Models Trained: {'Yes' if status['models_trained'] else 'No'}")

    elif args.train:
        print_banner()
        pipeline.run_training(incremental=True)

    elif args.analyze:
        print_banner()
        pipeline.initialize_components()
        result = pipeline.analyze_symbol(args.analyze)

        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print(f"\n📊 Analysis for {args.analyze}:")
            print(f"   Detected Concepts: {result['detected_concepts']}")

            if result.get('signal'):
                signal = result['signal']
                print(f"\n🎯 Signal:")
                print(f"   Direction: {signal['direction']}")
                print(f"   Strength: {signal['strength']:.2f}")
                print(f"   Confidence: {signal['confidence']:.2%}")
                print(f"   Entry Zone: {signal['entry_zone']}")
                print(f"   Stop Loss: {signal['stop_loss']}")
                print(f"   Take Profits: {signal['take_profit']}")
            else:
                print("\n   No signal (insufficient confluence)")

    elif args.report:
        print_banner()
        pipeline.initialize_components()
        report = pipeline.generate_daily_report()
        print(json.dumps(report, indent=2, default=str))

    else:
        # Full pipeline
        pipeline.run_full_pipeline(train=not args.no_train, analyze=True)


if __name__ == "__main__":
    main()
