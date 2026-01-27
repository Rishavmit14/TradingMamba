# ICT AI Trading System - Technical Implementation Guide

## Quick Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                                    │
│                                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│   │  YOUTUBE    │    │   MARKET    │    │    AI/ML    │    │   USER      │     │
│   │  PIPELINE   │───▶│   DATA      │───▶│   ENGINE    │───▶│   INTERFACE │     │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │                   │             │
│         ▼                  ▼                  ▼                   ▼             │
│   ┌─────────────────────────────────────────────────────────────────────┐      │
│   │                        POSTGRESQL + REDIS                            │      │
│   └─────────────────────────────────────────────────────────────────────┘      │
│                                    │                                            │
│                                    ▼                                            │
│                          ┌─────────────────┐                                   │
│                          │   WHATSAPP      │                                   │
│                          │   NOTIFICATIONS │                                   │
│                          └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ict-ai-trading/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py               # Configuration
│   │   ├── models/                 # Database models
│   │   │   ├── __init__.py
│   │   │   ├── video.py
│   │   │   ├── signal.py
│   │   │   ├── user.py
│   │   │   └── concept.py
│   │   ├── api/                    # API routes
│   │   │   ├── __init__.py
│   │   │   ├── signals.py
│   │   │   ├── analysis.py
│   │   │   ├── videos.py
│   │   │   └── users.py
│   │   ├── services/               # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── video_processor.py
│   │   │   ├── market_data.py
│   │   │   ├── ict_analyzer.py
│   │   │   ├── signal_generator.py
│   │   │   └── notification.py
│   │   ├── ml/                     # Machine learning
│   │   │   ├── __init__.py
│   │   │   ├── concept_model.py
│   │   │   ├── pattern_model.py
│   │   │   ├── prediction_model.py
│   │   │   └── training.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── chart_generator.py
│   │       └── helpers.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── app/                    # Next.js App Router
│   │   │   ├── page.tsx            # Dashboard
│   │   │   ├── signals/
│   │   │   ├── analysis/
│   │   │   ├── learning/
│   │   │   └── settings/
│   │   ├── components/
│   │   │   ├── ui/                 # shadcn components
│   │   │   ├── charts/
│   │   │   ├── signals/
│   │   │   └── layout/
│   │   ├── lib/
│   │   │   ├── api.ts
│   │   │   └── utils.ts
│   │   └── hooks/
│   ├── package.json
│   └── Dockerfile
│
├── ml-training/
│   ├── notebooks/
│   │   ├── concept_extraction.ipynb
│   │   ├── pattern_detection.ipynb
│   │   └── price_prediction.ipynb
│   ├── data/
│   └── models/
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Core Implementation Code

### 1. Video Processing Pipeline

```python
# backend/app/services/video_processor.py

import yt_dlp
import whisper
from anthropic import Anthropic
from pinecone import Pinecone
import json
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ProcessedVideo:
    video_id: str
    title: str
    transcript: List[Dict]
    concepts: List[Dict]
    embeddings: List[float]

class VideoProcessor:
    """Process ICT YouTube videos into structured knowledge"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model("large-v3")
        self.anthropic = Anthropic()
        self.pinecone = Pinecone()
        self.index = self.pinecone.Index("ict-knowledge")
        
        # ICT concept keywords for detection
        self.concept_keywords = self._load_concept_keywords()
    
    def process_playlist(self, playlist_url: str) -> List[ProcessedVideo]:
        """Process entire playlist"""
        videos = self._get_playlist_videos(playlist_url)
        processed = []
        
        for video in videos:
            try:
                result = self.process_single_video(video['id'])
                processed.append(result)
            except Exception as e:
                print(f"Error processing {video['id']}: {e}")
                continue
        
        return processed
    
    def process_single_video(self, video_id: str) -> ProcessedVideo:
        """Process single video"""
        
        # Step 1: Download audio
        audio_path = self._download_audio(video_id)
        
        # Step 2: Transcribe
        transcript = self._transcribe(audio_path)
        
        # Step 3: Extract ICT concepts using Claude
        concepts = self._extract_concepts(transcript)
        
        # Step 4: Generate embeddings
        embeddings = self._generate_embeddings(transcript, concepts)
        
        # Step 5: Store in vector DB
        self._store_in_vectordb(video_id, transcript, concepts, embeddings)
        
        return ProcessedVideo(
            video_id=video_id,
            title=self._get_video_title(video_id),
            transcript=transcript,
            concepts=concepts,
            embeddings=embeddings
        )
    
    def _download_audio(self, video_id: str) -> str:
        """Download audio from YouTube video"""
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'/tmp/{video_id}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        
        return f'/tmp/{video_id}.mp3'
    
    def _transcribe(self, audio_path: str) -> List[Dict]:
        """Transcribe audio with timestamps"""
        result = self.whisper_model.transcribe(
            audio_path,
            verbose=False,
            word_timestamps=True
        )
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'words': segment.get('words', [])
            })
        
        return segments
    
    def _extract_concepts(self, transcript: List[Dict]) -> List[Dict]:
        """Use Claude to extract ICT concepts from transcript"""
        
        full_text = "\n".join([s['text'] for s in transcript])
        
        prompt = f"""Analyze this ICT (Inner Circle Trader) video transcript and extract all trading concepts mentioned.

For each concept found, provide:
1. concept_name: The ICT term (e.g., "Order Block", "Fair Value Gap")
2. category: One of [market_structure, entry_model, key_level, time_based, institutional]
3. explanation: How it was explained in this video
4. timestamp_mentions: Approximate times it was discussed
5. related_concepts: Other concepts it connects to
6. trading_rules: Any specific rules or criteria mentioned

Transcript:
{full_text}

Return as JSON array of concepts."""

        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        concepts = json.loads(response.content[0].text)
        return concepts
    
    def _generate_embeddings(self, transcript: List[Dict], concepts: List[Dict]):
        """Generate embeddings for semantic search"""
        # Implementation using OpenAI embeddings or similar
        pass
    
    def _store_in_vectordb(self, video_id: str, transcript: List[Dict], 
                           concepts: List[Dict], embeddings: List[float]):
        """Store in Pinecone for retrieval"""
        
        vectors = []
        for i, (segment, embedding) in enumerate(zip(transcript, embeddings)):
            vectors.append({
                'id': f"{video_id}_{i}",
                'values': embedding,
                'metadata': {
                    'video_id': video_id,
                    'text': segment['text'],
                    'start_time': segment['start'],
                    'concepts': [c['concept_name'] for c in concepts 
                                if c.get('timestamp_mentions') and 
                                segment['start'] in c['timestamp_mentions']]
                }
            })
        
        self.index.upsert(vectors=vectors)
```

### 2. ICT Analysis Engine

```python
# backend/app/services/ict_analyzer.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class MarketStructure(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    CONSOLIDATION = "consolidation"

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'high' or 'low'
    timestamp: pd.Timestamp

@dataclass
class OrderBlock:
    start_index: int
    end_index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    mitigated: bool
    timestamp: pd.Timestamp

@dataclass
class FairValueGap:
    index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    filled: bool
    timestamp: pd.Timestamp

class ICTAnalyzer:
    """Core ICT methodology analysis engine"""
    
    def __init__(self, lookback_swing: int = 5):
        self.lookback_swing = lookback_swing
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Complete ICT analysis on OHLCV data
        
        Parameters:
        - data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
        - Complete analysis dictionary
        """
        
        # Find swing points
        swing_points = self.find_swing_points(data)
        
        # Determine market structure
        structure = self.analyze_market_structure(swing_points)
        
        # Find order blocks
        order_blocks = self.find_order_blocks(data, structure)
        
        # Find fair value gaps
        fvgs = self.find_fair_value_gaps(data)
        
        # Find liquidity levels
        liquidity = self.find_liquidity_levels(swing_points, data)
        
        # Calculate premium/discount
        premium_discount = self.calculate_premium_discount(data, swing_points)
        
        # Determine bias
        bias = self.determine_bias(structure, premium_discount)
        
        return {
            'swing_points': swing_points,
            'market_structure': structure,
            'order_blocks': order_blocks,
            'fair_value_gaps': fvgs,
            'liquidity_levels': liquidity,
            'premium_discount': premium_discount,
            'bias': bias,
            'current_price': data['close'].iloc[-1]
        }
    
    def find_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Identify swing highs and lows"""
        
        swing_points = []
        highs = data['high'].values
        lows = data['low'].values
        
        for i in range(self.lookback_swing, len(data) - self.lookback_swing):
            # Check for swing high
            if highs[i] == max(highs[i-self.lookback_swing:i+self.lookback_swing+1]):
                swing_points.append(SwingPoint(
                    index=i,
                    price=highs[i],
                    type='high',
                    timestamp=data.index[i]
                ))
            
            # Check for swing low
            if lows[i] == min(lows[i-self.lookback_swing:i+self.lookback_swing+1]):
                swing_points.append(SwingPoint(
                    index=i,
                    price=lows[i],
                    type='low',
                    timestamp=data.index[i]
                ))
        
        return sorted(swing_points, key=lambda x: x.index)
    
    def analyze_market_structure(self, swing_points: List[SwingPoint]) -> Dict:
        """
        Analyze market structure for BOS and CHoCH
        
        Returns structure analysis including:
        - Current structure (bullish/bearish/consolidation)
        - Recent BOS/CHoCH events
        - Structure breaks
        """
        
        if len(swing_points) < 4:
            return {'structure': MarketStructure.CONSOLIDATION, 'events': []}
        
        events = []
        recent_highs = [sp for sp in swing_points if sp.type == 'high'][-4:]
        recent_lows = [sp for sp in swing_points if sp.type == 'low'][-4:]
        
        # Check for bullish structure (HH, HL)
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1].price > recent_highs[-2].price
            hl = recent_lows[-1].price > recent_lows[-2].price
            
            lh = recent_highs[-1].price < recent_highs[-2].price
            ll = recent_lows[-1].price < recent_lows[-2].price
            
            if hh and hl:
                structure = MarketStructure.BULLISH
                events.append({
                    'type': 'bullish_structure',
                    'description': 'Higher High and Higher Low confirmed'
                })
            elif lh and ll:
                structure = MarketStructure.BEARISH
                events.append({
                    'type': 'bearish_structure',
                    'description': 'Lower High and Lower Low confirmed'
                })
            else:
                structure = MarketStructure.CONSOLIDATION
        else:
            structure = MarketStructure.CONSOLIDATION
        
        # Check for BOS
        bos = self._check_bos(swing_points, structure)
        if bos:
            events.append(bos)
        
        # Check for CHoCH
        choch = self._check_choch(swing_points, structure)
        if choch:
            events.append(choch)
        
        return {
            'structure': structure,
            'events': events,
            'recent_highs': recent_highs,
            'recent_lows': recent_lows
        }
    
    def _check_bos(self, swing_points: List[SwingPoint], 
                   current_structure: MarketStructure) -> Optional[Dict]:
        """Check for Break of Structure"""
        
        recent_highs = [sp for sp in swing_points if sp.type == 'high'][-3:]
        recent_lows = [sp for sp in swing_points if sp.type == 'low'][-3:]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None
        
        if current_structure == MarketStructure.BULLISH:
            # BOS = break above previous swing high
            if recent_highs[-1].price > recent_highs[-2].price:
                return {
                    'type': 'bos_bullish',
                    'level': recent_highs[-2].price,
                    'timestamp': recent_highs[-1].timestamp,
                    'description': f'Bullish BOS at {recent_highs[-2].price}'
                }
        
        elif current_structure == MarketStructure.BEARISH:
            # BOS = break below previous swing low
            if recent_lows[-1].price < recent_lows[-2].price:
                return {
                    'type': 'bos_bearish',
                    'level': recent_lows[-2].price,
                    'timestamp': recent_lows[-1].timestamp,
                    'description': f'Bearish BOS at {recent_lows[-2].price}'
                }
        
        return None
    
    def _check_choch(self, swing_points: List[SwingPoint], 
                     current_structure: MarketStructure) -> Optional[Dict]:
        """Check for Change of Character"""
        
        # CHoCH occurs when structure changes
        # e.g., in bullish trend, price breaks below last HL
        
        recent_highs = [sp for sp in swing_points if sp.type == 'high'][-3:]
        recent_lows = [sp for sp in swing_points if sp.type == 'low'][-3:]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None
        
        # Simplified CHoCH detection
        if current_structure == MarketStructure.BULLISH:
            if recent_lows[-1].price < recent_lows[-2].price:
                return {
                    'type': 'choch_bearish',
                    'level': recent_lows[-2].price,
                    'timestamp': recent_lows[-1].timestamp,
                    'description': 'Bearish CHoCH - potential trend reversal'
                }
        
        elif current_structure == MarketStructure.BEARISH:
            if recent_highs[-1].price > recent_highs[-2].price:
                return {
                    'type': 'choch_bullish',
                    'level': recent_highs[-2].price,
                    'timestamp': recent_highs[-1].timestamp,
                    'description': 'Bullish CHoCH - potential trend reversal'
                }
        
        return None
    
    def find_order_blocks(self, data: pd.DataFrame, 
                          structure: Dict) -> List[OrderBlock]:
        """
        Find order blocks
        
        Bullish OB: Last bearish candle before bullish move
        Bearish OB: Last bullish candle before bearish move
        """
        
        order_blocks = []
        current_price = data['close'].iloc[-1]
        
        for i in range(2, len(data) - 1):
            # Bullish Order Block
            # Bearish candle followed by strong bullish move
            if (data['close'].iloc[i] < data['open'].iloc[i] and  # Bearish candle
                data['close'].iloc[i+1] > data['high'].iloc[i]):  # Bullish break
                
                ob = OrderBlock(
                    start_index=i,
                    end_index=i,
                    high=data['high'].iloc[i],
                    low=data['low'].iloc[i],
                    type='bullish',
                    mitigated=current_price < data['low'].iloc[i],
                    timestamp=data.index[i]
                )
                order_blocks.append(ob)
            
            # Bearish Order Block
            # Bullish candle followed by strong bearish move
            if (data['close'].iloc[i] > data['open'].iloc[i] and  # Bullish candle
                data['close'].iloc[i+1] < data['low'].iloc[i]):  # Bearish break
                
                ob = OrderBlock(
                    start_index=i,
                    end_index=i,
                    high=data['high'].iloc[i],
                    low=data['low'].iloc[i],
                    type='bearish',
                    mitigated=current_price > data['high'].iloc[i],
                    timestamp=data.index[i]
                )
                order_blocks.append(ob)
        
        # Return only unmitigated order blocks
        return [ob for ob in order_blocks if not ob.mitigated][-10:]  # Last 10
    
    def find_fair_value_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """
        Find Fair Value Gaps (Imbalances)
        
        Bullish FVG: Gap between candle 1 high and candle 3 low
        Bearish FVG: Gap between candle 1 low and candle 3 high
        """
        
        fvgs = []
        current_price = data['close'].iloc[-1]
        
        for i in range(2, len(data)):
            # Bullish FVG
            if data['low'].iloc[i] > data['high'].iloc[i-2]:
                gap_high = data['low'].iloc[i]
                gap_low = data['high'].iloc[i-2]
                
                fvgs.append(FairValueGap(
                    index=i-1,
                    high=gap_high,
                    low=gap_low,
                    type='bullish',
                    filled=current_price < gap_low,
                    timestamp=data.index[i-1]
                ))
            
            # Bearish FVG
            if data['high'].iloc[i] < data['low'].iloc[i-2]:
                gap_high = data['low'].iloc[i-2]
                gap_low = data['high'].iloc[i]
                
                fvgs.append(FairValueGap(
                    index=i-1,
                    high=gap_high,
                    low=gap_low,
                    type='bearish',
                    filled=current_price > gap_high,
                    timestamp=data.index[i-1]
                ))
        
        # Return unfilled FVGs
        return [fvg for fvg in fvgs if not fvg.filled][-10:]
    
    def find_liquidity_levels(self, swing_points: List[SwingPoint], 
                              data: pd.DataFrame) -> Dict:
        """Find liquidity pools (equal highs/lows, swing points)"""
        
        current_price = data['close'].iloc[-1]
        
        # Buy-side liquidity (above current price)
        buy_side = [
            {
                'level': sp.price,
                'type': 'swing_high',
                'timestamp': sp.timestamp
            }
            for sp in swing_points 
            if sp.type == 'high' and sp.price > current_price
        ]
        
        # Sell-side liquidity (below current price)
        sell_side = [
            {
                'level': sp.price,
                'type': 'swing_low',
                'timestamp': sp.timestamp
            }
            for sp in swing_points 
            if sp.type == 'low' and sp.price < current_price
        ]
        
        # Find equal highs/lows (within tolerance)
        equal_highs = self._find_equal_levels(
            [sp for sp in swing_points if sp.type == 'high'],
            tolerance=0.001
        )
        
        equal_lows = self._find_equal_levels(
            [sp for sp in swing_points if sp.type == 'low'],
            tolerance=0.001
        )
        
        return {
            'buy_side_liquidity': sorted(buy_side, key=lambda x: x['level'])[:5],
            'sell_side_liquidity': sorted(sell_side, key=lambda x: x['level'], reverse=True)[:5],
            'equal_highs': equal_highs,
            'equal_lows': equal_lows
        }
    
    def _find_equal_levels(self, swing_points: List[SwingPoint], 
                           tolerance: float = 0.001) -> List[Dict]:
        """Find equal highs or lows within tolerance"""
        
        equal_levels = []
        
        for i, sp1 in enumerate(swing_points):
            for sp2 in swing_points[i+1:]:
                if abs(sp1.price - sp2.price) / sp1.price < tolerance:
                    equal_levels.append({
                        'level': (sp1.price + sp2.price) / 2,
                        'points': [sp1, sp2]
                    })
        
        return equal_levels
    
    def calculate_premium_discount(self, data: pd.DataFrame, 
                                   swing_points: List[SwingPoint]) -> Dict:
        """
        Calculate premium/discount zones
        
        Premium: Above equilibrium (50%) - look for sells
        Discount: Below equilibrium (50%) - look for buys
        """
        
        recent_highs = [sp for sp in swing_points if sp.type == 'high']
        recent_lows = [sp for sp in swing_points if sp.type == 'low']
        
        if not recent_highs or not recent_lows:
            return {'zone': 'neutral', 'percentage': 50}
        
        range_high = max(sp.price for sp in recent_highs[-5:])
        range_low = min(sp.price for sp in recent_lows[-5:])
        
        equilibrium = (range_high + range_low) / 2
        current_price = data['close'].iloc[-1]
        
        # Calculate position in range (0-100%)
        range_size = range_high - range_low
        if range_size == 0:
            percentage = 50
        else:
            percentage = (current_price - range_low) / range_size * 100
        
        # Determine zone
        if percentage >= 70:
            zone = 'premium'
        elif percentage <= 30:
            zone = 'discount'
        else:
            zone = 'equilibrium'
        
        return {
            'zone': zone,
            'percentage': round(percentage, 2),
            'range_high': range_high,
            'range_low': range_low,
            'equilibrium': equilibrium,
            'current_price': current_price
        }
    
    def determine_bias(self, structure: Dict, 
                       premium_discount: Dict) -> Dict:
        """
        Determine overall bias based on structure and price position
        """
        
        ms = structure['structure']
        zone = premium_discount['zone']
        
        # Simple bias logic
        if ms == MarketStructure.BULLISH:
            if zone == 'discount':
                bias = Bias.BULLISH
                confidence = 0.8
                reasoning = "Bullish structure + price in discount zone"
            elif zone == 'equilibrium':
                bias = Bias.BULLISH
                confidence = 0.6
                reasoning = "Bullish structure + price at equilibrium"
            else:  # premium
                bias = Bias.NEUTRAL
                confidence = 0.4
                reasoning = "Bullish structure but price in premium zone"
        
        elif ms == MarketStructure.BEARISH:
            if zone == 'premium':
                bias = Bias.BEARISH
                confidence = 0.8
                reasoning = "Bearish structure + price in premium zone"
            elif zone == 'equilibrium':
                bias = Bias.BEARISH
                confidence = 0.6
                reasoning = "Bearish structure + price at equilibrium"
            else:  # discount
                bias = Bias.NEUTRAL
                confidence = 0.4
                reasoning = "Bearish structure but price in discount zone"
        
        else:  # consolidation
            bias = Bias.NEUTRAL
            confidence = 0.3
            reasoning = "Market in consolidation"
        
        return {
            'bias': bias,
            'confidence': confidence,
            'reasoning': reasoning
        }
```

### 3. Signal Generator

```python
# backend/app/services/signal_generator.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uuid

@dataclass
class TradingSignal:
    id: str
    symbol: str
    timeframe: str
    direction: str  # 'BUY', 'SELL', 'WAIT'
    confidence: float
    entry_price: float
    entry_zone: tuple
    stop_loss: float
    take_profit: List[float]
    risk_reward: float
    factors: List[str]
    analysis_text: str
    mtf_bias: str
    key_levels: List[Dict]
    created_at: datetime
    valid_until: datetime

class SignalGenerator:
    """Generate trading signals based on ICT analysis"""
    
    def __init__(self, ict_analyzer: ICTAnalyzer):
        self.analyzer = ict_analyzer
        self.min_confidence = 0.65
        self.min_rr = 2.0  # Minimum risk:reward
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                        timeframe: str) -> TradingSignal:
        """
        Generate complete trading signal
        
        ICT Signal Criteria:
        1. Higher TF bias alignment
        2. Valid order block
        3. Fair value gap for entry
        4. Liquidity target identified
        5. Kill zone timing (bonus)
        """
        
        # Run ICT analysis
        analysis = self.analyzer.analyze(data)
        
        # Calculate signal score
        score, factors = self._calculate_signal_score(analysis)
        
        # Determine direction
        direction = self._determine_direction(analysis, score)
        
        # Calculate levels
        entry_zone, stop_loss, targets = self._calculate_levels(
            analysis, direction
        )
        
        # Calculate risk:reward
        if direction != 'WAIT' and stop_loss and targets:
            entry_mid = (entry_zone[0] + entry_zone[1]) / 2
            risk = abs(entry_mid - stop_loss)
            reward = abs(targets[0] - entry_mid)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
        
        # Adjust confidence based on R:R
        final_confidence = self._adjust_confidence(score / 100, risk_reward)
        
        # Generate analysis text
        analysis_text = self._generate_analysis_text(analysis, factors, direction)
        
        return TradingSignal(
            id=str(uuid.uuid4()),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            confidence=final_confidence,
            entry_price=data['close'].iloc[-1],
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit=targets,
            risk_reward=risk_reward,
            factors=factors,
            analysis_text=analysis_text,
            mtf_bias=analysis['bias']['bias'].value,
            key_levels=self._extract_key_levels(analysis),
            created_at=datetime.utcnow(),
            valid_until=self._calculate_validity(timeframe)
        )
    
    def _calculate_signal_score(self, analysis: Dict) -> tuple:
        """Calculate signal score based on ICT criteria"""
        
        score = 0
        factors = []
        
        # Factor 1: Market Structure (25 points)
        ms = analysis['market_structure']['structure']
        if ms != MarketStructure.CONSOLIDATION:
            score += 25
            factors.append(f"Clear {ms.value} market structure")
        
        # Factor 2: Order Block available (25 points)
        obs = analysis['order_blocks']
        bias = analysis['bias']['bias']
        
        if bias == Bias.BULLISH and any(ob.type == 'bullish' for ob in obs):
            score += 25
            factors.append("Valid bullish order block present")
        elif bias == Bias.BEARISH and any(ob.type == 'bearish' for ob in obs):
            score += 25
            factors.append("Valid bearish order block present")
        
        # Factor 3: Fair Value Gap (20 points)
        fvgs = analysis['fair_value_gaps']
        
        if bias == Bias.BULLISH and any(fvg.type == 'bullish' for fvg in fvgs):
            score += 20
            factors.append("Bullish FVG for entry")
        elif bias == Bias.BEARISH and any(fvg.type == 'bearish' for fvg in fvgs):
            score += 20
            factors.append("Bearish FVG for entry")
        
        # Factor 4: Premium/Discount zone (20 points)
        pd_zone = analysis['premium_discount']
        
        if bias == Bias.BULLISH and pd_zone['zone'] == 'discount':
            score += 20
            factors.append(f"Price in discount zone ({pd_zone['percentage']:.0f}%)")
        elif bias == Bias.BEARISH and pd_zone['zone'] == 'premium':
            score += 20
            factors.append(f"Price in premium zone ({pd_zone['percentage']:.0f}%)")
        
        # Factor 5: Liquidity target (10 points)
        liquidity = analysis['liquidity_levels']
        
        if bias == Bias.BULLISH and liquidity['buy_side_liquidity']:
            score += 10
            factors.append("Buy-side liquidity target identified")
        elif bias == Bias.BEARISH and liquidity['sell_side_liquidity']:
            score += 10
            factors.append("Sell-side liquidity target identified")
        
        return score, factors
    
    def _determine_direction(self, analysis: Dict, score: int) -> str:
        """Determine trade direction based on analysis"""
        
        if score < 50:
            return 'WAIT'
        
        bias = analysis['bias']['bias']
        
        if bias == Bias.BULLISH:
            return 'BUY'
        elif bias == Bias.BEARISH:
            return 'SELL'
        else:
            return 'WAIT'
    
    def _calculate_levels(self, analysis: Dict, 
                          direction: str) -> tuple:
        """Calculate entry, stop loss, and take profit levels"""
        
        if direction == 'WAIT':
            return (0, 0), 0, []
        
        current_price = analysis['current_price']
        obs = analysis['order_blocks']
        fvgs = analysis['fair_value_gaps']
        liquidity = analysis['liquidity_levels']
        
        if direction == 'BUY':
            # Entry at bullish OB or FVG
            bullish_obs = [ob for ob in obs if ob.type == 'bullish']
            bullish_fvgs = [fvg for fvg in fvgs if fvg.type == 'bullish']
            
            if bullish_obs:
                entry_zone = (bullish_obs[-1].low, bullish_obs[-1].high)
                stop_loss = bullish_obs[-1].low * 0.998  # Slightly below OB
            elif bullish_fvgs:
                entry_zone = (bullish_fvgs[-1].low, bullish_fvgs[-1].high)
                stop_loss = bullish_fvgs[-1].low * 0.998
            else:
                entry_zone = (current_price * 0.998, current_price)
                stop_loss = current_price * 0.99
            
            # Take profit at buy-side liquidity
            if liquidity['buy_side_liquidity']:
                targets = [level['level'] for level in liquidity['buy_side_liquidity'][:3]]
            else:
                targets = [current_price * 1.02, current_price * 1.04, current_price * 1.06]
        
        else:  # SELL
            bearish_obs = [ob for ob in obs if ob.type == 'bearish']
            bearish_fvgs = [fvg for fvg in fvgs if fvg.type == 'bearish']
            
            if bearish_obs:
                entry_zone = (bearish_obs[-1].low, bearish_obs[-1].high)
                stop_loss = bearish_obs[-1].high * 1.002
            elif bearish_fvgs:
                entry_zone = (bearish_fvgs[-1].low, bearish_fvgs[-1].high)
                stop_loss = bearish_fvgs[-1].high * 1.002
            else:
                entry_zone = (current_price, current_price * 1.002)
                stop_loss = current_price * 1.01
            
            if liquidity['sell_side_liquidity']:
                targets = [level['level'] for level in liquidity['sell_side_liquidity'][:3]]
            else:
                targets = [current_price * 0.98, current_price * 0.96, current_price * 0.94]
        
        return entry_zone, stop_loss, targets
    
    def _adjust_confidence(self, base_confidence: float, 
                           risk_reward: float) -> float:
        """Adjust confidence based on risk:reward ratio"""
        
        if risk_reward >= 3:
            return min(base_confidence * 1.1, 1.0)
        elif risk_reward >= 2:
            return base_confidence
        elif risk_reward >= 1.5:
            return base_confidence * 0.9
        else:
            return base_confidence * 0.7
    
    def _generate_analysis_text(self, analysis: Dict, 
                                factors: List[str], direction: str) -> str:
        """Generate human-readable analysis"""
        
        bias = analysis['bias']
        structure = analysis['market_structure']
        pd_zone = analysis['premium_discount']
        
        text = f"""
## ICT Analysis Summary

**Market Structure:** {structure['structure'].value.title()}
**Current Bias:** {bias['bias'].value.title()} ({bias['confidence']:.0%} confidence)
**Price Zone:** {pd_zone['zone'].title()} ({pd_zone['percentage']:.0f}% of range)

### Signal Factors:
{"".join([f"- {f}" for f in factors])}

### Key Observations:
- {bias['reasoning']}
- Range: {pd_zone['range_low']:.5f} - {pd_zone['range_high']:.5f}
- Equilibrium: {pd_zone['equilibrium']:.5f}

### Recommendation: **{direction}**
"""
        return text
    
    def _extract_key_levels(self, analysis: Dict) -> List[Dict]:
        """Extract key levels for charting"""
        
        levels = []
        
        # Add order block levels
        for ob in analysis['order_blocks'][:5]:
            levels.append({
                'type': f'{ob.type}_ob',
                'high': ob.high,
                'low': ob.low
            })
        
        # Add FVG levels
        for fvg in analysis['fair_value_gaps'][:5]:
            levels.append({
                'type': f'{fvg.type}_fvg',
                'high': fvg.high,
                'low': fvg.low
            })
        
        # Add liquidity levels
        for level in analysis['liquidity_levels']['buy_side_liquidity'][:3]:
            levels.append({
                'type': 'buy_liquidity',
                'price': level['level']
            })
        
        for level in analysis['liquidity_levels']['sell_side_liquidity'][:3]:
            levels.append({
                'type': 'sell_liquidity',
                'price': level['level']
            })
        
        return levels
    
    def _calculate_validity(self, timeframe: str) -> datetime:
        """Calculate signal validity duration"""
        
        validity_map = {
            'M1': timedelta(days=30),
            'W1': timedelta(days=7),
            'D1': timedelta(days=2),
            'H4': timedelta(hours=16),
            'H1': timedelta(hours=4)
        }
        
        return datetime.utcnow() + validity_map.get(timeframe, timedelta(hours=4))
```

---

## Deployment Configuration

### Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ict_trading
      - REDIS_URL=redis://redis:6379
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID}
      - TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN}
    depends_on:
      - db
      - redis
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend

  db:
    image: timescale/timescaledb:latest-pg14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=ict_trading
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  celery:
    build: ./backend
    command: celery -A app.celery worker -l info
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ict_trading
      - REDIS_URL=redis://redis:6379
    depends_on:
      - backend
      - redis

  celery-beat:
    build: ./backend
    command: celery -A app.celery beat -l info
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ict_trading
      - REDIS_URL=redis://redis:6379
    depends_on:
      - celery

volumes:
  postgres_data:
  redis_data:
```

---

## Environment Variables

```bash
# .env.example

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ict_trading

# Redis
REDIS_URL=redis://localhost:6379

# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

# Market Data
POLYGON_API_KEY=your_polygon_key
ALPHA_VANTAGE_KEY=your_alphavantage_key

# Notifications
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# App Settings
SECRET_KEY=your_secret_key
DEBUG=true
CORS_ORIGINS=http://localhost:3000
```

---

## Next Steps Checklist

```
□ Share YouTube playlist URL for video analysis
□ Confirm technology stack choices
□ Set up development environment
□ Create GitHub repository
□ Set up cloud infrastructure
□ Begin Phase 1 implementation
```
