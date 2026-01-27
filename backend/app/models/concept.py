"""ICT Concept models for knowledge base"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from uuid import uuid4


class ConceptCategory(Enum):
    """Categories of ICT concepts"""
    MARKET_STRUCTURE = "market_structure"
    KEY_LEVELS = "key_levels"
    ENTRY_MODELS = "entry_models"
    TIME_BASED = "time_based"
    INSTITUTIONAL = "institutional"
    RISK_MANAGEMENT = "risk_management"
    LIQUIDITY = "liquidity"
    PRICE_ACTION = "price_action"


@dataclass
class ICTConcept:
    """
    Represents an ICT trading concept

    This forms the core taxonomy that the ML model will learn to identify
    and apply to market analysis.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    short_name: str = ""  # Abbreviated form (e.g., "OB" for Order Block)
    category: ConceptCategory = ConceptCategory.MARKET_STRUCTURE

    # Hierarchical structure
    parent_concept_id: Optional[str] = None
    child_concept_ids: List[str] = field(default_factory=list)

    # Description and definition
    description: str = ""
    detailed_explanation: str = ""

    # Detection
    detection_keywords: List[str] = field(default_factory=list)
    detection_patterns: List[str] = field(default_factory=list)  # Regex patterns

    # Trading rules associated with this concept
    identification_rules: List[str] = field(default_factory=list)
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    confirmation_rules: List[str] = field(default_factory=list)

    # Related concepts for confluence
    related_concept_ids: List[str] = field(default_factory=list)

    # Learning metadata
    difficulty_level: int = 1  # 1-5, 1 being basic
    prerequisite_concept_ids: List[str] = field(default_factory=list)

    # Source tracking
    source_video_ids: List[str] = field(default_factory=list)
    mention_count: int = 0

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "short_name": self.short_name,
            "category": self.category.value,
            "description": self.description,
            "detection_keywords": self.detection_keywords,
            "difficulty_level": self.difficulty_level,
            "mention_count": self.mention_count,
        }


@dataclass
class ConceptMention:
    """
    Records where an ICT concept was mentioned/explained in a video

    This links video content to concepts for training and retrieval.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    video_id: str = ""
    concept_id: str = ""
    transcript_segment_id: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Context
    context_text: str = ""  # The actual text where concept was mentioned
    before_context: str = ""  # Text before for context
    after_context: str = ""  # Text after for context

    # Quality assessment
    confidence_score: float = 0.0  # How confident are we this is a valid mention
    explanation_quality: str = "brief"  # brief, detailed, with_example

    # What type of mention is this
    mention_type: str = "explanation"  # explanation, example, rule, reference

    # If this mention contains a trading rule
    extracted_rule: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "concept_id": self.concept_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "context_text": self.context_text,
            "confidence_score": self.confidence_score,
            "explanation_quality": self.explanation_quality,
            "mention_type": self.mention_type,
        }


@dataclass
class ConceptRule:
    """
    A codified trading rule extracted from ICT content

    These rules are what the signal generator will use to make decisions.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    concept_id: str = ""

    # Rule definition
    rule_type: str = ""  # identification, entry, exit, confirmation, filter
    rule_name: str = ""
    rule_description: str = ""

    # Codified rule (can be converted to code)
    rule_definition: Dict[str, Any] = field(default_factory=dict)

    # Natural language version
    rule_text: str = ""

    # Conditions for this rule
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    # Priority and confidence
    priority: int = 1  # Higher = more important
    confidence_score: float = 0.0

    # Source tracking
    source_video_ids: List[str] = field(default_factory=list)
    source_mentions: List[str] = field(default_factory=list)

    # Versioning (rules may be refined over time)
    version: int = 1
    previous_version_id: Optional[str] = None

    # Validation
    backtested: bool = False
    backtest_win_rate: Optional[float] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "concept_id": self.concept_id,
            "rule_type": self.rule_type,
            "rule_name": self.rule_name,
            "rule_description": self.rule_description,
            "rule_text": self.rule_text,
            "priority": self.priority,
            "confidence_score": self.confidence_score,
            "version": self.version,
        }


# Pre-defined ICT Concept Taxonomy
ICT_CONCEPT_TAXONOMY = {
    "market_structure": {
        "name": "Market Structure",
        "concepts": [
            {
                "name": "Break of Structure",
                "short_name": "BOS",
                "keywords": ["break of structure", "bos", "structure break", "broke structure"],
                "description": "Price breaking above/below a previous swing point confirming trend continuation",
            },
            {
                "name": "Change of Character",
                "short_name": "CHoCH",
                "keywords": ["change of character", "choch", "character change", "trend change"],
                "description": "First sign of potential trend reversal when price breaks structure in opposite direction",
            },
            {
                "name": "Market Structure Shift",
                "short_name": "MSS",
                "keywords": ["market structure shift", "mss", "structure shift"],
                "description": "Confirmed change in market direction after CHoCH",
            },
            {
                "name": "Higher High",
                "short_name": "HH",
                "keywords": ["higher high", "hh", "new high"],
                "description": "A swing high that is higher than the previous swing high",
            },
            {
                "name": "Higher Low",
                "short_name": "HL",
                "keywords": ["higher low", "hl"],
                "description": "A swing low that is higher than the previous swing low",
            },
            {
                "name": "Lower High",
                "short_name": "LH",
                "keywords": ["lower high", "lh"],
                "description": "A swing high that is lower than the previous swing high",
            },
            {
                "name": "Lower Low",
                "short_name": "LL",
                "keywords": ["lower low", "ll", "new low"],
                "description": "A swing low that is lower than the previous swing low",
            },
        ]
    },
    "key_levels": {
        "name": "Key Levels & Zones",
        "concepts": [
            {
                "name": "Order Block",
                "short_name": "OB",
                "keywords": ["order block", "ob", "orderblock", "bullish order block", "bearish order block"],
                "description": "The last candle before a significant price move, representing institutional orders",
            },
            {
                "name": "Bullish Order Block",
                "short_name": "Bullish OB",
                "keywords": ["bullish order block", "bullish ob", "demand zone"],
                "description": "Last bearish candle before a bullish impulse move",
            },
            {
                "name": "Bearish Order Block",
                "short_name": "Bearish OB",
                "keywords": ["bearish order block", "bearish ob", "supply zone"],
                "description": "Last bullish candle before a bearish impulse move",
            },
            {
                "name": "Fair Value Gap",
                "short_name": "FVG",
                "keywords": ["fair value gap", "fvg", "imbalance", "inefficiency"],
                "description": "A gap in price where there was no trading, representing imbalance",
            },
            {
                "name": "Breaker Block",
                "short_name": "BB",
                "keywords": ["breaker block", "breaker", "failed order block"],
                "description": "An order block that has been mitigated and now acts as opposite zone",
            },
            {
                "name": "Mitigation Block",
                "short_name": "MB",
                "keywords": ["mitigation block", "mitigation"],
                "description": "A zone where price returns to fill orders before continuing",
            },
        ]
    },
    "liquidity": {
        "name": "Liquidity Concepts",
        "concepts": [
            {
                "name": "Buy Side Liquidity",
                "short_name": "BSL",
                "keywords": ["buy side liquidity", "bsl", "buy stops", "liquidity above"],
                "description": "Stop losses of short sellers resting above swing highs",
            },
            {
                "name": "Sell Side Liquidity",
                "short_name": "SSL",
                "keywords": ["sell side liquidity", "ssl", "sell stops", "liquidity below"],
                "description": "Stop losses of long buyers resting below swing lows",
            },
            {
                "name": "Liquidity Pool",
                "short_name": "LP",
                "keywords": ["liquidity pool", "liquidity", "stops", "stop hunt"],
                "description": "Area where many stop losses are clustered",
            },
            {
                "name": "Liquidity Sweep",
                "short_name": "Sweep",
                "keywords": ["liquidity sweep", "sweep", "stop hunt", "liquidity grab", "raid"],
                "description": "When price moves to take out stop losses before reversing",
            },
            {
                "name": "Equal Highs",
                "short_name": "EQH",
                "keywords": ["equal highs", "eqh", "double top", "triple top"],
                "description": "Multiple swing highs at similar levels creating liquidity",
            },
            {
                "name": "Equal Lows",
                "short_name": "EQL",
                "keywords": ["equal lows", "eql", "double bottom", "triple bottom"],
                "description": "Multiple swing lows at similar levels creating liquidity",
            },
        ]
    },
    "entry_models": {
        "name": "Entry Models",
        "concepts": [
            {
                "name": "Optimal Trade Entry",
                "short_name": "OTE",
                "keywords": ["optimal trade entry", "ote", "fib retracement", "62%", "79%"],
                "description": "Entry at 62-79% Fibonacci retracement of an impulse move",
            },
            {
                "name": "Silver Bullet",
                "short_name": "SB",
                "keywords": ["silver bullet", "10am", "11am", "silver bullet setup"],
                "description": "Specific time-based entry model during NY session",
            },
            {
                "name": "ICT 2022 Model",
                "short_name": "2022",
                "keywords": ["2022 model", "ict 2022", "new model"],
                "description": "ICT's refined entry model from 2022",
            },
            {
                "name": "Power of Three",
                "short_name": "PO3",
                "keywords": ["power of three", "po3", "amd", "accumulation manipulation distribution"],
                "description": "Three-phase market cycle: Accumulation, Manipulation, Distribution",
            },
            {
                "name": "Judas Swing",
                "short_name": "Judas",
                "keywords": ["judas swing", "judas", "fake move", "false break"],
                "description": "False move to trap traders before real move",
            },
            {
                "name": "Turtle Soup",
                "short_name": "TS",
                "keywords": ["turtle soup", "failed breakout"],
                "description": "Failed breakout pattern that reverses",
            },
        ]
    },
    "time_based": {
        "name": "Time-Based Concepts",
        "concepts": [
            {
                "name": "Kill Zone",
                "short_name": "KZ",
                "keywords": ["kill zone", "killzone", "kz", "trading session"],
                "description": "Specific time windows with highest probability setups",
            },
            {
                "name": "Asian Session",
                "short_name": "Asian",
                "keywords": ["asian session", "asian range", "asia", "tokyo"],
                "description": "Trading session from ~7pm-2am EST",
            },
            {
                "name": "London Session",
                "short_name": "London",
                "keywords": ["london session", "london open", "london", "lo"],
                "description": "Trading session from ~2am-5am EST",
            },
            {
                "name": "New York Session",
                "short_name": "NY",
                "keywords": ["new york session", "ny session", "new york", "ny open"],
                "description": "Trading session from ~7am-10am EST",
            },
            {
                "name": "True Day Open",
                "short_name": "TDO",
                "keywords": ["true day open", "tdo", "midnight open"],
                "description": "The actual start of the trading day at midnight EST",
            },
            {
                "name": "IPDA Data Range",
                "short_name": "IPDA",
                "keywords": ["ipda", "ipda data range", "20 day", "40 day", "60 day"],
                "description": "Interbank Price Delivery Algorithm reference ranges",
            },
        ]
    },
    "institutional": {
        "name": "Institutional Concepts",
        "concepts": [
            {
                "name": "Smart Money Concept",
                "short_name": "SMC",
                "keywords": ["smart money", "smc", "institutional", "banks"],
                "description": "Trading concepts based on institutional behavior",
            },
            {
                "name": "Premium Zone",
                "short_name": "Premium",
                "keywords": ["premium", "premium zone", "expensive", "above equilibrium"],
                "description": "Price area above 50% of range - look for shorts",
            },
            {
                "name": "Discount Zone",
                "short_name": "Discount",
                "keywords": ["discount", "discount zone", "cheap", "below equilibrium"],
                "description": "Price area below 50% of range - look for longs",
            },
            {
                "name": "Equilibrium",
                "short_name": "EQ",
                "keywords": ["equilibrium", "eq", "50%", "midpoint"],
                "description": "The 50% level of a price range",
            },
            {
                "name": "Displacement",
                "short_name": "Disp",
                "keywords": ["displacement", "impulse", "aggressive move"],
                "description": "Strong, fast price movement showing institutional involvement",
            },
        ]
    },
}
