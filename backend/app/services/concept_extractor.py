"""
ICT Concept Extraction Service

Uses AI (Claude/GPT) to extract ICT trading concepts from video transcripts.
This is the core learning component that builds the knowledge base.
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..models.video import Transcript, TranscriptSegment
from ..models.concept import (
    ICTConcept,
    ConceptMention,
    ConceptRule,
    ConceptCategory,
    ICT_CONCEPT_TAXONOMY
)
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of concept extraction from a transcript"""
    video_id: str
    concepts_found: List[ConceptMention]
    rules_extracted: List[ConceptRule]
    new_concepts: List[ICTConcept]  # Previously unknown concepts
    quality_score: float
    processing_time: float


class ConceptExtractor:
    """
    Extract ICT concepts from video transcripts using AI

    This service:
    1. Analyzes transcripts to find mentions of known ICT concepts
    2. Extracts trading rules and guidelines
    3. Identifies new concepts not in the taxonomy
    4. Links concepts to specific timestamps for reference
    """

    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        self.concept_taxonomy = self._load_concept_taxonomy()
        self.concept_patterns = self._build_concept_patterns()

        # AI clients (lazy loaded)
        self._anthropic_client = None
        self._openai_client = None

    def _load_concept_taxonomy(self) -> Dict[str, ICTConcept]:
        """Load the ICT concept taxonomy"""
        concepts = {}

        for category_key, category_data in ICT_CONCEPT_TAXONOMY.items():
            category = ConceptCategory(category_key) if category_key in [c.value for c in ConceptCategory] else ConceptCategory.MARKET_STRUCTURE

            for concept_data in category_data.get('concepts', []):
                concept = ICTConcept(
                    name=concept_data['name'],
                    short_name=concept_data.get('short_name', ''),
                    category=category,
                    description=concept_data.get('description', ''),
                    detection_keywords=concept_data.get('keywords', []),
                )
                concepts[concept.name.lower()] = concept

        logger.info(f"Loaded {len(concepts)} concepts from taxonomy")
        return concepts

    def _build_concept_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Build regex patterns for concept detection"""
        patterns = []

        for concept_name, concept in self.concept_taxonomy.items():
            for keyword in concept.detection_keywords:
                # Create pattern that matches the keyword with word boundaries
                pattern = re.compile(
                    r'\b' + re.escape(keyword) + r'\b',
                    re.IGNORECASE
                )
                patterns.append((pattern, concept.name))

        return patterns

    @property
    def anthropic_client(self):
        """Lazy load Anthropic client"""
        if self._anthropic_client is None and settings.ANTHROPIC_API_KEY:
            try:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._anthropic_client

    def extract_concepts_basic(self, transcript: Transcript) -> List[ConceptMention]:
        """
        Basic concept extraction using keyword matching

        This is faster but less accurate than AI extraction.
        Good for initial pass or when AI is not available.
        """
        mentions = []

        for segment in transcript.segments:
            text = segment.text.lower()

            for pattern, concept_name in self.concept_patterns:
                matches = pattern.finditer(text)

                for match in matches:
                    concept = self.concept_taxonomy.get(concept_name.lower())
                    if not concept:
                        continue

                    mention = ConceptMention(
                        video_id=transcript.video_id,
                        concept_id=concept.id,
                        transcript_segment_id=segment.id,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        context_text=segment.text,
                        confidence_score=0.7,  # Medium confidence for keyword match
                        explanation_quality="brief",
                        mention_type="reference",
                    )
                    mentions.append(mention)

        # Deduplicate by concept and segment
        seen = set()
        unique_mentions = []
        for mention in mentions:
            key = (mention.concept_id, mention.transcript_segment_id)
            if key not in seen:
                seen.add(key)
                unique_mentions.append(mention)

        logger.info(f"Found {len(unique_mentions)} concept mentions using keyword matching")
        return unique_mentions

    def extract_concepts_ai(self, transcript: Transcript) -> Tuple[List[ConceptMention], List[ConceptRule]]:
        """
        AI-powered concept extraction using Claude

        This provides:
        - Higher accuracy concept detection
        - Context-aware mentions
        - Rule extraction
        - Quality assessment
        """
        if not self.anthropic_client:
            logger.warning("Anthropic client not available, falling back to basic extraction")
            return self.extract_concepts_basic(transcript), []

        # Process in chunks to handle long transcripts
        chunk_size = 10000  # characters
        full_text = transcript.full_text
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

        all_mentions = []
        all_rules = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            prompt = self._build_extraction_prompt(chunk, i, len(chunks))

            try:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = self._parse_ai_response(response.content[0].text, transcript)
                all_mentions.extend(result['mentions'])
                all_rules.extend(result['rules'])

            except Exception as e:
                logger.error(f"Error in AI extraction: {e}")
                continue

        return all_mentions, all_rules

    def _build_extraction_prompt(self, text: str, chunk_index: int, total_chunks: int) -> str:
        """Build the prompt for AI concept extraction"""

        concept_list = "\n".join([
            f"- {name}: {concept.description}"
            for name, concept in self.concept_taxonomy.items()
        ])

        return f"""You are an expert in ICT (Inner Circle Trader) trading methodology.
Analyze the following transcript from an ICT YouTube video and extract:

1. **Concept Mentions**: Identify where specific ICT concepts are mentioned or explained.
2. **Trading Rules**: Extract any specific trading rules, guidelines, or criteria mentioned.
3. **New Concepts**: Note any ICT concepts not in the known list that should be added.

## Known ICT Concepts:
{concept_list}

## Transcript (Part {chunk_index + 1} of {total_chunks}):
{text}

## Response Format (JSON):
{{
    "concept_mentions": [
        {{
            "concept_name": "Order Block",
            "context": "exact quote from transcript",
            "explanation_quality": "brief|detailed|with_example",
            "mention_type": "explanation|example|rule|reference",
            "confidence": 0.0-1.0
        }}
    ],
    "trading_rules": [
        {{
            "concept_name": "Order Block",
            "rule_type": "identification|entry|exit|confirmation",
            "rule_text": "The specific rule in natural language",
            "conditions": ["condition 1", "condition 2"]
        }}
    ],
    "new_concepts": [
        {{
            "name": "Concept Name",
            "category": "market_structure|key_levels|entry_models|time_based|institutional|liquidity",
            "description": "Brief description",
            "keywords": ["keyword1", "keyword2"]
        }}
    ]
}}

Only include high-quality extractions. For concept mentions, focus on actual explanations
and teaching moments, not just passing references. For rules, only include specific,
actionable trading rules.

Respond with valid JSON only."""

    def _parse_ai_response(self, response_text: str, transcript: Transcript) -> Dict:
        """Parse the AI response and create model objects"""
        mentions = []
        rules = []

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.error("No JSON found in AI response")
                return {'mentions': [], 'rules': []}

            data = json.loads(json_match.group())

            # Process concept mentions
            for mention_data in data.get('concept_mentions', []):
                concept_name = mention_data.get('concept_name', '').lower()
                concept = self.concept_taxonomy.get(concept_name)

                if not concept:
                    # Try to find by partial match
                    for name, c in self.concept_taxonomy.items():
                        if concept_name in name or name in concept_name:
                            concept = c
                            break

                if concept:
                    # Find the segment containing this context
                    context = mention_data.get('context', '')
                    segment = self._find_segment_for_context(transcript, context)

                    mention = ConceptMention(
                        video_id=transcript.video_id,
                        concept_id=concept.id,
                        transcript_segment_id=segment.id if segment else '',
                        start_time=segment.start_time if segment else 0,
                        end_time=segment.end_time if segment else 0,
                        context_text=context,
                        confidence_score=mention_data.get('confidence', 0.8),
                        explanation_quality=mention_data.get('explanation_quality', 'brief'),
                        mention_type=mention_data.get('mention_type', 'reference'),
                    )
                    mentions.append(mention)

            # Process trading rules
            for rule_data in data.get('trading_rules', []):
                concept_name = rule_data.get('concept_name', '').lower()
                concept = self.concept_taxonomy.get(concept_name)

                if concept:
                    rule = ConceptRule(
                        concept_id=concept.id,
                        rule_type=rule_data.get('rule_type', 'identification'),
                        rule_name=f"{concept.name} - {rule_data.get('rule_type', 'Rule')}",
                        rule_text=rule_data.get('rule_text', ''),
                        conditions=[{'description': c} for c in rule_data.get('conditions', [])],
                        source_video_ids=[transcript.video_id],
                        confidence_score=0.8,
                    )
                    rules.append(rule)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")

        return {'mentions': mentions, 'rules': rules}

    def _find_segment_for_context(
        self,
        transcript: Transcript,
        context: str
    ) -> Optional[TranscriptSegment]:
        """Find the transcript segment that contains the given context"""
        context_lower = context.lower()

        for segment in transcript.segments:
            if context_lower in segment.text.lower():
                return segment

        # Fuzzy match - find segment with most word overlap
        context_words = set(context_lower.split())
        best_segment = None
        best_overlap = 0

        for segment in transcript.segments:
            segment_words = set(segment.text.lower().split())
            overlap = len(context_words & segment_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment = segment

        return best_segment

    def extract_from_transcript(self, transcript: Transcript) -> ExtractionResult:
        """
        Main entry point for concept extraction

        Combines basic and AI extraction for best results.
        """
        import time
        start_time = time.time()

        # Basic extraction first (fast, always available)
        basic_mentions = self.extract_concepts_basic(transcript)

        # AI extraction if available
        ai_mentions = []
        rules = []

        if self.use_ai and self.anthropic_client:
            ai_mentions, rules = self.extract_concepts_ai(transcript)

        # Merge results, preferring AI extractions
        all_mentions = self._merge_mentions(basic_mentions, ai_mentions)

        # Calculate quality score
        quality_score = self._calculate_quality_score(all_mentions, rules)

        processing_time = time.time() - start_time

        logger.info(f"Extraction complete: {len(all_mentions)} mentions, {len(rules)} rules")

        return ExtractionResult(
            video_id=transcript.video_id,
            concepts_found=all_mentions,
            rules_extracted=rules,
            new_concepts=[],  # TODO: Handle new concept suggestions
            quality_score=quality_score,
            processing_time=processing_time
        )

    def _merge_mentions(
        self,
        basic: List[ConceptMention],
        ai: List[ConceptMention]
    ) -> List[ConceptMention]:
        """Merge basic and AI extractions, deduplicating"""
        # Create map of AI mentions by concept+segment
        ai_map = {
            (m.concept_id, m.transcript_segment_id): m
            for m in ai
        }

        # Start with AI mentions (higher quality)
        merged = list(ai)

        # Add basic mentions not in AI results
        for mention in basic:
            key = (mention.concept_id, mention.transcript_segment_id)
            if key not in ai_map:
                merged.append(mention)

        return merged

    def _calculate_quality_score(
        self,
        mentions: List[ConceptMention],
        rules: List[ConceptRule]
    ) -> float:
        """Calculate quality score for the extraction"""
        if not mentions:
            return 0.0

        # Factors:
        # - Number of unique concepts found
        # - Average confidence of mentions
        # - Number of rules extracted
        # - Quality of explanations

        unique_concepts = len(set(m.concept_id for m in mentions))
        avg_confidence = sum(m.confidence_score for m in mentions) / len(mentions)

        detailed_explanations = sum(
            1 for m in mentions
            if m.explanation_quality in ['detailed', 'with_example']
        )

        # Weighted score
        score = (
            (unique_concepts / 20) * 0.3 +  # Concept diversity
            avg_confidence * 0.3 +  # Confidence
            (len(rules) / 10) * 0.2 +  # Rules found
            (detailed_explanations / len(mentions)) * 0.2  # Quality
        )

        return min(score, 1.0)

    def get_concept_summary(self, mentions: List[ConceptMention]) -> Dict:
        """Generate a summary of concepts found"""
        concept_counts = {}
        category_counts = {}

        for mention in mentions:
            # Get concept info
            for name, concept in self.concept_taxonomy.items():
                if concept.id == mention.concept_id:
                    # Count by concept
                    if name not in concept_counts:
                        concept_counts[name] = 0
                    concept_counts[name] += 1

                    # Count by category
                    cat = concept.category.value
                    if cat not in category_counts:
                        category_counts[cat] = 0
                    category_counts[cat] += 1
                    break

        return {
            'total_mentions': len(mentions),
            'unique_concepts': len(concept_counts),
            'by_concept': dict(sorted(concept_counts.items(), key=lambda x: -x[1])),
            'by_category': category_counts,
        }
