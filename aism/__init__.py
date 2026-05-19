"""
AISM — Adaptive Interaction Style Modelling
===========================================
Pass 1 implementation.

Public API:
    AdaptiveInteractionStylePipeline  — the orchestrator
    SessionContext                    — integration contract with LTM
    ConversationTurn / TurnRole       — input types
    ResponsePolicy                    — output handed to the GenerationAdapter
    LocalJSONStore                    — persistence
    MockGenerationAdapter             — default adapter (no LLM)

BluePrint v2.1 is the authoritative design spec. Each stage module references
the relevant BluePrint section in its docstring.
"""

from .data_models import (
    ConversationTurn,
    EligibilityDecision,
    EvidenceSourceType,
    InteractionProfile,
    ResponsePolicy,
    SessionOverlay,
    StabilityClass,
    StyleEvidence,
    TraitState,
    TurnRole,
    WeightedEvidence,
)
from .embedder import (
    CharacterNgramEmbedder,
    Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
)
from .generation_adapter import GenerationAdapter, MockGenerationAdapter
from .pipeline import AdaptiveInteractionStylePipeline, TurnResult
from .session_context import SessionContext
from .stage2b_layer_b_extractor import LayerBExtractor
from .stage2c_pattern_aggregator import PatternAggregator
from .storage import LocalJSONStore

__all__ = [
    # Pass 1
    "AdaptiveInteractionStylePipeline",
    "ConversationTurn",
    "EligibilityDecision",
    "EvidenceSourceType",
    "GenerationAdapter",
    "InteractionProfile",
    "LocalJSONStore",
    "MockGenerationAdapter",
    "ResponsePolicy",
    "SessionContext",
    "SessionOverlay",
    "StabilityClass",
    "StyleEvidence",
    "TraitState",
    "TurnResult",
    "TurnRole",
    "WeightedEvidence",
    # Pass 2 — embedder + extractors
    "CharacterNgramEmbedder",
    "Embedder",
    "LayerBExtractor",
    "PatternAggregator",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
]
