"""
AISM — Data Models
==================
Shared dataclasses and enums used across all six stages.

BluePrint v2.1 references:
  - Section 2:  Observable dimensions, context-conditional traits
  - Section 3:  Evidence item schema (Stage 2), WeightedEvidence (Stage 3)
  - Section 4:  Storage structures (profile, evidence log, overlay)

Note on context-conditional traits:
  A trait is keyed by (trait_name, context) rather than trait_name alone.
  InteractionProfile stores this as a nested dict: traits[name][context] -> TraitState.
  Default context is "general"; context-specific values fall back to "general" if absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# ============================================================
# Enums
# ============================================================


class TurnRole(str, Enum):
  USER = "user"
  ASSISTANT = "assistant"
  SYSTEM = "system"


class EvidenceSourceType(str, Enum):
  """
    How we came to believe this evidence item. Drives the explicitness
    feature in Stage 3's log-linear scorer.
    """
  EXPLICIT_PREFERENCE = "explicit_preference"  # "I prefer short replies"
  EXPLICIT_DISLIKE = "explicit_dislike"  # "don't use bullet points"
  CORRECTION = "correction"  # "that was too long"
  POSITIVE_FEEDBACK = "positive_feedback"  # "this tone is better"
  IMPLICIT_PATTERN = "implicit_pattern"  # inferred from repetition


class StabilityClass(str, Enum):
  """
    Final classification produced by Stage 3's scorer/gater.
    Stage 2 tags evidence as CANDIDATE_STABLE; Stage 3 decides the final class.
    """
  CANDIDATE_STABLE = "candidate_stable"  # set by Stage 2 (extractor)
  STABLE = "stable"  # promoted by Stage 3 → updates profile
  SESSION = "session"  # stays in session overlay only
  IGNORE = "ignore"  # failed a gate; logged, not applied


# ============================================================
# Core conversation types
# ============================================================


@dataclass
class ConversationTurn:
  """A single turn in the conversation. The input unit for the pipeline."""
  turn_id: str
  role: TurnRole
  text: str
  timestamp: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc))
  metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Stage 1 output
# ============================================================


@dataclass
class EligibilityDecision:
  """
    Produced by Stage 1 (eligibility filter).
    `contamination_score` is the max of the three heuristic scores (keyword,
    structural, length). Lower = cleaner conversational turn.
    """
  turn_id: str
  is_eligible: bool
  reason: str
  contamination_score: float  # 0.0 (clean) .. 1.0 (clearly contaminated)
  eligibility_confidence: float  # used as a gate feature in Stage 3


# ============================================================
# Stage 2 output
# ============================================================


@dataclass
class StyleEvidence:
  """
    One observation that suggests something about the user's preferred style.
    Emitted by Stage 2; consumed (scored) by Stage 3.

    Note: `stability_class` here is always CANDIDATE_STABLE when produced by
    the extractor. Stage 3 decides the final class (STABLE / SESSION / IGNORE).
    """
  evidence_id: str
  turn_id: str
  trait: str
  context: str  # "general" | "technical" | "emotional" | ...
  value: Any  # scalar float or categorical string
  confidence: float  # extractor's confidence in this cue
  source_type: EvidenceSourceType
  stability_class: StabilityClass  # CANDIDATE_STABLE at this stage
  timestamp: datetime
  evidence_text: str  # the user text that triggered this
  metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Stage 3 output
# ============================================================


@dataclass
class WeightedEvidence:
  """
    Stage 3 output: evidence plus its computed weight and final stability
    class. If the gate failed, `gate_pass` is False and the evidence is
    classified as IGNORE.
    """
  evidence: StyleEvidence
  weight: float  # log-linear weight in [0, 1]
  gate_pass: bool  # did the evidence clear the gates?
  final_stability: StabilityClass  # STABLE | SESSION | IGNORE
  is_hard_override: bool = False
  features: Dict[str, float] = field(default_factory=dict)  # for auditability


# ============================================================
# Stage 4 / profile state
# ============================================================


@dataclass
class TraitState:
  """
    The stored state for a single (trait, context) pair.
    `recent_values` is kept so we can compute real volatility (stddev) rather
    than a decaying placeholder. `session_drift` tracks cumulative |Δ| within
    the current session to enforce the per-session drift cap (Stage 4).
    """
  value: Any
  confidence: float
  evidence_count: int
  volatility: float
  last_updated: datetime
  source_summary: List[str] = field(default_factory=list)
  recent_values: List[Any] = field(default_factory=list)
  session_drift: float = 0.0


@dataclass
class InteractionProfile:
  """
    Long-term stable profile. Traits are context-conditional:
    traits[trait_name][context] -> TraitState.
    """
  user_id: str
  traits: Dict[str, Dict[str, TraitState]] = field(default_factory=dict)
  boundaries: Dict[str, Any] = field(default_factory=dict)
  last_updated: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc))

  # ---- Helpers for context-conditional access ----

  def get_trait(self,
                trait_name: str,
                context: str = "general") -> Optional[TraitState]:
    """
        Return the TraitState for (trait_name, context). If no context-specific
        entry exists, fall back to (trait_name, 'general'). Returns None if
        the trait is not stored at all.
        """
    by_context = self.traits.get(trait_name)
    if by_context is None:
      return None
    if context in by_context:
      return by_context[context]
    return by_context.get("general")

  def set_trait(
      self, trait_name: str, context: str, state: TraitState) -> None:
    if trait_name not in self.traits:
      self.traits[trait_name] = {}
    self.traits[trait_name][context] = state

  def reset_session_drift(self) -> None:
    """Called at session end to clear per-session drift counters."""
    for by_context in self.traits.values():
      for trait_state in by_context.values():
        trait_state.session_drift = 0.0


@dataclass
class SessionOverlay:
  """
    Temporary style state for the current session. Expires on session end
    or after inactivity TTL (enforced by the pipeline, not stored here).
    """
  session_id: str
  state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
  last_updated: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc))


# ============================================================
# Stage 5 output
# ============================================================


@dataclass
class ResponsePolicy:
  """
    The compact style policy handed to the GenerationAdapter.
    Fields map to BluePrint Section 3 Stage 5's example policy.
    Any field may be None if no evidence supports a setting — this is what
    Precision@K selection produces.

    `mode` and `mode_confidence` (added in the turn-intent pass) carry the
    per-turn intent label from `aism.turn_intent`. The adapter uses them to
    decide whether to render advice-emitting blocks of the system prompt
    or switch to listen-first behaviour. Default 'ambiguous' is the
    listen-first default — explicit SEEKING_HELP is the only label that
    fully unlocks suggestion-style replies.
    """
  tone: Optional[str] = None
  verbosity: Optional[str] = None
  formatting: Optional[str] = None
  emotional_style: Optional[str] = None
  proactiveness: Optional[str] = None
  reasoning_presentation: Optional[str] = None
  mode: str = "ambiguous"
  mode_confidence: float = 0.0
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
  """A single remarkable memory stored for long-term personalization."""
  memory_id: str
  user_id: str
  utterance: str
  store: str
  memory_type: str
  importance: int
  frequency: int
  created_at: datetime
  last_seen: datetime
  status: str
  notes: str = ""
