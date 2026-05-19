"""
AISM — STAGE 6: Feedback Processor (Closed Loop)
================================================
Detects explicit (and coarse implicit) feedback signals in a user turn that
follows an assistant reply, converts them into fresh StyleEvidence, and hands
them back to Stage 3 → Stage 4 for immediate in-session profile adjustment.

BluePrint v2.1 reference: Section 3 — Stage 6.

This is the component that turns v1's "dangling feedback hook" into a real
closed loop.

Steps:
    Step 6.1 : Scan the user turn for feedback lexicons (correction / positive)
    Step 6.2 : Infer which trait the feedback targets
    Step 6.3 : Infer the target direction (increase / decrease)
    Step 6.4 : Convert to StyleEvidence with appropriate source_type
    Step 6.5 : Return evidence list for re-injection into Stage 3

Note: Pass 1 implements rule-based feedback detection. Sycophancy detection
and the four over-mimicry monitors are deferred to Pass 2.

Constraint: No LLM calls.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .data_models import (
  ConversationTurn,
  EvidenceSourceType,
  StabilityClass,
  StyleEvidence,
)
from .session_context import SessionContext
from .stage2_extraction import HUMOUR_LOW_PATTERNS, HUMOUR_HIGH_PATTERNS

# ============================================================
# Feedback pattern map (Step 6.1-6.3 data)
# ============================================================
#
# Each entry specifies:
#   patterns      : regex list that matches the feedback phrase
#   trait         : the trait being corrected
#   value         : the target value implied by the feedback
#   source_type   : CORRECTION or POSITIVE_FEEDBACK
# ============================================================

FEEDBACK_RULES: List[Dict] = [

  # ---- Verbosity corrections ----
  {
    "patterns": [
      r"\btoo\s+long\b",
      r"\btoo\s+wordy\b",
      r"\btoo\s+much\s+text\b",
      r"\bway\s+too\s+long\b",
      r"\bshorter\s+please\b",
      r"\btalk(?:ing)?\s+too\s+much\b",
      r"\btoo\s+much\s+talk(?:ing)?\b",
    ],
    "trait":
    "verbosity",
    "value":
    "short",
    "source_type":
    EvidenceSourceType.CORRECTION,
  },
  {
    "patterns": [
      r"\btoo\s+short\b",
      r"\bmore\s+detail\b",
      r"\belaborate\b",
      r"\btoo\s+brief\b",
    ],
    "trait":
    "verbosity",
    "value":
    "long",
    "source_type":
    EvidenceSourceType.CORRECTION,
  },

  # ---- Directness corrections ----
  {
    "patterns": [
      r"\bjust\s+answer\b",
      r"\bstop\s+hedging\b",
      r"\bget\s+to\s+the\s+point\b",
      r"\bdon'?t\s+beat\s+around\b",
    ],
    "trait":
    "directness",
    "value":
    0.9,
    "source_type":
    EvidenceSourceType.CORRECTION,
  },
  {
    "patterns": [
      r"\btoo\s+blunt\b",
      r"\bbe\s+gentler\b",
      r"\bsoften\s+that\b",
    ],
    "trait": "directness",
    "value": 0.35,
    "source_type": EvidenceSourceType.CORRECTION,
  },

  # ---- Tone / warmth corrections ----
  {
    "patterns": [
      r"\btoo\s+enthusiastic\b",
      r"\btoo\s+cheerful\b",
      r"\btoo\s+perky\b",
      r"\bdrop\s+the\s+(cheer|enthusiasm)\b",
      r"\bless\s+enthusiastic\b",
    ],
    "trait":
    "warmth",
    "value":
    0.3,
    "source_type":
    EvidenceSourceType.CORRECTION,
  },
  {
    "patterns": [
      r"\btoo\s+cold\b",
      r"\bbe\s+warmer\b",
      r"\bmore\s+empathy\b",
    ],
    "trait": "warmth",
    "value": 0.8,
    "source_type": EvidenceSourceType.CORRECTION,
  },

  # ---- Formatting corrections ----
  {
    "patterns": [
      r"\btoo\s+many\s+bullets?\b",
      r"\bno\s+more\s+bullets?\b",
      r"\bwrite\s+it\s+as\s+prose\b",
    ],
    "trait":
    "formatting",
    "value":
    "paragraphs",
    "source_type":
    EvidenceSourceType.CORRECTION,
  },
  {
    "patterns": [
      r"\bas\s+bullets?\b",
      r"\bmake\s+it\s+a\s+list\b",
    ],
    "trait": "formatting",
    "value": "bullets",
    "source_type": EvidenceSourceType.CORRECTION,
  },

  # ---- Humour / joke corrections ----
  # Source-of-truth lives in stage2_extraction. Stage 6 just classifies
  # the same phrases as CORRECTION (a stronger signal type) instead of
  # EXPLICIT_DISLIKE, and uses 0.0 as the target value (the user is
  # unambiguously rejecting humour).
  {
    "patterns": HUMOUR_LOW_PATTERNS,
    "trait":
    "humour",
    "value":
    0.0,
    "source_type":
    EvidenceSourceType.CORRECTION,
  },
  # Stage 6 only adopts the explicit-request subset of HUMOUR_HIGH_PATTERNS
  # — phrases like "tell me a joke" / "i want to hear a joke" are clear
  # post-reply requests for more humour. We exclude pure intensity phrases
  # ("be funnier") since those are caught by Stage 2 as preference, not
  # feedback on the just-given reply.
  {
    "patterns": [
      r"\btell\s+me\s+(?:a\s+)?(?:few\s+)?(?:\w+\s+){0,3}jokes?\b",
      r"\bi\s+(?:want|would\s+like|need)\s+(?:to\s+hear\s+)?(?:a\s+)?(?:few\s+)?(?:\w+\s+){0,3}jokes?\b",
      r"\b(?:\w+\s+){0,3}jokes?\s+please\b",
      r"\bnot\s+funny\s+enough\b",
      r"\b(?:more|much)\s+(?:funny|funnier)\b",
    ],
    "trait":
    "humour",
    "value":
    0.8,
    "source_type":
    EvidenceSourceType.EXPLICIT_PREFERENCE,
  },

  # ---- Positive feedback ----
  {
    "patterns": [
      r"\bperfect\s+length\b",
      r"\bgood\s+length\b",
      r"\bright\s+amount\b",
    ],
    "trait":
    "verbosity",
    "value":
    "medium",
    "source_type":
    EvidenceSourceType.POSITIVE_FEEDBACK,
  },
  {
    "patterns": [
      r"\bthis\s+tone\s+is\s+(good|better|right)\b",
      r"\bthat\s+was\s+(perfect|great|ideal)\b",
      r"\bmuch\s+better\b",
    ],
    # Positive tone feedback — anchors current warmth; we treat this as a
    # mid-confidence signal to strengthen whatever warmth is currently
    # being expressed (value left at neutral 0.5 unless other evidence
    # pins it down — it's primarily a confidence booster).
    "trait":
    "warmth",
    "value":
    0.5,
    "source_type":
    EvidenceSourceType.POSITIVE_FEEDBACK,
  },
]


class FeedbackProcessor:
  """
    Rule-based feedback detector. Emits StyleEvidence items that should be
    re-injected into Stage 3 (scoring) by the pipeline orchestrator.
    """

  def __init__(self) -> None:
    # Pre-compile patterns.
    self._rules: List[Tuple[List[re.Pattern], Dict]] = []
    for rule in FEEDBACK_RULES:
      compiled = [re.compile(p, re.IGNORECASE) for p in rule["patterns"]]
      self._rules.append((compiled, rule))

  # ============================================================
  # Public entry point
  # ============================================================

  def detect_feedback(
    self,
    turn: ConversationTurn,
    session_context: SessionContext,
  ) -> List[StyleEvidence]:
    """
        Scan `turn` for feedback signals and return StyleEvidence items ready
        for Stage 3 re-injection.
        """
    text = turn.text
    evidences: List[StyleEvidence] = []
    seen_keys: set = set()  # dedupe (trait, direction) within a single turn

    # --- Step 6.1: Lexicon scan ---
    for patterns, rule in self._rules:
      hit = self._first_match(patterns, text)
      if hit is None:
        continue

      # --- Step 6.2-6.3: Trait + target implied by rule ---
      trait = rule["trait"]
      value = rule["value"]
      source_type = rule["source_type"]

      key = (
        trait,
        value if not isinstance(value,
                                (int, float)) else round(float(value), 2))
      if key in seen_keys:
        continue
      seen_keys.add(key)

      # --- Step 6.4: Convert to StyleEvidence ---
      confidence = 0.94 if source_type == EvidenceSourceType.CORRECTION else 0.85
      evidences.append(
        StyleEvidence(
          evidence_id=f"fb-{turn.turn_id}-{trait}-{hit[:20]}",
          turn_id=turn.turn_id,
          trait=trait,
          context=session_context.current_topic,
          value=value,
          confidence=confidence,
          source_type=source_type,
          stability_class=StabilityClass.CANDIDATE_STABLE,
          timestamp=turn.timestamp,
          evidence_text=turn.text[:200],
          metadata={
            "source": "feedback_processor",
            "matched_pattern": hit
          },
        ))

    # If a turn contains both positive and negative humour cues, prefer the
    # correction. This prevents "no more jokes please" from creating a new
    # positive humour preference through the word "please".
    humour_correction_present = any(
      e.trait == "humour" and isinstance(e.value, (int, float)) and e.value <= 0.3
      for e in evidences
    )
    if humour_correction_present:
      evidences = [
        e for e in evidences
        if not (
          e.trait == "humour" and isinstance(e.value, (int, float)) and e.value > 0.3
        )
      ]

    # --- Step 6.5: Return for re-injection ---
    return evidences

  @staticmethod
  def _first_match(patterns: List[re.Pattern], text: str) -> str | None:
    for p in patterns:
      m = p.search(text)
      if m:
        return m.group(0)
    return None
