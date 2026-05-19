"""
AISM — STAGE 2: Style Evidence Extractor (Layer A)
==================================================
Extracts structured StyleEvidence items from an eligible user turn by matching
against curated per-trait lexicons.

BluePrint v2.1 reference: Section 3 — Stage 2 revision.
  - Pass 1 implements Layer A (lexicon matching) only.
  - Layer B (embedding-assisted semantic matching) is deferred to Pass 2.

Steps:
    Step 2.1 : Lowercase and normalise the turn text
    Step 2.2 : For each trait, check each direction's pattern list
    Step 2.3 : Tag evidence with context from SessionContext
    Step 2.4 : Classify source type (preference / dislike / correction / feedback)
    Step 2.5 : Emit StyleEvidence items with CANDIDATE_STABLE (Stage 3 decides final)

Constraint: No LLM calls. Pure deterministic pattern matching.
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

# ============================================================
# Shared trait pattern constants
# ============================================================
#
# These are exported at module level so other AISM stages (notably Stage 6
# feedback) can import the same patterns rather than maintaining their own
# copy. Until this refactor, Stage 6 had a literal duplicate of the humour
# low-list, which meant fixing the sign-flip on "not funny enough" had to
# be done in two places — and Stage 6's copy got missed.
#
# Source-of-truth rule: any per-trait pattern list referenced by both
# Stage 2 and Stage 6 lives here, and both stages import it.
# ============================================================

# --- Humour: low-direction patterns ---------------------------------------
# Used by Stage 2 (EXPLICIT_DISLIKE, value 0.2) and Stage 6 (CORRECTION,
# value 0.0). The bare "not funny" pattern uses a negative lookahead so it
# does NOT match "not funny enough" — that phrase means the user wants MORE
# humour, not less, and is handled by the high-direction list below.
HUMOUR_LOW_PATTERNS = [
  r"\bno\s+(?:more\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bdon'?t\s+(?:tell|make|give)\s+(?:me\s+)?(?:any\s+)?(?:more\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bdo\s+not\s+(?:tell|make|give)\s+(?:me\s+)?(?:any\s+)?(?:more\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bi\s+do\s+not\s+need\s+(?:any\s+)?(?:more\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bi\s+don'?t\s+need\s+(?:any\s+)?(?:more\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bstop\s+(?:telling|making|giving|talking\s+about)\s+(?:me\s+)?(?:any\s+)?(?:more\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bstop\s+joking\b",
  r"\bnever\s+(?:tell|make|give)\s+(?:me\s+)?(?:another\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bjokes?\s+(?:are|is)\s+not\s+funny\b",
  r"\bnot\s+funny\b(?!\s+enough)",
  r"\bbe\s+serious\b",
  r"\bless\s+humou?r\b",
]

# --- Humour: high-direction patterns --------------------------------------
# Includes the polarity-flipped phrasings the v1 lexicon got wrong:
#   "not funny enough"   → wants MORE humour
#   "we should make you more funnier" → wants MORE humour
#   "more humour please" → wants MORE humour
HUMOUR_HIGH_PATTERNS = [
  r"\bbe\s+funnier\b",
  r"\bnot\s+funny\s+enough\b",
  r"\b(?:more|much)\s+(?:funny|funnier)\b",
  r"\b(?:make\s+\w+(?:\s+\w+)?\s+)?(?:more\s+)?funnier\b",
  r"\bmore\s+humou?r(?:\s+please)?\b",
  r"\btell\s+me\s+(?:a\s+)?(?:few\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bcan\s+you\s+tell\s+me\s+(?:a\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\bi\s+(?:want|would\s+like|need)\s+(?:to\s+hear\s+)?(?:a\s+)?(?:few\s+)?(?:\w+\s+){0,3}jokes?\b",
  r"\b(?:another|more)\s+(?:\w+\s+){0,3}jokes?\s+please\b",
  r"\b(?:\w+\s+){0,3}jokes?\s+please\b",
  r"\blighten\s+up\b",
]


# ============================================================
# Trait lexicons (Step 2.2 data)
# ============================================================
#
# Structure:
#   TRAIT_LEXICONS[trait_name][direction_label] = {
#       "patterns": [list of regex strings],
#       "value": the observed value (scalar or categorical),
#       "source_type": how this cue should be classified,
#   }
#
# direction_label is a human-readable tag for auditing only (e.g. "high",
# "short", "paragraphs"). The "value" field is what gets stored.
# ============================================================

TRAIT_LEXICONS: Dict[str, Dict[str, Dict]] = {

  # ---- Directness ----
  "directness": {
    "high": {
      "patterns": [
        r"\bbe\s+(more\s+)?direct\b",
        r"\bjust\s+tell\s+me\b",
        r"\bdon'?t\s+sugarcoat\b",
        r"\bstop\s+hedging\b",
        r"\bget\s+to\s+the\s+point\b",
        r"\bcut\s+the\s+fluff\b",
        r"\bstraight\s+answer\b",
        r"\bno\s+bullshit\b",
        r"\bskip\s+the\s+preamble\b",
        # Polarity-flip phrasings ("not X enough" → wants more X).
        r"\bnot\s+direct\s+enough\b",
        r"\bnot\s+blunt\s+enough\b",
      ],
      "value":
      0.9,
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
    "low": {
      "patterns": [
        r"\bcould\s+you\s+maybe\b",
        r"\bif\s+it'?s\s+not\s+too\s+much\b",
        r"\bbe\s+more\s+tactful\b",
        r"\bsoften\s+the\b",
        r"\bdon'?t\s+be\s+so\s+blunt\b",
        r"\bmore\s+gently\b",
      ],
      "value":
      0.3,
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
  },

  # ---- Verbosity ----
  "verbosity": {
    "short": {
      "patterns": [
        r"\bbe\s+(more\s+)?concise\b",
        r"\btoo\s+wordy\b",
        r"\btoo\s+long\b",
        r"\bmake\s+it\s+shorter\b",
        r"\bkeep\s+it\s+brief\b",
        r"\bkeep\s+it\s+short\b",
        r"\bdon'?t\s+ramble\b",
        r"\bless\s+text\b",
        r"\bin\s+fewer\s+words\b",
        r"\btldr\b",
        # Polarity-flip phrasings.
        r"\bnot\s+(?:concise|brief|short)\s+enough\b",
      ],
      "value":
      "short",
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
    "long": {
      "patterns": [
        r"\bmore\s+detail(?:ed)?\b",
        r"\belaborate\b",
        r"\bexpand\s+on\b",
        r"\bgo\s+deeper\b",
        r"\bmore\s+thorough\b",
        r"\bin\s+depth\b",
        # Polarity-flip phrasings.
        r"\bnot\s+(?:detailed|thorough|long)\s+enough\b",
        r"\btoo\s+(?:short|brief|terse)\b",
      ],
      "value":
      "long",
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
  },

  # ---- Formatting ----
  "formatting": {
    "paragraphs": {
      "patterns": [
        r"\bno\s+bullet(s|\s+points)?\b",
        r"\bdon'?t\s+use\s+bullet\b",
        r"\bprefer\s+paragraphs\b",
        r"\bin\s+prose\b",
        r"\bnot\s+as\s+a\s+list\b",
      ],
      "value":
      "paragraphs",
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
    "bullets": {
      "patterns": [
        r"\buse\s+bullets\b",
        r"\bbullet\s+points\s+please\b",
        r"\blist\s+(them|it)\s+out\b",
        r"\bas\s+a\s+list\b",
      ],
      "value":
      "bullets",
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
  },

  # ---- Warmth / emotional style ----
  "warmth": {
    "high": {
      "patterns": [
        r"\bbe\s+warmer\b",
        r"\bmore\s+empathy\b",
        r"\bmore\s+supportive\b",
        r"\bmore\s+friendly\b",
        # Polarity-flip phrasings.
        r"\bnot\s+(?:warm|friendly|supportive|empathetic|kind)\s+enough\b",
      ],
      "value":
      0.8,
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
    "low": {
      "patterns": [
        r"\bstop\s+being\s+so\s+cheerful\b",
        r"\bless\s+enthusiastic\b",
        r"\bless\s+perky\b",
        r"\btoo\s+enthusiastic\b",
        r"\bdrop\s+the\s+(cheer|enthusiasm)\b",
      ],
      "value":
      0.3,
      "source_type":
      EvidenceSourceType.EXPLICIT_DISLIKE,
    },
  },

  # ---- Humour ----
  "humour": {
    "high": {
      "patterns": HUMOUR_HIGH_PATTERNS,
      "value": 0.8,
      "source_type": EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
    "low": {
      "patterns": HUMOUR_LOW_PATTERNS,
      "value": 0.2,
      "source_type": EvidenceSourceType.EXPLICIT_DISLIKE,
    },
  },

  # ---- Proactiveness ----
  "proactiveness": {
    "high": {
      "patterns": [
        r"\bsuggest\s+next\s+steps\b",
        r"\btell\s+me\s+what\s+to\s+do\s+next\b",
        r"\bbe\s+more\s+proactive\b",
        # Polarity-flip phrasings.
        r"\bnot\s+proactive\s+enough\b",
      ],
      "value":
      0.8,
      "source_type":
      EvidenceSourceType.EXPLICIT_PREFERENCE,
    },
    "low": {
      "patterns": [
        r"\bdon'?t\s+suggest\b",
        r"\bjust\s+answer\s+the\s+question\b",
        r"\bstop\s+offering\b",
      ],
      "value":
      0.2,
      "source_type":
      EvidenceSourceType.EXPLICIT_DISLIKE,
    },
  },
}

# ============================================================
# Correction / feedback patterns (Step 2.4)
# ============================================================
#
# These are generic "something was wrong with the last reply" markers that
# upgrade the source_type from PREFERENCE to CORRECTION when present.
# ============================================================

CORRECTION_MARKERS = [
  re.compile(r"\bthat\s+was\s+too\b", re.IGNORECASE),
  re.compile(r"\bstop\s+(doing\s+that|being\s+so)\b", re.IGNORECASE),
  re.compile(r"\bno,?\s+(be|don'?t|stop)\b", re.IGNORECASE),
  re.compile(r"\b(again|still)\s+(too|way\s+too)\b", re.IGNORECASE),
]

POSITIVE_FEEDBACK_MARKERS = [
  re.compile(r"\bthis\s+tone\s+is\s+(good|better)\b", re.IGNORECASE),
  re.compile(r"\bthat\s+was\s+(perfect|great|ideal)\b", re.IGNORECASE),
  re.compile(r"\bmuch\s+better\b", re.IGNORECASE),
  re.compile(r"\b(good|great)\s+length\b", re.IGNORECASE),
]

# ============================================================
# Extractor
# ============================================================


class StyleEvidenceExtractor:
  """
    Layer A (lexicon) extractor. Walks each trait's direction patterns and
    emits one StyleEvidence per matched direction.
    """

  def __init__(self) -> None:
    # Pre-compile all patterns.
    self._compiled: Dict[str, Dict[str, Dict]] = {}
    for trait, directions in TRAIT_LEXICONS.items():
      self._compiled[trait] = {}
      for direction_label, spec in directions.items():
        self._compiled[trait][direction_label] = {
          "patterns": [re.compile(p, re.IGNORECASE) for p in spec["patterns"]],
          "value": spec["value"],
          "source_type": spec["source_type"],
        }

  # ============================================================
  # Public entry point
  # ============================================================

  def extract(
    self,
    turn: ConversationTurn,
    session_context: SessionContext,
  ) -> List[StyleEvidence]:
    # --- Step 2.1: Normalise text ---
    text = turn.text  # case-insensitive regex handles casing

    # --- Step 2.4a: Detect correction/feedback envelope ---
    has_correction_marker = any(p.search(text) for p in CORRECTION_MARKERS)
    has_positive_marker = any(
      p.search(text) for p in POSITIVE_FEEDBACK_MARKERS)

    evidences: List[StyleEvidence] = []

    # --- Step 2.0 (NEW): polarity preprocessor -----------------------------
    # Detect "not X enough" / "too X" / "more X please" patterns and emit
    # polarity-aware evidence first. Tracks which (trait, direction) pairs
    # have already been claimed so the lexicon walk doesn't double-count.
    from .intensifier_handler import detect_polarity_evidence
    polarity_evidences = detect_polarity_evidence(
      turn, session_context, TRAIT_LEXICONS)
    polarity_claimed = {(e.trait, e.metadata.get("direction"))
                        for e in polarity_evidences}
    evidences.extend(polarity_evidences)

    # --- Step 2.2: Walk the lexicon ---
    for trait, directions in self._compiled.items():
      for direction_label, spec in directions.items():
        # Skip a (trait, direction) the polarity handler already claimed.
        if (trait, direction_label) in polarity_claimed:
          continue

        matched_pattern = self._first_match(spec["patterns"], text)
        if matched_pattern is None:
          continue

        # If the polarity handler claimed the OPPOSITE direction for this
        # same trait, the literal-lexicon match is almost certainly
        # collateral damage — e.g. "not funny enough" claimed humour=high,
        # so a stale "not funny" lexicon hit on the same turn must be
        # suppressed even though we patched the regex.
        opposite_direction_claimed = any(
          (e.trait == trait and e.metadata.get("direction") != direction_label)
          for e in polarity_evidences)
        if opposite_direction_claimed:
          continue

        # --- Step 2.4b: Upgrade source_type if correction marker present ---
        source_type = spec["source_type"]
        if has_correction_marker and source_type == EvidenceSourceType.EXPLICIT_PREFERENCE:
          source_type = EvidenceSourceType.CORRECTION
        if has_positive_marker:
          source_type = EvidenceSourceType.POSITIVE_FEEDBACK

        # --- Step 2.3: Tag with session context ---
        context = session_context.current_topic

        # --- Step 2.5: Emit evidence ---
        evidences.append(
          StyleEvidence(
            evidence_id=f"ev-{turn.turn_id}-{trait}-{direction_label}",
            turn_id=turn.turn_id,
            trait=trait,
            context=context,
            value=spec["value"],
            confidence=self._pattern_confidence(source_type),
            source_type=source_type,
            stability_class=StabilityClass.CANDIDATE_STABLE,
            timestamp=turn.timestamp,
            evidence_text=turn.text[:200],
            metadata={
              "matched_pattern": matched_pattern,
              "direction": direction_label
            },
          ))

    # If a turn contains both positive and negative humour cues, prefer the
    # negative cue. This prevents phrases like "no more jokes please" from
    # being misread as both "jokes please" and "no jokes".
    humour_low_present = any(
      e.trait == "humour" and isinstance(e.value, (int, float)) and e.value <= 0.3
      for e in evidences
    )
    if humour_low_present:
      evidences = [
        e for e in evidences
        if not (
          e.trait == "humour" and isinstance(e.value, (int, float)) and e.value > 0.3
        )
      ]

    return evidences

  # ============================================================
  # Helpers
  # ============================================================

  @staticmethod
  def _first_match(compiled_patterns: List[re.Pattern], text: str) -> str:
    """Return the matched string of the first matching pattern, or None."""
    for p in compiled_patterns:
      m = p.search(text)
      if m:
        return m.group(0)
    return None  # type: ignore[return-value]

  @staticmethod
  def _pattern_confidence(source_type: EvidenceSourceType) -> float:
    """
        Extractor confidence based on source type. These are Layer A confidences
        — the actual weight is decided in Stage 3. See BluePrint Section 3 Stage 3
        explicitness mapping.
        """
    return {
      EvidenceSourceType.EXPLICIT_PREFERENCE: 0.90,
      EvidenceSourceType.EXPLICIT_DISLIKE: 0.90,
      EvidenceSourceType.CORRECTION: 0.92,
      EvidenceSourceType.POSITIVE_FEEDBACK: 0.85,
      EvidenceSourceType.IMPLICIT_PATTERN: 0.55,
    }[source_type]
