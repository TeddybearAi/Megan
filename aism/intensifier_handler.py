"""
AISM — Negation / Intensifier Handler
=====================================

Deterministic polarity preprocessor. Detects phrasings like:

    "not funny enough"   → wants MORE humour
    "too long"           → wants LESS verbosity
    "more direct please" → wants MORE directness

…and emits the correct StyleEvidence regardless of which way the literal
words point.

Why this exists
---------------
The Stage 2 lexicon catches obvious patterns ("be more direct"), but
naturally-spoken English is full of polarity flips:

    "I'm trying to be funny but you are not funny enough"
        ^- the words "not funny" appear, but the user is asking for MORE
           humour, not less.

In v1 the bare ``\\bnot\\s+funny\\b`` pattern in Stage 2's humour-low
list fired and AISM stored ``humour=LOW`` from a turn that meant the
opposite. This module formalises the polarity flip so future cases —
"not concise enough", "not direct enough", "too warm", etc. — are handled
without each one needing a bespoke lexicon entry.

Design constraints
------------------
- R1: no LLM calls. Pure regex + an adjective→trait map.
- R2: no neural components.
- The handler emits StyleEvidence directly; the Stage 2 extractor adds
  these to its output so they flow through Stage 3 scoring like any other
  evidence.

Coverage today
--------------
- "not <adj> enough"  → wants MORE <adj>  (high-direction value)
- "too <adj>"         → wants LESS <adj>  (low/opposite value)
- "more <adj>" (with optional "please") → wants MORE
- "less <adj>" (with optional "please") → wants LESS

The adjective→(trait, high_direction) map below decides which trait each
adjective targets and which direction "more of it" means. Add to the map
to extend.
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
# Adjective → trait + high-direction map
# ============================================================
#
# Each entry says: when the user uses <adjective>, they're talking about
# <trait>, and "more <adjective>" means going in <high_direction>. The
# extractor then resolves <high_direction> against TRAIT_LEXICONS to get
# the actual stored value (0.8, "short", "bullets", etc.).

ADJECTIVE_TO_TRAIT: Dict[str, Tuple[str, str]] = {
  # directness
  "direct": ("directness", "high"),
  "blunt": ("directness", "high"),
  "tactful": ("directness", "low"),
  "diplomatic": ("directness", "low"),
  "gentle": ("directness", "low"),
  # verbosity
  "concise": ("verbosity", "short"),
  "brief": ("verbosity", "short"),
  "short": ("verbosity", "short"),
  "succinct": ("verbosity", "short"),
  "wordy": ("verbosity", "long"),
  "long": ("verbosity", "long"),
  "verbose": ("verbosity", "long"),
  "detailed": ("verbosity", "long"),
  "thorough": ("verbosity", "long"),
  # warmth
  "warm": ("warmth", "high"),
  "friendly": ("warmth", "high"),
  "supportive": ("warmth", "high"),
  "empathetic": ("warmth", "high"),
  "kind": ("warmth", "high"),
  "cheerful": ("warmth", "high"),
  "enthusiastic": ("warmth", "high"),
  "perky": ("warmth", "high"),
  "cold": ("warmth", "low"),
  "distant": ("warmth", "low"),
  # humour
  "funny": ("humour", "high"),
  "humorous": ("humour", "high"),
  "playful": ("humour", "high"),
  "witty": ("humour", "high"),
  "serious": ("humour", "low"),
  # proactiveness
  "proactive": ("proactiveness", "high"),
  "pushy": ("proactiveness", "high"),
}


def _opposite_direction(trait: str, direction: str) -> str:
  """Resolve the opposite direction label for a given trait."""
  pair_map = {
    "directness": {"high": "low", "low": "high"},
    "warmth": {"high": "low", "low": "high"},
    "humour": {"high": "low", "low": "high"},
    "proactiveness": {"high": "low", "low": "high"},
    "verbosity": {"short": "long", "long": "short"},
    "formatting": {"paragraphs": "bullets", "bullets": "paragraphs"},
  }
  return pair_map.get(trait, {}).get(direction, direction)


# ============================================================
# Polarity-pattern compilation
# ============================================================
#
# We pre-build the (compiled_regex, trait, direction) triples once at
# import time. Each trigger phrase tells us:
#   - which trait to emit evidence for
#   - which direction (high/low/short/long/...) to use the value of

_POLARITY_TRIGGERS: List[Tuple[re.Pattern, str, str, str]] = []
# tuple: (compiled_pattern, trait, direction_label, source_type_str)


def _compile_triggers() -> None:
  """Populate _POLARITY_TRIGGERS lazily on first import."""
  for adj, (trait, high_dir) in ADJECTIVE_TO_TRAIT.items():
    low_dir = _opposite_direction(trait, high_dir)

    # "not <adj> enough" → wants MORE adj → high_dir
    _POLARITY_TRIGGERS.append((
      re.compile(rf"\bnot\s+{adj}\s+enough\b", re.IGNORECASE),
      trait, high_dir, "correction"))

    # "too <adj>" → wants LESS adj → low_dir
    _POLARITY_TRIGGERS.append((
      re.compile(rf"\btoo\s+{adj}\b", re.IGNORECASE),
      trait, low_dir, "correction"))

    # "more <adj>" (optionally "please") → wants MORE adj → high_dir
    # Guard against "no more <adj>" / "don't need any more <adj>" by
    # disallowing immediately-preceding negation words.
    _POLARITY_TRIGGERS.append((
      re.compile(rf"(?<!\bno\s)(?<!\bany\s)\bmore\s+{adj}\b(?:\s+please)?",
                 re.IGNORECASE),
      trait, high_dir, "preference"))

    # "less <adj>" (optionally "please") → wants LESS adj → low_dir
    _POLARITY_TRIGGERS.append((
      re.compile(rf"\bless\s+{adj}\b(?:\s+please)?", re.IGNORECASE),
      trait, low_dir, "preference"))


_compile_triggers()


# ============================================================
# Public API
# ============================================================


def detect_polarity_evidence(
  turn: ConversationTurn,
  session_context: SessionContext,
  trait_lexicons: Dict[str, Dict[str, Dict]],
) -> List[StyleEvidence]:
  """Return polarity-aware StyleEvidence items for the given turn.

  Args:
    turn: the user turn being processed.
    session_context: AISM session context (for `current_topic`).
    trait_lexicons: the same TRAIT_LEXICONS dict Stage 2 uses, so we can
      look up the actual value for each (trait, direction) pair without
      duplicating it here.

  Each match emits one StyleEvidence. Duplicates within the same turn
  (e.g. "not funny enough... not funny enough!") are suppressed.
  """
  text = turn.text or ""
  if not text:
    return []

  out: List[StyleEvidence] = []
  seen: set[Tuple[str, str]] = set()

  for compiled, trait, direction, src_kind in _POLARITY_TRIGGERS:
    m = compiled.search(text)
    if not m:
      continue

    key = (trait, direction)
    if key in seen:
      continue
    seen.add(key)

    # Look up the actual value/source_type from the canonical lexicon.
    spec = trait_lexicons.get(trait, {}).get(direction)
    if spec is None:
      # Trait/direction we don't know about — silently skip rather than
      # emit nonsense.
      continue

    if src_kind == "correction":
      source_type = EvidenceSourceType.CORRECTION
      confidence = 0.92
    else:
      source_type = EvidenceSourceType.EXPLICIT_PREFERENCE
      confidence = 0.90

    out.append(
      StyleEvidence(
        evidence_id=f"ev-{turn.turn_id}-{trait}-{direction}-polarity",
        turn_id=turn.turn_id,
        trait=trait,
        context=session_context.current_topic,
        value=spec["value"],
        confidence=confidence,
        source_type=source_type,
        stability_class=StabilityClass.CANDIDATE_STABLE,
        timestamp=turn.timestamp,
        evidence_text=text[:200],
        metadata={
          "matched_pattern": m.group(0),
          "direction": direction,
          "polarity_handler": True,
        },
      ))

  return out
