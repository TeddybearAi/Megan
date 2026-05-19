"""
AISM — STAGE 1: Eligibility Filter
==================================
Determines whether a user turn is a valid signal for long-term style modelling
or content-for-rewrite that would pollute the profile.

BluePrint v2.1 reference: Section 2 — Stage 1 revision (rule-based only).

Steps:
    Step 1.1 : Role check (only USER turns are eligible)
    Step 1.2 : Keyword/regex patterns for obvious contamination
    Step 1.3 : Structural heuristics (quoted text, email headers/footers)
    Step 1.4 : Length/form heuristics (long blocks suggesting pasted content)
    Step 1.5 : Combined decision with conservative uncertain-handling

Constraint: No LLM calls. Pure deterministic logic.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .data_models import ConversationTurn, EligibilityDecision, TurnRole


class EligibilityFilter:
  """Rule-based eligibility filter. See module docstring for step breakdown."""

  # ---- Step 1.2: Ineligible keyword patterns ----
  # Patterns that strongly indicate content-for-rewrite or roleplay tasks.
  # Written as regex to allow word-boundary and variant matching.
  INELIGIBLE_PATTERNS: List[str] = [
    r"\brewrite\s+(this|that|it|the following|my|the)",
    r"\btranslate\s+(this|that|into|the following)",
    r"\bdraft\s+(an?\s+)?(email|letter|message|reply|response)",
    r"\bwrite\s+(an?\s+)?(email|letter|essay|article|story|poem|speech)\s+(for|to)\s+(my|our|the|a)",
    r"\bfor\s+my\s+(professor|supervisor|boss|teacher|client|manager|colleague)",
    r"\b(pretend|roleplay|role-play|act\s+as|you\s+are\s+now)\b",
    r"\bin\s+the\s+style\s+of\b",
    r"\bsummari[sz]e\s+(the|this|that|the\s+following)",
    r"\bparaphrase\s+(this|that|the following)",
    r"\bedit\s+(this|my|the\s+following)",
    r"\bproofread\s+(this|my|the)",
  ]

  # ---- Step 1.3: Structural heuristics ----
  # Email-style markers and quoted-block indicators.
  EMAIL_MARKER_PATTERNS: List[str] = [
    r"^\s*From:\s+",
    r"^\s*To:\s+",
    r"^\s*Subject:\s+",
    r"^\s*CC:\s+",
    r"^\s*Dear\s+(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|[A-Z][a-z]+)",
    r"(Sincerely|Best regards|Kind regards|Yours truly|Warm regards)\s*,?\s*$",
  ]

  QUOTED_BLOCK_MIN_CHARS = 50  # contiguous quoted content of this length triggers

  # ---- Step 1.4: Length/form heuristics ----
  LONG_TURN_CHAR_THRESHOLD = 500  # turns over this long are suspicious
  LONG_TURN_NEWLINE_THRESHOLD = 4  # paragraph-break count that confirms

  # ---- Step 1.5: Decision thresholds ----
  # contamination_score is the max of the per-heuristic scores.
  THRESHOLD_CLEAR_REJECT = 0.7  # >= this -> definitely ineligible
  THRESHOLD_UNCERTAIN = 0.4  # in [0.4, 0.7) -> conservatively ineligible

  def __init__(self) -> None:
    # Pre-compile for efficiency; multiline/case-insensitive where relevant.
    self._ineligible_re = [
      re.compile(p, re.IGNORECASE) for p in self.INELIGIBLE_PATTERNS
    ]
    self._email_re = [
      re.compile(p, re.IGNORECASE | re.MULTILINE)
      for p in self.EMAIL_MARKER_PATTERNS
    ]
    # Quoted content: lines starting with > OR text inside matching quotes.
    self._quoted_line_re = re.compile(r"^\s*>\s*.+$", re.MULTILINE)
    self._quoted_span_re = re.compile(
      r'"([^"]{%d,})"' % self.QUOTED_BLOCK_MIN_CHARS)

  # ============================================================
  # Public entry point
  # ============================================================

  def evaluate_turn(self, turn: ConversationTurn) -> EligibilityDecision:
    # --- Step 1.1: Role check ---
    if turn.role != TurnRole.USER:
      return EligibilityDecision(
        turn_id=turn.turn_id,
        is_eligible=False,
        reason="Non-user turn; only user turns update the profile.",
        contamination_score=1.0,
        eligibility_confidence=1.0,
      )

    text = turn.text

    # --- Step 1.2: Keyword score ---
    kw_score, kw_hit = self._keyword_score(text)

    # --- Step 1.3: Structural score ---
    struct_score, struct_hit = self._structural_score(text)

    # --- Step 1.4: Length/form score ---
    length_score, length_hit = self._length_score(text)

    # --- Step 1.5: Combined decision ---
    contamination = max(kw_score, struct_score, length_score)
    reasons: List[str] = [h for h in (kw_hit, struct_hit, length_hit) if h]

    if contamination >= self.THRESHOLD_CLEAR_REJECT:
      return EligibilityDecision(
        turn_id=turn.turn_id,
        is_eligible=False,
        reason="Ineligible: " + "; ".join(reasons),
        contamination_score=contamination,
        # High confidence that this decision is correct.
        eligibility_confidence=0.9,
      )

    if contamination >= self.THRESHOLD_UNCERTAIN:
      # Conservative default: uncertain -> exclude.
      # eligibility_confidence is LOW here — this gates Stage 3 scoring.
      return EligibilityDecision(
        turn_id=turn.turn_id,
        is_eligible=False,
        reason="Eligibility uncertain: " +
        "; ".join(reasons or ["borderline contamination score"]),
        contamination_score=contamination,
        eligibility_confidence=0.5,
      )

    return EligibilityDecision(
      turn_id=turn.turn_id,
      is_eligible=True,
      reason="No contamination rule matched.",
      contamination_score=contamination,
      eligibility_confidence=0.85,
    )

  # ============================================================
  # Step-level helpers
  # ============================================================

  def _keyword_score(self, text: str) -> Tuple[float, str]:
    """Step 1.2 internals. Returns (score, human-readable reason or '')."""
    for pattern in self._ineligible_re:
      m = pattern.search(text)
      if m:
        return 0.95, f"matched ineligible pattern '{m.group(0)}'"
    return 0.0, ""

  def _structural_score(self, text: str) -> Tuple[float, str]:
    """Step 1.3 internals: email markers and quoted content."""
    # Email-style markers are strong signals.
    for pattern in self._email_re:
      m = pattern.search(text)
      if m:
        return 0.85, f"email-style marker detected: '{m.group(0).strip()}'"

    # Quoted blocks via '>' line-prefix.
    quoted_lines = self._quoted_line_re.findall(text)
    quoted_chars = sum(len(line) for line in quoted_lines)
    if quoted_chars >= self.QUOTED_BLOCK_MIN_CHARS:
      return 0.75, f"substantial quoted block ({quoted_chars} chars)"

    # Quoted spans inside double quotes (only when long).
    if self._quoted_span_re.search(text):
      return 0.6, "long quoted span in double quotes"

    return 0.0, ""

  def _length_score(self, text: str) -> Tuple[float, str]:
    """Step 1.4 internals. Long + multi-paragraph -> likely pasted content."""
    if len(text) < self.LONG_TURN_CHAR_THRESHOLD:
      return 0.0, ""

    newline_count = text.count("\n")
    if newline_count >= self.LONG_TURN_NEWLINE_THRESHOLD:
      return 0.55, f"long multi-paragraph turn ({len(text)} chars, {newline_count} breaks)"

    # Long but not paragraphed — could be a single long question.
    return 0.25, ""
