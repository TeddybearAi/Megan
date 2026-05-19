"""
AISM — STAGE 2C: Cross-Turn Implicit Pattern Aggregator
=======================================================
Detects patterns the single-turn extractor misses because they only become
visible when you look across many turns. Example: the user has asked for
shorter answers 5 times in the last 20 turns even though no single turn used
a lexicon phrase strong enough to fire Layer A.

BluePrint v2.1 reference: Section 3 — Stage 2 aggregator note.

Steps:
    Step 2C.1 : Scan the last N evidence items (or conversation turns) for
                repeated (trait, context, value) signals.
    Step 2C.2 : For each trait that appears ≥ MIN_REPETITIONS times, compute
                the modal value and its support.
    Step 2C.3 : Emit a single IMPLICIT_PATTERN StyleEvidence with confidence
                proportional to the support fraction.
    Step 2C.4 : Dedup — only emit if the aggregated pattern differs from the
                current profile value, otherwise there's nothing to learn.

Design rule (BluePrint R1): operates on already-extracted structured evidence
only. Never parses raw text. If a pattern requires semantic text understanding,
it should have been caught by Layer A or B, not here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Optional, Sequence

from .data_models import (
  EvidenceSourceType,
  InteractionProfile,
  StabilityClass,
  StyleEvidence,
)
from .session_context import SessionContext


class PatternAggregator:
  """Emits aggregate IMPLICIT_PATTERN evidence from the evidence log."""

  LOOKBACK_WINDOW = 30  # look at last N evidence items
  MIN_REPETITIONS = 3  # how many times the same trait must appear
  MIN_MODE_FRACTION = 0.6  # mode must cover at least this fraction

  # Confidence scaling: a trait that hit 6/6 samples with the same value
  # should be more confident than 3/5.
  CONFIDENCE_BASE = 0.40
  CONFIDENCE_PER_FRACTION = 0.30  # mode_fraction × this, added to base

  def aggregate(
    self,
    evidence_log: Sequence[StyleEvidence],
    profile: InteractionProfile,
    session_context: SessionContext,
    now_timestamp,
  ) -> List[StyleEvidence]:
    if not evidence_log:
      return []

    # --- Step 2C.1: slice lookback window ---
    window = list(evidence_log[-self.LOOKBACK_WINDOW:])

    # --- Step 2C.2: group by (trait, context) ---
    groups: dict = defaultdict(list)
    for ev in window:
      # Only use already-explicit or correction evidence for aggregation;
      # aggregating implicit evidence would double-count.
      if ev.source_type == EvidenceSourceType.IMPLICIT_PATTERN:
        continue
      groups[(ev.trait, ev.context)].append(ev)

    emitted: List[StyleEvidence] = []

    for (trait, context), items in groups.items():
      if len(items) < self.MIN_REPETITIONS:
        continue

      # Modal value + support.
      values = [self._hashable_value(it.value) for it in items]
      counter = Counter(values)
      mode_value, mode_count = counter.most_common(1)[0]
      mode_fraction = mode_count / len(values)
      if mode_fraction < self.MIN_MODE_FRACTION:
        # Too inconsistent across the window — not a stable pattern.
        continue

      # --- Step 2C.4: only emit if this adds information ---
      current = profile.traits.get(trait, {}).get(context)
      current_value = current.value if current is not None else None
      if current_value is not None and self._hashable_value(
          current_value) == mode_value:
        # The profile already reflects this value; no new information.
        continue

      # Reconstruct original value (Counter keys are hashable forms).
      original_value = self._original_value_from_items(items, mode_value)

      confidence = min(
        1.0,
        self.CONFIDENCE_BASE + self.CONFIDENCE_PER_FRACTION * mode_fraction)

      # --- Step 2C.3: emit aggregate evidence ---
      emitted.append(
        StyleEvidence(
          evidence_id=f"agg-{trait}-{context}-{now_timestamp.timestamp():.0f}",
          turn_id="aggregate",
          trait=trait,
          context=context,
          value=original_value,
          confidence=round(confidence, 4),
          source_type=EvidenceSourceType.IMPLICIT_PATTERN,
          stability_class=StabilityClass.CANDIDATE_STABLE,
          timestamp=now_timestamp,
          evidence_text=(
            f"aggregate: {mode_count}/{len(items)} of recent "
            f"{trait}[{context}] signals supported '{original_value}'"),
          metadata={
            "source": "pattern_aggregator",
            "support_count": mode_count,
            "total_in_window": len(items),
            "mode_fraction": round(mode_fraction, 3),
          },
        ))

    return emitted

  # ============================================================
  # Helpers
  # ============================================================

  @staticmethod
  def _hashable_value(value):
    if isinstance(value, (int, float)):
      return ("num", round(float(value), 2))
    return ("str", value)

  @staticmethod
  def _original_value_from_items(items, mode_key):
    """Return the first original (non-hashed) value matching mode_key."""
    for it in items:
      if PatternAggregator._hashable_value(it.value) == mode_key:
        return it.value
    return None  # shouldn't happen
