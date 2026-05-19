"""
AISM — STAGE 3: Evidence Scoring and Gating
===========================================
Gated log-linear scorer. Applies hard gates (vetoes), then computes a weight
via a log-linear formula, then classifies the final stability class.

BluePrint v2.1 reference: Section 3 — Stage 3 (unchanged from v2).

Steps:
    Step 3.1 : Hard gates — eligibility, scope validity, extractor confidence
    Step 3.2 : Compute four features — explicitness, frequency, recency, consistency
    Step 3.3 : Log-linear combination → sigmoid → weight in [0, 1]
    Step 3.4 : Detect hard override (bypasses smoothed update in Stage 4)
    Step 3.5 : Final stability classification → STABLE | SESSION | IGNORE

Formula (BluePrint §3 Stage 3):
    logit(w) = β₀ + β_e·explicitness
                  + β_f·log(1+frequency)
                  + β_r·recency_decay
                  + β_c·consistency
    w = σ(logit(w))

Constraint: No LLM calls. All computation is deterministic.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import variance
from typing import Dict, List

from .data_models import (
  EvidenceSourceType,
  StabilityClass,
  StyleEvidence,
  WeightedEvidence,
)


class EvidenceScorer:
  """Gated log-linear evidence scorer."""

  # ============================================================
  # Step 3.1: Gate thresholds
  # ============================================================
  # These are genuine vetoes. Failing any gate → weight = 0, class = IGNORE,
  # evidence logged but never applied to the profile.
  THRESHOLD_ELIGIBILITY = 0.6
  THRESHOLD_SCOPE = 0.5
  THRESHOLD_CONFIDENCE = 0.5

  # ============================================================
  # Step 3.3: Log-linear coefficients
  # ============================================================
  # Hand-tuned values per BluePrint §3 Stage 3.
  # Upgrade path: fit via logistic regression on Tier 2 annotated data.
  BETA_0 = -1.5
  BETA_E = 3.0  # explicitness
  BETA_F = 1.2  # frequency (log-damped)
  BETA_R = 1.8  # recency decay
  BETA_C = 1.5  # consistency

  # ============================================================
  # Step 3.2 feature constants
  # ============================================================
  EXPLICITNESS_MAP: Dict[EvidenceSourceType, float] = {
    EvidenceSourceType.EXPLICIT_PREFERENCE: 1.0,
    EvidenceSourceType.EXPLICIT_DISLIKE: 1.0,
    EvidenceSourceType.CORRECTION: 0.9,
    EvidenceSourceType.POSITIVE_FEEDBACK: 0.7,
    EvidenceSourceType.IMPLICIT_PATTERN: 0.4,
  }

  # Half-life ~7 days for recency decay. λ = ln(2) / half_life_seconds.
  RECENCY_HALF_LIFE_SECONDS = 7 * 24 * 3600
  RECENCY_LAMBDA = math.log(2) / RECENCY_HALF_LIFE_SECONDS

  FREQUENCY_WINDOW = 50  # last-N evidence items for frequency count
  CONSISTENCY_MIN_SAMPLES = 3  # need at least N prior samples to compute

  # ============================================================
  # Step 3.4: Hard override criteria
  # ============================================================
  # Hard override bypasses Stage 4's smoothed update. Intent: "user is
  # telling me I got it wrong RIGHT NOW" should snap immediately; a
  # first-time preference statement should ease in via the smoothed path.
  #
  # With extractor confidences:
  #   EXPLICIT_PREFERENCE = 0.90  → below threshold → smoothed
  #   EXPLICIT_DISLIKE    = 0.90  → below threshold → smoothed
  #   CORRECTION          = 0.92  → above threshold → hard override
  #   (Feedback processor's corrections emit at 0.94.)
  # If a preference becomes a stable trait through repetition, the
  # repetition itself drives the smoothed update — no override needed.
  HARD_OVERRIDE_SOURCE_TYPES = {
    EvidenceSourceType.EXPLICIT_PREFERENCE,
    EvidenceSourceType.EXPLICIT_DISLIKE,
    EvidenceSourceType.CORRECTION,
  }
  HARD_OVERRIDE_MIN_CONFIDENCE = 0.91

  # ============================================================
  # Step 3.5: Stability class thresholds
  # ============================================================
  # Two paths to STABLE (via _classify_stability):
  #   (a) Hard override → STABLE regardless of frequency (snap-update).
  #   (b) Explicit evidence (preference/dislike/correction/positive) with
  #       weight >= STABLE_WEIGHT_THRESHOLD → STABLE on first occurrence.
  #       Rationale: users who state a preference expect it to persist.
  #   (c) Implicit pattern evidence with frequency >= IMPLICIT_STABLE_FREQUENCY
  #       → STABLE. Behavioural patterns need repetition to be trusted.
  # Everything else with weight >= IGNORE_THRESHOLD goes to SESSION.
  STABLE_WEIGHT_THRESHOLD = 0.5
  IMPLICIT_STABLE_FREQUENCY = 2  # implicit-path repetition requirement
  IGNORE_WEIGHT_THRESHOLD = 0.1  # below this → IGNORE

  _EXPLICIT_SOURCE_TYPES = {
    EvidenceSourceType.EXPLICIT_PREFERENCE,
    EvidenceSourceType.EXPLICIT_DISLIKE,
    EvidenceSourceType.CORRECTION,
    EvidenceSourceType.POSITIVE_FEEDBACK,
  }

  # ============================================================
  # Construction
  # ============================================================

  def __init__(self) -> None:
    # Evidence log for frequency/recency/consistency lookup.
    # The pipeline appends to this; we only read.
    self.evidence_log: List[StyleEvidence] = []

  # ============================================================
  # Public entry point
  # ============================================================

  def score(
    self,
    evidence: StyleEvidence,
    eligibility_confidence: float,
    scope_validity: float = 1.0,
  ) -> WeightedEvidence:
    """
        Score a single piece of evidence.

        Parameters
        ----------
        evidence : StyleEvidence
            The evidence item produced by Stage 2.
        eligibility_confidence : float
            From Stage 1's EligibilityDecision. Feeds the eligibility gate.
        scope_validity : float
            Defaults to 1.0 ("general" scope). Lower this if the evidence
            comes from a narrower scope that shouldn't update global traits.
        """
    features: Dict[str, float] = {
      "eligibility_confidence": eligibility_confidence,
      "scope_validity": scope_validity,
      "extractor_confidence": evidence.confidence,
    }

    # --- Step 3.1: Hard gates ---
    gate_pass = (
      eligibility_confidence >= self.THRESHOLD_ELIGIBILITY
      and scope_validity >= self.THRESHOLD_SCOPE
      and evidence.confidence >= self.THRESHOLD_CONFIDENCE)

    if not gate_pass:
      return WeightedEvidence(
        evidence=evidence,
        weight=0.0,
        gate_pass=False,
        final_stability=StabilityClass.IGNORE,
        is_hard_override=False,
        features=features,
      )

    # --- Step 3.2: Compute log-linear features ---
    explicitness = self.EXPLICITNESS_MAP[evidence.source_type]
    frequency = self._compute_frequency(evidence)
    recency = self._compute_recency(evidence)
    consistency = self._compute_consistency(evidence)

    features.update(
      {
        "explicitness": explicitness,
        "frequency": float(frequency),
        "recency_decay": recency,
        "consistency": consistency,
      })

    # --- Step 3.3: Log-linear weight ---
    logit_w = (
      self.BETA_0 + self.BETA_E * explicitness +
      self.BETA_F * math.log1p(frequency) + self.BETA_R * recency +
      self.BETA_C * consistency)
    weight = self._sigmoid(logit_w)
    features["logit_w"] = logit_w
    features["weight"] = weight

    # --- Step 3.4: Hard override detection ---
    is_hard_override = (
      evidence.source_type in self.HARD_OVERRIDE_SOURCE_TYPES
      and evidence.confidence >= self.HARD_OVERRIDE_MIN_CONFIDENCE)

    # --- Step 3.5: Final stability class ---
    final_stability = self._classify_stability(
      weight=weight,
      frequency=frequency,
      is_hard_override=is_hard_override,
      source_type=evidence.source_type,
    )

    return WeightedEvidence(
      evidence=evidence,
      weight=round(weight, 4),
      gate_pass=True,
      final_stability=final_stability,
      is_hard_override=is_hard_override,
      features=features,
    )

  # ============================================================
  # Evidence log management (used by feature computation)
  # ============================================================

  def register_evidence(self, evidence: StyleEvidence) -> None:
    """Called by the pipeline AFTER scoring to record this evidence for
        future frequency/consistency computations. Bounded size to keep lookups
        cheap — we only need the window size, plus a bit of headroom."""
    self.evidence_log.append(evidence)
    max_keep = self.FREQUENCY_WINDOW * 4
    if len(self.evidence_log) > max_keep:
      self.evidence_log = self.evidence_log[-max_keep:]

  # ============================================================
  # Step-level helpers
  # ============================================================

  def _compute_frequency(self, evidence: StyleEvidence) -> int:
    """
        Count prior evidence items for the same (trait, context) within the
        last FREQUENCY_WINDOW items. The current item is NOT counted (it hasn't
        been registered yet — prevents self-reinforcing loop).
        """
    recent = self.evidence_log[-self.FREQUENCY_WINDOW:]
    return sum(
      1 for e in recent
      if e.trait == evidence.trait and e.context == evidence.context)

  def _compute_recency(self, evidence: StyleEvidence) -> float:
    """
        Recency decay: exp(-λ · Δt) where Δt is seconds since the previous
        most-recent evidence for the same trait. New traits (no prior evidence)
        return 1.0 — maximum freshness.
        """
    same_trait = [
      e for e in self.evidence_log
      if e.trait == evidence.trait and e.context == evidence.context
    ]
    if not same_trait:
      return 1.0
    last = same_trait[-1]
    # Ensure both are tz-aware.
    now = evidence.timestamp
    if now.tzinfo is None:
      now = now.replace(tzinfo=timezone.utc)
    then = last.timestamp
    if then.tzinfo is None:
      then = then.replace(tzinfo=timezone.utc)
    dt = max(0.0, (now - then).total_seconds())
    return math.exp(-self.RECENCY_LAMBDA * dt)

  def _compute_consistency(self, evidence: StyleEvidence) -> float:
    """
        Consistency = 1 - normalised variance of the last N values for this
        (trait, context). For categorical traits, variance is computed as the
        fraction of disagreeing values with the mode.
        """
    same = [
      e for e in self.evidence_log
      if e.trait == evidence.trait and e.context == evidence.context
    ]
    if len(same) < self.CONSISTENCY_MIN_SAMPLES:
      # Not enough samples to judge — neutral score.
      return 0.7

    values = [e.value for e in same[-self.FREQUENCY_WINDOW:]]

    # Scalar path
    if all(isinstance(v, (int, float)) for v in values):
      if len(values) < 2:
        return 1.0
      # Normalise by assumed trait range [0, 1] → max variance is 0.25.
      var = variance([float(v) for v in values])
      normalised = min(1.0, var / 0.25)
      return max(0.0, 1.0 - normalised)

    # Categorical path
    mode_value = max(set(values), key=values.count)
    mode_count = values.count(mode_value)
    return mode_count / len(values)

  @staticmethod
  def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
      z = math.exp(-x)
      return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

  def _classify_stability(
    self,
    weight: float,
    frequency: int,
    is_hard_override: bool,
    source_type: EvidenceSourceType,
  ) -> StabilityClass:
    """
        Step 3.5 — final stability classification.

        Decision order:
          (1) Hard override                         → STABLE (snap-update path)
          (2) weight < IGNORE_WEIGHT_THRESHOLD      → IGNORE
          (3) weight < STABLE_WEIGHT_THRESHOLD      → SESSION
          (4) Explicit source + weight OK           → STABLE (first-occurrence ok)
          (5) Implicit source + frequency OK        → STABLE (repetition required)
          (6) Otherwise                             → SESSION
        """
    # (1)
    if is_hard_override:
      return StabilityClass.STABLE
    # (2)
    if weight < self.IGNORE_WEIGHT_THRESHOLD:
      return StabilityClass.IGNORE
    # (3)
    if weight < self.STABLE_WEIGHT_THRESHOLD:
      return StabilityClass.SESSION
    # (4) Explicit path — user stated it clearly; persist on first occurrence.
    if source_type in self._EXPLICIT_SOURCE_TYPES:
      return StabilityClass.STABLE
    # (5) Implicit path — require repetition.
    if frequency >= self.IMPLICIT_STABLE_FREQUENCY:
      return StabilityClass.STABLE
    # (6)
    return StabilityClass.SESSION
