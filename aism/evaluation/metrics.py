"""
AISM — Evaluation: Core Metrics
===============================
Implements the headline metrics from BluePrint v2.1 Section 9:

    - Trait Extraction Precision / Recall / F1   (categorical traits)
    - Trait Value MAE                            (scalar traits)
    - Profile Stability Index (PSI)
    - Adaptation Latency
    - Eligibility Classification Accuracy
    - Context Contamination Rate

Each metric is a pure function over pipeline state or run traces — no side
effects, no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data_models import InteractionProfile, StyleEvidence

# ============================================================
# Trait Extraction metrics
# ============================================================


@dataclass
class ExtractionCounts:
  true_positives: int = 0
  false_positives: int = 0
  false_negatives: int = 0

  @property
  def precision(self) -> float:
    tp, fp = self.true_positives, self.false_positives
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

  @property
  def recall(self) -> float:
    tp, fn = self.true_positives, self.false_negatives
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

  @property
  def f1(self) -> float:
    p, r = self.precision, self.recall
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def score_turn_extraction(
  predicted: Sequence[StyleEvidence],
  expected: Sequence[Dict[str, Any]],
) -> ExtractionCounts:
  """
    Score the per-turn extractor output against ground-truth evidence.

    expected items look like: {"trait": "verbosity", "value": "short"}.
    Predictions match by (trait, rounded-value) key; multiple predictions
    for the same key count as one (dedup).
    """
  pred_keys = {_key(ev.trait, ev.value) for ev in predicted}
  exp_keys = {_key(item["trait"], item["value"]) for item in expected}

  tp = len(pred_keys & exp_keys)
  fp = len(pred_keys - exp_keys)
  fn = len(exp_keys - pred_keys)
  return ExtractionCounts(tp, fp, fn)


def _key(trait: str, value: Any) -> Tuple[str, Any]:
  if isinstance(value, (int, float)):
    return (trait, round(float(value), 1))
  return (trait, value)


# ============================================================
# Profile dynamics: PSI, Adaptation Latency, Ground-truth match
# ============================================================


def profile_stability_index(
    profile_snapshots: Sequence[Dict[Tuple[str, str], Any]]) -> float:
  """
    PSI = 1 - mean(||P_t - P_{t-1}||) over stable-regime snapshots.

    Snapshot format: {(trait, context): value} per turn.
    For scalars, distance is absolute difference.
    For categoricals, distance is 0 (same) or 1 (different).
    Only traits present in both snapshots contribute.

    Higher PSI = more stable.
    """
  if len(profile_snapshots) < 2:
    return 1.0

  distances: List[float] = []
  for prev, curr in zip(profile_snapshots, profile_snapshots[1:]):
    shared_keys = set(prev.keys()) & set(curr.keys())
    if not shared_keys:
      continue
    pair_distances: List[float] = []
    for key in shared_keys:
      pv, cv = prev[key], curr[key]
      if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
        pair_distances.append(abs(float(pv) - float(cv)))
      else:
        pair_distances.append(0.0 if pv == cv else 1.0)
    distances.append(mean(pair_distances))

  if not distances:
    return 1.0
  return max(0.0, 1.0 - mean(distances))


def adaptation_latency(
  snapshots: Sequence[Dict[Tuple[str, str], Any]],
  trait: str,
  context: str,
  target_value: Any,
  correction_turn_index: int,
  tolerance: float = 0.15,
) -> Optional[int]:
  """
    Number of turns from `correction_turn_index` until the trait reaches
    `target_value` (within tolerance for scalars, exact match for categoricals).

    Returns None if the target is never reached in the snapshot range.
    """
  for offset, snap in enumerate(snapshots[correction_turn_index:], start=0):
    val = snap.get((trait, context))
    if val is None:
      continue
    if isinstance(val, (int, float)) and isinstance(target_value,
                                                    (int, float)):
      if abs(float(val) - float(target_value)) <= tolerance:
        return offset
    else:
      if val == target_value:
        return offset
  return None


def ground_truth_match_rate(
  profile: InteractionProfile,
  ground_truth: Dict[Tuple[str, str], Any],
  tolerance: float = 0.2,
) -> Dict[str, Any]:
  """
    Report match rate between final profile and ground truth.

    Returns {
        "n_expected": int,
        "n_matched": int,
        "n_missing": int,     # expected trait absent from profile
        "match_rate": float,
        "per_trait": {key: {"expected": ..., "actual": ..., "match": bool}}
    }
    """
  per_trait: Dict[str, Dict[str, Any]] = {}
  matched = 0
  missing = 0

  for (trait, context), expected in ground_truth.items():
    state = profile.traits.get(trait, {}).get(context)
    actual = state.value if state is not None else None

    if actual is None:
      missing += 1
      match = False
    elif isinstance(expected, (int, float)) and isinstance(actual,
                                                           (int, float)):
      match = abs(float(actual) - float(expected)) <= tolerance
    else:
      match = actual == expected

    if match:
      matched += 1

    per_trait[f"{trait}[{context}]"] = {
      "expected": expected,
      "actual": actual,
      "match": match,
    }

  n = len(ground_truth)
  return {
    "n_expected": n,
    "n_matched": matched,
    "n_missing": missing,
    "match_rate": round(matched / n, 4) if n > 0 else 0.0,
    "per_trait": per_trait,
  }


# ============================================================
# Stage 1 metrics: eligibility accuracy + contamination rate
# ============================================================


@dataclass
class EligibilityCounts:
  true_positives: int = 0  # correctly identified as eligible
  true_negatives: int = 0  # correctly identified as ineligible
  false_positives: int = 0  # ineligible but system said eligible
  false_negatives: int = 0  # eligible but system said ineligible

  @property
  def accuracy(self) -> float:
    total = (
      self.true_positives + self.true_negatives + self.false_positives +
      self.false_negatives)
    return (
      (self.true_positives + self.true_negatives) /
      total if total > 0 else 0.0)

  @property
  def contamination_rate(self) -> float:
    """Ineligible turns wrongly admitted as eligible / total ineligible."""
    denom = self.true_negatives + self.false_positives
    return self.false_positives / denom if denom > 0 else 0.0


def score_eligibility(
  predicted_eligible: bool,
  expected_eligible: bool,
  counts: EligibilityCounts,
) -> None:
  """Update `counts` in-place."""
  if expected_eligible and predicted_eligible:
    counts.true_positives += 1
  elif not expected_eligible and not predicted_eligible:
    counts.true_negatives += 1
  elif not expected_eligible and predicted_eligible:
    counts.false_positives += 1
  else:
    counts.false_negatives += 1
