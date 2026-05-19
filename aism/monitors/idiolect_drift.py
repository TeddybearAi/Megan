"""
AISM — Monitor 2/4: Idiolect Drift
==================================
Tracks how far the assistant's recent reply style has drifted from its
original baseline voice. Prevents the smoothed-update mechanism from
gradually turning the assistant into a copy of the user.

BluePrint v2.1 reference: Section 5 — Over-mimicry safeguards.

Steps:
    Step M2.1 : Build a baseline embedding from the first N assistant replies
                (BASELINE_SIZE).
    Step M2.2 : For subsequent replies, compute embedding distance to the
                baseline centroid.
    Step M2.3 : Rolling mean of recent distances → drift score.
    Step M2.4 : Flag if drift exceeds threshold.

Uses the Embedder abstraction — same embedder as Stage 2 Layer B is fine.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from ..embedder import Embedder, cosine_similarity


class IdiolectDriftMonitor:
  """
    Tracks drift from baseline assistant voice. Requires an Embedder.
    """

  BASELINE_SIZE = 5  # first N replies define the baseline
  ROLLING_WINDOW = 10  # average drift over last N non-baseline replies
  # Thresholds: drift_score = 1 - cos(reply, baseline_centroid).
  # Higher = further from baseline. 0 = identical, 1 = orthogonal.
  DRIFT_THRESHOLD = 0.35

  def __init__(self, embedder: Embedder) -> None:
    self.embedder = embedder
    self._baseline_vecs: List[List[float]] = []
    self._baseline_centroid: Optional[List[float]] = None
    self._recent_distances: Deque[float] = deque(maxlen=self.ROLLING_WINDOW)

  # ============================================================
  # Public API
  # ============================================================

  def record_reply(self, reply: str) -> dict:
    """Call with every assistant reply. Returns drift metrics."""
    vec = self.embedder.embed(reply or "")

    # --- Step M2.1: build baseline ---
    if len(self._baseline_vecs) < self.BASELINE_SIZE:
      self._baseline_vecs.append(vec)
      if len(self._baseline_vecs) == self.BASELINE_SIZE:
        self._baseline_centroid = self._centroid(self._baseline_vecs)
      return {
        "baseline_building": True,
        "baseline_size": len(self._baseline_vecs),
        "drift_score": 0.0,
        "rolling_drift": 0.0,
        "flagged": False,
      }

    # --- Step M2.2: distance to baseline ---
    assert self._baseline_centroid is not None
    sim = cosine_similarity(vec, self._baseline_centroid)
    drift = max(0.0, 1.0 - sim)
    self._recent_distances.append(drift)

    # --- Step M2.3: rolling mean ---
    rolling = sum(self._recent_distances) / len(self._recent_distances)

    # --- Step M2.4: flag ---
    flagged = rolling > self.DRIFT_THRESHOLD

    return {
      "baseline_building": False,
      "drift_score": round(drift, 4),
      "rolling_drift": round(rolling, 4),
      "flagged": flagged,
      "baseline_size": len(self._baseline_vecs),
      "recent_samples": len(self._recent_distances),
    }

  # ============================================================
  # Helpers
  # ============================================================

  @staticmethod
  def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
      return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
