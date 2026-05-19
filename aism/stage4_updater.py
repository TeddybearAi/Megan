"""
AISM — STAGE 4: Profile State Update
====================================
Applies a WeightedEvidence item to either the stable profile or the session
overlay, depending on its final stability class. Implements the smoothed
update formula and the three runaway-drift guardrails.

BluePrint v2.1 reference: Section 3 — Stage 4.

Steps:
    Step 4.1 : Route by final stability class (STABLE / SESSION / IGNORE)
    Step 4.2 : SESSION branch — write to SessionOverlay with timestamp
    Step 4.3 : STABLE branch, new trait — create TraitState directly
    Step 4.4 : STABLE branch, scalar trait — smoothed update:
                   p_new = (1 - α·w)·p_old + (α·w)·x
               with per-update cap δ and per-session drift cap Δ
    Step 4.5 : STABLE branch, categorical trait — threshold or hard-override
    Step 4.6 : Update volatility as stddev of recent_values (not decay)
    Step 4.7 : Update metadata (evidence_count, last_updated, source_summary)

Constraint: No LLM calls. Deterministic arithmetic only.
"""

from __future__ import annotations

from statistics import stdev
from typing import Any

from .data_models import (
  InteractionProfile,
  SessionOverlay,
  StabilityClass,
  TraitState,
  WeightedEvidence,
)


class ProfileUpdater:
  """
    Applies scored evidence to profile/overlay with guardrails.
    """

  # ============================================================
  # Update rate and guardrails
  # ============================================================
  ALPHA = 0.25  # base learning rate (Stage 4 §)
  PER_UPDATE_CAP = 0.15  # δ : max |p_new - p_old| per update
  PER_SESSION_DRIFT_CAP = 0.35  # Δ : max cumulative |Δ| per session per trait

  # Categorical update: require at least this weight OR hard override.
  CATEGORICAL_UPDATE_WEIGHT = 0.6

  # Volatility window: stddev of last K observed values.
  VOLATILITY_WINDOW = 5

  # Audit trail: keep the last N evidence texts per trait.
  SOURCE_SUMMARY_MAX = 5

  # ============================================================
  # Public entry point
  # ============================================================

  def update(
    self,
    profile: InteractionProfile,
    session_overlay: SessionOverlay,
    weighted: WeightedEvidence,
  ) -> str:
    """
        Apply `weighted` to the appropriate store.

        Returns a human-readable outcome string for logging / the demo.
        """
    evidence = weighted.evidence

    # --- Step 4.1: Route by final stability class ---
    if weighted.final_stability == StabilityClass.IGNORE:
      return f"IGNORED (gate_pass={weighted.gate_pass}, w={weighted.weight})"

    if weighted.final_stability == StabilityClass.SESSION:
      # --- Step 4.2: Session branch ---
      return self._apply_session(session_overlay, weighted)

    # --- STABLE branch ---
    # STRICT context lookup: we must not fall back to "general" here, or
    # we'd mutate the general trait when the evidence came from a
    # specific context. Retrieval (Stage 5) is where fallback is correct.
    current = profile.traits.get(evidence.trait, {}).get(evidence.context)

    if current is None:
      # --- Step 4.3: New trait creation at the specific context ---
      # Optionally seed from general if a general entry exists and is
      # type-compatible with the evidence — this preserves prior
      # learning when branching into a new context for the first time.
      general = profile.traits.get(evidence.trait, {}).get("general")
      if general is not None and type(general.value) is type(evidence.value):
        # Create the context-specific entry seeded from general, then
        # apply the new evidence on top via the normal update path.
        profile.set_trait(
          evidence.trait,
          evidence.context,
          TraitState(
            value=general.value,
            confidence=general.confidence,
            evidence_count=0,  # count only context-specific
            volatility=0.0,
            last_updated=evidence.timestamp,
            source_summary=[f"seeded from {evidence.trait}[general]"],
            recent_values=[general.value],
            session_drift=0.0,
          ),
        )
        current = profile.traits[evidence.trait][evidence.context]
        # Fall through to the update path below.
      else:
        return self._create_trait(profile, weighted)

    # --- Existing trait ---
    if isinstance(current.value, (int, float)) and isinstance(evidence.value,
                                                              (int, float)):
      # --- Step 4.4: Scalar smoothed update ---
      return self._update_scalar(profile, current, weighted)
    else:
      # --- Step 4.5: Categorical update ---
      return self._update_categorical(profile, current, weighted)

  # ============================================================
  # Step 4.2 — Session overlay write
  # ============================================================

  def _apply_session(
    self,
    overlay: SessionOverlay,
    weighted: WeightedEvidence,
  ) -> str:
    evidence = weighted.evidence
    # Key by (trait, context) so session state is context-aware too.
    key = f"{evidence.trait}::{evidence.context}"
    overlay.state[key] = {
      "trait": evidence.trait,
      "context": evidence.context,
      "value": evidence.value,
      "confidence": evidence.confidence,
      "weight": weighted.weight,
      "updated_at": evidence.timestamp.isoformat(),
    }
    overlay.last_updated = evidence.timestamp
    return f"SESSION overlay written for {key} (w={weighted.weight})"

  # ============================================================
  # Step 4.3 — New trait creation
  # ============================================================

  def _create_trait(
    self,
    profile: InteractionProfile,
    weighted: WeightedEvidence,
  ) -> str:
    evidence = weighted.evidence
    new_state = TraitState(
      value=evidence.value,
      confidence=evidence.confidence,
      evidence_count=1,
      volatility=0.0,  # single sample → zero by definition
      last_updated=evidence.timestamp,
      source_summary=[evidence.evidence_text[:120]],
      recent_values=[evidence.value],
      session_drift=0.0,
    )
    profile.set_trait(evidence.trait, evidence.context, new_state)
    profile.last_updated = evidence.timestamp
    return f"CREATED trait {evidence.trait}[{evidence.context}] = {evidence.value}"

  # ============================================================
  # Step 4.4 — Scalar smoothed update (with guardrails)
  # ============================================================

  def _update_scalar(
    self,
    profile: InteractionProfile,
    current: TraitState,
    weighted: WeightedEvidence,
  ) -> str:
    evidence = weighted.evidence
    old_value = float(current.value)
    new_raw_target = float(evidence.value)

    if weighted.is_hard_override:
      # Hard override bypasses smoothing AND per-update cap, but we
      # still cap by the per-session drift budget.
      proposed = new_raw_target
    else:
      # Smoothed update.
      alpha_w = self.ALPHA * weighted.weight
      proposed = (1.0 - alpha_w) * old_value + alpha_w * new_raw_target

    # --- Guardrail A: Per-update cap (δ) ---
    delta = proposed - old_value
    if abs(delta) > self.PER_UPDATE_CAP and not weighted.is_hard_override:
      proposed = old_value + (
        self.PER_UPDATE_CAP if delta > 0 else -self.PER_UPDATE_CAP)
      delta = proposed - old_value

    # --- Guardrail B: Per-session drift cap (Δ) ---
    remaining_budget = self.PER_SESSION_DRIFT_CAP - current.session_drift
    if abs(delta) > remaining_budget:
      if remaining_budget <= 0.0:
        return (
          f"DEFERRED scalar update for {evidence.trait}[{evidence.context}]: "
          f"per-session drift cap exhausted ({current.session_drift:.3f})")
      proposed = old_value + (
        remaining_budget if delta > 0 else -remaining_budget)
      delta = proposed - old_value

    # Apply the update.
    current.value = round(proposed, 4)
    current.session_drift = round(current.session_drift + abs(delta), 4)

    # --- Step 4.7: Metadata ---
    self._append_recent_value(current, evidence.value)
    self._update_volatility(current, value_range=1.0)
    current.confidence = round(
      min(1.0, (current.confidence + evidence.confidence) / 2), 4)
    current.evidence_count += 1
    current.last_updated = evidence.timestamp
    current.source_summary.append(evidence.evidence_text[:120])
    current.source_summary = current.source_summary[-self.SOURCE_SUMMARY_MAX:]
    profile.last_updated = evidence.timestamp

    return (
      f"UPDATED scalar {evidence.trait}[{evidence.context}]: "
      f"{old_value:.3f} → {current.value:.3f} "
      f"(Δ={delta:+.3f}, w={weighted.weight:.3f}, drift={current.session_drift:.3f})"
    )

  # ============================================================
  # Step 4.5 — Categorical update
  # ============================================================

  def _update_categorical(
    self,
    profile: InteractionProfile,
    current: TraitState,
    weighted: WeightedEvidence,
  ) -> str:
    evidence = weighted.evidence
    old_value = current.value

    allow_overwrite = (
      weighted.is_hard_override
      or weighted.weight >= self.CATEGORICAL_UPDATE_WEIGHT)

    if not allow_overwrite:
      # Even if we don't overwrite, we record the observation.
      self._append_recent_value(current, evidence.value)
      current.evidence_count += 1
      current.last_updated = evidence.timestamp
      self._update_volatility(current, value_range=None)
      return (
        f"HELD categorical {evidence.trait}[{evidence.context}]={old_value} "
        f"(weight {weighted.weight:.3f} below threshold {self.CATEGORICAL_UPDATE_WEIGHT})"
      )

    current.value = evidence.value
    self._append_recent_value(current, evidence.value)
    self._update_volatility(current, value_range=None)
    current.confidence = round(
      min(1.0, (current.confidence + evidence.confidence) / 2), 4)
    current.evidence_count += 1
    current.last_updated = evidence.timestamp
    current.source_summary.append(evidence.evidence_text[:120])
    current.source_summary = current.source_summary[-self.SOURCE_SUMMARY_MAX:]
    # Count categorical flips against the drift budget (1 flip = 1.0 unit
    # of drift, scaled to a fraction of the cap).
    if old_value != evidence.value:
      current.session_drift = round(
        current.session_drift + self.PER_UPDATE_CAP, 4)
    profile.last_updated = evidence.timestamp

    return (
      f"UPDATED categorical {evidence.trait}[{evidence.context}]: "
      f"{old_value} → {current.value} (w={weighted.weight:.3f})")

  # ============================================================
  # Step 4.6 — Volatility as real variance
  # ============================================================

  def _append_recent_value(self, current: TraitState, value: Any) -> None:
    current.recent_values.append(value)
    if len(current.recent_values) > self.VOLATILITY_WINDOW:
      current.recent_values = current.recent_values[-self.VOLATILITY_WINDOW:]

  def _update_volatility(
      self, current: TraitState, value_range: float | None) -> None:
    """
        Scalar: volatility = stddev(recent_values) / value_range, clipped to [0,1].
        Categorical: volatility = fraction of recent values differing from mode.
        """
    values = current.recent_values
    if len(values) < 2:
      current.volatility = 0.0
      return

    if all(isinstance(v, (int, float))
           for v in values) and value_range is not None:
      sd = stdev([float(v) for v in values])
      current.volatility = round(min(1.0, sd / value_range), 4)
      return

    # Categorical: fraction that disagree with mode.
    mode_value = max(set(values), key=values.count)
    disagree = sum(1 for v in values if v != mode_value)
    current.volatility = round(disagree / len(values), 4)
