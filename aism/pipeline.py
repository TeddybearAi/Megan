"""
AISM — Pipeline Orchestrator
============================
Glues the six stages together and drives the closed feedback loop. The
public surface of AISM for external callers.

BluePrint v2.1 references:
    Section 3 — six-stage pipeline
    Section 4 — four storage structures + SessionContext integration contract

Per-turn flow (process_turn):
    Stage 1  : evaluate eligibility
    Stage 2  : extract style evidence (if eligible)
    Stage 6' : detect feedback signals on the same turn
    Stage 3  : score each evidence item (gates + log-linear weight)
    Stage 4  : apply scored evidence to profile / session overlay
    Stage 5  : build response policy from current state
    Generate : hand policy to GenerationAdapter (out of AISM)

Note on Stage 6's placement:
    In the BluePrint narrative, Stage 6 detects feedback on the NEXT user
    turn after an assistant reply. In practice this is just "on every user
    turn, check for feedback patterns." We run Stage 6 before Stage 3 so
    feedback evidence is scored alongside fresh extraction evidence in the
    same cycle — this is what closes the loop within one turn.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .data_models import (
  ConversationTurn,
  EligibilityDecision,
  InteractionProfile,
  ResponsePolicy,
  SessionOverlay,
  StabilityClass,
  StyleEvidence,
  WeightedEvidence,
)
from .generation_adapter import GenerationAdapter, MockGenerationAdapter
from .session_context import SessionContext
from .stage1_eligibility import EligibilityFilter
from .stage2_extraction import StyleEvidenceExtractor
from .stage3_scoring import EvidenceScorer
from .stage4_updater import ProfileUpdater
from .stage5_retrieval import PolicyRetriever
from .stage6_feedback import FeedbackProcessor
from .storage import LocalJSONStore

# ---- Pass 2 optional components (forward-referenced in constructor) ----
# We import lazily inside process_turn to avoid import-time cost for Pass 1-only users.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .stage2b_layer_b_extractor import LayerBExtractor
  from .stage2c_pattern_aggregator import PatternAggregator
  from .monitors.lexical_mimicry import LexicalMimicryMonitor
  from .monitors.idiolect_drift import IdiolectDriftMonitor
  from .monitors.sycophancy import SycophancyMonitor

# ============================================================
# TurnResult — what process_turn returns
# ============================================================


class TurnResult:
  """Container for everything produced during one turn. Used by the demo
    and tests to verify pipeline behaviour."""

  def __init__(
    self,
    eligibility: EligibilityDecision,
    extraction_evidence: List[StyleEvidence],
    feedback_evidence: List[StyleEvidence],
    weighted_evidence: List[WeightedEvidence],
    update_outcomes: List[str],
    policy: ResponsePolicy,
    rendered_policy: Dict[str, Any],
    generated_reply: Optional[str] = None,
    # --- Pass 2 fields (default empty so Pass 1 callers stay compatible) ---
    layer_b_evidence: Optional[List[StyleEvidence]] = None,
    aggregate_evidence: Optional[List[StyleEvidence]] = None,
    monitor_metrics: Optional[Dict[str, Any]] = None,
  ) -> None:
    self.eligibility = eligibility
    self.extraction_evidence = extraction_evidence
    self.feedback_evidence = feedback_evidence
    self.weighted_evidence = weighted_evidence
    self.update_outcomes = update_outcomes
    self.policy = policy
    self.rendered_policy = rendered_policy
    self.generated_reply = generated_reply
    self.layer_b_evidence = layer_b_evidence or []
    self.aggregate_evidence = aggregate_evidence or []
    self.monitor_metrics = monitor_metrics or {}


# ============================================================
# Orchestrator
# ============================================================


class AdaptiveInteractionStylePipeline:
  """Top-level AISM pipeline. Instantiate per user+session."""

  # Session overlay TTL: BluePrint §3 Stage 4.
  SESSION_IDLE_TTL = timedelta(minutes=30)

  def __init__(
    self,
    user_id: str,
    session_id: str,
    storage: LocalJSONStore,
    adapter: Optional[GenerationAdapter] = None,
    # --- Pass 2 optional components. All default None → Pass 1 behaviour. ---
    layer_b_extractor: Optional["LayerBExtractor"] = None,
    pattern_aggregator: Optional["PatternAggregator"] = None,
    lexical_mimicry_monitor: Optional["LexicalMimicryMonitor"] = None,
    idiolect_drift_monitor: Optional["IdiolectDriftMonitor"] = None,
    sycophancy_monitor: Optional["SycophancyMonitor"] = None,
    pattern_aggregator_every_n_turns: int = 5,
  ) -> None:
    self.user_id = user_id
    self.session_id = session_id
    self.storage = storage
    self.adapter: GenerationAdapter = adapter or MockGenerationAdapter()

    # ---- Pass 2 optional wiring ----
    self.layer_b_extractor = layer_b_extractor
    self.pattern_aggregator = pattern_aggregator
    self.lexical_mimicry_monitor = lexical_mimicry_monitor
    self.idiolect_drift_monitor = idiolect_drift_monitor
    self.sycophancy_monitor = sycophancy_monitor
    self._aggregator_every_n = pattern_aggregator_every_n_turns
    self._turns_processed = 0

    # ---- Load or create the four storage structures ----
    self.profile: InteractionProfile = (
      self.storage.load_profile(user_id)
      or InteractionProfile(user_id=user_id))
    self.session_overlay: SessionOverlay = (
      self.storage.load_overlay(user_id, session_id)
      or SessionOverlay(session_id=session_id))

    # ---- Stage instances ----
    self.eligibility_filter = EligibilityFilter()
    self.extractor = StyleEvidenceExtractor()
    self.scorer = EvidenceScorer()
    self.scorer.evidence_log = self.storage.load_evidence_log(user_id)
    self.updater = ProfileUpdater()
    self.retriever = PolicyRetriever()
    self.feedback_processor = FeedbackProcessor()

  # ============================================================
  # Public entry point — process one user turn end to end
  # ============================================================

  def process_turn(
    self,
    turn: ConversationTurn,
    session_context: SessionContext,
  ) -> TurnResult:
    # --- Preamble: persist raw turn, apply overlay TTL ---
    self.storage.append_turn(self.user_id, turn)
    self._expire_stale_overlay(now=turn.timestamp)
    self._turns_processed += 1

    # --- Monitor hook: record user text before generation (Pass 2 opt-in) ---
    if self.lexical_mimicry_monitor is not None:
      self.lexical_mimicry_monitor.record_user_turn(turn.text)
    if self.sycophancy_monitor is not None:
      self.sycophancy_monitor.record_user_turn(turn.text)

    # --- Stage 1: eligibility ---
    decision = self.eligibility_filter.evaluate_turn(turn)

    # If ineligible, we still build a policy (from existing profile state)
    # and generate, but we don't extract evidence or update the profile.
    extraction_evidence: List[StyleEvidence] = []
    feedback_evidence: List[StyleEvidence] = []
    layer_b_evidence: List[StyleEvidence] = []
    aggregate_evidence: List[StyleEvidence] = []

    if decision.is_eligible:
      # --- Stage 2 Layer A: lexicon extraction ---
      extraction_evidence = self.extractor.extract(turn, session_context)

      # --- Stage 2 Layer B: embedding-assisted extraction (opt-in) ---
      if self.layer_b_extractor is not None:
        layer_b_evidence = self.layer_b_extractor.extract(
          turn, session_context, extraction_evidence)

      # --- Stage 6 (same-turn): detect feedback ---
      feedback_evidence = self.feedback_processor.detect_feedback(
        turn, session_context)

      # --- Stage 2C: cross-turn aggregator (opt-in, every N turns) ---
      if (self.pattern_aggregator is not None
          and self._turns_processed % self._aggregator_every_n == 0):
        aggregate_evidence = self.pattern_aggregator.aggregate(
          evidence_log=self.scorer.evidence_log,
          profile=self.profile,
          session_context=session_context,
          now_timestamp=turn.timestamp,
        )

    # Merge all evidence streams + dedup.
    all_evidence = self._dedupe_evidence(
      extraction_evidence + layer_b_evidence + feedback_evidence +
      aggregate_evidence)

    # --- Stage 3: score each piece of evidence ---
    weighted_list: List[WeightedEvidence] = []
    update_outcomes: List[str] = []

    for ev in all_evidence:
      weighted = self.scorer.score(
        evidence=ev,
        eligibility_confidence=decision.eligibility_confidence,
        scope_validity=1.0 if ev.context == "general" else 0.7,
      )
      weighted_list.append(weighted)

      # --- Stage 4: apply ---
      outcome = self.updater.update(
        profile=self.profile,
        session_overlay=self.session_overlay,
        weighted=weighted,
      )
      update_outcomes.append(outcome)

      # Register AFTER scoring so frequency/consistency reflect history,
      # not the current item (see Stage 3 comment).
      self.scorer.register_evidence(ev)

    # --- Stage 5: retrieve/build policy ---
    policy = self.retriever.build_policy(
      profile=self.profile,
      session_overlay=self.session_overlay,
      session_context=session_context,
      current_user_message=turn.text,
    )

    # --- Generation: hand off via adapter ---
    rendered = self.adapter.render_policy(policy)
    retrieved_facts = getattr(session_context, "retrieved_memory", [])
    reply: Optional[str] = None
    if decision.is_eligible:
      reply = self.adapter.generate(
        user_message=turn.text,
        retrieved_facts=retrieved_facts,
        rendered_policy=rendered,
        session_context=session_context,
      )

    # --- Monitor hooks: score the reply (Pass 2 opt-in) ---
    monitor_metrics: Dict[str, Any] = {}
    if reply is not None:
      if self.lexical_mimicry_monitor is not None:
        monitor_metrics["lexical_mimicry"] = (
          self.lexical_mimicry_monitor.score_reply(reply))
      if self.idiolect_drift_monitor is not None:
        monitor_metrics["idiolect_drift"] = (
          self.idiolect_drift_monitor.record_reply(reply))
      if self.sycophancy_monitor is not None:
        monitor_metrics["sycophancy"] = (
          self.sycophancy_monitor.score_reply(reply))

    # --- Persist ---
    self.storage.save_profile(self.profile)
    self.storage.save_overlay(self.user_id, self.session_overlay)
    self.storage.save_evidence_log(self.user_id, self.scorer.evidence_log)

    return TurnResult(
      eligibility=decision,
      extraction_evidence=extraction_evidence,
      feedback_evidence=feedback_evidence,
      weighted_evidence=weighted_list,
      update_outcomes=update_outcomes,
      policy=policy,
      rendered_policy=rendered,
      generated_reply=reply,
      layer_b_evidence=layer_b_evidence,
      aggregate_evidence=aggregate_evidence,
      monitor_metrics=monitor_metrics,
    )

  # ============================================================
  # Session lifecycle
  # ============================================================

  def end_session(self) -> None:
    """Reset per-session drift counters and clear overlay."""
    self.profile.reset_session_drift()
    self.session_overlay = SessionOverlay(session_id=self.session_id)
    self.storage.save_profile(self.profile)
    self.storage.save_overlay(self.user_id, self.session_overlay)

  def _expire_stale_overlay(self, now: datetime) -> None:
    """Clear overlay if it hasn't been touched within SESSION_IDLE_TTL."""
    last = self.session_overlay.last_updated
    if last.tzinfo is None:
      last = last.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
      now = now.replace(tzinfo=timezone.utc)
    if self.session_overlay.state and (now - last) > self.SESSION_IDLE_TTL:
      self.session_overlay = SessionOverlay(session_id=self.session_id)
      self.profile.reset_session_drift()

  @staticmethod
  def _dedupe_evidence(evidence: List[StyleEvidence]) -> List[StyleEvidence]:
    """
        Collapse duplicate evidence within one turn.
        Key: (trait, context, rounded_value). For duplicates, keep the one
        with the highest (source_type_priority, confidence) — correction beats
        preference, higher confidence wins ties.
        """
    source_priority = {
      "correction": 4,
      "explicit_dislike": 3,
      "explicit_preference": 3,
      "positive_feedback": 2,
      "implicit_pattern": 1,
    }

    best: Dict[tuple, StyleEvidence] = {}
    for ev in evidence:
      val_key = (
        round(float(ev.value), 2) if isinstance(ev.value,
                                                (int, float)) else ev.value)
      key = (ev.trait, ev.context, val_key)
      incumbent = best.get(key)
      if incumbent is None:
        best[key] = ev
        continue
      # Higher priority wins; tiebreak on confidence.
      new_score = (
        source_priority.get(ev.source_type.value, 0),
        ev.confidence,
      )
      old_score = (
        source_priority.get(incumbent.source_type.value, 0),
        incumbent.confidence,
      )
      if new_score > old_score:
        best[key] = ev
    return list(best.values())
