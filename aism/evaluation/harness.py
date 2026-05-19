"""
AISM — Evaluation Harness
=========================
Runs a SyntheticPersona's turn_plan through the pipeline and computes the
headline metrics from BluePrint v2.1 §9.

BluePrint v2.1 reference: Section 8 — Phase 1 offline benchmark.

Steps:
    Step H.1 : Snapshot the profile after every turn (for PSI & latency).
    Step H.2 : Score per-turn extraction against expected_evidence.
    Step H.3 : Score eligibility decisions against expected_eligible.
    Step H.4 : After the run, compute Trait Extraction F1/MAE,
               Profile Stability Index, and Ground-Truth Match Rate.
    Step H.5 : Aggregate and return a PersonaEvalResult.

This is the Tier 1 evaluation runner. For Tier 2 (real annotated conversations)
the persona's `turn_plan` is replaced by the annotated corpus; the metric code
is the same.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from ..data_models import ConversationTurn, InteractionProfile, TurnRole
from ..pipeline import AdaptiveInteractionStylePipeline
from ..session_context import SessionContext
from ..storage import LocalJSONStore
from .metrics import (
  EligibilityCounts,
  ExtractionCounts,
  adaptation_latency,
  ground_truth_match_rate,
  profile_stability_index,
  score_eligibility,
  score_turn_extraction,
)
from .personas import PersonaTurn, SyntheticPersona

# ============================================================
# Result container
# ============================================================


@dataclass
class PersonaEvalResult:
  persona_id: str
  extraction: ExtractionCounts
  eligibility: EligibilityCounts
  psi: float
  ground_truth: Dict[str, Any]
  per_turn_details: List[Dict[str, Any]] = field(default_factory=list)
  final_profile_snapshot: Dict[Tuple[str, str],
                               Any] = field(default_factory=dict)

  def pretty_summary(self) -> str:
    lines = [
      f"Persona: {self.persona_id}",
      f"  Extraction   P={self.extraction.precision:.3f}  "
      f"R={self.extraction.recall:.3f}  F1={self.extraction.f1:.3f}  "
      f"(TP={self.extraction.true_positives} "
      f"FP={self.extraction.false_positives} "
      f"FN={self.extraction.false_negatives})",
      f"  Eligibility  acc={self.eligibility.accuracy:.3f}  "
      f"contamination_rate={self.eligibility.contamination_rate:.3f}",
      f"  Profile Stability Index = {self.psi:.3f}",
      f"  Ground-truth match rate = "
      f"{self.ground_truth['match_rate']:.3f} "
      f"({self.ground_truth['n_matched']}/{self.ground_truth['n_expected']})",
    ]
    for trait_key, detail in self.ground_truth["per_trait"].items():
      mark = "✓" if detail["match"] else "✗"
      lines.append(
        f"    {mark} {trait_key}  expected={detail['expected']!r} "
        f"actual={detail['actual']!r}")
    return "\n".join(lines)


# ============================================================
# Harness
# ============================================================


class PersonaHarness:
  """Evaluates a single persona against an AISM pipeline instance."""

  def run(self, persona: SyntheticPersona) -> PersonaEvalResult:
    # --- Set up a fresh pipeline in a temp storage dir ---
    tmpdir = tempfile.mkdtemp(prefix=f"aism_eval_{persona.persona_id}_")
    storage = LocalJSONStore(tmpdir)
    pipeline = AdaptiveInteractionStylePipeline(
      user_id=f"eval_{persona.persona_id}",
      session_id=f"eval_session_{uuid.uuid4().hex[:6]}",
      storage=storage,
    )

    extraction_counts = ExtractionCounts()
    eligibility_counts = EligibilityCounts()
    snapshots: List[Dict[Tuple[str, str], Any]] = []
    per_turn_details: List[Dict[str, Any]] = []

    start = datetime.now(timezone.utc)

    for idx, persona_turn in enumerate(persona.turn_plan, start=1):
      ctx = SessionContext(
        current_topic=persona_turn.topic,
        task_type=persona_turn.task_type,
        turn_index=idx,
      )
      turn = ConversationTurn(
        turn_id=f"eval-t{idx}-{uuid.uuid4().hex[:6]}",
        role=TurnRole.USER,
        text=persona_turn.text,
        timestamp=start + timedelta(seconds=idx * 30),
      )
      result = pipeline.process_turn(turn, ctx)

      # --- Step H.2: extraction scoring ---
      predicted_all = (
        list(result.extraction_evidence) + list(result.feedback_evidence))
      turn_counts = score_turn_extraction(
        predicted_all, persona_turn.expected_evidence)
      extraction_counts.true_positives += turn_counts.true_positives
      extraction_counts.false_positives += turn_counts.false_positives
      extraction_counts.false_negatives += turn_counts.false_negatives

      # --- Step H.3: eligibility scoring ---
      score_eligibility(
        predicted_eligible=result.eligibility.is_eligible,
        expected_eligible=persona_turn.expected_eligible,
        counts=eligibility_counts,
      )

      # --- Step H.1: snapshot profile ---
      snap = _snapshot_profile(pipeline.profile)
      snapshots.append(snap)

      per_turn_details.append(
        {
          "turn_index":
          idx,
          "text":
          persona_turn.text[:60] +
          ("..." if len(persona_turn.text) > 60 else ""),
          "predicted_eligible":
          result.eligibility.is_eligible,
          "expected_eligible":
          persona_turn.expected_eligible,
          "extraction": {
            "tp": turn_counts.true_positives,
            "fp": turn_counts.false_positives,
            "fn": turn_counts.false_negatives,
          },
          "snapshot":
          snap,
        })

    # --- Step H.4: aggregate metrics ---
    psi = profile_stability_index(snapshots)
    gt = ground_truth_match_rate(pipeline.profile, persona.ground_truth)

    return PersonaEvalResult(
      persona_id=persona.persona_id,
      extraction=extraction_counts,
      eligibility=eligibility_counts,
      psi=psi,
      ground_truth=gt,
      per_turn_details=per_turn_details,
      final_profile_snapshot=snapshots[-1] if snapshots else {},
    )


def _snapshot_profile(
  profile: InteractionProfile, ) -> Dict[Tuple[str, str], Any]:
  """Convert profile state into a {(trait, context): value} dict."""
  snap: Dict[Tuple[str, str], Any] = {}
  for trait_name, by_context in profile.traits.items():
    for context, state in by_context.items():
      snap[(trait_name, context)] = state.value
  return snap


# ============================================================
# Convenience runner
# ============================================================


def evaluate_all(personas) -> List[PersonaEvalResult]:
  """Run the harness across a list of personas and return all results."""
  harness = PersonaHarness()
  return [harness.run(p) for p in personas]
