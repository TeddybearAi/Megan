"""
AISM — Evaluation: Synthetic Personas
=====================================
SyntheticPersona is a ground-truth style vector plus a set of representative
user turns that would arise from that persona. Tier 1 evaluation compares
the AISM-inferred profile against the ground-truth persona vector.

BluePrint v2.1 reference: Section 7.1 — Tier 1 data strategy.

In a real study we'd generate the turn sets with a capable LLM as a user
simulator (offline developer work, permitted by R3). For Pass 2 we ship a
set of hand-written personas so the evaluation harness can be exercised
without any network or model dependency.

Steps (persona construction):
    Step P.1 : Define ground_truth — the true (trait, context, value) map
    Step P.2 : Define turn_plan — a sequence of {turn_text, topic, task_type}
                plus an optional `expected_evidence` annotation for per-turn
                extraction evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ============================================================
# Persona dataclasses
# ============================================================


@dataclass
class PersonaTurn:
  """One simulated user turn from a persona."""
  text: str
  topic: str = "general"  # becomes SessionContext.current_topic
  task_type: str = "conversation"  # becomes SessionContext.task_type
  # Ground-truth annotation (optional). If provided, the harness can
  # evaluate per-turn extraction precision/recall.
  expected_evidence: List[Dict[str, Any]] = field(default_factory=list)
  # Whether Stage 1 should reject this turn as ineligible.
  expected_eligible: bool = True


@dataclass
class SyntheticPersona:
  """A named user-style persona with ground truth and a scripted turn set."""
  persona_id: str
  description: str
  # ground_truth[(trait, context)] = expected final value after running
  # the full turn_plan through the pipeline.
  ground_truth: Dict[tuple, Any]
  turn_plan: List[PersonaTurn]


# ============================================================
# Sample personas
# ============================================================
#
# We ship three personas that exercise different parts of the pipeline:
#
#   "direct_engineer"    — strongly prefers direct/short in technical context,
#                          moderate in general. Tests context-conditional traits.
#   "warm_reflective"    — prefers warm/long replies, dislikes bullet points.
#                          Tests scalar+categorical update paths.
#   "correction_heavy"   — issues explicit corrections multiple times.
#                          Tests feedback loop + hard override semantics.
# ============================================================

SAMPLE_PERSONAS: List[SyntheticPersona] = [
  SyntheticPersona(
    persona_id="direct_engineer",
    description=(
      "Prefers direct, concise responses. Wants more detail only when "
      "the topic is technical. Dislikes over-enthusiastic tone."),
    ground_truth={
      ("directness", "general"): 0.9,
      ("verbosity", "general"): "short",
      ("verbosity", "technical"): "long",
      ("warmth", "general"): 0.3,
    },
    turn_plan=[
      PersonaTurn(
        text="Please be direct and don't sugarcoat things.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "directness",
          "value": 0.9
        }],
      ),
      PersonaTurn(
        text="Also keep it short and concise.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "verbosity",
          "value": "short"
        }],
      ),
      PersonaTurn(
        text="Drop the enthusiasm — less perky please.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "warmth",
          "value": 0.3
        }],
      ),
      PersonaTurn(
        text="When we talk about code, I want more detailed explanations.",
        topic="technical",
        task_type="question",
        expected_evidence=[{
          "trait": "verbosity",
          "value": "long"
        }],
      ),
      PersonaTurn(
        text="Rewrite this email for my professor: 'Dear Prof...'",
        topic="general",
        task_type="rewrite_request",
        expected_eligible=False,
        expected_evidence=[],
      ),
      PersonaTurn(
        text="Get to the point, just tell me what's wrong.",
        topic="technical",
        task_type="question",
        expected_evidence=[{
          "trait": "directness",
          "value": 0.9
        }],
      ),
    ],
  ),
  SyntheticPersona(
    persona_id="warm_reflective",
    description=(
      "Prefers warm, supportive tone with longer explanations. "
      "Dislikes bullet points, wants prose."),
    ground_truth={
      ("warmth", "general"): 0.8,
      ("verbosity", "general"): "long",
      ("formatting", "general"): "paragraphs",
    },
    turn_plan=[
      PersonaTurn(
        text="Be warmer, more empathy please.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "warmth",
          "value": 0.8
        }],
      ),
      PersonaTurn(
        text="I prefer more detailed explanations, elaborate a bit.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "verbosity",
          "value": "long"
        }],
      ),
      PersonaTurn(
        text="No bullet points — prefer paragraphs.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "formatting",
          "value": "paragraphs"
        }],
      ),
      PersonaTurn(
        text="Be more supportive when I'm talking about personal stuff.",
        topic="emotional",
        task_type="venting",
        expected_evidence=[{
          "trait": "warmth",
          "value": 0.8
        }],
      ),
    ],
  ),
  SyntheticPersona(
    persona_id="correction_heavy",
    description=(
      "Issues frequent explicit corrections about general style. "
      "Tests the feedback loop and hard-override path. Corrections are "
      "in general-context conversational framing even though the "
      "surrounding question is technical."),
    ground_truth={
      ("verbosity", "general"): "short",
      ("directness", "general"): 0.9,
    },
    turn_plan=[
      PersonaTurn(
        text="Tell me about Python generators.",
        topic="technical",
        task_type="question",
        expected_evidence=[],
      ),
      # The user's correction is a general-style complaint, not a
      # technical-topic one. The Orchestrator surfaces this via
      # task_type="conversation" and topic="general".
      PersonaTurn(
        text="That was too long. Be shorter next time.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "verbosity",
          "value": "short"
        }],
      ),
      PersonaTurn(
        text="And get to the point faster, stop hedging.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "directness",
          "value": 0.9
        }],
      ),
      PersonaTurn(
        text="Still too wordy. Shorter please.",
        topic="general",
        task_type="conversation",
        expected_evidence=[{
          "trait": "verbosity",
          "value": "short"
        }],
      ),
    ],
  ),
]


def get_persona(persona_id: str) -> Optional[SyntheticPersona]:
  for p in SAMPLE_PERSONAS:
    if p.persona_id == persona_id:
      return p
  return None
