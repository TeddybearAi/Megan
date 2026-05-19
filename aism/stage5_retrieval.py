"""
AISM — STAGE 5: Policy Retrieval for Generation
===============================================
Compiles a compact ResponsePolicy from the stable profile, session overlay,
and any explicit override in the current turn. Implements the conflict-
resolution hierarchy and Precision@K trait selection.

BluePrint v2.1 reference: Section 3 — Stage 5.

Steps:
    Step 5.1 : Detect explicit overrides in the current user message
    Step 5.2 : Walk the priority list per policy field:
               (1) explicit override   → highest
               (2) hard boundary        → absolute
               (3) session overlay      → current-session state
               (4) context-specific stable trait  → profile.get_trait(ctx)
               (5) general stable trait → profile.get_trait("general")
    Step 5.3 : Task-relevance filtering (Precision@K — omit irrelevant fields)
    Step 5.4 : Render policy strings from resolved trait values
    Step 5.5 : Attach provenance metadata for auditability

Constraint: No LLM calls. All logic is deterministic rule-walking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .data_models import InteractionProfile, ResponsePolicy, SessionOverlay
from .session_context import SessionContext

# ============================================================
# Step 5.1 — Explicit override patterns (current-turn overrides)
# ============================================================
#
# These detect "right now, do X" instructions that should override stored
# profile values for this turn only. They mirror the Stage 2 extractor but
# are evaluated against the CURRENT message by the retriever.
# ============================================================

EXPLICIT_OVERRIDE_RULES: List[Dict[str, Any]] = [
  {
    "trait":
    "verbosity",
    "pattern":
    re.compile(
      r"\b(be brief|keep it short|tldr|just briefly|in one sentence)\b",
      re.IGNORECASE,
    ),
    "value":
    "short",
  },
  {
    "trait":
    "verbosity",
    "pattern":
    re.compile(
      r"\b(in detail|elaborate|thoroughly|deep dive)\b",
      re.IGNORECASE,
    ),
    "value":
    "long",
  },
  {
    "trait":
    "directness",
    "pattern":
    re.compile(
      r"\b(be direct|just tell me|straight answer|no preamble)\b",
      re.IGNORECASE,
    ),
    "value":
    0.9,
  },
  {
    "trait":
    "formatting",
    "pattern":
    re.compile(
      r"\b(in bullet points?|as a list|bullet(s|ed))\b",
      re.IGNORECASE,
    ),
    "value":
    "bullets",
  },
  {
    "trait":
    "formatting",
    "pattern":
    re.compile(
      r"\b(in prose|in paragraphs?|no bullets?)\b",
      re.IGNORECASE,
    ),
    "value":
    "paragraphs",
  },
]

# ============================================================
# Step 5.3 — Task-relevance map (which traits matter for which task)
# ============================================================
#
# For each task_type (from SessionContext), list the traits that are
# actually relevant. Irrelevant traits are omitted from the final policy —
# this is the Precision@K mechanism.
# ============================================================

TASK_RELEVANT_TRAITS: Dict[str, List[str]] = {
  "conversation": [
    "directness", "verbosity", "warmth", "humour", "formatting",
    "proactiveness"
  ],
  "question": ["directness", "verbosity", "formatting"],
  "request": ["directness", "verbosity", "formatting", "proactiveness"],
  "venting": ["warmth", "directness", "verbosity"],
  "rewrite_request": [],  # handled upstream by Stage 1; no policy emitted
}


class PolicyRetriever:
  """
    Conflict-aware policy compiler. Returns a compact ResponsePolicy, with
    provenance metadata attached so we can audit which source won each field.
    """

  # ============================================================
  # Public entry point
  # ============================================================

  def build_policy(
    self,
    profile: InteractionProfile,
    session_overlay: SessionOverlay,
    session_context: SessionContext,
    current_user_message: str,
  ) -> ResponsePolicy:
    # --- Step 5.1: Detect explicit overrides in the current turn ---
    overrides = self._detect_explicit_overrides(current_user_message)

    # --- Step 5.3 (first half): Which traits are relevant for this task? ---
    relevant_traits = TASK_RELEVANT_TRAITS.get(
      session_context.task_type,
      TASK_RELEVANT_TRAITS["conversation"],
    )

    # --- Step 5.2: Resolve each relevant trait via priority walk ---
    resolved: Dict[str, Dict[str, Any]] = {}
    for trait_name in relevant_traits:
      resolution = self._resolve_trait(
        trait_name=trait_name,
        profile=profile,
        session_overlay=session_overlay,
        session_context=session_context,
        overrides=overrides,
      )
      if resolution is not None:
        resolved[trait_name] = resolution

    # --- Step 5.4: Render policy strings from resolved values ---
    policy = ResponsePolicy(
      tone=self._render_tone(resolved),
      verbosity=self._render_verbosity(resolved),
      formatting=self._render_formatting(resolved),
      emotional_style=self._render_emotional_style(resolved),
      proactiveness=self._render_proactiveness(resolved),
      reasoning_presentation=self._render_reasoning(resolved, session_context),
    )

    # --- Step 5.4b: Per-turn intent classification (added pass 3) ---
    # Determines whether this turn is venting / seeking_help / casual /
    # decision / ambiguous. The adapter uses this to gate advice-emitting
    # parts of the system prompt. Defaults to 'ambiguous' (listen-first)
    # when current_user_message is empty.
    #
    # ``previous_assistant_reply`` is threaded through so the classifier
    # can recognise short affirmations ("yeah, sure, go ahead") that
    # accept a prior offer-to-help ("Want my take?") and route them to
    # SEEKING_HELP — without this, they fall through to AMBIGUOUS and
    # Megan loops on the off-ramp.
    if current_user_message:
      from .turn_intent import classify_turn_intent
      mode, mode_conf = classify_turn_intent(
        current_user_message,
        previous_assistant_reply=getattr(
          session_context, "previous_assistant_reply", "") or "",
      )
      policy.mode = mode
      policy.mode_confidence = mode_conf

    # --- Step 5.5: Provenance metadata ---
    policy.metadata = {
      "resolved_traits": {
        k: {
          "value": v["value"],
          "source": v["source"]
        }
        for k, v in resolved.items()
      },
      "task_type": session_context.task_type,
      "current_topic": session_context.current_topic,
      "turn_index": session_context.turn_index,
      "intent_mode": policy.mode,
      "intent_confidence": policy.mode_confidence,
    }
    return policy

  # ============================================================
  # Step 5.1 — Override detection
  # ============================================================

  def _detect_explicit_overrides(self, text: str) -> Dict[str, Any]:
    """
        Returns a dict {trait_name: override_value} for each override detected
        in the current message. First match per trait wins.
        """
    found: Dict[str, Any] = {}
    for rule in EXPLICIT_OVERRIDE_RULES:
      trait = rule["trait"]
      if trait in found:
        continue
      if rule["pattern"].search(text):
        found[trait] = rule["value"]
    return found

  # ============================================================
  # Step 5.2 — Conflict resolution priority walk
  # ============================================================

  def _resolve_trait(
    self,
    trait_name: str,
    profile: InteractionProfile,
    session_overlay: SessionOverlay,
    session_context: SessionContext,
    overrides: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    # --- Priority 1: Explicit override in current message ---
    if trait_name in overrides:
      return {
        "value": overrides[trait_name],
        "source": "explicit_override",
      }

    # --- Priority 2: Hard boundaries ---
    if trait_name in profile.boundaries:
      return {
        "value": profile.boundaries[trait_name],
        "source": "boundary",
      }

    # --- Priority 3: Session overlay (context-keyed) ---
    overlay_key_ctx = f"{trait_name}::{session_context.current_topic}"
    overlay_key_gen = f"{trait_name}::general"
    if overlay_key_ctx in session_overlay.state:
      return {
        "value": session_overlay.state[overlay_key_ctx]["value"],
        "source": f"session_overlay[{session_context.current_topic}]",
      }
    if overlay_key_gen in session_overlay.state:
      return {
        "value": session_overlay.state[overlay_key_gen]["value"],
        "source": "session_overlay[general]",
      }

    # --- Priority 4: Context-specific stable trait ---
    # get_trait automatically falls back to "general" if no context-specific
    # entry exists — see InteractionProfile.get_trait.
    ctx_state = profile.traits.get(trait_name,
                                   {}).get(session_context.current_topic)
    if ctx_state is not None:
      return {
        "value": ctx_state.value,
        "source": f"profile[{session_context.current_topic}]",
      }

    # --- Priority 5: General stable trait ---
    general_state = profile.traits.get(trait_name, {}).get("general")
    if general_state is not None:
      return {
        "value": general_state.value,
        "source": "profile[general]",
      }

    # No value available — this trait is omitted from the policy.
    return None

  # ============================================================
  # Step 5.4 — Render helpers (trait value → policy string)
  # ============================================================

  def _render_tone(self, resolved: Dict[str, Dict[str, Any]]) -> Optional[str]:
    directness = resolved.get("directness", {}).get("value")
    warmth = resolved.get("warmth", {}).get("value")

    if directness is None and warmth is None:
      return None

    parts: List[str] = []
    if isinstance(directness, (int, float)):
      if directness >= 0.75:
        parts.append("direct")
      elif directness <= 0.35:
        parts.append("gentle")
      else:
        parts.append("balanced")

    if isinstance(warmth, (int, float)):
      if warmth >= 0.7:
        parts.append("warm")
      elif warmth <= 0.35:
        parts.append("low-hype")
      else:
        parts.append("measured warmth")

    return ", ".join(parts) if parts else None

  def _render_verbosity(self, resolved: Dict[str, Dict[str,
                                                       Any]]) -> Optional[str]:
    v = resolved.get("verbosity", {}).get("value")
    if v is None:
      return None
    return str(v)

  def _render_formatting(self,
                         resolved: Dict[str, Dict[str, Any]]) -> Optional[str]:
    f = resolved.get("formatting", {}).get("value")
    if f is None:
      return None
    if f == "paragraphs":
      return "prefer paragraphs, minimal bullets"
    if f == "bullets":
      return "bullet points for multi-item content"
    return str(f)

  def _render_emotional_style(
      self, resolved: Dict[str, Dict[str, Any]]) -> Optional[str]:
    warmth = resolved.get("warmth", {}).get("value")
    humour = resolved.get("humour", {}).get("value")

    if warmth is None and humour is None:
      return None

    parts: List[str] = []
    if isinstance(warmth, (int, float)):
      if warmth >= 0.7:
        parts.append("warm but controlled")
      elif warmth <= 0.35:
        parts.append("calm, restrained")
      else:
        parts.append("neutral")
    if isinstance(humour, (int, float)):
      # Threshold lowered from 0.7 to 0.55 so that "light humour welcome"
      # reaches the prompt at organically-learned values in the 0.55-0.7
      # band, not just clearly-high values. Without this, a trait value
      # in the natural "mostly yes to humour" range produces no humour
      # cue at all — silent zone — and the LLM falls back to a safer,
      # drier register than the persona actually allows.
      if humour >= 0.55:
        parts.append("light humour welcome")
      elif humour <= 0.3:
        parts.append("no jokes")
    return "; ".join(parts) if parts else None

  def _render_proactiveness(
      self, resolved: Dict[str, Dict[str, Any]]) -> Optional[str]:
    p = resolved.get("proactiveness", {}).get("value")
    if p is None:
      return None
    if isinstance(p, (int, float)):
      if p >= 0.7:
        return "offer next steps and related suggestions"
      if p <= 0.3:
        return "answer only what was asked; no suggestions"
      return "offer one next step if clearly relevant"
    return str(p)

  def _render_reasoning(
    self,
    resolved: Dict[str, Dict[str, Any]],
    session_context: SessionContext,
  ) -> Optional[str]:
    # A light task-conditional rendering — no trait drives this directly
    # in Pass 1, but context should.
    if session_context.current_topic == "technical":
      return "show reasoning steps briefly; structured"
    if session_context.current_topic == "emotional":
      return "minimise analytical framing; lead with acknowledgement"
    return None
