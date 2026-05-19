"""
AISM — Generation Adapter Abstraction
=====================================
Defines the boundary between AISM and the language model used for reply
generation. AISM hands out a structured ResponsePolicy; the adapter
translates that into whatever the target model expects.

BluePrint v2.1 reference: Section 5 — GenerationAdapter abstraction.

This is THE ONLY place in the system that interacts with an LLM. No AISM
stage calls a generative model. Pass 1 ships the abstract Protocol and a
MockGenerationAdapter useful for tests / the demo.

Concrete adapters (Pass 2):
    OllamaAdapter      — for Ollama-served local models
    LlamaCppAdapter    — for direct llama.cpp binding
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .data_models import ResponsePolicy
from .session_context import SessionContext

# ============================================================
# Protocol
# ============================================================


class GenerationAdapter(Protocol):
  """
    The contract AISM depends on. Concrete adapters must NOT import anything
    AISM-internal except ResponsePolicy and SessionContext.
    """

  def render_policy(self, policy: ResponsePolicy) -> Dict[str, Any]:
    """
        Convert a ResponsePolicy into whatever the target LLM consumes
        (system prompt string, control tokens, sampling parameters, ...).
        """
    ...

  def generate(
    self,
    user_message: str,
    retrieved_facts: List[Dict[str, Any]],
    rendered_policy: Dict[str, Any],
    session_context: SessionContext,
  ) -> str:
    """Produce the assistant reply text."""
    ...


# ============================================================
# MockGenerationAdapter — for tests and the demo
# ============================================================


class MockGenerationAdapter:
  """
    A deterministic adapter that returns a canned reply echoing the rendered
    policy. Useful for verifying end-to-end pipeline flow without requiring
    a real LLM.
    """

  def render_policy(self, policy: ResponsePolicy) -> Dict[str, Any]:
    # --- Build a compact system-prompt-like string from the policy ---
    parts: List[str] = []
    if policy.tone:
      parts.append(f"Tone={policy.tone}")
    if policy.verbosity:
      parts.append(f"Verbosity={policy.verbosity}")
    if policy.formatting:
      parts.append(f"Formatting={policy.formatting}")
    if policy.emotional_style:
      parts.append(f"EmotionalStyle={policy.emotional_style}")
    if policy.proactiveness:
      parts.append(f"Proactiveness={policy.proactiveness}")
    if policy.reasoning_presentation:
      parts.append(f"Reasoning={policy.reasoning_presentation}")

    system_instruction = (
      "Respond according to this interaction policy: " +
      "; ".join(parts) if parts else "Respond naturally.")

    return {
      "system_instruction": system_instruction,
      "provenance": policy.metadata,
    }

  def generate(
    self,
    user_message: str,
    retrieved_facts: List[Dict[str, Any]],
    rendered_policy: Dict[str, Any],
    session_context: SessionContext,
  ) -> str:
    # Echo the policy so tests can verify it was wired correctly.
    fact_preview = (
      f" | LTM facts: {len(retrieved_facts)}" if retrieved_facts else "")
    return (
      f"[MOCK REPLY | {rendered_policy['system_instruction']} | "
      f"topic={session_context.current_topic}{fact_preview}]")
