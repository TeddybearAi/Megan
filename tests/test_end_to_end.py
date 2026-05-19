"""
AISM — End-to-End Demo
======================
Runs a scripted multi-turn conversation through the full pipeline and prints
the state after each turn. Verifies:

    - Stage 1: eligibility filter accepts normal conversation and rejects
               rewrite / roleplay / email-draft turns
    - Stage 2: lexicon extractor finds trait cues in natural phrasing
    - Stage 3: gated log-linear scorer assigns appropriate weights
    - Stage 4: scalar traits smooth-update; categorical traits threshold-update
    - Stage 5: policy retrieval resolves conflicts via the priority walk
    - Stage 6: feedback loop closes — "too long" becomes a new evidence item
               that immediately updates the verbosity preference

Run:
    python -m tests.test_end_to_end
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

from aism import (
  AdaptiveInteractionStylePipeline,
  ConversationTurn,
  LocalJSONStore,
  SessionContext,
  TurnRole,
)

# ============================================================
# Pretty-printing helpers
# ============================================================


def _banner(title: str) -> None:
  print("\n" + "=" * 78)
  print(title)
  print("=" * 78)


def _print_result(turn_num: int, turn: ConversationTurn, result) -> None:
  print(f"\n--- Turn {turn_num} ---")
  print(f"  USER: {turn.text!r}")
  print(
    f"  Stage 1: eligible={result.eligibility.is_eligible}  "
    f"(score={result.eligibility.contamination_score:.2f}, "
    f"conf={result.eligibility.eligibility_confidence:.2f})")
  if not result.eligibility.is_eligible:
    print(f"           reason: {result.eligibility.reason}")

  if result.extraction_evidence or result.feedback_evidence:
    print(
      f"  Stage 2: {len(result.extraction_evidence)} extraction evidence "
      f"+ {len(result.feedback_evidence)} feedback evidence")
    for ev in result.extraction_evidence + result.feedback_evidence:
      src = "fb" if ev.evidence_id.startswith("fb-") else "ex"
      print(
        f"           [{src}] {ev.trait}={ev.value} "
        f"({ev.source_type.value}, conf={ev.confidence:.2f})")

  if result.weighted_evidence:
    print(f"  Stage 3: weights computed")
    for w in result.weighted_evidence:
      print(
        f"           {w.evidence.trait}={w.evidence.value}  "
        f"w={w.weight:.3f}  class={w.final_stability.value}  "
        f"override={w.is_hard_override}")

  if result.update_outcomes:
    print(f"  Stage 4: profile updates")
    for outcome in result.update_outcomes:
      print(f"           {outcome}")

  print(f"  Stage 5: ResponsePolicy")
  for field in ("tone", "verbosity", "formatting", "emotional_style",
                "proactiveness", "reasoning_presentation"):
    val = getattr(result.policy, field)
    if val is not None:
      source = result.policy.metadata.get("resolved_traits", {})
      # Find any source trait that contributed to this field (best-effort).
      print(f"           {field}: {val}")

  if result.generated_reply:
    # Truncate for readability.
    reply = result.generated_reply
    if len(reply) > 140:
      reply = reply[:137] + "..."
    print(f"  Reply: {reply}")


def _print_profile(pipeline: AdaptiveInteractionStylePipeline) -> None:
  print("\n[ Stable Profile ]")
  if not pipeline.profile.traits:
    print("  (empty)")
    return
  for trait_name, by_context in pipeline.profile.traits.items():
    for context, state in by_context.items():
      print(
        f"  {trait_name}[{context}] = {state.value}  "
        f"(conf={state.confidence:.2f}, n={state.evidence_count}, "
        f"vol={state.volatility:.3f}, drift={state.session_drift:.3f})")
  print("\n[ Session Overlay ]")
  if not pipeline.session_overlay.state:
    print("  (empty)")
  else:
    for key, v in pipeline.session_overlay.state.items():
      print(f"  {key} = {v['value']}  (w={v['weight']:.3f})")


# ============================================================
# Scripted conversation
# ============================================================


def build_script():
  """
    Returns a list of (turn_text, session_context, turn_kind) tuples.

    turn_kind is a label for demo output; it's not used by the pipeline.

    The script is designed to exercise each stage:
      - T1-T3  : normal conversational style preferences (Stage 2, 3 STABLE path)
      - T4     : ineligible rewrite request (Stage 1 rejection)
      - T5     : a technical-context turn to test context-conditional traits
      - T6     : explicit override in the CURRENT message (Stage 5 priority 1)
      - T7     : a feedback turn that closes the loop (Stage 6 → 3 → 4)
      - T8     : verify the loop actually shifted the profile
    """
  ctx_general = SessionContext(
    current_topic="general", task_type="conversation")
  ctx_technical = SessionContext(
    current_topic="technical", task_type="question")

  script = [
    # --- T1: explicit directness preference ---
    (
      "Please be direct and don't sugarcoat things.",
      ctx_general,
      "explicit directness preference",
    ),
    # --- T2: explicit verbosity preference ---
    (
      "Also keep it short and concise. I don't like long-winded answers.",
      ctx_general,
      "explicit short-verbosity preference",
    ),
    # --- T3: reinforce directness (frequency → STABLE) ---
    (
      "Get to the point whenever you can, just tell me.",
      ctx_general,
      "reinforce directness (expect STABLE after frequency bump)",
    ),
    # --- T4: ineligible turn (rewrite task) ---
    (
      "Rewrite this for my professor: 'Dear Prof. Smith, I hope this finds you well...'",
      ctx_general,
      "rewrite task — should be rejected by Stage 1",
    ),
    # --- T5: technical context with different preference ---
    (
      "When we talk about code, I want more detailed explanations actually.",
      ctx_technical,
      "technical-context verbosity preference (long)",
    ),
    # --- T6: explicit override in current message ---
    (
      "Quick question though — in one sentence, what is a closure?",
      ctx_technical,
      "explicit override: brief, this turn only",
    ),
    # --- T7: feedback that closes the loop ---
    (
      "That last answer was too long, be shorter next time.",
      ctx_technical,
      "feedback: should produce CORRECTION evidence and update verbosity",
    ),
    # --- T8: normal turn to see updated policy ---
    (
      "Okay, what about decorators?",
      ctx_technical,
      "should reflect tightened verbosity from T7",
    ),
  ]
  return script


# ============================================================
# Main
# ============================================================


def main() -> None:
  # Use a fresh temp directory so repeated runs are reproducible.
  tmpdir = tempfile.mkdtemp(prefix="aism_demo_")
  try:
    _banner("AISM Pass 1 — End-to-End Demo")
    print(f"Storage: {tmpdir}")

    storage = LocalJSONStore(tmpdir)
    pipeline = AdaptiveInteractionStylePipeline(
      user_id="demo_user",
      session_id="session_001",
      storage=storage,
    )

    # Run the scripted conversation with staggered timestamps so recency
    # features have non-trivial behaviour.
    start = datetime.now(timezone.utc)
    script = build_script()

    for idx, (text, ctx, label) in enumerate(script, start=1):
      _banner(f"TURN {idx}: {label}")
      ctx.turn_index = idx
      turn = ConversationTurn(
        turn_id=f"t{idx}_{uuid.uuid4().hex[:6]}",
        role=TurnRole.USER,
        text=text,
        timestamp=start + timedelta(seconds=idx * 30),
      )
      result = pipeline.process_turn(turn, ctx)
      _print_result(idx, turn, result)
      _print_profile(pipeline)

    # ---- Final summary ----
    _banner("FINAL STATE AFTER 8 TURNS")
    _print_profile(pipeline)

    # ---- Verify persistence round-trip ----
    _banner("PERSISTENCE ROUND-TRIP CHECK")
    reloaded_profile = storage.load_profile("demo_user")
    assert reloaded_profile is not None
    assert reloaded_profile.traits.keys() == pipeline.profile.traits.keys()
    print("✓ Profile round-trip OK — traits match after reload")
    evidence_log = storage.load_evidence_log("demo_user")
    print(f"✓ Evidence log persisted: {len(evidence_log)} items")
    raw_turns = storage.load_turns("demo_user")
    print(f"✓ Raw log persisted: {len(raw_turns)} turns")

  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
  main()
