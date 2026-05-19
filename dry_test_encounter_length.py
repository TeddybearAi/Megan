"""Dry test for encounter-length feature — runs without Ollama.

Two parts:
  Part A — Prompt inspection: what does the system prompt LOOK LIKE at each
           encounter level? We assemble it the same way generate() does,
           then show the encounter block that gets appended.
  Part B — Flow simulation: walk through a 20-turn conversation, plus idle
           gap, plus page reload — show the bucket at each step.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.adapters.ollama_adapter import (
  OllamaAdapter,
  _encounter_context_block,
)
from aism.data_models import ResponsePolicy
from aism.session_context import SessionContext


# ============================================================
# PART A — Prompt inspection
# ============================================================


def show_prompt_for_bucket(label: str, ctx: SessionContext) -> None:
  """Build a system prompt as generate() would, and show the encounter tail."""
  adapter = OllamaAdapter(model="qwen2.5:32b")
  policy = ResponsePolicy(mode="casual", mode_confidence=0.0)
  rendered = adapter.render_policy(policy)
  system_prompt = rendered["system_prompt"]

  # This is exactly what generate() does:
  encounter_block = _encounter_context_block(ctx.encounter_length())
  if encounter_block:
    full_prompt = system_prompt + "\n\n" + encounter_block
  else:
    full_prompt = system_prompt

  bucket = ctx.encounter_length()
  print()
  print("=" * 70)
  print(f"  {label}  (bucket: {bucket!r})")
  print(f"  turn_index={ctx.turn_index}  user_words={ctx.user_word_count}  "
        f"elapsed_min={ctx.elapsed_minutes():.1f}")
  print("=" * 70)
  print()
  print(f"Total system prompt length: {len(full_prompt)} chars")
  print(f"Encounter block appended:   {len(encounter_block)} chars")
  print()
  print("--- THE APPENDED ENCOUNTER BLOCK (what's new) ---")
  print(encounter_block)
  print()


def part_a():
  print("\n" + "#" * 70)
  print("#  PART A — System prompt at each encounter level")
  print("#" * 70)

  # FRESH — turn 0, no history
  ctx = SessionContext()
  show_prompt_for_bucket("FRESH — first message of the day", ctx)

  # BRIEF — 2 turns in
  ctx = SessionContext()
  ctx.turn_index = 2
  ctx.user_word_count = 12
  show_prompt_for_bucket("BRIEF — 2-turn 'hey, how are you' exchange", ctx)

  # LIGHT — 4 turns in
  ctx = SessionContext()
  ctx.turn_index = 4
  ctx.user_word_count = 60
  show_prompt_for_bucket("LIGHT — 4-turn small chat", ctx)

  # REAL — 10 turns in
  ctx = SessionContext()
  ctx.turn_index = 10
  ctx.user_word_count = 250
  show_prompt_for_bucket("REAL — 10-turn proper conversation", ctx)

  # SUBSTANTIAL — 20 turns or many words
  ctx = SessionContext()
  ctx.turn_index = 20
  ctx.user_word_count = 1500
  show_prompt_for_bucket("SUBSTANTIAL — 20-turn deep chat", ctx)


# ============================================================
# PART B — Flow simulation
# ============================================================


def part_b():
  print("\n" + "#" * 70)
  print("#  PART B — Flow simulation over a 20-turn conversation")
  print("#" * 70)
  print()
  print(f"{'turn':<6}{'words+=':<12}{'elapsed':<12}{'bucket':<14}{'event'}")
  print("-" * 70)

  ctx = SessionContext()

  # Simulate page load
  ctx.reset_for_new_session()
  print(f"{'-':<6}{'-':<12}{'-':<12}{ctx.encounter_length():<14}"
        "← page loaded (POST /api/session/new)")

  # Simulate 20 turns. Each turn: assistant generates → note_turn → turn_index
  # increments via main.py for the NEXT turn.
  user_texts = [
    "hey",
    "how are you",
    "tell me about your day, i've got a moment",
    "no way, really?",
    "i feel like that's a stretch but ok",
    "let me think about that",
    "so anyway, work was rough",
    "the manager said something i didn't expect",
    "do you think i should push back?",
    "yeah you're right",
    "ok new topic — what are you reading?",
    "i love that one too",
    "well shit, i had a different take",
    "fair point",
    "ok i'm tired",
    "yeah let's keep going actually",
    "one more thing",
    "have you ever felt like that?",
    "good answer mate",
    "alright, time to go",
  ]
  assistant_texts = [
    "Hi there Teddy. What's on your mind?",
    "I'm doing alright. Yourself?",
    "Five hundred years and still cataloguing the small joys, mate. You?",
    "Trust me, the Tudors got worse.",
    "Hm. Maybe. Try me.",
    "Take your time.",
    "Oof. What flavour of rough?",
    "Yeah, that lands wrong. Walk me through it.",
    "Depends on the relationship. What's your read?",
    "Don't let me decide for you though.",
    "Currently re-reading a Wodehouse. Lazy comfort food.",
    "Knew there was a reason I liked you.",
    "Lay it on me.",
    "I yield, my dear.",
    "Alright. Want to keep going or call it?",
    "Right, off we go then.",
    "Shoot.",
    "Plenty. The trick is not letting it set the tone of the day.",
    "Cheers, Teddy.",
    "Catch you later, my girl.",
  ]

  for i in range(20):
    # Megan would generate his reply now — encounter_length is consulted here
    bucket_before = ctx.encounter_length()
    user_text = user_texts[i]
    assistant_text = assistant_texts[i]

    # Print state at the moment of generation
    print(f"{i+1:<6}+{len(user_text.split()):<11}"
          f"{ctx.elapsed_minutes():.1f}min{'':<5}"
          f"{bucket_before:<14}"
          f"user: {user_text[:40]!r}")

    # Now do what app.py does post-turn:
    ctx.note_turn(user_text, assistant_text)
    ctx.turn_index = i + 1  # main.py sets this for the next turn

    # Show bucket transitions
    bucket_after = ctx.encounter_length()
    if bucket_after != bucket_before:
      print(f"{'':6}{'':12}{'':12}{'':<14}"
            f"^^^ bucket transition: {bucket_before} → {bucket_after}")

  print()
  print(f"After 20 turns: turn_index={ctx.turn_index}, "
        f"user_words={ctx.user_word_count}, "
        f"bucket={ctx.encounter_length()}")

  # Simulate idle gap
  print()
  print("--- simulating 31-minute idle gap (would trigger backend reset) ---")
  ctx.last_activity_time = (
    datetime.now(timezone.utc) - timedelta(minutes=31))
  gap_min = (datetime.now(timezone.utc) -
             ctx.last_activity_time).total_seconds() / 60.0
  print(f"gap since last activity: {gap_min:.1f} minutes")
  print(f"_maybe_reset_for_idle_gap() in app.py would fire reset because "
        f"{gap_min:.1f} > 30")

  ctx.reset_for_new_session()
  print(f"after reset: turn_index={ctx.turn_index}, "
        f"user_words={ctx.user_word_count}, "
        f"bucket={ctx.encounter_length()}")


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
  part_a()
  part_b()
  print()
  print("=" * 70)
  print("Dry test complete. No Ollama calls were made.")
  print("=" * 70)
