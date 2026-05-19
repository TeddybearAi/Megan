"""Tests for encounter-length tracking (added 2026-05-12).

Covers:
- SessionContext bucket math (turn count, elapsed time, word volume)
- note_turn() word accumulation
- reset_for_new_session() clears everything correctly
- Encounter block injection into the system prompt via the adapter
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.adapters.ollama_adapter import (
  ENCOUNTER_CONTEXT_BRIEF,
  ENCOUNTER_CONTEXT_FRESH,
  ENCOUNTER_CONTEXT_LIGHT,
  ENCOUNTER_CONTEXT_REAL,
  ENCOUNTER_CONTEXT_SUBSTANTIAL,
  _encounter_context_block,
)
from aism.session_context import SessionContext


# ============================================================
# Bucket math — turn-count axis
# ============================================================


def test_fresh_when_no_turns_yet():
  ctx = SessionContext()
  assert ctx.encounter_length() == "fresh"


def test_brief_at_one_or_two_turns():
  ctx = SessionContext()
  ctx.turn_index = 1
  assert ctx.encounter_length() == "brief"
  ctx.turn_index = 2
  assert ctx.encounter_length() == "brief"


def test_light_at_three_to_five_turns():
  ctx = SessionContext()
  for n in (3, 4, 5):
    ctx.turn_index = n
    assert ctx.encounter_length() == "light", f"turn {n} should be light"


def test_real_at_six_to_fifteen_turns():
  ctx = SessionContext()
  for n in (6, 10, 15):
    ctx.turn_index = n
    assert ctx.encounter_length() == "real", f"turn {n} should be real"


def test_substantial_beyond_fifteen_turns():
  ctx = SessionContext()
  ctx.turn_index = 16
  assert ctx.encounter_length() == "substantial"
  ctx.turn_index = 50
  assert ctx.encounter_length() == "substantial"


# ============================================================
# Bucket math — wall-clock axis (OR-joined with turn count)
# ============================================================


def test_elapsed_time_can_push_to_real_even_with_few_turns():
  ctx = SessionContext()
  ctx.turn_index = 2  # would be "brief" by turn count alone
  # backdate the start so >5 minutes have elapsed
  ctx.session_start_time = (
    datetime.now(timezone.utc) - timedelta(minutes=10))
  assert ctx.encounter_length() == "real"


def test_long_idle_pushes_to_substantial():
  ctx = SessionContext()
  ctx.turn_index = 3  # would be "light" by turn count alone
  ctx.session_start_time = (
    datetime.now(timezone.utc) - timedelta(minutes=45))
  assert ctx.encounter_length() == "substantial"


# ============================================================
# Bucket math — word-volume axis
# ============================================================


def test_long_user_messages_can_push_to_substantial():
  ctx = SessionContext()
  ctx.turn_index = 4  # would be "light" by turn count
  ctx.user_word_count = 1500  # past _SUBSTANTIAL_USER_WORDS threshold
  assert ctx.encounter_length() == "substantial"


# ============================================================
# note_turn() accumulation
# ============================================================


def test_note_turn_accumulates_word_counts():
  ctx = SessionContext()
  ctx.note_turn("hello there friend", "hi yourself")
  assert ctx.user_word_count == 3
  assert ctx.assistant_word_count == 2

  ctx.note_turn("how are you", "fine thanks")
  assert ctx.user_word_count == 6
  assert ctx.assistant_word_count == 4


def test_note_turn_handles_empty_strings_gracefully():
  ctx = SessionContext()
  ctx.note_turn("", "")
  assert ctx.user_word_count == 0
  assert ctx.assistant_word_count == 0
  ctx.note_turn("ok", "")
  assert ctx.user_word_count == 1
  assert ctx.assistant_word_count == 0


def test_note_turn_updates_last_activity():
  ctx = SessionContext()
  before = ctx.last_activity_time
  ctx.note_turn("hi", "hello")
  assert ctx.last_activity_time >= before


# ============================================================
# reset_for_new_session()
# ============================================================


def test_reset_clears_all_counters():
  ctx = SessionContext()
  ctx.turn_index = 12
  ctx.user_word_count = 500
  ctx.assistant_word_count = 800
  ctx.previous_assistant_reply = "something"

  ctx.reset_for_new_session()

  assert ctx.turn_index == 0
  assert ctx.user_word_count == 0
  assert ctx.assistant_word_count == 0
  assert ctx.previous_assistant_reply == ""
  assert ctx.encounter_length() == "fresh"


def test_reset_updates_session_start_time():
  ctx = SessionContext()
  ctx.session_start_time = datetime.now(timezone.utc) - timedelta(hours=2)
  old_start = ctx.session_start_time
  ctx.reset_for_new_session()
  assert ctx.session_start_time > old_start


# ============================================================
# Encounter context block injection (prompt-level)
# ============================================================


def test_each_bucket_has_a_distinct_block():
  buckets = ["fresh", "brief", "light", "real", "substantial"]
  blocks = {b: _encounter_context_block(b) for b in buckets}
  # every bucket should yield non-empty guidance
  assert all(blocks.values())
  # every block should be distinct (no accidental aliasing)
  assert len(set(blocks.values())) == 5


def test_unknown_bucket_returns_empty_string():
  assert _encounter_context_block("nonsense") == ""
  assert _encounter_context_block("") == ""


def test_fresh_block_warns_against_today_phrasing():
  # The whole point: FRESH should explicitly forbid the "great chatting
  # today" pattern that prompted this feature.
  block = ENCOUNTER_CONTEXT_FRESH
  assert "today" in block.lower()
  assert "fresh" in block.lower()


def test_brief_block_lists_forbidden_farewells():
  # BRIEF should explicitly call out the failure mode from Teddy's screenshot.
  block = ENCOUNTER_CONTEXT_BRIEF
  assert "great chatting" in block.lower()


def test_substantial_block_permits_warmer_farewells():
  # By contrast SUBSTANTIAL should explicitly ALLOW the warmer shape.
  block = ENCOUNTER_CONTEXT_SUBSTANTIAL
  assert "warm" in block.lower() or "warranted" in block.lower()


def test_light_and_real_blocks_calibrate_warmth():
  # The middle buckets should mention warmth-scaling concepts.
  assert ENCOUNTER_CONTEXT_LIGHT  # non-empty
  assert ENCOUNTER_CONTEXT_REAL  # non-empty
  # LIGHT should be more cautious than REAL
  assert "small" in ENCOUNTER_CONTEXT_LIGHT.lower() or \
         "premature" in ENCOUNTER_CONTEXT_LIGHT.lower()


# ============================================================
# Integration: SessionContext.encounter_length() drives block selection
# ============================================================


def test_encounter_length_drives_correct_block_at_each_stage():
  """The flow this whole feature exists for: a session progresses,
  encounter_length() rolls up the buckets, and the right block is
  selected at each stage."""
  ctx = SessionContext()
  expectations = [
    (0, "fresh", ENCOUNTER_CONTEXT_FRESH),
    (1, "brief", ENCOUNTER_CONTEXT_BRIEF),
    (4, "light", ENCOUNTER_CONTEXT_LIGHT),
    (10, "real", ENCOUNTER_CONTEXT_REAL),
    (20, "substantial", ENCOUNTER_CONTEXT_SUBSTANTIAL),
  ]
  for turn_index, expected_bucket, expected_block in expectations:
    ctx.turn_index = turn_index
    bucket = ctx.encounter_length()
    block = _encounter_context_block(bucket)
    assert bucket == expected_bucket, (
      f"turn {turn_index}: expected {expected_bucket}, got {bucket}")
    assert block == expected_block
