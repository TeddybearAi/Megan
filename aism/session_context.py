"""
AISM — SessionContext (Integration Contract)
============================================
The shared object by which AISM receives conditioning information from the
Orchestrator. This is the ONLY channel through which context derived from
outside AISM (e.g. LTM retrieval, task classification) enters the pipeline.

BluePrint v2.1 reference: Section 4 — Integration Contract with LTM.

Design rule: AISM never reads the Long-Term Memory store directly. The
Orchestrator runs LTM retrieval, derives topic/task classification, and
populates this object. AISM uses it to select context-conditional traits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
  from .data_models import ConversationTurn  # noqa: F401

# ----------------------------------------------------------------------------
# Encounter-length thresholds
# ----------------------------------------------------------------------------
# These five buckets quantise "how long has this conversation been going" into
# a categorical signal the LLM can act on without us injecting raw numbers
# into the prompt. The thresholds are deliberately permissive — they're "OR"
# joined so a long conversation by ANY measure (turns, time, words) qualifies
# at the next-up bucket. Tunable; treat the numbers as policy, not gospel.
#
# Reset semantics: counters reset (a) when the page reloads (frontend signals
# session/new), and (b) after a 30-minute idle gap (matches the existing
# SESSION_IDLE_TTL in the pipeline). Long-term memory is NOT affected by
# these resets — only the in-session sense of "how long have we been talking
# right now" resets.

_BRIEF_TURNS = 2
_LIGHT_TURNS = 5
_REAL_TURNS = 15
# anything > _REAL_TURNS is SUBSTANTIAL

_LIGHT_MINUTES = 5.0
_REAL_MINUTES = 30.0
# anything > _REAL_MINUTES is SUBSTANTIAL

_SUBSTANTIAL_USER_WORDS = 1000
# user-word volume alone can push into SUBSTANTIAL even with few turns


@dataclass
class SessionContext:
  """
    Per-turn context supplied by the Orchestrator.

    current_topic : {"general", "technical", "emotional", "admin", "casual"}
                    Used to select context-conditional trait values.
    task_type     : {"conversation", "question", "request", "venting",
                     "rewrite_request"}  (rewrite_request should typically be
                     filtered upstream by Stage 1, but we track it here so
                     the retriever can detect edge cases.)
    turn_index    : 0-based turn counter within the session.
    previous_assistant_reply : The text of Megan's last reply, if any.
                    Used by the turn-intent classifier to recognise short
                    affirmations (e.g. "yeah, sure, go ahead") as acceptance
                    of a prior offer-to-help (e.g. "Want my take?"). Without
                    this, the affirmation falls through to AMBIGUOUS and
                    Megan loops on the off-ramp instead of delivering.
                    Orchestrator writes this after each turn.

    Encounter-length fields (added 2026-05-12):
    user_word_count       : cumulative word count of user turns this session.
    assistant_word_count  : cumulative word count of Megan's turns this session.
    last_activity_time    : timestamp of the most recent turn (user OR
                            assistant). Used by the orchestrator to detect
                            the 30-minute idle gap and reset.
    """
  current_topic: str = "general"
  task_type: str = "conversation"
  turn_index: int = 0
  session_start_time: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc))
  retrieved_memory: List[Dict[str, Any]] = field(default_factory=list)
  previous_assistant_reply: str = ""

  # ---- v2.2.1: Person Bank integration ----
  # The formatted Person Bank context block for this turn, ready to drop
  # into the system prompt. Empty string when Person Bank doesn't engage.
  # Written by app.py's _process_text_and_speak; read by the adapter's
  # _build_system_prompt.
  person_context_block: str = ""

  # When True, the adapter suppresses LTM fact prepending for this turn.
  # v2.2.4: True ONLY on actual introduction turns (was previously
  # True on any sticky/match turn, which broke all LTM access).
  skip_ltm_retrieval: bool = False

  # ---- v2.2.6: Identity block ----
  # A pre-rendered block of high-confidence user identity facts (age,
  # personality, languages, favourites, family). Injected directly into
  # the system prompt every turn so meta-queries like "what's my age?"
  # and "tell me about myself" don't depend on cosine retrieval. Built
  # from LTM in main.process_user_message via
  # storage.get_identity_block(). Empty string for fresh users or when
  # LTM has no qualifying facts.
  identity_block: str = ""

  # ---- v2.2.9: Explicit-remember signal for LTM boost ----
  # True when the user said "remember X" / "this is important" / etc.
  # AND no Person Bank record was engaged on this turn (because if a
  # person IS engaged, Person Bank's extractor handles it).
  # main.process_user_message reads this flag and forces the resulting
  # memory candidate to store=yes / importance=5 / frequency=2 so the
  # fact reaches the identity block on the next turn.
  explicit_remember: bool = False

  # ---- v2.2.4: Short-term / working memory ----
  # Recent ConversationTurn entries from THIS session (filtered by
  # session_start_time so prior sessions don't leak in). The adapter
  # splices these into the Ollama messages array between the system
  # prompt and the current user message, giving Megan working memory
  # of what was just said. Populated by main.process_user_message
  # before pipeline.process_turn. Empty list = no short-term context
  # (fresh session, or first turn after idle-gap reset).
  recent_turns: List[Any] = field(default_factory=list)

  # ---- Encounter-length tracking (added 2026-05-12) ----
  user_word_count: int = 0
  assistant_word_count: int = 0
  last_activity_time: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc))

  # ============================================================
  # Encounter-length API
  # ============================================================

  def note_turn(self, user_text: str, assistant_text: str) -> None:
    """Accumulate per-turn statistics. Call once per completed exchange.

    Splits on whitespace — good enough for English; we're not doing
    linguistics here, just bucketing for prompt conditioning.
    """
    self.user_word_count += len(user_text.split()) if user_text else 0
    self.assistant_word_count += (
      len(assistant_text.split()) if assistant_text else 0)
    self.last_activity_time = datetime.now(timezone.utc)

  def reset_for_new_session(self) -> None:
    """Wipe all in-session counters. Long-term memory is untouched.

    Called by the orchestrator when (a) the frontend signals page reload,
    or (b) the orchestrator detects a 30+ minute idle gap. After this,
    encounter_length() returns "fresh" again until the next note_turn().
    """
    now = datetime.now(timezone.utc)
    self.turn_index = 0
    self.user_word_count = 0
    self.assistant_word_count = 0
    self.session_start_time = now
    self.last_activity_time = now
    self.previous_assistant_reply = ""
    # v2.2.4: short-term memory is per-session — drop it on reset.
    self.recent_turns = []
    self.person_context_block = ""
    self.skip_ltm_retrieval = False
    # v2.2.6: identity_block reloads from LTM on the next turn anyway,
    # but clear it for tidiness.
    self.identity_block = ""
    # v2.2.9: explicit_remember is per-turn; reset for cleanliness.
    self.explicit_remember = False

  def elapsed_minutes(self) -> float:
    """Wall-clock minutes since session_start_time."""
    now = datetime.now(timezone.utc)
    start = self.session_start_time
    if start.tzinfo is None:
      start = start.replace(tzinfo=timezone.utc)
    return (now - start).total_seconds() / 60.0

  def encounter_length(self) -> str:
    """Return categorical bucket: fresh / brief / light / real / substantial.

    Computed from three signals (turn_index, elapsed_minutes, user_word_count)
    joined with OR — any signal can push the encounter "up" a bucket. The
    LLM prompt is conditioned on this string, NOT on the raw numbers, so
    the bucket boundaries are policy and easily tuned without touching
    the prompt or the adapter.
    """
    turns = self.turn_index
    minutes = self.elapsed_minutes()
    words = self.user_word_count

    if turns == 0:
      return "fresh"

    if (turns > _REAL_TURNS
        or minutes > _REAL_MINUTES
        or words > _SUBSTANTIAL_USER_WORDS):
      return "substantial"

    if turns > _LIGHT_TURNS or minutes > _LIGHT_MINUTES:
      return "real"

    if turns > _BRIEF_TURNS:
      return "light"

    return "brief"
