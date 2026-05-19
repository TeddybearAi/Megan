"""
AISM — Remarkable Memory Extraction (revised, Memory-pass v2)
=============================================================

Classifies a user's utterance as a memory candidate and assigns a storage
decision, importance score, and summary note.

Revisions in this pass (vs. v1):
- Bare ``\\bi'm\\b`` is no longer enough to land an utterance in
  ``profile_fact``. Whisper transcribes literally everything starting with
  "I'm", so the v1 classifier promoted greetings, throwaway feelings, and
  conversational filler to importance-5 stable life facts. The new profile
  patterns require structural follow-ups like "I am a/an/the …",
  "I work at/for/as …", "my job is …".
- New ``transient_state`` category catches "I'm tired", "I'm busy now",
  "I'm fine right now" so they don't pollute long-term memory.
  ``transient_state`` is checked BEFORE ``profile_fact``.
- Emotional patterns broadened slightly to cover the bare "I'm sad / anxious
  / worried" cases that v1 missed (those leaked into profile_fact).
- Imports remain side-effect-free; no LLM calls anywhere.
"""

from __future__ import annotations

import re
from typing import Dict, Any

STYLE_KEYWORDS = {
  "answer", "answers", "reply", "replies", "response", "responses",
  "tone", "direct", "warm", "concise", "shorter", "gentle", "calm",
}

# --- Pattern groups --------------------------------------------------------
#
# Order of evaluation in ``infer_memory_type`` is the order these are checked.
# Tighter, more specific categories come first.

# Transient: short-lived states that should NOT enter long-term memory.
# Crucially, this group catches the bare "i'm <feeling>" / "i'm <state>" cases
# that v1 mis-classified as profile_fact via the bare \bi'm\b rule.
TRANSIENT_STATE_PATTERNS = [
  # "i'm tired/busy/fine/sad/etc."  — short transient states
  r"\bi'?m\s+(?:tired|exhausted|sleepy|busy|free|fine|ok|okay|good|"
  r"great|alright|hungry|thirsty|cold|hot|bored|stressed\s+out)\b",
  # explicit transient markers anywhere in the utterance
  r"\bright now\b",
  r"\bat the moment\b",
  r"\bjust (?:now|saying|wanted|want|finished|going)\b",
  # short conversational fillers that often start with "i'm"
  r"\bi'?m\s+just\b",
  r"\bi'?m\s+kind\s+of\b",
  r"\bi'?m\s+sort\s+of\b",
]

# Emotional: feelings with potential staying power. Broader than v1, so the
# bare "I'm anxious / worried / down" cases get classified correctly instead
# of falling into profile_fact.
EMOTIONAL_PATTERNS = [
  r"\bi feel\b",
  r"\bi am feeling\b",
  r"\bi'?m feeling\b",
  r"\bi get (?:anxious|stressed|worried|nervous|down|upset)\b",
  r"\bi'?m\s+(?:anxious|worried|nervous|depressed|heartbroken|"
  r"devastated|grieving|lonely|miserable|scared|afraid)\b",
  r"\b(?:my )?(?:crush|partner|girlfriend|boyfriend|husband|wife|"
  r"ex)\b.*\b(?:has|left|broke up|cheated|gone)\b",
]

# Long-term goals
LONG_TERM_GOAL_PATTERNS = [
  r"\bi want to\b",
  r"\bi hope to\b",
  r"\bi plan to\b",
  r"\bmy goal is\b",
  r"\bi(?:'m| am) trying to\b",
]

# Profile facts: STABLE life facts. Tightened from v1 — bare "I'm X" no longer
# qualifies; we require a structural cue like "i am a/an/the …" or a clear
# job/location/identity verb.
#
# v2.2.5 — expanded substantially to cover identity claims that v1 missed:
# MBTI types ("I'm INTJ"), personality assertions ("my personality is X"),
# language ability ("I speak Mandarin"), demographic facets ("my age is..."),
# and hedge-tolerant variants ("I think my personality is...").
PROFILE_PATTERNS = [
  r"\bi am (?:a|an|the)\s+\w+",        # "I am a software engineer"
  r"\bi'?m (?:a|an|the)\s+\w+",        # "I'm a teacher"
  r"\bi work (?:at|for|as)\b",         # "I work at Google" / "as a lawyer"
  r"\bi study (?:at|in)\b",            # "I study at MIT"
  r"\bi live (?:in|at|on|near)\b",     # "I live in Sydney"
  r"\bmy (?:job|role|profession|career|name) is\b",
  r"\bi was born (?:in|on)\b",
  r"\bi grew up (?:in|on)\b",
  r"\bmy name(?:'?s| is)\s+\w+",       # "my name is Teddy"

  # ---- v2.2.5 additions ----
  # MBTI / personality-type direct claims. Note: input is lowercased by
  # _normalise_text before regex evaluation, so char classes use lowercase.
  r"\bi(?:'?m| am)\s+(?:an?\s+)?[ie][ns][tf][jp]\b",
  # Personality assertion, hedge-tolerant. The "my personality ... is"
  # tolerates inserted hedges like ", I believe," between subject and verb.
  r"\b(?:i (?:think|believe|guess|reckon) )?my personality\b",
  r"\b(?:i (?:think|believe|guess|reckon) )?my\s+(?:mbti|enneagram)\b",
  # Language ability — identity-stable, not transient
  r"\bi (?:speak|am fluent in|can speak)\s+\w+",
  r"\bi(?:'?m| am) (?:bilingual|trilingual|multilingual)\b",
  # Demographic facets stated as facts
  r"\bmy (?:age|height|weight|gender|sexuality|"
  r"nationality|culture|ethnicity|religion|background|"
  r"birthday|birth ?date|hometown|home town) (?:is|are|was|were)\b",
  # Explicit "I am N years old" age claims
  r"\bi(?:'?m| am)\s+\d{1,3}\s+(?:years?\s+old|y\.?o\.?)\b",
  # Identity declarations with copular "is"
  r"\bmy (?:identity|orientation|pronouns?) (?:is|are)\b",
]

PREFERENCE_PATTERNS = [
  r"\bi like\b",
  r"\bi love\b",
  r"\bi prefer\b",
  r"\bi dislike\b",
  r"\bi hate\b",
  r"\bi enjoy\b",
  # v2.2.5 — explicit favourites declared as facts:
  # "my favorite colour is red", "my favourite game is Cyberpunk".
  r"\bmy favou?rite\s+\w+(?:\s+\w+)?\s+(?:is|are)\s+\w+",
]

TEMPORARY_PATTERNS = [
  r"\btomorrow\b",
  r"\bthis weekend\b",
  r"\btoday\b",
  r"\btonight\b",
  r"\bnext week\b",
  r"\bmaybe\b",
  r"\bmight\b",
]


def _contains_pattern(text: str, patterns: list[str]) -> bool:
  return any(re.search(pattern, text) for pattern in patterns)


def _contains_any_keyword(text: str, keywords: set[str]) -> bool:
  tokens = set(text.split())
  return len(tokens.intersection(keywords)) > 0


def _normalise_text(text: str) -> str:
  return " ".join(text.strip().lower().split())


def _looks_like_voice_control_only(text: str) -> bool:
  """True if after rough cleaning there's almost nothing left.

  Catches utterances like "Megan over" or "make an over making over" that
  Whisper produced and ``strip_voice_controls`` should already have removed —
  this is a defence-in-depth check so memory.py never promotes a near-empty
  string.
  """
  return len(text.strip()) < 6


# v2.2.5 — You-directed utterance detection.
#
# Bug 4 (2026-05-15): utterances like "what's your favorite cup, I want to
# remember something about you" were being stored as Teddy profile facts /
# long_term_goals. Then on later turns, retrieval surfaced them and Megan
# treated his own previous questions as facts about Teddy.
#
# Heuristic: if the utterance is primarily ABOUT Megan (you-questions,
# your-attribute requests, "I want to remember about you" framing), suppress
# storage entirely. Be conservative — preserve emotional disclosures like
# "I love you" which are about Teddy's feelings, not Megan's attributes.
YOU_DIRECTED_PATTERNS = [
  # Direct questions about Megan's attributes
  r"\bwhat(?:'?s| is| are) your\b",       # "what's your favorite X"
  r"\bdo you (?:have|like|prefer|know|enjoy|love)\b",
  r"\bcan you (?:tell|describe|share)\b",
  r"\btell me (?:about )?your\b",         # "tell me your favorite"
  r"\byour (?:favou?rite|opinion|view|thought|preference|age|name)\b",
  r"\bhow (?:old|tall) are you\b",
  # "I want to know/remember about you" — goal is about Megan
  r"\bi want to (?:know|remember|learn|find out)\s+(?:more\s+)?(?:about )?you\b",
  r"\b(?:remember|learn|tell me)\s+something\s+about\s+you\b",
]

# Whitelist: phrases that are technically "you-directed" but are emotional
# disclosures FROM Teddy, which we DO want to keep as emotional content.
YOU_DIRECTED_WHITELIST = [
  r"\bi (?:love|like|miss|adore|hate|trust|need) you\b",
  r"\byou (?:make me|hurt me|help me|saved me|matter)\b",
]


def _is_you_directed(text: str) -> bool:
  """True iff the utterance is primarily asking about / referring to Megan's
  attributes, rather than disclosing something about Teddy.

  Returns False for emotional disclosures (whitelist) even when those
  contain "you", since those carry information about Teddy's relationship
  to Megan that's worth keeping as emotional_pattern.
  """
  # Emotional disclosure trumps you-direction
  for pat in YOU_DIRECTED_WHITELIST:
    if re.search(pat, text, re.IGNORECASE):
      return False
  # Otherwise check you-directed cues
  for pat in YOU_DIRECTED_PATTERNS:
    if re.search(pat, text, re.IGNORECASE):
      return True
  return False


# High-signal emotional events. These are checked BEFORE transient state
# so that "my crush has a girlfriend right now" doesn't get demoted to
# transient on the strength of the trailing "right now" marker — the
# underlying event is a real emotional moment, not a fleeting state.
EMOTIONAL_EVENT_PATTERNS = [
  r"\b(?:my )?(?:crush|partner|girlfriend|boyfriend|husband|wife|ex)\b.*"
  r"\b(?:has|left|broke up|cheated|gone|passed away|died)\b",
  r"\b(?:lost|losing) my (?:job|partner|parent|mum|mom|dad|father|"
  r"mother|brother|sister|friend)\b",
  r"\b(?:diagnosed with|cancer|miscarriage|miscarried)\b",
  r"\b(?:depressed|grieving|heartbroken|devastated)\b",
  # Medical journeys / fertility — long-running life context, not transient.
  r"\bivf\b",
  r"\bembryos?\b",
  r"\bfertility\b",
  r"\bpregnan(?:t|cy)\b",
  r"\bchemo(?:therapy)?\b",
  r"\bsurgery\b",
]

# Aspirations / application events — long-term-goal-shaped, not transient.
# Caught before TRANSIENT so "I'm applying for being a police officer right
# now" doesn't get demoted on the trailing "right now".
ASPIRATION_PATTERNS = [
  r"\b(?:applied|applying)\s+(?:for|to)\b",
  r"\b(?:got|landed)\s+(?:the\s+)?(?:job|offer|role)\b",
  r"\bstarting\s+(?:a\s+)?new\s+(?:job|role|business)\b",
]


def infer_memory_type(utterance: str) -> str:
  text = _normalise_text(utterance)

  # Defence-in-depth: empty / voice-control-only utterance never classifies
  # to anything that would be stored.
  if _looks_like_voice_control_only(text):
    return "daily_detail"

  # v2.2.5 Bug 4 — You-directed utterances. "What's your favorite cup",
  # "I want to remember something about you", "do you like X" — these are
  # NOT facts about Teddy. They're questions/requests directed at Megan.
  # Without this check, v1 was storing them as long_term_goal or
  # profile_fact and surfacing them on later turns as if they were Teddy
  # disclosures. Emotional disclosures like "I love you" are whitelisted
  # inside _is_you_directed so they still flow to emotional_pattern.
  if _is_you_directed(text):
    return "daily_detail"

  # 1. Strong emotional events first — these always win. We don't want a
  #    crush-news utterance to be demoted to transient because the user
  #    happened to say "right now".
  if _contains_pattern(text, EMOTIONAL_EVENT_PATTERNS):
    return "emotional_pattern"

  # 2. Aspiration / application events — these are long-term goals even
  #    when the surrounding sentence has transient markers ("I'm tired
  #    right now and I just applied for the police academy").
  if _contains_pattern(text, ASPIRATION_PATTERNS):
    return "long_term_goal"

  # 3. Transient state — short-lived feelings that mention "right now",
  #    "i'm tired", etc. Pre-empts profile_fact's "i'm a/the" patterns.
  if _contains_pattern(text, TRANSIENT_STATE_PATTERNS):
    return "transient_state"

  # 4. General emotional patterns ("i feel", "i get anxious").
  if _contains_pattern(text, EMOTIONAL_PATTERNS):
    return "emotional_pattern"

  # 5. Long-term goals.
  if _contains_pattern(text, LONG_TERM_GOAL_PATTERNS):
    return "long_term_goal"

  # 6. Stable life facts.
  if _contains_pattern(text, PROFILE_PATTERNS):
    return "profile_fact"

  # 7. Preferences (style-flavoured ones bumped to style_preference).
  if _contains_pattern(text, PREFERENCE_PATTERNS):
    return "style_preference" if _contains_any_keyword(
      text, STYLE_KEYWORDS) else "preference"

  # 8. Temporary plans.
  if _contains_pattern(text, TEMPORARY_PATTERNS):
    return "temporary_plan"

  return "daily_detail"


def infer_store_label(utterance: str, memory_type: str) -> str:
  text = _normalise_text(utterance)
  if memory_type in {
      "profile_fact",
      "style_preference",
      "long_term_goal",
      "emotional_pattern",
  }:
    return "yes"
  if memory_type == "temporary_plan":
    return "maybe"
  if memory_type == "transient_state":
    # Transient states never reach long-term memory and rarely deserve to
    # sit in candidates either. Skip persistence entirely.
    return "no"
  if memory_type == "daily_detail":
    return "no"
  if memory_type == "preference":
    return "maybe" if _contains_pattern(text, TEMPORARY_PATTERNS) else "yes"
  return "no"


def infer_importance(utterance: str, memory_type: str) -> int:
  if memory_type in {"profile_fact", "long_term_goal"}:
    return 4  # was 5 in v1; reserved 5 for explicitly user-confirmed facts
  if memory_type == "style_preference":
    return 4
  if memory_type == "emotional_pattern":
    return 4
  if memory_type == "preference":
    return 3
  if memory_type == "transient_state":
    return 1
  if memory_type == "temporary_plan":
    important = _contains_pattern(
      text := _normalise_text(utterance), [r"\bimportant\b", r"\breally\b"])
    return 3 if important else 2
  return 1


def infer_notes(utterance: str, memory_type: str, store: str) -> str:
  if store == "no":
    return "Not promoted to long-term memory."
  note_map = {
    "profile_fact": "Stable background fact about the user.",
    "style_preference": "Likely communication or response-style preference.",
    "temporary_plan": "Possible short-lived plan or context-dependent detail.",
    "preference": "Personal like or dislike.",
    "long_term_goal": "Potential long-term goal.",
    "emotional_pattern": "Emotion-related pattern.",
    "transient_state": "Short-lived feeling/state — not stored.",
  }
  return note_map.get(memory_type, "General extracted memory candidate.")


def extract_memory_candidate(user_id: str, utterance: str) -> Dict[str, Any]:
  memory_type = infer_memory_type(utterance)
  store = infer_store_label(utterance, memory_type)
  importance = infer_importance(utterance, memory_type)
  notes = infer_notes(utterance, memory_type, store)

  return {
    "user_id": user_id,
    "utterance": utterance,
    "store": store,
    "memory_type": memory_type,
    "importance": importance,
    "notes": notes,
  }
