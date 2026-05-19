"""
AISM — Person Bank
==================
A separate memory store for *people in the user's life* — judges, friends,
coaches, crushes, family. Sits alongside candidate_memory.json and
long_term_memory.json but is keyed on **named entity lookup**, not on
fuzzy semantic similarity over self-statements.

Why this exists
---------------
The existing AISM memory pipeline is built around first-person statements
the user makes about *themselves* ("I like classic music", "I'm feeling
anxious", "I applied for the academy"). Its memory_type vocabulary —
profile_fact, preference, emotional_pattern, long_term_goal — has no slot
for "Vahid is the course judge, age 35, charming, brilliant, his mantra
is 'that's right'". When the user introduces a person and asks Megan to
remember them, the existing extractor classifies the utterance as
daily_detail (storage.no) and drops it on the floor.

The Person Bank fixes this by being explicitly relational: each record
describes a *person other than the user*, with a name, relationship, and
free-form attribute dict. Retrieval is by name (with STT-tolerant
fuzzy match), not by semantic similarity. Injection into the system
prompt is high-authority — when the user mentions a known name, the
bank's facts about that person become ground truth for that turn.

Pipeline hooks
--------------
Two hooks in app.py's turn handler:

  1. BEFORE generation: bank.lookup(user_text)
     -> attach matched PersonRecords to session_context.person_context
     -> adapter injects them as a high-authority system-prompt block

  2. AFTER generation: bank.observe(user_text)
     -> detect introduction patterns in the user's utterance
     -> if found, fire a focused LLM call to extract structured info
     -> upsert into the bank (new person, or attribute update on existing)

The observe step also tracks mention counts on existing people without
firing the LLM call, so the bank knows who's been talked about recently.

Extraction trigger model
------------------------
Hybrid, in three layers (highest-confidence first):

  EXPLICIT       "remember Vahid"            -> always extract
                 "don't forget X"
                 "add to memory: X"

  INTRODUCTION   "his/her/their name is X"   -> always extract
                 "this is X"
                 "X is my [role]"
                 "the [role] is X"
                 "let me tell you about X"

  ATTACHMENT     "he's 35", "her mantra is"  -> attach to most-recently
                 (within 5 turns of intro)      introduced person, no LLM call
                                                (v1: not implemented; intro
                                                always triggers a single
                                                extraction with all attrs
                                                in the same utterance)

Storage shape
-------------
Each PersonRecord is serialized as a dict to person_bank.json, list-of-
dicts at the top level. The file lives in the same per-user directory as
candidate_memory.json and long_term_memory.json.
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import error, request


# ============================================================
# PersonRecord — the unit of storage
# ============================================================

@dataclass
class PersonRecord:
  """A single person in the user's life.

  Designed for relational facts ("Vahid is the judge"), not for facts
  the user states about themselves. Lives in person_bank.json, separate
  from long_term_memory.json and candidate_memory.json.

  Attributes:
    person_id: stable UUID-based ID, never reused.
    canonical_name: the form Megan will say back. The "true" spelling.
    name_variants: all observed transcriptions, including STT mishears
      ("Vahid", "Bahid", "Bahi"). Used for fuzzy lookup. Includes the
      canonical form.
    relationship: short noun phrase ("judge", "course coordinator",
      "crush", "gym coach"). Optional — sometimes only a name is known.
    attributes: free-form dict of facts. Common keys: age, personality
      (list), appearance, mantra, other_facts (list). The shape is
      deliberately loose so the LLM extractor can put new fact-shapes
      in without a schema migration.
    salient_notes: free-form short notes the user explicitly flagged
      ("for the demo next week"). Distinct from attributes — these are
      situational reminders, not stable facts.
    created_at, last_mentioned_at, mention_count: lifecycle metadata.
  """
  person_id: str
  canonical_name: str
  name_variants: List[str] = field(default_factory=list)
  relationship: Optional[str] = None
  attributes: Dict[str, Any] = field(default_factory=dict)
  salient_notes: List[str] = field(default_factory=list)
  created_at: str = ""
  last_mentioned_at: str = ""
  mention_count: int = 0

  def to_prompt_summary(self) -> str:
    """One-line factual summary suitable for injecting into the system
    prompt. Keep concise — these stack up if multiple people are
    mentioned in a single turn."""
    parts = [self.canonical_name]
    if self.relationship:
      parts.append(self.relationship)

    # Pull a handful of the most useful attrs, in stable order.
    attr_bits = []
    for key in ("age", "appearance", "personality", "mantra"):
      val = self.attributes.get(key)
      if not val:
        continue
      if isinstance(val, list):
        val = ", ".join(str(x) for x in val)
      attr_bits.append(f"{key}: {val}")
    # Any other attrs the LLM stored beyond the canonical four.
    for key, val in self.attributes.items():
      if key in {"age", "appearance", "personality", "mantra"}:
        continue
      if isinstance(val, list):
        val = ", ".join(str(x) for x in val)
      attr_bits.append(f"{key}: {val}")

    if attr_bits:
      parts.append("; ".join(attr_bits))

    if self.salient_notes:
      parts.append("notes: " + " | ".join(self.salient_notes))

    return " — ".join(parts)


# ============================================================
# Lookup-side: detecting which people are mentioned in a turn
# ============================================================
#
# This is the read path. Cheap, runs every turn before generation,
# no LLM calls. The goal is: given the user's utterance, return any
# PersonRecords whose name (canonical or variant) appears in the text.
#
# STT-tolerant via:
#   1. variants list — every form we've ever heard for a person
#   2. case-insensitive matching
#   3. fuzzy fallback using difflib.SequenceMatcher when no exact
#      match is found (catches new STT mishears)
# ============================================================

# Tokens that look like names but aren't — common English words that
# would otherwise match name-shaped patterns and produce false positives
# in introduction detection. NOT used in lookup (lookup matches against
# known names only). Used in observe() to filter introduction candidates.
_NON_NAME_TOKENS = frozenset({
  "i", "me", "my", "mine", "myself",
  "you", "your", "yours", "yourself",
  "he", "him", "his", "she", "her", "hers",
  "we", "us", "our", "they", "them", "their",
  "it", "its", "this", "that", "these", "those",
  "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure",
  "right", "wrong", "true", "false", "good", "bad",
  "the", "a", "an", "some", "any", "every", "all", "no",
  "and", "or", "but", "so", "if", "then", "than", "as",
  "is", "are", "was", "were", "be", "been", "being", "am",
  "have", "has", "had", "do", "does", "did", "doing",
  "going", "trying", "thinking", "feeling", "saying", "doing",
  "today", "tomorrow", "yesterday", "now", "later", "soon",
  "morning", "evening", "afternoon", "night",
  "really", "very", "much", "just", "only", "still", "even",
  "well", "actually", "basically", "literally", "honestly",
  "hmm", "uh", "um", "oh", "ah", "eh", "huh",
  # v2.2.4 — interrogatives. Without these, "what is my favorite color"
  # captures "what" as a name candidate via the
  # `\b{NAME}\s+is\s+(?:my|our|the)\s+\w+` pattern. Same story for
  # "remember how old I am" → "how", and "remember anything about me"
  # → "anything". All flagged as introductions, all blocking LTM.
  "what", "who", "whom", "whose", "which",
  "where", "when", "why", "how",
  "anything", "something", "nothing", "everything",
  "anyone", "someone", "noone", "everyone",
  "anybody", "somebody", "nobody", "everybody",
  "megan",  # Megan's own name — never treat as a third person
})


def _tokenize_for_lookup(text: str) -> List[str]:
  """Lowercase, strip punctuation, split on whitespace. Cheap."""
  cleaned = re.sub(r"[^\w\s\-']", " ", text.lower())
  return [t for t in cleaned.split() if t]


def _fuzzy_match_score(a: str, b: str) -> float:
  """Stdlib similarity ratio in [0, 1]. Used for STT-tolerant fallback."""
  return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Empirical: SequenceMatcher ratio threshold above which two name
# tokens are considered the same person. Tuned for short names where
# a single-letter difference (V/B in "Vahid"/"Bahid") gives ~0.8 ratio.
_FUZZY_NAME_THRESHOLD = 0.78


# v2.2.7 — Distinctive role tokens for Pass 3 role-match.
#
# The existing Pass 3 rule (overlap >= 2 tokens OR any 7+-char token)
# correctly identifies "gym coach" → Tahnay and "course coordinator" →
# Vahid, but silently misses single short role words like "dad" → Jianhua
# and "mom" → Meilan. Those are 3 chars, so the 7+-char rule doesn't
# trigger, and "my dad's name" gives only 1 token overlap.
#
# The fix: any single token that appears in this set AND in both the
# query and the record's relationship is enough to match. These tokens
# are short but unambiguous as person references.
#
# False-positive risk: someone might say "this kid bullied me" while
# having a person record whose relationship contains "kid". The risk is
# tolerable — bringing the record into prompt context still lets Megan
# disambiguate by surrounding language; the anti-conflation rules from
# v2.2.1 prevent details from grafting onto the wrong subject.
_DISTINCTIVE_ROLE_TOKENS = frozenset({
  # Immediate family
  "dad", "dads", "daddy",
  "mom", "moms", "mum", "mums", "mommy", "mummy",
  "son", "sons", "kid", "kids",
  # Extended family — short enough to miss the 7-char rule
  "bro", "sis", "aunt", "uncle", "niece", "nephew", "cousin",
  # Grandparents (informal short forms)
  "nan", "gran", "papa", "nana", "pop",
})


# v2.2.7 — Role-token cross-variants. When a record's relationship contains
# one of the keys, treat any of the values as also matching that role.
# Handles the mum/mom dialect split (Aussie speakers say "mum" but the
# record may have been stored with "mom") and informal variants. Symmetric:
# each variant lists the others as alternates.
_ROLE_VARIANTS: Dict[str, tuple] = {
  "mom":    ("mum", "mommy", "mummy"),
  "mum":    ("mom", "mommy", "mummy"),
  "mommy":  ("mom", "mum", "mummy"),
  "mummy":  ("mom", "mum", "mommy"),
  "dad":    ("daddy",),
  "daddy":  ("dad",),
  "grandma":    ("granny", "nana", "nan"),
  "grandpa":    ("granddad", "grandad", "papa", "pop"),
  "grandmother": ("grandma", "granny", "nana"),
  "grandfather": ("grandpa", "granddad", "papa"),
}


def _lookup_records(
  text: str,
  records: List[PersonRecord],
) -> List[PersonRecord]:
  """Return records whose canonical_name or any variant appears in the
  text, with case-insensitive exact match preferred and a fuzzy fallback
  for STT-tolerance.

  v2.2.2: also matches records by ROLE — if the user says "the UTS
  course coordinator" and a record has that as its relationship, the
  record is returned. Catches the "remember the guy I told you about
  the UTS course coordinator? what's his name?" case where the user
  refers to a known person without using their name.

  Tokens in _NON_NAME_TOKENS never trigger matches, even if a person
  happens to be named e.g. "Will" or "Hope" — those would just be
  noisy false positives on common English. Edge cases users hit will
  get handled by storing the actual heard form in name_variants.
  """
  if not records:
    return []

  tokens = _tokenize_for_lookup(text)
  if not tokens:
    return []

  # Build lookup: lowercased variant -> record
  variant_to_record: Dict[str, PersonRecord] = {}
  for rec in records:
    for variant in [rec.canonical_name, *rec.name_variants]:
      key = variant.lower()
      if key and key not in _NON_NAME_TOKENS:
        variant_to_record.setdefault(key, rec)

  matched_ids = set()
  hits: List[PersonRecord] = []

  # Pass 1: exact (lowercased) name match.
  if variant_to_record:
    for token in tokens:
      if token in _NON_NAME_TOKENS:
        continue
      if token in variant_to_record:
        rec = variant_to_record[token]
        if rec.person_id not in matched_ids:
          matched_ids.add(rec.person_id)
          hits.append(rec)

    # Pass 2: fuzzy fallback for any token that didn't exact-match.
    # Only fires for tokens >= 4 chars to avoid matching "he"/"hi"/"on" to
    # short names by accident.
    for token in tokens:
      if len(token) < 4 or token in _NON_NAME_TOKENS:
        continue
      if token in variant_to_record:
        continue  # already exact-matched
      best_score = 0.0
      best_rec: Optional[PersonRecord] = None
      for variant, rec in variant_to_record.items():
        if len(variant) < 4:
          continue
        score = _fuzzy_match_score(token, variant)
        if score > best_score:
          best_score = score
          best_rec = rec
      if best_rec is not None and best_score >= _FUZZY_NAME_THRESHOLD:
        if best_rec.person_id not in matched_ids:
          matched_ids.add(best_rec.person_id)
          hits.append(best_rec)

  # Pass 3 (v2.2.2, extended v2.2.7): role-based match.
  # For records with a relationship phrase, return the record if enough
  # distinctive content words from that phrase appear in the utterance.
  # Catches "the UTS course coordinator" referring to Vahid, "my gym
  # coach" referring to Tahnay, "my dad" referring to Jianhua, etc.
  #
  # Trigger threshold:
  #   - 2+ overlapping content tokens ("gym coach"), OR
  #   - any single 7+-char overlapping token ("coordinator" alone), OR
  #   - any single token in _DISTINCTIVE_ROLE_TOKENS ("dad", "mom", "sis"
  #     — short but unambiguous as person references; v2.2.7).
  #
  # The token set also includes possessive- and plural-stripped forms so
  # "dad's" and "dads" in "my dad's name" / "my dads name" both overlap
  # a record relationship of "dad".
  text_token_set = set(tokens)
  for t in list(text_token_set):
    if t.endswith("'s") and len(t) > 2:
      text_token_set.add(t[:-2])
    elif t.endswith("'") and len(t) > 1:
      text_token_set.add(t[:-1])
    # Plural-s strip (e.g. "dads" -> "dad"). Require length 4+ so we
    # don't generate "i" from "is" or similar 1-2 char garbage.
    if t.endswith("s") and len(t) >= 4 and not t.endswith("ss"):
      text_token_set.add(t[:-1])

  for rec in records:
    if rec.person_id in matched_ids:
      continue
    if not rec.relationship:
      continue
    rel_tokens = [
      t for t in _tokenize_for_lookup(rec.relationship)
      if t not in _NON_NAME_TOKENS and len(t) >= 3
    ]
    if not rel_tokens:
      continue
    # Expand role tokens with cross-variants (mum/mom etc.) so a record
    # stored with relationship="mom" matches an utterance with "mum".
    rel_tokens_with_variants = set(rel_tokens)
    for t in rel_tokens:
      rel_tokens_with_variants.update(_ROLE_VARIANTS.get(t, ()))

    overlap = [t for t in rel_tokens_with_variants if t in text_token_set]
    if (len(overlap) >= 2
        or any(len(t) >= 7 for t in overlap)
        or any(t in _DISTINCTIVE_ROLE_TOKENS for t in overlap)):
      matched_ids.add(rec.person_id)
      hits.append(rec)

  return hits


# ============================================================
# Observe-side: detecting introductions and triggering extraction
# ============================================================
#
# This is the write path. Runs every turn AFTER generation (so the
# user is hearing Megan's reply while extraction happens — latency
# is invisible). Two stages:
#
#   1. Pattern detection (cheap, regex) — does this utterance look
#      like a person introduction?
#   2. If yes, fire an LLM call to extract structured info, then
#      upsert into the bank.
#
# Pattern detection is intentionally permissive — false positives are
# cheap (the LLM call returns is_person_introduction=false and nothing
# is written). False negatives are costly (a person never enters the
# bank).
# ============================================================

# Each pattern captures the candidate name in group 1. STT often
# produces lowercase output without proper-noun capitalisation, so we
# match case-insensitively and accept any word-shaped token.
_NAME_TOKEN = r"([a-zA-Z][a-zA-Z\-']{1,29})"

_INTRODUCTION_PATTERNS = [
  # Explicit remember/forget commands.
  re.compile(
    rf"\bremember\s+(?:that\s+|about\s+)?{_NAME_TOKEN}\b",
    re.IGNORECASE),
  re.compile(
    rf"\bdon'?t\s+forget\s+(?:about\s+)?{_NAME_TOKEN}\b",
    re.IGNORECASE),
  re.compile(
    rf"\badd\s+(?:to\s+memory[:\s]+)?{_NAME_TOKEN}\b",
    re.IGNORECASE),

  # "His/her/their name is X"
  re.compile(
    rf"\b(?:his|her|their|its)\s+name\s+is\s+{_NAME_TOKEN}\b",
    re.IGNORECASE),

  # v2.2.5 — "My/our/the [optional adjective] <relationship>['s|s'|s] name is X"
  # Catches "my dad's name is Jianhua", "the coach's name is Tahnay",
  # "our tutor's name is Zoe", "my best friend's name is Sarah".
  #
  # The relationship slot is whitelisted to known person-role words so
  # we don't accept "my cup's name is Big Mug" or "my game's name is
  # Cyberpunk". The optional adjective slot allows "my best friend",
  # "my old roommate" etc; the role word still has to be on the list.
  #
  # Possessive form is permissive: 's, s', plain trailing s (Whisper
  # often drops apostrophes), or nothing.
  re.compile(
    r"\b(?:my|our|the)\s+"
    r"(?:\w+\s+)?"  # optional adjective: "best", "old", "close" etc.
    r"(?:dad|dads|mom|moms|mum|mums|mother|mothers|father|fathers|"
    r"parent|parents|"
    r"brother|brothers|sister|sisters|sibling|siblings|"
    r"son|sons|daughter|daughters|child|children|kid|kids|"
    r"husband|wife|partner|spouse|"
    r"boyfriend|girlfriend|fiance|fiancee|"
    r"friend|friends|bestie|mate|buddy|"
    r"coach|trainer|tutor|teacher|instructor|mentor|"
    r"professor|lecturer|coordinator|judge|advisor|adviser|"
    r"colleague|coworker|workmate|boss|manager|employee|"
    r"cousin|uncle|aunt|aunty|auntie|nephew|niece|"
    r"grandma|grandpa|grandmother|grandfather|granny|grandad|"
    r"neighbour|neighbor|roommate|flatmate|housemate|"
    r"classmate|schoolmate|teammate|"
    r"doctor|therapist|dentist|nurse|"
    r"client|customer|landlord|tenant)"
    r"(?:'s|s'|s|')?"  # optional possessive: 's, s', plain s, or '
    rf"\s+name\s+is\s+{_NAME_TOKEN}\b",
    re.IGNORECASE),

  # "This is X" / "That's X"
  re.compile(
    rf"\bthis\s+is\s+{_NAME_TOKEN}\b",
    re.IGNORECASE),
  re.compile(
    rf"\bthat'?s\s+{_NAME_TOKEN}\b",
    re.IGNORECASE),

  # "Let me tell you about X" / "Tell you about X"
  re.compile(
    rf"\btell(?:ing)?\s+you\s+about\s+{_NAME_TOKEN}\b",
    re.IGNORECASE),

  # "X is my [role]" / "X is the [role]"
  # The role pattern is loose — anything that follows is/was a
  # noun phrase identifying their role.
  re.compile(
    rf"\b{_NAME_TOKEN}\s+is\s+(?:my|our|the)\s+\w+",
    re.IGNORECASE),

  # "The [role] is X" — captures the X.
  re.compile(
    rf"\b(?:the|our|my)\s+\w+(?:\s+\w+)?\s+is\s+(?:named|called)\s+{_NAME_TOKEN}\b",
    re.IGNORECASE),
]


def _detect_introduction_candidates(text: str) -> List[str]:
  """Scan the text for introduction patterns and return candidate
  names. Filters out _NON_NAME_TOKENS so common English words don't
  trigger LLM extraction calls.

  Returns possibly-empty list, deduplicated, lowercased.
  """
  seen = set()
  out: List[str] = []
  for pat in _INTRODUCTION_PATTERNS:
    for m in pat.finditer(text):
      name = m.group(1).strip().lower()
      if not name or name in _NON_NAME_TOKENS:
        continue
      if name not in seen:
        seen.add(name)
        out.append(name)
  return out


# ============================================================
# LLM extraction call
# ============================================================

_EXTRACTOR_PROMPT = """You extract structured information about a PERSON \
that the user has mentioned. The user said:

"{utterance}"

Return ONE JSON object and nothing else — no markdown fences, no \
preamble, no commentary. Schema:

{{
  "is_person_introduction": true | false,
  "name": "<the person's name as they said it>" | null,
  "relationship": "<short noun phrase like 'friend', 'judge', 'crush', \
'course coordinator', 'gym coach', 'colleague'>" | null,
  "attributes": {{
    "age": "<their age, if mentioned>",
    "personality": [<list of personality descriptors mentioned>],
    "appearance": "<physical description, if mentioned>",
    "mantra": "<a phrase they say often, if mentioned>",
    "other_facts": [<any other facts about them as short strings>]
  }}
}}

Rules:
- If no specific OTHER person is being introduced or described, set \
"is_person_introduction": false. Statements about the user themselves \
(e.g. "I'm feeling tired") are NOT person introductions.
- Omit any attribute fields the utterance doesn't mention. Don't \
invent facts.
- The "name" field should be JUST the person's name, not "his name is X".
- Do not include the speaker (the user) as the person — only third \
parties.
- SPELLED-OUT NAMES: if the user spells the name out with single \
letters (e.g. "his name is X, spelled J-O-H-N" or "V-A-H-I-D"), use \
the SPELLED-OUT form as the "name" field. The user is correcting the \
speech-to-text transcription — respect their spelling over the \
phonetic transcription Whisper produced. Example: "the gym coach his \
name is Taney spelled T-A-H-N-A-Y, Tane" → "name": "Tahnay" (NOT \
"Taney" or "Tane").

JSON:"""


class PersonBankError(RuntimeError):
  """Raised when the bank can't reach Ollama or parse the extraction."""


def _call_extractor(
  utterance: str,
  ollama_host: str,
  model: str,
  timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
  """Call the extraction LLM and return the parsed JSON, or None if
  extraction declines (is_person_introduction=false) or fails.

  Failures are logged but never raised — Megan has already replied to
  the user; an extraction crash must not break the turn.
  """
  prompt = _EXTRACTOR_PROMPT.format(utterance=utterance.strip())

  payload = {
    "model": model,
    "messages": [
      {"role": "system",
       "content": "You output ONLY a JSON object. No commentary."},
      {"role": "user", "content": prompt},
    ],
    "options": {
      "temperature": 0.1,   # extraction wants determinism
      "num_ctx": 8192,
    },
    "stream": False,
  }

  url = f"{ollama_host.rstrip('/')}/api/chat"
  data = json.dumps(payload).encode("utf-8")
  req = request.Request(
    url, data=data,
    headers={"Content-Type": "application/json"},
    method="POST")

  try:
    with request.urlopen(req, timeout=timeout) as resp:
      body = resp.read().decode("utf-8")
  except error.URLError as exc:
    print(f"[person_bank] extractor unreachable: {exc}")
    return None

  try:
    parsed = json.loads(body)
  except json.JSONDecodeError:
    print(f"[person_bank] non-JSON wrapper from Ollama: {body[:200]}")
    return None

  message = parsed.get("message") or {}
  content = message.get("content", "") or parsed.get("response", "")
  content = content.strip()

  # Strip any markdown code fences if the model added them despite
  # instructions.
  if content.startswith("```"):
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```\s*$", "", content)

  # Strip any leaked <think> blocks (Qwen3 sometimes emits these).
  content = re.sub(
    r"<think\b[^>]*>.*?</think>", "",
    content, flags=re.DOTALL | re.IGNORECASE).strip()

  try:
    result = json.loads(content)
  except json.JSONDecodeError:
    print(f"[person_bank] extractor returned non-JSON: {content[:200]}")
    return None

  if not isinstance(result, dict):
    return None
  if not result.get("is_person_introduction"):
    return None
  if not result.get("name"):
    return None

  return result


# ============================================================
# PersonBank — public facade
# ============================================================

class PersonBank:
  """The Person Bank.

  Thin wrapper around the storage layer that handles all reads, writes,
  introduction detection, extractor calls, and prompt formatting. The
  app-level turn handler calls just two methods on it: lookup() before
  generation, observe() after.

  All storage is via the injected `storage` object (LocalJSONStore),
  through its load_person_bank / save_person_bank methods. The bank
  does not hold an in-memory cache between calls — every lookup
  reloads from disk so external edits (manual cleanup, sync from
  another process) are picked up.
  """

  def __init__(
    self,
    storage,
    user_id: str,
    ollama_host: str = "http://localhost:11434",
    extractor_model: str = "qwen2.5:32b",
    extractor_timeout: float = 30.0,
  ) -> None:
    self.storage = storage
    self.user_id = user_id
    self.ollama_host = ollama_host
    self.extractor_model = extractor_model
    self.extractor_timeout = extractor_timeout

  # --- public read path ---

  def lookup(self, user_text: str) -> List[PersonRecord]:
    """Return PersonRecords matching names mentioned in the text.
    Empty list if nothing matches. Cheap; no LLM calls."""
    records = self._load_records()
    return _lookup_records(user_text, records)

  def format_for_prompt(
    self, hits: List[PersonRecord]) -> str:
    """Format a list of hits as a high-authority system-prompt block.
    Returns an empty string if no hits, so the caller can append
    unconditionally.

    v2.2.1: rewritten with explicit anti-conflation rules. The previous
    version only said "treat as ground truth" which let the model graft
    details from other LTM entries (e.g. gym-coach memories) onto a
    newly-introduced person who shared emotional context.
    """
    if not hits:
      return ""
    summaries = [f"- {rec.to_prompt_summary()}" for rec in hits]
    return (
      "REFERENCED PEOPLE (Megan's Person Bank — the COMPLETE record of "
      "what you know about these people):\n"
      + "\n".join(summaries)
      + "\n\nRules for these people — follow strictly:\n"
      "1. The bullets above are the COMPLETE set of facts you have on "
      "each person. If a detail is NOT in the bullet, you do NOT know "
      "it. Do not invent ages, occupations, histories, or feelings.\n"
      "2. Do NOT import details from other people in long-term memory "
      "(friends, coaches, colleagues, crushes, family) into these "
      "profiles. Each person above is their own person; treat them as "
      "completely separate from anyone else stored in memory.\n"
      "3. Do NOT assume any two named people are the same person "
      "unless the user has explicitly told you they are.\n"
      "4. Do NOT infer relationships, history, or emotional dynamics "
      "that aren't stated above. No 'you have a crush on them', no "
      "'they've been lying to you', no 'you've been bullying them' "
      "unless that is literally in the bullet for this person.\n"
      "5. If the user is currently REGISTERING or CORRECTING a person, "
      "your job is to acknowledge concisely and confirm what you've "
      "recorded — not to launch into analysis or pull in old memories "
      "about that person.\n"
      "6. If the user contradicts a fact above, accept the correction "
      "without arguing. They know their own life."
    )

  def is_introduction(self, user_text: str) -> bool:
    """Cheap pattern check — does this utterance look like a person
    introduction?

    v2.2.1 NEW: called by app.py BEFORE the pipeline runs to set
    session_context.skip_ltm_retrieval. Side-effect free; no LLM call,
    no disk I/O.

    Authoritative classification still comes from the LLM extractor
    inside observe() — this is a fast pre-check for the orchestration
    layer.
    """
    return bool(_detect_introduction_candidates(user_text))

  def get_by_id(self, person_id: str) -> Optional[PersonRecord]:
    """Look up a record by stable person_id. Returns None if not found.

    v2.2.1 NEW: used by app.py's sticky-person tracker to retrieve
    records pinned from previous turns.
    """
    for rec in self._load_records():
      if rec.person_id == person_id:
        return rec
    return None

  def all_records(self) -> List[PersonRecord]:
    """Return every record currently in the bank.

    v2.2.3 NEW: used by greeting code paths to ground the LLM in the
    full set of known people. Without this, a greeting prompt fed only
    a single memory snippet can hallucinate names — e.g. inventing
    "Dr. Chen" for "the AI Studio coordinator" when Vahid is in the
    bank but not in scope of that single snippet.
    """
    return self._load_records()

  # --- public write path ---

  def observe(self, user_text: str, *,
              explicit_remember: bool = False) -> Dict[str, Any]:
    """Process a user turn for person-related updates.

    Two outcomes possible (returned as a status dict for logging):
      - 'mention_only': name(s) matched existing records; mention
        counts bumped, no LLM call.
      - 'extracted_new' / 'extracted_update': introduction pattern
        fired OR explicit_remember=True; LLM extraction ran and a
        record was created/updated.
      - 'nothing': no patterns matched and no known names mentioned.

    v2.2.1: result dict now includes "affected_person_id" — the id of
    the record created or updated by the extractor on this turn, if any.
    None when nothing was extracted. Used by app.py's sticky tracker.

    v2.2.8: explicit_remember=True forces the LLM extractor to run even
    when no introduction pattern fires. Used when the user has signalled
    an explicit memory request ("I need you to remember Vahid also works
    at Domain", "this is important: X") so updates to existing records
    via attribute-merge land correctly without re-introducing the
    person.

    Never raises. Errors are logged and the turn proceeds.
    """
    try:
      records = self._load_records()
      now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
      status = "nothing"
      affected_id: Optional[str] = None

      # 1. Always update mention counts on existing matches.
      hits = _lookup_records(user_text, records)
      for rec in hits:
        rec.mention_count += 1
        rec.last_mentioned_at = now_iso
        # Save back to its slot in records (records are PersonRecord
        # instances; we mutated in place).
      if hits:
        status = "mention_only"

      # 2. Check for introduction patterns OR explicit-remember request.
      # The latter (v2.2.8) lets the user force attribute updates on an
      # existing record by saying "remember X also does Y".
      intro_candidates = _detect_introduction_candidates(user_text)
      if intro_candidates or explicit_remember:
        # Fire extraction. If it succeeds, upsert.
        result = _call_extractor(
          user_text, self.ollama_host,
          self.extractor_model, self.extractor_timeout)
        if result is not None:
          status, affected_id = self._upsert_from_extraction(
            records, result, now_iso, user_text)

      # 3. Persist.
      self._save_records(records)
      return {
        "status": status,
        "hits": [r.person_id for r in hits],
        "affected_person_id": affected_id,
      }

    except Exception as exc:
      print(f"[person_bank] observe failed: {exc}")
      return {"status": "error", "error": str(exc),
              "affected_person_id": None}

  # --- internals ---

  def _load_records(self) -> List[PersonRecord]:
    raw = self.storage.load_person_bank(self.user_id)
    out: List[PersonRecord] = []
    for d in raw:
      try:
        out.append(PersonRecord(**d))
      except TypeError:
        # Forward-compatibility: silently skip records with extra
        # fields from a future schema rather than crashing.
        print(f"[person_bank] skipping incompatible record: "
              f"{d.get('canonical_name', '?')}")
    return out

  def _save_records(self, records: List[PersonRecord]) -> None:
    self.storage.save_person_bank(
      self.user_id, [asdict(r) for r in records])

  def _upsert_from_extraction(
    self,
    records: List[PersonRecord],
    extraction: Dict[str, Any],
    now_iso: str,
    source_utterance: str,
  ) -> tuple[str, Optional[str]]:
    """Apply an extraction result to the records list (in-place).
    Returns ('extracted_new' | 'extracted_update' | 'nothing',
             person_id of affected record or None)."""
    heard_name = str(extraction.get("name", "")).strip()
    if not heard_name:
      return "nothing", None

    # Try to match this extraction to an existing record via fuzzy
    # name match — if STT rendered "Bahid" today but we already have
    # "Vahid" from yesterday, this is the same person.
    existing = self._find_record_by_name(records, heard_name)

    extracted_rel = extraction.get("relationship")
    extracted_attrs = extraction.get("attributes") or {}
    # Drop empty/null fields so we don't overwrite real data with blanks.
    extracted_attrs = {
      k: v for k, v in extracted_attrs.items()
      if v not in (None, "", [], {})
    }

    if existing is not None:
      # UPDATE existing person.
      if heard_name.lower() not in [
          v.lower() for v in existing.name_variants
          + [existing.canonical_name]]:
        existing.name_variants.append(heard_name)
      if extracted_rel and not existing.relationship:
        existing.relationship = str(extracted_rel)
      # Merge attributes: list attrs append-and-dedupe, others overwrite
      # only if currently empty.
      for k, v in extracted_attrs.items():
        if isinstance(v, list):
          existing_list = existing.attributes.get(k, [])
          if not isinstance(existing_list, list):
            existing_list = [existing_list]
          merged = list(existing_list)
          for item in v:
            if item not in merged:
              merged.append(item)
          existing.attributes[k] = merged
        else:
          if not existing.attributes.get(k):
            existing.attributes[k] = v
      existing.last_mentioned_at = now_iso
      existing.mention_count += 1
      return "extracted_update", existing.person_id

    # CREATE new person.
    new_id = f"person_{uuid.uuid4().hex[:8]}"
    rec = PersonRecord(
      person_id=new_id,
      canonical_name=heard_name,
      name_variants=[heard_name],
      relationship=str(extracted_rel) if extracted_rel else None,
      attributes=extracted_attrs,
      salient_notes=[],
      created_at=now_iso,
      last_mentioned_at=now_iso,
      mention_count=1,
    )
    records.append(rec)
    return "extracted_new", new_id

  def _find_record_by_name(
    self,
    records: List[PersonRecord],
    name: str,
  ) -> Optional[PersonRecord]:
    """Match a heard name to an existing record via exact-or-fuzzy
    comparison against all known variants. Returns None if no match
    or the match is too weak."""
    name_lower = name.lower()
    best_score = 0.0
    best_rec: Optional[PersonRecord] = None
    for rec in records:
      for variant in [rec.canonical_name, *rec.name_variants]:
        if variant.lower() == name_lower:
          return rec
        score = _fuzzy_match_score(name_lower, variant.lower())
        if score > best_score:
          best_score = score
          best_rec = rec
    if best_rec is not None and best_score >= _FUZZY_NAME_THRESHOLD:
      return best_rec
    return None
