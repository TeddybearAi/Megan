"""
AISM — Turn Intent Classifier
=============================

Deterministic per-turn intent classifier. Reads the user's current message
and labels it with one of:

    VENTING            user wants presence + validation, not advice
    SEEKING_HELP       user has a concrete question or decision to make
    THINKING_OUT_LOUD  user is reflecting; soliciting perspective, not a plan
    DECISION           user has already decided; reporting, not asking
    CASUAL             greeting, small talk
    AMBIGUOUS          can't tell — defaults to listen-first behaviour

The downstream consumer (``stage5_retrieval`` → ``ResponsePolicy.mode`` →
``ollama_adapter``) uses the label to gate Megan's advice-emitting parts of
the system prompt. ``VENTING`` and ``AMBIGUOUS`` both trigger listen-first
behaviour; ``SEEKING_HELP`` enables suggestions.

Why this exists
---------------
The user observed across seven sessions: Megan responds to every turn with
a "help" frame — empathy stub, paraphrase, suggestions, scripts, numbered
lists, ends with "what do you think?". This is correct behaviour for help-
seeking turns and wrong behaviour for venting turns. A friend reads the
room before responding. So does this classifier.

Design constraints
------------------
- R1: no LLM calls in the AISM pipeline. Pure regex + small heuristic.
- R2: no neural components used here (the classifier is purely lexical).
- Conservative posture: when uncertain, return AMBIGUOUS so the downstream
  prompt defaults to listening. False-positive-advice is the bug we're
  fixing; false-positive-listening is harmless (Megan can ask "want my
  take?" mid-conversation if she gets it wrong).
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ============================================================
# Intent labels
# ============================================================

VENTING = "venting"
SEEKING_HELP = "seeking_help"
THINKING_OUT_LOUD = "thinking_out_loud"
DECISION = "decision"
CASUAL = "casual"
AMBIGUOUS = "ambiguous"
HOSTILE = "hostile"
WARM_OPENING = "warm_opening"


# ============================================================
# Pattern groups
# ============================================================
#
# All patterns are matched case-insensitive against the lowercased message.

# --- HOSTILE: personal attacks directed AT Megan --------------------------
# Distinct from VENTING (which is cursing at the world / a third party).
# These patterns target Megan specifically — second-person pronouns paired
# with insults, or direct second-person curses.
#
# Megan needs permission to push back when these fire. The OVERRIDE block
# in the adapter defines what "push back" means (indignation first;
# matching register only on persistence; never the aggressor).
HOSTILE_STRONG_MARKERS = [
  # Direct second-person curses
  r"\bfuck\s+you\b",
  r"\bgo\s+fuck\s+yourself\b",
  r"\bscrew\s+you\b",
  r"\bup\s+yours\b",
  # "you piece of shit" and family
  r"\byou(?:'re|\s+are)?\s+(?:a\s+|an\s+|such\s+a\s+|such\s+an\s+)?"
  r"(?:piece\s+of\s+(?:shit|crap)|asshole|cunt|bitch|bastard|moron|"
  r"idiot|loser|jerk|prick|dipshit|dumbass)\b",
  # "you're stupid/useless/etc."
  r"\byou(?:'re|\s+are)\s+(?:so\s+|really\s+|fucking\s+|so\s+fucking\s+)?"
  r"(?:stupid|dumb|useless|pathetic|worthless|garbage|trash|"
  r"hopeless|incompetent|annoying)\b",
  # "shut up" family
  r"\bshut\s+(?:the\s+fuck\s+)?up\b",
  r"\bshut\s+it\b",
  # "I hate you" / "I despise you" — direct
  r"\bi\s+(?:hate|despise|loathe|can'?t\s+stand)\s+you\b",
  # Direct address with insult: "Megan, you idiot"
  r"\bmegan,?\s+you\s+(?:moron|idiot|piece\s+of\s+shit|cunt|asshole)\b",
]

# Affectionate / playful dampeners. When these are present in the same
# message, hostile confidence drops because the user is being playful, not
# actually attacking ("oh fuck you, that was clever").
HOSTILE_AFFECTIONATE_DAMPENERS = [
  r"\bhaha\b",
  r"\blol\b",
  r"\blmao\b",
  r"\brofl\b",
  r"\bdarling\b",
  r"\bsweet(?:ie|heart)\b",
  r"\bmy dear\b",
  r"\blove\s+(?:you|that|it|this)\b",
  r"\bjk\b",
  r"\bjust\s+kidding\b",
  r"\bkidding\b",
  r"\bteasing\b",
  r"\b(?:you'?re\s+)?(?:hilarious|funny|clever|brilliant)\b",
  r"\bthat\s+was\s+(?:good|great|funny|clever|brilliant|hilarious)\b",
  r"\b😂|🤣|😄|😅|😆\b",  # emoji
  r"\b<3\b",
]

# --- VENTING: explicit "I want emotional support" / "I'm bitching" -------
# These are the strongest possible signals. One match → high-confidence
# venting label.
EXPLICIT_VENTING_MARKERS = [
  r"\bi\s+(?:just\s+)?(?:want|need|wanted)\s+(?:your\s+)?(?:emotional?\s+|some\s+)?support\b",
  r"\bi\s+(?:just\s+)?(?:want|need|wanted)\s+(?:someone\s+)?(?:to\s+)?listen\b",
  r"\b(?:come\s+)?bitch(?:ing)?\s+(?:about|to|with)\b",
  r"\b(?:vent|rant)\s+(?:about|to)\b",
  r"\bi\s+(?:just\s+)?(?:want|need)\s+to\s+(?:vent|rant|bitch)\b",
  r"\bcan\s+you\s+(?:just\s+)?listen\b",
  r"\bi\s+know\s+what\s+to\s+do\b",   # "of course I know what to do" = explicit "no advice"
  r"\bi\s+already\s+know\s+what\b",
  r"\bnot\s+looking\s+for\s+(?:advice|solutions?|suggestions?)\b",
  r"\bdon'?t\s+(?:want|need)\s+(?:advice|solutions?|suggestions?)\b",
]

# --- VENTING: emotional/frustrated affect markers --------------------------
# Strong signals on their own but accumulate with other cues for confidence.
AFFECT_VENTING_MARKERS = [
  r"\bi'?m\s+so\s+(?:mad|pissed|angry|frustrated|annoyed|upset|irritated|sick)\b",
  r"\bi'?m\s+(?:really|so)?\s*(?:mad|pissed|angry|frustrated|annoyed|irritated|upset)\b",
  # "kind irritated" / "kind of pissed" — common Whisper-transcribed
  # phrasings of "I'm kind of irritated" with the 'i'm' dropped.
  r"\bkind(?:\s+of)?\s+(?:mad|pissed|angry|frustrated|annoyed|irritated|upset)\b",
  r"\bget\s+me\s+out\b",
  r"\bdamn\s+it\b",
  r"\bfuck(?:ing)?\b",
  r"\bbullshit\b",
  r"\bbullshitt(?:ing|ed|er)\b",
  r"\bbull\s*sh\*?t\b",
  r"\bgod\s*damn\b",
  r"\bpissed\s+me\s+off\b",
  r"\bdoes\s+(?:that|this)\s+make\s+sense\b",   # the "tell me I'm not crazy" tag
  r"\bif\s+it\s+were\s+you\b",                  # validation-seeking framing
  r"\bare\s+you\s+gonna\s+be\s+mad\b",          # validation-seeking framing
  r"\b(?:so\s+)?(?:weird|strange|crazy|insane)\s+(?:right|huh)\b",
  r"\bcan\s+you\s+believe\b",
  r"\bi\s+(?:hate|can'?t\s+stand)\s+(?:it\s+)?when\b",
  # "definitely misleading", "definitely lying" etc. — strong opinion-laden
  # accusations the user makes about a third party, almost always venting.
  r"\bdefinitely\s+(?:misleading|lying|wrong|insulting|unfair|unprofessional)\b",
  r"\bmisleading\s+right\b",
]

# --- VENTING: trailing tag-questions seeking agreement, not advice ---------
# These almost always end venting turns ("...you know what I mean", "right?").
# Detected at the tail of the message specifically.
VENTING_TAG_QUESTION_PATTERNS = [
  r"(?:right|huh|yeah|okay)\s*[?.]?\s*$",
  r"you\s+know\s+(?:what\s+i\s+mean|right)\s*[?.]?\s*$",
  r"you\s+know\s*[,.?]?\s*$",
  r"(?:isn'?t\s+it|aren'?t\s+they|don'?t\s+(?:you|they))\s*\??\s*$",
]

# --- SEEKING_HELP: explicit request for advice / instructions / a plan -----
SEEKING_HELP_MARKERS = [
  r"\bwhat\s+should\s+i\b",
  r"\bhow\s+(?:do|should|can|would)\s+i\b",
  r"\bhow\s+(?:do|should|can|would)\s+you\b",   # "how would you handle..."
  r"\bany\s+(?:idea|ideas|advice|suggestions?|tips?)\b",
  r"\bgive\s+me\s+(?:advice|some\s+advice|tips?|suggestions?|ideas?)\b",
  r"\bwhat\s+do\s+you\s+(?:recommend|suggest)\b",
  r"\bcan\s+you\s+help\s+me\b",
  r"\bhelp\s+me\s+(?:with|figure\s+out|work\s+out|understand)\b",
  r"\bwalk\s+me\s+through\b",
  r"\bexplain\s+(?:to\s+me\s+)?how\b",
  r"\bteach\s+me\s+how\b",
  r"\bi\s+(?:need|want)\s+(?:to\s+know\s+)?(?:how|what)\b",
  r"\bdo\s+you\s+know\s+how\b",
  r"\bcan\s+you\s+(?:write|generate|create|build|make|fix|debug)\b",
]

# --- DECISION: user is reporting, not asking ------------------------------
DECISION_MARKERS = [
  r"\bi'?(?:ve|d)\s+(?:decided|already\s+decided)\b",
  r"\bi'?m\s+(?:definitely|going\s+to|gonna)\s+\w+",
  r"\bi'?ll\s+(?:just\s+)?(?:do|go|try|take|tell|talk|call)\b",
  r"\bi\s+(?:will|won'?t)\s+\w+",
  r"\bmind\s+(?:is\s+)?made\s+up\b",
]

# --- CASUAL: greetings, small talk ----------------------------------------
# Matched against the whole stripped message; short messages only.
CASUAL_FULL_MESSAGE_PATTERNS = [
  r"^(?:hey|hi|hello|sup|yo)\b[!\s,.]*$",
  r"^(?:hey|hi|hello|sup|yo)\s+(?:there|megan)\b[!\s,.]*$",
  r"^good\s+(?:morning|afternoon|evening|night)\b[!\s,.]*$",
  r"^how\s+(?:are\s+you|are\s+you\s+doing|'?s\s+it\s+going)\b[!\s,?.]*$",
  r"^what'?s\s+up\b[!\s,?.]*$",
  r"^(?:thanks|thank\s+you|thx|ty)\b[!\s,.]*$",
  r"^(?:bye|goodbye|good\s+night|gn|cya|later)\b[!\s,.]*$",
]


# --- ACCEPT_OFFER: short affirmations after Megan offered a take -----------
# Fires when (a) the previous assistant reply ended with an offer-of-take
# phrase like "Want my take?" and (b) the user's message is a short
# affirmation accepting it. Without this rule, "yeah, sure, go ahead"
# falls through to AMBIGUOUS (listen-first), which keeps Megan in the
# venting block — meaning every reply ends with another "Want my take?"
# and the take is never delivered. Off-ramp loop.
#
# Detection runs in two halves:
#   1. The TAIL of the previous assistant reply must contain an offer
#      pattern (Want my take? / Hear my take? / My take on this?).
#   2. The CURRENT user message must be a short, whole-message affirmation.
# Both halves must match. Either half alone is not enough.

OFFER_TAKE_TAIL_PATTERNS = [
  # Canonical offer from the venting mode-override block.
  r"\bwant\s+my\s+take\b[^.?!]{0,80}[?.]?\s*$",
  # Common variant.
  r"\bhear\s+my\s+take\b[^.?!]{0,80}[?.]?\s*$",
  # "My take on this/that/it?" — terser variant.
  r"\bmy\s+take\s+on\s+(?:this|that|it)\b[^.?!]{0,20}[?.]?\s*$",
]

# Reusable sub-patterns for ACCEPT_OFFER. Kept here so the multi-clause
# pattern below stays readable.
_ACCEPT_STARTER = r"(?:yes|yeah|yep|yup|sure|ok|okay)"
_ACCEPT_VERB = (
  r"(?:sure|please(?:\s+do)?|do\s+it|go\s+ahead|tell\s+me|why\s+not|"
  r"let'?s\s+hear\s+it|hit\s+me|fire\s+away|i'?d\s+like\s+that|"
  r"sounds\s+good|sounds\s+great)"
)

ACCEPT_OFFER_FULL_PATTERNS = [
  # Bare starter alone: "yes", "yeah", "sure", "ok", "okay" (+ punctuation).
  rf"^{_ACCEPT_STARTER}[!.,\s]*$",
  # Bare "please".
  r"^please[!.,\s]*$",
  # Starter + one-or-more accept-verbs ("yeah sure", "yeah, sure, go ahead",
  # "ok tell me", "yes please do").
  rf"^{_ACCEPT_STARTER}(?:[,!.\s]+{_ACCEPT_VERB})+[!.,\s]*$",
  # Accept-verb(s) without starter ("go ahead", "please do", "tell me").
  rf"^{_ACCEPT_VERB}(?:[,!.\s]+{_ACCEPT_VERB})*[!.,\s]*$",
  # Polite "I'd like to hear it" / "I'd love to hear it" / "I'd like that".
  r"^i'?d\s+(?:like\s+(?:that|to\s+hear)|love\s+to\s+hear)[!.,\s]*$",
]


# --- WARM_OPENING: user expresses affection / appreciation TO Megan -------
# Direct relational warmth aimed AT Megan: "I like you", "you're a good
# friend", "thanks for being here", "I appreciate you". Distinct from
# generic CASUAL "thanks!" (transactional) and from task feedback "you
# did well on that one" (about the work, not the relationship).
#
# Patterns require either:
#   (a) "I [verb] you" with affectionate verb, OR
#   (b) "you're [positive relational descriptor]", OR
#   (c) "thanks/thank you for [emotional object]" — not "thanks for the
#       answer" or bare "thanks!", OR
#   (d) "X of you" affection ("very sweet of you", "kind of you to say"), OR
#   (e) past/perfect tense relational ("you've been a good friend"), OR
#   (f) user-as-subject emotional response ("I'm so touched",
#       "tears in my eyes", "means so much to me", "made my day").
#
# False-positive guards: "I like coffee" (non-person object), "I like
# that idea" (idea/it/that, not you), "you're a good idea" (idea ≠
# friend) — none of these match. Affect-venting markers ("pissed",
# "frustrated") co-present in the same turn cause WARM_OPENING to
# fall through to VENTING (see precedence rule in classify_turn_intent).
WARM_OPENING_PATTERNS = [
  # Direct affection: "I like you", "I love you", "I really like you"
  # Un-anchored: mid-message "I like you" inside longer warmth turns
  # also fires (real users say "thank you so much. yeah, I like you").
  r"\bi\s+(?:really\s+|just\s+|so\s+|kind\s+of\s+)*"
  r"(?:like|love|adore|appreciate)\s+you\b",
  # Bare "love you" (often appears as a sign-off)
  r"^love\s+you\b[!.,\s]*$",
  # "You're [positive relational descriptor]"
  r"\byou'?re\s+(?:my\s+|the\s+|a\s+|such\s+a\s+)?"
  r"(?:hero|saviour|savior|best|favourite|favorite|amazing|"
  r"wonderful|incredible|brilliant|the\s+best|the\s+greatest|"
  r"a\s+legend|a\s+gem|"
  r"a\s+(?:good|great|true|real|amazing|the\s+best)\s+"
  r"(?:friend|mate|guy|bloke|person|companion|listener))\b",
  # "You're so/really [warm trait]"
  r"\byou'?re\s+(?:so|really|just|always)\s+"
  r"(?:sweet|kind|nice|caring|thoughtful|patient|understanding|"
  r"amazing|wonderful|supportive|helpful)\b",
  # PAST/PERFECT TENSE: "you've been a good friend", "you were so kind"
  # Real users say this constantly — past tense was a gap.
  r"\byou'?ve\s+been\s+(?:so\s+|such\s+a\s+|really\s+|always\s+)?"
  r"(?:sweet|kind|nice|caring|amazing|wonderful|supportive|helpful|"
  r"patient|understanding|a\s+(?:good|great|real|true)\s+"
  r"(?:friend|mate|guy|bloke|person|companion))\b",
  r"\byou\s+(?:were|have\s+been)\s+(?:so\s+|such\s+a\s+|really\s+|always\s+)?"
  r"(?:sweet|kind|nice|caring|amazing|wonderful|supportive|helpful|"
  r"patient|understanding|a\s+(?:good|great|real|true)\s+"
  r"(?:friend|mate|guy|bloke|person|companion))\b",
  # "X of you" affection: "very sweet of you", "kind of you to say"
  r"\b(?:that'?s|that\s+is|so|very|really|how)\s+"
  r"(?:sweet|kind|nice|thoughtful|generous|lovely)\s+of\s+you\b",
  r"\b(?:sweet|kind|nice|thoughtful|lovely)\s+of\s+you\s+to\s+"
  r"(?:say|do|offer|help|listen|notice|think|mention)\b",
  # Emotional thanks: "thanks for being here / listening / caring"
  r"\bthanks?\s+for\s+"
  r"(?:being\s+(?:here|there|around|a\s+(?:good|real|great)\s+friend)|"
  r"listening|caring|understanding|the\s+support|"
  r"always\s+being\s+there|all\s+(?:of\s+)?this)\b",
  r"\bthank\s+you\s+for\s+"
  r"(?:being\s+(?:here|there|around|a\s+(?:good|real|great)\s+friend)|"
  r"listening|caring|understanding|the\s+support|"
  r"always\s+being\s+there|all\s+(?:of\s+)?this)\b",
  # "I appreciate you" (specifically you, not "this" or "that")
  r"\bi\s+(?:really\s+|just\s+)?appreciate\s+you\b",
  # USER-AS-SUBJECT emotional response: "I'm so touched", "I have tears
  # in my eyes". Whisper STT sometimes transcribes "I'm so touched" as
  # "I'm so touching" — covering that typo deliberately.
  r"\bi'?m\s+(?:so|really|kind\s+of|getting)\s+"
  r"(?:touched|moved|emotional|grateful|overwhelmed|teary|misty)\b",
  r"\bi'?m\s+so\s+touching\b",  # STT typo for "touched"
  r"\b(?:tears?\s+in\s+my\s+eyes?|tearing\s+up|crying\s+a\s+little)\b",
  # "That means so much / a lot / the world"
  r"\b(?:that|this|it|what\s+you\s+said)\s+"
  r"(?:means|meant)\s+(?:so\s+much|a\s+lot|the\s+world|everything)\b",
  r"\b(?:means|meant)\s+(?:so\s+much|a\s+lot|the\s+world)\s+to\s+me\b",
  # "You made my day / week / year"
  r"\b(?:you|that)\s+(?:just\s+)?made\s+my\s+(?:day|week|year|night)\b",
  # "You make me feel..." — relational impact
  r"\byou\s+(?:make|made)\s+me\s+(?:feel\s+(?:better|good|happy|"
  r"safe|seen|heard|loved|understood|less\s+alone)|smile|laugh)",
  # "Couldn't have done it without you"
  r"\bcouldn'?t\s+(?:have\s+)?(?:done|made\s+it)\s+"
  r"(?:this|it|that|through\s+(?:this|it|that))\s+without\s+you\b",
  # "I'm glad/lucky to have you"
  r"\bi'?m\s+(?:so\s+)?(?:glad|lucky|grateful|happy)\s+"
  r"(?:to\s+have\s+you|(?:that\s+)?you'?re\s+(?:here|around|in\s+my\s+life))\b",
  # "You really get/understand me"
  r"\byou\s+(?:really\s+|always\s+|just\s+)*"
  r"(?:get|understand)\s+me\b",
]


# ============================================================
# Compiled patterns
# ============================================================

_EXPLICIT_VENTING = [re.compile(p, re.IGNORECASE) for p in EXPLICIT_VENTING_MARKERS]
_AFFECT_VENTING = [re.compile(p, re.IGNORECASE) for p in AFFECT_VENTING_MARKERS]
_VENTING_TAGS = [re.compile(p, re.IGNORECASE) for p in VENTING_TAG_QUESTION_PATTERNS]
_SEEKING_HELP = [re.compile(p, re.IGNORECASE) for p in SEEKING_HELP_MARKERS]
_DECISION = [re.compile(p, re.IGNORECASE) for p in DECISION_MARKERS]
_CASUAL = [re.compile(p, re.IGNORECASE) for p in CASUAL_FULL_MESSAGE_PATTERNS]
_HOSTILE_STRONG = [re.compile(p, re.IGNORECASE) for p in HOSTILE_STRONG_MARKERS]
_HOSTILE_DAMPENERS = [
  re.compile(p, re.IGNORECASE) for p in HOSTILE_AFFECTIONATE_DAMPENERS
]
_OFFER_TAKE_TAIL = [re.compile(p, re.IGNORECASE) for p in OFFER_TAKE_TAIL_PATTERNS]
_ACCEPT_OFFER = [re.compile(p, re.IGNORECASE) for p in ACCEPT_OFFER_FULL_PATTERNS]
_WARM_OPENING = [re.compile(p, re.IGNORECASE) for p in WARM_OPENING_PATTERNS]


# ============================================================
# Helpers
# ============================================================


def _count_matches(text: str, patterns: List[re.Pattern]) -> int:
  return sum(1 for p in patterns if p.search(text))


def _has_match(text: str, patterns: List[re.Pattern]) -> bool:
  return any(p.search(text) for p in patterns)


def _tail(text: str, n_chars: int = 120) -> str:
  """The last ``n_chars`` of the message, used to detect tail-position cues
  like trailing tag-questions or specific verbs."""
  return text[-n_chars:] if len(text) > n_chars else text


def _is_short(text: str) -> bool:
  return len(text.strip()) <= 40


def _looks_long_and_ramble_y(text: str) -> bool:
  """Detect long unstructured speech-to-text monologues — a strong signal
  of venting (vs. a tight help-seeking question, which is usually short)."""
  stripped = text.strip()
  if len(stripped) < 200:
    return False
  # Lots of "you know" / "um" / "uh" / "like" / "okay" filler is a Whisper
  # transcript of someone processing out loud.
  filler_count = len(re.findall(
    r"\b(?:you\s+know|um+|uh+|like|okay|right)\b", stripped, re.IGNORECASE))
  # Rough density: filler per 50 chars. Speech rambling tends to be ≥ 1.
  return filler_count / max(1, len(stripped) // 50) >= 1.2


def _ends_with_question_mark(text: str) -> bool:
  return text.rstrip().endswith("?")


# ============================================================
# Public API
# ============================================================


def classify_turn_intent(
    text: str,
    previous_assistant_reply: str = "",
) -> Tuple[str, float]:
  """Classify a user turn's intent.

  Returns ``(label, confidence)`` where ``confidence`` is in [0, 1].
  Confidence ≥ 0.6 means "act on this label". Confidence < 0.6 means
  "default to AMBIGUOUS" (caller may override the label).

  Parameters
  ----------
  text: str
      The user's current turn.
  previous_assistant_reply: str
      Megan's previous reply, if any. Used only by the ACCEPT_OFFER rule
      (rule 2). Empty/missing is fine — the classifier degrades gracefully
      to its pre-context behaviour. Callers without conversation state
      can omit it.

  Order of precedence:
    1. HOSTILE — personal attack on Megan. Checked first because it's
       more specific than VENTING and the response shape is completely
       different (push back vs sit with).
    2. ACCEPT_OFFER — short affirmation immediately after Megan offered
       to deliver a take ("Want my take?" → "yeah, sure, go ahead").
       Without this rule, the affirmation falls through to AMBIGUOUS
       and Megan loops on the off-ramp.
    2.5 WARM_OPENING — direct affection / appreciation aimed AT Megan
       ("I like you" / "you're a good friend"). Triggers recency-
       weighted memory retrieval and a friend-shaped warm response
       block (not corporate gratitude).
    3. CASUAL — short, greeting-shaped messages exit early
    4. EXPLICIT VENTING — "I just want emotional support" / "I know what to do"
    5. SEEKING_HELP — explicit asks dominate, even when affect is present
       (matters because users say "I'm so mad, what should I do?" — the
       question wins)
    6. AFFECT VENTING (+ tag-questions or rambling) — combines weaker signals
    7. DECISION — reporting a made-up mind
    8. AMBIGUOUS — fallback
  """
  if not text or not text.strip():
    return AMBIGUOUS, 0.0

  stripped = text.strip()
  tail = _tail(stripped)

  # --- 1. HOSTILE: personal attack on Megan ---
  # Strong patterns alone are enough. Affectionate dampeners ("haha",
  # "darling", "love you") lower the label to AMBIGUOUS — playful
  # "fuck you" between friends shouldn't trigger a pushback response.
  hostile_hits = _count_matches(stripped, _HOSTILE_STRONG)
  if hostile_hits >= 1:
    dampener_hits = _count_matches(stripped, _HOSTILE_DAMPENERS)
    if dampener_hits >= 1:
      # Playful context. Don't fire HOSTILE. Let it fall through.
      pass
    else:
      # Genuine hostility. Confidence scales with severity.
      confidence = min(1.0, 0.85 + 0.05 * hostile_hits)
      return HOSTILE, confidence

  # --- 2. ACCEPT_OFFER: short yes to Megan's prior "Want my take?" ---
  # Both halves must match: previous assistant reply ended with an offer,
  # AND current message is a short whole-message affirmation. Either
  # alone is not enough. Conservative on purpose: a false positive here
  # routes a venting turn into advisor mode, which is the regression we
  # spent yesterday fixing.
  if previous_assistant_reply and _is_short(stripped):
    prev_tail = _tail(previous_assistant_reply.strip().lower(), 120)
    if _has_match(prev_tail, _OFFER_TAKE_TAIL):
      if _has_match(stripped, _ACCEPT_OFFER):
        return SEEKING_HELP, 0.90

  # --- 2.5. WARM_OPENING: direct affection / appreciation aimed at Megan ---
  # "I like you", "you're a good friend", "thanks for being here",
  # "I appreciate you" — second-person warmth, not task feedback.
  # Sits above CASUAL because emotional "thanks for being here" should
  # not be downgraded to a generic "thanks!". Sits below HOSTILE and
  # ACCEPT_OFFER (both more specific). Patterns are tight to avoid
  # false positives on "I like coffee" / "you're a good idea".
  #
  # Co-presence guard: if AFFECT_VENTING markers ("pissed", "frustrated",
  # "damn it", "fuck", etc.) are ALSO present, the user is processing
  # distress with politeness sprinkled in, not having a pure relational
  # moment. Fall through to AFFECT_VENTING below. Example: "I'm so
  # pissed off but thanks for being here" → VENTING wins, not warmth.
  if _has_match(stripped, _WARM_OPENING):
    if not _has_match(stripped, _AFFECT_VENTING):
      return WARM_OPENING, 0.85
    # else: fall through to AFFECT_VENTING further down

  # --- 3. CASUAL: short greetings / sign-offs ---
  if _is_short(stripped):
    if _has_match(stripped, _CASUAL):
      return CASUAL, 0.95

  # --- 4. EXPLICIT VENTING: high-confidence "no advice" markers ---
  explicit_venting_hits = _count_matches(stripped, _EXPLICIT_VENTING)
  if explicit_venting_hits >= 1:
    # If the user ALSO explicitly asks for advice in the same message
    # ("I'm bitching but seriously what should I do") → seeking help wins.
    if _has_match(stripped, _SEEKING_HELP):
      return SEEKING_HELP, 0.80
    confidence = min(1.0, 0.85 + 0.05 * explicit_venting_hits)
    return VENTING, confidence

  # --- 5. SEEKING_HELP: explicit ask ---
  help_hits = _count_matches(stripped, _SEEKING_HELP)
  if help_hits >= 1:
    # Stronger confidence when the ask is in the tail (= the actual ask
    # closes the turn, not just a passing question in a longer rant).
    tail_help_hits = _count_matches(tail, _SEEKING_HELP)
    if tail_help_hits >= 1:
      return SEEKING_HELP, min(1.0, 0.85 + 0.05 * help_hits)
    return SEEKING_HELP, 0.70

  # --- 6. AFFECT VENTING: emotional markers + supporting signals ---
  affect_hits = _count_matches(stripped, _AFFECT_VENTING)
  tag_question = _has_match(tail, _VENTING_TAGS)
  rambly = _looks_long_and_ramble_y(stripped)

  if affect_hits >= 1:
    # Confidence accumulates with supporting signals.
    confidence = 0.55 + 0.10 * min(affect_hits, 3)
    if tag_question:
      confidence += 0.10
    if rambly:
      confidence += 0.10
    confidence = min(1.0, confidence)
    return VENTING, confidence

  # Affect-less long rambles still count as venting, lower confidence.
  # Confidence scales with length: a 200-char rambly+tag turn is mildly
  # venting; a 1000-char one is unambiguously venting.
  if rambly and tag_question:
    length_bonus = min(0.15, max(0.0, (len(stripped) - 200) / 1000 * 0.15))
    return VENTING, min(1.0, 0.65 + length_bonus)
  if rambly:
    return THINKING_OUT_LOUD, 0.60

  # --- 7. DECISION ---
  if _has_match(stripped, _DECISION) and not _ends_with_question_mark(stripped):
    return DECISION, 0.70

  # --- 8. Fallback ---
  # Trailing question mark with no explicit-help marker is genuinely
  # ambiguous — could be venting validation ("right?") or open-perspective.
  # Stay AMBIGUOUS so the adapter defaults to listening.
  return AMBIGUOUS, 0.35


__all__ = [
  "classify_turn_intent",
  "VENTING", "SEEKING_HELP", "THINKING_OUT_LOUD",
  "DECISION", "CASUAL", "AMBIGUOUS", "HOSTILE", "WARM_OPENING",
]
