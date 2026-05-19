import os
import sys

# Windows: pip-installed NVIDIA packages put their DLLs under site-packages,
# but neither the DLL loader nor CTranslate2 (faster-whisper's backend) looks
# there by default. We belt-and-braces this:
#   1. os.add_dll_directory() — for code that uses LoadLibraryEx with the
#      LOAD_LIBRARY_SEARCH_USER_DIRS flag (modern Python convention).
#   2. Prepend to PATH — for older LoadLibrary calls and transitive deps that
#      don't honour add_dll_directory.
# A debug line prints the registered directories so we can confirm at startup
# that the fix actually ran. No-op on Linux/macOS.
if sys.platform == "win32":
  import site
  _site_dirs = list(site.getsitepackages())
  try:
    _site_dirs.append(site.getusersitepackages())
  except Exception:
    pass

  _nvidia_bin_dirs: list[str] = []
  for _pkg_root in _site_dirs:
    _nvidia_root = os.path.join(_pkg_root, "nvidia")
    if not os.path.isdir(_nvidia_root):
      continue
    for _sub in os.listdir(_nvidia_root):
      _bin_dir = os.path.join(_nvidia_root, _sub, "bin")
      if os.path.isdir(_bin_dir):
        _nvidia_bin_dirs.append(_bin_dir)
        if hasattr(os, "add_dll_directory"):
          try:
            os.add_dll_directory(_bin_dir)
          except (OSError, FileNotFoundError):
            pass

  if _nvidia_bin_dirs:
    os.environ["PATH"] = (
      os.pathsep.join(_nvidia_bin_dirs) + os.pathsep + os.environ.get("PATH", "")
    )
    print(f"[DLL setup] Registered {len(_nvidia_bin_dirs)} NVIDIA bin dir(s):")
    for _d in _nvidia_bin_dirs:
      print(f"  - {_d}")
  else:
    print("[DLL setup] WARNING: no nvidia/*/bin directories found under site-packages.")

import uuid
import uvicorn
import re
import collections
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from faster_whisper import WhisperModel

from aism.tts import PiperTTS
from aism.utils.precision_tools import answer_if_precision_task

# Import your core logic from main.py
from main import (
  process_user_message, _generate_contextual_greeting, LocalJSONStore,
  OllamaAdapter, AdaptiveInteractionStylePipeline, SessionContext)



def clean_text_for_tts(text: str) -> str:
  """Return a TTS-safe version of assistant text.

  The UI can still display the original reply, but the TTS engine should
  not receive emojis or markdown emphasis characters — they get pronounced
  literally (e.g. "smiley face emoji", or "asterisk word asterisk").
  Engine-agnostic: applies equally to Piper, Edge TTS, or any successor.

  v2.2.3: also strips stage-direction phrases like *chuckles*, *sighs*,
  *winks softly* — the asterisks-only strip leaves the word behind and
  Piper reads it out loud. We detect action verbs inside asterisks and
  drop the whole block (asterisks + content) before falling through to
  the asterisks-only strip that handles plain emphasis like *so* tired.
  """
  if not text:
    return "."

  emoji_pattern = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"  # symbols and pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport and map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "]+",
    flags=re.UNICODE,
  )

  # Remove emojis and the joiner/variation-selector characters often attached
  # to emoji sequences.
  cleaned = emoji_pattern.sub("", text)
  cleaned = re.sub(r"[\u200d\ufe0e\ufe0f]", "", cleaned)

  # v2.2.3: drop *stage_direction* blocks BEFORE the bare-asterisk strip.
  # Matches an asterisk-wrapped phrase whose first word is a known action
  # verb (with optional -s/-ed/-ing inflection). Catches *chuckles*,
  # *chuckled softly*, *winks*, *sighs deeply*, etc. Doesn't touch
  # plain emphasis like *so* tired or *that* part because "so" / "that"
  # aren't in the verb list.
  cleaned = _TTS_STAGE_DIRECTION_RE.sub("", cleaned)

  # Remove remaining markdown emphasis characters that can sound odd in
  # speech. The content stays, only the asterisks/backticks go.
  cleaned = cleaned.replace("*", "").replace("`", "")

  # Normalise spacing after emoji removal.
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  return cleaned or "."


# Stage-direction verbs Megan emotes with in italics. The pattern matches
# *verb*, *verb softly*, *verbed*, *verbing*, etc. — the whole block gets
# stripped from TTS. Expand the list as new emote shapes appear.
_TTS_STAGE_DIRECTION_RE = re.compile(
    r"\*\s*(?:"
    r"chuckle|laugh|smile|grin|wink|sigh|pause|nod|shrug|frown|"
    r"gasp|whisper|mutter|beam|smirk|snort|hum|sniff|yawn|cough|"
    r"tap|clap|cheer|salute|wave|blink|breathe|"
    r"think|reflect|consider|ponder|leans?\s+\w+|raises?\s+\w+|"
    r"rolls?\s+\w+|shakes?\s+\w+|tilts?\s+\w+"
    r")(?:s|es|ed|ing)?\b[^*]*\*",
    re.IGNORECASE,
)


# ============================================================
# v2.2.1 NEW — Sticky person tracker
# ============================================================
#
# Keeps the last N introduced or matched person_ids in scope so the
# Person Bank context block still appears in the system prompt for
# follow-up turns that use only pronouns ("this guy", "he", "him") and
# contain no name token.
#
# Why this exists: turn 1 of the 2026-05-15 transcript registered Vahid;
# turn 2 was "let me tell you about this guy ... he is 35 ... charming"
# — no name in the text, so lookup() returned nothing, person_block was
# empty, LTM retrieval ran unsuppressed, gym-coach memories bled into
# Vahid's context, and Megan said "you've been bullying him for years".
# Sticky fixes that — Vahid stays in scope through the next 2 turns
# even with only pronouns, so the Person Bank block is non-empty and
# skip_ltm_retrieval stays True.
#
# Not persisted across process restart. If app.py restarts mid-session
# the sticky resets to empty; the bank itself is intact, so the only
# loss is the 2-turn "pronoun fallback" window.
# ============================================================


# v2.2.4 — Third-person pronoun detector. Used to decide whether Person Bank
# should engage on a turn that has no name match but might still be about
# a known person ("tell me more about him", "what's her age").
#
# Pronouns NOT included: "it"/"its" (false-positive heavy on object refs),
# "you"/"your" (always Megan), "I"/"my" (always Teddy). Reflexive forms
# (himself/herself/themselves) included.
_THIRD_PERSON_PRONOUN_RE = re.compile(
    r"\b(?:he|him|his|himself"
    r"|she|her|hers|herself"
    r"|they|them|their|theirs|themselves)\b",
    re.IGNORECASE,
)


def _has_third_person_reference(text: str) -> bool:
  """Cheap regex check: does this turn contain a third-person pronoun?

  Used by the lazy Person Bank gate. A turn that's pure first/second
  person ("what is my favorite color", "are you mad") should not pull
  Person Bank into scope at all.
  """
  return bool(_THIRD_PERSON_PRONOUN_RE.search(text))


# v2.2.8 — Explicit user-driven memory requests.
#
# When the user says "I need you to remember X", "this is important",
# "don't forget Y", we treat the current turn as a high-priority memory
# write. The Person Bank extractor is invoked even when no introduction
# pattern fires, so updates to existing records ("remember Vahid also
# works at Domain") land via attribute-merge without requiring a
# re-introduction.
#
# Positive patterns are restricted to clear imperative or importance
# markers. Bare "remember X" is excluded because it's ambiguous
# (imperative vs. query: "remember when..." / "do you remember...").

_EXPLICIT_REMEMBER_PATTERNS = [
  # Direct save-this-now requests
  re.compile(
    r"\bi (?:need|want|wanted|would like|'?d like) you to remember\b",
    re.IGNORECASE),
  # "please remember" without requiring that/this after — also matches
  # "please remember I love X" and "please remember I'm traveling..."
  re.compile(r"\bplease remember\b", re.IGNORECASE),
  re.compile(r"\bmake sure (?:to |you )?remember\b", re.IGNORECASE),
  # Negative form ("don't forget")
  re.compile(r"\bdon'?t (?:you )?forget\b", re.IGNORECASE),
  re.compile(r"\bnever forget\b", re.IGNORECASE),
  # Imperative store commands
  re.compile(r"\bkeep (?:this|that|it) in mind\b", re.IGNORECASE),
  re.compile(r"\badd (?:this|that|it) to (?:your )?memory\b", re.IGNORECASE),
  re.compile(r"\bnote (?:this|that|it) down\b", re.IGNORECASE),
  re.compile(r"\bcommit (?:this|that|it) to memory\b", re.IGNORECASE),
  # Importance markers
  re.compile(
    r"\b(?:this|that|it)(?:'?s| is)\s+"
    r"(?:very |really |super |so |quite )?important\b",
    re.IGNORECASE),
  re.compile(
    r"\b(?:very|really|super|extremely) important\b", re.IGNORECASE),
  re.compile(
    r"\bimportant\s+(?:to|that)\s+(?:remember|know|note)\b",
    re.IGNORECASE),

  # v2.2.9 — Correction patterns. User is updating something previously
  # stated. Treated identically to explicit-remember: trigger an extractor
  # pass + boost LTM storage. The extractor's merge-with-dedup behaviour
  # for list-valued attributes means corrections to lists ("his jobs
  # include X, Y, and Z") apply cleanly. Scalar overwrites (correcting
  # age 35→36) still require manual edit — by design, to avoid silent
  # data overwrites from a misclassified correction.
  re.compile(
    r"\bactually,?\s+(?:he|she|they|it|that|this|his|her|their|i)\b",
    re.IGNORECASE),
  # Also matches "actually Vahid is 36" / "actually the meeting was..."
  # The second-word + copula structure is the correction signature.
  # Allows up to 3 words between "actually" and the copula to cover
  # "actually the company is" / "actually his real name was".
  re.compile(
    r"\bactually,?\s+(?:\w+\s+){1,3}"
    r"(?:is|was|are|were|isn'?t|wasn'?t|aren'?t|weren'?t)\b",
    re.IGNORECASE),
  re.compile(r"\bi need you to correct\b", re.IGNORECASE),
  re.compile(r"\blet me correct\b", re.IGNORECASE),
  re.compile(
    r"\bto correct\s+(?:that|this|you|the)\b", re.IGNORECASE),
  re.compile(r"\b(?:^|\W)correction\s*:\s*", re.IGNORECASE),
  re.compile(r"\bjust to correct\b", re.IGNORECASE),
  re.compile(
    r"\bnope,?\s+(?:he|she|they|it|that|this|his|her|their)\b",
    re.IGNORECASE),
]


def _is_explicit_remember(text: str) -> bool:
  """True if the user is explicitly asking Megan to commit something
  to memory, OR correcting previously stored information. Both signals
  flow through the same downstream path (trigger Person Bank extractor
  pass; if no person engaged, boost the LTM write).

  Conservative — only fires on unambiguous imperative, importance, or
  correction markers. "I remember", "do you remember", "remember when"
  all fall through as no-ops because they're queries or self-statements,
  not save requests.
  """
  return any(p.search(text) for p in _EXPLICIT_REMEMBER_PATTERNS)


class _StickyPersonTracker:
  """Per-session deque of recently-touched person_ids. In-memory only."""

  def __init__(self, max_age_turns: int = 2) -> None:
    self.max_age_turns = max_age_turns
    # (turn_idx, person_id) tuples
    self._entries: collections.deque = collections.deque(maxlen=8)

  def add(self, turn_idx: int, person_id: str) -> None:
    # Remove any prior entry for this same id (refresh, not duplicate).
    self._entries = collections.deque(
      ((t, pid) for (t, pid) in self._entries if pid != person_id),
      maxlen=self._entries.maxlen,
    )
    self._entries.append((turn_idx, person_id))

  def active_ids(self, current_turn_idx: int) -> list[str]:
    """IDs still within max_age_turns of the current turn."""
    return [
      pid for (t, pid) in self._entries
      if current_turn_idx - t <= self.max_age_turns
    ]

  def reset(self) -> None:
    """Clear all entries. Called on session reset / idle gap."""
    self._entries.clear()


# 2. Paths and AISM Setup
AUDIO_DIR = Path("audio_out")
AUDIO_DIR.mkdir(exist_ok=True)
USER_ID = "interactive_user"

# --- TTS (local, Piper) ----------------------------------------------------
# Local neural TTS via Piper, replacing the previous Edge TTS cloud call.
# The voice model lives on disk; fetch it once with:
#   python scripts/download_piper_voice.py --apply
# Default voice: en_GB-alan-medium — British male, single-speaker, 22 kHz.
# Megan's persona frames him as "British by origin, lived in Australia for
# centuries", so the British accent matches the voice while the LLM-generated
# text leans Australian. See aism/companion_persona.py for the backstory.
PIPER_VOICE_DIR = Path("voices")
PIPER_VOICE_NAME = "en_GB-alan-medium"
PIPER_MODEL_PATH = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
# Default length_scale 1.0 = Piper's trained speed. Individual call sites
# below override this where they want a different cadence (e.g. the chat
# pipeline uses 0.85 to approximate Edge TTS's previous rate="+20%" feel).
PIPER_DEFAULT_LENGTH_SCALE = 1.0
PIPER_CHAT_LENGTH_SCALE = 0.85

MODEL_NAME = "qwen2.5:32b"
# MODEL_NAME = "llama3.2:latest"

# Whisper (local STT) configuration. With a 5090/4090 we use the largest
# turbo model on CUDA at float16 — ~12-15x realtime, ~50ms for a 5s utterance.
# Drop to "distil-large-v3" or "medium.en" if you ever run on a thinner GPU.
WHISPER_MODEL_SIZE = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

storage = LocalJSONStore("./aism_data")
adapter = OllamaAdapter(model=MODEL_NAME)

# --- Embedder (R2: small non-generative neural component) ----------------
# Used by:
#   - storage.retrieve_relevant_memories  (semantic retrieval)
#   - consolidation.consolidate_memories  (near-duplicate detection)
#   - AISM Layer B / Pattern Aggregator   (wired in the AISM pass)
#
# Try the real sentence-transformer model first, fall back to the
# zero-dependency CharacterNgramEmbedder if it's not installed. Both satisfy
# the same Embedder Protocol so callers don't care which one they get.
def _build_embedder():
  try:
    from aism.embedder import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder()
    print(f"Embedder: SentenceTransformer (dim={emb.dim})")
    return emb
  except Exception as e:
    from aism.embedder import CharacterNgramEmbedder
    emb = CharacterNgramEmbedder(dim=256)
    print(f"Embedder: CharacterNgram (dim={emb.dim}) — "
          f"sentence-transformers unavailable ({type(e).__name__})")
    return emb

embedder = _build_embedder()
storage.embedder = embedder  # storage uses it on memory write + retrieval

# AISM Pass 2 components (R2 — non-generative neural component is the
# embedder; everything else here is deterministic).
from aism.stage2b_layer_b_extractor import LayerBExtractor
from aism.stage2c_pattern_aggregator import PatternAggregator
_layer_b = LayerBExtractor(embedder)
_pattern_aggregator = PatternAggregator()
print(f"AISM Layer B: enabled (encoded "
      f"{len(_layer_b.encoded_cues)} cue exemplars)")
print(f"AISM Pattern Aggregator: enabled (every 5 turns)")

# Load the Piper voice model. ~63 MB held in memory for process lifetime;
# subsequent synthesis calls are purely local (no network, no cloud).
# If the model file isn't on disk yet, fail loud with a helpful pointer
# to the download script rather than crashing on the first /api/voice call.
tts = PiperTTS(
  PIPER_MODEL_PATH,
  default_length_scale=PIPER_DEFAULT_LENGTH_SCALE,
)
print(f"Piper TTS: loaded {PIPER_VOICE_NAME} from {PIPER_MODEL_PATH}")

# Periodic consolidation trigger (every N turns).
# Threshold 0.85 is calibrated for SentenceTransformerEmbedder. If you fall
# back to CharacterNgramEmbedder, raise this to 0.95 or disable consolidation
# (CharacterNgramEmbedder clusters on stylistic similarity, not topic).
from aism import consolidation as _consolidation
from aism.memory import extract_memory_candidate
from aism.person_bank import PersonBank
CONSOLIDATE_EVERY_N_TURNS = 20
CONSOLIDATE_THRESHOLD = 0.85

# Module-global Whisper model handle. Loaded once at startup via the lifespan
# handler below so the first user turn doesn't pay the load cost.
_whisper_model: WhisperModel | None = None


def get_whisper() -> WhisperModel:
  """Return the singleton Whisper model, loading it lazily if startup hasn't run."""
  global _whisper_model
  if _whisper_model is None:
    print(
      f"Loading faster-whisper: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} ({WHISPER_COMPUTE})"
    )
    _whisper_model = WhisperModel(
      WHISPER_MODEL_SIZE,
      device=WHISPER_DEVICE,
      compute_type=WHISPER_COMPUTE,
    )
    print("faster-whisper loaded.")
  return _whisper_model


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Pre-load Whisper into VRAM and warm Ollama so the first turn is fast."""
  # Load Whisper and force CUDA kernel compilation with a tiny dummy run.
  model = get_whisper()
  try:
    warmup_audio = (np.random.randn(16000) * 0.01).astype(np.float32)
    segments, _ = model.transcribe(warmup_audio, language="en", vad_filter=False)
    list(segments)  # consume the generator to actually run inference
    print("faster-whisper warmed up.")
  except Exception as e:
    print(f"faster-whisper warmup failed (non-fatal): {e}")

  # Warm Ollama so qwen2.5:32b is hot in VRAM.
  try:
    adapter._post_chat({
      "model": adapter.model,
      "messages": [{"role": "user", "content": "hi"}],
      "options": {"temperature": 0.1},
      "stream": False,
    })
    print(f"Ollama ({MODEL_NAME}) warmed up.")
  except Exception as e:
    print(f"Ollama warmup failed (non-fatal): {e}")

  yield
  # No teardown needed — model handle is freed when the process exits.


app = FastAPI(lifespan=lifespan)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

print("=" * 60)
print(f"Megan is using Ollama model: {MODEL_NAME}")
print(f"Megan is using Whisper model: {WHISPER_MODEL_SIZE} ({WHISPER_DEVICE}/{WHISPER_COMPUTE})")
print("=" * 60)

pipeline = AdaptiveInteractionStylePipeline(
    USER_ID, str(uuid.uuid4()), storage, adapter,
    layer_b_extractor=_layer_b,
    pattern_aggregator=_pattern_aggregator,
)

# Person Bank: separate memory store for people in the user's life.
# Lookup runs before each turn (cheap, no LLM); observe runs after (may
# fire an LLM extraction call when introduction patterns match).
# Shares the main adapter's Ollama host + model, so the warm model is
# reused — no second VRAM load.
person_bank = PersonBank(
  storage=storage,
  user_id=USER_ID,
  ollama_host=adapter.host,
  extractor_model=MODEL_NAME,
)

# v2.2.1 NEW — Sticky person tracker. Keeps recently-touched people in
# scope for ~2 follow-up turns so pronoun-only utterances ("this guy",
# "him") still benefit from the Person Bank context block. Defined
# above near clean_text_for_tts.
_sticky_persons = _StickyPersonTracker(max_age_turns=2)

session_context = SessionContext()
turn_tracker = {"count": 0}
conversation_state = {"previous_user_input": None}


# Encounter-length idle-gap (matches AISM SESSION_IDLE_TTL in pipeline.py).
# After this much wall-clock silence, the next message starts a "fresh"
# conversation from Megan's perspective — counters reset, encounter_length
# returns to "fresh". LTM is untouched.
SESSION_IDLE_GAP_MINUTES = 30.0


def _maybe_reset_for_idle_gap() -> bool:
  """Reset session counters if SESSION_IDLE_GAP_MINUTES have passed since
  the last activity. Returns True if a reset happened.

  Belt-and-braces companion to the explicit POST /api/session/new the
  frontend sends on page load. If the user goes to bed without closing
  the app, this fires the next morning when they come back.
  """
  from datetime import datetime, timezone
  now = datetime.now(timezone.utc)
  last = session_context.last_activity_time
  if last.tzinfo is None:
    last = last.replace(tzinfo=timezone.utc)
  gap_minutes = (now - last).total_seconds() / 60.0
  if gap_minutes > SESSION_IDLE_GAP_MINUTES:
    session_context.reset_for_new_session()
    turn_tracker["count"] = 0
    conversation_state["previous_user_input"] = None
    _sticky_persons.reset()  # v2.2.1
    return True
  return False


class Msg(BaseModel):
  user_input: str


# --- ROUTES ---


@app.post("/api/session/new")
async def new_session():
  """Frontend calls this on page load. Resets in-session counters so Megan's
  encounter_length goes back to 'fresh' regardless of how recently the last
  message was. Long-term memory is untouched — Megan still remembers
  everything about prior days. Only the right-now sense of 'how long have
  we been chatting in this app session' resets."""
  session_context.reset_for_new_session()
  turn_tracker["count"] = 0
  conversation_state["previous_user_input"] = None
  _sticky_persons.reset()  # v2.2.1
  return {"status": "ok", "encounter_length": session_context.encounter_length()}


@app.get("/api/greeting")
async def greeting():
  try:
    top = storage.get_top_priority_memory(USER_ID)

    # v2.2.3: pre-format the full Person Bank as grounding for the
    # greeting LLM call. Without this, the model has no way to know that
    # "our course coordinator" maps to Vahid (or that Tahnay exists),
    # and fills the gap by inventing names like "Dr. Chen". Empty string
    # if the bank is empty or unreachable — _generate_contextual_greeting
    # handles that fine.
    person_block = ""
    try:
      all_records = person_bank.all_records()
      if all_records:
        person_block = person_bank.format_for_prompt(all_records)
    except Exception as e:
      print(f"[greeting] person_bank grounding non-fatal error: {e}")

    msg = _generate_contextual_greeting(
      top, adapter, person_block=person_block,
    ) if top else "Hey! Ready to continue our project?"
    clean_msg = msg.replace("AI: ", "")

    # Optional: Generate audio for greeting (might be blocked by browser autoplay)
    audio_id = "greeting_voice"
    audio_filename = f"{audio_id}.wav"
    audio_path = AUDIO_DIR / audio_filename
    tts_msg = clean_text_for_tts(clean_msg)
    # Greeting uses Piper's default (trained) speed — slightly slower than the
    # chat pipeline, giving the hello a calmer, less hurried feel.
    tts.synthesize(tts_msg, audio_path)

    return {
      "response": clean_msg,
      "audio_url": f"/api/audio/{audio_filename}"
    }
  except Exception:
    return {"response": "Welcome back! How's the AI research going?"}


async def _process_text_and_speak(user_input: str) -> dict:
  """Shared core: text in -> reply text + reply audio URL out.

  Used by both /api/chat (text in) and /api/voice (audio in, transcribed first).
  Runs the precision guard, then the AISM pipeline if needed, then TTS.
  """
  user_input = user_input.strip()

  # Belt-and-braces: if the user's been silent for >30min, treat this
  # message as the start of a fresh conversation. Counters and turn_tracker
  # reset so encounter_length() returns "fresh" for Megan's reply shape.
  # Frontend also explicitly resets on page load; this catches the case
  # where the user just leaves the tab open and comes back hours later.
  _maybe_reset_for_idle_gap()

  # Precision guard: exact tasks should not be guessed by the language model.
  # This catches things like letter counts, word counts, arithmetic, and
  # correction turns such as "you are wrong, count again" before Ollama sees them.
  precision = answer_if_precision_task(
    user_input,
    previous_user_text=conversation_state.get("previous_user_input"),
  )

  if precision.handled and precision.should_bypass_llm:
    reply_text = precision.answer
    result = {"status": f"Precision: {precision.kind}"}
  else:
    # v2.2.4 — Person Bank engagement is now LAZY.
    #
    # The previous v2.2.1 design ran Person Bank unconditionally on every
    # turn AND used "any person in scope (including sticky)" as the gate
    # to suppress LTM retrieval. Combined with `is_introduction` false-
    # positives on interrogatives ("what is my name" → captured "what" as
    # a name candidate), the gate fired on essentially every turn —
    # including pure first-person turns like "what's my favorite color"
    # and "how old am I". Result: Megan stopped accessing his LTM at all,
    # losing all 43 profile facts about Teddy.
    #
    # New rules:
    #   1. Engage Person Bank ONLY if the turn contains a third-person
    #      reference — a known name (via lookup), a role match (via
    #      lookup's Pass 3), OR a third-person pronoun (he/she/they/...)
    #      combined with someone sticky from the previous 2 turns.
    #   2. When engaged, suppress LTM only on actual INTRODUCTION turns
    #      (the cross-person grafting failure mode v2.2.1 was solving).
    #      On follow-up turns where Person Bank is engaged via sticky or
    #      pronoun, let LTM run too — the anti-conflation rules in
    #      `format_for_prompt()` will keep cross-person details from
    #      bleeding across.
    #   3. When NOT engaged, Person Bank stays completely silent: no
    #      person_context_block in the system prompt, no skip_ltm flag.
    #      Pure first/second-person turns get unimpeded LTM access.
    try:
      person_hits = person_bank.lookup(user_input)
      matched_ids = {r.person_id for r in person_hits}
      has_pronoun_ref = _has_third_person_reference(user_input)
      current_turn = turn_tracker["count"]

      # Sticky records only matter if there's something to anchor them to.
      # A pronoun resolves to whoever's recently in scope. A direct name
      # match is its own anchor. With neither, we don't drag sticky people
      # into a turn that's clearly about Teddy herself.
      sticky_records = []
      if person_hits or has_pronoun_ref:
        for pid in _sticky_persons.active_ids(current_turn):
          if pid in matched_ids:
            continue
          rec = person_bank.get_by_id(pid)
          if rec is not None:
            sticky_records.append(rec)

      all_person_records = person_hits + sticky_records

      if all_person_records:
        # Person Bank engages: inject the context block, decide LTM gate
        # based on whether THIS turn is an actual introduction.
        session_context.person_context_block = (
          person_bank.format_for_prompt(all_person_records))

        is_intro = person_bank.is_introduction(user_input)
        # v2.2.4: gate only on introduction, NOT on sticky/match. This
        # fixes the "Megan can't access memory" bug from 2026-05-15.
        session_context.skip_ltm_retrieval = is_intro

        # Register matched records in sticky tracking for follow-up turns.
        for rec in person_hits:
          _sticky_persons.add(current_turn, rec.person_id)

        matched_names = ", ".join(r.canonical_name for r in person_hits)
        sticky_names = ", ".join(r.canonical_name for r in sticky_records)
        bits = []
        if matched_names:
          bits.append(f"matched: {matched_names}")
        if sticky_names:
          bits.append(f"sticky: {sticky_names}")
        print(f"[person_bank] engaged {', '.join(bits)} "
              f"(intro={is_intro}, skip_ltm={session_context.skip_ltm_retrieval})")
      else:
        # No person reference, no sticky carryover. Person Bank silent.
        # LTM runs unimpeded. This is the common case for self-talk.
        session_context.person_context_block = ""
        session_context.skip_ltm_retrieval = False
    except Exception as e:
      print(f"[person_bank] lookup non-fatal error: {e}")
      session_context.person_context_block = ""
      session_context.skip_ltm_retrieval = False

    # v2.2.9 — Set the explicit-remember flag for main.process_user_message.
    # The signal fires ONLY when:
    #   (a) the user said "remember X" / "this is important" / etc., AND
    #   (b) Person Bank did NOT engage on this turn.
    # When Person Bank engaged, the extractor pass (in observe() below)
    # handles attribute updates on the relevant record — boosting LTM
    # too would pollute the identity block with third-party facts as if
    # they were Teddy's own. When Person Bank didn't engage, this IS a
    # self-fact the user wants persisted, so main.py boosts the write.
    try:
      _explicit_remember_detected = _is_explicit_remember(user_input)
      _person_bank_engaged = bool(session_context.person_context_block)
      session_context.explicit_remember = (
        _explicit_remember_detected and not _person_bank_engaged)
      if session_context.explicit_remember:
        print("[memory] explicit_remember signal — "
              "no person engaged, will boost LTM write")
    except Exception as e:
      print(f"[memory] explicit_remember detection error: {e}")
      session_context.explicit_remember = False

    # AISM Core Logic
    result = process_user_message(
      user_input, USER_ID, storage, adapter, pipeline, session_context,
      turn_tracker["count"])
    reply_text = result["reply"]

  conversation_state["previous_user_input"] = user_input
  # Stash the reply so the next turn's classifier can detect short
  # affirmations ("yeah, sure, go ahead") that accept a prior offer-to-help
  # ("Want my take?") and route them to SEEKING_HELP. Without this, the
  # affirmation falls through to AMBIGUOUS and Megan loops on the off-ramp.
  session_context.previous_assistant_reply = reply_text
  turn_tracker["count"] += 1

  # Accumulate per-turn statistics for encounter_length() bucketing on the
  # NEXT turn. user_word_count and assistant_word_count drive the SUBSTANTIAL
  # threshold when a few long exchanges should count as more than a few
  # short ones; last_activity_time drives the idle-gap reset above.
  session_context.note_turn(user_input, reply_text)

  # Memory candidate extraction + routing.
  #
  # This block is the voice-app counterpart to main.py's process_user_turn
  # (lines 147-154). Without it, app.py runs the AISM pipeline (style
  # adaptation, profile updates, evidence log) but never writes anything
  # into candidate_memory.json or long_term_memory.json. That meant every
  # voice conversation was memory-write-disabled: raw_log.json kept
  # growing but nothing the user said ever became a retrievable memory.
  #
  # Wrapped in try/except because memory failure must NEVER kill a turn —
  # Megan is mid-conversation and the user is waiting on the TTS.
  try:
    memory_candidate = extract_memory_candidate(USER_ID, user_input)
    memory_status = storage.route_memory(USER_ID, memory_candidate)
    storage.promote_candidate_memories(USER_ID)
    storage.decay_candidate_memories(USER_ID)
    if memory_status != "ignored":
      print(f"[memory] {memory_status}")
  except Exception as e:
    print(f"[memory] non-fatal error: {e}")

  # Person Bank observe — detect introduction patterns and fire LLM
  # extraction if found. Updates mention counts on existing people
  # regardless. The user is already hearing Megan's TTS reply at this
  # point, so the extractor's ~1-2s latency is invisible.
  # Failures must not break the turn — observe() catches everything
  # internally and returns a status dict; this try/except is defence
  # in depth in case observe() itself raises.
  #
  # v2.2.1: register the affected person_id in sticky tracking so the
  # next 2 turns can refer to this person by pronoun only and still
  # benefit from the Person Bank context block.
  # v2.2.8: forward explicit-remember signal so attribute updates on
  # existing records ("remember Vahid also works at Domain") fire the
  # extractor without requiring a re-introduction.
  # v2.2.9: reuse the _explicit_remember_detected local computed above
  # (one regex pass per turn instead of two).
  try:
    _er = locals().get("_explicit_remember_detected")
    if _er is None:
      _er = _is_explicit_remember(user_input)
    if _er:
      print(f"[person_bank] explicit_remember detected — forcing extractor pass")
    pb_result = person_bank.observe(
      user_input, explicit_remember=_er)
    if pb_result.get("status") not in (None, "nothing"):
      print(f"[person_bank] {pb_result}")
    affected_id = pb_result.get("affected_person_id")
    if affected_id:
      _sticky_persons.add(turn_tracker["count"], affected_id)
  except Exception as e:
    print(f"[person_bank] observe non-fatal error: {e}")

  # Periodic memory consolidation. No-op unless the trigger condition is
  # met; cheap to call every turn.
  try:
    n_merged = _consolidation.maybe_consolidate(
      storage,
      USER_ID,
      every_n_turns=CONSOLIDATE_EVERY_N_TURNS,
      turn_count=turn_tracker["count"],
      similarity_threshold=CONSOLIDATE_THRESHOLD,
    )
    if n_merged:
      print(f"[consolidation] merged {n_merged} near-duplicate memories")
  except Exception as e:
    print(f"[consolidation] non-fatal error: {e}")

  # TTS Generation
  audio_id = uuid.uuid4().hex
  audio_filename = f"{audio_id}.wav"
  audio_path = AUDIO_DIR / audio_filename

  tts_text = clean_text_for_tts(reply_text)
  # Chat pipeline uses length_scale=0.85 (~+20% faster than trained speed),
  # approximating the previous Edge TTS rate="+20%" cadence. Note: Piper
  # changes speed during synthesis, so pitch shifts slightly with the rate
  # (Edge did this as post-synthesis time-stretch, which preserved pitch).
  tts.synthesize(tts_text, audio_path, length_scale=PIPER_CHAT_LENGTH_SCALE)

  return {
    "response": reply_text,
    "status": result["status"],
    "audio_url": f"/api/audio/{audio_filename}"
  }


def _transcribe_upload_to_text(audio: UploadFile, raw_bytes: bytes) -> str:
  """Run faster-whisper on an uploaded audio blob and return the transcript.

  Uses VAD filtering to suppress silence-induced hallucinations such as the
  classic "Thanks for watching" artifact that Whisper emits on quiet input.
  """
  # Pick a sensible suffix so faster-whisper / ffmpeg can sniff the format.
  ctype = (audio.content_type or "").lower()
  if "webm" in ctype:
    suffix = ".webm"
  elif "ogg" in ctype:
    suffix = ".ogg"
  elif "wav" in ctype:
    suffix = ".wav"
  elif "mp4" in ctype or "m4a" in ctype:
    suffix = ".m4a"
  else:
    suffix = ".webm"  # MediaRecorder default in Chromium

  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(raw_bytes)
    tmp_path = tmp.name

  try:
    model = get_whisper()
    segments, _info = model.transcribe(
      tmp_path,
      language="en",
      vad_filter=True,
      vad_parameters=dict(min_silence_duration_ms=300),
      beam_size=5,
      condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
  finally:
    Path(tmp_path).unlink(missing_ok=True)


# Patterns that strip Megan's voice-control phrases from a Whisper transcript
# before the text is handed to memory/chat. These are conversation control
# tokens, not message content — they should not pollute long-term memory.
#
# Pattern ORDER MATTERS. Multi-word mishears ("make an over") must match before
# the single-word trailing "over" pattern, otherwise "over" gets stripped first
# and leaves "make an" stranded in the transcript.
_VOICE_CONTROL_PATTERNS = (
  # Common Chrome STT mishears for "Megan over" / "Megan your turn" — match
  # these first as full phrases so we don't half-strip them.
  re.compile(r"\bmake\s+(?:an|and|in|it)\s+(?:over|your\s+turn)\b[,.!?\s]*", re.IGNORECASE),
  re.compile(r"\b(?:maggie|maken|magna|making|again|may\s+again)\s+(?:over|your\s+turn)\b[,.!?\s]*", re.IGNORECASE),
  # Canonical "Megan over" / "Megan your turn" anywhere.
  re.compile(r"\bmegan[,\s]+(?:over|your\s+turn)\b[,.!?\s]*", re.IGNORECASE),
  # Trailing "over" / "your turn" with optional preceding "Megan".
  re.compile(r"[,.!?\s]*\b(?:megan'?s?\s+)?(?:over|your\s+turn)\b[,.!?\s]*$", re.IGNORECASE),
  # Leading wake phrases.
  re.compile(r"^[\s,.!?]*(?:hey|hi|hello|ok|okay)\s+megan\b[,.!?\s]*", re.IGNORECASE),
  re.compile(r"^[\s,.!?]*megan\b[,.!?\s]+", re.IGNORECASE),
  # Trailing bare "Megan" preceded by sentence punctuation. This catches the
  # case where the user said "Megan over" multiple times because Chrome STT
  # missed the first attempt — the audio Whisper transcribed contains an
  # earlier vocative "Megan," that we want to drop. Requires preceding
  # punctuation so we don't accidentally strip a real sentence-final reference.
  re.compile(r"[.!?,]+\s*megan\b[\s,.!?]*$", re.IGNORECASE),
)


def strip_voice_controls(text: str) -> str:
  """Remove wake words ('Hey Megan', 'Megan,') and turn-end phrases ('Megan over',
  'Megan your turn') from a Whisper transcript before it reaches the chat logic."""
  if not text:
    return ""
  cleaned = text
  for pattern in _VOICE_CONTROL_PATTERNS:
    cleaned = pattern.sub(" ", cleaned)
  cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?")
  return cleaned


@app.post("/api/chat")
async def chat(request: Msg):
  return await _process_text_and_speak(request.user_input)


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...)):
  """Audio in -> transcript + reply text + reply audio URL out.

  Single round-trip endpoint used by the React UI. The frontend records a
  turn with MediaRecorder, POSTs the blob here when "Megan over" fires, and
  receives both Megan's text reply and her TTS audio URL in one response.
  """
  raw = await audio.read()
  if not raw:
    raise HTTPException(400, "Empty audio upload.")

  try:
    transcript = _transcribe_upload_to_text(audio, raw)
  except Exception as e:
    print(f"Whisper transcription failed: {e}")
    raise HTTPException(500, f"Transcription failed: {e}")

  if not transcript:
    return {
      "transcript": "",
      "response": "I didn't catch that. Could you say it again?",
      "status": "empty_transcript",
      "audio_url": None,
    }

  # Strip wake words and turn-end phrases ("Hey Megan", "Megan over", etc.)
  # before the transcript reaches memory and the LLM. The raw transcript is
  # still returned in the response for the UI to display.
  cleaned = strip_voice_controls(transcript)

  if not cleaned:
    return {
      "transcript": transcript,
      "response": "I heard the wake word but didn't catch a question. Could you repeat?",
      "status": "wake_only",
      "audio_url": None,
    }

  reply = await _process_text_and_speak(cleaned)
  reply["transcript"] = cleaned
  reply["transcript_raw"] = transcript
  return reply


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
  """Debug endpoint: audio in -> transcript only. No LLM, no TTS.

  Useful for `curl`-testing Whisper accuracy in isolation:
    curl -F audio=@sample.webm http://127.0.0.1:8000/api/transcribe
  """
  raw = await audio.read()
  if not raw:
    raise HTTPException(400, "Empty audio upload.")
  try:
    transcript = _transcribe_upload_to_text(audio, raw)
  except Exception as e:
    raise HTTPException(500, f"Transcription failed: {e}")
  return {"transcript": transcript}


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
  path = AUDIO_DIR / filename
  if not path.exists():
    raise HTTPException(404)
  return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)
