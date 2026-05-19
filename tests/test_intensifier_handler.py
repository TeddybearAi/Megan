"""Tests for the negation/intensifier handler (intensifier_handler.py)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.data_models import ConversationTurn, TurnRole, EvidenceSourceType
from aism.session_context import SessionContext
from aism.intensifier_handler import detect_polarity_evidence
from aism.stage2_extraction import TRAIT_LEXICONS, StyleEvidenceExtractor


def _turn(text):
  return ConversationTurn(
    turn_id="t1",
    role=TurnRole.USER,
    text=text,
    timestamp=datetime.now(timezone.utc),
  )


def _evidences_for_trait(evidences, trait):
  return [e for e in evidences if e.trait == trait]


# ---------- "not X enough" → wants more X ---------------------------------

def test_not_concise_enough_wants_short_verbosity():
  evs = detect_polarity_evidence(
    _turn("Your answers are not concise enough."),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "verbosity")
  assert v, "Expected verbosity evidence"
  assert v[0].value == "short"


def test_not_warm_enough_wants_high_warmth():
  evs = detect_polarity_evidence(
    _turn("You're not warm enough when I'm having a hard time."),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "warmth")
  assert v, "Expected warmth evidence"
  assert v[0].value >= 0.7


def test_not_direct_enough_wants_high_directness():
  evs = detect_polarity_evidence(
    _turn("Your answers are not direct enough — get to the point."),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "directness")
  assert v
  assert v[0].value >= 0.7


# ---------- "too X" → wants less X ----------------------------------------

def test_too_warm_wants_low_warmth():
  evs = detect_polarity_evidence(
    _turn("You're too warm — tone it down a bit."),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "warmth")
  assert v
  assert v[0].value <= 0.4


def test_too_long_wants_short_verbosity():
  evs = detect_polarity_evidence(
    _turn("That answer was too long."),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "verbosity")
  assert v
  assert v[0].value == "short"


def test_too_brief_wants_long_verbosity():
  evs = detect_polarity_evidence(
    _turn("That was too brief."),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "verbosity")
  assert v
  assert v[0].value == "long"


# ---------- "more X please" → wants more X --------------------------------

def test_more_warm_please_wants_high_warmth():
  evs = detect_polarity_evidence(
    _turn("Could you be more warm please?"),
    SessionContext(), TRAIT_LEXICONS,
  )
  v = _evidences_for_trait(evs, "warmth")
  assert v
  assert v[0].value >= 0.7


# ---------- guard: "no more X" / "any more X" must not flip ---------------

def test_no_more_jokes_does_not_trigger_more_funny():
  """Adversarial: 'no more X' should NOT match the 'more X please' rule."""
  evs = detect_polarity_evidence(
    _turn("No more jokes please."),
    SessionContext(), TRAIT_LEXICONS,
  )
  # If the polarity handler fires high-humour here, the negative-lookbehind
  # in the regex is broken.
  high_humour = [
    e for e in evs
    if e.trait == "humour" and isinstance(e.value, (int, float)) and e.value > 0.5
  ]
  assert not high_humour, (
    f"'no more jokes' must not produce high-humour evidence: {high_humour}")


# ---------- end-to-end: full StyleEvidenceExtractor uses preprocessor -----

def test_extractor_uses_polarity_preprocessor_end_to_end():
  """Smoke test: the v1 ltm_020 phrase must yield humour=HIGH (not LOW) via
  the full StyleEvidenceExtractor.extract pipeline."""
  ext = StyleEvidenceExtractor()
  evs = ext.extract(
    _turn(
      "Okay, I'm trying to be funny, but apparently you are not funny enough. "
      "We should make you more funnier."
    ),
    SessionContext(),
  )
  humour = [e for e in evs if e.trait == "humour"]
  assert humour, "Expected humour evidence"
  assert all(e.value >= 0.7 for e in humour), (
    f"All humour evidence should be HIGH; got {[e.value for e in humour]}")
  # And specifically the polarity handler should have done the work.
  has_polarity = any(
    e.metadata.get("polarity_handler") for e in humour)
  assert has_polarity, (
    "Expected at least one humour evidence to be tagged "
    "polarity_handler=True")


if __name__ == "__main__":
  ns = dict(globals())
  failures = []
  passed = 0
  for name, fn in ns.items():
    if name.startswith("test_") and callable(fn):
      try:
        fn()
        passed += 1
        print(f"PASS  {name}")
      except AssertionError as e:
        failures.append((name, e))
        print(f"FAIL  {name}: {e}")
      except Exception as e:
        failures.append((name, e))
        print(f"ERROR {name}: {type(e).__name__}: {e}")
  print(f"\n{passed} passed, {len(failures)} failed")
  sys.exit(0 if not failures else 1)
