"""Tests for the humour lexicon — sign-flip fix and Stage 2/6 dedup."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.data_models import (
  ConversationTurn, EvidenceSourceType, TurnRole,
)
from aism.session_context import SessionContext
from aism.stage2_extraction import (
  StyleEvidenceExtractor, HUMOUR_LOW_PATTERNS, HUMOUR_HIGH_PATTERNS,
)
from aism.stage6_feedback import FeedbackProcessor


def _turn(text):
  return ConversationTurn(
    turn_id="t1",
    role=TurnRole.USER,
    text=text,
    timestamp=datetime.now(timezone.utc),
  )


def _humour_evidence(evidences):
  return [e for e in evidences if e.trait == "humour"]


# ---------- Stage 2: humour sign-flip fix ----------------------------------

def test_stage2_not_funny_enough_is_high_humour():
  """The v1 bug: 'not funny enough' was stored as humour=LOW. v2 must
  recognise this as a polarity flip and emit humour=HIGH."""
  ext = StyleEvidenceExtractor()
  evs = ext.extract(
    _turn("Okay, I'm trying to be funny, but apparently you are not funny enough."),
    SessionContext(),
  )
  hev = _humour_evidence(evs)
  assert hev, "Expected at least one humour evidence"
  assert all(e.value >= 0.7 for e in hev), (
    f"All humour evidence should be HIGH, got values "
    f"{[e.value for e in hev]}")


def test_stage2_bare_not_funny_is_still_low():
  """'not funny' on its own (no 'enough') should still register as low."""
  ext = StyleEvidenceExtractor()
  evs = ext.extract(_turn("That joke was not funny."), SessionContext())
  hev = _humour_evidence(evs)
  assert hev, "Expected humour evidence for 'not funny'"
  assert all(e.value <= 0.3 for e in hev), (
    f"Bare 'not funny' should be LOW, got {[e.value for e in hev]}")


def test_stage2_be_funnier_is_high():
  ext = StyleEvidenceExtractor()
  evs = ext.extract(_turn("I want you to be funnier."), SessionContext())
  hev = _humour_evidence(evs)
  assert hev
  assert all(e.value >= 0.7 for e in hev)


def test_stage2_more_funnier_is_high():
  """The actual ltm_020 phrasing: 'we should make you more funnier'."""
  ext = StyleEvidenceExtractor()
  evs = ext.extract(
    _turn("We should make you more funnier next time."),
    SessionContext(),
  )
  hev = _humour_evidence(evs)
  assert hev, "Expected humour evidence"
  assert all(e.value >= 0.7 for e in hev), (
    f"'more funnier' should be HIGH, got {[e.value for e in hev]}")


def test_stage2_more_humour_please_is_high():
  ext = StyleEvidenceExtractor()
  evs = ext.extract(_turn("more humour please"), SessionContext())
  hev = _humour_evidence(evs)
  assert hev
  assert all(e.value >= 0.7 for e in hev)


def test_stage2_no_more_jokes_is_low():
  """'no more jokes' must NOT be tripped up by 'more X' polarity rule."""
  ext = StyleEvidenceExtractor()
  evs = ext.extract(
    _turn("No more jokes please, just tell me the answer."),
    SessionContext(),
  )
  hev = _humour_evidence(evs)
  assert hev
  assert all(e.value <= 0.3 for e in hev), (
    f"'no more jokes' should be LOW, got {[e.value for e in hev]}")


# ---------- Stage 6: same lexicon as Stage 2 -------------------------------

def test_stage6_uses_shared_low_patterns():
  """Stage 6 should reference the same HUMOUR_LOW_PATTERNS constant."""
  from aism.stage6_feedback import HUMOUR_LOW_PATTERNS as s6_low
  assert s6_low is HUMOUR_LOW_PATTERNS, (
    "Stage 6 must import HUMOUR_LOW_PATTERNS from stage2_extraction "
    "(single source of truth) — not duplicate it.")


def test_stage6_not_funny_enough_does_not_correct_to_low():
  """The Stage 6 bug: 'not funny enough' would have triggered the
  correction rule and stored humour=0.0. After the dedup + lookahead fix,
  Stage 6 must NOT treat it as a low correction."""
  fp = FeedbackProcessor()
  evs = fp.detect_feedback(
    _turn("you are not funny enough"),
    SessionContext(),
  )
  # Either no humour-low evidence at all, or only humour-high evidence.
  for e in evs:
    if e.trait == "humour":
      assert e.value >= 0.5, (
        f"Stage 6 must not classify 'not funny enough' as humour-low; "
        f"got value={e.value}")


def test_stage6_bare_not_funny_still_triggers_correction():
  fp = FeedbackProcessor()
  evs = fp.detect_feedback(
    _turn("That was not funny."),
    SessionContext(),
  )
  humour_low = [
    e for e in evs
    if e.trait == "humour" and e.value <= 0.3
  ]
  assert humour_low, "Bare 'not funny' should still trigger Stage 6 correction"


# ---------- regression: shared constants are non-empty and well-formed ----

def test_shared_constants_present_and_compilable():
  import re
  assert HUMOUR_LOW_PATTERNS, "HUMOUR_LOW_PATTERNS must not be empty"
  assert HUMOUR_HIGH_PATTERNS, "HUMOUR_HIGH_PATTERNS must not be empty"
  for p in HUMOUR_LOW_PATTERNS + HUMOUR_HIGH_PATTERNS:
    re.compile(p, re.IGNORECASE)  # raises if malformed


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
