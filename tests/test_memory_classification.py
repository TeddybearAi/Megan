"""Tests for the revised memory candidate classifier."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.memory import (
  extract_memory_candidate, infer_importance, infer_memory_type,
  infer_store_label,
)


# ---------- transient_state catches the v1 over-classification cases -------

def test_transient_im_tired():
  m = extract_memory_candidate("u", "I'm tired today")
  assert m["memory_type"] == "transient_state", m
  assert m["store"] == "no", m
  assert m["importance"] == 1, m


def test_transient_im_busy():
  m = extract_memory_candidate("u", "I'm busy right now")
  assert m["memory_type"] == "transient_state"
  assert m["store"] == "no"


def test_transient_im_just_saying_hi():
  # ltm_010 in the real data — was profile_fact/imp5 in v1, must be
  # transient/store=no in v2.
  m = extract_memory_candidate("u", "i'm just saying hi how are you")
  assert m["memory_type"] == "transient_state"
  assert m["store"] == "no"


def test_transient_im_kind_of_tired():
  # ltm_011-style real utterance.
  m = extract_memory_candidate(
    "u", "yeah i just finished my gym session i'm kind of tired")
  assert m["memory_type"] == "transient_state"


# ---------- profile_fact still works for genuinely stable facts ------------

def test_profile_fact_im_a_software_engineer():
  m = extract_memory_candidate("u", "I'm a software engineer at Google")
  assert m["memory_type"] == "profile_fact"
  assert m["store"] == "yes"
  assert m["importance"] == 4   # tightened from 5 in v1


def test_profile_fact_my_name_is_teddy():
  m = extract_memory_candidate("u", "my name is Teddy")
  assert m["memory_type"] == "profile_fact"
  assert m["store"] == "yes"


def test_profile_fact_i_work_at():
  m = extract_memory_candidate("u", "I work at Anthropic as a researcher")
  assert m["memory_type"] == "profile_fact"


def test_profile_fact_i_live_in():
  m = extract_memory_candidate("u", "I live in Sydney")
  assert m["memory_type"] == "profile_fact"


# ---------- emotional_pattern catches the bare 'i'm sad' / 'crush' case ----

def test_emotional_im_anxious():
  m = extract_memory_candidate("u", "I'm anxious about the interview")
  assert m["memory_type"] == "emotional_pattern", m


def test_emotional_crush_has_girlfriend():
  # ltm_001 in real data. v1 classified this as profile_fact (because of
  # "i'm"). v2 should classify as emotional_pattern.
  m = extract_memory_candidate(
    "u",
    "i'm kind a little bit sad right now because i just find out my crush "
    "has a girlfriend",
  )
  assert m["memory_type"] in ("emotional_pattern", "transient_state"), m
  # Either is acceptable: the "right now" makes it transient, which is also
  # a fine outcome (don't promote). The CRITICAL property is that it does
  # not become profile_fact.
  assert m["memory_type"] != "profile_fact", m


def test_emotional_i_feel_stressed():
  m = extract_memory_candidate("u", "I feel stressed about work")
  assert m["memory_type"] == "emotional_pattern"


# ---------- IVF / medical context wins over stray temporary markers --------

def test_ivf_is_emotional_event():
  # ltm_023 in real data — was profile_fact in v1, then mis-classified as
  # temporary_plan because of an incidental "next week". IVF context must
  # win.
  m = extract_memory_candidate(
    "u",
    "I was doing IVF for a while now. Last month was my fourth round. "
    "The doctor will call next week.",
  )
  assert m["memory_type"] == "emotional_pattern", m


def test_pregnancy_is_emotional_event():
  m = extract_memory_candidate("u", "I'm pregnant!")
  assert m["memory_type"] == "emotional_pattern"


def test_application_is_long_term_goal():
  # ltm_012 in real data — must win over the "right now"/"i'm" transient
  # markers.
  m = extract_memory_candidate(
    "u",
    "I'm kind of tired right now but I'm actually recently applied for "
    "being a police officer in New South Wales",
  )
  assert m["memory_type"] == "long_term_goal", m


def test_got_the_job_is_long_term_goal():
  m = extract_memory_candidate("u", "I got the job at the law firm")
  assert m["memory_type"] == "long_term_goal"


# ---------- voice-control noise must not promote ---------------------------

def test_voice_control_only_does_not_promote():
  m = extract_memory_candidate("u", "Megan over")
  assert m["store"] == "no", m


def test_short_noise_does_not_promote():
  m = extract_memory_candidate("u", "okay")
  assert m["store"] == "no"


# ---------- preferences and goals still classify correctly -----------------

def test_long_term_goal():
  m = extract_memory_candidate("u", "I want to learn Spanish")
  assert m["memory_type"] == "long_term_goal"


def test_preference_i_like():
  m = extract_memory_candidate("u", "I like dark chocolate")
  assert m["memory_type"] in ("preference", "style_preference")
  assert m["store"] in ("yes", "maybe")


def test_style_preference():
  m = extract_memory_candidate(
    "u", "I prefer concise, direct answers please")
  # The style keywords ("concise", "direct") promote this to style_preference.
  assert m["memory_type"] in ("style_preference", "preference")


def test_temporary_plan():
  m = extract_memory_candidate("u", "Maybe I'll go for a run tomorrow")
  # "i'll" doesn't match transient (which only matches "i'm") so this
  # should fall to temporary_plan.
  assert m["memory_type"] == "temporary_plan", m


# ---------- regression: bare 'i'm' alone no longer fires profile_fact ------

def test_bare_im_alone_is_not_profile_fact():
  # The v1 trap. Without any structural cue (a/an/the + noun, work at, etc),
  # bare "i'm" must NOT classify as profile_fact.
  m = extract_memory_candidate("u", "i'm not sure about that")
  assert m["memory_type"] != "profile_fact", m


if __name__ == "__main__":
  # Hand-rolled runner so we don't need pytest.
  import inspect
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
