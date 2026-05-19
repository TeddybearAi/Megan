"""
Unit tests for aism.person_bank.

These tests cover the bank's read path (lookup) and pattern detection
in full, since both are pure-Python and need no external services.
The end-to-end observe() path is tested with a stub extractor that
returns a canned LLM response — we want to verify the upsert logic
without needing Ollama running.

Run from the project root:
    python -m pytest tests/test_person_bank.py -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from aism.person_bank import (
  PersonBank, PersonRecord,
  _detect_introduction_candidates,
  _lookup_records,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_storage(tmp_path):
  """A minimal stand-in for LocalJSONStore that only implements the
  two methods PersonBank uses. Backed by an in-memory dict keyed by
  user_id."""
  class _StubStorage:
    def __init__(self):
      self._banks: Dict[str, List[Dict[str, Any]]] = {}

    def load_person_bank(self, user_id: str) -> List[Dict[str, Any]]:
      return list(self._banks.get(user_id, []))

    def save_person_bank(
      self, user_id: str, records: List[Dict[str, Any]]) -> None:
      self._banks[user_id] = list(records)

  return _StubStorage()


@pytest.fixture
def bank(tmp_storage):
  return PersonBank(
    storage=tmp_storage,
    user_id="test_user",
    ollama_host="http://localhost:11434",
    extractor_model="qwen2.5:32b",
  )


@pytest.fixture
def vahid_record():
  return PersonRecord(
    person_id="person_001",
    canonical_name="Vahid",
    name_variants=["Vahid", "Bahid", "Bahi"],
    relationship="judge",
    attributes={"age": "35", "personality": ["charming", "brilliant"]},
    salient_notes=["for the demo next week"],
    created_at="2026-05-15T00:00:00+00:00",
    last_mentioned_at="2026-05-15T00:00:00+00:00",
    mention_count=3,
  )


@pytest.fixture
def zoe_record():
  return PersonRecord(
    person_id="person_002",
    canonical_name="Zoe",
    name_variants=["Zoe"],
    relationship="tutor",
    attributes={"appearance": "long black hair", "personality": ["smart", "lovely"]},
    salient_notes=[],
    created_at="2026-05-15T00:00:00+00:00",
    last_mentioned_at="2026-05-15T00:00:00+00:00",
    mention_count=2,
  )


# ============================================================
# PersonRecord.to_prompt_summary
# ============================================================

class TestPromptSummary:
  def test_includes_name_and_relationship(self, vahid_record):
    summary = vahid_record.to_prompt_summary()
    assert "Vahid" in summary
    assert "judge" in summary

  def test_includes_canonical_attributes_in_order(self, vahid_record):
    summary = vahid_record.to_prompt_summary()
    assert "35" in summary
    assert "charming" in summary

  def test_includes_salient_notes(self, vahid_record):
    summary = vahid_record.to_prompt_summary()
    assert "demo next week" in summary

  def test_handles_minimal_record(self):
    rec = PersonRecord(
      person_id="x", canonical_name="Alex",
      name_variants=["Alex"],
    )
    summary = rec.to_prompt_summary()
    assert summary == "Alex"

  def test_includes_non_canonical_attributes(self):
    rec = PersonRecord(
      person_id="x", canonical_name="Pat",
      name_variants=["Pat"],
      attributes={"hobby": "chess", "drink_of_choice": "espresso"},
    )
    summary = rec.to_prompt_summary()
    assert "chess" in summary
    assert "espresso" in summary


# ============================================================
# Lookup — the read path
# ============================================================

class TestLookup:
  def test_empty_bank_returns_empty(self):
    assert _lookup_records("tell me about Vahid", []) == []

  def test_empty_text_returns_empty(self, vahid_record):
    assert _lookup_records("", [vahid_record]) == []

  def test_exact_canonical_match(self, vahid_record):
    hits = _lookup_records("what about Vahid?", [vahid_record])
    assert len(hits) == 1
    assert hits[0].person_id == "person_001"

  def test_case_insensitive(self, vahid_record):
    hits = _lookup_records("VAHID is great", [vahid_record])
    assert len(hits) == 1

  def test_variant_match_stt_mishear(self, vahid_record):
    # "Bahid" is in name_variants — should match Vahid
    hits = _lookup_records("tell me about bahid", [vahid_record])
    assert len(hits) == 1
    assert hits[0].canonical_name == "Vahid"

  def test_variant_match_short_form(self, vahid_record):
    # "Bahi" is in name_variants
    hits = _lookup_records("how's bahi doing?", [vahid_record])
    assert len(hits) == 1

  def test_fuzzy_match_new_stt_form(self, vahid_record):
    # "Vahed" is NOT in variants but is fuzzy-close to "Vahid".
    # SequenceMatcher ratio for these should exceed 0.78.
    hits = _lookup_records("what about Vahed?", [vahid_record])
    assert len(hits) == 1, "fuzzy match should catch this STT variant"

  def test_no_match_unrelated_text(self, vahid_record):
    hits = _lookup_records("I went to the gym today", [vahid_record])
    assert hits == []

  def test_no_match_common_words(self, vahid_record, zoe_record):
    # "He" and "the" and "is" should never trigger fuzzy matches
    # against any name.
    hits = _lookup_records("he is going to the show", [vahid_record, zoe_record])
    assert hits == []

  def test_multiple_people_in_one_turn(self, vahid_record, zoe_record):
    hits = _lookup_records(
      "compare Vahid and Zoe", [vahid_record, zoe_record])
    ids = {r.person_id for r in hits}
    assert ids == {"person_001", "person_002"}

  def test_same_person_mentioned_twice_dedupes(self, vahid_record):
    hits = _lookup_records(
      "Vahid is fun, Vahid is great", [vahid_record])
    assert len(hits) == 1

  def test_megan_self_reference_not_matched(self):
    # If we had a person literally named "Megan", lookup should
    # still skip it as a third-person reference because Megan
    # is the assistant's own name.
    fake_megan = PersonRecord(
      person_id="px", canonical_name="Megan",
      name_variants=["Megan"],
    )
    hits = _lookup_records("Hi Megan, how are you?", [fake_megan])
    assert hits == []

  def test_short_token_no_false_fuzzy(self):
    # Very short variants shouldn't fuzzy-match short words.
    rec = PersonRecord(
      person_id="p", canonical_name="Al", name_variants=["Al"])
    # Even with "all" or "as" in text, "Al" shouldn't fuzzy-collide.
    hits = _lookup_records("all of us went", [rec])
    # We disallow fuzzy match on tokens <4 chars, so "Al" is only
    # findable via exact match.
    assert hits == []


# ============================================================
# Introduction pattern detection
# ============================================================

class TestIntroductionDetection:
  def test_no_patterns_returns_empty(self):
    assert _detect_introduction_candidates("hello there") == []

  def test_his_name_is(self):
    cands = _detect_introduction_candidates("his name is Vahid")
    assert "vahid" in cands

  def test_her_name_is(self):
    cands = _detect_introduction_candidates("her name is Zoe")
    assert "zoe" in cands

  def test_their_name_is(self):
    cands = _detect_introduction_candidates("their name is Alex")
    assert "alex" in cands

  def test_this_is(self):
    cands = _detect_introduction_candidates("this is Vahid")
    assert "vahid" in cands

  def test_remember_pattern(self):
    cands = _detect_introduction_candidates("please remember Bahid")
    assert "bahid" in cands

  def test_dont_forget_pattern(self):
    cands = _detect_introduction_candidates("don't forget about Vahid")
    assert "vahid" in cands

  def test_x_is_my_role(self):
    cands = _detect_introduction_candidates("Vahid is my judge")
    assert "vahid" in cands

  def test_let_me_tell_you_about(self):
    cands = _detect_introduction_candidates("let me tell you about Zoe")
    assert "zoe" in cands

  def test_lowercase_stt_output(self):
    # STT often produces lowercase even for proper nouns.
    cands = _detect_introduction_candidates("his name is bahi")
    assert "bahi" in cands

  def test_common_words_filtered(self):
    # Trigger phrase followed by a common English word should NOT
    # match — "he is the one" shouldn't fire as an introduction
    # of someone named "one".
    cands = _detect_introduction_candidates("she is the one")
    assert "one" not in cands
    # "I" / "you" / "the" should all be filtered.
    cands = _detect_introduction_candidates("this is the deal")
    assert "the" not in cands

  def test_multiple_introductions_in_one_turn(self):
    text = "his name is Vahid, and her name is Zoe"
    cands = _detect_introduction_candidates(text)
    assert "vahid" in cands
    assert "zoe" in cands

  def test_megan_excluded(self):
    cands = _detect_introduction_candidates("her name is Megan")
    assert "megan" not in cands

  def test_dedup_in_single_call(self):
    cands = _detect_introduction_candidates(
      "Vahid is my judge. Remember Vahid.")
    # Vahid appears twice; should only be listed once.
    assert cands.count("vahid") == 1


# ============================================================
# observe() with a stubbed extractor — write path
# ============================================================

class TestObserve:
  def test_no_intro_no_existing_returns_nothing(self, bank):
    result = bank.observe("the weather is nice today")
    assert result["status"] == "nothing"

  def test_known_name_mention_bumps_count(self, bank, vahid_record, tmp_storage):
    # Seed the bank with Vahid.
    tmp_storage.save_person_bank(
      "test_user", [vahid_record.__dict__])

    result = bank.observe("how is Vahid doing today?")
    assert result["status"] == "mention_only"

    # Reload and verify mention_count went up.
    raw = tmp_storage.load_person_bank("test_user")
    assert len(raw) == 1
    assert raw[0]["mention_count"] == vahid_record.mention_count + 1

  def test_introduction_triggers_extractor_and_creates_record(self, bank, tmp_storage):
    fake_extraction = {
      "is_person_introduction": True,
      "name": "Vahid",
      "relationship": "judge",
      "attributes": {"age": "35", "personality": ["charming"]},
    }
    with mock.patch(
        "aism.person_bank._call_extractor",
        return_value=fake_extraction):
      result = bank.observe("his name is Vahid")

    assert result["status"] == "extracted_new"
    raw = tmp_storage.load_person_bank("test_user")
    assert len(raw) == 1
    assert raw[0]["canonical_name"] == "Vahid"
    assert raw[0]["relationship"] == "judge"
    assert raw[0]["attributes"]["age"] == "35"
    assert raw[0]["mention_count"] == 1

  def test_introduction_with_extractor_decline_does_nothing(self, bank, tmp_storage):
    # Extractor returned None — model didn't classify this as an intro.
    with mock.patch(
        "aism.person_bank._call_extractor",
        return_value=None):
      result = bank.observe("remember to buy milk")

    assert result["status"] == "nothing"
    assert tmp_storage.load_person_bank("test_user") == []

  def test_re_extraction_on_existing_updates_not_duplicates(
      self, bank, vahid_record, tmp_storage):
    tmp_storage.save_person_bank("test_user", [vahid_record.__dict__])

    # Extractor reports another attribute we didn't have before.
    fake_extraction = {
      "is_person_introduction": True,
      "name": "Vahid",
      "relationship": "judge",
      "attributes": {"mantra": "that's right"},
    }
    with mock.patch(
        "aism.person_bank._call_extractor",
        return_value=fake_extraction):
      result = bank.observe("remember Vahid says 'that's right' a lot")

    assert result["status"] == "extracted_update"
    raw = tmp_storage.load_person_bank("test_user")
    assert len(raw) == 1, "must not create a duplicate Vahid record"
    assert raw[0]["attributes"]["mantra"] == "that's right"
    # Pre-existing attrs are preserved.
    assert raw[0]["attributes"]["age"] == "35"

  def test_stt_variant_in_intro_attaches_to_existing(
      self, bank, vahid_record, tmp_storage):
    # Existing record has variants Vahid/Bahid/Bahi. New intro uses
    # "Bahid" — should attach to the same record, not create a new one.
    tmp_storage.save_person_bank("test_user", [vahid_record.__dict__])

    fake_extraction = {
      "is_person_introduction": True,
      "name": "Bahid",   # Whisper rendered V as B again
      "relationship": "judge",
      "attributes": {"new_fact": "wears glasses"},
    }
    with mock.patch(
        "aism.person_bank._call_extractor",
        return_value=fake_extraction):
      result = bank.observe("remember Bahid wears glasses")

    assert result["status"] == "extracted_update"
    raw = tmp_storage.load_person_bank("test_user")
    assert len(raw) == 1
    assert raw[0]["attributes"]["new_fact"] == "wears glasses"

  def test_observe_never_raises_on_extractor_crash(self, bank):
    def boom(*args, **kwargs):
      raise RuntimeError("simulated network failure")

    with mock.patch("aism.person_bank._call_extractor", side_effect=boom):
      result = bank.observe("his name is Vahid")

    # observe() catches everything internally.
    assert result["status"] == "error"


# ============================================================
# format_for_prompt
# ============================================================

class TestFormatForPrompt:
  def test_empty_returns_empty_string(self, bank):
    assert bank.format_for_prompt([]) == ""

  def test_includes_facts_and_safety_clause(self, bank, vahid_record):
    block = bank.format_for_prompt([vahid_record])
    assert "REFERENCED PEOPLE" in block
    assert "Vahid" in block
    assert "judge" in block
    # The "accept the correction" safety clause should be present —
    # if the bank is wrong and the user corrects, Megan must defer.
    assert "correction" in block.lower()

  def test_multiple_people_listed(self, bank, vahid_record, zoe_record):
    block = bank.format_for_prompt([vahid_record, zoe_record])
    assert "Vahid" in block
    assert "Zoe" in block
