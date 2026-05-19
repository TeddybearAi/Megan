"""
AISM — Memory Storage Tests
============================
Verifies that remarkable memory candidates are extracted, persisted, and
retrieved from the local AISM JSON store.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from aism import LocalJSONStore
from aism.memory import extract_memory_candidate


def test_memory_storage_and_retrieval() -> None:
  tmpdir = Path(tempfile.mkdtemp(prefix="aism_memory_test_"))
  try:
    store = LocalJSONStore(tmpdir)
    user_id = "memory_test_user"

    candidate = extract_memory_candidate(
      user_id, "I prefer short, direct answers and simple explanations.")
    assert candidate["memory_type"] in {"style_preference", "preference"}
    assert candidate["store"] == "yes"

    status = store.route_memory(user_id, candidate)
    assert status == "saved_yes"

    long_term = store.load_long_term_memory(user_id)
    assert len(long_term) == 1
    assert long_term[0]["frequency"] == 1
    assert long_term[0]["utterance"] == candidate["utterance"]

    status = store.route_memory(user_id, candidate)
    assert status == "updated_yes"
    assert store.load_long_term_memory(user_id)[0]["frequency"] == 2

    candidate_2 = extract_memory_candidate(user_id, "I might travel tomorrow.")
    assert candidate_2["store"] == "maybe"
    status = store.route_memory(user_id, candidate_2)
    assert status == "saved_maybe"
    assert len(store.load_candidate_memory(user_id)) == 1

    retrieved = store.retrieve_relevant_memories(user_id, "travel plans")
    assert isinstance(retrieved, list)
    assert any("travel" in item["summary"].lower() for item in retrieved)
  finally:
    shutil.rmtree(tmpdir)
