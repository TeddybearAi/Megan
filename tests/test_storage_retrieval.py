"""Tests for the revised LocalJSONStore retrieval/ID/recency behaviour."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.storage import LocalJSONStore
from aism.embedder import CharacterNgramEmbedder


def _store_with_records(records):
  """Build a fresh store in a tempdir, populated with given LTM records."""
  tmp = Path(tempfile.mkdtemp(prefix="aism_test_"))
  emb = CharacterNgramEmbedder(dim=128)
  st = LocalJSONStore(tmp, embedder=emb)
  st.save_long_term_memory("u", records)
  return st, tmp


def _now_iso():
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _days_ago_iso(days):
  return (datetime.now(timezone.utc)
          - timedelta(days=days)).isoformat(timespec="seconds")


# ---------- ID generation: must not collide after archive ------------------

def test_id_generation_no_collision_after_archive():
  """
  Reproduces the v1 bug: v1 uses len(records)+1 to generate IDs. After
  popping an LTM record (e.g. via _detect_resolution), the count drops
  and the next promotion collides with an existing id.
  """
  records = [
    {"memory_id": "ltm_001", "user_id": "u", "utterance": "alpha",
     "store": "yes", "memory_type": "profile_fact", "importance": 4,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
    {"memory_id": "ltm_002", "user_id": "u", "utterance": "beta",
     "store": "yes", "memory_type": "profile_fact", "importance": 4,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
    {"memory_id": "ltm_003", "user_id": "u", "utterance": "gamma",
     "store": "yes", "memory_type": "profile_fact", "importance": 4,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
  ]
  st, _ = _store_with_records(records)

  # Simulate an archive: drop ltm_002 (the middle one).
  ltm = st.load_long_term_memory("u")
  ltm = [r for r in ltm if r["memory_id"] != "ltm_002"]
  st.save_long_term_memory("u", ltm)

  # Now route a new memory. v1 would generate ltm_003 (collision). v2 must
  # generate ltm_004.
  status = st.route_memory("u", {
    "user_id": "u", "utterance": "delta",
    "store": "yes", "memory_type": "profile_fact", "importance": 4,
    "notes": "",
  })
  assert status.startswith("saved_"), status

  ids = [r["memory_id"] for r in st.load_long_term_memory("u")]
  assert ids.count("ltm_003") == 1, f"id collision: {ids}"
  assert "ltm_004" in ids, f"expected ltm_004, got {ids}"


# ---------- Embedder retrieval: semantic match, no spurious match ----------

def test_retrieval_finds_semantic_match():
  records = [
    {"memory_id": "ltm_001", "user_id": "u",
     "utterance": "I just finished my gym session, I'm tired",
     "store": "yes", "memory_type": "transient_state", "importance": 1,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
    {"memory_id": "ltm_002", "user_id": "u",
     "utterance": "I love sourdough bread",
     "store": "yes", "memory_type": "preference", "importance": 3,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
  ]
  # Need to attach embeddings (records came from "old" data so we embed by
  # hand to simulate the migration path).
  emb = CharacterNgramEmbedder(dim=128)
  for r in records:
    r["embedding"] = list(emb.embed(r["utterance"]))
  st, _ = _store_with_records(records)
  st.embedder = emb

  results = st.retrieve_relevant_memories("u", "going to the gym today",
                                          top_k=2)
  assert results, "Expected the gym memory to surface for a gym query."
  assert "gym" in results[0]["summary"].lower()


def test_retrieval_filters_consolidated():
  records = [
    {"memory_id": "ltm_001", "user_id": "u",
     "utterance": "I love sourdough bread",
     "store": "yes", "memory_type": "preference", "importance": 3,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "consolidated", "notes": ""},
    {"memory_id": "ltm_002", "user_id": "u",
     "utterance": "I love sourdough bread",
     "store": "yes", "memory_type": "preference", "importance": 3,
     "frequency": 2, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
  ]
  emb = CharacterNgramEmbedder(dim=128)
  for r in records:
    r["embedding"] = list(emb.embed(r["utterance"]))
  st, _ = _store_with_records(records)
  st.embedder = emb

  results = st.retrieve_relevant_memories("u", "sourdough bread", top_k=5)
  ids = [r.get("summary") for r in results]
  # Only one should surface (the active one), even though both have the
  # same utterance.
  # We can't tell ids apart from `summary`; just assert count.
  assert len(results) == 1, f"Expected 1, got {len(results)}: {results}"


# ---------- Top-priority: recency must temper importance over time ---------

def test_top_priority_recent_beats_stale_high_importance():
  """
  An old importance-4 emotional memory must NOT win over a recent
  importance-3 memory once it ages past the half-life.
  """
  records = [
    # Old high-importance memory (60 days old).
    {"memory_id": "ltm_old", "user_id": "u",
     "utterance": "I was sad about the breakup",
     "store": "yes", "memory_type": "emotional_pattern", "importance": 4,
     "frequency": 1, "created_at": _days_ago_iso(60),
     "last_seen": _days_ago_iso(60),
     "status": "active", "notes": ""},
    # Recent moderate-importance memory (1 day old).
    {"memory_id": "ltm_new", "user_id": "u",
     "utterance": "I started a new pottery class",
     "store": "yes", "memory_type": "preference", "importance": 3,
     "frequency": 1, "created_at": _days_ago_iso(1),
     "last_seen": _days_ago_iso(1),
     "status": "active", "notes": ""},
  ]
  st, _ = _store_with_records(records)

  top = st.get_top_priority_memory("u")
  assert top is not None
  assert top["memory_id"] == "ltm_new", (
    f"Expected ltm_new (recent), got {top['memory_id']}")


def test_top_priority_filters_consolidated():
  records = [
    {"memory_id": "ltm_001", "user_id": "u",
     "utterance": "old fact", "store": "yes",
     "memory_type": "profile_fact", "importance": 5,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "consolidated", "notes": ""},
    {"memory_id": "ltm_002", "user_id": "u",
     "utterance": "active fact", "store": "yes",
     "memory_type": "profile_fact", "importance": 3,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "active", "notes": ""},
  ]
  st, _ = _store_with_records(records)
  top = st.get_top_priority_memory("u")
  assert top["memory_id"] == "ltm_002"


def test_top_priority_returns_none_when_only_consolidated():
  records = [
    {"memory_id": "ltm_001", "user_id": "u",
     "utterance": "old", "store": "yes",
     "memory_type": "profile_fact", "importance": 5,
     "frequency": 1, "created_at": _now_iso(), "last_seen": _now_iso(),
     "status": "consolidated", "notes": ""},
  ]
  st, _ = _store_with_records(records)
  assert st.get_top_priority_memory("u") is None


# ---------- New memories get an embedding when written --------------------

def test_route_memory_attaches_embedding():
  st, _ = _store_with_records([])
  st.route_memory("u", {
    "user_id": "u", "utterance": "I work at Anthropic",
    "store": "yes", "memory_type": "profile_fact", "importance": 4,
    "notes": "",
  })
  rec = st.load_long_term_memory("u")[0]
  assert "embedding" in rec
  assert isinstance(rec["embedding"], list)
  assert len(rec["embedding"]) == st.embedder.dim


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
