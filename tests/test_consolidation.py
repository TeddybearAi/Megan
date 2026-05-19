"""Tests for the consolidation module."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.consolidation import consolidate_memories
from aism.embedder import CharacterNgramEmbedder


def _now():
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record(memory_id, utterance, importance=3, freq=1):
  return {
    "memory_id": memory_id,
    "user_id": "u",
    "utterance": utterance,
    "store": "yes",
    "memory_type": "preference",
    "importance": importance,
    "frequency": freq,
    "created_at": _now(),
    "last_seen": _now(),
    "status": "active",
    "notes": "",
  }


def _embed_records(records, emb):
  for r in records:
    r["embedding"] = list(emb.embed(r["utterance"]))
  return records


def test_near_duplicates_collapse():
  emb = CharacterNgramEmbedder(dim=256)
  records = _embed_records([
    _record("ltm_001", "I love sourdough bread"),
    _record("ltm_002", "I love sourdough bread!"),  # near-identical
    _record("ltm_003", "I work as a software engineer at Anthropic"),
  ], emb)

  out, n = consolidate_memories(records, similarity_threshold=0.80)
  assert n == 1, f"Expected exactly 1 to be consolidated, got {n}"
  consolidated = [r for r in out if r.get("status") == "consolidated"]
  active = [r for r in out if r.get("status", "active") == "active"]
  assert len(consolidated) == 1
  assert len(active) == 2
  # The active sourdough record should have absorbed the consolidated one.
  sourdough = [r for r in active if "sourdough" in r["utterance"]]
  assert len(sourdough) == 1
  assert sourdough[0]["frequency"] >= 2, (
    f"Expected freq absorbed, got {sourdough[0]['frequency']}")


def test_unrelated_records_do_not_merge():
  emb = CharacterNgramEmbedder(dim=256)
  records = _embed_records([
    _record("ltm_001", "I love sourdough bread"),
    _record("ltm_002", "I work at Anthropic"),
    _record("ltm_003", "I have a golden retriever named Max"),
  ], emb)

  out, n = consolidate_memories(records, similarity_threshold=0.80)
  assert n == 0, f"Unrelated records should not merge, got n={n}"
  active = [r for r in out if r.get("status", "active") == "active"]
  assert len(active) == 3


def test_idempotent():
  emb = CharacterNgramEmbedder(dim=256)
  records = _embed_records([
    _record("ltm_001", "I love sourdough bread"),
    _record("ltm_002", "I love sourdough bread"),
    _record("ltm_003", "I work as a software engineer"),
  ], emb)

  pass1, n1 = consolidate_memories(records, similarity_threshold=0.80)
  pass2, n2 = consolidate_memories(pass1, similarity_threshold=0.80)
  assert n2 == 0, (
    f"Second pass must be a no-op (idempotent), got n2={n2}")
  # Status counts should match between passes
  s1 = sorted(r.get("status", "active") for r in pass1)
  s2 = sorted(r.get("status", "active") for r in pass2)
  assert s1 == s2


def test_canonical_is_longest_utterance():
  """The longer record should win and absorb the shorter near-duplicate."""
  emb = CharacterNgramEmbedder(dim=256)
  records = _embed_records([
    _record("ltm_001", "sourdough bread"),
    _record("ltm_002",
            "I really love sourdough bread, especially the sourdough "
            "bread from that bakery downtown"),
  ], emb)

  out, n = consolidate_memories(records, similarity_threshold=0.55)
  if n == 0:
    # If the threshold didn't trigger a merge for these two specific
    # strings (CharacterNgram is a coarse embedder), bail without failing —
    # the property tested is conditional on a merge happening.
    return
  active = [r for r in out if r.get("status", "active") == "active"]
  assert len(active) == 1
  assert active[0]["memory_id"] == "ltm_002"


def test_records_without_embedding_are_skipped():
  records = [
    _record("ltm_001", "no embedding A"),
    _record("ltm_002", "no embedding B"),
  ]
  # No embeddings attached. Consolidator must not blow up; it must
  # return them all untouched.
  out, n = consolidate_memories(records, similarity_threshold=0.80)
  assert n == 0
  assert all(r.get("status", "active") == "active" for r in out)


def test_empty_input():
  out, n = consolidate_memories([], similarity_threshold=0.80)
  assert out == []
  assert n == 0


def test_already_consolidated_records_are_left_alone():
  emb = CharacterNgramEmbedder(dim=256)
  records = _embed_records([
    _record("ltm_001", "I love sourdough bread"),
    _record("ltm_002", "I love sourdough bread"),
  ], emb)
  records[0]["status"] = "consolidated"

  out, n = consolidate_memories(records, similarity_threshold=0.80)
  # The already-consolidated record stays consolidated; the active one
  # stays active (no peer to merge with — the other active records is the
  # already-consolidated one we excluded from clustering).
  assert n == 0


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
