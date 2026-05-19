"""
AISM — Memory Consolidation
===========================

Periodic deterministic merge of near-duplicate memories using cosine
similarity over stored embeddings. The HLD calls for a consolidation module
that the v1 system was missing. This is its first cut.

What this does
--------------
- Walks the long-term memory store for a user.
- Computes pairwise cosine similarity between record embeddings.
- Finds clusters where every pair is above ``similarity_threshold``.
- Within each cluster, keeps one canonical record and marks the others with
  ``status="consolidated"``. Retrieval helpers in ``storage.py`` filter
  consolidated records out, so they no longer surface — but they are not
  deleted, so we keep an audit trail.
- The canonical record absorbs ``frequency`` from its consolidated peers,
  reflecting how often the underlying topic was raised across turns.

What this does NOT do
---------------------
- No LLM summarisation. R1 forbids LLM calls inside the AISM pipeline.
  Picking the longest utterance as canonical is a deliberately dumb
  heuristic; smarter selection is a follow-up.
- No clustering for records without embeddings. They are skipped.
- No cross-user merging. Per-user only.

Triggering
----------
``consolidate_memories`` is meant to be called periodically (e.g. every
N turns or on session boundary). It is idempotent: a second call right
after the first does nothing because there are no eligible duplicate pairs
left.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .embedder import cosine_similarity


def _select_canonical(records: List[Dict[str, Any]]) -> int:
  """Pick the index of the canonical record from a cluster.

  Heuristic: longest utterance wins (more information density). Tie-break
  by lowest memory_id (deterministic).
  """
  best_idx = 0
  best_key: Tuple[int, str] = (-1, "~")
  for i, r in enumerate(records):
    length = len(r.get("utterance", ""))
    mid = r.get("memory_id", "~")
    key = (length, "-" + mid)  # invert mid so smaller comes "larger"
    if key > best_key:
      best_key = key
      best_idx = i
  return best_idx


def _find_clusters(
  records: List[Dict[str, Any]],
  threshold: float,
) -> List[List[int]]:
  """Single-link agglomerative clustering on cosine similarity.

  Returns a list of clusters; each cluster is a list of indices into
  ``records``. Records without an ``embedding`` field are skipped (they
  appear in no cluster).
  """
  n = len(records)
  parent = list(range(n))

  def find(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a: int, b: int) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
      parent[ra] = rb

  for i in range(n):
    vi = records[i].get("embedding")
    if not vi:
      continue
    for j in range(i + 1, n):
      vj = records[j].get("embedding")
      if not vj:
        continue
      if len(vi) != len(vj):
        continue
      sim = cosine_similarity(vi, vj)
      if sim >= threshold:
        union(i, j)

  groups: Dict[int, List[int]] = {}
  for i in range(n):
    if not records[i].get("embedding"):
      continue
    root = find(i)
    groups.setdefault(root, []).append(i)

  return [g for g in groups.values() if len(g) >= 2]


def consolidate_memories(
  records: List[Dict[str, Any]],
  *,
  similarity_threshold: float = 0.80,
) -> Tuple[List[Dict[str, Any]], int]:
  """Consolidate near-duplicate memories.

  Args:
    records: long-term memory records as loaded from disk.
    similarity_threshold: cosine threshold to treat two records as
      duplicates. 0.80 is a reasonable starting value for
      ``CharacterNgramEmbedder``; for sentence-transformer embeddings,
      0.85+ is usually better.

  Returns:
    (updated_records, n_consolidated). The original list is not mutated.
    ``n_consolidated`` is the number of records flipped to
    ``status="consolidated"`` in this pass. Records without embeddings are
    passed through unchanged.
  """
  if not records:
    return list(records), 0

  # Only consider currently-active records for clustering. Already-consolidated
  # ones stay consolidated; we don't resurrect them.
  active_indices = [
    i for i, r in enumerate(records)
    if r.get("status", "active") == "active" and r.get("embedding")
  ]
  active_records = [records[i] for i in active_indices]

  clusters = _find_clusters(active_records, similarity_threshold)

  # Make a deep-ish copy of records (shallow per-record but we won't share
  # mutations on individual fields with the caller).
  out = [dict(r) for r in records]
  consolidated = 0
  now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

  for cluster in clusters:
    # cluster contains indices into active_records; map back to ``out``.
    real_indices = [active_indices[ci] for ci in cluster]
    cluster_recs = [out[i] for i in real_indices]
    canonical_local = _select_canonical(cluster_recs)
    canonical_idx = real_indices[canonical_local]

    extra_freq = 0
    for ri in real_indices:
      if ri == canonical_idx:
        continue
      extra_freq += int(out[ri].get("frequency", 0) or 0)
      out[ri]["status"] = "consolidated"
      out[ri]["consolidated_into"] = out[canonical_idx].get("memory_id", "")
      out[ri]["consolidated_at"] = now_iso
      consolidated += 1

    # Roll absorbed frequency into the canonical record so its salience
    # reflects how often the topic came up across the cluster.
    out[canonical_idx]["frequency"] = (
      int(out[canonical_idx].get("frequency", 0) or 0) + extra_freq
    )
    out[canonical_idx]["last_seen"] = now_iso

  return out, consolidated


def maybe_consolidate(
  storage: Any,
  user_id: str,
  *,
  every_n_turns: int = 20,
  turn_count: Optional[int] = None,
  force: bool = False,
  similarity_threshold: float = 0.80,
) -> int:
  """Consolidate if the trigger condition is met. Returns # consolidated.

  Cheap to call every turn; no-ops unless ``turn_count % every_n_turns == 0``
  or ``force=True``.
  """
  if not force:
    if turn_count is None or turn_count == 0:
      return 0
    if turn_count % every_n_turns != 0:
      return 0

  records = storage.load_long_term_memory(user_id)
  updated, n = consolidate_memories(
    records, similarity_threshold=similarity_threshold)
  if n > 0:
    storage.save_long_term_memory(user_id, updated)
  return n
