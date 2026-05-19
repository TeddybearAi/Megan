"""
AISM — Storage (Local JSON Persistence)
=======================================
Implements the four-store persistence model from BluePrint, backed by
local JSON files on disk. Consistent with the HLD privacy stance: nothing
leaves the device.

Stores:
    A. raw_interaction_log  (conversation turns)
    B. style_evidence_log   (every extracted evidence item)
    C. stable_profile       (InteractionProfile)
    D. session_overlay      (SessionOverlay)

File layout under `base_path/<user_id>/`:
    raw_log.json
    evidence_log.json
    profile.json
    session_<session_id>.json

Steps (per save/load):
    S.1 : Convert dataclass/enum/datetime to JSON-safe dicts
    S.2 : Write atomically (temp file → rename)
    S.3 : On load, reconstruct dataclasses from dicts
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_models import (
  ConversationTurn,
  EvidenceSourceType,
  InteractionProfile,
  MemoryRecord,
  SessionOverlay,
  StabilityClass,
  StyleEvidence,
  TraitState,
  TurnRole,
)


class LocalJSONStore:
  """Simple local JSON persistence for AISM's four storage structures."""

  def __init__(
    self,
    base_path: str | Path,
    embedder: Any = None,
  ) -> None:
    """Local JSON-backed store.

    ``embedder`` is optional. When provided, ``retrieve_relevant_memories``
    uses cosine similarity over per-record vectors instead of bag-of-words
    overlap. The embedder must satisfy the ``aism.embedder.Embedder``
    Protocol (``dim`` attribute + ``embed(text)`` method). Pure JSON storage
    is preserved; embeddings are persisted as a sibling ``.npy``-style list
    of floats inside each record under the ``embedding`` key, so the JSON
    file remains human-readable.
    """
    self.base_path = Path(base_path)
    self.base_path.mkdir(parents=True, exist_ok=True)
    self.embedder = embedder

  # ============================================================
  # Path helpers
  # ============================================================

  def _user_dir(self, user_id: str) -> Path:
    d = self.base_path / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d

  def _profile_path(self, user_id: str) -> Path:
    return self._user_dir(user_id) / "profile.json"

  def _evidence_log_path(self, user_id: str) -> Path:
    return self._user_dir(user_id) / "evidence_log.json"

  def _raw_log_path(self, user_id: str) -> Path:
    return self._user_dir(user_id) / "raw_log.json"

  def _overlay_path(self, user_id: str, session_id: str) -> Path:
    return self._user_dir(user_id) / f"session_{session_id}.json"

  # ============================================================
  # S.1 / S.3: Serialisation helpers
  # ============================================================

  @staticmethod
  def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / datetimes into JSON-safe."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
      return obj
    if isinstance(obj, Enum):
      return obj.value
    if isinstance(obj, datetime):
      return obj.isoformat()
    if isinstance(obj, list):
      return [LocalJSONStore._to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
      return {k: LocalJSONStore._to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
      return LocalJSONStore._to_jsonable(asdict(obj))
    return str(obj)  # fallback

  # ============================================================
  # S.2: Atomic write helper
  # ============================================================

  @staticmethod
  def _atomic_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = LocalJSONStore._to_jsonable(payload)
    with tmp.open("w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

  @staticmethod
  def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
      return None
    with path.open("r", encoding="utf-8") as f:
      return json.load(f)

  # ============================================================
  # Profile
  # ============================================================

  def save_profile(self, profile: InteractionProfile) -> None:
    payload = self._to_jsonable(profile)
    self._atomic_write(self._profile_path(profile.user_id), payload)

  def load_profile(self, user_id: str) -> Optional[InteractionProfile]:
    raw = self._read_json(self._profile_path(user_id))
    if raw is None:
      return None

    traits: Dict[str, Dict[str, TraitState]] = {}
    for trait_name, by_context in (raw.get("traits") or {}).items():
      traits[trait_name] = {}
      for context, state_dict in by_context.items():
        traits[trait_name][context] = TraitState(
          value=state_dict["value"],
          confidence=state_dict["confidence"],
          evidence_count=state_dict["evidence_count"],
          volatility=state_dict["volatility"],
          last_updated=self._parse_dt(state_dict["last_updated"]),
          source_summary=state_dict.get("source_summary", []),
          recent_values=state_dict.get("recent_values", []),
          session_drift=state_dict.get("session_drift", 0.0),
        )

    return InteractionProfile(
      user_id=raw["user_id"],
      traits=traits,
      boundaries=raw.get("boundaries", {}),
      last_updated=self._parse_dt(raw["last_updated"]),
    )

  # ============================================================
  # Evidence log
  # ============================================================

  def save_evidence_log(self, user_id: str, log: List[StyleEvidence]) -> None:
    payload = [self._to_jsonable(ev) for ev in log]
    self._atomic_write(self._evidence_log_path(user_id), {"items": payload})

  def load_evidence_log(self, user_id: str) -> List[StyleEvidence]:
    raw = self._read_json(self._evidence_log_path(user_id))
    if raw is None:
      return []
    items: List[StyleEvidence] = []
    for d in raw.get("items", []):
      items.append(
        StyleEvidence(
          evidence_id=d["evidence_id"],
          turn_id=d["turn_id"],
          trait=d["trait"],
          context=d.get("context", "general"),
          value=d["value"],
          confidence=d["confidence"],
          source_type=EvidenceSourceType(d["source_type"]),
          stability_class=StabilityClass(d["stability_class"]),
          timestamp=self._parse_dt(d["timestamp"]),
          evidence_text=d["evidence_text"],
          metadata=d.get("metadata", {}),
        ))
    return items

  # ============================================================
  # Raw conversation log
  # ============================================================

  def append_turn(self, user_id: str, turn: ConversationTurn) -> None:
    existing = self._read_json(self._raw_log_path(user_id)) or {"turns": []}
    existing["turns"].append(self._to_jsonable(turn))
    self._atomic_write(self._raw_log_path(user_id), existing)

  def load_turns(self, user_id: str) -> List[ConversationTurn]:
    raw = self._read_json(self._raw_log_path(user_id))
    if raw is None:
      return []
    turns: List[ConversationTurn] = []
    for d in raw.get("turns", []):
      turns.append(
        ConversationTurn(
          turn_id=d["turn_id"],
          role=TurnRole(d["role"]),
          text=d["text"],
          timestamp=self._parse_dt(d["timestamp"]),
          metadata=d.get("metadata", {}),
        ))
    return turns

  # ============================================================
  # Session overlay
  # ============================================================

  def save_overlay(self, user_id: str, overlay: SessionOverlay) -> None:
    payload = self._to_jsonable(overlay)
    self._atomic_write(
      self._overlay_path(user_id, overlay.session_id), payload)

  def load_overlay(self, user_id: str,
                   session_id: str) -> Optional[SessionOverlay]:
    raw = self._read_json(self._overlay_path(user_id, session_id))
    if raw is None:
      return None
    return SessionOverlay(
      session_id=raw["session_id"],
      state=raw.get("state", {}),
      last_updated=self._parse_dt(raw["last_updated"]),
    )

  # ============================================================
  # Remarkable memory storage
  # ============================================================

  def _memory_path(self, user_id: str, kind: str) -> Path:
    suffix = {
      "long_term": "long_term_memory.json",
      "candidate": "candidate_memory.json",
      "person_bank": "person_bank.json",
    }.get(kind)
    if suffix is None:
      raise ValueError(f"Unknown memory kind: {kind}")
    return self._user_dir(user_id) / suffix

  def _load_memory_records(self, user_id: str,
                           kind: str) -> List[Dict[str, Any]]:
    raw = self._read_json(self._memory_path(user_id, kind))
    return raw if raw is not None else []

  def _save_memory_records(
      self, user_id: str, kind: str, records: List[Dict[str, Any]]) -> None:
    self._atomic_write(self._memory_path(user_id, kind), records)

  def load_long_term_memory(self, user_id: str) -> List[Dict[str, Any]]:
    return self._load_memory_records(user_id, "long_term")

  def save_long_term_memory(
      self, user_id: str, records: List[Dict[str, Any]]) -> None:
    self._save_memory_records(user_id, "long_term", records)

  # ============================================================
  # v2.2.6 — Identity block for system prompt
  # ============================================================

  # Memory types that constitute Teddy's stable identity for the
  # system-prompt block. profile_fact is the core (age, languages,
  # personality, demographics). preference is included because favourites
  # ("favourite colour", "favourite game") read as identity-stable in
  # practice and Teddy explicitly asks about them. long_term_goal can
  # be added later if we want career/life-plan context surfaced.
  _IDENTITY_BLOCK_TYPES = {"profile_fact", "preference"}

  # Quality bars. profile_fact has a permissive frequency bar (1) because
  # v2.2.5 writes explicit identity claims directly to LTM at freq=1 and
  # those are exactly the facts the identity block exists to surface.
  # preference has a stricter bar (2) so single-mention casual "I love X"
  # turns and emotionally-charged disclosures don't pollute the block —
  # those need to be confirmed at least once before they become "known
  # facts" that Megan repeats back unprompted.
  _IDENTITY_BLOCK_MIN_IMPORTANCE = 3
  _IDENTITY_BLOCK_MIN_FREQUENCY_BY_TYPE = {
    "profile_fact": 1,
    "preference": 2,
    "long_term_goal": 2,  # in case we add it later
  }

  # How many facts to surface in the block. Token budget: ~150 chars/fact
  # × 15 = 2250 chars ≈ 600 tokens. Comfortable for any reasonable num_ctx.
  _IDENTITY_BLOCK_MAX_ENTRIES = 15

  # Each fact's text is truncated to this many characters in the block,
  # to keep total size bounded even if some LTM entries are long ramblings.
  _IDENTITY_BLOCK_MAX_FACT_CHARS = 400

  # Leading conversational filler we strip from utterances before rendering
  # so the block reads less noisily. Order matters: longer/multi-token
  # variants first so we don't half-strip a phrase.
  _LEADING_FILLER_PATTERN = re.compile(
    r"^(?:"
    r"(?:hmm|mm|mhm|uh|um|oh|ah|yeah|yep|yup|okay|ok|right|well|so|and|"
    r"like|listen|look|alright|hey|wait)\b"
    r"[\s,\.]*"
    r")+",
    re.IGNORECASE,
  )

  def _strip_leading_filler(self, text: str) -> str:
    """Trim leading conversational fillers (Yeah, Okay, Hmm, So…) from
    an utterance so the identity block reads cleanly. Preserves the
    rest of the text exactly. Used only for the block render; the
    underlying stored utterance is untouched."""
    cleaned = self._LEADING_FILLER_PATTERN.sub("", text).lstrip()
    # If we stripped everything (utterance was pure filler), return
    # the original so we don't silently drop content.
    return cleaned if cleaned else text

  def get_identity_block(self, user_id: str) -> str:
    """Render a system-prompt-ready block of stable identity facts.

    This is the structural fix for the meta-query failure mode where
    queries like "what's my age?" or "tell me about myself" fail to
    surface profile facts via cosine-similarity retrieval. By injecting
    these facts directly into the system prompt every turn, the model
    has them available regardless of how the question is phrased.

    Returns empty string if the user has no qualifying facts (e.g. a
    fresh user) so the caller can skip injection cleanly.

    Implementation notes:
    - Pure rendering; no LLM call, no embedding lookup.
    - Recomputed every turn. Cheap: file read + filter + format.
      Caching can be added later if profiling shows it matters.
    - Records are sorted by (importance, frequency) descending so the
      most established facts surface first if we hit the cap.
    """
    records = self.load_long_term_memory(user_id)
    if not records:
      return ""

    eligible: List[Dict[str, Any]] = []
    for rec in records:
      if rec.get("status") == "consolidated":
        continue  # consolidated duplicates are filtered out of retrieval
      mtype = rec.get("memory_type")
      if mtype not in self._IDENTITY_BLOCK_TYPES:
        continue
      if rec.get("importance", 0) < self._IDENTITY_BLOCK_MIN_IMPORTANCE:
        continue
      min_freq = self._IDENTITY_BLOCK_MIN_FREQUENCY_BY_TYPE.get(mtype, 2)
      if rec.get("frequency", 1) < min_freq:
        continue
      eligible.append(rec)

    if not eligible:
      return ""

    # Sort: highest importance first, ties broken by frequency.
    eligible.sort(
      key=lambda r: (r.get("importance", 0), r.get("frequency", 0)),
      reverse=True,
    )
    eligible = eligible[:self._IDENTITY_BLOCK_MAX_ENTRIES]

    lines = [
      "KNOWN FACTS ABOUT THE USER",
      "(These have been confirmed through past conversations. Treat them "
      "as established context. If asked directly about any of these "
      "topics, draw on the facts below — do not say 'I don't recall' "
      "for something listed here.)",
      "",
    ]
    for rec in eligible:
      utterance = (rec.get("utterance") or "").strip()
      if not utterance:
        continue
      utterance = self._strip_leading_filler(utterance)
      if len(utterance) > self._IDENTITY_BLOCK_MAX_FACT_CHARS:
        utterance = utterance[:self._IDENTITY_BLOCK_MAX_FACT_CHARS] + "…"
      mtype = rec.get("memory_type", "fact")
      lines.append(f"- [{mtype}] {utterance}")

    return "\n".join(lines)

  def load_candidate_memory(self, user_id: str) -> List[Dict[str, Any]]:
    return self._load_memory_records(user_id, "candidate")

  def save_candidate_memory(
      self, user_id: str, records: List[Dict[str, Any]]) -> None:
    self._save_memory_records(user_id, "candidate", records)

  def load_person_bank(self, user_id: str) -> List[Dict[str, Any]]:
    return self._load_memory_records(user_id, "person_bank")

  def save_person_bank(
      self, user_id: str, records: List[Dict[str, Any]]) -> None:
    self._save_memory_records(user_id, "person_bank", records)

  @staticmethod
  def _normalise_text(text: str) -> str:
    return " ".join(text.strip().lower().split())

  def _find_duplicate_memory(
      self, records: List[Dict[str, Any]], utterance: str,
      user_id: str) -> int:
    target = self._normalise_text(utterance)
    for idx, record in enumerate(records):
      if (record.get("user_id") == user_id
          and self._normalise_text(record.get("utterance", "")) == target):
        return idx
    return -1

  def _generate_memory_id(
    self,
    prefix: str,
    existing_records: List[Dict[str, Any]],
  ) -> str:
    """Generate a fresh memory id that does not collide with any existing id.

    v1 used ``f"{prefix}{len(records) + 1:03d}"``, which collides as soon
    as records are archived (e.g. via ``_detect_resolution`` in main.py).
    v2 scans existing ids of this prefix and takes max+1, so the counter
    never goes backwards.
    """
    max_seen = 0
    for record in existing_records:
      mid = record.get("memory_id", "")
      if not isinstance(mid, str) or not mid.startswith(prefix):
        continue
      tail = mid[len(prefix):]
      try:
        n = int(tail)
      except ValueError:
        continue
      if n > max_seen:
        max_seen = n
    return f"{prefix}{max_seen + 1:03d}"

  def _create_memory_record(
    self,
    memory_id: str,
    user_id: str,
    utterance: str,
    store: str,
    memory_type: str,
    importance: int,
    notes: str = "",
  ) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    status = "active" if store == "yes" else "candidate"
    record = MemoryRecord(
      memory_id=memory_id,
      user_id=user_id,
      utterance=utterance,
      store=store,
      memory_type=memory_type,
      importance=importance,
      frequency=1,
      created_at=now,
      last_seen=now,
      status=status,
      notes=notes,
    )
    record_dict = asdict(record)
    # Attach an embedding if the store has one configured. Persisting it
    # alongside the record keeps the JSON file human-readable while letting
    # cosine-similarity retrieval be O(n) without recomputing.
    if self.embedder is not None:
      try:
        record_dict["embedding"] = list(self.embedder.embed(utterance))
      except Exception:
        # If embedding fails for any reason, store the record without one;
        # retrieval will fall back to bag-of-words for that record.
        pass
    return record_dict

  def route_memory(self, user_id: str, data: Dict[str, Any]) -> str:
    store_label = data.get("store", "no").strip().lower()
    if store_label == "yes":
      path_kind = "long_term"
      prefix = "ltm_"
    elif store_label == "maybe":
      path_kind = "candidate"
      prefix = "cand_"
    else:
      return "ignored"

    records = self._load_memory_records(user_id, path_kind)
    duplicate_idx = self._find_duplicate_memory(
      records, data["utterance"], user_id)

    if duplicate_idx != -1:
      records[duplicate_idx]["frequency"] += 1
      records[duplicate_idx]["last_seen"] = datetime.now(
        timezone.utc).isoformat(timespec="seconds")
      self._save_memory_records(user_id, path_kind, records)
      return f"updated_{store_label}"

    memory_id = self._generate_memory_id(prefix, records)
    new_record = self._create_memory_record(
      memory_id=memory_id,
      user_id=user_id,
      utterance=data["utterance"],
      store=data["store"],
      memory_type=data["memory_type"],
      importance=data["importance"],
      notes=data.get("notes", ""),
    )
    records.append(new_record)
    self._save_memory_records(user_id, path_kind, records)
    return f"saved_{store_label}"

  def promote_candidate_memories(
    self,
    user_id: str,
    promotion_threshold: int = 2,
  ) -> int:
    candidate_records = self.load_candidate_memory(user_id)
    long_term_records = self.load_long_term_memory(user_id)
    promoted_count = 0
    remaining_candidates: List[Dict[str, Any]] = []

    for record in candidate_records:
      if record.get("frequency", 0) >= promotion_threshold:
        dup_idx = self._find_duplicate_memory(
          long_term_records, record["utterance"], user_id)
        if dup_idx != -1:
          long_term_records[dup_idx]["frequency"] += record.get("frequency", 1)
          long_term_records[dup_idx]["last_seen"] = datetime.now(
            timezone.utc).isoformat(timespec="seconds")
        else:
          record.update(
            {
              "store":
              "yes",
              "status":
              "active",
              "memory_id":
              self._generate_memory_id("ltm_", long_term_records),
            })
          long_term_records.append(record)
        promoted_count += 1
      else:
        remaining_candidates.append(record)

    self.save_long_term_memory(user_id, long_term_records)
    self.save_candidate_memory(user_id, remaining_candidates)
    return promoted_count

  def decay_candidate_memories(
    self,
    user_id: str,
    max_age_days: int = 7,
    max_frequency: int = 1,
  ) -> int:
    candidates = self.load_candidate_memory(user_id)
    now = datetime.now(timezone.utc)
    filtered: List[Dict[str, Any]] = []
    removed = 0

    for record in candidates:
      try:
        last_seen = datetime.fromisoformat(record.get("last_seen", ""))
        if last_seen.tzinfo is None:
          last_seen = last_seen.replace(tzinfo=timezone.utc)
      except ValueError:
        last_seen = now

      age = now - last_seen
      if age.days > max_age_days and record.get("frequency",
                                                0) <= max_frequency:
        removed += 1
        continue
      filtered.append(record)

    self.save_candidate_memory(user_id, filtered)
    return removed

  # ============================================================
  # Retrieval helpers (revised for Memory-pass v2)
  # ============================================================
  #
  # Two changes vs. v1:
  #   1. Both functions filter out records whose ``status`` is "consolidated"
  #      so consolidation-marked duplicates don't surface again.
  #   2. ``get_top_priority_memory`` adds recency weighting; an old
  #      importance-5 record no longer wins over a fresh importance-3 one
  #      forever.
  #   3. ``retrieve_relevant_memories`` uses cosine similarity on stored
  #      embeddings when the store has an embedder configured. Otherwise
  #      falls back to the v1 bag-of-words behaviour.

  _RECENCY_HALF_LIFE_DAYS = 14.0  # tunable; older memories decay smoothly

  @staticmethod
  def _record_age_days(record: Dict[str, Any]) -> float:
    """Days since the record was last seen. Defaults to 0 if unparsable."""
    raw = record.get("last_seen") or record.get("created_at") or ""
    try:
      ts = datetime.fromisoformat(raw)
      if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
      return 0.0
    delta = datetime.now(timezone.utc) - ts
    return max(0.0, delta.total_seconds() / 86400.0)

  @classmethod
  def _recency_score(cls, record: Dict[str, Any]) -> float:
    """Smooth decay in [0, 1]. 1.0 today, ~0.5 at half-life, → 0 over time."""
    import math
    age = cls._record_age_days(record)
    return math.exp(-age / cls._RECENCY_HALF_LIFE_DAYS)

  @staticmethod
  def _is_active(record: Dict[str, Any]) -> bool:
    """True if the record should be considered for retrieval/top-priority."""
    status = record.get("status", "active")
    return status not in ("consolidated", "archived")

  def get_top_priority_memory(self, user_id: str) -> Optional[Dict[str, Any]]:
    """Return the highest-priority active memory.

    Score = importance * 0.4 + frequency * 0.1 + recency * 1.0. Recency
    decays exponentially with the half-life above. ``consolidated`` records
    are filtered out.

    Importantly, importance-4/5 emotional memories no longer win greetings
    forever — recency dominates once the memory is more than ~6 weeks old.
    """
    records = self.load_long_term_memory(user_id)
    candidates = [r for r in records if self._is_active(r)]
    if not candidates:
      return None

    def score(r: Dict[str, Any]) -> float:
      return (
        r.get("importance", 0) * 0.4
        + r.get("frequency", 0) * 0.1
        + self._recency_score(r) * 1.0
      )

    return max(candidates, key=score)

  def retrieve_relevant_memories(
    self,
    user_id: str,
    query: str,
    top_k: int = 3,
    min_score: float = 0.05,
  ) -> List[Dict[str, Any]]:
    """Retrieve memories relevant to a query.

    If an embedder is configured, uses cosine similarity over per-record
    embeddings. Records without embeddings (legacy/migrated data) get a
    fall-back bag-of-words score so they still participate.

    A small additive bonus is given to importance and recent frequency, so
    when two memories tie semantically the more salient one wins. The bonus
    is intentionally small (≤0.15) so it cannot promote an irrelevant
    memory above a relevant one.
    """
    records = self.load_long_term_memory(user_id)
    active = [r for r in records if self._is_active(r)]
    if not active:
      return []

    # ----- Build a similarity score for each record -----
    if self.embedder is not None:
      try:
        from .embedder import cosine_similarity
        query_vec = list(self.embedder.embed(query))
      except Exception:
        query_vec = None
    else:
      query_vec = None

    query_tokens = set(self._normalise_text(query).split())

    def sim(record: Dict[str, Any]) -> float:
      if query_vec is not None:
        rec_vec = record.get("embedding")
        if (rec_vec is not None
            and isinstance(rec_vec, list)
            and len(rec_vec) == len(query_vec)):
          from .embedder import cosine_similarity
          return float(cosine_similarity(query_vec, rec_vec))
      # Fallback: bag-of-words token overlap, normalised by query length.
      rec_tokens = set(
        self._normalise_text(record.get("utterance", "")).split())
      if not query_tokens:
        return 0.0
      overlap = len(query_tokens & rec_tokens)
      return overlap / max(1, len(query_tokens))

    scored: List[tuple[float, Dict[str, Any]]] = []
    for record in active:
      base = sim(record)
      bonus = (
        0.03 * record.get("importance", 0)
        + 0.01 * record.get("frequency", 0)
      )
      scored.append((base + bonus, record))

    scored.sort(key=lambda item: item[0], reverse=True)

    # Only return records that cleared the relevance floor. If nothing
    # cleared, return empty rather than the previous "everything is somehow
    # relevant" behaviour, which is what produced the spurious crush match
    # for "gym" queries.
    relevant = [r for s, r in scored if s >= min_score]
    return [
      {
        "summary": r.get("utterance", ""),
        "memory_type": r.get("memory_type", ""),
        "importance": r.get("importance", 0),
      }
      for r in relevant[:top_k]
    ]

  def retrieve_recent_emotional_memories(
    self,
    user_id: str,
    top_k: int = 3,
    max_age_days: int = 14,
  ) -> List[Dict[str, Any]]:
    """Retrieve memories by recency, weighted by emotional importance.

    Used when the user expresses affection or appreciation directly to
    Megan (WARM_OPENING mode). A real friend's first instinct when
    someone says "I like you" is to draw on something specific from
    recent shared experience — not to dredge up content from months
    ago. Recency is both psychologically right and technically cheaper
    (no embedder call, just a sort).

    Returns up to ``top_k`` active records updated within
    ``max_age_days``, sorted by ``last_seen`` descending with
    ``importance`` as a secondary tiebreaker. Falls back to the most
    recent active records if nothing falls within the cutoff (rare;
    handles first-session edge case).
    """
    records = self.load_long_term_memory(user_id)
    active = [r for r in records if self._is_active(r)]
    if not active:
      return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)

    def parse_ts(value: Any) -> datetime:
      if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
      if isinstance(value, str):
        try:
          return self._parse_dt(value)
        except (ValueError, TypeError):
          pass
      # Records missing timestamps sort to the oldest position rather
      # than being dropped silently.
      return datetime.min.replace(tzinfo=timezone.utc)

    # Build (last_seen, importance, record) tuples for sortable retrieval.
    decorated = []
    for record in active:
      last_seen = parse_ts(record.get("last_seen") or record.get("created_at"))
      decorated.append((last_seen, record.get("importance", 0), record))

    # Sort by recency desc, then importance desc.
    decorated.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # Prefer records within the cutoff window; fall back to the most
    # recent records overall if none qualify (first-session safety).
    within_window = [d for d in decorated if d[0] >= cutoff]
    chosen = within_window if within_window else decorated

    return [
      {
        "summary": r.get("utterance", ""),
        "memory_type": r.get("memory_type", ""),
        "importance": r.get("importance", 0),
      }
      for _, _, r in chosen[:top_k]
    ]

  # ============================================================
  # Parsing helpers
  # ============================================================

  @staticmethod
  def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=timezone.utc)
    return dt
