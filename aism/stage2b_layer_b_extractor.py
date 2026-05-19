"""
AISM — STAGE 2 LAYER B: Embedding-Assisted Semantic Extractor
=============================================================
Opt-in semantic matching on top of Layer A's lexicon rules. When Layer A
misses a phrasing the user used (e.g. "keep it tight" for verbosity=short),
Layer B embeds candidate n-grams from the user turn and compares them
against embeddings of the lexicon cue exemplars.

BluePrint v2.1 reference: Section 3 — Stage 2, Layer B.

Steps:
    Step 2B.1 : At construction, encode every cue phrase from TRAIT_LEXICONS
                into a vector. Store (trait, direction, value, source_type,
                vector) tuples.
    Step 2B.2 : At extraction time, generate candidate phrases from the user
                turn (n-gram windows over words).
    Step 2B.3 : For each candidate, find the nearest cue (max cosine similarity).
    Step 2B.4 : If similarity ≥ threshold, emit IMPLICIT_PATTERN evidence
                with confidence derived from the similarity score.
    Step 2B.5 : Dedup so Layer B doesn't re-emit evidence Layer A already found.

Constraint: Uses an Embedder (non-generative neural OK). No LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from .data_models import (
  ConversationTurn,
  EvidenceSourceType,
  StabilityClass,
  StyleEvidence,
)
from .embedder import Embedder, cosine_similarity
from .session_context import SessionContext
from .stage2_extraction import TRAIT_LEXICONS

# ============================================================
# Internal: encoded cue entry
# ============================================================


@dataclass
class _EncodedCue:
  trait: str
  direction: str
  cue_text: str
  value: object  # scalar float or categorical string
  source_type: EvidenceSourceType
  vector: List[float]


class LayerBExtractor:
  """
    Embedding-assisted extractor. Construct once, reuse across turns — the
    per-cue embeddings are computed at init and cached.
    """

  # --- Step 2B.2: candidate-phrase generation parameters ---
  CANDIDATE_WORD_NGRAMS = (3, 4, 5, 6)
  MIN_CANDIDATE_WORDS = 3
  MAX_CANDIDATES_PER_TURN = 60

  # --- Step 2B.4: similarity threshold ---
  SIMILARITY_THRESHOLD = 0.72

  # --- Step 2B.4: confidence mapping ---
  # Layer B is always implicit_pattern (semantic inference, not literal
  # match). Its confidence tracks how close the match was.
  #   similarity 0.72 → confidence ~0.55
  #   similarity 1.00 → confidence ~0.75
  # This keeps Layer B below Layer A's 0.90 so Stage 3's gates and
  # log-linear scorer naturally down-weight it.
  CONFIDENCE_AT_THRESHOLD = 0.55
  CONFIDENCE_AT_ONE = 0.75

  _WORD_SPLIT = re.compile(r"[^\w']+")

  def __init__(self, embedder: Embedder) -> None:
    self.embedder = embedder
    self.encoded_cues: List[_EncodedCue] = []

    # --- Step 2B.1: Pre-encode cue exemplars ---
    for trait, directions in TRAIT_LEXICONS.items():
      for direction, spec in directions.items():
        for raw_pattern in spec["patterns"]:
          # Strip regex syntax for a cleaner natural-language exemplar.
          cue_text = self._regex_to_exemplar(raw_pattern)
          if not cue_text:
            continue
          vector = self.embedder.embed(cue_text)
          self.encoded_cues.append(
            _EncodedCue(
              trait=trait,
              direction=direction,
              cue_text=cue_text,
              value=spec["value"],
              source_type=spec["source_type"],
              vector=vector,
            ))

  # ============================================================
  # Public entry point
  # ============================================================

  def extract(
    self,
    turn: ConversationTurn,
    session_context: SessionContext,
    layer_a_evidences: Sequence[StyleEvidence],
  ) -> List[StyleEvidence]:
    """
        Produce Layer B evidence not already covered by Layer A.

        `layer_a_evidences` is the output of Layer A for the same turn; we use
        it to dedup so Layer B doesn't echo matches Layer A already made.
        """
    # --- Step 2B.5: dedup keys from Layer A ---
    layer_a_keys = {
      self._dedup_key(ev.trait, ev.value)
      for ev in layer_a_evidences
    }

    # --- Step 2B.2: candidate phrases from the turn ---
    candidates = self._generate_candidates(turn.text)
    if not candidates:
      return []

    candidate_vecs = self.embedder.embed_batch(candidates)

    # --- Step 2B.3 + 2B.4: nearest-cue match per candidate ---
    best_per_key: dict = {}  # dedup to best match per (trait, value)
    for cand_text, cand_vec in zip(candidates, candidate_vecs):
      best_cue: _EncodedCue | None = None
      best_sim = -1.0
      for cue in self.encoded_cues:
        sim = cosine_similarity(cand_vec, cue.vector)
        if sim > best_sim:
          best_sim = sim
          best_cue = cue
      if best_cue is None or best_sim < self.SIMILARITY_THRESHOLD:
        continue

      key = self._dedup_key(best_cue.trait, best_cue.value)
      if key in layer_a_keys:
        continue
      incumbent = best_per_key.get(key)
      if incumbent is None or best_sim > incumbent["similarity"]:
        best_per_key[key] = {
          "cue": best_cue,
          "similarity": best_sim,
          "candidate_text": cand_text,
        }

    # --- Step 2B.4 (finish): emit evidence ---
    evidences: List[StyleEvidence] = []
    for key, match in best_per_key.items():
      cue: _EncodedCue = match["cue"]
      sim = match["similarity"]
      confidence = self._similarity_to_confidence(sim)
      evidences.append(
        StyleEvidence(
          evidence_id=f"evB-{turn.turn_id}-{cue.trait}-{cue.direction}",
          turn_id=turn.turn_id,
          trait=cue.trait,
          context=session_context.current_topic,
          value=cue.value,
          confidence=round(confidence, 4),
          # Layer B always emits as implicit_pattern — semantic inference,
          # not an exact lexicon match.
          source_type=EvidenceSourceType.IMPLICIT_PATTERN,
          stability_class=StabilityClass.CANDIDATE_STABLE,
          timestamp=turn.timestamp,
          evidence_text=turn.text[:200],
          metadata={
            "layer": "B",
            "matched_cue": cue.cue_text,
            "candidate_phrase": match["candidate_text"],
            "similarity": round(sim, 4),
          },
        ))
    return evidences

  # ============================================================
  # Helpers
  # ============================================================

  @staticmethod
  def _dedup_key(trait: str, value) -> tuple:
    if isinstance(value, (int, float)):
      return (trait, round(float(value), 2))
    return (trait, value)

  @staticmethod
  def _similarity_to_confidence(sim: float) -> float:
    """Linear interpolation between CONFIDENCE_AT_THRESHOLD and CONFIDENCE_AT_ONE."""
    t = (sim - LayerBExtractor.SIMILARITY_THRESHOLD) / (
      1.0 - LayerBExtractor.SIMILARITY_THRESHOLD)
    t = max(0.0, min(1.0, t))
    return (
      LayerBExtractor.CONFIDENCE_AT_THRESHOLD + t * (
        LayerBExtractor.CONFIDENCE_AT_ONE -
        LayerBExtractor.CONFIDENCE_AT_THRESHOLD))

  @classmethod
  def _regex_to_exemplar(cls, pattern: str) -> str:
    """
        Convert a regex cue into a natural-language exemplar for embedding.
        Strips common regex syntax and picks the first alternative.
        """
    s = pattern
    # Strip word boundaries and anchors.
    s = s.replace(r"\b", " ").replace(r"\s+", " ").replace(r"\s*", " ")
    s = s.replace(r"\.", ".").replace(r"\?", "")
    # Strip character classes we don't care about.
    s = re.sub(r"\[[^\]]+\]", "", s)
    # Strip optional-group syntax.
    s = re.sub(r"\(\?:([^)]+)\)\??", r"\1", s)
    # In alternations (a|b|c), just take the first alternative.
    s = re.sub(r"\(([^)]+)\)\??", lambda m: m.group(1).split("|")[0], s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s

  @classmethod
  def _generate_candidates(cls, text: str) -> List[str]:
    """
        Word-level n-gram windows over the turn. Skips empty/too-short
        candidates and caps total count to keep per-turn cost bounded.
        """
    words = [w for w in cls._WORD_SPLIT.split(text) if w]
    if len(words) < cls.MIN_CANDIDATE_WORDS:
      # Whole short turn is the only candidate.
      return [" ".join(words)] if words else []

    candidates: List[str] = []
    seen = set()
    for n in cls.CANDIDATE_WORD_NGRAMS:
      if len(words) < n:
        continue
      for i in range(len(words) - n + 1):
        phrase = " ".join(words[i:i + n]).lower()
        if phrase in seen:
          continue
        seen.add(phrase)
        candidates.append(phrase)
        if len(candidates) >= cls.MAX_CANDIDATES_PER_TURN:
          return candidates
    return candidates
