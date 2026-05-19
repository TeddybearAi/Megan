"""
AISM — Embedder Abstraction
===========================
Pluggable embedding interface used by:
  - Stage 2 Layer B (semantic matching against lexicon cues)
  - Idiolect Drift monitor (tracking response-voice drift over time)

BluePrint v2.1 reference: Section 2 (R2) — small non-generative neural
components are permitted. The embedder is the ONLY neural component AISM
uses at runtime.

Pass 2 ships two implementations:
    CharacterNgramEmbedder   — deterministic, zero external deps, pure Python
    SentenceTransformerEmbedder — optional, requires sentence-transformers

Both satisfy the Embedder Protocol, so the rest of the system doesn't care
which one is wired in.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import List, Protocol, Sequence

# ============================================================
# Protocol
# ============================================================


class Embedder(Protocol):
  """
    Produces a fixed-length float vector for a text string.
    Must be deterministic: same input → same output.
    """
  dim: int

  def embed(self, text: str) -> List[float]:
    ...

  def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
    return [self.embed(t) for t in texts]


# ============================================================
# Utility: cosine similarity
# ============================================================


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
  if len(a) != len(b):
    raise ValueError(f"Vector dim mismatch: {len(a)} vs {len(b)}")
  dot = sum(x * y for x, y in zip(a, b))
  na = math.sqrt(sum(x * x for x in a))
  nb = math.sqrt(sum(y * y for y in b))
  if na == 0.0 or nb == 0.0:
    return 0.0
  return dot / (na * nb)


# ============================================================
# CharacterNgramEmbedder — deterministic, no external deps
# ============================================================


class CharacterNgramEmbedder:
  """
    A simple hashed character-n-gram embedder. Works well enough for
    short-phrase semantic similarity in English without requiring any
    downloaded model. Uses the hashing trick to keep dim fixed.

    Steps:
        E.1 : Lowercase + strip; pad edges with '_' for boundary n-grams
        E.2 : Generate character n-grams for n in NGRAM_RANGE
        E.3 : Hash each n-gram modulo dim, count occurrences
        E.4 : L2-normalise the resulting vector
    """

  NGRAM_RANGE = (3, 4, 5)

  def __init__(self, dim: int = 256) -> None:
    self.dim = dim

  def embed(self, text: str) -> List[float]:
    # --- E.1 ---
    s = "_" + (text or "").lower().strip() + "_"
    if not s.strip("_"):
      return [0.0] * self.dim

    # --- E.2 + E.3: bucketed n-gram counts ---
    counts = [0.0] * self.dim
    for n in self.NGRAM_RANGE:
      if len(s) < n:
        continue
      for i in range(len(s) - n + 1):
        ng = s[i:i + n]
        h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
        bucket = h % self.dim
        counts[bucket] += 1.0

    # --- E.4: L2 normalise ---
    norm = math.sqrt(sum(c * c for c in counts))
    if norm == 0.0:
      return counts
    return [c / norm for c in counts]

  def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
    return [self.embed(t) for t in texts]


# ============================================================
# SentenceTransformerEmbedder — optional, requires pip install
# ============================================================


class SentenceTransformerEmbedder:
  """
    Real semantic embedder using sentence-transformers. Uses all-MiniLM-L6-v2
    by default (~90MB model, sub-100ms inference on CPU).

    Import is deferred to __init__ so the AISM package works without
    sentence-transformers installed. If you want to use this in production,
    pip install sentence-transformers and pass an instance into the pipeline.
    """

  def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
    try:
      from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
      raise ImportError(
        "sentence-transformers is not installed. Either install it "
        "(`pip install sentence-transformers`) or use "
        "CharacterNgramEmbedder instead.") from exc
    self._model = SentenceTransformer(model_name)
    # Cache dim by doing a probe encode.
    self.dim = int(self._model.get_sentence_embedding_dimension())

  def embed(self, text: str) -> List[float]:
    vec = self._model.encode(text, normalize_embeddings=True)
    return [float(x) for x in vec]

  def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
    vecs = self._model.encode(list(texts), normalize_embeddings=True)
    return [[float(x) for x in v] for v in vecs]
