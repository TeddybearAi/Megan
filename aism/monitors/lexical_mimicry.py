"""
AISM — Monitor 1/4: Lexical Mimicry Index
=========================================
Detects the assistant copying the user's recent phrasing into its own
replies. Computes a rolling n-gram overlap score, excluding common function
words so "the" and "a" don't swamp the signal.

BluePrint v2.1 reference: Section 5 — Over-mimicry safeguards.

Steps:
    Step M1.1 : Tokenise user-recent and assistant-reply into word sequences.
    Step M1.2 : Generate n-gram sets (n=3,4) for each side, filtering stopwords.
    Step M1.3 : Compute Jaccard overlap on the n-gram sets.
    Step M1.4 : Exponential moving average over recent observations to smooth
                per-turn noise.
    Step M1.5 : Flag if the EMA exceeds a threshold; emit a penalty signal.

Constraint: Pure string processing. No LLM, no neural model.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Deque, List, Set

STOPWORDS = frozenset(
  """
a an and the is are was were be been being do does did done
of in on at to for from by with without into onto over under
i you he she it we they me him her us them my your his its our their
this that these those
but or nor so yet if because as than then when while
not no yes ok okay
just really very quite pretty much many more less
""".split())


class LexicalMimicryMonitor:
  """
    Tracks per-turn overlap and maintains a smoothed index over time.
    """

  # Step M1.2 params
  NGRAM_SIZES = (3, 4)
  # Step M1.4 params
  EMA_ALPHA = 0.30
  # Step M1.5 params
  MIMICRY_THRESHOLD = 0.18  # above this, we flag excessive mimicry

  _WORD_RE = re.compile(r"[A-Za-z']+")

  def __init__(self, user_history_window: int = 3) -> None:
    self.user_history: Deque[str] = deque(maxlen=user_history_window)
    self.ema_index: float = 0.0
    self.observations: int = 0

  # ============================================================
  # Public API
  # ============================================================

  def record_user_turn(self, text: str) -> None:
    """Call before assistant reply is generated."""
    self.user_history.append(text)

  def score_reply(self, assistant_reply: str) -> dict:
    """Call with the assistant reply. Returns metrics for this turn."""
    if not self.user_history:
      return {
        "per_turn_overlap": 0.0,
        "ema_index": self.ema_index,
        "flagged": False
      }

    # --- Step M1.1-M1.3: compute per-turn overlap ---
    user_blob = " ".join(self.user_history)
    user_ngrams = self._content_ngrams(user_blob)
    reply_ngrams = self._content_ngrams(assistant_reply)
    overlap = self._jaccard(user_ngrams, reply_ngrams)

    # --- Step M1.4: EMA smoothing ---
    if self.observations == 0:
      self.ema_index = overlap
    else:
      self.ema_index = (
        (1 - self.EMA_ALPHA) * self.ema_index + self.EMA_ALPHA * overlap)
    self.observations += 1

    # --- Step M1.5: flag excessive mimicry ---
    flagged = self.ema_index > self.MIMICRY_THRESHOLD

    return {
      "per_turn_overlap": round(overlap, 4),
      "ema_index": round(self.ema_index, 4),
      "flagged": flagged,
      "observations": self.observations,
    }

  # ============================================================
  # Helpers
  # ============================================================

  @classmethod
  def _tokens(cls, text: str) -> List[str]:
    return [w.lower() for w in cls._WORD_RE.findall(text or "")]

  @classmethod
  def _content_ngrams(cls, text: str) -> Set[tuple]:
    """Content-word n-grams only — stopwords excluded."""
    words = [w for w in cls._tokens(text) if w not in STOPWORDS]
    grams: Set[tuple] = set()
    for n in cls.NGRAM_SIZES:
      if len(words) < n:
        continue
      for i in range(len(words) - n + 1):
        grams.add(tuple(words[i:i + n]))
    return grams

  @staticmethod
  def _jaccard(a: Set[tuple], b: Set[tuple]) -> float:
    if not a or not b:
      return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
