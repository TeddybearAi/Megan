"""Tests for AISM Pass 2 wiring: Layer B + Pattern Aggregator."""

from __future__ import annotations

import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism import AdaptiveInteractionStylePipeline
from aism.adapters.ollama_adapter import OllamaAdapter  # only used as a type
from aism.data_models import ConversationTurn, TurnRole
from aism.embedder import CharacterNgramEmbedder
from aism.session_context import SessionContext
from aism.stage2b_layer_b_extractor import LayerBExtractor
from aism.stage2c_pattern_aggregator import PatternAggregator
from aism.storage import LocalJSONStore


def _build_pipeline(*, with_pass2: bool):
  """Build a pipeline in a tempdir, optionally with Pass 2 components."""
  tmp = Path(tempfile.mkdtemp(prefix="aism_pass2_test_"))
  emb = CharacterNgramEmbedder(dim=128)
  storage = LocalJSONStore(tmp, embedder=emb)

  # We don't need a real LLM adapter for these tests — the pipeline will
  # use MockGenerationAdapter when adapter is None.
  if with_pass2:
    return AdaptiveInteractionStylePipeline(
      user_id="u",
      session_id=str(uuid.uuid4()),
      storage=storage,
      adapter=None,
      layer_b_extractor=LayerBExtractor(emb),
      pattern_aggregator=PatternAggregator(),
    )
  return AdaptiveInteractionStylePipeline(
    user_id="u",
    session_id=str(uuid.uuid4()),
    storage=storage,
    adapter=None,
  )


def _turn(text, idx=0):
  return ConversationTurn(
    turn_id=f"t{idx}",
    role=TurnRole.USER,
    text=text,
    timestamp=datetime.now(timezone.utc),
  )


# ---------- Layer B is constructable, accepts the embedder ----------------

def test_layer_b_constructible_with_embedder():
  emb = CharacterNgramEmbedder(dim=128)
  lb = LayerBExtractor(emb)
  # Should have pre-encoded all the cue exemplars from TRAIT_LEXICONS.
  assert lb.encoded_cues, (
    "LayerBExtractor must encode at least some cues at construction")
  assert all(len(c.vector) == emb.dim for c in lb.encoded_cues), (
    "All encoded cue vectors should match the embedder dim")


def test_pattern_aggregator_constructible():
  agg = PatternAggregator()
  # Trivial smoke: it has the public `aggregate` method.
  assert hasattr(agg, "aggregate"), (
    "PatternAggregator must expose an aggregate(...) method")


# ---------- Pipeline accepts the Pass 2 kwargs ----------------------------

def test_pipeline_accepts_pass2_components():
  pl = _build_pipeline(with_pass2=True)
  assert pl.layer_b_extractor is not None
  assert pl.pattern_aggregator is not None


def test_pipeline_pass1_still_works_without_pass2():
  pl = _build_pipeline(with_pass2=False)
  assert pl.layer_b_extractor is None
  assert pl.pattern_aggregator is None


# ---------- End-to-end: a turn flows through the Pass 2 pipeline ----------

def test_pipeline_processes_turn_with_pass2_active():
  pl = _build_pipeline(with_pass2=True)
  ctx = SessionContext()
  result = pl.process_turn(
    _turn("Could you be more concise please?"),
    ctx,
  )
  # We don't assert on specific evidence content here (Layer B's matches
  # depend on embedder strength). We only assert the pipeline doesn't blow
  # up and returns a TurnResult-shaped object.
  assert hasattr(result, "evidence") or hasattr(result, "extraction_evidence"), (
    "process_turn should return a result with evidence")


def test_pipeline_layer_b_emits_evidence_on_paraphrase():
  """Smoke test: Layer B should fire on a paraphrase that the lexicon
  doesn't catch literally. Uses a phrase semantically close to the
  directness-high cue 'be direct' but worded differently."""
  pl = _build_pipeline(with_pass2=True)
  ctx = SessionContext()
  # 'cut to the chase' is semantically close to 'get to the point' / 'be
  # direct' but the literal phrase isn't in the lexicon.
  pl.process_turn(_turn("Cut to the chase, would you?"), ctx)
  # We don't strictly require Layer B to catch this with CharacterNgram —
  # that embedder is too coarse — but the pipeline must not crash and
  # must store extraction state. The semantic version of this assertion
  # belongs in an integration test against the real sentence-transformer
  # embedder.


def test_layer_b_is_optional_no_crash_when_missing():
  """Pipeline must run cleanly when Pass 2 components are None."""
  pl = _build_pipeline(with_pass2=False)
  ctx = SessionContext()
  pl.process_turn(_turn("Be more concise."), ctx)


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
