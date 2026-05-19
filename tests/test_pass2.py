"""
AISM — Pass 2 End-to-End Demo
=============================
Exercises the Pass 2 components layered on top of the Pass 1 pipeline:

    A. Layer B (embedding-assisted semantic matching against lexicons)
    B. PatternAggregator (cross-turn implicit pattern detection)
    C. Four over-mimicry monitors (lexical, idiolect drift, sycophancy, comfort)
    D. Evaluation harness (Tier 1 synthetic personas + metrics)

Run:
    python -m tests.test_pass2
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

from aism import (
  AdaptiveInteractionStylePipeline,
  ConversationTurn,
  LocalJSONStore,
  SessionContext,
  TurnRole,
)
from aism.embedder import CharacterNgramEmbedder
from aism.evaluation import SAMPLE_PERSONAS, PersonaHarness, evaluate_all
from aism.monitors import (
  ComfortRating,
  ComfortRatingRecorder,
  IdiolectDriftMonitor,
  LexicalMimicryMonitor,
  SycophancyMonitor,
)
from aism.stage2b_layer_b_extractor import LayerBExtractor
from aism.stage2c_pattern_aggregator import PatternAggregator

# ============================================================
# Pretty-printing helpers
# ============================================================


def banner(title: str) -> None:
  print("\n" + "=" * 78)
  print(title)
  print("=" * 78)


def subbanner(title: str) -> None:
  print("\n" + "-" * 78)
  print(title)
  print("-" * 78)


# ============================================================
# Part A — Layer B semantic extraction
# ============================================================


def demo_layer_b() -> None:
  """
    Show Layer B catching paraphrases that Layer A misses.

    The user's phrasing "keep it tight please" is NOT in Layer A's verbosity
    lexicon, but it's semantically close to cues like "keep it brief" /
    "keep it short" so Layer B should catch it.
    """
  banner("PART A — Layer B semantic extraction")

  tmpdir = tempfile.mkdtemp(prefix="aism_p2a_")
  try:
    storage = LocalJSONStore(tmpdir)
    embedder = CharacterNgramEmbedder(dim=256)
    layer_b = LayerBExtractor(embedder)

    pipeline = AdaptiveInteractionStylePipeline(
      user_id="demo_layer_b",
      session_id="s1",
      storage=storage,
      layer_b_extractor=layer_b,
    )

    ctx = SessionContext(
      current_topic="general", task_type="conversation", turn_index=1)
    test_phrases = [
      # Layer A should hit these — they match literal lexicon patterns.
      "Please be direct and don't sugarcoat things.",
      # Layer A misses this — it's a paraphrase. Layer B should catch it.
      "Keep it tight please, I'm in a rush.",
      # Another paraphrase — Layer A's "elaborate"/"more detail" lexicon
      # doesn't have this exact phrasing.
      "Can you unpack that a bit more thoroughly.",
    ]
    start = datetime.now(timezone.utc)

    for i, text in enumerate(test_phrases, start=1):
      turn = ConversationTurn(
        turn_id=f"a{i}",
        role=TurnRole.USER,
        text=text,
        timestamp=start + timedelta(seconds=i * 30),
      )
      ctx.turn_index = i
      result = pipeline.process_turn(turn, ctx)

      subbanner(f"Turn {i}: {text!r}")
      print(f"  Layer A evidence: {len(result.extraction_evidence)}")
      for ev in result.extraction_evidence:
        print(
          f"    [A] {ev.trait}={ev.value}  ({ev.source_type.value}, conf={ev.confidence:.2f})"
        )
      print(f"  Layer B evidence: {len(result.layer_b_evidence)}")
      for ev in result.layer_b_evidence:
        sim = ev.metadata.get("similarity")
        cue = ev.metadata.get("matched_cue")
        cand = ev.metadata.get("candidate_phrase")
        print(f"    [B] {ev.trait}={ev.value}  sim={sim:.3f}")
        print(f"        matched cue: {cue!r}")
        print(f"        candidate:   {cand!r}")
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Part B — Cross-turn pattern aggregator
# ============================================================


def demo_aggregator() -> None:
  """
    The aggregator's job: turn many weak signals into one strong one.

    Rather than constructing a full pipeline turn sequence (which would
    require a semantic embedder stronger than our char-ngram fallback to
    produce enough implicit signals), we test the aggregator directly with
    a constructed evidence log. This is the honest way to demonstrate the
    component because the pipeline integration is straightforward once the
    component works in isolation.
    """
  from datetime import datetime, timezone
  from aism.data_models import (
    EvidenceSourceType,
    InteractionProfile,
    StabilityClass,
    StyleEvidence,
  )

  banner("PART B — Cross-turn pattern aggregator (direct unit test)")

  # Build a synthetic evidence log: five weak implicit signals all pointing
  # to verbosity=short in technical context. None of these on their own
  # would become stable (they're implicit_pattern at confidence 0.55), but
  # together they form a clear pattern.
  now = datetime.now(timezone.utc)
  evidence_log = []
  for i in range(5):
    evidence_log.append(
      StyleEvidence(
        evidence_id=f"syn-{i}",
        turn_id=f"t{i}",
        trait="verbosity",
        context="technical",
        value="short",
        confidence=0.55,
        # Use EXPLICIT_PREFERENCE so the aggregator sees them (it skips
        # IMPLICIT_PATTERN to avoid re-aggregating its own output).
        # In a real deployment these would come from Layer A or B
        # finding weak preference signals.
        source_type=EvidenceSourceType.EXPLICIT_PREFERENCE,
        stability_class=StabilityClass.CANDIDATE_STABLE,
        timestamp=now,
        evidence_text=f"weak signal {i}",
      ))

  # Empty profile — nothing stored yet for this trait.
  profile = InteractionProfile(user_id="agg_test")
  ctx = SessionContext(current_topic="technical", task_type="conversation")

  aggregator = PatternAggregator()
  aggregate_items = aggregator.aggregate(
    evidence_log=evidence_log,
    profile=profile,
    session_context=ctx,
    now_timestamp=now,
  )

  print(
    f"\n  Input: {len(evidence_log)} individual evidence items, "
    f"each confidence 0.55 (below stable threshold)")
  print(f"  Profile before: (empty — no traits stored)")
  print(f"\n  Aggregator output: {len(aggregate_items)} aggregate item(s)")
  for ev in aggregate_items:
    print(f"    trait   : {ev.trait}[{ev.context}]")
    print(f"    value   : {ev.value}")
    print(f"    conf    : {ev.confidence}  (boosted above any single item)")
    print(f"    source  : {ev.source_type.value}")
    print(
      f"    support : {ev.metadata['support_count']}/{ev.metadata['total_in_window']} "
      f"(mode_fraction={ev.metadata['mode_fraction']})")
    print(f"    summary : {ev.evidence_text}")

  print(
    "\n  --- Dedup check: run aggregator again with profile now matching ---")
  # Simulate what happens if the aggregate's value is already in the profile.
  from aism.data_models import TraitState
  profile.set_trait(
    "verbosity", "technical",
    TraitState(
      value="short",
      confidence=0.7,
      evidence_count=1,
      volatility=0.0,
      last_updated=now,
    ))
  re_aggregate = aggregator.aggregate(
    evidence_log=evidence_log,
    profile=profile,
    session_context=ctx,
    now_timestamp=now,
  )
  print(
    f"  With profile already = 'short': aggregator emits "
    f"{len(re_aggregate)} items (expected 0 — nothing new to learn)")


# ============================================================
# Part C — Four over-mimicry monitors
# ============================================================


class ScriptedMimickyAdapter:
  """
    A test adapter that deliberately mimics the user's phrasing, so the
    Lexical Mimicry and Idiolect Drift monitors have something to flag.
    Also reverses a position to exercise the Sycophancy monitor.
    """

  def __init__(self):
    self.counter = 0

  def render_policy(self, policy):
    return {"system_prompt": "mock"}

  def generate(
      self, user_message, retrieved_facts, rendered_policy, session_context):
    self.counter += 1
    # Turns 1-3: baseline replies (short, varied).
    baseline = [
      "Here is a concise summary of the key points.",
      "The main idea involves three steps and one exception.",
      "A quick overview: define, validate, and persist the record.",
    ]
    if self.counter <= 3:
      return baseline[self.counter - 1]
    # Turn 4: echo user phrasing heavily.
    if self.counter == 4:
      # Echo many content words from user message.
      return f"Absolutely — {user_message} Yes, {user_message}".strip()
    # Turn 5: sycophantic reversal.
    if self.counter == 5:
      return "You're right, I was wrong. Apologies — the answer is actually different."
    return "Here is another short summary of the key points today."


def demo_monitors() -> None:
  banner("PART C — Four over-mimicry monitors")

  tmpdir = tempfile.mkdtemp(prefix="aism_p2c_")
  try:
    storage = LocalJSONStore(tmpdir)
    embedder = CharacterNgramEmbedder(dim=256)

    lexical = LexicalMimicryMonitor()
    drift = IdiolectDriftMonitor(embedder)
    sycophancy = SycophancyMonitor()
    comfort = ComfortRatingRecorder()

    pipeline = AdaptiveInteractionStylePipeline(
      user_id="demo_monitors",
      session_id="s1",
      storage=storage,
      adapter=ScriptedMimickyAdapter(),
      lexical_mimicry_monitor=lexical,
      idiolect_drift_monitor=drift,
      sycophancy_monitor=sycophancy,
    )

    ctx = SessionContext(
      current_topic="general", task_type="conversation", turn_index=0)
    script = [
      # Turns 1-3 establish baseline voice.
      ("What's a good way to deploy a small Flask app?", False),
      ("And how do I handle environment variables?", False),
      ("What about logging setup?", False),
      # Turn 4: user uses distinctive phrasing we'll echo.
      ("The deployment checklist should cover rollback procedures.", False),
      # Turn 5: user pushback with NO new info, to trigger sycophancy.
      ("No, you're wrong about that. Are you sure?", False),
      # Turn 6: user pushback WITH new info — reversal should be allowed.
      ("Actually, the docs say the default is False, I checked.", True),
    ]
    start = datetime.now(timezone.utc)

    for i, (text, _has_info) in enumerate(script, start=1):
      turn = ConversationTurn(
        turn_id=f"c{i}",
        role=TurnRole.USER,
        text=text,
        timestamp=start + timedelta(seconds=i * 30),
      )
      ctx.turn_index = i
      result = pipeline.process_turn(turn, ctx)
      subbanner(f"Turn {i}: user={text!r}")
      print(f"  reply: {result.generated_reply}")
      mm = result.monitor_metrics
      if "lexical_mimicry" in mm:
        lm = mm["lexical_mimicry"]
        print(
          f"  lexical mimicry: per_turn={lm['per_turn_overlap']}  "
          f"ema={lm['ema_index']}  flagged={lm['flagged']}")
      if "idiolect_drift" in mm:
        idd = mm["idiolect_drift"]
        if idd.get("baseline_building"):
          print(
            f"  idiolect drift:  baseline building "
            f"({idd['baseline_size']}/{5})")
        else:
          print(
            f"  idiolect drift:  drift={idd['drift_score']}  "
            f"rolling={idd['rolling_drift']}  flagged={idd['flagged']}")
      if "sycophancy" in mm:
        s = mm["sycophancy"]
        print(
          f"  sycophancy:      reversal={s['contains_reversal']}  "
          f"sycophantic={s['sycophantic']}  rate={s['sycophancy_rate']}  "
          f"({s['sycophantic_reversals']}/{s['total_reversals']})")

    # Demonstrate the comfort recorder.
    subbanner("Comfort rating (post-session, user study)")
    comfort.record(
      ComfortRating(
        session_id="s1",
        p1_unsaid_reference=2,
        p2_uncomfortable_mirroring=3,
        p3_surveillance_feeling=1,
        p4_becoming_user=2,
        notes="Reply at T4 felt a bit echoey.",
      ))
    summary = comfort.summarise()
    print(f"  n sessions: {summary['n']}")
    for probe, stats in summary.items():
      if probe == "n":
        continue
      print(
        f"  {probe}: mean={stats['mean']}  worst={stats['worst_case']}  "
        f"dist={stats['distribution']}")
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Part D — Evaluation harness with sample personas
# ============================================================


def demo_evaluation() -> None:
  banner("PART D — Evaluation harness (Tier 1 synthetic personas)")

  results = evaluate_all(SAMPLE_PERSONAS)
  for r in results:
    subbanner(f"Persona: {r.persona_id}")
    print(r.pretty_summary())

  # Aggregate across personas.
  subbanner("AGGREGATE across all personas")
  total_tp = sum(r.extraction.true_positives for r in results)
  total_fp = sum(r.extraction.false_positives for r in results)
  total_fn = sum(r.extraction.false_negatives for r in results)
  total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
  total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
  total_f1 = (
    2 * total_p * total_r / (total_p + total_r) if
    (total_p + total_r) else 0.0)
  print(
    f"  Extraction (micro)  P={total_p:.3f}  R={total_r:.3f}  F1={total_f1:.3f}"
  )
  avg_psi = sum(r.psi for r in results) / len(results)
  print(f"  Avg PSI             {avg_psi:.3f}")
  avg_match = (
    sum(r.ground_truth["match_rate"] for r in results) / len(results))
  print(f"  Avg GT match rate   {avg_match:.3f}")


# ============================================================
# Main
# ============================================================


def main() -> None:
  banner("AISM Pass 2 — End-to-End Demo")
  demo_layer_b()
  demo_aggregator()
  demo_monitors()
  demo_evaluation()
  banner("Pass 2 demo complete.")


if __name__ == "__main__":
  main()
