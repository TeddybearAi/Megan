# AISM — Adaptive Interaction Style Modelling

Pass 1 + Pass 2 complete implementation for the Megan AI-companion project.

Implements BluePrint v2.1. **Fully on-device, deterministic, zero LLM calls inside the pipeline.** The only runtime neural component is an optional sentence-embedder used by Stage 2 Layer B and the Idiolect Drift monitor.

## Package layout

```
aism/
├── aism/
│   ├── __init__.py              Public API (Pass 1 + Pass 2)
│   ├── data_models.py           Dataclasses + enums
│   ├── session_context.py       Integration contract (AISM ↔ LTM)
│   ├── embedder.py              Embedder Protocol + 2 implementations
│   ├── stage1_eligibility.py    STAGE 1 — rule-based eligibility
│   ├── stage2_extraction.py     STAGE 2 Layer A — lexicon matching
│   ├── stage2b_layer_b_extractor.py  STAGE 2 Layer B — semantic matching
│   ├── stage2c_pattern_aggregator.py STAGE 2C — cross-turn aggregator
│   ├── stage3_scoring.py        STAGE 3 — gated log-linear scorer
│   ├── stage4_updater.py        STAGE 4 — smoothed update + guardrails
│   ├── stage5_retrieval.py      STAGE 5 — policy retrieval with priority walk
│   ├── stage6_feedback.py       STAGE 6 — closed feedback loop
│   ├── storage.py               Local JSON persistence
│   ├── generation_adapter.py    GenerationAdapter Protocol + MockAdapter
│   ├── pipeline.py              AdaptiveInteractionStylePipeline orchestrator
│   ├── monitors/
│   │   ├── lexical_mimicry.py   Monitor 1 — n-gram overlap
│   │   ├── idiolect_drift.py    Monitor 2 — embedding distance from baseline
│   │   ├── sycophancy.py        Monitor 3 — position-reversal heuristic
│   │   └── comfort_rating.py    Monitor 4 — Likert recorder
│   ├── adapters/
│   │   └── ollama_adapter.py    Concrete adapter for local Ollama server
│   └── evaluation/
│       ├── personas.py          SyntheticPersona + SAMPLE_PERSONAS
│       ├── metrics.py           F1, PSI, Adaptation Latency, ...
│       └── harness.py           PersonaHarness runner
└── tests/
    ├── test_end_to_end.py       Pass 1 demo (6 stages, 8 turns)
    └── test_pass2.py            Pass 2 demo (Layer B, aggregator,
                                 4 monitors, evaluation harness)
```

## Running the demos

```bash
cd aism/
python -m tests.test_end_to_end      # Pass 1 core pipeline
python -m tests.test_pass2           # Pass 2 advanced features
```

## What's in each pass

### Pass 1 (core pipeline)

- All six deterministic pipeline stages
- Gated log-linear scorer with source-aware stability classification
- Smoothed profile update with per-update and per-session drift caps
- Closed feedback loop (Stage 6 feeds back into Stage 3)
- Context-conditional traits with strict-update / fallback-read semantics
- Four local JSON storage structures
- MockGenerationAdapter for LLM-free testing

### Pass 2 (advanced features, all opt-in)

- **Layer B extractor** — embedding-assisted semantic matching against lexicon cues. Catches paraphrases Layer A misses.
- **PatternAggregator** — cross-turn implicit pattern detection. Turns many weak signals into one stronger aggregate evidence item.
- **Four over-mimicry monitors** — Lexical Mimicry, Idiolect Drift, Sycophancy (all runtime), and Comfort Rating (user-study recorder).
- **OllamaAdapter** — HTTP adapter for local Ollama-served open-source LLMs (Llama, Mistral, Phi, Gemma, ...).
- **Evaluation harness** — synthetic personas, Tier 1 metrics (Trait Extraction F1, Profile Stability Index, Adaptation Latency, Eligibility Accuracy, Contamination Rate, Ground-Truth Match Rate).

## Design constraints honoured

- **R1: No LLM calls anywhere inside the pipeline.** Pass 1 uses pattern matching and arithmetic only. Pass 2 adds an optional sentence-embedder, which is non-generative and runs locally.
- **R2: Small neural components permitted.** The embedder is the only neural runtime component. `CharacterNgramEmbedder` is a zero-dependency deterministic fallback; `SentenceTransformerEmbedder` is the real one (requires `pip install sentence-transformers`).
- **R3: LLMs for development are unrestricted.** The evaluation harness uses hand-crafted personas for Pass 2; a production harness would generate persona conversations with a capable LLM offline.
- **AISM ↔ LTM boundary**: AISM only reads `SessionContext` for conditioning; the Orchestrator bridges the two subsystems.

## Honest caveats about Pass 2

The `CharacterNgramEmbedder` is a deterministic stand-in — it catches some paraphrases but not all. For production use, swap in `SentenceTransformerEmbedder` (~90MB, one-time model download). The Protocol guarantees that nothing else in the code needs to change.

The structural sycophancy detector (Monitor 3) is coarse. It will have both false positives (legitimate position changes correlated with pushback) and false negatives (sycophantic reversals without the lexical markers it recognises). The BluePrint acknowledges this; the LLM-judged version is evaluation-time only and lives in the development toolkit, not the runtime.

`correction_heavy` persona corrections use `topic="general"` because the user's complaint ("that was too long") is stylistic not technical. This is an Orchestrator responsibility — classifying task_type and topic so style-feedback turns update the general-context trait. AISM follows the `SessionContext` it's given.

## Running the Frontend App (Preact + Vite)

The frontend provides the voice interface with Safari-optimized audio playback and "Megan" wake-word detection.

1. **Navigate & Install**:

   ```bash
   cd frontend/
   npm install
   ```

2. **Start Dev Server**:

   ```bash
   npm run dev
   ```

3. **Activation**:
   Open the app in your browser. **Click the "ACTIVATE MEGAN" header** once. This is required by the browser to unlock the microphone and audio engine for hands-free mode.

4. **Commands**:
   - **Wake Up**: Say "Hey Megan".
   - **Conversation**: Talk naturally; the mic stays active after she speaks.
   - **Standby**: Say "Goodbye" or "See you later" to clear the chat and return to wake-word mode.
