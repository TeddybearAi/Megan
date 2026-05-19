# Sprint 3 Report — MLOps & ClearML Section (Draft)

## Context

Megan is a deterministic-by-design AI companion (BluePrint v2.1, Design
Principle R1): no generative language model is invoked inside the AISM
personalisation pipeline at runtime. Traditional model training therefore
sits outside the runtime path. The AISM pipeline does, however, contain a
small number of hyperparameters that were hand-tuned during system design
and earmarked in the BluePrint for data-driven refinement. The most
consequential of these is the Layer B paraphrase-matcher's cosine-similarity
threshold (`SIMILARITY_THRESHOLD = 0.72`).

Investigation of Megan's live evidence log during Sprint 3 revealed that
this single threshold dominates the system's behavioural adaptation: 43 of
49 directness evidence items came from Layer B paraphrase matches, not from
the literal Layer A lexicon. The threshold controls a real precision/recall
trade-off and is therefore a meaningful hyperparameter in the
machine-learning sense, despite the deterministic runtime constraint.

## Approach

We built a six-component MLOps pipeline in `mlops/`:

1. **Data/Feature Engineering** — `build_dataset.py` produces 41 labelled
   user utterances (20 positive, 21 negative) drawn from authentic Sprint 3
   conversation logs. Each row carries the expected trait and value if a
   profile update should occur. Features are computed by AISM Stages 1 and 2
   themselves.
2. **Model Training & Evaluation Pipeline** — `train_and_eval.py` runs the
   labelled set through Stages 1+2 at a given threshold and embedder choice,
   computes precision/recall/F1, and logs scalars to ClearML.
3. **Hyperparameter Tuning** — `hpo_sweep.py` invokes ClearML's
   `HyperParameterOptimizer` over the threshold range [0.50, 0.85] using
   grid search, maximising F1 on the labelled set.
4. **Multi-Model Selection** — `multi_embedder.py` repeats the sweep across
   three embedder choices (character-n-gram fallback, all-MiniLM-L6-v2, and
   all-mpnet-base-v2), producing a 3 × 7 grid of ClearML tasks.
5. **CI/CD** — `.github/workflows/ci.yml` runs the full pytest suite on
   every push and pull request to main via GitHub Actions.
6. **Product UI** — the existing `aism_ui/` React + FastAPI frontend remains
   the product surface.

## Result

[Insert two screenshots here: (a) the ClearML parallel-coordinates plot
showing F1 across swept thresholds; (b) the multi-embedder scalar
comparison showing the three F1 curves overlaid.]

The HPO sweep at the production-grade `all-MiniLM-L6-v2` embedder identified
[INSERT THRESHOLD VALUE] as the F1-maximising threshold on the labelled set,
producing F1 = [INSERT VALUE] versus the hand-tuned baseline of F1 =
[INSERT BASELINE] at 0.72. [If plateau: "The result lies inside a flat F1
plateau spanning thresholds 0.65 to 0.85, confirming the hand-tuned value
is defensible without modification."]

The multi-embedder comparison showed [INSERT WINNING EMBEDDER] consistently
above the other two across the swept range, supporting its selection as the
production embedder over the character-n-gram fallback currently used as a
no-dependency default.

## Honest assessment

The labelled evaluation set is small (n=41) and authored from a single
user's speech patterns; results do not generalise to a broader user
population without a larger, multi-annotator corpus. The HPO outcome should
be read as confirmation of a defensible production value rather than as a
universally optimal setting. The exercise demonstrates the MLOps tooling
(ClearML experiment tracking, HPO, GitHub Actions CI) on a real, narrow,
well-bounded hyperparameter in Megan's adaptation pipeline.
