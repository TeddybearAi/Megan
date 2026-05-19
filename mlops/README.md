# Megan — MLOps & ClearML Pipeline

This folder contains the MLOps deliverables for AT4 (Sprint 3 Artefact).
It demonstrates the use of **ClearML** and **GitHub Actions** to evaluate
and optimise a hyperparameter in Megan's AISM personalisation pipeline.

## What's tuned

`SIMILARITY_THRESHOLD = 0.72` in `aism/stage2b_layer_b_extractor.py` line 68.

This is the cosine-similarity threshold that the Layer B paraphrase matcher
uses to decide whether a candidate phrase from a user turn counts as a
style-preference match. It was hand-tuned per BluePrint v2.1 §3 and earmarked
for data-driven refinement. Investigation of the live `evidence_log.json`
showed Layer B paraphrase matches dominate the actual adaptation (43 of 49
directness evidence items came from Layer B, not the literal Layer A lexicon),
so this threshold is the highest-leverage hyperparameter in the system.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Writes 41 labelled user turns to `dataset.csv` |
| `dataset.csv` | The labelled evaluation set (20 positive, 21 negative) |
| `train_and_eval.py` | Runs the labelled set through AISM Stages 1+2 at a given threshold + embedder, logs P/R/F1 to ClearML |
| `hpo_sweep.py` | ClearML `HyperParameterOptimizer` over `SIMILARITY_THRESHOLD` |
| `multi_embedder.py` | Same sweep across three embedder choices (multi-model selection) |

## Prerequisites

1. **ClearML credentials in `~/clearml.conf`.** Generate via app.clear.ml →
   Settings → Workspace → Create new credentials, then paste the config
   block into `~/clearml.conf`. (Windows: `C:\Users\<you>\clearml.conf`.)
2. **`clearml` Python package installed.** `pip install clearml`.
3. **The Megan codebase importable as `aism.*`.** Run scripts from the
   repository root: `cd Megan; python mlops/train_and_eval.py ...`.
4. **For the `minilm` and `mpnet` embedder runs**: a working Hugging Face
   download path. First run pulls weights; subsequent runs are cached.

## Run order

### Step 1 — Build the dataset

```bash
python mlops/build_dataset.py
```
Outputs `mlops/dataset.csv`. Idempotent; safe to rerun.

### Step 2 — Seed the base task

ClearML's HPO needs an existing task to clone from. Run train_and_eval once
to create it:

```bash
python mlops/train_and_eval.py --threshold 0.72 --embedder minilm
```
ClearML prints `ClearML Task: created new task id=...` — copy that task ID.

### Step 3 — Run the HPO sweep

```bash
python mlops/hpo_sweep.py --base-task-id <task_id_from_step_2>
```
Launches 11 trials across thresholds 0.50 → 0.85. Total runtime ~10 minutes
on the reference hardware. Each trial appears on the ClearML dashboard as a
separate task; the optimiser task aggregates them.

### Step 4 — Run the multi-embedder comparison

```bash
python mlops/multi_embedder.py
```
21 ClearML tasks (3 embedders × 7 thresholds). Runtime ~30 minutes,
dominated by the first `minilm`/`mpnet` runs that download the model.

## Dashboard screenshots for the report

After all runs complete, capture the following from app.clear.ml:

1. **HPO sweep — parallel-coordinates plot.** Project `Megan-AISM` →
   Experiments tab → select the `hpo_layerb_threshold_sweep` task → Plots
   → Parallel Coordinates. Shows F1 across the swept thresholds.
2. **Multi-embedder scalar comparison.** Project `Megan-AISM` → Experiments
   tab → select all 21 `multi_emb=*` tasks → Compare → Scalars → F1.
   Three coloured lines on the same axes, one per embedder.
3. **Top experiments table.** From either view, the sortable scalar
   summary showing the best (threshold, embedder, F1) combinations.

## What the result tells us

The headline finding from a clean run is one of two shapes:

- **Plateau result**: F1 is flat across the whole threshold range, with
  the current 0.72 inside the flat region. Conclusion: the hand-tuned
  value is defensible; HPO confirms there's no measurable benefit to
  changing it on this evaluation set.
- **Optimum result**: F1 peaks at some threshold ≠ 0.72. Conclusion: the
  identified value would be the empirical replacement; whether to update
  the live code in `aism/stage2b_layer_b_extractor.py` is a separate
  deployment decision and is **not done automatically**.

Either result is a valid MLOps outcome. The first is a negative result;
the second is a positive one. Both are reported honestly in the report.
