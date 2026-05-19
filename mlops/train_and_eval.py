"""
mlops/train_and_eval.py
=======================
Runs the labelled evaluation set through AISM Stages 1 and 2 at a given
Layer B similarity threshold and embedder choice, then computes:

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)

Each labelled row is one ground-truth label. Stage 2's verdict per row is
the prediction:
    - "expects_evidence == 1" → positive label
    - len(extracted_evidence) > 0 after run → positive prediction

ClearML wiring:
    Task.init(...) at the top makes this a tracked experiment. Hyperparameters
    are reported via task.connect(). Final metrics are logged via task.get_logger().
    The HPO sweep (hpo_sweep.py) varies the hyperparameters across many runs.

Standalone use:
    python mlops/train_and_eval.py --threshold 0.72 --embedder charngram

ClearML HPO use:
    Invoked automatically by hpo_sweep.py with varying --threshold values.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import aism.*
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aism.data_models import ConversationTurn, TurnRole
from aism.session_context import SessionContext
from aism.stage1_eligibility import EligibilityFilter
from aism.stage2_extraction import StyleEvidenceExtractor
from aism.stage2b_layer_b_extractor import LayerBExtractor

# ClearML is optional at import time so the script can run locally without it
# (e.g. for quick sanity checks). When run under HPO, Task.init() must succeed.
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False


# ─── Embedder factory ────────────────────────────────────────────────────
def make_embedder(name: str):
    """Return an Embedder instance by name."""
    if name == "charngram":
        from aism.embedder import CharacterNgramEmbedder
        return CharacterNgramEmbedder()
    elif name == "minilm":
        from aism.embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif name == "mpnet":
        from aism.embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        raise ValueError(f"Unknown embedder: {name!r}")


# ─── Dataset loader ──────────────────────────────────────────────────────
def load_dataset(csv_path: Path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "text": row["text"],
                "expects_evidence": int(row["expects_evidence"]),
                "trait": row["trait"],
                "expected_value": row["expected_value"],
                "notes": row["notes"],
            })
    return rows


# ─── Evaluation ──────────────────────────────────────────────────────────
def evaluate(rows, threshold: float, embedder_name: str):
    """Run Stages 1+2 on each row at given Layer B threshold; compute metrics."""
    embedder = make_embedder(embedder_name)

    # Monkey-patch the threshold for this run. Layer B reads it as a class attr.
    LayerBExtractor.SIMILARITY_THRESHOLD = threshold

    eligibility = EligibilityFilter()
    extractor = StyleEvidenceExtractor()
    layer_b = LayerBExtractor(embedder=embedder)

    tp = fp = tn = fn = 0
    trait_tp = trait_fp = 0  # per-trait accuracy (more stringent than just yes/no)

    per_row_results = []

    for row in rows:
        text = row["text"]
        expects = row["expects_evidence"]
        expected_trait = row["trait"]

        turn = ConversationTurn(
            turn_id="t0", role=TurnRole.USER, text=text,
            timestamp=datetime.now(timezone.utc),
        )
        ctx = SessionContext(current_topic="general", task_type="conversation")

        # Stage 1
        decision = eligibility.evaluate_turn(turn)
        if not decision.is_eligible:
            # Stage 1 vetoed; treat as no-evidence prediction
            predicted_evidence = []
        else:
            # Stage 2 Layer A
            layer_a_evs = extractor.extract(turn=turn, session_context=ctx)
            # Stage 2 Layer B (adds paraphrase matches not already in Layer A)
            layer_b_evs = layer_b.extract(
                turn=turn, session_context=ctx, layer_a_evidences=layer_a_evs,
            )
            predicted_evidence = layer_a_evs + layer_b_evs

        predicted = 1 if predicted_evidence else 0
        predicted_traits = {ev.trait for ev in predicted_evidence}

        # Per-row confusion matrix
        if expects == 1 and predicted == 1:
            tp += 1
            if expected_trait in predicted_traits:
                trait_tp += 1
            else:
                trait_fp += 1  # right that *something* fired but wrong trait
        elif expects == 0 and predicted == 1:
            fp += 1
        elif expects == 0 and predicted == 0:
            tn += 1
        elif expects == 1 and predicted == 0:
            fn += 1

        per_row_results.append({
            "text": text,
            "expects": expects,
            "predicted": predicted,
            "expected_trait": expected_trait,
            "predicted_traits": ",".join(sorted(predicted_traits)),
        })

    # Aggregate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    trait_acc = trait_tp / tp if tp > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "trait_accuracy": trait_acc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "per_row": per_row_results,
    }


# ─── Entry point ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.72,
                    help="Layer B similarity threshold (default 0.72, the hand-tuned value)")
    ap.add_argument("--embedder", default="charngram",
                    choices=["charngram", "minilm", "mpnet"],
                    help="Embedder to use for Layer B")
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).parent / "dataset.csv")
    ap.add_argument("--project", default="Megan-AISM",
                    help="ClearML project name")
    ap.add_argument("--task-name", default=None,
                    help="ClearML task name (auto-generated if omitted)")
    ap.add_argument("--no-clearml", action="store_true",
                    help="Run without ClearML (for local smoke-testing)")
    args = ap.parse_args()

    # ClearML task initialisation — happens BEFORE any heavy work so that
    # all the prints + scalars are captured in the task's console + scalars log.
    task = None
    if CLEARML_AVAILABLE and not args.no_clearml:
        task_name = args.task_name or f"eval_thr={args.threshold}_emb={args.embedder}"
        task = Task.init(
            project_name=args.project,
            task_name=task_name,
            reuse_last_task_id=False,
        )
        task.connect({
            "threshold": args.threshold,
            "embedder": args.embedder,
            "dataset_path": str(args.dataset),
        })

    # Load + evaluate
    rows = load_dataset(args.dataset)
    print(f"Loaded {len(rows)} labelled examples from {args.dataset}")
    print(f"Layer B SIMILARITY_THRESHOLD = {args.threshold}")
    print(f"Embedder: {args.embedder}")
    print()

    results = evaluate(rows, threshold=args.threshold, embedder_name=args.embedder)

    print("Confusion matrix:")
    print(f"  TP={results['tp']}  FP={results['fp']}  TN={results['tn']}  FN={results['fn']}")
    print()
    print(f"Precision      : {results['precision']:.4f}")
    print(f"Recall         : {results['recall']:.4f}")
    print(f"F1             : {results['f1']:.4f}")
    print(f"Accuracy       : {results['accuracy']:.4f}")
    print(f"Trait accuracy : {results['trait_accuracy']:.4f}  (correct trait given any-evidence-fired)")

    # ClearML scalar logging — these are what the HPO optimiser reads
    if task is not None:
        logger = task.get_logger()
        # Report as proper scalars so ClearML's HPO can read them as the
        # optimisation target. Iteration=0 since we run a single evaluation.
        for name in ("precision", "recall", "f1", "accuracy", "trait_accuracy"):
            logger.report_scalar(title=name, series=name, value=results[name], iteration=0)
        for name in ("tp", "fp", "tn", "fn"):
            logger.report_scalar(title=name, series=name, value=results[name], iteration=0)
        # Also report as single values for the nicer dashboard summary panel.
        for name in ("precision", "recall", "f1", "accuracy", "trait_accuracy"):
            logger.report_single_value(name, results[name])
        task.close()

    return results


if __name__ == "__main__":
    main()
