"""
mlops/multi_embedder.py
=======================
Multi-model comparison: runs the same threshold sweep across three embedder
choices and reports which embedder × threshold combination scores best on
the labelled evaluation set.

The three "models" being compared:

    charngram :  CharacterNgramEmbedder         — zero-dependency fallback
    minilm    :  all-MiniLM-L6-v2 (~90 MB)      — production sentence encoder
    mpnet     :  all-mpnet-base-v2 (~420 MB)    — larger, slower, possibly better

Each combination is logged to ClearML as its own Task so the dashboard shows
3 × N trials, comparable in one parallel-coordinates plot.

Usage:
    python mlops/multi_embedder.py --thresholds 0.55 0.60 0.65 0.70 0.72 0.75 0.80

(The first run with minilm or mpnet will download the model weights from
HuggingFace; subsequent runs are offline.)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
TRAIN_EVAL = THIS_DIR / "train_and_eval.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80],
                    help="Layer B similarity thresholds to sweep")
    ap.add_argument("--embedders", nargs="+",
                    default=["charngram", "minilm", "mpnet"],
                    choices=["charngram", "minilm", "mpnet"])
    ap.add_argument("--project", default="Megan-AISM",
                    help="ClearML project name")
    args = ap.parse_args()

    print(f"Running {len(args.embedders)} embedders × {len(args.thresholds)} thresholds"
          f" = {len(args.embedders) * len(args.thresholds)} ClearML tasks")
    print()

    for embedder in args.embedders:
        for thr in args.thresholds:
            task_name = f"multi_emb={embedder}_thr={thr}"
            print(f"▶ {task_name}")
            cmd = [
                sys.executable, str(TRAIN_EVAL),
                "--threshold", str(thr),
                "--embedder", embedder,
                "--project", args.project,
                "--task-name", task_name,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Extract & echo just the F1 line for progress visibility
            for line in result.stdout.splitlines():
                if line.startswith("F1") or line.startswith("Precision") or line.startswith("Recall"):
                    print(f"    {line}")
            if result.returncode != 0:
                print(f"    !! task failed:\n{result.stderr[-400:]}")
            print()


if __name__ == "__main__":
    main()
