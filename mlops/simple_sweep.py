"""
mlops/simple_sweep.py
=====================
Local sweep over Layer B SIMILARITY_THRESHOLD. Calls train_and_eval.py once
per threshold value. Each call creates its own ClearML task. Compare them on
the dashboard afterward.

No queue, no worker, no HyperParameterOptimizer. Just runs.
"""

import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
TRAIN_EVAL = THIS_DIR / "train_and_eval.py"

THRESHOLDS = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85]
EMBEDDER = "minilm"

print(f"Sweeping {len(THRESHOLDS)} thresholds with embedder={EMBEDDER}")
print()

for thr in THRESHOLDS:
    task_name = f"sweep_thr={thr}_emb={EMBEDDER}"
    print(f"-> {task_name}")
    cmd = [
        sys.executable, str(TRAIN_EVAL),
        "--threshold", str(thr),
        "--embedder", EMBEDDER,
        "--task-name", task_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith(("Precision", "Recall", "F1", "Confusion")):
            print(f"   {line}")
    if result.returncode != 0:
        print(f"   FAILED: {result.stderr[-300:]}")
    print()

print("Done. View all 11 tasks at https://app.clear.ml/projects/")
