"""
mlops/hpo_sweep.py
==================
ClearML HyperParameterOptimizer sweep over the Layer B similarity threshold.

This script does NOT run the evaluations itself — it tells ClearML to clone
the base evaluation task (created once from train_and_eval.py) many times
with different --threshold values, and to collect their f1 scalar metrics.

ClearML's Optimizer chooses the next set of trials based on the strategy
(random or Bayesian). For this small search (one continuous hyperparameter)
random search is more than adequate and cheaper than Bayesian.

Prerequisite: a single execution of `train_and_eval.py` must have already
created a Task in ClearML (the "base task" we clone from). Get its task_id
from the ClearML UI or from the printed output of train_and_eval.py.

Usage:
    # Step 1 — run once to seed a base task:
    python mlops/train_and_eval.py --threshold 0.72 --embedder minilm

    # Step 2 — copy the printed Task ID, then:
    python mlops/hpo_sweep.py --base-task-id <task_id>
"""

from __future__ import annotations

import argparse
import time
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    UniformParameterRange,
    HyperParameterOptimizer,
)
from clearml.automation.optimization import GridSearch, RandomSearch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-task-id", required=True,
                    help="ClearML task ID of a seeded train_and_eval.py run")
    ap.add_argument("--project", default="Megan-AISM",
                    help="ClearML project name for the optimiser task")
    ap.add_argument("--total-jobs", type=int, default=15,
                    help="Number of trial runs to launch")
    ap.add_argument("--strategy", default="grid", choices=["grid", "random"],
                    help="Optimizer search strategy")
    args = ap.parse_args()

    optimizer_task = Task.init(
        project_name=args.project,
        task_name="hpo_layerb_threshold_sweep",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    # The hyperparameter search space.
    # Layer B SIMILARITY_THRESHOLD lives in [0, 1]; we sweep the meaningful
    # range based on the production default of 0.72 and the locally observed
    # curve transitions in the 0.50–0.70 zone.
    if args.strategy == "grid":
        # Fixed grid — reproducible and easy to read on the dashboard.
        thresholds = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85]
        hyperparams = [DiscreteParameterRange(
            "Args/threshold", values=thresholds,
        )]
        method_class = GridSearch
    else:
        # Random search over a continuous range.
        hyperparams = [UniformParameterRange(
            "Args/threshold", min_value=0.50, max_value=0.85, step_size=0.025,
        )]
        method_class = RandomSearch

    optimizer = HyperParameterOptimizer(
        base_task_id=args.base_task_id,
        hyper_parameters=hyperparams,
        # The scalar metric to optimise. Must match the title/series of a
        # scalar reported by train_and_eval.py.
        objective_metric_title="f1",
        objective_metric_series="f1",
        objective_metric_sign="max",
        # Search method
        optimizer_class=method_class,
        # Concurrency: 1 keeps things simple for a local single-GPU machine.
        max_number_of_concurrent_tasks=1,
        # Stop after this many trials.
        total_max_jobs=args.total_jobs,
        execution_queue="default",
        pool_period_min=0.2,   # poll every 12s, not every 5min
    )

    optimizer.start()
    optimizer.wait()
    optimizer.stop()

    # Pull top results and write them into the optimiser task's console log.
    top = optimizer.get_top_experiments(top_k=5)
    print("\nTop 5 trials by F1:")
    for i, t in enumerate(top, 1):
        params = t.get_parameters() or {}
        f1 = t.get_last_scalar_metrics().get("f1", {}).get("f1", {}).get("last")
        print(f"  {i}. threshold={params.get('Args/threshold')}  F1={f1}")


if __name__ == "__main__":
    main()
