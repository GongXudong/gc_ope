"""Offline KDE/GMM prediction check on vanilla SAC evaluation CSVs.

Example:
  conda run -n gc_ope python scripts/evaluate_gmm_prediction.py --env push --seed 1 \
      --checkpoint 100000 --output-dir logs/gmm_prediction
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_gmm import GMMEvaluator
from gc_ope.evaluate.evaluator_kde import KDEEvaluator


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"rl_model_(\d+)_steps_eval_res_on_fixed\.csv$")
# Push/Slide keep z fixed.  Comparing densities in the effective x-y goal
# space avoids treating that deterministic coordinate as a continuous random
# variable with an arbitrary GMM regularization width.
GOAL_COLUMNS = ["x", "y"]


def checkpoint_file(env: str, seed: int, timestep: int) -> Path:
    path = ROOT / "checkpoints" / f"my_{env}" / "sac" / f"seed_{seed}" / f"rl_model_{timestep}_steps_eval_res_on_fixed.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def historical_files(env: str, seed: int, checkpoint: int) -> list[tuple[int, Path]]:
    directory = ROOT / "checkpoints" / f"my_{env}" / "sac" / f"seed_{seed}"
    found = []
    for path in directory.glob("rl_model_*_steps_eval_res_on_fixed.csv"):
        match = CHECKPOINT_RE.search(path.name)
        if match and int(match.group(1)) < checkpoint:
            found.append((int(match.group(1)), path))
    return sorted(found)


def add_records(
    evaluator,
    frames: list[tuple[int, pd.DataFrame]],
    checkpoint: int,
    kappa: float,
    goal_columns: list[str] | None = None,
):
    columns = GOAL_COLUMNS if goal_columns is None else list(goal_columns)
    goals, successes, weights = [], [], []
    for timestep, frame in frames:
        goals.append(frame[columns].to_numpy(dtype=float))
        successes.extend((frame["termination"].to_numpy() == "reach target").tolist())
        weights.extend([float(kappa ** ((checkpoint - timestep) / 10000.0))] * len(frame))
    if not goals:
        raise ValueError("no historical evaluation files precede the selected checkpoint")
    goals_arr = np.concatenate(goals, axis=0)
    evaluator.eval_res_container.add_batch(
        goals_arr,
        successes,
        [0.0] * len(goals_arr),
        [0.0] * len(goals_arr),
        [1.0] * len(goals_arr),
    )
    # Set explicit absolute time weights after insertion.  The legacy container
    # applies decay between batches, which is not the desired checkpoint-time rule.
    evaluator.eval_res_container.desired_goal_weights = np.asarray(weights, dtype=float)
    return goals_arr, np.asarray(successes, dtype=bool), np.asarray(weights, dtype=float)


def discrete_kl(reference_goals: np.ndarray, grid: np.ndarray, predicted_density: np.ndarray) -> float:
    """KL(q||p) on the fixed evaluation grid, q from reference successes.

    注意：这是离散格点近似，不是连续 KL。两个估计器在同一格点集上可比，
    但绝对值会偏大，只作为辅助诊断。
    """
    if len(reference_goals) == 0:
        return float("nan")
    # Fixed CSVs share the same goal ordering; matching by rows also tolerates
    # floating point serialization differences through nearest exact lookup.
    q = np.zeros(len(grid), dtype=float)
    for goal in reference_goals:
        idx = int(np.argmin(np.sum((grid - goal) ** 2, axis=1)))
        q[idx] += 1.0
    q /= q.sum()
    p = np.maximum(np.asarray(predicted_density, dtype=float), np.finfo(float).tiny)
    p /= p.sum()
    return float(np.sum(q * (np.log(q + np.finfo(float).tiny) - np.log(p))))


def run(args: argparse.Namespace) -> dict:
    ref_path = Path(args.reference_eval_file) if args.reference_eval_file else checkpoint_file(args.env, args.seed, args.checkpoint)
    reference = pd.read_csv(ref_path)
    history = [(t, pd.read_csv(path)) for t, path in historical_files(args.env, args.seed, args.checkpoint)]
    kde = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": args.kappa},
        kde_bandwidth=args.kde_bandwidth,
    )
    gmm = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": args.kappa},
        n_components=args.n_components,
        resample_size=args.resample_size,
        random_state=args.random_state,
        reg_covar=args.gmm_reg_covar,
    )
    hist_goals, hist_success, hist_weights = add_records(gmm, history, args.checkpoint, args.kappa)
    # Feed exactly the same records into KDE.
    kde.eval_res_container.add_batch(hist_goals, hist_success.tolist(), [0.0] * len(hist_goals), [0.0] * len(hist_goals), [1.0] * len(hist_goals))
    kde.eval_res_container.desired_goal_weights = hist_weights
    gmm.fit_evaluator()
    kde.fit_evaluator()
    grid = reference[GOAL_COLUMNS].to_numpy(dtype=float)
    _, gmm_density = gmm.evaluate(grid)
    _, kde_density = kde.evaluate(grid)
    ref_success = reference.loc[reference["termination"] == "reach target", GOAL_COLUMNS].to_numpy(dtype=float)
    return {
        "env": args.env,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "reference_file": str(ref_path),
        "goal_columns": GOAL_COLUMNS,
        "historical_files": len(history),
        "historical_rows": int(len(hist_goals)),
        "historical_successes": int(hist_success.sum()),
        "effective_weight_sum": float(hist_weights[hist_success].sum()),
        "reference_rows": int(len(reference)),
        "reference_successes": int(len(ref_success)),
        "reference_success_rate": float(len(ref_success) / len(reference)),
        "gmm_components": int(gmm.gmm.n_components),
        "gmm_reg_covar": float(args.gmm_reg_covar),
        "gmm_fit_diagnostics": gmm.fit_diagnostics_,
        "gmm_converged": bool(gmm.gmm.converged_),
        "gmm_kl_reference_to_prediction": discrete_kl(ref_success, grid, gmm_density),
        "kde_kl_reference_to_prediction": discrete_kl(ref_success, grid, kde_density),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["push", "slide"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--kappa", type=float, default=0.9)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--resample-size", type=int, default=1000)
    # Keep the baseline identical to the paper/configured PE-GCRL KDE.
    parser.add_argument("--kde-bandwidth", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    parser.add_argument("--reference-eval-file")
    parser.add_argument("--output-dir", default="logs/gmm_prediction")
    args = parser.parse_args()
    result = run(args)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.env}_seed_{args.seed}_{args.checkpoint}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
