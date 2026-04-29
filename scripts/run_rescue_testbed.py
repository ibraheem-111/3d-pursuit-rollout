from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rescue import load_problem_from_config, run_all_strategies, run_rescue_simulation
from src.rescue.signaling import RescueLearnedSignalingPolicy
from src.rescue.testbed import RESCUE_STRATEGIES
from src.rescue.visualization import plot_rescue_trajectory, save_rescue_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Run the rescue graph-search test bed.")
    parser.add_argument("--config", type=str, default="rescue_config.yaml", help="Path to rescue test-bed config.")
    parser.add_argument(
        "--strategy",
        choices=["all", *RESCUE_STRATEGIES],
        default="all",
        help="Strategy to run. Use all to compare baseline strategies.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--signaling-model", type=str, default=None, help="Path to rescue learned-signaling model.")
    parser.add_argument(
        "--signaling-model-type",
        choices=["kernel_knn", "mlp"],
        default=None,
        help="Expected rescue learned-signaling model type.",
    )
    parser.add_argument("--plot", action="store_true", help="Save a trajectory PNG for each strategy.")
    parser.add_argument("--save-gif", action="store_true", help="Save an animated GIF for each strategy.")
    parser.add_argument(
        "--show-hidden-lost",
        action="store_true",
        help="Show unfound lost-individual nodes in unknown-target visualizations.",
    )
    return parser.parse_args()


def create_output_dir(output_dir=None):
    if output_dir is not None:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        path = Path("outputs") / f"rescue_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_signaling_policy(args, config):
    if args.strategy != "autonomous_learned_signaling":
        return None

    rescue_config = config.get("rescue", config)
    signaling_config = rescue_config.get("signaling", {})
    model_path = args.signaling_model or signaling_config.get("model_path")
    model_type = args.signaling_model_type or signaling_config.get("model_type")
    if model_path is None or not Path(model_path).exists():
        raise ValueError(_missing_signaling_model_message())
    return RescueLearnedSignalingPolicy.load(model_path, model_type=model_type)


def _missing_signaling_model_message():
    return (
        "autonomous_learned_signaling requires a trained rescue signaling model.\n"
        "Collect data:\n"
        "  uv run python scripts/collect_rescue_signaling_data.py --config rescue_config.yaml "
        "--output models/rescue_signaling_dataset.npz --episodes 50\n"
        "Train a model:\n"
        "  uv run python scripts/train_rescue_signaling_model.py --dataset models/rescue_signaling_dataset.npz "
        "--output models/rescue_signaling_kernel.npz --model-type kernel_knn\n"
        "Then rerun with --signaling-model models/rescue_signaling_kernel.npz"
    )


def write_metrics_csv(path, results):
    rows = [result.metrics for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    problem = load_problem_from_config(config)
    signaling_policy = load_signaling_policy(args, config)

    if args.strategy == "all":
        results = run_all_strategies(problem)
    else:
        results = [run_rescue_simulation(problem, args.strategy, signaling_policy=signaling_policy)]

    output_dir = create_output_dir(args.output_dir)
    shutil.copy2(args.config, output_dir / "config.yaml")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump([result.metrics for result in results], f, indent=2)

    write_metrics_csv(output_dir / "metrics.csv", results)

    trajectories_path = output_dir / "trajectories.json"
    with open(trajectories_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                result.strategy: result.trajectory
                for result in results
            },
            f,
            indent=2,
        )

    for result in results:
        if args.plot:
            plot_rescue_trajectory(
                problem,
                result,
                output_path=output_dir / f"{result.strategy}_trajectory.png",
                show_hidden_lost=args.show_hidden_lost,
            )
        if args.save_gif:
            save_rescue_gif(
                problem,
                result,
                output_path=output_dir / f"{result.strategy}_animation.gif",
                show_hidden_lost=args.show_hidden_lost,
            )

    for result in results:
        print(
            f"{result.strategy}: all_found={result.metrics['all_found']} "
            f"time={result.metrics['time_to_find_all']} "
            f"total_cost={result.metrics['total_search_cost']} "
            f"explored={result.metrics['explored_fraction']:.3f}"
        )
    print(f"saved rescue test-bed outputs to {output_dir}")


if __name__ == "__main__":
    main()
