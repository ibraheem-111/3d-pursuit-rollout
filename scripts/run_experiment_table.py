from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_signaling_data import collect_episode
from src.grid import Grid3D
from src.planner.signaling_policy import KernelSignalingModel
from src.simulation.simulation import run_simulation


STRATEGIES = (
    "greedy",
    "non_autonomous_rollout",
    "autonomous_greedy_signaling",
    "autonomous_learned_signaling",
)

METHOD_LABELS = {
    "greedy": "Greedy",
    "non_autonomous_rollout": "Non-autonomous rollout",
    "autonomous_greedy_signaling": "Autonomous rollout + greedy signaling",
    "autonomous_learned_signaling": "Autonomous rollout + learned signaling",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated pursuit experiments and produce final-table metrics."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Experiment config.")
    parser.add_argument("--runs", type=int, default=50, help="Runs per strategy.")
    parser.add_argument("--base-seed", type=int, default=0, help="First seed used for each strategy.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="Strategies to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/experiments/<timestamp>.",
    )
    parser.add_argument("--signaling-model", type=str, default=None, help="Learned signaling model path.")
    parser.add_argument(
        "--signaling-episodes",
        type=int,
        default=50,
        help="Expert episodes used when a signaling model must be collected.",
    )
    parser.add_argument(
        "--refresh-signaling-model",
        action="store_true",
        help="Always rebuild the kernel signaling model before learned-signaling runs.",
    )
    parser.add_argument(
        "--keep-rollout-logs",
        action="store_true",
        help="Do not suppress verbose rollout candidate-action logs.",
    )
    parser.add_argument(
        "--capture-time-policy",
        choices=["captured_only", "max_steps_for_failures"],
        default="captured_only",
        help="How to average capture time when some runs fail to capture.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dir(path):
    if path is None:
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        path = Path("outputs") / "experiments" / timestamp
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configured_signaling_model_path(args, config):
    if args.signaling_model is not None:
        return Path(args.signaling_model)
    planner_config = config.get("planner", {})
    return Path(planner_config.get("signaling_model_path", "models/signaling_kernel.npz"))


def model_matches_config(model_path, config):
    if not model_path.exists():
        return False
    try:
        model = KernelSignalingModel.load(model_path)
    except Exception:
        return False

    grid_size = tuple(int(v) for v in config["simulation"]["grid_size"])
    num_pursuers = len(config["pursuers"])
    planner_config = config.get("planner", {})
    include_evader_position = bool(planner_config.get("signaling_include_evader_position", True))
    normalize = bool(planner_config.get("signaling_normalize", True))
    expected_width = 3 * num_pursuers
    if include_evader_position:
        expected_width += 3

    return (
        tuple(model.grid_size) == grid_size
        and model.X_train.shape[1] == expected_width
        and bool(model.include_evader_position) == include_evader_position
        and bool(model.normalize) == normalize
    )


def ensure_signaling_model(args, config, model_path):
    if "autonomous_learned_signaling" not in args.strategies:
        return False
    if not args.refresh_signaling_model and model_matches_config(model_path, config):
        return False

    planner_config = config.get("planner", {})
    collector_args = SimpleNamespace(
        max_time_steps=None,
        k=int(planner_config.get("signaling_k", 25)),
        sigma=float(planner_config.get("signaling_sigma", 5.0)),
        include_evader_position=bool(planner_config.get("signaling_include_evader_position", True)),
        normalize=bool(planner_config.get("signaling_normalize", True)),
        verbose=args.keep_rollout_logs,
    )

    all_X = []
    all_y = []
    for episode_idx in range(args.signaling_episodes):
        episode_seed = int(args.base_seed + 100_000 + episode_idx)
        X_rows, y_rows = collect_episode(config, collector_args, episode_seed)
        all_X.extend(X_rows)
        all_y.extend(y_rows)
        print(f"signaling data episode={episode_idx + 1}/{args.signaling_episodes} rows={len(X_rows)}")

    if len(all_X) == 0:
        raise RuntimeError("collected zero signaling examples")

    metadata = {
        "source_strategy": "non_autonomous_rollout",
        "episodes": int(args.signaling_episodes),
        "seed": int(args.base_seed + 100_000),
        "k": int(collector_args.k),
        "sigma": float(collector_args.sigma),
        "include_evader_position": bool(collector_args.include_evader_position),
        "normalize": bool(collector_args.normalize),
        "num_evaders": len(config["evaders"]),
        "num_pursuers": len(config["pursuers"]),
        "config": str(args.config),
    }
    model = KernelSignalingModel(
        X_train=np.vstack(all_X),
        y_train=np.asarray(all_y, dtype=int),
        grid_size=tuple(int(v) for v in config["simulation"]["grid_size"]),
        k=int(collector_args.k),
        sigma=float(collector_args.sigma),
        include_evader_position=bool(collector_args.include_evader_position),
        normalize=bool(collector_args.normalize),
        metadata=metadata,
    )
    model.save(model_path)
    with open(model_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({**metadata, "rows": len(all_y)}, f, indent=2)
    return True


def run_one(config, strategy, seed, signaling_model_path, keep_rollout_logs):
    run_config = copy.deepcopy(config)
    run_config["strategy"] = strategy
    width, height, depth = run_config["simulation"]["grid_size"]
    grid = Grid3D(width, height, depth)
    args = SimpleNamespace(
        strategy=strategy,
        evader_type=None,
        pursuer_type=None,
        seed=int(seed),
        signaling_model=str(signaling_model_path),
        signaling_k=None,
        signaling_sigma=None,
    )

    log_context = contextlib.nullcontext()
    if not keep_rollout_logs:
        log_context = contextlib.redirect_stdout(io.StringIO())
    with log_context:
        result = run_simulation(grid, args, config=run_config)

    metrics = dict(result["metrics"])
    metrics["method"] = METHOD_LABELS[strategy]
    metrics["run_index"] = int(seed)
    return metrics


def summarize(raw_df, capture_time_policy):
    rows = []
    for strategy in STRATEGIES:
        strategy_df = raw_df[raw_df["strategy"] == strategy]
        if strategy_df.empty:
            continue

        capture_rate = float(strategy_df["capture_occurred"].mean())
        if capture_time_policy == "captured_only":
            capture_times = strategy_df.loc[strategy_df["capture_occurred"], "time_to_capture"].dropna()
            avg_capture_time = float(capture_times.mean()) if len(capture_times) > 0 else np.nan
        else:
            capture_time_values = strategy_df["time_to_capture"].fillna(strategy_df["max_time_steps"])
            avg_capture_time = float(capture_time_values.mean())

        rows.append(
            {
                "Method": METHOD_LABELS[strategy],
                "Capture Rate": capture_rate,
                "Avg. Capture Time": avg_capture_time,
                "Avg. Cost": float(strategy_df["total_stage_cost"].mean()),
                "Runtime / Step": float(strategy_df["mean_runtime_per_step"].mean()),
                "Runs": int(len(strategy_df)),
            }
        )

    return pd.DataFrame(rows)


def format_summary(summary_df):
    formatted = summary_df.copy()
    formatted["Capture Rate"] = formatted["Capture Rate"].map(lambda value: f"{100.0 * value:.1f}%")
    formatted["Avg. Capture Time"] = formatted["Avg. Capture Time"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
    )
    formatted["Avg. Cost"] = formatted["Avg. Cost"].map(lambda value: f"{value:.2f}")
    formatted["Runtime / Step"] = formatted["Runtime / Step"].map(lambda value: f"{value:.4f}s")
    return formatted


def write_outputs(output_dir, config_path, config, raw_df, summary_df):
    shutil.copy2(config_path, output_dir / "config.yaml")
    raw_df.to_csv(output_dir / "raw_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)

    formatted = format_summary(summary_df)
    with open(output_dir / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(formatted.drop(columns=["Runs"])))
    with open(output_dir / "summary_table.tex", "w", encoding="utf-8") as f:
        f.write(formatted.drop(columns=["Runs"]).to_latex(index=False, escape=False))

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(config_path),
                "grid_size": config["simulation"]["grid_size"],
                "num_evaders": len(config["evaders"]),
                "num_pursuers": len(config["pursuers"]),
            },
            f,
            indent=2,
        )

    print("\nFinal table:")
    print(formatted.drop(columns=["Runs"]).to_string(index=False))
    print(f"\nWrote experiment outputs to {output_dir}")


def markdown_table(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _idx, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = make_output_dir(args.output_dir)
    signaling_model_path = configured_signaling_model_path(args, config)
    signaling_model_path.parent.mkdir(parents=True, exist_ok=True)

    rebuilt_model = ensure_signaling_model(args, config, signaling_model_path)
    if rebuilt_model:
        print(f"saved signaling model to {signaling_model_path}")

    raw_rows = []
    for strategy in args.strategies:
        for run_idx in range(args.runs):
            seed = int(args.base_seed + run_idx)
            metrics = run_one(
                config=config,
                strategy=strategy,
                seed=seed,
                signaling_model_path=signaling_model_path,
                keep_rollout_logs=args.keep_rollout_logs,
            )
            raw_rows.append(metrics)
            print(
                f"{METHOD_LABELS[strategy]} run={run_idx + 1}/{args.runs} "
                f"seed={seed} capture={metrics['capture_occurred']} "
                f"cost={metrics['total_stage_cost']:.2f}"
            )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df, args.capture_time_policy)
    write_outputs(output_dir, config_path, config, raw_df, summary_df)


if __name__ == "__main__":
    main()
