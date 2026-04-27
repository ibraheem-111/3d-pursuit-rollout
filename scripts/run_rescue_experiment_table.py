from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_rescue_data import make_sampled_problem
from src.rescue.testbed import RESCUE_STRATEGIES, load_problem_from_config, run_rescue_simulation


METHOD_LABELS = {
    "greedy": "Greedy closest-unexplored",
    "non_autonomous_rollout": "Non-autonomous rollout",
    "autonomous_greedy_signaling": "Autonomous rollout + greedy signaling",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated rescue-search experiments and produce final-table metrics."
    )
    parser.add_argument("--config", type=str, default="rescue_config.yaml", help="Rescue experiment config.")
    parser.add_argument("--runs", type=int, default=50, help="Sampled scenarios per strategy.")
    parser.add_argument("--base-seed", type=int, default=0, help="First seed for sampled scenarios.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=RESCUE_STRATEGIES,
        default=list(RESCUE_STRATEGIES),
        help="Strategies to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/rescue_experiments/<timestamp>.",
    )
    parser.add_argument(
        "--sample-lost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample lost-individual locations each run.",
    )
    parser.add_argument(
        "--sample-agents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample agent starts each run.",
    )
    parser.add_argument(
        "--find-time-policy",
        choices=["found_only", "max_steps_for_failures"],
        default="found_only",
        help="How to average find time when some runs fail.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_output_dir(path):
    if path is None:
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        path = Path("outputs") / "rescue_experiments" / timestamp
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sampled_problems(base_problem, args):
    rng = np.random.default_rng(args.base_seed)
    return [
        make_sampled_problem(
            base_problem,
            rng,
            sample_lost=args.sample_lost,
            sample_agents=args.sample_agents,
        )
        for _run_idx in range(args.runs)
    ]


def run_one(problem, strategy, run_idx):
    result = run_rescue_simulation(problem, strategy)
    row = dict(result.metrics)
    row.update(
        {
            "method": METHOD_LABELS[strategy],
            "run_index": int(run_idx),
            "agent_start_nodes": json.dumps(list(problem.agent_start_nodes)),
            "lost_individual_nodes": json.dumps(list(problem.lost_individual_nodes)),
        }
    )
    return row


def summarize(raw_df, strategies, find_time_policy):
    rows = []
    for strategy in strategies:
        strategy_df = raw_df[raw_df["strategy"] == strategy]
        if strategy_df.empty:
            continue

        find_rate = float(strategy_df["all_found"].mean())
        if find_time_policy == "found_only":
            find_times = strategy_df.loc[strategy_df["all_found"], "time_to_find_all"].dropna()
            avg_find_time = float(find_times.mean()) if len(find_times) > 0 else np.nan
        else:
            find_time_values = strategy_df["time_to_find_all"].fillna(strategy_df["max_time_steps"])
            avg_find_time = float(find_time_values.mean())

        rows.append(
            {
                "Method": METHOD_LABELS[strategy],
                "Find Rate": find_rate,
                "Avg. Time to Find All": avg_find_time,
                "Avg. Search Cost": float(strategy_df["total_search_cost"].mean()),
                "Avg. Discounted Cost": float(strategy_df["discounted_search_cost"].mean()),
                "Avg. Explored Fraction": float(strategy_df["explored_fraction"].mean()),
                "Runtime / Step": float(strategy_df["mean_runtime_per_step"].mean()),
                "Runs": int(len(strategy_df)),
            }
        )

    return pd.DataFrame(rows)


def format_summary(summary_df):
    formatted = summary_df.copy()
    formatted["Find Rate"] = formatted["Find Rate"].map(lambda value: f"{100.0 * value:.1f}%")
    formatted["Avg. Time to Find All"] = formatted["Avg. Time to Find All"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
    )
    formatted["Avg. Search Cost"] = formatted["Avg. Search Cost"].map(lambda value: f"{value:.2f}")
    formatted["Avg. Discounted Cost"] = formatted["Avg. Discounted Cost"].map(lambda value: f"{value:.2f}")
    formatted["Avg. Explored Fraction"] = formatted["Avg. Explored Fraction"].map(lambda value: f"{value:.3f}")
    formatted["Runtime / Step"] = formatted["Runtime / Step"].map(lambda value: f"{value:.4f}s")
    return formatted


def markdown_table(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _idx, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(output_dir, config_path, config, raw_df, summary_df, args):
    shutil.copy2(config_path, output_dir / "config.yaml")
    raw_df.to_csv(output_dir / "raw_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)

    formatted = format_summary(summary_df)
    table = formatted.drop(columns=["Runs"])
    with open(output_dir / "summary_table.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(table))
    with open(output_dir / "summary_table.tex", "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

    rescue_config = config.get("rescue", config)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(config_path),
                "runs": int(args.runs),
                "base_seed": int(args.base_seed),
                "sample_lost": bool(args.sample_lost),
                "sample_agents": bool(args.sample_agents),
                "graph": rescue_config["graph"],
                "simulation": rescue_config["simulation"],
                "num_agents": len(rescue_config["agents"]["starting_nodes"]),
                "num_lost_individuals": len(rescue_config["lost_individuals"]["nodes"]),
            },
            f,
            indent=2,
        )

    print("\nFinal rescue table:")
    print(table.to_string(index=False))
    print(f"\nWrote rescue experiment outputs to {output_dir}")


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    base_problem = load_problem_from_config(config)
    output_dir = make_output_dir(args.output_dir)
    problems = sampled_problems(base_problem, args)

    raw_rows = []
    for strategy in args.strategies:
        for run_idx, problem in enumerate(problems):
            metrics = run_one(problem, strategy, run_idx)
            raw_rows.append(metrics)
            print(
                f"{METHOD_LABELS[strategy]} run={run_idx + 1}/{args.runs} "
                f"found={metrics['all_found']} cost={metrics['total_search_cost']:.2f}"
            )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df, args.strategies, args.find_time_policy)
    write_outputs(output_dir, config_path, config, raw_df, summary_df, args)


if __name__ == "__main__":
    main()
