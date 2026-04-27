from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rescue.testbed import (
    RESCUE_STRATEGIES,
    GraphSearchProblem,
    grid_coordinates,
    grid_graph,
    grid_node_id,
    load_problem_from_config,
    run_all_strategies,
    run_rescue_simulation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect rescue test-bed metrics over sampled scenarios.")
    parser.add_argument("--config", type=str, default="rescue_config.yaml", help="Base rescue config.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of sampled scenarios.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--strategy",
        choices=["all", *RESCUE_STRATEGIES],
        default="all",
        help="Strategy to run for each sampled scenario.",
    )
    parser.add_argument(
        "--sample-lost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample lost-individual nodes each episode.",
    )
    parser.add_argument(
        "--sample-agents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample agent start nodes each episode.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Also save full trajectories to a JSON file next to the CSV.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_sampled_problem(base_problem: GraphSearchProblem, rng, *, sample_lost: bool, sample_agents: bool):
    occupied = set()
    num_agents = len(base_problem.agent_start_nodes)
    num_lost = len(base_problem.lost_individual_nodes)
    all_nodes = np.array(sorted(base_problem.adjacency), dtype=int)

    if sample_agents:
        sampled_agents = _sample_unique_nodes(rng, all_nodes, num_agents, occupied)
    else:
        sampled_agents = tuple(base_problem.agent_start_nodes)
    occupied.update(sampled_agents)

    if sample_lost:
        sampled_lost = _sample_unique_nodes(rng, all_nodes, num_lost, occupied)
    else:
        sampled_lost = tuple(base_problem.lost_individual_nodes)

    return GraphSearchProblem(
        adjacency=base_problem.adjacency,
        agent_start_nodes=sampled_agents,
        lost_individual_nodes=sampled_lost,
        max_time_steps=base_problem.max_time_steps,
        target_knowledge=base_problem.target_knowledge,
        discount_factor=base_problem.discount_factor,
        revisit_penalty=base_problem.revisit_penalty,
        grid_width=base_problem.grid_width,
        grid_height=base_problem.grid_height,
    )


def _sample_unique_nodes(rng, all_nodes, count, excluded):
    available = np.array([node for node in all_nodes if int(node) not in excluded], dtype=int)
    if count > available.shape[0]:
        raise ValueError("not enough available graph nodes to sample scenario")
    chosen = rng.choice(available, size=count, replace=False)
    return tuple(int(node) for node in chosen)


def run_strategies(problem, strategy):
    if strategy == "all":
        return run_all_strategies(problem)
    return [run_rescue_simulation(problem, strategy)]


def metric_rows_for_episode(episode_idx, problem, results):
    rows = []
    for result in results:
        row = dict(result.metrics)
        row.update(
            {
                "episode": episode_idx,
                "agent_start_nodes": json.dumps(list(problem.agent_start_nodes)),
                "lost_individual_nodes": json.dumps(list(problem.lost_individual_nodes)),
            }
        )
        if problem.grid_width is not None:
            row["agent_start_coordinates"] = json.dumps(
                [list(grid_coordinates(node, problem.grid_width)) for node in problem.agent_start_nodes]
            )
            row["lost_individual_coordinates"] = json.dumps(
                [list(grid_coordinates(node, problem.grid_width)) for node in problem.lost_individual_nodes]
            )
        rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    base_problem = load_problem_from_config(config)
    rng = np.random.default_rng(args.seed)

    metric_rows = []
    trajectory_rows = []
    for episode_idx in range(args.episodes):
        problem = make_sampled_problem(
            base_problem,
            rng,
            sample_lost=args.sample_lost,
            sample_agents=args.sample_agents,
        )
        results = run_strategies(problem, args.strategy)
        metric_rows.extend(metric_rows_for_episode(episode_idx, problem, results))

        if args.save_trajectories:
            for result in results:
                trajectory_rows.append(
                    {
                        "episode": episode_idx,
                        "strategy": result.strategy,
                        "trajectory": result.trajectory,
                    }
                )

        print(f"episode={episode_idx + 1}/{args.episodes} rows={len(results)}")

    write_csv(args.output, metric_rows)
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2)

    if args.save_trajectories:
        trajectory_path = Path(args.output).with_name(Path(args.output).stem + "_trajectories.json")
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_rows, f, indent=2)

    print(f"saved {len(metric_rows)} metric rows to {args.output}")


if __name__ == "__main__":
    main()
