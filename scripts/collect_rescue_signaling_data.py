from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_rescue_data import make_sampled_problem
from src.rescue.signaling import (
    RescueSignalingDataset,
    action_label,
    rescue_signaling_metadata,
    rescue_state_features,
    save_rescue_signaling_dataset,
)
from src.rescue.testbed import (
    ShortestPathOracle,
    initial_state,
    is_terminal,
    load_problem_from_config,
    non_autonomous_rollout_action,
    transition,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect rescue learned-signaling examples from rollout expert.")
    parser.add_argument("--config", type=str, default="rescue_config.yaml", help="Base rescue config.")
    parser.add_argument("--output", type=str, required=True, help="Output dataset .npz path.")
    parser.add_argument("--episodes", type=int, default=25, help="Number of sampled expert episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Scenario sampling seed.")
    parser.add_argument("--max-time-steps", type=int, default=None, help="Optional expert episode length cap.")
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
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_episode(problem, *, max_time_steps=None):
    oracle = ShortestPathOracle(problem.adjacency)
    state = initial_state(problem)
    max_steps = int(max_time_steps if max_time_steps is not None else problem.max_time_steps)
    X_rows = []
    y_rows = []

    while not is_terminal(problem, state) and state.step_idx < max_steps:
        expert_action = non_autonomous_rollout_action(problem, state, oracle)
        for agent_idx, next_node in enumerate(expert_action):
            X_rows.append(rescue_state_features(problem, state, agent_idx))
            y_rows.append(action_label(problem, state.agent_nodes[agent_idx], next_node))
        state = transition(problem, state, expert_action)

    return X_rows, y_rows


def collect_dataset(
    base_problem,
    *,
    episodes: int,
    seed: int,
    max_time_steps=None,
    sample_lost: bool = True,
    sample_agents: bool = False,
    source_config=None,
) -> RescueSignalingDataset:
    rng = np.random.default_rng(seed)
    all_X = []
    all_y = []

    for episode_idx in range(int(episodes)):
        problem = make_sampled_problem(
            base_problem,
            rng,
            sample_lost=sample_lost,
            sample_agents=sample_agents,
        )
        X_rows, y_rows = collect_episode(problem, max_time_steps=max_time_steps)
        all_X.extend(X_rows)
        all_y.extend(y_rows)
        print(f"signaling episode={episode_idx + 1}/{episodes} rows={len(y_rows)}")

    if len(all_X) == 0:
        raise RuntimeError("collected zero rescue signaling examples")

    metadata = rescue_signaling_metadata(
        base_problem,
        source_config=source_config,
        episodes=episodes,
        seed=seed,
        sample_lost=sample_lost,
        sample_agents=sample_agents,
    )
    return RescueSignalingDataset(
        X_train=np.vstack(all_X),
        y_train=np.asarray(all_y, dtype=int),
        metadata=metadata,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    base_problem = load_problem_from_config(config)
    dataset = collect_dataset(
        base_problem,
        episodes=args.episodes,
        seed=args.seed,
        max_time_steps=args.max_time_steps,
        sample_lost=args.sample_lost,
        sample_agents=args.sample_agents,
        source_config=args.config,
    )
    save_rescue_signaling_dataset(args.output, dataset.X_train, dataset.y_train, dataset.metadata)

    metadata_path = Path(args.output).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({**dataset.metadata, "rows": int(dataset.y_train.shape[0])}, f, indent=2)

    print(f"saved {dataset.y_train.shape[0]} rescue signaling examples to {args.output}")


if __name__ == "__main__":
    main()
