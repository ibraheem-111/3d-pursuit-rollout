from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.factory import AgentFactory
from src.data_types import GameState
from src.data_types.postion import Position
from src.grid import Grid3D
from src.planner.base_policy_evaluator import BasePolicyEvaluator
from src.planner.grid_model import GridModel
from src.planner.nonautonomous_rollout import NonAutonomousRolloutPlanner
from src.planner.signaling_kernel import action_label, state_features
from src.planner.signaling_policy import KernelSignalingModel, closest_evader_position


def parse_args():
    parser = argparse.ArgumentParser(description="Collect kernel signaling data from non-autonomous rollout.")
    parser.add_argument("--config", type=str, required=True, help="Path to the simulation config.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path.")
    parser.add_argument("--episodes", type=int, default=25, help="Number of sampled expert episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--max-time-steps", type=int, default=None, help="Optional override for episode length.")
    parser.add_argument("--k", type=int, default=None, help="Default k saved with the model.")
    parser.add_argument("--sigma", type=float, default=None, help="Default kernel bandwidth saved with the model.")
    parser.add_argument(
        "--include-evader-position",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include normalized reference evader position as boundary context.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize coordinates by grid dimensions.",
    )
    parser.add_argument("--verbose", action="store_true", help="Keep expert rollout debug prints.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_agents(config):
    evaders = [
        AgentFactory.create_agent(
            agent_type="evader",
            strategy=evader_config["strategy"],
            name=f"evader_{idx}",
            agent_id=idx + 1,
            position=Position(
                x=int(evader_config["starting_position"][0]),
                y=int(evader_config["starting_position"][1]),
                z=int(evader_config["starting_position"][2]),
            ),
        )
        for idx, evader_config in enumerate(config["evaders"])
    ]

    pursuers = [
        AgentFactory.create_agent(
            agent_type="pursuer",
            strategy=pursuer_config["strategy"],
            name=f"pursuer_{idx}",
            agent_id=len(evaders) + idx + 1,
            position=Position(
                x=int(pursuer_config["starting_position"][0]),
                y=int(pursuer_config["starting_position"][1]),
                z=int(pursuer_config["starting_position"][2]),
            ),
        )
        for idx, pursuer_config in enumerate(config["pursuers"])
    ]

    return evaders, pursuers


def place_agents(grid, evaders, pursuers):
    for agent in evaders + pursuers:
        placed = grid.place_agent(agent.position, agent_id=agent.agent_id, role=agent.role)
        if not placed:
            raise RuntimeError(f"failed to place {agent.name} at {agent.position}")


def collect_episode(config, args, episode_seed):
    grid_size = tuple(int(v) for v in config["simulation"]["grid_size"])
    grid = Grid3D(*grid_size)
    grid_model = GridModel(width=grid.width, height=grid.height, depth=grid.depth)
    evaders, pursuers = make_agents(config)
    place_agents(grid, evaders, pursuers)

    discount_factor = float(config["planner"]["discount_factor"])
    evaluator = BasePolicyEvaluator(grid_model=grid_model, alpha=discount_factor)
    expert = NonAutonomousRolloutPlanner(grid_model=grid_model, base_evaluator=evaluator, alpha=discount_factor)

    planner_rng = np.random.default_rng(episode_seed)
    evader_rng = np.random.default_rng(episode_seed + 10_000)
    max_time_steps = args.max_time_steps or int(config["simulation"]["time_steps"])

    X_rows = []
    y_rows = []
    active_evaders = list(evaders)

    for step_idx in range(max_time_steps):
        if len(active_evaders) == 0:
            break

        state = GameState(
            pursuer_positions=tuple(pursuer.position for pursuer in pursuers),
            evader_positions=tuple(evader.position for evader in active_evaders),
            step_idx=step_idx,
        )

        print_context = contextlib.nullcontext()
        if not args.verbose:
            print_context = contextlib.redirect_stdout(io.StringIO())
        with print_context:
            decision = expert.improve_joint_action(
                state=state,
                pursuer_agents=pursuers,
                evader_agents=active_evaders,
                rng=planner_rng,
            )

        for pursuer_idx, next_position in enumerate(decision.pursuer_next_positions):
            reference_evader = closest_evader_position(state.pursuer_positions[pursuer_idx], state.evader_positions)
            X_rows.append(
                state_features(
                    state,
                    reference_evader_position=reference_evader,
                    focus_pursuer_index=pursuer_idx,
                    grid_size=grid_size,
                    include_evader_position=args.include_evader_position,
                    normalize=args.normalize,
                )
            )
            y_rows.append(action_label(state.pursuer_positions[pursuer_idx], next_position))

        for evader in active_evaders:
            next_evader_position = evader.choose_action_from_state(
                current_position=evader.position,
                grid_model=grid_model,
                pursuer_positions=list(state.pursuer_positions),
                evader_positions=list(state.evader_positions),
                pursuer_agent_ids=[pursuer.agent_id for pursuer in pursuers],
                rng=evader_rng,
            )
            moved = grid.move_agent(evader.position, next_evader_position, agent_id=evader.agent_id)
            if moved:
                evader.move(next_evader_position)

        for pursuer, next_pursuer_position in zip(pursuers, decision.pursuer_next_positions):
            moved = grid.move_agent(pursuer.position, next_pursuer_position, agent_id=pursuer.agent_id)
            if moved:
                pursuer.move(next_pursuer_position)

        active_evaders = [
            evader for evader in active_evaders
            if int(evader.agent_id) in grid.agent_roles_by_id
        ]

    return X_rows, y_rows


def main():
    args = parse_args()
    config = load_config(args.config)
    planner_config = config.get("planner", {})
    if args.k is None:
        args.k = int(planner_config.get("signaling_k", 25))
    if args.sigma is None:
        args.sigma = float(planner_config.get("signaling_sigma", 5.0))
    if args.include_evader_position is None:
        args.include_evader_position = bool(planner_config.get("signaling_include_evader_position", True))
    if args.normalize is None:
        args.normalize = bool(planner_config.get("signaling_normalize", True))

    all_X = []
    all_y = []

    for episode_idx in range(args.episodes):
        episode_seed = int(args.seed + episode_idx)
        X_rows, y_rows = collect_episode(config, args, episode_seed)
        all_X.extend(X_rows)
        all_y.extend(y_rows)
        print(f"episode={episode_idx + 1}/{args.episodes} rows={len(X_rows)}")

    if len(all_X) == 0:
        raise RuntimeError("collected zero signaling examples")

    grid_size = tuple(int(v) for v in config["simulation"]["grid_size"])
    metadata = {
        "source_strategy": "non_autonomous_rollout",
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "k": int(args.k),
        "sigma": float(args.sigma),
        "include_evader_position": bool(args.include_evader_position),
        "normalize": bool(args.normalize),
        "num_evaders": len(config["evaders"]),
        "num_pursuers": len(config["pursuers"]),
        "config": str(Path(args.config)),
    }
    model = KernelSignalingModel(
        X_train=np.vstack(all_X),
        y_train=np.asarray(all_y, dtype=int),
        grid_size=grid_size,
        k=int(args.k),
        sigma=float(args.sigma),
        include_evader_position=bool(args.include_evader_position),
        normalize=bool(args.normalize),
        metadata=metadata,
    )
    model.save(args.output)

    metadata_path = Path(args.output).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({**metadata, "rows": len(all_y)}, f, indent=2)

    print(f"saved {len(all_y)} examples to {args.output}")


if __name__ == "__main__":
    main()
