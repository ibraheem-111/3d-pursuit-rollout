import logging
import time
import numpy as np
from src.agents.factory import AgentFactory
from src.data_types.postion import Position
from src.simulation.metrics import build_run_metrics, summarize_policy
from src.utils.math_utils import find_closest_agent
from src.simulation.planner import normalize_strategy, planner_run_simulation

logger = logging.getLogger(__name__)

def _configured_strategy(args, config):
    strategy = getattr(args, "strategy", None)
    if strategy is None:
        strategy = config.get("strategy", "greedy")
    return normalize_strategy(strategy)


def run_simulation(grid, args, config, **kwargs):
    logger.info("Starting simulation")
    run_start = time.perf_counter()
    strategy = _configured_strategy(args, config)

    if strategy in {"non_autonomous_rollout", "autonomous_greedy_signaling"}:
        return planner_run_simulation(grid, args, config, strategy=strategy)
    if strategy != "greedy":
        raise RuntimeError(f"unknown simulation strategy: {strategy}")

    max_time_steps = config["simulation"]["time_steps"]

    evader_configs = config["evaders"]
    if len(evader_configs) == 0:
        raise RuntimeError("config['evaders'] must contain at least one evader")

    pursuer_configs = config["pursuers"]

    evader_strategy_override = args.evader_type
    pursuer_strategy_override = args.pursuer_type

    seed = args.seed
    if seed is not None:
        seed = int(seed)
    rng = np.random.default_rng(seed)

    evaders = [
        AgentFactory.create_agent(
            agent_type="evader",
            strategy=evader_strategy_override if evader_strategy_override is not None else evader_config["strategy"],
            name=f"evader_{idx}",
            agent_id=idx + 1,
            position=Position(
                x=int(evader_config["starting_position"][0]),
                y=int(evader_config["starting_position"][1]),
                z=int(evader_config["starting_position"][2]),
            ),
        )
        for idx, evader_config in enumerate(evader_configs)
    ]

    pursuers = [
        AgentFactory.create_agent(
            agent_type="pursuer",
            strategy=pursuer_strategy_override if pursuer_strategy_override is not None else pursuer_config["strategy"],
            name=f"pursuer_{idx}",
            agent_id=len(evaders) + idx + 1,
            position=Position(
                x=int(pursuer_config["starting_position"][0]),
                y=int(pursuer_config["starting_position"][1]),
                z=int(pursuer_config["starting_position"][2]),
            ),
        )
        for idx, pursuer_config in enumerate(pursuer_configs)
    ]

    for agent in evaders + pursuers:
        placed = grid.place_agent(agent.position, agent_id=agent.agent_id, role=agent.role)
        if placed:
            continue
        raise RuntimeError(f"failed to place {agent.name} at the start position")

    active_evaders = list(evaders)

    snapshots = [grid.grid.copy()]
    positions = [{
        "evaders": [evader.position for evader in active_evaders],
        "pursuers": [pursuer.position for pursuer in pursuers],
    }]

    time_steps = 0
    while len(active_evaders) > 0 and time_steps < max_time_steps:

        for evader in active_evaders:
            next_evader_position = evader.choose_action(
                grid,
                rng=rng,
                pursuers=pursuers,
                evaders=active_evaders,
            )
            moved = grid.move_agent(evader.position, next_evader_position, agent_id=evader.agent_id)
            if not moved:
                continue
            evader.move(next_evader_position)

        for pursuer in pursuers:
            if len(active_evaders) == 0:
                break
            target_evader = find_closest_agent(pursuer.position, active_evaders)
            next_pursuer_position = pursuer.choose_action(
                grid,
                target_position=target_evader.position,
                rng=rng,
                pursuers=pursuers,
                evaders=active_evaders,
            )
            moved = grid.move_agent(pursuer.position, next_pursuer_position, agent_id=pursuer.agent_id)
            if not moved:
                continue
            pursuer.move(next_pursuer_position)

        captured_evader_ids = {
            evader.agent_id
            for evader in active_evaders
            if any(pursuer.position == evader.position for pursuer in pursuers)
        }
        if len(captured_evader_ids) > 0:
            active_evaders = [
                evader for evader in active_evaders
                if evader.agent_id not in captured_evader_ids
            ]

        snapshots.append(grid.grid.copy())
        positions.append({
            "evaders": [evader.position for evader in active_evaders],
            "pursuers": [pursuer.position for pursuer in pursuers],
        })

        logger.debug("Time step %d: Active evaders: %d", time_steps, len(active_evaders))
        for evader in active_evaders:
            logger.debug("  %s at %s", evader.name, evader.position)
        for pursuer in pursuers:
            logger.debug("  %s at %s", pursuer.name, pursuer.position)

        time_steps += 1

    capture_occurred = len(active_evaders) == 0
    total_runtime = time.perf_counter() - run_start
    metrics = build_run_metrics(
        strategy="greedy",
        seed=seed,
        grid=grid,
        num_evaders=len(evaders),
        num_pursuers=len(pursuers),
        evader_policy=summarize_policy(evader_configs, evader_strategy_override),
        pursuer_policy=summarize_policy(pursuer_configs, pursuer_strategy_override),
        discount_factor=None,
        max_time_steps=max_time_steps,
        capture_occurred=capture_occurred,
        time_steps=time_steps,
        positions=positions,
        total_runtime=total_runtime,
        rollout_horizon=None,
        num_rollout_samples=None,
        common_random_numbers=False,
        tie_breaking_rule="first_valid_min_manhattan_distance",
        parallel_agent_rollout=False,
    )

    return {
        "snapshots": snapshots,
        "positions": positions,
        "grid_size": [grid.width, grid.height, grid.depth],
        "time_steps": time_steps,
        "capture_occurred": capture_occurred,
        "remaining_evaders": len(active_evaders),
        "metrics": metrics,
    }
