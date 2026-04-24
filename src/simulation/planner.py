import logging

import numpy as np

from src.agents.factory import AgentFactory
from src.data_types import GameState
from src.data_types.postion import Position
from src.planner.base_policy_evaluator import BasePolicyEvaluator
from src.planner.grid_model import GridModel
from src.planner.nonautonomous_rollout import NonAutonomousRolloutPlanner


logger = logging.getLogger(__name__)

def planner_run_simulation(grid, args, config):
    """Run planner-based simulation for one evader and multiple pursuers."""

    max_time_steps = config["simulation"]["time_steps"]

    evader_configs = config["evaders"]
    if len(evader_configs) == 0:
        raise RuntimeError("config['evaders'] must contain at least one evader")
    if len(evader_configs) != 1:
        raise RuntimeError("planner mode supports exactly one evader")

    pursuer_configs = config["pursuers"]

    evader_strategy_override = args.evader_type
    pursuer_strategy_override = args.pursuer_type

    seed = args.seed
    if seed is not None:
        seed = int(seed)
    rng = np.random.default_rng(seed)

    rollout_horizon = args.horizon
    if rollout_horizon is None:
        rollout_horizon = 8
    rollout_horizon = int(rollout_horizon)
    if rollout_horizon < 0:
        raise RuntimeError("--horizon must be non-negative")

    discount_factor = float(config["planner"]["discount_factor"])
   
    grid_model = GridModel(width=grid.width, height=grid.height, depth=grid.depth)

    base_evaluator = BasePolicyEvaluator(grid_model=grid_model, alpha=discount_factor, rollout_horizon=rollout_horizon)
    rollout_planner = NonAutonomousRolloutPlanner(grid_model=grid_model, base_evaluator=base_evaluator, alpha=discount_factor)

    evader_config = evader_configs[0]
    evader = AgentFactory.create_agent(
        agent_type="evader",
        strategy=evader_strategy_override if evader_strategy_override is not None else evader_config["strategy"],
        name="evader_0",
        agent_id=1,
        position=Position(
            x=int(evader_config["starting_position"][0]),
            y=int(evader_config["starting_position"][1]),
            z=int(evader_config["starting_position"][2]),
        ),
    )

    pursuers = [
        AgentFactory.create_agent(
            agent_type="pursuer",
            strategy=pursuer_strategy_override if pursuer_strategy_override is not None else pursuer_config["strategy"],
            name=f"pursuer_{idx}",
            agent_id=idx + 2,
            position=Position(
                x=int(pursuer_config["starting_position"][0]),
                y=int(pursuer_config["starting_position"][1]),
                z=int(pursuer_config["starting_position"][2]),
            ),
        )
        for idx, pursuer_config in enumerate(pursuer_configs)
    ]

    placed = grid.place_agent(evader.position, agent_id=evader.agent_id, role=evader.role)
    if not placed:
        raise RuntimeError(f"failed to place {evader.name} at the start position")

    for pursuer in pursuers:
        placed = grid.place_agent(pursuer.position, agent_id=pursuer.agent_id, role=pursuer.role)
        if placed:
            continue
        raise RuntimeError(f"failed to place {pursuer.name} at the start position")

    snapshots = [grid.grid.copy()]
    positions = [{
        "evaders": [evader.position],
        "pursuers": [pursuer.position for pursuer in pursuers],
    }]

    capture_occurred = False
    time_steps = 0
    while not capture_occurred and time_steps < max_time_steps:
        state = GameState(
            pursuer_positions=tuple(p.position for p in pursuers),
            evader_position=evader.position,
            step_idx=time_steps,
        )

        decision = rollout_planner.improve_joint_action(
            state=state,
            pursuer_agents=pursuers,
            evader_agent=evader,
            rng=rng,
        )

        next_evader_position = evader.choose_action_from_state(
            current_position=evader.position,
            grid_model=grid_model,
            pursuer_positions=list(state.pursuer_positions),
            rng=rng,
        )
        moved = grid.move_agent(evader.position, next_evader_position, agent_id=evader.agent_id)
        if moved:
            evader.move(next_evader_position)

        for pursuer, next_pursuer_position in zip(pursuers, decision.pursuer_next_positions):
            moved = grid.move_agent(pursuer.position, next_pursuer_position, agent_id=pursuer.agent_id)
            if moved:
                pursuer.move(next_pursuer_position)

            if pursuer.position == evader.position:
                capture_occurred = True
                break

        snapshots.append(grid.grid.copy())
        positions.append({
            "evaders": [] if capture_occurred else [evader.position],
            "pursuers": [pursuer.position for pursuer in pursuers],
        })

        logger.debug(
            "Planner timestep %d: capture=%s, pursuers=%d",
            time_steps,
            capture_occurred,
            len(pursuers),
        )
        time_steps += 1

    return {
        "snapshots": snapshots,
        "positions": positions,
        "grid_size": [grid.width, grid.height, grid.depth],
        "time_steps": time_steps,
        "capture_occurred": capture_occurred,
        "remaining_evaders": 0 if capture_occurred else 1,
    }

    
