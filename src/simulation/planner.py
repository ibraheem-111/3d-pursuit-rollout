import logging
import time
import numpy as np
from src.agents.factory import AgentFactory
from src.data_types import GameState
from src.data_types.postion import Position
from src.planner.autonomous_rollout import AutonomousGreedySignalingRolloutPlanner
from src.planner.base_policy_evaluator import BasePolicyEvaluator
from src.planner.grid_model import GridModel
from src.planner.nonautonomous_rollout import NonAutonomousRolloutPlanner
from src.planner.signaling_policy import KernelLearnedSignalingPolicy
from src.simulation.metrics import build_run_metrics, summarize_policy


logger = logging.getLogger(__name__)


def normalize_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_")
    aliases = {
        "nonautonomous": "non_autonomous_rollout",
        "nonautonomous_rollout": "non_autonomous_rollout",
        "autonomous_greedy": "autonomous_greedy_signaling",
        "learned_signaling": "autonomous_learned_signaling",
    }
    return aliases.get(normalized, normalized)


def _signaling_model_config(args, config):
    planner_config = config.get("planner", {})
    model_path = getattr(args, "signaling_model", None) or planner_config.get("signaling_model_path")
    k = getattr(args, "signaling_k", None)
    if k is None:
        k = planner_config.get("signaling_k", 25)
    sigma = getattr(args, "signaling_sigma", None)
    if sigma is None:
        sigma = planner_config.get("signaling_sigma", 5.0)
    return model_path, int(k), float(sigma)


def _create_rollout_planner(strategy, grid_model, base_evaluator, discount_factor, args=None, config=None):
    normalized_strategy = normalize_strategy(strategy)
    if normalized_strategy == "non_autonomous_rollout":
        return NonAutonomousRolloutPlanner(
            grid_model=grid_model,
            base_evaluator=base_evaluator,
            alpha=discount_factor,
        )
    if normalized_strategy == "autonomous_greedy_signaling":
        return AutonomousGreedySignalingRolloutPlanner(
            grid_model=grid_model,
            base_evaluator=base_evaluator,
            alpha=discount_factor,
        )
    if normalized_strategy == "autonomous_learned_signaling":
        model_path, k, sigma = _signaling_model_config(args, config)
        if model_path is None:
            raise RuntimeError(
                "autonomous_learned_signaling requires --signaling-model "
                "or planner.signaling_model_path in config"
            )
        signaling_policy = KernelLearnedSignalingPolicy.load(
            grid_model=grid_model,
            path=model_path,
            k=k,
            sigma=sigma,
        )
        return AutonomousGreedySignalingRolloutPlanner(
            grid_model=grid_model,
            base_evaluator=base_evaluator,
            alpha=discount_factor,
            signaling_policy=signaling_policy,
        )
    raise RuntimeError(f"unknown planner strategy: {strategy}")

def planner_run_simulation(grid, args, config, strategy):
    """Run planner-based simulation for one evader and multiple pursuers."""
    run_start = time.perf_counter()

    max_time_steps = config["simulation"]["time_steps"]

    evader_configs = config["evaders"]
    pursuer_configs = config["pursuers"]

    evader_strategy_override = args.evader_type
    pursuer_strategy_override = args.pursuer_type

    seed = args.seed
    if seed is not None:
        seed = int(seed)

    planner_rng = np.random.default_rng(seed)
    evader_rng = np.random.default_rng(seed + 10_000 if seed is not None else None)

    discount_factor = float(config["planner"]["discount_factor"])
    strategy = normalize_strategy(strategy)
   
    grid_model = GridModel(width=grid.width, height=grid.height, depth=grid.depth)

    base_evaluator = BasePolicyEvaluator(grid_model=grid_model, alpha=discount_factor)
    rollout_planner = _create_rollout_planner(
        strategy=strategy,
        grid_model=grid_model,
        base_evaluator=base_evaluator,
        discount_factor=discount_factor,
        args=args,
        config=config,
    )
    logger.info("Using strategy=%s", strategy)

    if len(evader_configs) == 0:
        raise RuntimeError("config['evaders'] must contain at least one evader")

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

    for evader in evaders:
        placed = grid.place_agent(evader.position, agent_id=evader.agent_id, role=evader.role)
        if placed:
            continue
        raise RuntimeError(f"failed to place {evader.name} at the start position")

    for pursuer in pursuers:
        placed = grid.place_agent(pursuer.position, agent_id=pursuer.agent_id, role=pursuer.role)
        if placed:
            continue
        raise RuntimeError(f"failed to place {pursuer.name} at the start position")

    active_evaders = list(evaders)

    snapshots = [grid.grid.copy()]
    positions = [{
        "evaders": [evader.position for evader in active_evaders],
        "pursuers": [pursuer.position for pursuer in pursuers],
    }]

    capture_occurred = False
    time_steps = 0
    while len(active_evaders) > 0 and time_steps < max_time_steps:
        state = GameState(
            pursuer_positions=tuple(p.position for p in pursuers),
            evader_positions=tuple(evader.position for evader in active_evaders),
            step_idx=time_steps,
        )
        decision = rollout_planner.improve_joint_action(
            state=state,
            pursuer_agents=pursuers,
            evader_agents=active_evaders,
            rng=planner_rng,
        )
        for idx, (selected_position, selected_q) in enumerate(
            zip(decision.pursuer_next_positions, decision.selected_q_values), start=1
        ):
            print(f"SELECTED Agent {idx}: {selected_position}, Q={selected_q}")

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
            evader
            for evader in active_evaders
            if int(evader.agent_id) in grid.agent_roles_by_id
        ]

        snapshots.append(grid.grid.copy())
        positions.append({
            "evaders": [evader.position for evader in active_evaders],
            "pursuers": [pursuer.position for pursuer in pursuers],
        })

        capture_occurred = len(active_evaders) == 0

        logger.debug(
            "Planner timestep %d: capture=%s, pursuers=%d",
            time_steps,
            capture_occurred,
            len(pursuers),
        )
        time_steps += 1

    capture_occurred = len(active_evaders) == 0
    total_runtime = time.perf_counter() - run_start
    metrics = build_run_metrics(
        strategy=strategy,
        seed=seed,
        grid=grid,
        num_evaders=len(evaders),
        num_pursuers=len(pursuers),
        evader_policy=summarize_policy(evader_configs, evader_strategy_override),
        pursuer_policy=summarize_policy(pursuer_configs, pursuer_strategy_override),
        discount_factor=discount_factor,
        max_time_steps=max_time_steps,
        capture_occurred=capture_occurred,
        time_steps=time_steps,
        positions=positions,
        total_runtime=total_runtime,
        parallel_agent_rollout=strategy in {"autonomous_greedy_signaling", "autonomous_learned_signaling"},
    )
    if strategy == "autonomous_learned_signaling":
        model_path, k, sigma = _signaling_model_config(args, config)
        signaling_prediction_count = getattr(rollout_planner, "signaling_prediction_count", 0)
        signaling_invalid_prediction_count = getattr(rollout_planner, "signaling_invalid_prediction_count", 0)
        signaling_invalid_prediction_rate = None
        if signaling_prediction_count > 0:
            signaling_invalid_prediction_rate = signaling_invalid_prediction_count / signaling_prediction_count
        metrics.update(
            {
                "signaling_model_type": "kernel_knn",
                "signaling_model_path": model_path,
                "signaling_k": k,
                "signaling_sigma": sigma,
                "signaling_prediction_count": signaling_prediction_count,
                "signaling_invalid_prediction_count": signaling_invalid_prediction_count,
                "signaling_invalid_prediction_rate": signaling_invalid_prediction_rate,
            }
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

    
