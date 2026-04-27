from __future__ import annotations

from src.utils.math_utils import manhattan_distance


def summarize_policy(configs, override=None):
    if override is not None:
        return override

    strategies = [config["strategy"] for config in configs]
    unique_strategies = sorted(set(strategies))
    if len(unique_strategies) == 1:
        return unique_strategies[0]
    return ",".join(unique_strategies)


def stage_costs_from_positions(positions):
    return [float(len(state["evaders"])) for state in positions[:-1]]


def discounted_stage_cost(stage_costs, discount_factor):
    if discount_factor is None:
        return None
    if len(stage_costs) == 0:
        return 0.0
    return float(
        sum((discount_factor ** step_idx) * cost for step_idx, cost in enumerate(stage_costs))
    )


def final_min_distance_to_evader(positions, capture_occurred):
    if capture_occurred:
        return 0.0

    if len(positions) == 0:
        return None

    final_state = positions[-1]
    evaders = final_state["evaders"]
    pursuers = final_state["pursuers"]
    if len(evaders) == 0 or len(pursuers) == 0:
        return None

    min_distance = min(
        manhattan_distance(pursuer_position, evader_position)
        for pursuer_position in pursuers
        for evader_position in evaders
    )
    return float(min_distance)


def build_run_metrics(
    *,
    strategy,
    seed,
    grid,
    num_evaders,
    num_pursuers,
    evader_policy,
    pursuer_policy,
    discount_factor,
    max_time_steps,
    capture_occurred,
    time_steps,
    positions,
    total_runtime,
    rollout_horizon="infinite",
    num_rollout_samples=1,
    common_random_numbers=True,
    tie_breaking_rule="min_manhattan_distance_to_evader",
    parallel_agent_rollout=False,
):
    stage_costs = stage_costs_from_positions(positions)
    total_stage_cost = float(sum(stage_costs))
    return {
        "strategy": strategy,
        "seed": seed,
        "grid_width": grid.width,
        "grid_height": grid.height,
        "grid_depth": grid.depth,
        "num_evaders": num_evaders,
        "num_pursuers": num_pursuers,
        "evader_policy": evader_policy,
        "pursuer_policy": pursuer_policy,
        "rollout_horizon": rollout_horizon,
        "discount_factor": discount_factor,
        "num_rollout_samples": num_rollout_samples,
        "common_random_numbers": common_random_numbers,
        "tie_breaking_rule": tie_breaking_rule,
        "parallel_agent_rollout": parallel_agent_rollout,
        "capture_occurred": capture_occurred,
        "time_to_capture": time_steps if capture_occurred else None,
        "max_time_steps": max_time_steps,
        "discounted_cost": discounted_stage_cost(stage_costs, discount_factor),
        "total_stage_cost": total_stage_cost,
        "mean_runtime_per_step": total_runtime / max(time_steps, 1),
        "total_runtime": total_runtime,
        "final_min_distance_to_evader": final_min_distance_to_evader(positions, capture_occurred),
    }
