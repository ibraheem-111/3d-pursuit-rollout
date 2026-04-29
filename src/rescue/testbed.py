from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


NodeId = int

RESCUE_STRATEGIES = (
    "greedy",
    "non_autonomous_rollout",
    "autonomous_greedy_signaling",
    "autonomous_learned_signaling",
)
DEFAULT_RESCUE_STRATEGIES = RESCUE_STRATEGIES[:3]


@dataclass(frozen=True)
class RescueState:
    agent_nodes: Tuple[NodeId, ...]
    found_individuals: Tuple[int, ...]
    explored_nodes: Tuple[NodeId, ...]
    step_idx: int = 0


@dataclass(frozen=True)
class GraphSearchProblem:
    adjacency: Mapping[NodeId, Tuple[NodeId, ...]]
    agent_start_nodes: Tuple[NodeId, ...]
    lost_individual_nodes: Tuple[NodeId, ...]
    max_time_steps: int
    target_knowledge: str = "unknown"
    discount_factor: float = 0.99
    revisit_penalty: float = 0.25
    grid_width: int | None = None
    grid_height: int | None = None

    def __post_init__(self):
        if self.target_knowledge not in {"known", "unknown"}:
            raise ValueError("target_knowledge must be 'known' or 'unknown'")
        if not (0.0 < self.discount_factor < 1.0):
            raise ValueError("discount_factor must be in (0, 1)")
        if self.revisit_penalty < 0.0:
            raise ValueError("revisit_penalty must be non-negative")
        normalized_adjacency = {
            int(node): tuple(sorted(int(neighbor) for neighbor in neighbors))
            for node, neighbors in self.adjacency.items()
        }
        object.__setattr__(self, "adjacency", normalized_adjacency)
        object.__setattr__(self, "agent_start_nodes", tuple(int(n) for n in self.agent_start_nodes))
        object.__setattr__(self, "lost_individual_nodes", tuple(int(n) for n in self.lost_individual_nodes))
        for node in (*self.agent_start_nodes, *self.lost_individual_nodes):
            if node not in self.adjacency:
                raise ValueError(f"node {node} is not in the graph")


@dataclass(frozen=True)
class RescueResult:
    strategy: str
    metrics: Dict[str, float | int | str | bool | None]
    trajectory: List[Dict[str, object]]


class ShortestPathOracle:
    def __init__(self, adjacency: Mapping[NodeId, Tuple[NodeId, ...]]):
        self.adjacency = adjacency

    @lru_cache(maxsize=None)
    def distance(self, source: NodeId, target: NodeId) -> int:
        if source == target:
            return 0

        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            node, distance = queue.popleft()
            for neighbor in self.adjacency[node]:
                if neighbor == target:
                    return distance + 1
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return 10**9

    def next_step_toward(self, source: NodeId, target: NodeId) -> NodeId:
        if source == target:
            return source

        candidates = [source, *self.adjacency[source]]
        return min(
            candidates,
            key=lambda candidate: (
                self.distance(candidate, target),
                candidate,
            ),
        )


def grid_graph(width: int, height: int) -> Dict[NodeId, Tuple[NodeId, ...]]:
    adjacency = {}
    for y in range(height):
        for x in range(width):
            node = grid_node_id(x=x, y=y, width=width)
            neighbors = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append(grid_node_id(x=nx, y=ny, width=width))
            adjacency[node] = tuple(sorted(neighbors))
    return adjacency


def sparse_grid_graph(
    width: int,
    height: int,
    *,
    extra_edge_probability: float = 0.25,
    seed: int | None = None,
) -> Dict[NodeId, Tuple[NodeId, ...]]:
    if not (0.0 <= extra_edge_probability <= 1.0):
        raise ValueError("extra_edge_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    all_edges = _grid_edges(width, height)
    shuffled_edges = list(all_edges)
    rng.shuffle(shuffled_edges)

    parent = {node: node for node in range(width * height)}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    selected_edges = set()
    remaining_edges = []
    for edge in shuffled_edges:
        a, b = edge
        if union(a, b):
            selected_edges.add(tuple(sorted(edge)))
        else:
            remaining_edges.append(tuple(sorted(edge)))

    for edge in remaining_edges:
        if rng.random() <= extra_edge_probability:
            selected_edges.add(edge)

    adjacency = {node: set() for node in range(width * height)}
    for a, b in selected_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    return {node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()}


def _grid_edges(width: int, height: int) -> List[Tuple[NodeId, NodeId]]:
    edges = []
    for y in range(height):
        for x in range(width):
            node = grid_node_id(x=x, y=y, width=width)
            if x + 1 < width:
                edges.append((node, grid_node_id(x=x + 1, y=y, width=width)))
            if y + 1 < height:
                edges.append((node, grid_node_id(x=x, y=y + 1, width=width)))
    return edges


def grid_node_id(*, x: int, y: int, width: int) -> NodeId:
    return int(y * width + x)


def grid_coordinates(node: NodeId, width: int) -> Tuple[int, int]:
    return int(node % width), int(node // width)


def initial_state(problem: GraphSearchProblem) -> RescueState:
    explored = tuple(sorted(set(problem.agent_start_nodes)))
    found = found_individual_indices(problem, explored)
    return RescueState(
        agent_nodes=problem.agent_start_nodes,
        found_individuals=found,
        explored_nodes=explored,
        step_idx=0,
    )


def found_individual_indices(problem: GraphSearchProblem, explored_nodes: Iterable[NodeId]) -> Tuple[int, ...]:
    explored = set(explored_nodes)
    return tuple(
        idx
        for idx, node in enumerate(problem.lost_individual_nodes)
        if node in explored
    )


def remaining_individual_count(problem: GraphSearchProblem, state: RescueState) -> int:
    return len(problem.lost_individual_nodes) - len(state.found_individuals)


def is_terminal(problem: GraphSearchProblem, state: RescueState) -> bool:
    return remaining_individual_count(problem, state) == 0


def stage_cost(problem: GraphSearchProblem, state: RescueState) -> float:
    return float(remaining_individual_count(problem, state))


def valid_moves(problem: GraphSearchProblem, node: NodeId) -> Tuple[NodeId, ...]:
    return tuple(sorted((node, *problem.adjacency[node])))


def transition(problem: GraphSearchProblem, state: RescueState, next_agent_nodes: Sequence[NodeId]) -> RescueState:
    for current_node, next_node in zip(state.agent_nodes, next_agent_nodes):
        if next_node not in valid_moves(problem, current_node):
            raise ValueError(f"invalid move from {current_node} to {next_node}")

    explored = tuple(sorted(set(state.explored_nodes).union(int(node) for node in next_agent_nodes)))
    found = found_individual_indices(problem, explored)
    return RescueState(
        agent_nodes=tuple(int(node) for node in next_agent_nodes),
        found_individuals=found,
        explored_nodes=explored,
        step_idx=state.step_idx + 1,
    )


def belief_transition(state: RescueState, next_agent_nodes: Sequence[NodeId]) -> RescueState:
    explored = tuple(sorted(set(state.explored_nodes).union(int(node) for node in next_agent_nodes)))
    return RescueState(
        agent_nodes=tuple(int(node) for node in next_agent_nodes),
        found_individuals=state.found_individuals,
        explored_nodes=explored,
        step_idx=state.step_idx + 1,
    )


def greedy_joint_action(problem: GraphSearchProblem, state: RescueState, oracle: ShortestPathOracle) -> Tuple[NodeId, ...]:
    targets = _search_targets(problem, state)
    if len(targets) == 0:
        return state.agent_nodes

    unclaimed_targets = set(targets)
    actions = []
    for agent_node in state.agent_nodes:
        target_pool = unclaimed_targets if len(unclaimed_targets) > 0 else set(targets)
        target = min(
            target_pool,
            key=lambda node: (
                oracle.distance(agent_node, node),
                node,
            ),
        )
        unclaimed_targets.discard(target)
        actions.append(oracle.next_step_toward(agent_node, target))
    return tuple(actions)


def _search_targets(problem: GraphSearchProblem, state: RescueState) -> Tuple[NodeId, ...]:
    if problem.target_knowledge == "known":
        return tuple(
            node
            for idx, node in enumerate(problem.lost_individual_nodes)
            if idx not in set(state.found_individuals)
        )

    explored = set(state.explored_nodes)
    return tuple(sorted(node for node in problem.adjacency if node not in explored))


def evaluate_greedy_cost_to_go(
    problem: GraphSearchProblem,
    state: RescueState,
    oracle: ShortestPathOracle,
    belief_remaining: float | None = None,
) -> float:
    if problem.target_knowledge == "unknown":
        return evaluate_unknown_belief_cost_to_go(problem, state, oracle, belief_remaining=belief_remaining)

    total = 0.0
    discount = 1.0
    sim_state = state
    while not is_terminal(problem, sim_state) and sim_state.step_idx < problem.max_time_steps:
        total += discount * stage_cost(problem, sim_state)
        action = greedy_joint_action(problem, sim_state, oracle)
        sim_state = transition(problem, sim_state, action)
        discount *= problem.discount_factor
    return total


def evaluate_unknown_belief_cost_to_go(
    problem: GraphSearchProblem,
    state: RescueState,
    oracle: ShortestPathOracle,
    belief_remaining: float | None = None,
) -> float:
    total = 0.0
    discount = 1.0
    sim_state = state
    if belief_remaining is None:
        belief_remaining = float(remaining_individual_count(problem, state))

    while belief_remaining > 1e-9 and sim_state.step_idx < problem.max_time_steps:
        total += discount * belief_remaining
        unexplored_before = unexplored_count(problem, sim_state)
        action = greedy_joint_action(problem, sim_state, oracle)
        sim_state = belief_transition(sim_state, action)
        unexplored_after = unexplored_count(problem, sim_state)
        belief_remaining = updated_unknown_belief_remaining(
            belief_remaining,
            unexplored_before,
            unexplored_after,
        )
        discount *= problem.discount_factor
    return total


def q_value(
    problem: GraphSearchProblem,
    state: RescueState,
    joint_action: Sequence[NodeId],
    oracle: ShortestPathOracle,
) -> float:
    penalty = revisit_penalty(problem, state, joint_action)
    if problem.target_knowledge == "unknown":
        unexplored_before = unexplored_count(problem, state)
        next_state = belief_transition(state, joint_action)
        unexplored_after = unexplored_count(problem, next_state)
        next_belief_remaining = updated_unknown_belief_remaining(
            float(remaining_individual_count(problem, state)),
            unexplored_before,
            unexplored_after,
        )
        return stage_cost(problem, state) + penalty + problem.discount_factor * evaluate_greedy_cost_to_go(
            problem,
            next_state,
            oracle,
            belief_remaining=next_belief_remaining,
        )

    next_state = transition(problem, state, joint_action)
    return stage_cost(problem, state) + penalty + problem.discount_factor * evaluate_greedy_cost_to_go(
        problem,
        next_state,
        oracle,
    )


def revisit_penalty(problem: GraphSearchProblem, state: RescueState, joint_action: Sequence[NodeId]) -> float:
    if problem.revisit_penalty == 0.0:
        return 0.0
    explored_nodes = set(state.explored_nodes)
    revisits = sum(1 for node in joint_action if int(node) in explored_nodes)
    return float(problem.revisit_penalty * revisits)


def unexplored_count(problem: GraphSearchProblem, state: RescueState) -> int:
    return len(problem.adjacency) - len(state.explored_nodes)


def updated_unknown_belief_remaining(
    belief_remaining: float,
    unexplored_before: int,
    unexplored_after: int,
) -> float:
    if unexplored_before <= 0:
        return 0.0
    return float(belief_remaining) * (float(unexplored_after) / float(unexplored_before))


def non_autonomous_rollout_action(
    problem: GraphSearchProblem,
    state: RescueState,
    oracle: ShortestPathOracle,
) -> Tuple[NodeId, ...]:
    base_action = greedy_joint_action(problem, state, oracle)
    chosen: Dict[int, NodeId] = {}

    for agent_idx, agent_node in enumerate(state.agent_nodes):
        best_move = base_action[agent_idx]
        best_score = float("inf")
        for candidate in valid_moves(problem, agent_node):
            joint_action = []
            for other_idx, base_move in enumerate(base_action):
                if other_idx in chosen:
                    joint_action.append(chosen[other_idx])
                elif other_idx == agent_idx:
                    joint_action.append(candidate)
                else:
                    joint_action.append(base_move)
            score = q_value(problem, state, joint_action, oracle)
            if score < best_score or (
                score == best_score
                and _target_distance(problem, state, candidate, oracle) < _target_distance(problem, state, best_move, oracle)
            ):
                best_score = score
                best_move = candidate
        chosen[agent_idx] = best_move

    return tuple(chosen[idx] for idx in range(len(state.agent_nodes)))


def autonomous_greedy_signaling_action(
    problem: GraphSearchProblem,
    state: RescueState,
    oracle: ShortestPathOracle,
) -> Tuple[NodeId, ...]:
    base_action = greedy_joint_action(problem, state, oracle)
    return autonomous_signaling_action(problem, state, oracle, base_action)


def autonomous_learned_signaling_action(
    problem: GraphSearchProblem,
    state: RescueState,
    oracle: ShortestPathOracle,
    signaling_policy,
) -> Tuple[NodeId, ...]:
    fallback_action = greedy_joint_action(problem, state, oracle)
    base_action = signaling_policy.predict_joint_action(
        problem,
        state,
        oracle,
        fallback_action=fallback_action,
    )
    return autonomous_signaling_action(problem, state, oracle, base_action)


def autonomous_signaling_action(
    problem: GraphSearchProblem,
    state: RescueState,
    oracle: ShortestPathOracle,
    base_action: Sequence[NodeId],
) -> Tuple[NodeId, ...]:
    actions = []
    for agent_idx, agent_node in enumerate(state.agent_nodes):
        best_move = base_action[agent_idx]
        best_score = float("inf")
        for candidate in valid_moves(problem, agent_node):
            joint_action = list(base_action)
            joint_action[agent_idx] = candidate
            score = q_value(problem, state, joint_action, oracle)
            if score < best_score or (
                score == best_score
                and _target_distance(problem, state, candidate, oracle) < _target_distance(problem, state, best_move, oracle)
            ):
                best_score = score
                best_move = candidate
        actions.append(best_move)
    return tuple(actions)


def _target_distance(problem: GraphSearchProblem, state: RescueState, node: NodeId, oracle: ShortestPathOracle) -> int:
    targets = _search_targets(problem, state)
    if len(targets) == 0:
        return 0
    return min(oracle.distance(node, target) for target in targets)


def select_action(
    problem: GraphSearchProblem,
    state: RescueState,
    strategy: str,
    oracle: ShortestPathOracle,
    signaling_policy=None,
):
    if strategy == "greedy":
        return greedy_joint_action(problem, state, oracle)
    if strategy == "non_autonomous_rollout":
        return non_autonomous_rollout_action(problem, state, oracle)
    if strategy == "autonomous_greedy_signaling":
        return autonomous_greedy_signaling_action(problem, state, oracle)
    if strategy == "autonomous_learned_signaling":
        if signaling_policy is None:
            raise ValueError(_missing_rescue_signaling_model_message())
        return autonomous_learned_signaling_action(problem, state, oracle, signaling_policy)
    raise ValueError(f"unknown rescue strategy: {strategy}")


def run_rescue_simulation(problem: GraphSearchProblem, strategy: str, signaling_policy=None) -> RescueResult:
    strategy = strategy.strip().lower().replace("-", "_")
    if strategy not in RESCUE_STRATEGIES:
        raise ValueError(f"unknown rescue strategy: {strategy}")
    if strategy == "autonomous_learned_signaling" and signaling_policy is None:
        raise ValueError(_missing_rescue_signaling_model_message())

    started_at = time.perf_counter()
    oracle = ShortestPathOracle(problem.adjacency)
    state = initial_state(problem)
    trajectory = [_trajectory_row(problem, state)]
    total_cost = 0.0
    discounted_cost = 0.0
    discount = 1.0
    signaling_prediction_count_before = getattr(signaling_policy, "prediction_count", 0)
    signaling_invalid_count_before = getattr(signaling_policy, "invalid_prediction_count", 0)

    while not is_terminal(problem, state) and state.step_idx < problem.max_time_steps:
        cost = stage_cost(problem, state)
        total_cost += cost
        discounted_cost += discount * cost
        action = select_action(problem, state, strategy, oracle, signaling_policy=signaling_policy)
        state = transition(problem, state, action)
        trajectory.append(_trajectory_row(problem, state))
        discount *= problem.discount_factor

    runtime = time.perf_counter() - started_at
    all_found = is_terminal(problem, state)
    metrics = {
        "strategy": strategy,
        "target_knowledge": problem.target_knowledge,
        "num_nodes": len(problem.adjacency),
        "num_agents": len(problem.agent_start_nodes),
        "num_lost_individuals": len(problem.lost_individual_nodes),
        "max_time_steps": problem.max_time_steps,
        "time_to_find_all": state.step_idx if all_found else None,
        "all_found": all_found,
        "remaining_unfound": remaining_individual_count(problem, state),
        "explored_fraction": len(state.explored_nodes) / len(problem.adjacency),
        "total_search_cost": total_cost,
        "discounted_search_cost": discounted_cost,
        "revisit_penalty": problem.revisit_penalty,
        "total_runtime": runtime,
        "mean_runtime_per_step": runtime / max(state.step_idx, 1),
    }
    if problem.grid_width is not None:
        metrics["grid_width"] = problem.grid_width
    if problem.grid_height is not None:
        metrics["grid_height"] = problem.grid_height
    if strategy == "autonomous_learned_signaling":
        signaling_prediction_count = (
            getattr(signaling_policy, "prediction_count", 0) - signaling_prediction_count_before
        )
        signaling_invalid_prediction_count = (
            getattr(signaling_policy, "invalid_prediction_count", 0) - signaling_invalid_count_before
        )
        metrics.update(
            {
                "signaling_model_type": getattr(signaling_policy, "model_type", None),
                "signaling_model_path": getattr(signaling_policy, "model_path", None),
                "signaling_prediction_count": signaling_prediction_count,
                "signaling_invalid_prediction_count": signaling_invalid_prediction_count,
                "signaling_invalid_prediction_rate": (
                    signaling_invalid_prediction_count / signaling_prediction_count
                    if signaling_prediction_count > 0
                    else 0.0
                ),
            }
        )

    return RescueResult(strategy=strategy, metrics=metrics, trajectory=trajectory)


def run_all_strategies(
    problem: GraphSearchProblem,
    strategies: Sequence[str] = DEFAULT_RESCUE_STRATEGIES,
    signaling_policy=None,
):
    return [run_rescue_simulation(problem, strategy, signaling_policy=signaling_policy) for strategy in strategies]


def _missing_rescue_signaling_model_message() -> str:
    return (
        "autonomous_learned_signaling requires a trained rescue signaling model. "
        "Collect data with: uv run python scripts/collect_rescue_signaling_data.py "
        "--config rescue_config.yaml --output models/rescue_signaling_dataset.npz --episodes 50. "
        "Train a model with: uv run python scripts/train_rescue_signaling_model.py "
        "--dataset models/rescue_signaling_dataset.npz --output models/rescue_signaling_kernel.npz "
        "--model-type kernel_knn. Then pass --signaling-model models/rescue_signaling_kernel.npz."
    )


def _trajectory_row(problem: GraphSearchProblem, state: RescueState) -> Dict[str, object]:
    row = {
        "step": state.step_idx,
        "agent_nodes": list(state.agent_nodes),
        "found_individuals": list(state.found_individuals),
        "explored_nodes": list(state.explored_nodes),
        "remaining_unfound": remaining_individual_count(problem, state),
    }
    if problem.grid_width is not None:
        row["agent_coordinates"] = [
            list(grid_coordinates(node, problem.grid_width))
            for node in state.agent_nodes
        ]
    return row


def load_problem_from_config(config: Mapping[str, object]) -> GraphSearchProblem:
    rescue_config = config.get("rescue", config)
    graph_config = rescue_config["graph"]
    graph_type = graph_config.get("type", "grid")
    if graph_type not in {"grid", "sparse_grid"}:
        raise ValueError("graph.type must be 'grid' or 'sparse_grid'")

    width = int(graph_config["width"])
    height = int(graph_config["height"])
    if graph_type == "sparse_grid":
        adjacency = sparse_grid_graph(
            width=width,
            height=height,
            extra_edge_probability=float(graph_config.get("extra_edge_probability", 0.25)),
            seed=graph_config.get("seed"),
        )
    else:
        adjacency = grid_graph(width=width, height=height)

    agents_config = rescue_config["agents"]
    individuals_config = rescue_config["lost_individuals"]
    simulation_config = rescue_config["simulation"]

    agent_start_nodes = tuple(
        _parse_grid_node(node, width=width)
        for node in agents_config["starting_nodes"]
    )
    lost_individual_nodes = tuple(
        _parse_grid_node(node, width=width)
        for node in individuals_config["nodes"]
    )

    return GraphSearchProblem(
        adjacency=adjacency,
        agent_start_nodes=agent_start_nodes,
        lost_individual_nodes=lost_individual_nodes,
        max_time_steps=int(simulation_config["time_steps"]),
        target_knowledge=str(individuals_config.get("knowledge", "unknown")),
        discount_factor=float(simulation_config.get("discount_factor", 0.99)),
        revisit_penalty=float(simulation_config.get("revisit_penalty", 0.25)),
        grid_width=width,
        grid_height=height,
    )


def _parse_grid_node(node, *, width: int) -> NodeId:
    if isinstance(node, int):
        return int(node)
    if len(node) != 2:
        raise ValueError(f"grid node must be an integer id or [x, y], got {node}")
    return grid_node_id(x=int(node[0]), y=int(node[1]), width=width)
