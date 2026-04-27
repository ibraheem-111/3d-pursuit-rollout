from __future__ import annotations

import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np

from src.data_types import GameState
from src.data_types.postion import Position
from src.utils.math_utils import manhattan_distance


@dataclass
class JointDecision:
    pursuer_next_positions: List[Position]
    estimated_q_value: float
    selected_q_values: List[float]


@dataclass
class AgentImprovement:
    pursuer_index: int
    best_move: Position
    best_q: float
    log_lines: List[str]


def _process_pool_context():
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return None


def _improve_agent_worker(args):
    planner, state, pursuer_index, pursuer_agents, evader_agents, comparison_rng_state = args
    return planner._improve_single_agent(
        state=state,
        pursuer_index=pursuer_index,
        pursuer_agents=pursuer_agents,
        evader_agents=evader_agents,
        comparison_rng_state=comparison_rng_state,
        emit_logs=False,
    )


class AutonomousGreedySignalingRolloutPlanner:
    def __init__(self, grid_model, base_evaluator, alpha: float = 0.95):
        self.grid_model = grid_model
        self.base_evaluator = base_evaluator
        self.alpha = alpha

    def _rng_from_state(self, rng_state):
        if rng_state is None:
            return None

        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(rng_state)
        return rng

    def _capture_rng_state(self, rng):
        if rng is None:
            return None
        return copy.deepcopy(rng.bit_generator.state)

    def _distance_to_evader(self, position, state):
        if len(state.evader_positions) == 0:
            return 0
        return min(
            manhattan_distance(position, evader_position)
            for evader_position in state.evader_positions
        )

    def _target_for_pursuer(self, position, state):
        return min(
            state.evader_positions,
            key=lambda evader_position: manhattan_distance(position, evader_position),
        )

    def _active_evader_agents_after_capture(self, evader_agents, evader_moves, pursuer_moves):
        return [
            evader_agent
            for evader_agent, evader_move in zip(evader_agents, evader_moves)
            if not any(pursuer_move == evader_move for pursuer_move in pursuer_moves)
        ]

    def improve_joint_action(self, state: GameState, pursuer_agents, evader_agents, rng=None) -> JointDecision:
        comparison_rng_state = self._capture_rng_state(rng)
        improvements = self._improve_agents_parallel(
            state=state,
            pursuer_agents=pursuer_agents,
            evader_agents=evader_agents,
            comparison_rng_state=comparison_rng_state,
        )

        selected_moves = []
        selected_q_values = []
        for improvement in improvements:
            for line in improvement.log_lines:
                print(line)
            selected_moves.append(improvement.best_move)
            selected_q_values.append(improvement.best_q)

        q_val = self._joint_q_value(
            state=state,
            pursuer_agents=pursuer_agents,
            evader_agents=evader_agents,
            pursuer_moves=selected_moves,
            rng=rng,
        )
        return JointDecision(selected_moves, q_val, selected_q_values)

    def _improve_agents_parallel(self, state, pursuer_agents, evader_agents, comparison_rng_state):
        if len(pursuer_agents) == 0:
            return []

        if len(pursuer_agents) <= 1:
            return [
                self._improve_single_agent(
                    state=state,
                    pursuer_index=0,
                    pursuer_agents=pursuer_agents,
                    evader_agents=evader_agents,
                    comparison_rng_state=comparison_rng_state,
                    emit_logs=True,
                )
            ]

        tasks = [
            (self, state, i, pursuer_agents, evader_agents, comparison_rng_state)
            for i, _agent in enumerate(pursuer_agents)
        ]
        executor_kwargs = {
            "max_workers": min(len(pursuer_agents), os.cpu_count() or 1),
        }
        mp_context = _process_pool_context()
        if mp_context is not None:
            executor_kwargs["mp_context"] = mp_context

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            improvements = list(executor.map(_improve_agent_worker, tasks))

        return sorted(improvements, key=lambda improvement: improvement.pursuer_index)

    def _improve_single_agent(
        self,
        state,
        pursuer_index,
        pursuer_agents,
        evader_agents,
        comparison_rng_state,
        emit_logs=True,
    ):
        log_lines = []

        def emit(line):
            if emit_logs:
                print(line)
            else:
                log_lines.append(line)

        current_pos = state.pursuer_positions[pursuer_index]
        candidate_moves = self.grid_model.get_valid_moves(
            position=current_pos,
            agent_id=pursuer_agents[pursuer_index].agent_id,
            occupied_positions=list(state.pursuer_positions),
            evader_positions=list(state.evader_positions),
            occupied_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
        )

        if len(candidate_moves) == 0:
            return AgentImprovement(pursuer_index, current_pos, float("inf"), log_lines)

        best_move = current_pos
        best_q = float("inf")
        best_distance = self._distance_to_evader(best_move, state)
        tie_tolerance = 1e-9

        emit(f"State step={state.step_idx}, evaders={state.evader_positions}")
        emit(f"pursuers={state.pursuer_positions}")
        emit(f"Autonomous Agent {pursuer_index}")

        for candidate in candidate_moves:
            candidate_rng = self._rng_from_state(comparison_rng_state)
            pursuer_moves = self._assemble_joint_moves(
                state=state,
                pursuer_index=pursuer_index,
                candidate_move=candidate,
                pursuer_agents=pursuer_agents,
            )

            q_val = self._joint_q_value(
                state=state,
                pursuer_agents=pursuer_agents,
                evader_agents=evader_agents,
                pursuer_moves=pursuer_moves,
                rng=candidate_rng,
            )

            candidate_distance = self._distance_to_evader(candidate, state)
            if q_val < best_q - tie_tolerance or (
                abs(q_val - best_q) <= tie_tolerance and candidate_distance < best_distance
            ):
                best_q = q_val
                best_move = candidate
                best_distance = candidate_distance

            emit(f"{candidate} {q_val} dist_to_evader= {candidate_distance}")

        return AgentImprovement(pursuer_index, best_move, best_q, log_lines)

    def _assemble_joint_moves(self, state, pursuer_index, candidate_move, pursuer_agents):
        joint_moves = []

        for j, agent in enumerate(pursuer_agents):
            if j == pursuer_index:
                joint_moves.append(candidate_move)
            else:
                base_move = agent.choose_action_from_state(
                    current_position=state.pursuer_positions[j],
                    target_position=self._target_for_pursuer(state.pursuer_positions[j], state),
                    grid_model=self.grid_model,
                    pursuer_positions=list(state.pursuer_positions),
                    evader_positions=list(state.evader_positions),
                    pursuer_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
                )
                joint_moves.append(base_move)

        return joint_moves

    def _joint_q_value(self, state, pursuer_agents, evader_agents, pursuer_moves, rng=None):
        evader_moves = []
        for evader_agent, evader_position in zip(evader_agents, state.evader_positions):
            evader_move = evader_agent.choose_action_from_state(
                current_position=evader_position,
                grid_model=self.grid_model,
                pursuer_positions=list(state.pursuer_positions),
                evader_positions=list(state.evader_positions),
                pursuer_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
                rng=rng,
            )
            evader_moves.append(evader_move)

        next_state = self.grid_model.transition(state, pursuer_moves, evader_moves)
        stage_cost = self.grid_model.stage_cost(state)
        next_evader_agents = self._active_evader_agents_after_capture(
            evader_agents,
            evader_moves,
            pursuer_moves,
        )

        future_cost = self.base_evaluator.evaluate_cost_to_go(
            state=next_state,
            pursuer_agents=pursuer_agents,
            evader_agents=next_evader_agents,
            rng=rng,
        )

        return stage_cost + self.alpha * future_cost
