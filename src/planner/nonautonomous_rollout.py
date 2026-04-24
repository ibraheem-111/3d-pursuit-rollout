from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from src.data_types.postion import Position
from src.data_types import GameState


@dataclass
class JointDecision:
    pursuer_next_positions: List[Position]
    estimated_q_value: float


class NonAutonomousRolloutPlanner:
    def __init__(self, grid_model, base_evaluator, alpha: float = 0.95):
        self.grid_model = grid_model
        self.base_evaluator = base_evaluator
        self.alpha = alpha

    def improve_joint_action(self, state: GameState, pursuer_agents, evader_agent, rng=None) -> JointDecision:
        chosen_moves: Dict[int, Position] = {}

        for i, _agent in enumerate(pursuer_agents):
            best_move = self._improve_single_agent(
                state=state,
                pursuer_index=i,
                chosen_moves=chosen_moves,
                pursuer_agents=pursuer_agents,
                evader_agent=evader_agent,
                rng=rng,
            )
            chosen_moves[i] = best_move

        ordered_moves = [chosen_moves[i] for i in range(len(pursuer_agents))]
        q_val = self._joint_q_value(
            state=state,
            pursuer_agents=pursuer_agents,
            evader_agent=evader_agent,
            pursuer_moves=ordered_moves,
            rng=rng,
        )
        return JointDecision(ordered_moves, q_val)

    def _improve_single_agent(self, state, pursuer_index, chosen_moves, pursuer_agents, evader_agent, rng=None):
        current_pos = state.pursuer_positions[pursuer_index]
        candidate_moves = self.grid_model.get_valid_moves(
            position=current_pos,
            agent_id=pursuer_agents[pursuer_index].agent_id,
            occupied_positions=list(state.pursuer_positions),
            evader_position=state.evader_position,
        )

        if len(candidate_moves) == 0:
            return current_pos

        best_move = current_pos
        best_q = float("inf")

        for candidate in candidate_moves:
            pursuer_moves = self._assemble_joint_moves(
                state=state,
                pursuer_index=pursuer_index,
                candidate_move=candidate,
                chosen_moves=chosen_moves,
                pursuer_agents=pursuer_agents,
            )

            q_val = self._joint_q_value(
                state=state,
                pursuer_agents=pursuer_agents,
                evader_agent=evader_agent,
                pursuer_moves=pursuer_moves,
                rng=rng,
            )

            if q_val < best_q:
                best_q = q_val
                best_move = candidate

        return best_move

    def _assemble_joint_moves(self, state, pursuer_index, candidate_move, chosen_moves, pursuer_agents):
        joint_moves = []

        for j, agent in enumerate(pursuer_agents):
            if j in chosen_moves:
                joint_moves.append(chosen_moves[j])
            elif j == pursuer_index:
                joint_moves.append(candidate_move)
            else:
                base_move = agent.choose_action_from_state(
                    current_position=state.pursuer_positions[j],
                    target_position=state.evader_position,
                    grid_model=self.grid_model,
                    pursuer_positions=list(state.pursuer_positions),
                )
                joint_moves.append(base_move)

        return joint_moves

    def _joint_q_value(self, state, pursuer_agents, evader_agent, pursuer_moves, rng=None):
        evader_move = evader_agent.choose_action_from_state(
            current_position=state.evader_position,
            grid_model=self.grid_model,
            pursuer_positions=list(state.pursuer_positions),
            rng=rng,
        )

        next_state = self.grid_model.transition(state, pursuer_moves, evader_move)
        stage_cost = self.grid_model.stage_cost(state)

        future_cost = self.base_evaluator.evaluate_cost_to_go(
            state=next_state,
            pursuer_agents=pursuer_agents,
            evader_agent=evader_agent,
            rng=rng,
        )

        return stage_cost + self.alpha * future_cost