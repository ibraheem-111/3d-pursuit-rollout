from __future__ import annotations
import numpy as np
from src.data_types import GameState
from src.utils.math_utils import manhattan_distance

class BasePolicyEvaluator:
    def __init__(self, grid_model, alpha: float = 0.95):
        self.grid_model = grid_model
        self.alpha = alpha
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1) for infinite-horizon evaluation")

    def _closest_evader_position(self, position, evader_positions):
        return min(
            evader_positions,
            key=lambda evader_position: manhattan_distance(position, evader_position),
        )

    def _active_evader_agents_after_capture(self, evader_agents, evader_moves, pursuer_moves):
        return [
            evader_agent
            for evader_agent, evader_move in zip(evader_agents, evader_moves)
            if not any(pursuer_move == evader_move for pursuer_move in pursuer_moves)
        ]

    def _rollout_step(self, sim_state: GameState, pursuer_agents, evader_agents, rng):
        pursuer_moves = []
        for idx, agent in enumerate(pursuer_agents):
            target_position = self._closest_evader_position(
                sim_state.pursuer_positions[idx],
                sim_state.evader_positions,
            )
            move = agent.choose_action_from_state(
                current_position=sim_state.pursuer_positions[idx],
                target_position=target_position,
                grid_model=self.grid_model,
                pursuer_positions=list(sim_state.pursuer_positions),
                evader_positions=list(sim_state.evader_positions),
                pursuer_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
            )
            pursuer_moves.append(move)

        evader_moves = []
        for evader_agent, evader_position in zip(evader_agents, sim_state.evader_positions):
            evader_move = evader_agent.choose_action_from_state(
                current_position=evader_position,
                grid_model=self.grid_model,
                pursuer_positions=list(sim_state.pursuer_positions),
                evader_positions=list(sim_state.evader_positions),
                pursuer_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
                rng=rng,
            )
            evader_moves.append(evader_move)

        stage_cost = self.grid_model.stage_cost(sim_state)
        next_state = self.grid_model.transition(sim_state, pursuer_moves, evader_moves)
        next_evader_agents = self._active_evader_agents_after_capture(evader_agents, evader_moves, pursuer_moves)
        return stage_cost, next_state, next_evader_agents

    def evaluate_cost_to_go(self, state: GameState, pursuer_agents, evader_agents, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng(0)

        total = 0.0
        discount = 1.0
        sim_state = state
        sim_evader_agents = list(evader_agents)

        tail_tolerance = 1e-6
        while True:
            if self.grid_model.is_capture(sim_state):
                break

            stage_cost, sim_state, sim_evader_agents = self._rollout_step(
                sim_state,
                pursuer_agents,
                sim_evader_agents,
                rng,
            )
            total += discount * stage_cost
            discount *= self.alpha

            if discount / (1.0 - self.alpha) <= tail_tolerance:
                break

        return total
