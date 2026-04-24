from __future__ import annotations
from typing import List
import numpy as np
from src.data_types import GameState

class BasePolicyEvaluator:
    def __init__(self, grid_model, alpha: float = 0.95, rollout_horizon: int = 8):
        self.grid_model = grid_model
        self.alpha = alpha
        self.rollout_horizon = rollout_horizon
        if self.rollout_horizon < 0:
            raise ValueError("rollout_horizon must be non-negative")
        if self.rollout_horizon == 0 and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1] for infinite-horizon evaluation")

    def _rollout_step(self, sim_state: GameState, pursuer_agents, evader_agent, rng):
        pursuer_moves = []
        for idx, agent in enumerate(pursuer_agents):
            move = agent.choose_action_from_state(
                current_position=sim_state.pursuer_positions[idx],
                target_position=sim_state.evader_position,
                grid_model=self.grid_model,
                pursuer_positions=list(sim_state.pursuer_positions),
            )
            pursuer_moves.append(move)

        evader_move = evader_agent.choose_action_from_state(
            current_position=sim_state.evader_position,
            grid_model=self.grid_model,
            pursuer_positions=list(sim_state.pursuer_positions),
            rng=rng,
        )

        stage_cost = self.grid_model.stage_cost(sim_state)
        next_state = self.grid_model.transition(sim_state, pursuer_moves, evader_move)
        return stage_cost, next_state

    def evaluate_cost_to_go(self, state: GameState, pursuer_agents, evader_agent, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng(0)

        total = 0.0
        discount = 0.8
        sim_state = state

        if self.rollout_horizon == 0:
            tail_tolerance = 1e-6
            while True:
                if self.grid_model.is_capture(sim_state):
                    break

                stage_cost, sim_state = self._rollout_step(sim_state, pursuer_agents, evader_agent, rng)
                total += discount * stage_cost
                discount *= self.alpha

                if discount / (1.0 - self.alpha) <= tail_tolerance:
                    break

            return total

        for _ in range(self.rollout_horizon):
            if self.grid_model.is_capture(sim_state):
                break

            stage_cost, sim_state = self._rollout_step(sim_state, pursuer_agents, evader_agent, rng)
            total += discount * stage_cost
            discount *= self.alpha

        return total