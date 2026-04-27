import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.agents.factory import AgentFactory
from src.data_types import GameState
from src.data_types.postion import Position
from src.planner.grid_model import GridModel
from src.planner.signaling_kernel import action_label, state_features
from src.planner.signaling_policy import KernelLearnedSignalingPolicy, KernelSignalingModel


class MultiEvaderSignalingTest(unittest.TestCase):
    def test_stage_cost_is_active_evader_count(self):
        grid_model = GridModel(width=5, height=5, depth=1)
        state = GameState(
            pursuer_positions=(Position(0, 0, 0),),
            evader_positions=(Position(1, 1, 0), Position(4, 4, 0)),
        )

        self.assertEqual(grid_model.stage_cost(state), 2.0)

    def test_transition_removes_captured_evaders(self):
        grid_model = GridModel(width=5, height=5, depth=1)
        state = GameState(
            pursuer_positions=(Position(0, 0, 0),),
            evader_positions=(Position(1, 0, 0), Position(4, 4, 0)),
        )

        next_state = grid_model.transition(
            state,
            pursuer_next_positions=[Position(1, 0, 0)],
            evader_next_positions=[Position(1, 0, 0), Position(4, 4, 0)],
        )

        self.assertEqual(next_state.evader_positions, (Position(4, 4, 0),))

    def test_kernel_policy_predicts_valid_learned_move(self):
        grid_model = GridModel(width=5, height=5, depth=1)
        state = GameState(
            pursuer_positions=(Position(3, 2, 0), Position(4, 4, 0)),
            evader_positions=(Position(2, 2, 0),),
        )
        target_next = Position(2, 2, 0)
        feature = state_features(
            state,
            reference_evader_position=Position(2, 2, 0),
            focus_pursuer_index=0,
            grid_size=(5, 5, 1),
            include_evader_position=True,
            normalize=True,
        )
        model = KernelSignalingModel(
            X_train=np.array([feature]),
            y_train=np.array([action_label(state.pursuer_positions[0], target_next)]),
            grid_size=(5, 5, 1),
            k=1,
            sigma=5.0,
            include_evader_position=True,
            normalize=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.npz"
            model.save(model_path)
            policy = KernelLearnedSignalingPolicy.load(grid_model, model_path)

            pursuers = [
                AgentFactory.create_agent(
                    agent_type="pursuer",
                    strategy="greedy",
                    name="pursuer_0",
                    agent_id=2,
                    position=state.pursuer_positions[0],
                ),
                AgentFactory.create_agent(
                    agent_type="pursuer",
                    strategy="greedy",
                    name="pursuer_1",
                    agent_id=3,
                    position=state.pursuer_positions[1],
                ),
            ]
            predicted = policy.predict_move(state, pursuer_index=0, pursuer_agents=pursuers)

        self.assertEqual(predicted, target_next)


if __name__ == "__main__":
    unittest.main()
