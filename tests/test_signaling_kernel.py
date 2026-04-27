import unittest

import numpy as np

from src.data_types import GameState
from src.data_types.postion import Position
from src.planner.signaling_kernel import (
    action_label,
    apply_action_label,
    kernel_distance,
    kernel_weight,
    state_features,
    weighted_knn_vote,
)


class SignalingKernelTest(unittest.TestCase):
    def test_relative_features_are_translation_invariant(self):
        state_a = GameState(
            pursuer_positions=(Position(4, 2, 0), Position(2, 4, 0), Position(0, 2, 0)),
            evader_position=Position(2, 2, 0),
        )
        state_b = GameState(
            pursuer_positions=(Position(8, 6, 0), Position(6, 8, 0), Position(4, 6, 0)),
            evader_position=Position(6, 6, 0),
        )

        features_a = state_features(state_a)
        features_b = state_features(state_b)

        self.assertTrue(np.array_equal(features_a, features_b))
        self.assertEqual(kernel_distance(features_a, features_b), 0.0)
        self.assertEqual(kernel_weight(0.0, sigma=5.0), 1.0)

    def test_focus_pursuer_is_first_in_feature_vector(self):
        state = GameState(
            pursuer_positions=(Position(5, 5, 0), Position(7, 5, 0), Position(5, 8, 0)),
            evader_position=Position(5, 5, 0),
        )

        features = state_features(state, focus_pursuer_index=1)

        self.assertEqual(features[:3].tolist(), [2.0, 0.0, 0.0])

    def test_weighted_vote_uses_manhattan_exponential_kernel(self):
        X_train = np.array([[0.0, 0.0], [10.0, 10.0], [1.0, 0.0]])
        y_train = np.array([2, 3, 2])

        result = weighted_knn_vote(X_train, y_train, np.array([0.0, 0.0]), k=3, sigma=5.0)

        self.assertEqual(result.label, 2)
        self.assertGreater(result.scores[2], result.scores[3])

    def test_action_label_round_trip(self):
        current = Position(3, 3, 1)
        next_position = Position(3, 4, 1)

        label = action_label(current, next_position)

        self.assertEqual(apply_action_label(current, label), next_position)


if __name__ == "__main__":
    unittest.main()
