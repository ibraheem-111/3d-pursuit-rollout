import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.collect_rescue_signaling_data import collect_dataset
from scripts.train_rescue_signaling_model import train_model_from_dataset
from src.rescue.signaling import (
    ACTION_NAMES,
    KernelRescueSignalingModel,
    MLPRescueSignalingModel,
    RescueLearnedSignalingPolicy,
    RescueSignalingDataset,
    action_label,
    apply_action_label,
    load_rescue_signaling_model,
    rescue_state_features,
)
from src.rescue.testbed import (
    GraphSearchProblem,
    RescueState,
    grid_graph,
    run_rescue_simulation,
)


class RescueSignalingTest(unittest.TestCase):
    def test_unknown_features_do_not_include_hidden_lost_locations(self):
        state = RescueState(agent_nodes=(1,), found_individuals=(), explored_nodes=(1,))
        left_hidden = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(1,),
            lost_individual_nodes=(0,),
            max_time_steps=5,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
        )
        right_hidden = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
        )

        np.testing.assert_allclose(
            rescue_state_features(left_hidden, state, 0),
            rescue_state_features(right_hidden, state, 0),
        )

    def test_action_label_round_trip(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=3),
            agent_start_nodes=(4,),
            lost_individual_nodes=(8,),
            max_time_steps=5,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=3,
        )
        current_node = 4

        for label in range(len(ACTION_NAMES)):
            next_node = apply_action_label(problem, current_node, label)
            self.assertIsNotNone(next_node)
            self.assertEqual(action_label(problem, current_node, next_node), label)

    def test_kernel_policy_predicts_legal_move_and_falls_back_from_illegal_label(self):
        state = RescueState(agent_nodes=(1,), found_individuals=(), explored_nodes=(1,))
        full_problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
        )
        feature = rescue_state_features(full_problem, state, 0)
        model = KernelRescueSignalingModel(
            X_train=np.asarray([feature]),
            y_train=np.asarray([action_label(full_problem, 1, 2)]),
            num_nodes=3,
            num_agents=1,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
            k=1,
            sigma=5.0,
        )
        policy = RescueLearnedSignalingPolicy(model)

        self.assertEqual(policy.predict_joint_action(full_problem, state, fallback_action=(1,)), (2,))

        sparse_problem = GraphSearchProblem(
            adjacency={0: (1,), 1: (0,), 2: ()},
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
        )
        sparse_feature = rescue_state_features(sparse_problem, state, 0)
        fallback_model = KernelRescueSignalingModel(
            X_train=np.asarray([sparse_feature, sparse_feature]),
            y_train=np.asarray(
                [
                    action_label(sparse_problem, 1, 2),
                    action_label(sparse_problem, 1, 0),
                ]
            ),
            num_nodes=3,
            num_agents=1,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
            k=2,
            sigma=5.0,
        )
        fallback_policy = RescueLearnedSignalingPolicy(fallback_model)

        self.assertEqual(fallback_policy.predict_joint_action(sparse_problem, state, fallback_action=(1,)), (0,))

    def test_mlp_model_overfits_tiny_dataset_and_reloads(self):
        X_train = np.eye(len(ACTION_NAMES), dtype=float)
        y_train = np.arange(len(ACTION_NAMES), dtype=int)
        dataset = RescueSignalingDataset(
            X_train=X_train,
            y_train=y_train,
            metadata={
                "num_nodes": 5,
                "num_agents": 1,
                "target_knowledge": "unknown",
                "grid_width": 5,
                "grid_height": 1,
            },
        )
        model = MLPRescueSignalingModel.train(
            dataset,
            hidden_layers=(),
            learning_rate=0.5,
            epochs=800,
            seed=3,
        )

        predictions = [
            max(model.predict_label_scores(feature), key=model.predict_label_scores(feature).get)
            for feature in X_train
        ]
        self.assertEqual(predictions, list(y_train))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "mlp_model.npz"
            model.save(model_path)
            loaded = load_rescue_signaling_model(model_path, model_type="mlp")

            loaded_predictions = [
                max(loaded.predict_label_scores(feature), key=loaded.predict_label_scores(feature).get)
                for feature in X_train
            ]
            self.assertEqual(loaded_predictions, list(y_train))

    def test_autonomous_learned_signaling_records_policy_metrics(self):
        class FakePolicy:
            model_type = "fake"
            model_path = "fake-model"

            def __init__(self):
                self.prediction_count = 0
                self.invalid_prediction_count = 0

            def predict_joint_action(self, problem, state, oracle=None, fallback_action=None):
                self.prediction_count += len(state.agent_nodes)
                return tuple(fallback_action)

        problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(0,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
        )

        result = run_rescue_simulation(
            problem,
            "autonomous_learned_signaling",
            signaling_policy=FakePolicy(),
        )

        self.assertEqual(result.strategy, "autonomous_learned_signaling")
        self.assertEqual(result.metrics["signaling_model_type"], "fake")
        self.assertGreater(result.metrics["signaling_prediction_count"], 0)
        self.assertIn("signaling_invalid_prediction_rate", result.metrics)

    def test_collect_train_and_run_learned_signaling_for_both_model_types(self):
        base_problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(0,),
            lost_individual_nodes=(2,),
            max_time_steps=4,
            target_knowledge="unknown",
            grid_width=3,
            grid_height=1,
        )
        dataset = collect_dataset(
            base_problem,
            episodes=1,
            seed=0,
            max_time_steps=2,
            sample_lost=False,
            sample_agents=False,
        )
        self.assertGreater(dataset.y_train.shape[0], 0)

        kernel_model = train_model_from_dataset(dataset, model_type="kernel_knn", k=1, sigma=5.0)
        mlp_model = train_model_from_dataset(
            dataset,
            model_type="mlp",
            hidden_layers=(),
            learning_rate=0.5,
            epochs=400,
            seed=1,
        )

        for model in (kernel_model, mlp_model):
            policy = RescueLearnedSignalingPolicy(model)
            result = run_rescue_simulation(
                base_problem,
                "autonomous_learned_signaling",
                signaling_policy=policy,
            )
            self.assertIn("signaling_model_type", result.metrics)
            self.assertGreater(result.metrics["signaling_prediction_count"], 0)


if __name__ == "__main__":
    unittest.main()
