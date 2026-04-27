import unittest
import tempfile
from pathlib import Path

from src.rescue.testbed import (
    GraphSearchProblem,
    RescueState,
    ShortestPathOracle,
    autonomous_greedy_signaling_action,
    grid_graph,
    greedy_joint_action,
    initial_state,
    load_problem_from_config,
    non_autonomous_rollout_action,
    q_value,
    run_all_strategies,
    run_rescue_simulation,
    sparse_grid_graph,
    stage_cost,
    transition,
    revisit_penalty,
)
from src.rescue.visualization import plot_rescue_trajectory


class RescueTestBedTest(unittest.TestCase):
    def test_stage_cost_counts_unfound_individuals(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(0,),
            lost_individual_nodes=(1, 2),
            max_time_steps=5,
            target_knowledge="unknown",
        )
        state = initial_state(problem)

        self.assertEqual(stage_cost(problem, state), 2.0)

        next_state = transition(problem, state, [1])
        self.assertEqual(stage_cost(problem, next_state), 1.0)

    def test_greedy_moves_to_closest_unexplored_node(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(0,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
        )
        state = initial_state(problem)
        oracle = ShortestPathOracle(problem.adjacency)

        self.assertEqual(greedy_joint_action(problem, state, oracle), (1,))

    def test_sparse_grid_graph_is_connected_and_missing_edges(self):
        width = 5
        height = 5
        adjacency = sparse_grid_graph(width=width, height=height, extra_edge_probability=0.0, seed=4)
        oracle = ShortestPathOracle(adjacency)
        full_grid_edge_count = sum(len(neighbors) for neighbors in grid_graph(width, height).values()) // 2
        sparse_edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2

        self.assertLess(sparse_edge_count, full_grid_edge_count)
        for node in adjacency:
            self.assertLess(oracle.distance(0, node), 10**9)

    def test_load_sparse_grid_from_config(self):
        problem = load_problem_from_config(
            {
                "rescue": {
                    "graph": {
                        "type": "sparse_grid",
                        "width": 4,
                        "height": 4,
                        "extra_edge_probability": 0.2,
                        "seed": 9,
                    },
                    "agents": {"starting_nodes": [[0, 0]]},
                    "lost_individuals": {"knowledge": "unknown", "nodes": [[3, 3]]},
                    "simulation": {"time_steps": 20, "discount_factor": 0.99},
                }
            }
        )

        self.assertEqual(problem.grid_width, 4)
        self.assertEqual(problem.grid_height, 4)
        self.assertLess(sum(len(neighbors) for neighbors in problem.adjacency.values()) // 2, 24)

    def test_known_target_greedy_moves_toward_lost_individual(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=5, height=1),
            agent_start_nodes=(0,),
            lost_individual_nodes=(4,),
            max_time_steps=10,
            target_knowledge="known",
        )
        result = run_rescue_simulation(problem, "greedy")

        self.assertTrue(result.metrics["all_found"])
        self.assertEqual(result.metrics["time_to_find_all"], 4)
        self.assertEqual(result.metrics["total_search_cost"], 4.0)

    def test_revisit_penalty_counts_already_explored_next_nodes(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=1),
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            revisit_penalty=0.5,
        )
        state = RescueState(
            agent_nodes=(1,),
            found_individuals=(),
            explored_nodes=(0, 1),
        )

        self.assertEqual(revisit_penalty(problem, state, (0,)), 0.5)
        self.assertEqual(revisit_penalty(problem, state, (2,)), 0.0)

    def test_rollout_q_value_includes_revisit_penalty(self):
        adjacency = grid_graph(width=3, height=1)
        state = RescueState(
            agent_nodes=(1,),
            found_individuals=(),
            explored_nodes=(0, 1),
        )
        penalized_problem = GraphSearchProblem(
            adjacency=adjacency,
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            revisit_penalty=0.5,
        )
        unpenalized_problem = GraphSearchProblem(
            adjacency=adjacency,
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
            revisit_penalty=0.0,
        )
        oracle = ShortestPathOracle(adjacency)

        penalized_q = q_value(penalized_problem, state, (0,), oracle)
        unpenalized_q = q_value(unpenalized_problem, state, (0,), oracle)

        self.assertAlmostEqual(penalized_q - unpenalized_q, 0.5)

    def test_all_rescue_strategies_run(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=4, height=4),
            agent_start_nodes=(0, 15),
            lost_individual_nodes=(5, 10),
            max_time_steps=20,
            target_knowledge="unknown",
        )

        results = run_all_strategies(problem)

        self.assertEqual(
            [result.strategy for result in results],
            ["greedy", "non_autonomous_rollout", "autonomous_greedy_signaling"],
        )
        self.assertTrue(all("total_search_cost" in result.metrics for result in results))

    def test_unknown_rollout_action_does_not_depend_on_hidden_location(self):
        adjacency = grid_graph(width=3, height=1)
        state = RescueState(
            agent_nodes=(1,),
            found_individuals=(),
            explored_nodes=(1,),
        )
        left_hidden = GraphSearchProblem(
            adjacency=adjacency,
            agent_start_nodes=(1,),
            lost_individual_nodes=(0,),
            max_time_steps=5,
            target_knowledge="unknown",
        )
        right_hidden = GraphSearchProblem(
            adjacency=adjacency,
            agent_start_nodes=(1,),
            lost_individual_nodes=(2,),
            max_time_steps=5,
            target_knowledge="unknown",
        )

        left_oracle = ShortestPathOracle(left_hidden.adjacency)
        right_oracle = ShortestPathOracle(right_hidden.adjacency)

        self.assertEqual(
            non_autonomous_rollout_action(left_hidden, state, left_oracle),
            non_autonomous_rollout_action(right_hidden, state, right_oracle),
        )
        self.assertEqual(
            autonomous_greedy_signaling_action(left_hidden, state, left_oracle),
            autonomous_greedy_signaling_action(right_hidden, state, right_oracle),
        )

    def test_static_visualization_writes_png(self):
        problem = GraphSearchProblem(
            adjacency=grid_graph(width=3, height=3),
            agent_start_nodes=(0,),
            lost_individual_nodes=(8,),
            max_time_steps=10,
            target_knowledge="known",
            grid_width=3,
            grid_height=3,
        )
        result = run_rescue_simulation(problem, "greedy")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rescue.png"
            plot_rescue_trajectory(problem, result, output_path=path)

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
