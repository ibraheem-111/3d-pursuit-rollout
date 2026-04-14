from src.data_types import GameState
from src.planner.grid_model import GridModel
from src.planner.base_policy_evaluator import BasePolicyEvaluator
from src.planner.nonautonomous_rollout import NonAutonomousRolloutPlanner
import numpy as np
from src.agents.factory import AgentFactory
from src.data_types.postion import Position

def planner_run_simulation(grid, args, config):
    """
    For one evader and multiple pursuser.
    TODO: support multiple evaders in the future.
    """

    max_time_steps = config["simulation"]["time_steps"]
    evader_configs = config["evaders"]
    pursuer_configs = config["pursuers"]
    rng = np.random.default_rng(args.seed)
   
    grid_model = GridModel(width=grid.width, height=grid.height, depth=grid.depth)

    base_evaluator = BasePolicyEvaluator(grid_model=grid_model, alpha=0.95, rollout_horizon=8)
    rollout_planner = NonAutonomousRolloutPlanner(grid_model=grid_model, base_evaluator=base_evaluator, alpha=0.95)

    pursuers = [
        AgentFactory.create_agent(
            agent_type="pursuer",
            strategy=pursuer_config["strategy"],
            name=f"pursuer_{idx}",
            agent_id= idx + 2,
            position=Position(
                x=int(pursuer_config["starting_position"][0]),
                y=int(pursuer_config["starting_position"][1]),
                z=int(pursuer_config["starting_position"][2]),
            ),
        )
        for idx, pursuer_config in enumerate(pursuer_configs)
    ]

    evader = AgentFactory.create_agent(
        agent_type="evader",
        strategy=evader_configs[0]["strategy"],
        name="evader_0",
        agent_id=1,
        position=Position(
            x=int(evader_configs[0]["starting_position"][0]),
            y=int(evader_configs[0]["starting_position"][1]),
            z=int(evader_configs[0]["starting_position"][2]),
        ),
    )

    for time_steps in range(max_time_steps):
        
        state = GameState(
            pursuer_positions=tuple(p.position for p in pursuers),
            evader_position=evader.position,
            step_idx=time_steps,
        )

        decision = rollout_planner.improve_joint_action(
            state=state,
            pursuer_agents=pursuers,
            evader_agent=evader,
            rng=rng,
        )

        for pursuer, next_pursuer_position in zip(pursuers, decision.pursuer_next_positions):
            moved = grid.move_agent(pursuer.position, next_pursuer_position, agent_id=pursuer.agent_id)
            if moved:
                pursuer.move(next_pursuer_position)

            if pursuer.position == evader.position:
                capture_occurred = True
                break


    