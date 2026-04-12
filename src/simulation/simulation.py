import logging
import numpy as np

from src.agents.factory import AgentFactory
from src.data_types.postion import Position

logger = logging.getLogger(__name__)


def run_simulation(grid, args, config):
    logger.info("Starting simulation")
    simulation_config = config["simulation"]
    evader_config = config["evader"]

    evader_type = args.evader_type
    time_steps = simulation_config["time_steps"]
    seed = args.seed

    if seed is not None:
        seed = int(seed)

    start_xyz = evader_config["starting_position"]

    start_position = Position(
        x=int(start_xyz[0]),
        y=int(start_xyz[1]),
        z=int(start_xyz[2]),
    )

    rng = np.random.default_rng(seed)

    evader = AgentFactory.create_agent(
        agent_type="evader",
        solver_type=evader_type,
        name="evader",
        position=start_position,
    )

    placed = grid.place_agent(evader.position, agent_id=1)
    if not placed:
        raise RuntimeError("failed to place evader at the start position")

    snapshots = [grid.grid.copy()]
    positions = [evader.position]

    # Main simulation loop
    for _ in range(time_steps):

        next_position = evader.choose_action(grid, rng=rng)

        moved = grid.move_agent(evader.position, next_position, agent_id=1)
        if moved:
            evader.move(next_position)

        snapshots.append(grid.grid.copy())
        positions.append(evader.position)

    return {
        "snapshots": snapshots,
        "positions": positions,
        "grid_size": [grid.width, grid.height, grid.depth],
    }