import logging
import numpy as np

from src.agents.factory import AgentFactory
from src.data_types.postion import Position

logger = logging.getLogger(__name__)


def run_simulation(grid, args, config):
    logger.info("Starting simulation")
    simulation_config = config["simulation"]
    evader_config = config["evader"]
    pursuer_config = config["pursuer"]
    num_pursuers = pursuer_config["num_pursuers"]
    pursuer_starting_positions = pursuer_config["starting_positions"]


    evader_type = args.evader_type
    max_time_steps = simulation_config["time_steps"]
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
        strategy=evader_type,
        name="evader",
        agent_id=1,
        position=start_position,
    )

    pursuers = [AgentFactory.create_agent(
        agent_type="pursuer",
        strategy=pursuer_config["strategy"],
        name=f"pursuer_{i}",
        agent_id=i+2,
        position=Position(x=int(pursuer_starting_positions[i][0]), y=int(pursuer_starting_positions[i][1]), z=int(pursuer_starting_positions[i][2])), 
    ) for i in range(num_pursuers)]

    evader_placed = grid.place_agent(evader.position, agent_id=1)
    if not evader_placed:
        raise RuntimeError("failed to place evader at the start position")

    if not all(grid.place_agent(p.position, agent_id=p.agent_id) for p in pursuers):
        raise RuntimeError("failed to place one or more pursuers at their starting positions")

    snapshots = [grid.grid.copy()]
    positions = [{
        "evader": evader.position,
        "pursuers": [p.position for p in pursuers]
    }]

    capture_occurred = False

    time_steps = 0

    # Main simulation loop
    while not capture_occurred and time_steps < max_time_steps:

        next_evader_position = evader.choose_action(grid, rng=rng)

        moved = grid.move_agent(evader.position, next_evader_position, agent_id=1)
        if moved:
            evader.move(next_evader_position)
        
        pursuer_positions = []        
        for pursuer in pursuers:
            next_pursuer_position = pursuer.choose_action(grid, target_position=evader.position)

            moved = grid.move_agent(pursuer.position, next_pursuer_position, agent_id=pursuer.agent_id)
            if moved:
                pursuer.move(next_pursuer_position)

        snapshots.append(grid.grid.copy())
        positions.append({
            "evader": evader.position,
            "pursuers": [p.position for p in pursuers]
        })

        logger.debug(f"Time step {time_steps}: Evader at {evader.position} \n Pursuers at ")
        for p in pursuers:
            logger.debug(f"  {p.name} at {p.position}")

        capture_occurred = any(p.position == evader.position for p in pursuers)
        time_steps += 1



    return {
        "snapshots": snapshots,
        "positions": positions,
        "grid_size": [grid.width, grid.height, grid.depth],
        "time_steps": time_steps,
        "capture_occurred": capture_occurred,
    }