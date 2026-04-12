from src.grid import Grid3D
from src.agents.evader import RandomWalkEvaderAgent
from src.agents.pursuer import GreedyAgent
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def grid_array(string):
    
    arr = eval(string)
    if not isinstance(arr, list):
        raise argparse.ArgumentTypeError("Grid size must be a list of three integers: [width, height, depth]")
    if len(arr) != 3:
        raise argparse.ArgumentTypeError("Grid size must have 3 integers")
    if not all(isinstance(x, int) for x in arr):
        raise argparse.ArgumentTypeError("All elements in grid size must be integers")

    return arr

def parse_args():
    parser = argparse.ArgumentParser(description="Run the grid application.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    parser.add_argument("--pursuer-type", type=str, help="Type of the pursuer.")
    parser.add_argument("--evader-type", type=str, help="Type of the evader.",
                        default="random")

    parser.add_argument("--grid-size", type=grid_array, help="Size of the grid. Format: " \
    "[ width, height, depth ]", default="[10, 10, 10]")
        
    return parser.parse_args()


def main():
    args = parse_args()
    grid = None    

    if args.grid_size: 
        grid_size = args.grid_size
        width, height, depth = grid_size
        logger.info(f"Using grid size from command line: {grid_size}")
        grid = Grid3D(width, height, depth)

    run_simulation(grid, args)

if __name__ == "__main__":
    main()
