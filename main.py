from grid import Grid
from agents.evader import RandomWalkEvaderAgent
from agents.pursuer import GreedyAgent
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the grid application.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    parser.add_argument("--pursuer-type", type=str, help="Type of the pursuer.")
    parser.add_argument("--evader-type", type=str, help="Type of the evader.",
                        default="random")

    parser.add_argument("--grid-size", type=str, help="Size of the grid. Format: " \
    "[ width, height, depth ]", default="[10, 10, 10]")
        
    return parser.parse_args()


def main():
    args = parse_args()
    grid = Grid() 

if __name__ == "__main__":
    main()
