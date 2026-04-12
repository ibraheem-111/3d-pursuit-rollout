from src.grid import Grid3D
from src.simulation.simulation import run_simulation
from src.visualization import save_snapshots_and_gif
import logging
import argparse
import ast
import yaml
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_run_output_dir(base_dir: str = "outputs") -> Path:
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = Path(base_dir) / timestamp
    if run_dir.exists():
        raise RuntimeError(f"run output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_config(run_dir: Path, config_path: str):
    source_path = Path(config_path)
    destination_path = run_dir / "config.yaml"
    shutil.copy2(source_path, destination_path)
    return destination_path

def grid_array(string):

    arr = ast.literal_eval(string)
    if not isinstance(arr, list):
        raise argparse.ArgumentTypeError("Grid size must be a list of three integers: [width, height, depth]")
    if len(arr) != 3:
        raise argparse.ArgumentTypeError("Grid size must have 3 integers")
    if not all(isinstance(x, int) for x in arr):
        raise argparse.ArgumentTypeError("All elements in grid size must be integers")

    return arr

def parse_args():
    parser = argparse.ArgumentParser(description="Run the grid application.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.", default="./config.yaml")

    parser.add_argument("--pursuer-type", type=str, help="Type of the pursuer.")
    parser.add_argument("--evader-type", type=str, help="Type of the evader.",
                        default="random")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of simulation steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible movement.")

    parser.add_argument("--grid-size", type=grid_array, help="Size of the grid. Format: " \
    "[ width, height, depth ]", default=[10, 10, 3])

    parser.add_argument("--save-snapshots", action="store_true", help="Whether to save snapshots of the grid at each timestep.")
        
    return parser.parse_args()


def load_args_and_config():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return args, config


def main():
    args, config = load_args_and_config()
    run_dir = create_run_output_dir()

    grid_size = args.grid_size
    width, height, depth = grid_size
    logger.info(f"Using grid size from command line: {grid_size}")
    grid = Grid3D(width, height, depth)

    saved_config_path = save_run_config(run_dir=run_dir, config_path=args.config)
    logger.info(f"Run output directory: {run_dir}")
    logger.info(f"Saved run config to: {saved_config_path}")

    result = run_simulation(grid, args, config=config)

    if args.save_snapshots:
        export_result = save_snapshots_and_gif(
            snapshots=result["snapshots"],
            snapshot_dir=str(run_dir / "snapshots"),
            gif_path=str(run_dir / "snapshots.gif"),
            max_tiles=6,
        )
        logger.info(
            "Saved snapshot frames to %s and GIF to %s",
            export_result["snapshot_dir"],
            export_result["gif_path"],
        )

    logger.info(f"Simulation complete. Final evader position: {result['positions'][-1].as_tuple()}")

if __name__ == "__main__":
    main()
