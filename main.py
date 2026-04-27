from src.grid import Grid3D
from src.simulation.simulation import run_simulation
from src.visualization import plot_3d_trajectories, plot_visit_heatmaps, save_gif
import matplotlib.pyplot as plt
import logging
import argparse
import ast
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

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
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    parser.add_argument(
        "--strategy",
        choices=["greedy", "non_autonomous_rollout", "autonomous_greedy_signaling"],
        default=None,
        help="Simulation strategy. Overrides strategy in the config.",
    )

    parser.add_argument("--pursuer-type", type=str, help="Optional pursuer strategy override.")
    parser.add_argument("--evader-type", type=str, help="Type of the evader.",
                        default=None)
    parser.add_argument("--num-steps", type=int, default=20, help="Number of simulation steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible movement.")

    parser.add_argument("--save-gif", action="store_true", help="Whether to save a GIF of the rollout.")
    parser.add_argument("--plot-heatmap", action="store_true", help="Whether to save pursuer visit heatmaps.")
    parser.add_argument("--plot-3d-trajectory", action="store_true", help="Whether to save 3D trajectories.")
    parser.add_argument("--plot-all", action="store_true", help="Whether to save all plot outputs.")
        
    return parser.parse_args()


def load_args_and_config():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return args, config


def main():
    args, config = load_args_and_config()
    run_dir = create_run_output_dir()

    grid_size = config["simulation"]["grid_size"]
    width, height, depth = grid_size
    logger.info(f"Using grid size from config: {grid_size}")
    grid = Grid3D(width, height, depth)

    saved_config_path = save_run_config(run_dir=run_dir, config_path=args.config)
    logger.info(f"Run output directory: {run_dir}")
    logger.info(f"Saved run config to: {saved_config_path}")

    result = run_simulation(grid, args, config=config)

    logger.info(f"Simulation finished after {result['time_steps']} time steps. Capture occurred: {result['capture_occurred']}")

    plot_heatmap = args.plot_all or args.plot_heatmap
    plot_3d_trajectory = args.plot_all or args.plot_3d_trajectory
    plot_gif = args.save_gif or args.plot_all

    visualization_config = config["visualization"]
    evader_color = visualization_config["evader_color"]
    pursuer_color = visualization_config.get("pursuer_color", "tab:blue")

    # save positions into a dataframe and export as csv
    positions = result["positions"]
    positions_path = run_dir / "positions.csv"
    logger.info(positions[0])
    logger.info(type(positions[0]))
    df = pd.DataFrame(positions)
    df.to_csv(positions_path, index=False)
    logger.info(f"Saved positions to {positions_path}")

    metrics = result.get("metrics")
    if metrics is not None:
        metrics_json_path = run_dir / "metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", metrics_json_path)

        metrics_csv_path = run_dir / "metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
        logger.info("Saved metrics CSV to %s", metrics_csv_path)

    if plot_heatmap:
        fig, _ = plot_visit_heatmaps(
            positions_history=result["positions"],
            grid_size=result["grid_size"],
            show=False,
        )
        heatmap_path = run_dir / "visit_heatmap.png"
        fig.savefig(heatmap_path, dpi=140)
        plt.close(fig)
        logger.info("Saved visit heatmap to %s", heatmap_path)

    if plot_3d_trajectory:
        fig, _ = plot_3d_trajectories(
            positions_history=result["positions"],
            show=False,
            evader_color=evader_color,
            pursuer_color=pursuer_color,
        )
        trajectory_path = run_dir / "trajectory_3d.png"
        fig.savefig(trajectory_path, dpi=140)
        plt.close(fig)
        logger.info("Saved 3D trajectory plot to %s", trajectory_path)

    if plot_gif:
        export_result = save_gif(
            positions_history=result["positions"],
            grid_size=result["grid_size"],
            gif_path=str(run_dir / "simulation.gif"),
            max_tiles=6,
            evader_color=evader_color,
            pursuer_color=pursuer_color,
        )
        logger.info(
            "Saved GIF to %s",
            export_result["gif_path"],
        )

    final_state = result["positions"][-1]
    final_evaders = final_state["evaders"]

    if len(final_evaders) == 0:
        logger.info("Simulation complete. All evaders captured.")
    elif len(final_evaders) == 1:
        logger.info("Simulation complete. Final evader position: %s", final_evaders[0].as_tuple())
    else:
        logger.info(
            "Simulation complete. Remaining evaders: %d. Positions: %s",
            len(final_evaders),
            [position.as_tuple() for position in final_evaders],
        )

if __name__ == "__main__":
    main()
