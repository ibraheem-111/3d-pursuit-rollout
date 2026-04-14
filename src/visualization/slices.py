import logging
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML, clear_output, display


logger = logging.getLogger(__name__)
DEFAULT_PURSUER_COLOR = "tab:blue"


def _select_slice_indices(depth: int, max_tiles: int | None):
    if max_tiles is None:
        return np.arange(depth, dtype=int)

    tile_count = min(depth, int(max_tiles))
    return np.linspace(0, depth - 1, num=tile_count, dtype=int)


def plot_timestep_slices(
    snapshot: np.ndarray,
    timestep: int,
    show: bool = True,
    max_tiles: int | None = None,
):
    if snapshot.ndim != 3:
        raise ValueError("snapshot must be a 3D numpy array")

    depth = snapshot.shape[2]
    slice_indices = _select_slice_indices(depth=depth, max_tiles=max_tiles)
    tile_count = int(slice_indices.size)

    cols = min(3, tile_count)
    rows = int(np.ceil(tile_count / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    vmax = max(1.0, float(np.max(snapshot)))
    mappable = None

    for plot_idx, z in enumerate(slice_indices):
        ax = flat_axes[plot_idx]
        z_slice = snapshot[:, :, z].T
        mappable = ax.imshow(z_slice, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"z={z}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in flat_axes[tile_count:]:
        ax.axis("off")

    fig.suptitle(f"Timestep {timestep}")
    if mappable is not None:
        fig.colorbar(mappable, ax=flat_axes[:tile_count].tolist(), fraction=0.03, pad=0.03)

    if show:
        plt.show()
    return fig, axes


def playback_slices(snapshots, pause: float = 0.35):
    for timestep, snapshot in enumerate(snapshots):
        clear_output(wait=True)
        fig, _ = plot_timestep_slices(snapshot, timestep, show=False)
        display(fig)
        plt.pause(pause)
        plt.close(fig)


def _render_frame_rgb(snapshot: np.ndarray, timestep: int, max_tiles: int):
    fig, _ = plot_timestep_slices(
        snapshot=snapshot,
        timestep=timestep,
        show=False,
        max_tiles=max_tiles,
    )
    fig.canvas.draw()
    frame_rgba = np.asarray(fig.canvas.buffer_rgba())
    frame_rgb = np.ascontiguousarray(frame_rgba[:, :, :3])
    plt.close(fig)
    return frame_rgb


def _render_positions_frame_rgb(
    positions_history,
    grid_size,
    frame_idx: int,
    evader_color: str,
    pursuer_color: str,
):
    positions_timestep = positions_history[frame_idx]
    num_pursuers = len(positions_timestep["pursuers"])
    trails = [
        [positions_history[t]["pursuers"][i] for t in range(frame_idx + 1)]
        for i in range(num_pursuers)
    ]
    fig, _ = plot_positions_slices(
        positions_timestep=positions_timestep,
        grid_size=grid_size,
        timestep=frame_idx,
        trails=trails,
        show=False,
        evader_color=evader_color,
        pursuer_color=pursuer_color,
    )
    fig.canvas.draw()
    frame_rgba = np.asarray(fig.canvas.buffer_rgba())
    frame_rgb = np.ascontiguousarray(frame_rgba[:, :, :3])
    plt.close(fig)
    return frame_rgb


def save_snapshots_and_gif(
    snapshots,
    gif_path: str = "outputs/snapshots.gif",
    max_tiles: int = 6,
    frame_duration_ms: int = 250,
):
    return save_gif(
        snapshots=snapshots,
        gif_path=gif_path,
        max_tiles=max_tiles,
        frame_duration_ms=frame_duration_ms,
    )


def save_gif(
    snapshots=None,
    positions_history=None,
    grid_size=None,
    gif_path: str = "outputs/simulation.gif",
    max_tiles: int = 6,
    frame_duration_ms: int = 250,
    evader_color: str = "red",
    pursuer_color: str = DEFAULT_PURSUER_COLOR,
):
    if snapshots is None and positions_history is None:
        raise ValueError("Either snapshots or positions_history must be provided")
    if snapshots is not None and positions_history is not None:
        raise ValueError("Provide only one of snapshots or positions_history")
    if positions_history is not None and grid_size is None:
        raise ValueError("grid_size is required when positions_history is provided")

    if snapshots is not None and len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one frame")
    if positions_history is not None and len(positions_history) == 0:
        raise ValueError("positions_history must contain at least one frame")

    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for save_gif")

    gif_file = Path(gif_path)
    gif_file.parent.mkdir(parents=True, exist_ok=True)

    if snapshots is not None:
        first_rgb = _render_frame_rgb(snapshots[0], timestep=0, max_tiles=max_tiles)
    else:
        first_rgb = _render_positions_frame_rgb(
            positions_history=positions_history,
            grid_size=grid_size,
            frame_idx=0,
            evader_color=evader_color,
            pursuer_color=pursuer_color,
        )

    frame_height, frame_width = first_rgb.shape[0], first_rgb.shape[1]
    fps = 1000.0 / float(frame_duration_ms)

    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-filter_complex",
        "[0:v]split[p][v];[p]palettegen=stats_mode=diff[pal];[v][pal]paletteuse=dither=bayer",
        "-loop",
        "0",
        str(gif_file),
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        process.kill()
        raise RuntimeError("Failed to open ffmpeg stdin")

    frame_count = 0
    try:
        process.stdin.write(first_rgb.tobytes())
        frame_count += 1

        if snapshots is not None:
            for timestep, snapshot in enumerate(snapshots[1:], start=1):
                frame_rgb = _render_frame_rgb(snapshot, timestep=timestep, max_tiles=max_tiles)
                process.stdin.write(frame_rgb.tobytes())
                frame_count += 1
        else:
            for frame_idx in range(1, len(positions_history)):
                frame_rgb = _render_positions_frame_rgb(
                    positions_history=positions_history,
                    grid_size=grid_size,
                    frame_idx=frame_idx,
                    evader_color=evader_color,
                    pursuer_color=pursuer_color,
                )
                process.stdin.write(frame_rgb.tobytes())
                frame_count += 1

        process.stdin.close()
        stderr_output = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise

    if return_code != 0:
        stderr_text = stderr_output.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed while saving GIF: {stderr_text}")

    logger.info("Saved GIF with %d frames to %s", frame_count, gif_file)

    return {
        "gif_path": str(gif_file),
        "frame_count": frame_count,
    }


def plot_positions_slices(
    positions_timestep,
    grid_size,
    timestep=0,
    trails=None,
    show=True,
    evader_color="red",
    pursuer_color=DEFAULT_PURSUER_COLOR,
):
    width, height, depth = grid_size
    cols = min(3, depth)
    rows = int(np.ceil(depth / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    evaders = positions_timestep.get("evaders")
    if evaders is None:
        evaders = [positions_timestep["evader"]]
    pursuers = positions_timestep["pursuers"]

    for z in range(depth):
        ax = flat_axes[z]

        background = np.zeros((width, height))
        ax.imshow(background.T, origin="lower", cmap="Greys", vmin=0, vmax=1)

        if trails is not None:
            for trail in trails:
                pts = [(p.x, p.y) for p in trail if p.z == z]
                if len(pts) > 1:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, linewidth=2, alpha=0.7)

        for i, p in enumerate(pursuers):
            if p.z == z:
                ax.scatter(p.x, p.y, s=80, marker="o", color=pursuer_color)
                ax.text(p.x + 0.1, p.y + 0.1, f"P{i}", fontsize=9)

        for i, e in enumerate(evaders):
            if e.z == z:
                ax.scatter(e.x, e.y, s=100, marker="x", color=evader_color)
                ax.text(e.x + 0.1, e.y + 0.1, f"E{i}", fontsize=9)

        ax.set_title(f"z = {z}")
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.grid(True)

    for ax in flat_axes[depth:]:
        ax.axis("off")

    fig.suptitle(f"Timestep {timestep}")

    if show:
        plt.show()

    return fig, axes


def animate_positions_slices(
    positions_history,
    grid_size,
    interval=400,
    evader_color="red",
    pursuer_color=DEFAULT_PURSUER_COLOR,
):
    width, height, depth = grid_size
    cols = min(3, depth)
    rows = int(np.ceil(depth / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    def update(frame_idx):
        positions_timestep = positions_history[frame_idx]

        num_pursuers = len(positions_timestep["pursuers"])
        trails = []
        for i in range(num_pursuers):
            trail = [positions_history[t]["pursuers"][i] for t in range(frame_idx + 1)]
            trails.append(trail)

        evaders = positions_timestep.get("evaders")
        if evaders is None:
            evaders = [positions_timestep["evader"]]

        for z in range(depth):
            ax = flat_axes[z]
            ax.clear()

            background = np.zeros((width, height))
            ax.imshow(background.T, origin="lower", cmap="Greys", vmin=0, vmax=1)

            for trail in trails:
                pts = [(p.x, p.y) for p in trail if p.z == z]
                if len(pts) > 1:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, linewidth=2, alpha=0.7)

            for i, p in enumerate(positions_timestep["pursuers"]):
                if p.z == z:
                    ax.scatter(p.x, p.y, s=80, marker="o", color=pursuer_color)
                    ax.text(p.x + 0.1, p.y + 0.1, f"P{i}", fontsize=9)

            for i, e in enumerate(evaders):
                if e.z == z:
                    ax.scatter(e.x, e.y, s=100, marker="x", color=evader_color)
                    ax.text(e.x + 0.1, e.y + 0.1, f"E{i}", fontsize=9)

            ax.set_title(f"z = {z}, t = {frame_idx}")
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(-0.5, height - 0.5)
            ax.set_xticks(range(width))
            ax.set_yticks(range(height))
            ax.grid(True)

        for ax in flat_axes[depth:]:
            ax.axis("off")

    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=interval)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_visit_heatmaps(positions_history, grid_size, show=True):
    width, height, depth = grid_size
    visits = np.zeros((width, height, depth), dtype=int)

    for step in positions_history:
        for p in step["pursuers"]:
            visits[p.x, p.y, p.z] += 1

    cols = min(3, depth)
    rows = int(np.ceil(depth / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    for z in range(depth):
        ax = flat_axes[z]
        ax.imshow(visits[:, :, z].T, origin="lower", cmap="viridis")
        ax.set_title(f"Pursuer visits at z={z}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

    for ax in flat_axes[depth:]:
        ax.axis("off")

    if show:
        plt.show()
    return fig, axes

def plot_3d_trajectories(
    positions_history,
    show=True,
    evader_color="red",
    pursuer_color=DEFAULT_PURSUER_COLOR,
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _evaders_for_step(step):
        evaders = step.get("evaders")
        if evaders is not None:
            return evaders
        if "evader" in step:
            return [step["evader"]]
        return []

    num_pursuers = len(positions_history[0]["pursuers"])

    for i in range(num_pursuers):
        xs = [step["pursuers"][i].x for step in positions_history]
        ys = [step["pursuers"][i].y for step in positions_history]
        zs = [step["pursuers"][i].z for step in positions_history]
        ax.plot(xs, ys, zs, marker="o", label=f"P{i}", color=pursuer_color)

    max_evaders = max(len(_evaders_for_step(step)) for step in positions_history)
    for i in range(max_evaders):
        evader_positions = []
        for step in positions_history:
            evaders = _evaders_for_step(step)
            if i < len(evaders):
                evader_positions.append(evaders[i])
        if len(evader_positions) == 0:
            continue

        ex = [position.x for position in evader_positions]
        ey = [position.y for position in evader_positions]
        ez = [position.z for position in evader_positions]
        ax.plot(ex, ey, ez, marker="x", linestyle="--", label=f"E{i}", color=evader_color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax