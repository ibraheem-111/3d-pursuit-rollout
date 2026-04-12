import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

from PIL import Image
from IPython.display import clear_output, display


logger = logging.getLogger(__name__)


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


def save_snapshots_and_gif(
    snapshots,
    snapshot_dir: str = "outputs/snapshots",
    gif_path: str = "outputs/snapshots.gif",
    max_tiles: int = 6,
    frame_duration_ms: int = 250,
):
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one frame")

    output_dir = Path(snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_png in output_dir.glob("snapshot_*.png"):
        existing_png.unlink()

    gif_file = Path(gif_path)
    gif_file.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for timestep, snapshot in enumerate(snapshots):
        fig, _ = plot_timestep_slices(
            snapshot=snapshot,
            timestep=timestep,
            show=False,
            max_tiles=max_tiles,
        )
        frame_path = output_dir / f"snapshot_{timestep:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    first_frame = Image.open(frame_paths[0])
    append_frames = [Image.open(path) for path in frame_paths[1:]]
    first_frame.save(
        gif_file,
        save_all=True,
        append_images=append_frames,
        duration=frame_duration_ms,
        loop=0,
    )
    first_frame.close()
    for frame in append_frames:
        frame.close()

    logger.info(
        "Saved %d snapshots to %s and GIF to %s",
        len(frame_paths),
        output_dir,
        gif_file,
    )

    return {
        "snapshot_dir": str(output_dir),
        "gif_path": str(gif_file),
        "frame_count": len(frame_paths),
    }
