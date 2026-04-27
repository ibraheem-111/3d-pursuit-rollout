from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from src.data_types import GameState
from src.data_types.postion import Position


ACTION_DELTAS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
ACTION_LABEL_BY_DELTA = {delta: idx for idx, delta in enumerate(ACTION_DELTAS)}


@dataclass(frozen=True)
class KernelVoteResult:
    label: int
    scores: Dict[int, float]
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    neighbor_weights: np.ndarray


def _axis_scale(grid_size, axis: int) -> float:
    if grid_size is None:
        return 1.0

    max_coordinate = float(grid_size[axis] - 1)
    if max_coordinate <= 0.0:
        return 1.0
    return max_coordinate


def state_features(
    state: GameState,
    *,
    reference_evader_position: Position | None = None,
    focus_pursuer_index: int | None = None,
    grid_size=None,
    include_evader_position: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """Relative pursuer-evader geometry used by the signaling similarity kernel."""
    evader_position = reference_evader_position if reference_evader_position is not None else state.evader_position
    scales = (
        _axis_scale(grid_size, 0) if normalize else 1.0,
        _axis_scale(grid_size, 1) if normalize else 1.0,
        _axis_scale(grid_size, 2) if normalize else 1.0,
    )

    features = []
    pursuer_positions = _ordered_pursuer_positions(state.pursuer_positions, focus_pursuer_index)
    for pursuer_position in pursuer_positions:
        features.extend(
            [
                (pursuer_position.x - evader_position.x) / scales[0],
                (pursuer_position.y - evader_position.y) / scales[1],
                (pursuer_position.z - evader_position.z) / scales[2],
            ]
        )

    if include_evader_position:
        features.extend(
            [
                evader_position.x / scales[0],
                evader_position.y / scales[1],
                evader_position.z / scales[2],
            ]
        )

    return np.array(features, dtype=float)


def _ordered_pursuer_positions(pursuer_positions, focus_pursuer_index):
    pursuer_positions = tuple(pursuer_positions)
    if focus_pursuer_index is None:
        return pursuer_positions

    focus_pursuer_index = int(focus_pursuer_index)
    if focus_pursuer_index < 0 or focus_pursuer_index >= len(pursuer_positions):
        raise IndexError(f"focus_pursuer_index out of range: {focus_pursuer_index}")

    return (
        pursuer_positions[focus_pursuer_index],
        *pursuer_positions[:focus_pursuer_index],
        *pursuer_positions[focus_pursuer_index + 1:],
    )


def kernel_distance(x_feat: np.ndarray, y_feat: np.ndarray) -> float:
    x_feat = np.asarray(x_feat, dtype=float)
    y_feat = np.asarray(y_feat, dtype=float)
    if x_feat.shape != y_feat.shape:
        raise ValueError(f"feature shapes must match: {x_feat.shape} != {y_feat.shape}")
    return float(np.sum(np.abs(x_feat - y_feat)))


def kernel_weight(distance: float, sigma: float = 5.0) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return float(np.exp(-float(distance) / sigma))


def kernel_distances(X_train: np.ndarray, x_feat: np.ndarray) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    x_feat = np.asarray(x_feat, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")
    if x_feat.ndim != 1:
        raise ValueError("x_feat must be a 1D array")
    if X_train.shape[1] != x_feat.shape[0]:
        raise ValueError(f"feature width mismatch: {X_train.shape[1]} != {x_feat.shape[0]}")
    return np.sum(np.abs(X_train - x_feat), axis=1)


def weighted_knn_vote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_feat: np.ndarray,
    *,
    k: int,
    sigma: float = 5.0,
) -> KernelVoteResult:
    if k <= 0:
        raise ValueError("k must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    y_train = np.asarray(y_train)
    if y_train.ndim != 1:
        raise ValueError("y_train must be a 1D array of action labels")

    distances = kernel_distances(X_train, x_feat)
    if distances.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of rows")
    if distances.shape[0] == 0:
        raise ValueError("cannot vote with an empty training set")

    neighbor_count = min(k, distances.shape[0])
    neighbor_indices = np.argsort(distances, kind="stable")[:neighbor_count]
    neighbor_distances = distances[neighbor_indices]
    neighbor_weights = np.exp(-neighbor_distances / sigma)

    scores: Dict[int, float] = {}
    for label, weight in zip(y_train[neighbor_indices], neighbor_weights):
        label = int(label)
        scores[label] = scores.get(label, 0.0) + float(weight)

    best_label = min(
        scores,
        key=lambda label: (
            -scores[label],
            _mean_distance_for_label(label, y_train, neighbor_indices, neighbor_distances),
            label,
        ),
    )

    return KernelVoteResult(
        label=int(best_label),
        scores=scores,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        neighbor_weights=neighbor_weights,
    )


def _mean_distance_for_label(label: int, y_train, neighbor_indices, neighbor_distances) -> float:
    label_mask = y_train[neighbor_indices].astype(int) == int(label)
    return float(np.mean(neighbor_distances[label_mask]))


def action_label(current_position: Position, next_position: Position) -> int:
    delta = (
        next_position.x - current_position.x,
        next_position.y - current_position.y,
        next_position.z - current_position.z,
    )
    if delta not in ACTION_LABEL_BY_DELTA:
        raise ValueError(f"unsupported action delta for signaling label: {delta}")
    return ACTION_LABEL_BY_DELTA[delta]


def apply_action_label(current_position: Position, label: int) -> Position:
    label = int(label)
    if label < 0 or label >= len(ACTION_DELTAS):
        raise ValueError(f"unknown action label: {label}")

    try:
        dx, dy, dz = ACTION_DELTAS[label]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unknown action label: {label}") from exc

    return Position(
        x=current_position.x + dx,
        y=current_position.y + dy,
        z=current_position.z + dz,
    )


def action_labels(current_positions: Iterable[Position], next_positions: Iterable[Position]) -> np.ndarray:
    return np.array(
        [
            action_label(current_position, next_position)
            for current_position, next_position in zip(current_positions, next_positions)
        ],
        dtype=int,
    )
