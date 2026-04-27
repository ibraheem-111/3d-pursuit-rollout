from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_types import GameState
from src.data_types.postion import Position
from src.planner.signaling_kernel import apply_action_label, state_features, weighted_knn_vote
from src.utils.math_utils import manhattan_distance


def closest_evader_position(position: Position, evader_positions) -> Position:
    return min(
        evader_positions,
        key=lambda evader_position: manhattan_distance(position, evader_position),
    )


@dataclass(frozen=True)
class KernelSignalingModel:
    X_train: np.ndarray
    y_train: np.ndarray
    grid_size: tuple[int, int, int]
    k: int = 25
    sigma: float = 5.0
    include_evader_position: bool = True
    normalize: bool = True
    metadata: dict | None = None

    @classmethod
    def load(cls, path, *, k=None, sigma=None):
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
        return cls(
            X_train=np.asarray(data["X_train"], dtype=float),
            y_train=np.asarray(data["y_train"], dtype=int),
            grid_size=tuple(int(v) for v in data["grid_size"]),
            k=int(k if k is not None else metadata.get("k", 25)),
            sigma=float(sigma if sigma is not None else metadata.get("sigma", 5.0)),
            include_evader_position=bool(metadata.get("include_evader_position", True)),
            normalize=bool(metadata.get("normalize", True)),
            metadata=metadata,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata or {})
        metadata.update(
            {
                "k": int(self.k),
                "sigma": float(self.sigma),
                "include_evader_position": bool(self.include_evader_position),
                "normalize": bool(self.normalize),
            }
        )
        np.savez_compressed(
            path,
            X_train=np.asarray(self.X_train, dtype=float),
            y_train=np.asarray(self.y_train, dtype=int),
            grid_size=np.asarray(self.grid_size, dtype=int),
            metadata=json.dumps(metadata, sort_keys=True),
        )

    def validate_for_state(self, state: GameState):
        expected_width = 3 * len(state.pursuer_positions)
        if self.include_evader_position:
            expected_width += 3
        if self.X_train.ndim != 2:
            raise ValueError("signaling model X_train must be 2D")
        if self.X_train.shape[1] != expected_width:
            raise ValueError(
                f"signaling model feature width {self.X_train.shape[1]} does not match "
                f"state feature width {expected_width}"
            )


class GreedySignalingPolicy:
    def __init__(self, grid_model):
        self.grid_model = grid_model

    def predict_move(self, state: GameState, pursuer_index: int, pursuer_agents):
        current_position = state.pursuer_positions[pursuer_index]
        target_position = closest_evader_position(current_position, state.evader_positions)
        return pursuer_agents[pursuer_index].choose_action_from_state(
            current_position=current_position,
            target_position=target_position,
            grid_model=self.grid_model,
            pursuer_positions=list(state.pursuer_positions),
            evader_positions=list(state.evader_positions),
            pursuer_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
        )


class KernelLearnedSignalingPolicy:
    def __init__(self, grid_model, model: KernelSignalingModel):
        self.grid_model = grid_model
        self.model = model
        self.invalid_prediction_count = 0
        self.prediction_count = 0

    @classmethod
    def load(cls, grid_model, path, *, k=None, sigma=None):
        return cls(grid_model=grid_model, model=KernelSignalingModel.load(path, k=k, sigma=sigma))

    def _feature_for(self, state: GameState, pursuer_index: int):
        current_position = state.pursuer_positions[pursuer_index]
        reference_evader = closest_evader_position(current_position, state.evader_positions)
        return state_features(
            state,
            reference_evader_position=reference_evader,
            focus_pursuer_index=pursuer_index,
            grid_size=self.model.grid_size,
            include_evader_position=self.model.include_evader_position,
            normalize=self.model.normalize,
        )

    def _valid_moves(self, state: GameState, pursuer_index: int, pursuer_agents):
        return self.grid_model.get_valid_moves(
            position=state.pursuer_positions[pursuer_index],
            agent_id=pursuer_agents[pursuer_index].agent_id,
            occupied_positions=list(state.pursuer_positions),
            evader_positions=list(state.evader_positions),
            occupied_agent_ids=[pursuer.agent_id for pursuer in pursuer_agents],
        )

    def predict_move(self, state: GameState, pursuer_index: int, pursuer_agents):
        self.model.validate_for_state(state)
        self.prediction_count += 1

        current_position = state.pursuer_positions[pursuer_index]
        valid_moves = self._valid_moves(state, pursuer_index, pursuer_agents)
        if len(valid_moves) == 0:
            return current_position

        valid_moves_by_tuple = {move.as_tuple(): move for move in valid_moves}
        x_feat = self._feature_for(state, pursuer_index)
        vote = weighted_knn_vote(
            self.model.X_train,
            self.model.y_train,
            x_feat,
            k=self.model.k,
            sigma=self.model.sigma,
        )

        labels_by_score = sorted(vote.scores, key=lambda label: (-vote.scores[label], label))
        for label in labels_by_score:
            predicted_position = apply_action_label(current_position, label)
            valid_move = valid_moves_by_tuple.get(predicted_position.as_tuple())
            if valid_move is not None:
                return valid_move

        self.invalid_prediction_count += 1
        return GreedySignalingPolicy(self.grid_model).predict_move(state, pursuer_index, pursuer_agents)

    @property
    def invalid_prediction_rate(self):
        if self.prediction_count == 0:
            return 0.0
        return self.invalid_prediction_count / self.prediction_count
