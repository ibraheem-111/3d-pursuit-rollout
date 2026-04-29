from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from src.planner.signaling_kernel import weighted_knn_vote


NodeId = int

ACTION_NAMES = ("stay", "east", "west", "north", "south")
ACTION_DELTAS = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, -1),
    (0, 1),
)
ACTION_LABEL_BY_DELTA = {delta: idx for idx, delta in enumerate(ACTION_DELTAS)}


@dataclass(frozen=True)
class RescueSignalingDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    metadata: dict


def rescue_state_features(problem, state, focus_agent_index: int) -> np.ndarray:
    """Observable rescue-search features for learned signaling."""
    width, height = _require_grid_geometry(problem)
    focus_agent_index = int(focus_agent_index)
    if focus_agent_index < 0 or focus_agent_index >= len(state.agent_nodes):
        raise IndexError(f"focus_agent_index out of range: {focus_agent_index}")

    x_scale = _axis_scale(width)
    y_scale = _axis_scale(height)
    focus_node = int(state.agent_nodes[focus_agent_index])
    focus_x, focus_y = _grid_coordinates(focus_node, width)
    ordered_agent_nodes = (
        focus_node,
        *state.agent_nodes[:focus_agent_index],
        *state.agent_nodes[focus_agent_index + 1 :],
    )

    features = []
    for node in ordered_agent_nodes:
        x, y = _grid_coordinates(int(node), width)
        features.extend(
            [
                x / x_scale,
                y / y_scale,
                (x - focus_x) / x_scale,
                (y - focus_y) / y_scale,
            ]
        )

    features.extend(_node_mask(problem, state.explored_nodes))
    features.extend(valid_action_mask(problem, state, focus_agent_index))

    if problem.target_knowledge == "known":
        unfound_targets = [
            node
            for idx, node in enumerate(problem.lost_individual_nodes)
            if idx not in set(state.found_individuals)
        ]
        features.extend(_node_mask(problem, unfound_targets))

    return np.asarray(features, dtype=float)


def valid_action_mask(problem, state, focus_agent_index: int) -> np.ndarray:
    current_node = int(state.agent_nodes[focus_agent_index])
    valid_nodes = set(_valid_moves(problem, current_node))
    mask = []
    for label in range(len(ACTION_NAMES)):
        predicted_node = apply_action_label(problem, current_node, label)
        mask.append(1.0 if predicted_node is not None and predicted_node in valid_nodes else 0.0)
    return np.asarray(mask, dtype=float)


def action_label(problem, current_node: NodeId, next_node: NodeId) -> int:
    width, _height = _require_grid_geometry(problem)
    current_x, current_y = _grid_coordinates(int(current_node), width)
    next_x, next_y = _grid_coordinates(int(next_node), width)
    delta = (next_x - current_x, next_y - current_y)
    if delta not in ACTION_LABEL_BY_DELTA:
        raise ValueError(f"unsupported rescue action delta for signaling label: {delta}")
    return int(ACTION_LABEL_BY_DELTA[delta])


def apply_action_label(problem, current_node: NodeId, label: int) -> NodeId | None:
    width, height = _require_grid_geometry(problem)
    label = int(label)
    if label < 0 or label >= len(ACTION_DELTAS):
        raise ValueError(f"unknown rescue action label: {label}")

    x, y = _grid_coordinates(int(current_node), width)
    dx, dy = ACTION_DELTAS[label]
    next_x = x + dx
    next_y = y + dy
    if not (0 <= next_x < width and 0 <= next_y < height):
        return None

    next_node = _grid_node_id(next_x, next_y, width)
    if next_node not in problem.adjacency:
        return None
    return int(next_node)


def save_rescue_signaling_dataset(path, X_train, y_train, metadata: Mapping[str, object]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.update(
        {
            "action_names": list(ACTION_NAMES),
            "feature_width": int(np.asarray(X_train).shape[1]),
            "num_examples": int(np.asarray(y_train).shape[0]),
        }
    )
    np.savez_compressed(
        path,
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=int),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def load_rescue_signaling_dataset(path) -> RescueSignalingDataset:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    return RescueSignalingDataset(
        X_train=np.asarray(data["X_train"], dtype=float),
        y_train=np.asarray(data["y_train"], dtype=int),
        metadata=metadata,
    )


@dataclass(frozen=True)
class KernelRescueSignalingModel:
    X_train: np.ndarray
    y_train: np.ndarray
    num_nodes: int
    num_agents: int
    target_knowledge: str
    grid_width: int
    grid_height: int
    k: int = 25
    sigma: float = 5.0
    metadata: dict | None = None

    @property
    def model_type(self) -> str:
        return "kernel_knn"

    @classmethod
    def load(cls, path, *, k=None, sigma=None):
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
        return cls(
            X_train=np.asarray(data["X_train"], dtype=float),
            y_train=np.asarray(data["y_train"], dtype=int),
            num_nodes=int(metadata["num_nodes"]),
            num_agents=int(metadata["num_agents"]),
            target_knowledge=str(metadata["target_knowledge"]),
            grid_width=int(metadata["grid_width"]),
            grid_height=int(metadata["grid_height"]),
            k=int(k if k is not None else metadata.get("k", 25)),
            sigma=float(sigma if sigma is not None else metadata.get("sigma", 5.0)),
            metadata=metadata,
        )

    @classmethod
    def from_dataset(cls, dataset: RescueSignalingDataset, *, k: int = 25, sigma: float = 5.0):
        metadata = dict(dataset.metadata)
        return cls(
            X_train=np.asarray(dataset.X_train, dtype=float),
            y_train=np.asarray(dataset.y_train, dtype=int),
            num_nodes=int(metadata["num_nodes"]),
            num_agents=int(metadata["num_agents"]),
            target_knowledge=str(metadata["target_knowledge"]),
            grid_width=int(metadata["grid_width"]),
            grid_height=int(metadata["grid_height"]),
            k=int(k),
            sigma=float(sigma),
            metadata=metadata,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata or {})
        metadata.update(_model_metadata(self))
        metadata.update({"k": int(self.k), "sigma": float(self.sigma)})
        np.savez_compressed(
            path,
            X_train=np.asarray(self.X_train, dtype=float),
            y_train=np.asarray(self.y_train, dtype=int),
            metadata=json.dumps(metadata, sort_keys=True),
        )

    def validate_for_problem_state(self, problem, state):
        _validate_model_geometry(self, problem, state)
        if self.X_train.ndim != 2:
            raise ValueError("rescue signaling model X_train must be 2D")
        expected_width = rescue_state_features(problem, state, 0).shape[0]
        if self.X_train.shape[1] != expected_width:
            raise ValueError(
                f"rescue signaling feature width {self.X_train.shape[1]} does not match "
                f"state feature width {expected_width}"
            )

    def predict_label_scores(self, x_feat: np.ndarray) -> Dict[int, float]:
        vote = weighted_knn_vote(
            self.X_train,
            self.y_train,
            np.asarray(x_feat, dtype=float),
            k=self.k,
            sigma=self.sigma,
        )
        return {int(label): float(score) for label, score in vote.scores.items()}


@dataclass(frozen=True)
class MLPRescueSignalingModel:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    num_nodes: int
    num_agents: int
    target_knowledge: str
    grid_width: int
    grid_height: int
    feature_width: int
    metadata: dict | None = None

    @property
    def model_type(self) -> str:
        return "mlp"

    @classmethod
    def train(
        cls,
        dataset: RescueSignalingDataset,
        *,
        hidden_layers: Sequence[int] = (32,),
        learning_rate: float = 0.05,
        epochs: int = 500,
        seed: int = 0,
    ):
        X_train = np.asarray(dataset.X_train, dtype=float)
        y_train = np.asarray(dataset.y_train, dtype=int)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D")
        if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
            raise ValueError("y_train must be 1D and match X_train rows")
        if np.any(y_train < 0) or np.any(y_train >= len(ACTION_NAMES)):
            raise ValueError("y_train contains unsupported rescue action labels")

        hidden_layers = tuple(int(size) for size in hidden_layers)
        if any(size <= 0 for size in hidden_layers):
            raise ValueError("hidden layer sizes must be positive")

        rng = np.random.default_rng(seed)
        layer_sizes = (X_train.shape[1], *hidden_layers, len(ACTION_NAMES))
        weights = []
        biases = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / float(in_size + out_size))
            weights.append(rng.uniform(-limit, limit, size=(in_size, out_size)))
            biases.append(np.zeros(out_size, dtype=float))

        y_one_hot = np.zeros((y_train.shape[0], len(ACTION_NAMES)), dtype=float)
        y_one_hot[np.arange(y_train.shape[0]), y_train] = 1.0

        for _epoch in range(int(epochs)):
            activations = [X_train]
            pre_activations = []
            current = X_train
            for weight, bias in zip(weights[:-1], biases[:-1]):
                z = current @ weight + bias
                pre_activations.append(z)
                current = np.maximum(z, 0.0)
                activations.append(current)

            logits = current @ weights[-1] + biases[-1]
            probabilities = _softmax(logits)
            grad = (probabilities - y_one_hot) / float(X_train.shape[0])

            grad_weights = [None] * len(weights)
            grad_biases = [None] * len(biases)
            grad_weights[-1] = activations[-1].T @ grad
            grad_biases[-1] = np.sum(grad, axis=0)
            grad_activation = grad @ weights[-1].T

            for layer_idx in range(len(weights) - 2, -1, -1):
                grad_z = grad_activation * (pre_activations[layer_idx] > 0.0)
                grad_weights[layer_idx] = activations[layer_idx].T @ grad_z
                grad_biases[layer_idx] = np.sum(grad_z, axis=0)
                if layer_idx > 0:
                    grad_activation = grad_z @ weights[layer_idx].T

            for layer_idx in range(len(weights)):
                weights[layer_idx] = weights[layer_idx] - learning_rate * grad_weights[layer_idx]
                biases[layer_idx] = biases[layer_idx] - learning_rate * grad_biases[layer_idx]

        metadata = dict(dataset.metadata)
        metadata.update(
            {
                "hidden_layers": list(hidden_layers),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
                "seed": int(seed),
            }
        )
        return cls(
            weights=tuple(np.asarray(weight, dtype=float) for weight in weights),
            biases=tuple(np.asarray(bias, dtype=float) for bias in biases),
            num_nodes=int(metadata["num_nodes"]),
            num_agents=int(metadata["num_agents"]),
            target_knowledge=str(metadata["target_knowledge"]),
            grid_width=int(metadata["grid_width"]),
            grid_height=int(metadata["grid_height"]),
            feature_width=int(X_train.shape[1]),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
        layer_count = int(metadata["layer_count"])
        weights = tuple(np.asarray(data[f"weight_{idx}"], dtype=float) for idx in range(layer_count))
        biases = tuple(np.asarray(data[f"bias_{idx}"], dtype=float) for idx in range(layer_count))
        return cls(
            weights=weights,
            biases=biases,
            num_nodes=int(metadata["num_nodes"]),
            num_agents=int(metadata["num_agents"]),
            target_knowledge=str(metadata["target_knowledge"]),
            grid_width=int(metadata["grid_width"]),
            grid_height=int(metadata["grid_height"]),
            feature_width=int(metadata["feature_width"]),
            metadata=metadata,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata or {})
        metadata.update(_model_metadata(self))
        metadata.update(
            {
                "feature_width": int(self.feature_width),
                "layer_count": int(len(self.weights)),
            }
        )
        arrays = {
            "metadata": json.dumps(metadata, sort_keys=True),
        }
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"weight_{idx}"] = np.asarray(weight, dtype=float)
            arrays[f"bias_{idx}"] = np.asarray(bias, dtype=float)
        np.savez_compressed(path, **arrays)

    def validate_for_problem_state(self, problem, state):
        _validate_model_geometry(self, problem, state)
        expected_width = rescue_state_features(problem, state, 0).shape[0]
        if self.feature_width != expected_width:
            raise ValueError(
                f"rescue signaling feature width {self.feature_width} does not match "
                f"state feature width {expected_width}"
            )

    def predict_label_scores(self, x_feat: np.ndarray) -> Dict[int, float]:
        probabilities = self._predict_proba(np.asarray(x_feat, dtype=float).reshape(1, -1))[0]
        return {int(label): float(probability) for label, probability in enumerate(probabilities)}

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        current = np.asarray(X, dtype=float)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            current = np.maximum(current @ weight + bias, 0.0)
        return _softmax(current @ self.weights[-1] + self.biases[-1])


class RescueLearnedSignalingPolicy:
    def __init__(self, model, model_path=None):
        self.model = model
        self.model_path = str(model_path) if model_path is not None else None
        self.prediction_count = 0
        self.invalid_prediction_count = 0

    @classmethod
    def load(cls, path, *, model_type=None, k=None, sigma=None):
        model = load_rescue_signaling_model(path, model_type=model_type, k=k, sigma=sigma)
        return cls(model=model, model_path=path)

    @property
    def model_type(self) -> str:
        return self.model.model_type

    @property
    def invalid_prediction_rate(self) -> float:
        if self.prediction_count == 0:
            return 0.0
        return self.invalid_prediction_count / self.prediction_count

    def predict_joint_action(self, problem, state, oracle=None, fallback_action=None) -> tuple[NodeId, ...]:
        self.model.validate_for_problem_state(problem, state)
        if fallback_action is None:
            fallback_action = state.agent_nodes
        return tuple(
            self.predict_action(problem, state, agent_idx, fallback_action=fallback_action)
            for agent_idx in range(len(state.agent_nodes))
        )

    def predict_action(self, problem, state, agent_idx: int, *, fallback_action=None) -> NodeId:
        self.prediction_count += 1
        current_node = int(state.agent_nodes[agent_idx])
        valid_nodes = set(_valid_moves(problem, current_node))
        x_feat = rescue_state_features(problem, state, agent_idx)
        scores = self.model.predict_label_scores(x_feat)
        labels_by_score = sorted(scores, key=lambda label: (-scores[label], label))

        for label in labels_by_score:
            predicted_node = apply_action_label(problem, current_node, label)
            if predicted_node is not None and predicted_node in valid_nodes:
                return int(predicted_node)

        self.invalid_prediction_count += 1
        if fallback_action is not None:
            fallback_node = int(fallback_action[agent_idx])
            if fallback_node in valid_nodes:
                return fallback_node
        if current_node in valid_nodes:
            return current_node
        return int(min(valid_nodes))


def load_rescue_signaling_model(path, *, model_type=None, k=None, sigma=None):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    actual_model_type = str(metadata.get("model_type", "kernel_knn" if "X_train" in data else "mlp"))
    if model_type is not None and str(model_type) != actual_model_type:
        raise ValueError(f"rescue signaling model type {actual_model_type!r} does not match {model_type!r}")
    if actual_model_type == "kernel_knn":
        return KernelRescueSignalingModel.load(path, k=k, sigma=sigma)
    if actual_model_type == "mlp":
        return MLPRescueSignalingModel.load(path)
    raise ValueError(f"unsupported rescue signaling model_type: {actual_model_type}")


def rescue_signaling_metadata(problem, *, source_config=None, episodes=None, seed=None, sample_lost=None, sample_agents=None):
    width, height = _require_grid_geometry(problem)
    metadata = {
        "num_nodes": int(width * height),
        "num_agents": int(len(problem.agent_start_nodes)),
        "num_lost_individuals": int(len(problem.lost_individual_nodes)),
        "target_knowledge": str(problem.target_knowledge),
        "grid_width": int(width),
        "grid_height": int(height),
        "source_strategy": "non_autonomous_rollout",
    }
    if source_config is not None:
        metadata["config"] = str(source_config)
    if episodes is not None:
        metadata["episodes"] = int(episodes)
    if seed is not None:
        metadata["seed"] = int(seed)
    if sample_lost is not None:
        metadata["sample_lost"] = bool(sample_lost)
    if sample_agents is not None:
        metadata["sample_agents"] = bool(sample_agents)
    return metadata


def _validate_model_geometry(model, problem, state):
    width, height = _require_grid_geometry(problem)
    if int(model.grid_width) != int(width) or int(model.grid_height) != int(height):
        raise ValueError(
            f"rescue signaling model grid {(model.grid_width, model.grid_height)} does not match "
            f"problem grid {(width, height)}"
        )
    if int(model.num_nodes) != int(width * height):
        raise ValueError("rescue signaling model num_nodes does not match problem")
    if int(model.num_agents) != len(state.agent_nodes):
        raise ValueError("rescue signaling model num_agents does not match state")
    if str(model.target_knowledge) != str(problem.target_knowledge):
        raise ValueError("rescue signaling model target_knowledge does not match problem")


def _model_metadata(model) -> dict:
    return {
        "model_type": model.model_type,
        "num_nodes": int(model.num_nodes),
        "num_agents": int(model.num_agents),
        "target_knowledge": str(model.target_knowledge),
        "grid_width": int(model.grid_width),
        "grid_height": int(model.grid_height),
        "action_names": list(ACTION_NAMES),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _valid_moves(problem, node: NodeId) -> tuple[NodeId, ...]:
    return tuple(sorted((int(node), *problem.adjacency[int(node)])))


def _node_mask(problem, nodes) -> np.ndarray:
    width, height = _require_grid_geometry(problem)
    mask = np.zeros(width * height, dtype=float)
    for node in nodes:
        node = int(node)
        if 0 <= node < mask.shape[0]:
            mask[node] = 1.0
    return mask


def _require_grid_geometry(problem) -> tuple[int, int]:
    if problem.grid_width is None or problem.grid_height is None:
        raise ValueError("rescue learned signaling requires grid_width and grid_height")
    return int(problem.grid_width), int(problem.grid_height)


def _axis_scale(size: int) -> float:
    max_coordinate = float(size - 1)
    if max_coordinate <= 0.0:
        return 1.0
    return max_coordinate


def _grid_coordinates(node: NodeId, width: int) -> tuple[int, int]:
    return int(node) % int(width), int(node) // int(width)


def _grid_node_id(x: int, y: int, width: int) -> NodeId:
    return int(y) * int(width) + int(x)
