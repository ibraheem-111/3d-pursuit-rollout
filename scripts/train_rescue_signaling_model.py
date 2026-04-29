from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rescue.signaling import (
    KernelRescueSignalingModel,
    MLPRescueSignalingModel,
    load_rescue_signaling_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a rescue learned-signaling model.")
    parser.add_argument("--config", type=str, default=None, help="Optional rescue config for signaling defaults.")
    parser.add_argument("--dataset", type=str, required=True, help="Input rescue signaling dataset .npz.")
    parser.add_argument("--output", type=str, default=None, help="Output rescue signaling model .npz.")
    parser.add_argument("--model-type", choices=["kernel_knn", "mlp"], default=None)
    parser.add_argument("--k", type=int, default=None, help="k for kernel kNN.")
    parser.add_argument("--sigma", type=float, default=None, help="Exponential kernel bandwidth.")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="*",
        default=None,
        help="MLP hidden layer sizes. Pass no values for a linear softmax model.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="MLP learning rate.")
    parser.add_argument("--epochs", type=int, default=None, help="MLP training epochs.")
    parser.add_argument("--seed", type=int, default=None, help="MLP initialization seed.")
    return parser.parse_args()


def load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def signaling_config(config):
    rescue_config = config.get("rescue", config) if config else {}
    return rescue_config.get("signaling", {})


def train_model_from_dataset(
    dataset,
    *,
    model_type: str,
    k: int = 25,
    sigma: float = 5.0,
    hidden_layers=(32,),
    learning_rate: float = 0.05,
    epochs: int = 500,
    seed: int = 0,
):
    if model_type == "kernel_knn":
        return KernelRescueSignalingModel.from_dataset(dataset, k=k, sigma=sigma)
    if model_type == "mlp":
        return MLPRescueSignalingModel.train(
            dataset,
            hidden_layers=tuple(hidden_layers),
            learning_rate=learning_rate,
            epochs=epochs,
            seed=seed,
        )
    raise ValueError(f"unsupported rescue signaling model_type: {model_type}")


def main():
    args = parse_args()
    config = load_config(args.config)
    defaults = signaling_config(config)
    output_path = args.output or defaults.get("model_path")
    if output_path is None:
        raise ValueError("--output is required unless rescue.signaling.model_path is set in --config")
    model_type = args.model_type or defaults.get("model_type", "kernel_knn")
    k = int(args.k if args.k is not None else defaults.get("k", 25))
    sigma = float(args.sigma if args.sigma is not None else defaults.get("sigma", 5.0))
    hidden_layers = args.hidden_layers
    if hidden_layers is None:
        hidden_layers = defaults.get("hidden_layers", [32])
    learning_rate = float(args.learning_rate if args.learning_rate is not None else defaults.get("learning_rate", 0.05))
    epochs = int(args.epochs if args.epochs is not None else defaults.get("epochs", 500))
    seed = int(args.seed if args.seed is not None else defaults.get("seed", 0))

    dataset = load_rescue_signaling_dataset(args.dataset)
    model = train_model_from_dataset(
        dataset,
        model_type=model_type,
        k=k,
        sigma=sigma,
        hidden_layers=tuple(hidden_layers),
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )
    model.save(output_path)

    metadata = dict(model.metadata or {})
    metadata.update(
        {
            "model_type": model.model_type,
            "dataset": str(args.dataset),
            "rows": int(dataset.y_train.shape[0]),
        }
    )
    metadata_path = Path(output_path).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"saved {model.model_type} rescue signaling model to {output_path}")


if __name__ == "__main__":
    main()
