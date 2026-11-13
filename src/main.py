"""Run a reproducible training experiment; notebooks consume its saved metrics."""

import argparse
import csv
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from descent_algorithms import SFN, SGD, MiniBatchSGD
from src.models import build_mlp
from utils.load import PROJECT_ROOT, load_data


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    parser.add_argument(
        "--algorithm", choices=("sgd", "minibatch_sgd", "sfn"), default="minibatch_sgd"
    )
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=64, help="SGD always uses 1")
    parser.add_argument("--lr", type=positive_float, default=0.01)
    parser.add_argument("--damping", type=positive_float, default=0.1, help="SFN only")
    parser.add_argument("--hidden-units", type=positive_int, default=5)
    parser.add_argument(
        "--image-size", type=positive_int, default=10, help="resized square edge length"
    )
    parser.add_argument("--train-limit", type=positive_int, help="optional training subset size")
    parser.add_argument(
        "--eval-limit", type=positive_int, help="optional limit for validation and test"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--download", action="store_true", help="download missing dataset files")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.algorithm == "minibatch_sgd" and args.batch_size < 2:
        parser.error("minibatch_sgd requires --batch-size >= 2; use --algorithm sgd for size 1")
    if args.device == "mps" and args.dtype == "float64":
        parser.error("MPS does not support float64; select --dtype float32 or --device cpu")
    if args.algorithm == "sfn" and args.device != "cpu":
        parser.error("The exact SFN reference runs on CPU; use --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment")
    return args


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype
) -> dict:
    """Compute sample-weighted mean cross entropy and accuracy at current weights."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device=device, dtype=dtype), targets.to(device)
        logits = model(inputs)
        total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += targets.numel()
    return {"loss": total_loss / count, "accuracy": correct / count}


def run_experiment(args: argparse.Namespace) -> Path:
    """Train, save CSV/JSON artifacts, and return the unique run directory."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    batch_size = 1 if args.algorithm == "sgd" else args.batch_size
    loaders = load_data(
        args.dataset,
        batch_size=batch_size,
        data_dir=args.data_dir,
        download=args.download,
        image_size=args.image_size,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model = build_mlp(loaders.input_shape, args.hidden_units).to(device=device, dtype=dtype)
    if args.algorithm == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.algorithm == "minibatch_sgd":
        optimizer = MiniBatchSGD(model.parameters(), lr=args.lr, batch_size=batch_size)
    else:
        optimizer = SFN(model.parameters(), lr=args.lr, damping=args.damping, batch_size=batch_size)

    run_name = (
        f"{args.dataset}_{args.algorithm}_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "effective_batch_size": batch_size,
        "input_shape": loaders.input_shape,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_sizes": {
            "train": len(loaders.train.dataset),
            "validation": len(loaders.validation.dataset),
            "test": len(loaders.test.dataset),
        },
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    train_evaluation = DataLoader(
        loaders.train.dataset,
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    fields = [
        "epoch",
        "updates",
        "examples_seen",
        "train_seconds",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]
    updates, examples_seen, train_seconds = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    print(
        f"{args.dataset} | {args.algorithm} | {config['parameter_count']:,} parameters | {device}"
    )

    with (run_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        # Epoch zero records the common initialization before any optimizer updates.
        for epoch in range(args.epochs + 1):
            if epoch > 0:
                model.train()
                start = perf_counter()
                for inputs, targets in loaders.train:
                    inputs, targets = inputs.to(device=device, dtype=dtype), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if args.algorithm == "sfn":
                        optimizer.step(lambda x=inputs, y=targets: criterion(model(x), y))
                    else:
                        loss = criterion(model(inputs), targets)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss; try a smaller lr.")
                        loss.backward()
                        optimizer.step()
                    updates += 1
                    examples_seen += targets.numel()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                train_seconds += perf_counter() - start
            train_metrics = evaluate(model, train_evaluation, device, dtype)
            validation_metrics = evaluate(model, loaders.validation, device, dtype)
            if not all(
                math.isfinite(metric)
                for metric in (*train_metrics.values(), *validation_metrics.values())
            ):
                raise FloatingPointError("Non-finite evaluation metrics; try a smaller lr.")
            writer.writerow(
                {
                    "epoch": epoch,
                    "updates": updates,
                    "examples_seen": examples_seen,
                    "train_seconds": train_seconds,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                }
            )
            stream.flush()
            print(
                f"epoch {epoch:3d} | train loss {train_metrics['loss']:.4f} | "
                f"validation accuracy {validation_metrics['accuracy']:.1%}"
            )
    # Use validation for tuning; evaluate the official test split only after training.
    test_metrics = evaluate(model, loaders.test, device, dtype)
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    print(f"Saved metrics to {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
