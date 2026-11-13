import csv
import json
import math

from src.main import parse_args, run_experiment


def test_all_algorithms_train_and_save_comparable_metrics(offline_datasets, tmp_path):
    initial_losses = []
    for algorithm, expected_updates in [("sgd", 8), ("minibatch_sgd", 2), ("sfn", 2)]:
        args = parse_args(
            [
                "--algorithm",
                algorithm,
                "--epochs",
                "1",
                "--image-size",
                "2",
                "--hidden-units",
                "2",
                "--train-limit",
                "8",
                "--eval-limit",
                "3",
                "--batch-size",
                "4",
                "--output-dir",
                str(tmp_path),
            ]
        )
        run_dir = run_experiment(args)
        config = json.loads((run_dir / "config.json").read_text())
        with (run_dir / "metrics.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
        assert len(metrics) == 2
        assert int(metrics[0]["updates"]) == 0
        assert int(metrics[1]["updates"]) == expected_updates
        assert int(metrics[1]["examples_seen"]) == 8
        assert all(math.isfinite(float(value)) for row in metrics for value in row.values())
        assert 0 <= test_metrics["accuracy"] <= 1
        assert math.isfinite(test_metrics["loss"])
        assert config["effective_batch_size"] == (1 if algorithm == "sgd" else 4)
        assert config["split_sizes"] == {"train": 8, "validation": 3, "test": 3}
        initial_losses.append(float(metrics[0]["train_loss"]))
    assert initial_losses[0] == initial_losses[1] == initial_losses[2]
