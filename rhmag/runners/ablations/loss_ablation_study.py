"""Ablation study for the adapted loss function.

Runs trainings for all materials for the submission model style except that the
MSE loss instead of the adapted_RMS loss is used.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
from rhmag.runners.ablations.runner_structure import run_ablation_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recursive NNs")
    parser.add_argument(
        "--gpu_id",
        default=-1,
        type=int,
        required=False,
        help="id of the gpu to use for the experiments. '-1' for using the CPU.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    run_ablation_experiment(
        gpu_id=args.gpu_id,
        tag="ablation-loss-function",
        loss_function="MSE",
        init_type="default",
        feature_type="reduce",
    )
