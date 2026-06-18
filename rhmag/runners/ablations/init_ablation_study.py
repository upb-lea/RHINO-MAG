"""Ablation study for the initialization concept.

Runs trainings for all materials for the submission model style except that the
hidden state is initialized with zeros instead of inserting the true H value
at the first hidden state index.
"""

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
        tag="ablation-init-zeros",
        loss_function="adapted_RMS",
        init_type="ignore_warmup",
    )
