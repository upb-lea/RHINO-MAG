import argparse
import pathlib
from copy import deepcopy
import logging as log

import jax
import jax.numpy as jnp
import numpy as np
import json
import optax
from uuid import uuid4

from mc2.data_management import AVAILABLE_MATERIALS, MODEL_DUMP_ROOT, EXPERIMENT_LOGS_ROOT
from mc2.training.jax_routine import train_model
from mc2.runners.model_setup_jax import get_GRU_setup, get_HNODE_setup
from mc2.metrics import evaluate_model
from mc2.models.model_interface import save_model

supported_model_types = ["GRU", "HNODE"]  # TODO: ["EulerNODE", "HNODE", "GRU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recursive NNs")
    # parser.add_argument("-t", "--tag", default=None, required=False, help="an identifier/tag/comment for the trials")
    parser.add_argument(
        "--material",
        default=None,
        required=True,
        help=f"Material label to train on. One of {AVAILABLE_MATERIALS}",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        required=True,
        help=f"Model type to train with. One of {supported_model_types}",
    )
    parser.add_argument(
        "--gpu_id",
        default=-1,
        type=int,
        required=False,
        help="id of the gpu to use for the experiments. '-1' for not setting a GPU.",
    )
    # TODO: Enable epochs over sampling
    # parser.add_argument("-e", "--epochs", default=100, required=False, type=int, help="Number of epochs to train")
    # parser.add_argument("-d", "--debug", action="store_true", default=False, help="Run in debug mode with reduced data")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.gpu_id != -1:
        gpus = jax.devices()
        jax.config.update("jax_default_device", gpus[args.gpu_id])

    # setup
    seed = 5
    key = jax.random.PRNGKey(seed)
    key, training_key, model_key = jax.random.split(key, 3)

    assert (
        args.material in AVAILABLE_MATERIALS
    ), f"Material {args.material} is not available. Choose on of {AVAILABLE_MATERIALS}."

    # TODO: params as .yaml files?
    if args.model_type == "GRU":
        wrapped_model, optimizer, params = get_GRU_setup(args.material, model_key)
    elif args.model_type == "HNODE":
        wrapped_model, optimizer, params = get_HNODE_setup(args.material, model_key)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}. Choose on of {supported_model_types}")

    # run training
    logs, model, (train_set, val_set, test_set) = train_model(
        model=wrapped_model,
        optimizer=optimizer,
        material_name=args.material,
        key=training_key,
        seed=seed,
        **params["training_params"],
    )

    log.info("Training done. Proceeding with evaluation..")

    exp_id = str(uuid4())[:16]

    # TODO: eval, sample shorter trajectories as well?
    eval_metrics = {}
    for frequency in test_set.frequencies:
        test_set_at_frequency = test_set.at_frequency(frequency)
        eval_metrics[frequency.item()] = {
            sequence_length.item(): evaluate_model(
                model,
                B_past=test_set_at_frequency.B[:, :1],
                H_past=test_set_at_frequency.H[:, :1],
                B_future=test_set_at_frequency.B[:, 1:sequence_length],
                H_future=test_set_at_frequency.H[:, 1:sequence_length],
                T=test_set_at_frequency.T[:],
                reduce_to_scalar=True,
            )
            for sequence_length in np.linspace(10, test_set_at_frequency.H.shape[-1], 10, dtype=int)
        }

    log.info("Evaluation done. Proceeding with storing experiment data..")

    # TODO: store results
    data = dict(params=params, logs=logs, metrics=eval_metrics)

    with open(EXPERIMENT_LOGS_ROOT / "jax_experiments" / pathlib.Path(exp_id + ".json"), "w") as f:
        json.dump(data, f)

    # store model
    save_model_params = deepcopy(params["model_params"])
    del save_model_params["key"]
    save_model(MODEL_DUMP_ROOT / pathlib.Path(exp_id + ".eqx"), save_model_params, model.model)

    log.info(
        f"Experiment with id '{exp_id}' finished successfully. "
        + "Parameters, logs, evaluation metrics, and the model "
        + "have been stored successfully."
    )


if __name__ == "__main__":
    main()
