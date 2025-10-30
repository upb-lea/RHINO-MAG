import argparse
import pathlib
from copy import deepcopy
import logging as log
import os

# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)

import jax.numpy as jnp
import numpy as np
import json
import optax
from uuid import uuid4

from mc2.data_management import AVAILABLE_MATERIALS, MODEL_DUMP_ROOT, EXPERIMENT_LOGS_ROOT, book_keeping
from mc2.training.jax_routine import train_model
from mc2.runners.model_setup_jax import setup_loss, setup_model, SUPPORTED_MODELS, SUPPORTED_LOSSES
from mc2.metrics import evaluate_model_on_test_set
from mc2.models.model_interface import save_model


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
        help=f"Model type to train with. One of {SUPPORTED_MODELS}",
    )
    parser.add_argument(
        "--loss_type",
        default="adapted_RMS",
        required=False,
        help=f"Loss type to train with. One of {SUPPORTED_LOSSES}",
    )
    parser.add_argument(
        "--gpu_id",
        default=-1,
        type=int,
        required=False,
        help="id of the gpu to use for the experiments. '-1' for not setting a GPU.",
    )
    # TODO: Enable epochs over sampling
    parser.add_argument("-e", "--epochs", default=100, required=False, type=int, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", default=256, required=False, type=int, help="Batch size for training")
    parser.add_argument(
        "-t", "--tbptt_size", default=1024, required=False, type=int, help="Truncated backpropagation through time size"
    )
    parser.add_argument(
        "-ts",
        "--tbptt_size_start",
        default=None,
        nargs=2,
        type=int,
        help="Starting tbptt size and number of epochs/steps to use it for, before switching to tbptt_size. Format: size n_epochs",
    )
    # parser.add_argument("-d", "--debug", action="store_true", default=False, help="Run in debug mode with reduced data")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.gpu_id != -1:
        gpus = jax.devices()
        jax.config.update("jax_default_device", gpus[args.gpu_id])
    # jax.config.update("jax_platform_name", "cpu")

    # setup
    seed = 0
    key = jax.random.PRNGKey(seed)
    key, training_key, model_key = jax.random.split(key, 3)

    assert (
        args.material in AVAILABLE_MATERIALS
    ), f"Material {args.material} is not available. Choose on of {AVAILABLE_MATERIALS}."

    # TODO: params as .yaml files?
    wrapped_model, optimizer, params, data_tuple = setup_model(
        args.model_type,
        args.material,
        model_key,
        n_epochs=args.epochs,
        tbptt_size=args.tbptt_size,
        batch_size=args.batch_size,
        tbptt_size_start=args.tbptt_size_start,
    )

    loss_function = setup_loss(args.loss_type)

    # run training
    logs, model = train_model(
        model=wrapped_model,
        loss_function=loss_function,
        optimizer=optimizer,
        material_name=args.material,
        data_tuple=data_tuple,
        key=training_key,
        seed=seed,
        **params["training_params"],
    )
    train_set, val_set, test_set = data_tuple
    log.info("Training done. Proceeding with evaluation..")

    exp_id = str(uuid4())[:16]
    eval_metrics = evaluate_model_on_test_set(model, test_set)

    log.info("Evaluation done. Proceeding with storing experiment data..")

    json_dump_d = dict(params=params, metrics=eval_metrics)
    book_keeping(logs, exp_id=exp_id)

    # create missing folders
    experiment_path = EXPERIMENT_LOGS_ROOT / "jax_experiments"
    experiment_path.mkdir(parents=True, exist_ok=True)
    MODEL_DUMP_ROOT.mkdir(parents=True, exist_ok=True)

    # TODO: automatically turn all jax arrays to lists...
    # store expeirment params + logs + eval_metrics
    with open(experiment_path / f"{exp_id}.json", "w") as f:
        json.dump(json_dump_d, f)

    # store model
    print(model)
    save_model_params = deepcopy(params["model_params"])
    del save_model_params["key"]
    save_model(MODEL_DUMP_ROOT / f"{exp_id}.eqx", save_model_params, model.model)

    log.info(
        f"Experiment with id '{exp_id}' finished successfully. "
        + "Parameters, logs, evaluation metrics, and the model "
        + "have been stored successfully."
    )


if __name__ == "__main__":
    main()
