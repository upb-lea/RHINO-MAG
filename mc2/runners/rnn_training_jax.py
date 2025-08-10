import argparse
import pathlib
from copy import deepcopy
import jax
import json
import optax
from uuid import uuid4

from mc2.data_management import AVAILABLE_MATERIALS, MODEL_DUMP_ROOT
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
    # TODO: Enable epochs over sampling
    # parser.add_argument("-e", "--epochs", default=100, required=False, type=int, help="Number of epochs to train")
    # parser.add_argument("-d", "--debug", action="store_true", default=False, help="Run in debug mode with reduced data")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

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

    # TODO: eval, sample shorter trajectories as well?
    eval_metrics = {
        frequency.item(): evaluate_model(
            model,
            B_past=test_set.at_frequency(frequency).B[:, :1],
            H_past=test_set.at_frequency(frequency).H[:, :1],
            B_future=test_set.at_frequency(frequency).B[:, 1:],
            H_future=test_set.at_frequency(frequency).H[:, 1:],
            T=test_set.at_frequency(frequency).T[:],
            reduce_to_scalar=True,
        )
        for frequency in test_set.frequencies
    }
    print(eval_metrics)

    # TODO: store results
    # json.dumps() # params, logs, eval
    # store model
    save_model_params = deepcopy(params["model_params"])
    del save_model_params["key"]
    save_model(MODEL_DUMP_ROOT / pathlib.Path(str(uuid4())[:8] + ".eqx"), save_model_params, model.model)

    print("Experiment finished successfully.")


if __name__ == "__main__":
    main()
