import argparse
import jax
import optax

from mc2.data_management import AVAILABLE_MATERIALS
from mc2.training.jax_routine import train_model
from mc2.runners.model_setup_jax import get_GRU_setup

supported_model_types = ["GRU"]  # TODO: ["EulerNODE", "HiddenStateEulerNODE", "GRU"]


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
    else:
        raise ValueError(f"Unknown model type: {args.model_type}. Choose on of {supported_model_types}")

    # run training
    logs, model = train_model(
        model=wrapped_model,
        optimizer=optimizer,
        material_name=args.material,
        key=training_key,
        seed=seed,
        **params["training_params"],
    )

    # TODO: eval ?

    # TODO: store results
    #
    # store model
    # store logs


if __name__ == "__main__":
    main()
