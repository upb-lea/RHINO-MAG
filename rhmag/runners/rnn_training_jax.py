"""Main training script.

Either use the function `train_model_jax` in another python script / jupyter-notebook or run the script via commandline:

```
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"  # disable preallocation of memory

from mc2.runners.rnn_training_jax import train_model_jax


train_model_jax(
    material="A",
    model_types=["GRU4", "JA"],
    seeds=[1, 2, 3],
    exp_name="demonstration",
    loss_type="adapted_RMS",
    gpu_id=0,
    epochs=10,
    batch_size=512,
    tbptt_size=156,
    past_size=28,
    time_shift=0,
    noise_on_data=0.0,
    tbptt_size_start=None,
    dyn_avg_kernel_size=11,
    disable_f64=True,
    disable_features="reduce",
    transform_H=False,
    use_all_data=False,
)
```

or, e.g.:

```
python mc2/runners/rnn_training_jax.py --material "A" --model_types "GRU4" "JA" --seeds 1 2 3 --exp_name "demonstration"

"""

import argparse
import pathlib
from copy import deepcopy
import logging as log
import os

# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)

import jax.numpy as jnp
import numpy as np
import json
import optax
from uuid import uuid4

from rhmag.data_management import AVAILABLE_MATERIALS, MODEL_DUMP_ROOT, EXPERIMENT_LOGS_ROOT, book_keeping
from rhmag.training.jax_routine import train_model
from rhmag.model_setup import setup_experiment, SUPPORTED_MODELS, SUPPORTED_LOSSES
from rhmag.metrics import evaluate_model_on_test_set
from rhmag.model_interfaces.model_interface import save_model
from rhmag.utils.model_evaluation import store_model_to_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recursive NNs")
    # parser.add_argument("-t", "--tag", default=None, required=False, help="an identifier/tag/comment for the trials")
    parser.add_argument(
        "--material",
        required=True,
        help=f"Material label to train on. One of {AVAILABLE_MATERIALS}",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        required=True,
        help=f"Model types to train with. One or multiple of {SUPPORTED_MODELS}",
    )
    parser.add_argument(
        "--loss_type",
        default="adapted_RMS",
        required=False,
        help=f"Loss type to train with. One of {SUPPORTED_LOSSES}",
    )
    # parser.add_argument(
    #     "-f",
    #     "--features",
    #     nargs="+",
    #     default=SUPPORTED_FEATURES,
    #     required=False,
    #     help=f"The features to use. Multiple (at least one) of {SUPPORTED_FEATURES}",
    # )
    parser.add_argument(
        "--exp_name",
        default=None,
        required=False,
        help=f"Experiment name to appear in exp_id.",
    )
    parser.add_argument(
        "--gpu_id",
        default=-1,
        type=int,
        required=False,
        help="id of the gpu to use for the experiments. '-1' for using the CPU.",
    )
    # TODO: Enable epochs over sampling
    parser.add_argument("-e", "--epochs", default=100, required=False, type=int, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", default=256, required=False, type=int, help="Batch size for training")
    parser.add_argument(
        "-t", "--tbptt_size", default=1024, required=False, type=int, help="Truncated backpropagation through time size"
    )
    parser.add_argument(
        "-p",
        "--past_size",
        default=10,
        required=False,
        type=int,
        help="Number of steps to use in the past trajectory (warmup steps).",
    )
    parser.add_argument(
        "--time_shift",
        default=0,
        required=False,
        type=int,
        help="Time shift for the B trajectory in featurize",
    )
    parser.add_argument(
        "--noise_on_data",
        default=0.0,
        required=False,
        type=float,
        help="Standard deviation of gaussian noise applied to the B trajectory.",
    )
    parser.add_argument(
        "-ts",
        "--tbptt_size_start",
        default=None,
        nargs=2,
        type=int,
        help="Starting tbptt size and number of epochs/steps to use it for, before switching to tbptt_size. Format: size n_epochs",
    )
    parser.add_argument(
        "--seeds",
        default=[0],
        nargs="+",
        type=int,
        required=False,
        help="One or more seeds to run the experiments with. Default is [0].",
    )
    parser.add_argument("--disable_f64", action="store_true", default=False)
    parser.add_argument("--disable_features", action="store_true", default=False)
    parser.add_argument("--transform_H", action="store_true", default=False)
    # parser.add_argument("-d", "--debug", action="store_true", default=False, help="Run in debug mode with reduced data")
    args = parser.parse_args()
    return args


def run_experiment_for_seed(
    seed: int,
    base_id: str,
    material: str,
    model_type: str,
    loss_type: str,
    gpu_id: int,
    epochs: int,
    batch_size: int,
    tbptt_size: int,
    past_size: int,
    time_shift: int,
    noise_on_data: float,
    tbptt_size_start: tuple[int, int] | None,
    disable_features: bool | str,
    dyn_avg_kernel_size: int,
    transform_H: bool,
    use_all_data: bool,
):

    if gpu_id != -1:
        gpus = jax.devices()
        jax.config.update("jax_default_device", gpus[gpu_id])
    elif gpu_id == -1:
        jax.config.update("jax_platform_name", "cpu")

    # setup
    # seed = 0
    key = jax.random.PRNGKey(seed)
    key, training_key, model_key = jax.random.split(key, 3)

    assert material in AVAILABLE_MATERIALS, f"Material {material} is not available. Choose on of {AVAILABLE_MATERIALS}."

    wrapped_model, optimizer, loss_function, params, data_tuple = setup_experiment(
        model_type,
        material,
        loss_type,
        model_key,
        n_epochs=epochs,
        tbptt_size=tbptt_size,
        batch_size=batch_size,
        past_size=past_size,
        time_shift=time_shift,
        tbptt_size_start=tbptt_size_start,
        disable_features=disable_features,
        transform_H=transform_H,
        noise_on_data=noise_on_data,
        dyn_avg_kernel_size=dyn_avg_kernel_size,
        use_all_data=use_all_data,
    )

    exp_id = f"{base_id}_seed{seed}"
    log.info(f"Training starting. Experiment ID is '{exp_id}'.")

    # run training
    logs, model = train_model(
        model=wrapped_model,
        loss_function=loss_function,
        optimizer=optimizer,
        material_name=material,
        data_tuple=data_tuple,
        key=training_key,
        seed=seed,
        **params["training_params"],
    )
    train_set, val_set, test_set = data_tuple
    log.info("Training done. Proceeding with evaluation..")

    if test_set is None:
        eval_metrics = {}
    else:
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
    save_model_params = deepcopy(params)
    store_model_to_file(
        filename=MODEL_DUMP_ROOT / f"{exp_id}.eqx",
        wrapped_model=model,
        params=save_model_params,
    )

    log.info(
        f"Experiment with id '{exp_id}' finished successfully. "
        + "Parameters, logs, evaluation metrics, and the model "
        + "have been stored successfully."
    )


def train_model_jax(
    material_name: str,
    model_types: list[str],
    seeds: list[int],
    exp_name: str | None = None,
    loss_type: str = "adapted_RMS",
    gpu_id: int = -1,
    epochs: int = 100,
    batch_size: int = 256,
    tbptt_size: int = 1024,
    past_size: int = 10,
    time_shift: int = 0,
    noise_on_data: float = 0.0,
    dyn_avg_kernel_size: int = 11,
    tbptt_size_start: tuple[int, int] | None = None,
    disable_f64: bool = False,
    disable_features: bool | str = False,
    transform_H: bool = False,
    use_all_data: bool = False,
):
    """Train a model based on the specified parameterization.

    Args:
        material_name (str): The name of the material. See `mc2.datamanagement.AVAILABLE_MATERIALS`.
        model_types (list[str]): List of identifiers of the model types to be trained.
            See `mc2.runners.model_setup.SUPPORTED_MODELS` for all available models. The trainings for
            each specified model type are done sequentially, i.e., each model_type is trained for all
            seeds specified individually.
        seeds (list[int]): List of seeds for which a model should be trained.
        exp_name (str): Additional identifier for the experiment (There is a randomly generated identifer for each
            experiment, but this can still be useful for sorting/finding experiments after training)
        loss_type (str): Identifier for the loss to use in training. See `mc2.runners.model_setup.SUPPORTED_LOSSES`.
        gpu_id (int): The index of the CUDA device / GPU to use. Specifying "-1" uses the CPU instead.
        epochs (int): The number of epochs to train for.
        batch_size (int): Number of parallel sequences to process per parameter update (i.e., per gradient calculation).
        tbptt_size (int): Length of the sequences to process per parameter update (i.e., per gradient calculation).
        past_size (int): Number of warmup steps before the prediction starts.
        time_shift (int): When specifying a value `!=0`, a feature is added where the `B` trajectory is shifted by that
            number of time steps
        noise_on_data (float): The standard deviation of noise added to the `B` trajectories.
        dyn_avg_kernel_size (int): The kernel size of the dynamic average feature.
        tbptt_size_start (tuple[int, int] | None): Optional training with specified sequence length (first element of tuple)
            and the number of epochs to train with this sequence length (second element of tuple). This might be helpful when
            the model diverges on the full sequence length and needs to start training with shorter sequences to stabilize first.
        disable_f64 (bool): Whether f64 should be disabled. When `True` float32 is used for all jax.Arrays instead
        disable_features (bool): One of (True, False, "reduce"), True uses no features, False uses all default features,
            "reduce" uses the dB/dt and d^2 B / dt^2 as features.
        transform_H (bool): Whether a tanh transform for H should be utilized.
        use_all_data (bool): Whether all data should be used for training or if instead a train, eval, test split should be performed.

    Returns:
        None
    """
    jax.config.update("jax_enable_x64", not disable_f64)

    if not seeds:
        log.warning("No seeds provided. Using default seed 0.")
        seeds_to_run = [0]
    else:
        seeds_to_run = seeds

    log.info(
        f"Starting experiments for {len(model_types)} model type(s) and {len(seeds_to_run)} seeds: {model_types}, {seeds_to_run}"
    )
    for model_type in model_types:
        log.info(f"--- Starting experiments for Model Type: {model_type} ---")
        if exp_name is None:
            base_id = f"{material_name}_{model_type}_{str(uuid4())[:8]}"
        else:
            base_id = f"{material_name}_{model_type}_{exp_name}_{str(uuid4())[:8]}"
        for seed in seeds_to_run:
            # try:
            run_experiment_for_seed(
                seed=seed,
                base_id=base_id,
                material=material_name,
                model_type=model_type,
                loss_type=loss_type,
                gpu_id=gpu_id,
                epochs=epochs,
                batch_size=batch_size,
                tbptt_size=tbptt_size,
                past_size=past_size,
                time_shift=time_shift,
                noise_on_data=noise_on_data,
                dyn_avg_kernel_size=dyn_avg_kernel_size,
                tbptt_size_start=tbptt_size_start,
                disable_features=disable_features,
                transform_H=transform_H,
                use_all_data=use_all_data,
            )
            # except Exception as e:
            #     log.error(f"Experiment for model {model_type} and seed {seed} failed with error: {e}")
            jax.clear_caches()

    log.info("All scheduled experiments completed.")


if __name__ == "__main__":
    args = parse_args()
    train_model_jax(
        material_name=args.material,
        model_types=args.model_types,
        seeds=args.seeds,
        exp_name=args.exp_name,
        loss_type=args.loss_type,
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        tbptt_size=args.tbptt_size,
        past_size=args.past_size,
        time_shift=args.time_shift,
        noise_on_data=args.noise_on_data,
        tbptt_size_start=args.tbptt_size_start,
        disable_f64=args.disable_f64,
        disable_features=args.disable_features,
        transform_H=args.transform_H,
    )
