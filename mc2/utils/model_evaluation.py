import json
from copy import deepcopy
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import DATA_ROOT, EXPERIMENT_LOGS_ROOT, MODEL_DUMP_ROOT, MaterialSet, Normalizer
from mc2.training.data_sampling import draw_data_uniformly
from mc2.model_setup import setup_model, setup_experiment, setup_featurize
from mc2.model_interfaces.model_interface import load_model, ModelInterface, save_model
from mc2.metrics import sre, nere

from functools import partial


def get_exp_ids(
    material_name: str | list[str] | None = None,
    model_type: str | list[str] | None = None,
    exp_name: str | None = None,
    legacy_mode: bool = False,
):
    if legacy_mode:
        model_paths = list((DATA_ROOT / "legacy_model_dump").glob("*.eqx"))
    else:
        model_paths = list(MODEL_DUMP_ROOT.glob("*.eqx"))

    exp_ids = [model_path.stem for model_path in model_paths]

    if material_name is None:
        relevant_exp_ids = exp_ids
    elif isinstance(material_name, str):
        relevant_exp_ids = [exp_id for exp_id in exp_ids if exp_id.split("_")[0] == material_name]
    elif isinstance(material_name, list):
        relevant_exp_ids = [exp_id for exp_id in exp_ids if exp_id.split("_")[0] in material_name]
    else:
        raise ValueError("'material_name' needs to be a string, list of strings, or None.")

    exp_ids = relevant_exp_ids

    if model_type is None:
        relevant_exp_ids = exp_ids
    elif isinstance(model_type, str):
        relevant_exp_ids = []
        for exp_id in exp_ids:
            if len(exp_id.split("_")) < 2:
                continue
            elif exp_id.split("_")[1] == model_type:
                relevant_exp_ids.append(exp_id)
    elif isinstance(model_type, list):
        relevant_exp_ids = []
        for exp_id in exp_ids:
            if len(exp_id.split("_")) < 2:
                continue
            elif exp_id.split("_")[1] in model_type:
                relevant_exp_ids.append(exp_id)
    else:
        raise ValueError("'model_type' needs to be a string, list of strings, or None.")

    exp_ids = relevant_exp_ids

    if exp_name is None:
        relevant_exp_ids = exp_ids
    elif isinstance(exp_name, str):
        relevant_exp_ids = []
        for exp_id in exp_ids:
            if len(exp_id.split("_")) < 3:
                continue
            elif exp_id.split("_")[2] == exp_name:
                relevant_exp_ids.append(exp_id)
    else:
        raise ValueError("'exp_name' needs to be a string or None.")

    return relevant_exp_ids


from functools import partial
from mc2.model_interfaces.model_interface import filter_spec


def load_parameterization(exp_id):
    experiment_path = EXPERIMENT_LOGS_ROOT / "jax_experiments"
    with open(experiment_path / f"{exp_id}.json", "r") as f:
        params = json.load(f)["params"]
    return params


def reconstruct_model_from_file(filename: pathlib.Path) -> ModelInterface:
    """Reconstruct a model from its file stored on disk.

    Args:
        filename (pathlib.Path): The path of file to load. If the suffix is missing, `.eqx` is
            added by default. One can  give the full path of the model file or, if the
            file cannot be found at the specified path, the function looks for the model in
            the `mc2.data_management.MODEL_DUMP_ROOT`, which is the default folder models are
            stored in.

            The commands:
            ```
            from mc2.data_management import MODEL_DUMP_ROOT
            reconstruct_model_from_file('A_GRU8_reduced-features-f32_2a1473b6_seed12')
            reconstruct_model_from_file('A_GRU8_reduced-features-f32_2a1473b6_seed12.eqx')
            reconstruct_model_from_file(MODEL_DUMP_ROOT / 'A_GRU8_reduced-features-f32_2a1473b6_seed12.eqx')
            ```
            All load the same model at `data/models` unless a file with the same name is present in the cwd
            in which case the first two calls load that model from the cwd and the last one loads the model
            from `data/models`. Generally, the intended way is to call using the first version.

    Returns:
        The ModelInterface object (i.e., the model wrapped into the corresponing interface)
    """

    filename = pathlib.Path(filename)

    # append the '.eqx' suffix if it is missing
    if filename.suffix == "":
        filename = filename.with_name(f"{filename.name}.eqx")

    # check for the filename in the 'MODEL_DUMP_ROOT' if it is not already an existing file
    if filename.is_file():
        filename = filename
    else:
        search_path = MODEL_DUMP_ROOT / filename
        if search_path.is_file():
            print(f"Found model file at '{search_path}'. Loading model..")
            filename = search_path
        else:
            raise ValueError(f"No model could be found for the specified filepath: '{filename}'")

    with open(filename, "rb") as f:
        params = json.loads(f.readline().decode())

        normalizer = Normalizer.from_dict(params["normalizer_dict"])
        featurize = setup_featurize(
            disable_features=params["training_params"]["disable_features"],
            dyn_avg_kernel_size=params["training_params"]["dyn_avg_kernel_size"],
            time_shift=params["training_params"]["time_shift"],
        )
        fresh_wrapped_model, _ = setup_model(
            model_label=params["model_type"],
            model_key=jax.random.PRNGKey(0),
            normalizer=normalizer,
            featurize=featurize,
            time_shift=params["training_params"]["time_shift"],
        )

        loading_params = deepcopy(params["model_params"])
        loading_params["key"] = jnp.array(loading_params["key"], dtype=jnp.uint32)
        model = type(fresh_wrapped_model.model)(**loading_params)
        model = eqx.tree_deserialise_leaves(f, model, partial(filter_spec, f64_enabled=jax.config.x64_enabled))
        wrapped_model = eqx.tree_at(lambda t: t.model, fresh_wrapped_model, model)

    return wrapped_model


def store_model_to_file(filename: pathlib.Path, wrapped_model: ModelInterface, params: dict):
    """Store the given model at the specified filepath.

    NOTE: This needs the parameter dict from which the model was created!

    Args:
        filename (pathlib.Path): Filepath at which to store the model
        wrapped_model (ModelInterface): Model object to save on disk
        params (dict): Parameter dict that was used to create the model

    Returns:
        None

    """
    assert "model_type" in params.keys()
    assert "material_name" in params.keys()

    params["normalizer_dict"] = wrapped_model.normalizer.to_dict(params["training_params"]["transform_H"])
    save_model(filename, params, wrapped_model)


def reconstruct_model_from_exp_id(exp_id, **kwargs):
    """NOTE: Deprecated in favor of 'reconstruct_model_from_file' and 'store_model_to_file' for
    saving and loading models.
    """

    material_name = exp_id.split("_")[0]
    model_type = exp_id.split("_")[1]

    # check if params are available:
    experiment_path = EXPERIMENT_LOGS_ROOT / "jax_experiments"
    try:
        with open(experiment_path / f"{exp_id}.json", "r") as f:
            params = json.load(f)["params"]
        print(f"Parameters for the model setup were found at '{experiment_path / exp_id}' and are utilized.")
        fresh_wrapped_model, _, _, data_tuple = setup_experiment(
            model_label=model_type,
            material_name=material_name,
            model_key=jax.random.PRNGKey(0),
            loss_type=params["loss_type"] if "loss_type" in params.keys() else "adapted_RMS",
            **params["training_params"],
            **kwargs,
        )
    except FileNotFoundError:
        print(
            f"No parameters could be found under '{experiment_path}' for exp_id: '{exp_id}', "
            + "continues with default setup for the given model type specified in 'setup_model'."
        )
        fresh_wrapped_model, _, _, data_tuple = setup_experiment(
            model_label=model_type,
            material_name=material_name,
            model_key=jax.random.PRNGKey(0),
            loss_type=params["loss_type"] if "loss_type" in params.keys() else "adapted_RMS",
            **kwargs,
        )

    model_path = DATA_ROOT / "legacy_model_dump" / f"{exp_id}.eqx"
    try:
        model = load_model(model_path, type(fresh_wrapped_model.model))
    except TypeError:
        with open(model_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model = type(fresh_wrapped_model.model)(
                key=jax.random.PRNGKey(0), normalizer=fresh_wrapped_model.normalizer, **hyperparams
            )
            model = eqx.tree_deserialise_leaves(f, model, partial(filter_spec, f64_enabled=jax.config.x64_enabled))

    wrapped_model = eqx.tree_at(lambda t: t.model, fresh_wrapped_model, model)
    return wrapped_model, data_tuple


def load_gt_and_pred(exp_id, seed, freq_idx):
    gt = EXPERIMENT_LOGS_ROOT / f"{exp_id}/seed_{seed}_seq_{freq_idx}_gt.parquet"
    pred = EXPERIMENT_LOGS_ROOT / f"{exp_id}/seed_{seed}_seq_{freq_idx}_preds.parquet"
    gt = pd.read_parquet(gt).to_numpy()
    pred = pd.read_parquet(pred).to_numpy()
    return gt, pred


def plot_loss_trends(exp_id, seed, plot_together: bool = False, figsize=(8, 8)):
    loss_trend = EXPERIMENT_LOGS_ROOT / f"{exp_id}/seed_{seed}_loss_trends.parquet"
    loss_train_val = pd.read_parquet(loss_trend).to_numpy()

    epochs = np.arange(1, len(loss_train_val) + 1)
    train_loss = loss_train_val[:, 0]
    val_loss = loss_train_val[:, 1]

    # Fig with two subplots
    if plot_together:
        n_plots = 1
    else:
        n_plots = 2

    fig, axes = plt.subplots(n_plots, 1, squeeze=False, figsize=figsize, sharex=True)

    # Training Loss Plot
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="tab:blue", marker="o", markersize=1, label="Training Loss (normalized)")
    ax.tick_params(axis="y")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{RMSE}}$")

    ax = axes[0, 0] if plot_together else axes[1, 0]
    # Validation Loss Plot
    ax.plot(epochs, val_loss, color="tab:red", marker="s", markersize=1, label="Validation Loss (normalized)")
    ax.set_xlabel("Epoch")
    ax.tick_params(axis="y")
    ax.legend()
    ax.set_yscale("log")

    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{RMSE}}$")

    # Layout und Anzeige
    fig.tight_layout()
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    return fig, axes


def load_and_plot_worst_predictions(exp_id, seed, freq_idx):
    gt, pred = load_gt_and_pred(exp_id, seed, freq_idx)
    fig, axes = plot_first_predictions(gt, pred)
    return fig, axes


def load_and_plot_first_prediction(exp_id, seed, freq_idx):
    gt, pred = load_gt_and_pred(exp_id, seed, freq_idx)
    fig, axes = plot_first_predictions(gt, pred)
    return fig, axes


def plot_worst_predictions(gt, pred, metric="MSE"):
    mae_M = np.mean(np.abs(gt - pred), axis=-1)
    mse_M = np.mean((gt - pred) ** 2, axis=-1)
    rmse_M = np.sqrt(np.mean((gt - pred) ** 2, axis=-1)) / np.sqrt(np.mean(gt**2, axis=-1))
    if metric == "RMSE":
        sorted_idx = np.argsort(rmse_M)[::-1]
    elif metric == "MAE":
        sorted_idx = np.argsort(mae_M)[::-1]
    else:
        sorted_idx = np.argsort(mse_M)[::-1]
    worst_idx = sorted_idx[:5]

    print(f"MAE {mae_M.mean():.1f} A/m | MSE {mse_M.mean():.1f} (A/m)² | RMSE {rmse_M.mean():.3f}")

    fig, axes = plt.subplots(5, 1, sharex=True, sharey="col", figsize=(10, 15))
    axes[0].set_title(f"Worst {metric}")
    for i, idx in enumerate(worst_idx):
        ax = axes[i]
        ax.plot(gt[idx], label="gt")
        ax.plot(pred[idx], label="pred", ls="dashed")
        ax.annotate(
            f"MAE {mae_M[idx]:.1f} A/m | " f"MSE {mse_M[idx]:.1f} (A/m)² | RMSE {rmse_M[idx]:.3f}",
            (0.3, 0.1),
            xycoords=ax.transAxes,
        )

    axes.flatten()[0].legend()
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    for ax in axes:
        ax.set_ylabel("H in A/m")
    for ax in [axes[-1]]:
        ax.set_xlabel("Sequence step")

    fig.tight_layout()
    return fig, axes


def plot_first_predictions(gt, pred):
    fig, axes = plt.subplots(5, 1, sharex=True, sharey="col", figsize=(10, 15))
    mae_M = np.mean(np.abs(gt - pred), axis=-1)
    mse_M = np.mean((gt - pred) ** 2, axis=-1)

    n_plots = min(axes.shape[0], gt.shape[0])

    print(f"MAE {mae_M.mean():.1f} A/m | MSE {mse_M.mean():.1f} (A/m)²")
    for tst_idx in range(n_plots):
        ax = axes[tst_idx]
        ax.plot(gt[tst_idx], label="gt")
        ax.plot(pred[tst_idx], label="pred", ls="dashed")
        ax.annotate(
            f"MAE {mae_M[tst_idx]:.1f} A/m | " f"MSE {mse_M[tst_idx]:.1f} (A/m)²", (0.3, 0.1), xycoords=ax.transAxes
        )

    axes.flatten()[0].legend()
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    for ax in axes:
        ax.set_ylabel("H in A/m")
    for ax in [axes[-1]]:
        ax.set_xlabel("Sequence step")
    fig.tight_layout()
    return fig, axes


def get_mixed_frequency_arrays(test_set: MaterialSet, sequence_length: int, batch_size: int, key: jax.random.PRNGKey):
    H_list, B_list, T_list = [], [], []

    for freq_set in test_set:
        H, B, T, _, key = draw_data_uniformly(freq_set, sequence_length, batch_size, key)

        H_list.append(H[None, ...])
        B_list.append(B[None, ...])
        T_list.append(T[None, ...])

    H = jnp.concatenate(H_list, axis=0)
    B = jnp.concatenate(B_list, axis=0)
    T = jnp.concatenate(T_list, axis=0)

    return H, B, T


def get_metrics_per_sequence(
    wrapped_model: ModelInterface,
    test_set: MaterialSet,
    scenarios: dict[str, tuple[int]],
    sequence_length: int,
    batch_size_per_frequency: int,
    loader_key: jax.random.PRNGKey,
) -> dict[str, jax.Array]:
    metrics_per_sequence = {}

    H, B, T = get_mixed_frequency_arrays(
        test_set,
        sequence_length=sequence_length,
        batch_size=batch_size_per_frequency,
        key=loader_key,
    )
    H = H.reshape((-1, H.shape[-1]))
    B = B.reshape((-1, B.shape[-1]))
    T = T.flatten()

    for scenario_label, scenario_values in scenarios.items():

        past_size = scenario_values[0]
        future_size = scenario_values[1]

        batch_size = H.shape[0]

        H_past = H[:, :past_size]
        B_past = B[:, :past_size]

        B_future = B[:, past_size:]
        H_future = H[:, past_size:]

        T = T

        H_pred = wrapped_model(B_past, H_past, B_future, T)

        ## check array sizes

        assert H.shape == (batch_size, past_size + future_size)
        assert B.shape == (batch_size, past_size + future_size)
        assert T.shape == (batch_size,)

        assert H_past.shape == (batch_size, past_size)
        assert H_future.shape == (batch_size, future_size)
        assert H_pred.shape == H_future.shape

        assert B_past.shape == (batch_size, past_size)
        assert B_future.shape == (batch_size, future_size)

        ## metrics

        wce_per_sequence = np.max(np.abs(H_pred - H_future), axis=1)
        mse_per_sequence = np.mean((H_pred - H_future) ** 2, axis=1)
        sre_per_sequence = eqx.filter_vmap(sre)(H_pred, H_future)

        dbdt_full = np.gradient(B, axis=1)
        dbdt = dbdt_full[:, past_size:]
        print(
            "Normalized Energy relative error cannot be properly computed as the core losses are unknown. Setting 'true_core_loss=1'."
        )
        nere_per_sequence = eqx.filter_vmap(nere)(H_pred, H_future, dbdt, 1.0)  # np.abs(true_core_loss))

        metrics_per_sequence[scenario_label] = {
            "mse": jnp.array(mse_per_sequence),
            "wce": jnp.array(wce_per_sequence),
            "sre": jnp.array(sre_per_sequence),
            "nere": jnp.array(nere_per_sequence),
        }

    return metrics_per_sequence


def evaluate_cross_validation(
    wrapped_model: ModelInterface,
    test_set: MaterialSet,
    scenarios: dict[str, tuple[int]],
    sequence_length: int,
    batch_size_per_frequency: int,
    loader_key: jax.random.PRNGKey,
) -> dict[str, np.ndarray]:

    metrics_per_sequence = get_metrics_per_sequence(
        wrapped_model,
        test_set,
        scenarios,
        sequence_length,
        batch_size_per_frequency,
        loader_key,
    )

    metrics_reduced = {}

    for scenario_label, metric_values in metrics_per_sequence.items():

        mse_per_sequence = metric_values["mse"]
        wce_per_sequence = metric_values["wce"]
        sre_per_sequence = metric_values["sre"]
        nere_per_sequence = metric_values["nere"]

        # reduce metrics
        mse = np.mean(mse_per_sequence)
        wce = np.max(wce_per_sequence)

        sre_avg = np.mean(sre_per_sequence)
        sre_95th = np.percentile(sre_per_sequence, 95)

        nere_avg = np.mean(nere_per_sequence)
        nere_95th = np.percentile(nere_per_sequence, 95)

        print(f"\tMSE : {mse:>7.2f} (A/m)²")
        print(f"\tWCE : {wce:>7.2f} A/m")

        metrics_reduced[scenario_label] = {
            "mse": np.round(mse, 4).item(),
            "wce": np.round(wce, 4).item(),
            "sre_avg": np.round(sre_avg, 4).item(),
            "sre_95th": np.round(sre_95th, 4).item(),
            "nere_avg": np.round(nere_avg, 4).item(),
            "nere_95th": np.round(nere_95th, 4).item(),
        }

    return metrics_reduced


def plot_model_frequency_sweep(wrapped_model, test_set, loader_key, past_size, figsize=(30, 8)):
    # gather data

    H, B, T = get_mixed_frequency_arrays(test_set, sequence_length=1000, batch_size=1, key=loader_key)

    H_past = H[:, :past_size]
    B_past = B[:, :past_size]

    B_future = B[:, past_size:]
    H_future = H[:, past_size:]

    H_pred = wrapped_model(B_past, H_past, B_future, T)

    # plot
    fig, axs = plt.subplots(3, 7, figsize=figsize)
    for freq_idx in range(len(test_set.frequencies)):
        axs[0, freq_idx].plot(B_future[freq_idx])
        axs[1, freq_idx].plot(H_future[freq_idx])
        axs[1, freq_idx].plot(H_pred[freq_idx])
        #axs[1, freq_idx].plot(H_future[freq_idx] - H_pred[freq_idx], color="tab:red", linestyle="--")

        axs[2, freq_idx].plot(H_future[freq_idx], B_future[freq_idx])
        axs[2, freq_idx].plot(H_pred[freq_idx], B_future[freq_idx])

        axs[0, freq_idx].grid(True, alpha=0.3)
        axs[1, freq_idx].grid(True, alpha=0.3)
        axs[2, freq_idx].grid(True, alpha=0.3)

        axs[0, freq_idx].set_ylabel("B")
        axs[0, freq_idx].set_xlabel("k")
        axs[1, freq_idx].set_ylabel("H")
        axs[1, freq_idx].set_xlabel("k")
        axs[2, freq_idx].set_ylabel("B")
        axs[2, freq_idx].set_xlabel("H")

        axs[0, freq_idx].set_title("frequency: " + str(int(test_set.frequencies[freq_idx] / 1e3)) + " kHz")

    fig.tight_layout()
    return fig, axs
