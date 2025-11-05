import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import EXPERIMENT_LOGS_ROOT, MODEL_DUMP_ROOT
from mc2.training.data_sampling import draw_data_uniformly
from mc2.runners.model_setup_jax import setup_model
from mc2.model_interfaces.model_interface import load_model


def get_exp_ids(material_name: str | list[str] | None = None, model_type: str | list[str] | None = None):
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

    return relevant_exp_ids


def reconstruct_model_from_exp_id(exp_id):
    material_name = exp_id.split("_")[0]
    model_type = exp_id.split("_")[1]

    fresh_wrapped_model, _, params, (train_set, val_set, test_set) = setup_model(
        model_label=model_type,
        material_name=material_name,
        model_key=jax.random.PRNGKey(0),
    )

    model_path = MODEL_DUMP_ROOT / f"{exp_id}.eqx"
    model = load_model(model_path, type(fresh_wrapped_model.model))

    wrapped_model = eqx.tree_at(lambda t: t.model, fresh_wrapped_model, model)
    return wrapped_model


def load_gt_and_pred(exp_id, seed, freq_idx):
    gt = EXPERIMENT_LOGS_ROOT / f"{exp_id}/seed_{seed}_seq_{freq_idx}_gt.parquet"
    pred = EXPERIMENT_LOGS_ROOT / f"{exp_id}/seed_{seed}_seq_{freq_idx}_preds.parquet"
    gt = pd.read_parquet(gt).to_numpy()
    pred = pd.read_parquet(pred).to_numpy()
    return gt, pred


def plot_loss_trends(exp_id, seed):
    loss_trend = EXPERIMENT_LOGS_ROOT / f"{exp_id}/seed_{seed}_loss_trends.parquet"
    loss_train_val = pd.read_parquet(loss_trend).to_numpy()

    epochs = np.arange(1, len(loss_train_val) + 1)
    train_loss = loss_train_val[:, 0]
    val_loss = loss_train_val[:, 1]

    # Fig with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Training Loss Plot
    axes[0].plot(epochs, train_loss, color="tab:blue", marker="o", label="Training Loss (normalized)")
    axes[0].set_ylabel("Training Loss", color="tab:blue")
    axes[0].tick_params(axis="y", labelcolor="tab:blue")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale("log")

    # Validation Loss Plot
    axes[1].plot(epochs, val_loss, color="tab:red", marker="s", label="Validation (non-normalized)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation", color="tab:red")
    axes[1].tick_params(axis="y", labelcolor="tab:red")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Validation Loss")
    axes[1].set_yscale("log")
    # Layout und Anzeige
    fig.suptitle("Training vs Validation Loss", fontsize=14)
    fig.tight_layout()
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
    print(f"MAE {mae_M.mean():.1f} A/m | MSE {mse_M.mean():.1f} (A/m)²")
    for tst_idx in range(axes.shape[0]):
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


def plot_model_frequency_sweep(wrapped_model, test_set, loader_key, past_size, figsize=(30, 8)):
    # gather data

    H_list, B_list, T_list = [], [], []

    for freq_idx, frequency in enumerate(test_set.frequencies):
        test_set_at_frequency = test_set.at_frequency(frequency)
        H, B, T, _, loader_key = draw_data_uniformly(test_set_at_frequency, 1000, 1, loader_key)

        H_list.append(H[None, ...])
        B_list.append(B[None, ...])
        T_list.append(T[None, ...])

    H = jnp.concatenate(H_list, axis=0)
    B = jnp.concatenate(B_list, axis=0)
    T = jnp.concatenate(T_list, axis=0)

    H_past = H[:, :past_size]
    B_past = B[:, :past_size]

    B_future = B[:, past_size:]
    H_future = H[:, past_size:]

    H_pred = wrapped_model.call_with_warmup(B_past, H_past, B_future, T)

    # plot
    fig, axs = plt.subplots(3, 7, figsize=figsize)
    for freq_idx in range(len(test_set.frequencies)):
        axs[2, freq_idx].plot(B_future[freq_idx])
        axs[0, freq_idx].plot(H_future[freq_idx])
        axs[0, freq_idx].plot(H_pred[freq_idx])

        axs[1, freq_idx].plot(B_future[freq_idx], H_future[freq_idx])
        axs[1, freq_idx].plot(B_future[freq_idx], H_pred[freq_idx])

        axs[0, freq_idx].grid(True, alpha=0.3)
        axs[1, freq_idx].grid(True, alpha=0.3)

        axs[0, freq_idx].set_ylabel("H")
        axs[1, freq_idx].set_ylabel("H")
        axs[0, freq_idx].set_xlabel("k")
        axs[1, freq_idx].set_xlabel("B")

        axs[0, freq_idx].grid(True, alpha=0.3)
        axs[1, freq_idx].grid(True, alpha=0.3)

    fig.tight_layout(pad=-0.2)
    return fig, axs
