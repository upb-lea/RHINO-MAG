import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mc2.data_management import EXPERIMENT_LOGS_ROOT


def load_gt_and_pred(exp_id, material_name, seed, freq_idx):
    gt = EXPERIMENT_LOGS_ROOT / f"{material_name}_{exp_id}/seed_{seed}_seq_{freq_idx}_gt.parquet"
    pred = EXPERIMENT_LOGS_ROOT / f"{material_name}_{exp_id}/seed_{seed}_seq_{freq_idx}_preds.parquet"
    gt = pd.read_parquet(gt).to_numpy()
    pred = pd.read_parquet(pred).to_numpy()
    return gt, pred


def plot_loss_trends(exp_id, material_name, seed):
    loss_trend = EXPERIMENT_LOGS_ROOT / f"{material_name}_{exp_id}/seed_{seed}_loss_trends.parquet"
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


def load_and_plot_worst_predictions(exp_id, material_name, seed, freq_idx):
    gt, pred = load_gt_and_pred(exp_id, material_name, seed, freq_idx)
    fig, axes = plot_first_predictions(gt, pred)
    return fig, axes


def load_and_plot_first_prediction(exp_id, material_name, seed, freq_idx):
    gt, pred = load_gt_and_pred(exp_id, material_name, seed, freq_idx)
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
