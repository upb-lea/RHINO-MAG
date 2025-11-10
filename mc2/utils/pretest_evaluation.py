import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.model_interfaces.model_interface import ModelInterface

from mc2.metrics import sre, nere

SCENARIO_LABELS = ["10% unknown", "50% unknown", "90% unknown"]

HOSTS_VALUES_DICT = {
    "3C90": {
        SCENARIO_LABELS[0]: {
            "mse": None,
            "wce": None,
            "sre_avg": 0.1305,
            "sre_95th": 0.347,
            "nere_avg": 0.007623,
            "nere_95th": 0.01928,
        },  # 90 % known
        SCENARIO_LABELS[1]: {
            "mse": None,
            "wce": None,
            "sre_avg": 0.1602,
            "sre_95th": 0.3443,
            "nere_avg": 0.0341,
            "nere_95th": 0.05603,
        },  # 50 % known
        SCENARIO_LABELS[2]: {
            "mse": None,
            "wce": None,
            "sre_avg": 0.1704,
            "sre_95th": 0.3476,
            "nere_avg": 0.0618,
            "nere_95th": 0.07476,
        },
    },  # 10 % known
    "N87": {
        SCENARIO_LABELS[0]: {
            "mse": None,
            "wce": None,
            "sre_avg": 0.1962,
            "sre_95th": 0.521,
            "nere_avg": 0.007805,
            "nere_95th": 0.0157,
        },  # 90 % known
        SCENARIO_LABELS[1]: {
            "mse": None,
            "wce": None,
            "sre_avg": 0.2767,
            "sre_95th": 0.8498,
            "nere_avg": 0.02577,
            "nere_95th": 0.05509,
        },  # 50 % known
        SCENARIO_LABELS[2]: {
            "mse": None,
            "wce": None,
            "sre_avg": 0.3028,
            "sre_95th": 0.9999,
            "nere_avg": 0.04828,
            "nere_95th": 0.0681,
        },  # 10 % known
    },
}


def create_multilevel_df(nested_dict):
    """Convert 3-level nested dict to DataFrame with outer keys as index and 2-level columns"""
    dfs_by_model = []
    for model_name, model_metrics in nested_dict.items():
        # Create tuples for MultiIndex columns (scenario, metric)
        tuples = [(scenario, metric) for scenario in model_metrics.keys() for metric in model_metrics[scenario].keys()]
        values = [
            model_metrics[scenario][metric]
            for scenario in model_metrics.keys()
            for metric in model_metrics[scenario].keys()
        ]

        # Create DataFrame for this model
        df_model = pd.DataFrame([values], columns=pd.MultiIndex.from_tuples(tuples), index=[model_name])
        dfs_by_model.append(df_model)

    # Concatenate and set column names
    df = pd.concat(dfs_by_model, axis=0)
    # df.columns.names = ['Scenario', 'Metric']
    return df


def get_metrics_per_sequence(
    model_all, B, T, H_init, H_true, loss, msks_scenarios_N_tup, scenario_labels, show_plots: bool = False
) -> dict[str, jax.Array]:
    metrics_per_sequence = {}

    for scenario_i, msk_N in enumerate(msks_scenarios_N_tup):
        print(f"  Scenario {scenario_i} - {scenario_labels[scenario_i]}: ")

        B_scenario = B[msk_N]
        T_scenario = T[msk_N]
        H_init_scenario = H_init[msk_N]
        H_true_scenario = H_true[msk_N]

        true_core_loss = np.squeeze(loss[msk_N])

        warm_up_len = np.sum(~np.isnan(H_init_scenario[0]))
        print(warm_up_len / B_scenario.shape[1])
        print(f"    -> warm_up_len = {warm_up_len}")

        B_past = B_scenario[:, :warm_up_len]
        H_past = H_init_scenario[:, :warm_up_len]
        B_future = B_scenario[:, warm_up_len:]
        T_batch = T_scenario.reshape(-1)

        preds = model_all(
            B_past=B_past,
            H_past=H_past,
            B_future=B_future,
            T=T_batch,
        )

        H_gt = H_true_scenario[:, warm_up_len:]

        # ---- metrics ----
        wce_per_sequence = np.max(np.abs(preds - H_gt), axis=1)
        mse_per_sequence = np.mean((preds - H_gt) ** 2, axis=1)
        sre_per_sequence = eqx.filter_vmap(sre)(preds, H_gt)

        dbdt_full = np.gradient(B_scenario, axis=1)
        dbdt = dbdt_full[:, warm_up_len:]
        nere_per_sequence = eqx.filter_vmap(nere)(preds, H_gt, dbdt, np.abs(true_core_loss))

        metrics_per_sequence[scenario_labels[scenario_i]] = {
            "mse": jnp.array(mse_per_sequence),
            "wce": jnp.array(wce_per_sequence),
            "sre": jnp.array(sre_per_sequence),
            "nere": jnp.array(nere_per_sequence),
        }

        # optional plots
        if show_plots:
            n_plots = min(5, preds.shape[0])
            idx_argmax = np.argpartition(wce_per_sequence, -n_plots)[-n_plots:]

            fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(10, 2.5 * n_plots))
            if n_plots == 1:
                axes = [axes]
            for j, idx in enumerate(idx_argmax):
                ax = axes[j]
                ax.plot(H_gt[idx], label="gt")
                ax.plot(preds[idx], label="pred", ls="dashed")
                ax.annotate(
                    f"MSE {mse_per_sequence[idx]:.1f} (A/m)² | WCE {wce_per_sequence[idx]:.1f} A/m",
                    (0.3, 0.1),
                    xycoords=ax.transAxes,
                )
                ax.grid(alpha=0.3)
                ax.set_ylabel("H in A/m")
            axes[0].set_title(f"Worst-case predictions - {scenario_labels[scenario_i]}")
            axes[-1].set_xlabel("Sequence step")
            axes[0].legend()
            fig.tight_layout()

    return metrics_per_sequence


def evaluate_pretest_scenarios(
    model_all, B, T, H_init, H_true, loss, msks_scenarios_N_tup, scenario_labels, show_plots: bool = False
):
    """
    Evaluates the given model on pretest scenarios with different amounts of unknown samples.
    Works with batched NumPy inputs (model_all takes arrays of shape (batch, time)).
    """

    metrics_per_sequence = get_metrics_per_sequence(
        model_all,
        B,
        T,
        H_init,
        H_true,
        loss,
        msks_scenarios_N_tup,
        scenario_labels,
        show_plots=show_plots,
    )

    metrics_reduced = {}

    for scenario_i, _ in enumerate(msks_scenarios_N_tup):

        mse_per_sequence = metrics_per_sequence[scenario_labels[scenario_i]]["mse"]
        wce_per_sequence = metrics_per_sequence[scenario_labels[scenario_i]]["wce"]
        sre_per_sequence = metrics_per_sequence[scenario_labels[scenario_i]]["sre"]
        nere_per_sequence = metrics_per_sequence[scenario_labels[scenario_i]]["nere"]

        # reduce metrics
        mse = np.mean(mse_per_sequence)
        wce = np.max(wce_per_sequence)

        sre_avg = np.mean(sre_per_sequence)
        sre_95th = np.percentile(sre_per_sequence, 95)

        nere_avg = np.mean(nere_per_sequence)
        nere_95th = np.percentile(nere_per_sequence, 95)

        print(f"\tMSE : {mse:>7.2f} (A/m)²")
        print(f"\tWCE : {wce:>7.2f} A/m")

        metrics_reduced[scenario_labels[scenario_i]] = {
            "mse": np.round(mse, 4).item(),
            "wce": np.round(wce, 4).item(),
            "sre_avg": np.round(sre_avg, 4).item(),
            "sre_95th": np.round(sre_95th, 4).item(),
            "nere_avg": np.round(nere_avg, 4).item(),
            "nere_95th": np.round(nere_95th, 4).item(),
        }

    return metrics_reduced


def produce_pretest_histograms(
    material_name: str,
    model_all: ModelInterface,
    B: np.ndarray | jax.Array,
    T: np.ndarray | jax.Array,
    H_init: np.ndarray | jax.Array,
    H_true: np.ndarray | jax.Array,
    loss: np.ndarray | jax.Array,
    msks_scenarios_N_tup: list[np.ndarray] | list[jax.Array],
    scenario_labels: list[str],
    adapted_scenario_labels: list[str],
    show_plots: bool = False,
):
    metrics_per_sequence = get_metrics_per_sequence(
        model_all,
        B,
        T,
        H_init,
        H_true,
        loss,
        msks_scenarios_N_tup,
        scenario_labels,
        show_plots=show_plots,
    )

    fig, axs = plt.subplots(3, 2, figsize=(8, 10))
    rel_pos_line_avg = 0.66
    rel_pos_line_95th = 0.5

    for scenario_i, _ in enumerate(msks_scenarios_N_tup):
        sre_per_sequence = metrics_per_sequence[scenario_labels[scenario_i]]["sre"] * 100
        nere_per_sequence = metrics_per_sequence[scenario_labels[scenario_i]]["nere"] * 100

        n_sequences = nere_per_sequence.shape[0]

        sre_avg = np.mean(sre_per_sequence)
        sre_95th = np.percentile(sre_per_sequence, 95)

        nere_avg = np.mean(nere_per_sequence)
        nere_95th = np.percentile(nere_per_sequence, 95)

        # SRE
        ax = axs[scenario_i, 0]
        ax.hist(
            sre_per_sequence,
            100,
            range=[0, 100],
            density=False,
            weights=1 / n_sequences * np.ones(sre_per_sequence.shape),
        )

        ax.vlines(sre_avg, *(0, (rel_pos_line_avg - 0.02) * ax.get_ylim()[-1]), color="red", linestyle="dashed")
        ax.text(sre_avg, rel_pos_line_avg * ax.get_ylim()[-1], f"Avg = {sre_avg:.3f}\%", color="red")
        ax.vlines(sre_95th, *(0, (rel_pos_line_95th - 0.02) * ax.get_ylim()[-1]), color="red", linestyle="dashed")
        ax.text(sre_95th, rel_pos_line_95th * ax.get_ylim()[-1], f".95 Prct = {sre_95th:.3f}\%", color="red")

        ax.set_xlabel("Sequence Relative Error [\%]")
        ax.set_ylabel("Ratio of Data Points")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            "\\textbf{Sequence Relative Error for }"
            + material_name
            + "\n"
            + adapted_scenario_labels[scenario_i]
            + "\n"
            + (f"Avg={sre_avg:.3f}\%, \n .95 Prct={sre_95th:.3f}\%, \n Max={np.max(np.abs(sre_per_sequence)):.3f}\%")
        )

        # NERE
        ax = axs[scenario_i, 1]
        ax.hist(
            nere_per_sequence,
            100,
            density=False,
            weights=1 / n_sequences * np.ones(sre_per_sequence.shape),
        )
        ax.vlines(nere_avg, *(0, (rel_pos_line_avg - 0.02) * ax.get_ylim()[-1]), color="red", linestyle="dashed")
        ax.text(nere_avg, rel_pos_line_avg * ax.get_ylim()[-1], f"Avg = {nere_avg:.3f}\%", color="red")
        ax.vlines(nere_95th, *(0, (rel_pos_line_95th - 0.02) * ax.get_ylim()[-1]), color="red", linestyle="dashed")
        ax.text(nere_95th, rel_pos_line_95th * ax.get_ylim()[-1], f".95 Prct = {nere_95th:.3f}\%", color="red")

        ax.set_xlabel("Normalized Energy Relative Error [\%]")
        ax.set_ylabel("Ratio of Data Points")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            "\\textbf{Normalized Energy Relative Error for }"
            + material_name
            + "\n"
            + adapted_scenario_labels[scenario_i]
            + "\n"
            + (f"Avg={nere_avg:.3f}\%, \n .95 Prct={nere_95th:.3f}\%, \n Max={np.max(np.abs(nere_per_sequence)):.3f}\%")
        )

    fig.suptitle(
        f"Material: {material_name}",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    return fig, axs


def store_predictions_to_csv(
    exp_id,
    model_all,
    B,
    T,
    H_init,
    H_true,
    loss,
    msks_scenarios_N_tup,
    scenario_labels,
) -> dict[str, jax.Array]:

    for scenario_i, msk_N in enumerate(msks_scenarios_N_tup):
        print(f"  Scenario {scenario_i} - {scenario_labels[scenario_i]}: ")

        B_scenario = B[msk_N]
        T_scenario = T[msk_N]
        H_init_scenario = H_init[msk_N]
        H_true_scenario = H_true[msk_N]

        warm_up_len = np.sum(~np.isnan(H_init_scenario[0]))
        print(warm_up_len / B_scenario.shape[1])
        print(f"    -> warm_up_len = {warm_up_len}")

        B_past = B_scenario[:, :warm_up_len]
        H_past = H_init_scenario[:, :warm_up_len]
        B_future = B_scenario[:, warm_up_len:]
        T_batch = T_scenario.reshape(-1)

        preds = model_all(
            B_past=B_past,
            H_past=H_past,
            B_future=B_future,
            T=T_batch,
        )

        H_gt = H_true_scenario[:, warm_up_len:]

        for seq_pred, seq_gt in zip(preds, H_gt):
            with open(f"{exp_id}_pred.csv", "a") as f:
                np.savetxt(f, seq_pred[None, :], delimiter=",")
                f.close()
            with open(f"{exp_id}_meas.csv", "a") as f:
                np.savetxt(f, seq_gt[None, :], delimiter=",")
                f.close()
