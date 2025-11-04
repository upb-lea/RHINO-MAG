import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def evaluate_pretest_scenarios_custom(
    model_all, B, T, H_init, H_true, loss, msks_scenarios_N_tup, scenario_labels, show_plots: bool = False
):
    """
    Evaluates the given model on pretest scenarios with different amounts of unknown samples.
    Works with batched NumPy inputs (model_all takes arrays of shape (batch, time)).
    """

    metrics_d = {}

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

        # preds = model_all(
        #     B_past=B_past,
        #     H_past=H_past,
        #     B_future=B_future,
        #     T=T_batch,
        # )
        preds = model_all.call_with_warmup(
            B_past=B_past,
            H_past=H_past,
            B_future=B_future,
            T=T_batch,
        )

        H_gt = H_true_scenario[:, warm_up_len:]

        # ---- metrics ----
        wce_per_sequence = np.max(np.abs(preds - H_gt), axis=1)
        mse_per_sequence = np.mean((preds - H_gt) ** 2, axis=1)

        mse = np.mean(mse_per_sequence)
        wce = np.max(np.abs(preds - H_gt))
        sre_per_sequence = np.sqrt(mse_per_sequence) / np.sqrt(np.mean(H_gt**2, axis=1))
        sre_avg = np.mean(sre_per_sequence)
        sre_95th = np.percentile(sre_per_sequence, 95)

        dbdt_full = np.gradient(B_scenario, axis=1)
        dbdt = dbdt_full[:, warm_up_len:]
        nere_per_sequence = np.abs(
            (((dbdt * preds) - (dbdt * H_gt)).sum(axis=1)) / np.abs(loss[msk_N][:, 0])
        )  # added abs
        nere_avg = np.mean(nere_per_sequence)
        nere_95th = np.percentile(nere_per_sequence, 95)

        print(f"\tMSE : {mse:>7.2f} (A/m)²")
        print(f"\tWCE : {wce:>7.2f} A/m")

        metrics_d[scenario_labels[scenario_i]] = {
            "mse": np.round(mse, 3).item(),
            "wce": np.round(wce, 3).item(),
            "sre_avg": np.round(sre_avg, 3).item(),
            "sre_95th": np.round(sre_95th, 3).item(),
            "nere_avg": np.round(nere_avg, 3).item(),
            "nere_95th": np.round(nere_95th, 3).item(),
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

    return metrics_d
