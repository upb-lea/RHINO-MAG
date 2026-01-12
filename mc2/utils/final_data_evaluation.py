import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import DATA_ROOT, FINAL_MATERIALS, load_data_into_pandas_based_on_path, MaterialSet
from mc2.model_interfaces.model_interface import ModelInterface, count_model_parameters
from mc2.utils.model_evaluation import reconstruct_model_from_exp_id, get_exp_ids, evaluate_cross_validation


FINAL_SCENARIOS_PER_MATERIAL = {
    "A": {"10%known_90%unknown": [100, 900], "50%known_50%unknown": [500, 500], "90%known_10%unknown": [900, 100]},
    "B": {"10%known_90%unknown": [100, 900], "50%known_50%unknown": [500, 500], "90%known_10%unknown": [900, 100]},
    "C": {
        "1%known_99%unknown": [1, 999],
    },  # how to start at H=0?
    "D": {"10%known_90%unknown": [100, 900], "50%known_50%unknown": [500, 500], "90%known_10%unknown": [900, 100]},
    "E": {"10%known_90%unknown": [100, 900], "50%known_50%unknown": [500, 500], "90%known_10%unknown": [900, 100]},
}


def load_test_data_into_pandas_df(
    material: str,
    number: int = None,
) -> dict:
    """Load data selectively from raw CSV files if cache does not exist yet. Caches loaded data for next time."""
    return load_data_into_pandas_based_on_path(
        raw_path=DATA_ROOT / "final_testing_data" / "raw",
        cache_path=DATA_ROOT / "final_testing_data" / "cache",
        material=material,
        number=number,
    )


class TestSet(eqx.Module):
    """Class for holding the final test data.

    These are sequences without frequency information and the missing values (which are
        to be predicted) are padded with NaNs.
    """

    class TestScenario(eqx.Module):
        material_name: str
        mask: jax.Array
        H_past: jax.Array
        B_past: jax.Array
        B_future: jax.Array
        T: jax.Array
        N_unknown: int

        @property
        def N_known(self):
            return self.H_past.shape[-1]

    material_name: str
    H: jax.Array
    B: jax.Array
    T: jax.Array

    @classmethod
    def from_dict(cls, data_dict: dict) -> "TestSet":
        """Create a TestSet from a dictionary."""
        return cls(
            material_name=data_dict["material_name"],
            H=jnp.array(data_dict["H"]),
            B=jnp.array(data_dict["B"]),
            T=jnp.array(data_dict["T"]),
        )

    @classmethod
    def from_material_name(cls, material_name: str):
        data_dict = load_test_data_into_pandas_df(material_name)
        return cls.from_dict(
            data_dict={
                "material_name": material_name,
                "H": data_dict[f"{material_name}_Padded_H_seq"],
                "B": data_dict[f"{material_name}_True_B_seq"],
                "T": data_dict[f"{material_name}_True_T"],
            }
        )

    @property
    def scenarios(self):
        unknowns_N = jnp.isnan(self.H).sum(axis=1)
        unknown_samples_variants = jnp.array(pd.unique(np.array(unknowns_N)))

        scenarios = []

        for N_unknown in unknown_samples_variants:

            mask = unknowns_N == N_unknown

            scenarios.append(
                self.TestScenario(
                    material_name=self.material_name,
                    mask=mask,
                    H_past=self.H[mask, :-N_unknown],
                    B_past=self.B[mask, :-N_unknown],
                    B_future=self.B[mask, -N_unknown:],
                    T=self.T[mask],
                    N_unknown=N_unknown,
                )
            )
        return scenarios


class ResultSet(eqx.Module):
    material_name: str
    H: jax.Array
    B: jax.Array
    T: jax.Array
    exp_id: str


def predict_test_scenarios(
    models: dict[str, ModelInterface],
    test_data: dict[str, TestSet],
    exp_ids: dict[str, str],
) -> dict[str, ResultSet]:

    result_sets = {}

    for material_name in FINAL_MATERIALS:
        print("Evaluate test data for material: ", material_name)
        test_set = test_data[material_name]
        model = models[material_name]
        exp_id = exp_ids[material_name]
        assert exp_id.split("_")[0] == material_name

        print(f"The model has {model.n_params} parameters.")

        filled_H_trajectories = []
        for scenario in test_set.scenarios:
            print(
                f"Running scenario with a warmup of {scenario.N_known} steps and {scenario.N_unknown} unknown elements."
            )
            H_pred = model(
                B_past=scenario.B_past,
                H_past=scenario.H_past,
                B_future=scenario.B_future,
                T=jnp.squeeze(scenario.T),
            )
            filled_H_trajectories.append(jnp.concatenate([scenario.H_past, H_pred], axis=1))

        material_result_set = ResultSet(
            material_name=material_name,
            H=jnp.concatenate(filled_H_trajectories),
            B=test_set.B,
            T=test_set.T,
            exp_id=exp_id,
        )

        result_sets[material_name] = material_result_set
        print("Done with material:", material_name, "\n")
    return result_sets


def validate_result_set(
    result_set: ResultSet,
    test_set: TestSet,
) -> None:
    material_name = result_set.material_name
    print("Sanity checking results for material:", material_name)

    assert material_name == test_set.material_name
    assert material_name == result_set.exp_id.split("_")[0]

    assert jnp.all(result_set.B == test_set.B)
    assert jnp.all(result_set.T == test_set.T)

    for scenario in test_set.scenarios:
        assert result_set.H[scenario.mask].shape == test_set.H[scenario.mask].shape
        assert jnp.all(
            result_set.H[scenario.mask][:, : scenario.N_known] == test_set.H[scenario.mask][:, : scenario.N_known]
        )
        assert jnp.all(jnp.isnan(test_set.H[scenario.mask][:, scenario.N_known :]))
        assert jnp.all(~jnp.isnan(result_set.H[scenario.mask][:, scenario.N_known :]))

    print(f"Results for '{material_name}' seem consistent with the test data.")


def visualize_result_set(result_set: ResultSet, figsize=(30, 8)):
    fig, axs = plt.subplots(3, 7, figsize=figsize)

    n_plots_per_row = 7

    n_sequences = result_set.H.shape[0]
    if n_sequences < n_plots_per_row:
        n_plots_per_row = n_sequences

    for seq_idx in range(n_plots_per_row):
        axs[0, seq_idx].plot(result_set.B[seq_idx], color="tab:blue")
        axs[1, seq_idx].plot(result_set.H[seq_idx], color="tab:orange")

        axs[2, seq_idx].plot(result_set.H[seq_idx], result_set.B[seq_idx], color="tab:orange")

        axs[0, seq_idx].grid(True, alpha=0.3)
        axs[1, seq_idx].grid(True, alpha=0.3)
        axs[2, seq_idx].grid(True, alpha=0.3)

        axs[0, seq_idx].set_ylabel("B")
        axs[0, seq_idx].set_xlabel("k")
        axs[1, seq_idx].set_ylabel("H")
        axs[1, seq_idx].set_xlabel("k")
        axs[2, seq_idx].set_ylabel("B")
        axs[2, seq_idx].set_xlabel("H")

    fig.tight_layout(pad=-0.2)
    return fig, axs


def generate_metrics_from_exp_ids_without_seed(
    exp_ids_without_seed: list[str], material_name: str, loader_key: jax.random.PRNGKey
):
    assert np.all([material_name == exp_id.split("_")[0] for exp_id in exp_ids_without_seed])
    exp_ids = [exp_id for exp_id in get_exp_ids() if "_".join(exp_id.split("_")[:-1]) in exp_ids_without_seed]
    models = {exp_id: reconstruct_model_from_exp_id(exp_id)[0] for exp_id in exp_ids}

    mat_set = MaterialSet.from_material_name(material_name)
    _, _, test_set = mat_set.split_into_train_val_test(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0)

    all_results = []
    all_metrics = {}

    for exp_id, wrapped_model in models.items():
        metrics = evaluate_cross_validation(
            wrapped_model=wrapped_model,
            test_set=test_set,
            scenarios=FINAL_SCENARIOS_PER_MATERIAL[test_set.material_name],
            sequence_length=1000,
            batch_size_per_frequency=1000,
            loader_key=loader_key,
        )
        model_params = wrapped_model.n_params

        seed = exp_id.split("seed")[-1]
        model_type = exp_id.split("_")[1]
        exp_name = exp_id.split("_")[2]
        num_id = exp_id.split("_")[-2]

        for scenario, values in metrics.items():
            all_results.append(
                {
                    "exp_id_without_seed": [e for e in exp_ids_without_seed if "_".join(exp_id.split("_")[:-1]) == e][
                        0
                    ],
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    "num_id": num_id,
                    "material": material_name,
                    "model_type": model_type,
                    "seed": seed,
                    "n_params": model_params,
                    "scenario": scenario,
                    "sre_avg": values["sre_avg"],
                    "sre_95th": values["sre_95th"],
                    "nere_avg": values["nere_avg"],
                    "nere_95th": values["nere_95th"],
                }
            )

        all_metrics[exp_id] = metrics

    if len(all_results) == 0:
        raise ValueError("No results could be found for the specified experiment IDs.")

    df = pd.DataFrame(all_results)
    df["params_label"] = df["n_params"].astype(str)

    df = df.sort_values(by="exp_id")
    df = df.reset_index(drop=True)

    return df, all_metrics


def visualize_df(df, scenarios, metrics, x_label=None):

    fig, axs = plt.subplots(nrows=len(scenarios), ncols=len(metrics), figsize=(12, 12 / 3 * len(scenarios)))
    axs = np.atleast_2d(axs)

    available_exp_ids = list(str(element) for element in np.unique(list(df["exp_id_without_seed"])))

    for i, scenario in enumerate(scenarios):
        df_scenario = df[df["scenario"] == scenario]

        colors = plt.rcParams["axes.prop_cycle"]()

        for exp_id in available_exp_ids:
            c = next(colors)["color"]
            df_exp_id = df_scenario[df_scenario["exp_id_without_seed"] == exp_id]

            for j, metric in enumerate(metrics):
                ax = axs[i, j]
                ax.set_title(f"Scenario: {scenario}", fontsize=14)
                ax.set_ylabel(f"{metric}", fontsize=12)

                metric_avg = df_exp_id[f"{metric}_avg"]
                metric_95th = df_exp_id[f"{metric}_95th"]

                if x_label is not None:
                    ax.plot(df_exp_id[x_label], metric_avg, marker="o", alpha=0.6, c=c, label=exp_id)
                    ax.plot(df_exp_id[x_label], metric_95th, marker="^", alpha=0.6, c=c)
                else:
                    ax.plot(metric_avg, marker="o", alpha=0.6, c=c, label=exp_id)
                    ax.plot(metric_95th, marker="^", alpha=0.6, c=c)

                ax.grid(True, alpha=0.3)
                ax.legend()
    fig.tight_layout()
    return fig, axs
