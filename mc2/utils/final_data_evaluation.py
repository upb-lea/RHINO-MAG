import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

from mc2.data_management import DATA_ROOT, FINAL_MATERIALS, load_data_into_pandas_based_on_path
from mc2.model_interfaces.model_interface import ModelInterface


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

        if material_name == "C":
            print("WARNING: REMOVING LAST ELEMENT OF TEMPERATURE ARRAY")
            test_set = eqx.tree_at(lambda x: x.T, test_set, test_set.T[:-1])

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

    if material_name == "C":
        print("WARNING: REMOVING LAST ELEMENT OF TEMPERATURE ARRAY")
        test_set = eqx.tree_at(lambda x: x.T, test_set, test_set.T[:-1])

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
