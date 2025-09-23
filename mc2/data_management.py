import tqdm
import pandas as pd
import pickle
import json
from typing import Dict, Tuple, Callable
from pathlib import Path
import logging
import sys
from uuid import uuid4
from sklearn.model_selection import train_test_split

import h5py
import jax
import jax.numpy as jnp
import torch
import numpy as np
import equinox as eqx

from mc2.utils.data_inspection import load_and_process_single_from_full_file_overview


DATA_ROOT = Path(__file__).parent.parent / "data"
PRETEST_DATA_ROOT = DATA_ROOT / "pretest_raw"
MODEL_DUMP_ROOT = DATA_ROOT / "models"
CACHE_ROOT = DATA_ROOT / "cache"
PRETEST_CACHE_ROOT = DATA_ROOT / "pretest_cache"
EXPERIMENT_LOGS_ROOT = DATA_ROOT / "experiment_logs"
NORMALIZATION_ROOT = DATA_ROOT / "normalization_values"

for root_dir in (
    CACHE_ROOT,
    MODEL_DUMP_ROOT,
    EXPERIMENT_LOGS_ROOT,
    PRETEST_DATA_ROOT,
    PRETEST_CACHE_ROOT,
    NORMALIZATION_ROOT,
):
    root_dir.mkdir(parents=True, exist_ok=True)


AVAILABLE_MATERIALS = [
    "3C90",
    "3C94",
    "3E6",
    "3F4",
    "77",
    "78",
    "N27",
    "N30",
    "N49",
    "N87",
]

DESIRED_DT_FMT = "%Y-%m-%d %H:%M:%S"  # desired datetime format
LOG_FORMATTER = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s", DESIRED_DT_FMT)
LOG_COLUMN_WIDTH = 40


def setup_package_logging():
    """Configure package-wide logging settings."""
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(LOG_FORMATTER)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler],
    )


setup_package_logging()


class Normalizer(eqx.Module):
    B_max: float
    H_max: float
    T_max: float
    # f_max: float
    norm_fe_max: list[float] = eqx.field(static=True)
    H_transform: callable = eqx.field(static=True)
    H_inverse_transform: callable = eqx.field(static=True)

    def normalize(self, B, H, T):  # ,f
        return (B / self.B_max, self.H_transform(H / self.H_max), T / self.T_max)  # , f / self.f_max

    def normalize_H(self, H):
        return self.H_transform(H / self.H_max)

    def denormalize(self, B, H, T):  # ,f
        H = self.H_inverse_transform(H)
        return B * self.B_max, H * self.H_max, T * self.T_max  # , f / self.f_max

    def denormalize_H(self, H):
        H = self.H_inverse_transform(H)
        return H * self.H_max

    def normalize_fe(self, features):
        fe_norm = features / jnp.array(self.norm_fe_max)
        return fe_norm

    def denormalize_fe(self, features):
        return features * jnp.array(self.norm_fe_max)


class FrequencySet(eqx.Module):
    """Class to store measurement data for a single material for a single frequency
    but with potentially variable temperatures.

    Args:
        material_name (str): Name of the material as a scalar string.
        frequency (float): Frequency of the measurement as a scalar float.
        H (jax.Array): Magnetic field strength data in shape (n_sequences, sequence_length).
        B (jax.Array): Magnetic flux density data in space (n_sequences, sequence_length).
        T (jax.Array): Temperature data in shape (n_sequences).
    """

    material_name: str
    frequency: float
    H: jax.Array
    B: jax.Array
    T: jax.Array

    @classmethod
    def from_dict(cls, data_dict: dict) -> "FrequencySet":
        """Create a FrequencySet from a dictionary."""
        return cls(
            material_name=data_dict["material_name"],
            frequency=data_dict["frequency"],
            H=jnp.array(data_dict["H"]),
            B=jnp.array(data_dict["B"]),
            T=jnp.array(data_dict["T"]),
        )

    @classmethod
    def load_from_raw_data(cls, file_overview: pd.DataFrame, material_name: str, frequency: float) -> "FrequencySet":
        """Load a FrequencySet from raw data."""
        data_dict = load_and_process_single_from_full_file_overview(
            file_overview, material_name=material_name, data_type=["B", "T", "H"], frequency=[frequency]
        )
        return cls.from_dict(data_dict)

    def filter_temperatures(self, temperatures: list[float] | jax.Array) -> "FrequencySet":
        """Filter the frequency set by temperatures."""
        if isinstance(temperatures, list):
            temperatures = jnp.array(temperatures)

        temperature_mask = jnp.isin(self.T, temperatures)
        return FrequencySet(
            material_name=self.material_name,
            frequency=self.frequency,
            H=self.H[temperature_mask],
            B=self.B[temperature_mask],
            T=self.T[temperature_mask],
        )

    def split_into_train_val_test(
        self,
        train_frac: float,
        val_frac: float,
        test_frac: float,
        seed: int = 0,
    ) -> Tuple["FrequencySet", "FrequencySet", "FrequencySet"]:
        """Split a FrequencySet into train, validation and test sets, stratified by temperature.

        For each temperature, sequences are split into train, val and test separately,
        then combined across temperatures.
        """
        unique_temps = jnp.unique(self.T)

        train_H, train_B, train_T = [], [], []
        val_H, val_B, val_T = [], [], []
        test_H, test_B, test_T = [], [], []

        for temp in unique_temps:
            # Use the class method to filter sequences for this temperature
            freq_temp = self.filter_temperatures([temp])

            # Indices for this temperature sequences
            indices = list(range(freq_temp.H.shape[0]))

            # First split into train+val and test
            train_val_idx, test_idx = train_test_split(indices, test_size=test_frac, random_state=seed, shuffle=True)
            # Then split train+val into train and val
            val_relative_frac = val_frac / (train_frac + val_frac)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_relative_frac,
                random_state=seed,
                shuffle=True,
            )

            # Append split sequences for this temperature to respective sets
            train_idx = jnp.array(train_idx)
            val_idx = jnp.array(val_idx)
            test_idx = jnp.array(test_idx)

            train_H.append(freq_temp.H[train_idx])
            train_B.append(freq_temp.B[train_idx])
            train_T.append(freq_temp.T[train_idx])

            val_H.append(freq_temp.H[val_idx])
            val_B.append(freq_temp.B[val_idx])
            val_T.append(freq_temp.T[val_idx])

            test_H.append(freq_temp.H[test_idx])
            test_B.append(freq_temp.B[test_idx])
            test_T.append(freq_temp.T[test_idx])

        # Concatenate all temperature splits per dataset
        train_set = FrequencySet(
            material_name=self.material_name,
            frequency=self.frequency,
            H=jnp.concatenate(train_H),
            B=jnp.concatenate(train_B),
            T=jnp.concatenate(train_T),
        )

        val_set = FrequencySet(
            material_name=self.material_name,
            frequency=self.frequency,
            H=jnp.concatenate(val_H),
            B=jnp.concatenate(val_B),
            T=jnp.concatenate(val_T),
        )

        test_set = FrequencySet(
            material_name=self.material_name,
            frequency=self.frequency,
            H=jnp.concatenate(test_H),
            B=jnp.concatenate(test_B),
            T=jnp.concatenate(test_T),
        )

        return train_set, val_set, test_set

    def normalize(self, normalizer: Normalizer = None, transform_H: bool = False, featurize: Callable = None):
        if normalizer is None:
            H_max = jnp.max(jnp.abs(self.H))
            B_max = jnp.max(jnp.abs(self.B))
            T_max = jnp.max(jnp.abs(self.T))

            if transform_H:
                transform = lambda h: jnp.tanh(h * 1.2)
                inverse_transform = lambda h: jnp.atanh(h) / 1.2
            else:
                transform = lambda h: h
                inverse_transform = lambda h: h

            normalizer = Normalizer(
                B_max=B_max.item(),
                H_max=H_max.item(),
                T_max=T_max.item(),
                norm_fe_max=[],
                H_transform=transform,
                H_inverse_transform=inverse_transform,
            )  # f_max=800_000,

            norm_B, norm_H, norm_T = normalizer.normalize(self.B, self.H, self.T)  # , norm_f , self.frequency
            if featurize is not None:
                features = jax.vmap(featurize, in_axes=(0, 0, 0, 0))(
                    norm_B[:, :10], norm_H, norm_B[:, 10:], norm_T
                )  # , norm_f
                max_features = jnp.max(jnp.abs(features), axis=(0, 1))
                normalizer = Normalizer(
                    B_max=B_max.item(),
                    H_max=H_max.item(),
                    T_max=T_max.item(),
                    norm_fe_max=max_features.tolist(),
                    H_transform=transform,
                    H_inverse_transform=inverse_transform,
                )  # f_max=800_000,
        else:
            norm_B, norm_H, norm_T = normalizer.normalize(self.B, self.H, self.T)  # , norm_f , self.frequency
            if featurize is not None:
                features = jax.vmap(featurize, in_axes=(0, 0, 0, 0))(
                    norm_B[:, :10], norm_H, norm_B[:, 10:], norm_T
                )  # , norm_f
                max_features = jnp.max(jnp.abs(features), axis=(0, 1))
                normalizer = Normalizer(
                    B_max=normalizer.B_max,
                    H_max=normalizer.H_max,
                    T_max=normalizer.T_max,
                    norm_fe_max=max_features.tolist(),
                    H_transform=normalizer.H_transform,
                    H_inverse_transform=normalizer.H_inverse_transform,
                )  # f_max=normalizer.f_max,

        return NormalizedFrequencySet(
            material_name=self.material_name,
            frequency=self.frequency,
            H=norm_H,
            B=norm_B,
            T=norm_T,
            normalizer=normalizer,
        )


class NormalizedFrequencySet(FrequencySet):
    normalizer: Normalizer

    def denormalize(self):
        B, H, T = self.normalizer.denormalize(self.B, self.H, self.T)  # , frequency , self.frequency
        return FrequencySet(
            material_name=self.material_name,
            frequency=self.frequency,
            H=H,
            B=B,
            T=T,
        )


class MaterialSet(eqx.Module):
    """Class to store measurement data for a single material but with variable
    frequencies and temperatures.

    Args:
        material_name (str): Name of the material as a scalar string.
        frequency_sets (list[FrequencySet]): List of FrequencySet objects.
        frequencies (jax.Array): Frequencies in Hz as a 1D array.
    """

    material_name: str
    frequency_sets: list[FrequencySet]
    frequencies: jax.Array

    @classmethod
    def load_from_file(cls, file_path: str):
        """Load a MaterialSet from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def save_to_file(self, file_path: str):
        """Save the MaterialSet to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_raw_data(
        cls,
        file_overview: pd.DataFrame,
        material_name: str,
        frequencies: list[float] | jax.Array,
    ) -> "MaterialSet":
        """Load a MaterialSet from raw data."""
        frequency_sets = []
        for frequency in frequencies:
            frequency_set = FrequencySet.load_from_raw_data(file_overview, material_name, frequency)
            frequency_sets.append(frequency_set)

        return cls(
            material_name=material_name,
            frequency_sets=frequency_sets,
            frequencies=jnp.array(frequencies),
        )

    @classmethod
    def from_pandas_dict(
        cls,
        data_d: dict[str, pd.DataFrame],
        frequencies: list[float] = (50000.0, 80000.0, 125000.0, 200000.0, 320000.0, 500000.0, 800000.0),
    ) -> "MaterialSet":
        """Create a MaterialSet from a dictionary of pandas DataFrames."""

        # Extract the material name from the first key in the data dictionary
        sample_key = next(iter(data_d))
        material_name = sample_key.split("_")[0]

        frequency_sets = []
        for idx, freq in enumerate(frequencies):
            key_base = f"{material_name}_{idx + 1}"
            B_key = f"{key_base}_B"
            H_key = f"{key_base}_H"
            T_key = f"{key_base}_T"

            assert B_key in data_d and H_key in data_d and T_key in data_d, f"Missing data for frequency {freq} Hz"

            B = jnp.array(data_d[B_key].values)
            H = jnp.array(data_d[H_key].values)
            T = jnp.array(data_d[T_key].values)[:, 0]

            freq_set = FrequencySet(material_name=material_name, frequency=freq, B=B, H=H, T=T)

            frequency_sets.append(freq_set)

        return cls(
            material_name=material_name,
            frequencies=jnp.array(frequencies),
            frequency_sets=frequency_sets,
        )

    def to_pandas_dict(self) -> dict[str, pd.DataFrame]:
        """Convert the MaterialSet to a dictionary of pandas DataFrames."""
        data_dict = {}

        for idx, freq_set in enumerate(self.frequency_sets):
            prefix = f"{freq_set.material_name}_{idx + 1}"

            B_df = pd.DataFrame(jnp.array(freq_set.B))
            H_df = pd.DataFrame(jnp.array(freq_set.H))
            T_df = pd.DataFrame(jnp.array(freq_set.T))

            data_dict[f"{prefix}_B"] = B_df
            data_dict[f"{prefix}_H"] = H_df
            data_dict[f"{prefix}_T"] = T_df

        return data_dict

    def __getitem__(self, idx: int) -> FrequencySet:
        """Return the frequency set at the given index."""
        return self.frequency_sets[idx]

    def __iter__(self):
        """Return an iterator over the frequency sets."""
        return iter(self.frequency_sets)

    def at_frequency(self, frequency: float) -> FrequencySet:
        """Return the frequency set at the given index."""
        return self.frequency_sets[jnp.where(self.frequencies == frequency)[0][0]]

    def filter_temperatures(self, temperatures: list[float] | jax.Array) -> "MaterialSet":
        """Filter the material set by temperatures."""
        filtered_frequency_sets = [
            frequency_set.filter_temperatures(temperatures) for frequency_set in self.frequency_sets
        ]

        return MaterialSet(
            material_name=self.material_name,
            frequency_sets=filtered_frequency_sets,
            frequencies=jnp.array([fs.frequency for fs in filtered_frequency_sets]),
        )

    def filter_frequencies(self, frequencies: list[float] | jax.Array) -> "MaterialSet":
        """Filter the material set by frequencies."""
        filtered_frequency_sets = []

        frequencies = jnp.array(frequencies)

        for frequency_set in self.frequency_sets:
            if frequency_set.frequency in frequencies:
                filtered_frequency_sets.append(frequency_set)

        return MaterialSet(
            material_name=self.material_name,
            frequency_sets=filtered_frequency_sets,
            frequencies=jnp.array([fs.frequency for fs in filtered_frequency_sets]),
        )

    def split_into_train_val_test(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 0,
    ) -> Tuple["MaterialSet", "MaterialSet", "MaterialSet"]:
        """
        Split a MaterialSet into train, val, test MaterialSets.
        Each FrequencySet inside is split stratified by temperature sequences.
        """

        train_frequency_sets = []
        val_frequency_sets = []
        test_frequency_sets = []

        for freq_set in self.frequency_sets:
            train_fs, val_fs, test_fs = freq_set.split_into_train_val_test(train_frac, val_frac, test_frac, seed)
            train_frequency_sets.append(train_fs)
            val_frequency_sets.append(val_fs)
            test_frequency_sets.append(test_fs)

        frequencies = self.frequencies
        train_material_set = MaterialSet(
            material_name=self.material_name,
            frequency_sets=train_frequency_sets,
            frequencies=frequencies,
        )
        val_material_set = MaterialSet(
            material_name=self.material_name,
            frequency_sets=val_frequency_sets,
            frequencies=frequencies,
        )
        test_material_set = MaterialSet(
            material_name=self.material_name,
            frequency_sets=test_frequency_sets,
            frequencies=frequencies,
        )

        return train_material_set, val_material_set, test_material_set

    def normalize(
        self,
        normalizer: Normalizer = None,
        transform_H: bool = False,
        featurize: Callable = None,
        feature_names: list[str] = None,
    ) -> "NormalizedMaterialSet":
        if normalizer is None:
            if featurize is None:
                if feature_names is None:
                    raise ValueError(
                        "You must specify either 'featurize' or 'feature_names'. "
                        "Without one of them, it is not possible to determine how to obtain "
                        "the normalization values."
                    )
                else:
                    file_json = NORMALIZATION_ROOT / "norm_values.json"
                    with open(file_json, "r") as file:
                        norm_values = json.load(file)
                    norm_values_mat = norm_values[self.material_name]["train_set"]
                    B_max = norm_values_mat["B_max"]
                    H_max = norm_values_mat["H_max"]
                    T_max = norm_values_mat["T_max"]
                    # f_max = norm_values_mat["f_max"]
                    H_transform = lambda h: jnp.tanh(h * 1.2)
                    H_inverse_transform = lambda h: jnp.atanh(h) / 1.2
                    max_norm_fe_max = [norm_values_mat["features"][name] for name in feature_names]
                    normalizer = Normalizer(
                        B_max=B_max,
                        H_max=H_max,
                        T_max=T_max,
                        norm_fe_max=max_norm_fe_max,
                        H_transform=H_transform,
                        H_inverse_transform=H_inverse_transform,
                    )  # f_max=f_max,
            else:
                if feature_names is not None:
                    raise ValueError(
                        "Specify either 'featurize' OR 'feature_names', but not both. "
                        "Otherwise, it is unclear whether normalization values should be "
                        "recomputed using 'featurize' or loaded based on the provided 'feature_names'."
                    )
                frequency_sets_different_norm = [
                    freq_set.normalize(transform_H=transform_H, featurize=featurize) for freq_set in self.frequency_sets
                ]
                B_max = max(freq_set.normalizer.B_max for freq_set in frequency_sets_different_norm)
                H_max = max(freq_set.normalizer.H_max for freq_set in frequency_sets_different_norm)
                T_max = max(freq_set.normalizer.T_max for freq_set in frequency_sets_different_norm)
                # f_max = max(freq_set.normalizer.f_max for freq_set in frequency_sets_different_norm)

                H_transform = frequency_sets_different_norm[0].normalizer.H_transform
                H_inverse_transform = frequency_sets_different_norm[0].normalizer.H_inverse_transform

                pre_normalizer = Normalizer(
                    B_max=B_max,
                    H_max=H_max,
                    T_max=T_max,
                    norm_fe_max=[],
                    H_transform=H_transform,
                    H_inverse_transform=H_inverse_transform,
                )
                # f_max=f_max,
                frequency_sets_norm_pre = [
                    freq_set.normalize(normalizer=pre_normalizer, transform_H=transform_H, featurize=featurize)
                    for freq_set in self.frequency_sets
                ]

                max_norm_fe_max = jnp.max(
                    jnp.array([freq_set.normalizer.norm_fe_max for freq_set in frequency_sets_norm_pre]), axis=0
                )
                normalizer = Normalizer(
                    B_max=B_max,
                    H_max=H_max,
                    T_max=T_max,
                    norm_fe_max=max_norm_fe_max.tolist(),
                    H_transform=H_transform,
                    H_inverse_transform=H_inverse_transform,
                )  # f_max=f_max,

        frequency_sets_norm = [
            freq_set.normalize(normalizer=normalizer, transform_H=transform_H) for freq_set in self.frequency_sets
        ]

        frequencies_norm = [freq_set_norm.frequency for freq_set_norm in frequency_sets_norm]

        return NormalizedMaterialSet(
            material_name=self.material_name,
            frequency_sets=frequency_sets_norm,
            frequencies=frequencies_norm,
            normalizer=normalizer,
        )

    def subsample(self, sampling_freq: int) -> "MaterialSet":
        subsampled_freq_set_list = [
            FrequencySet(
                freq_set.material_name,
                freq_set.frequency,
                freq_set.H[:, ::sampling_freq],
                freq_set.B[:, ::sampling_freq],
                freq_set.T[:],
            )
            for freq_set in self
        ]

        return MaterialSet(
            self.material_name,
            subsampled_freq_set_list,
            self.frequencies,
        )


class NormalizedMaterialSet(MaterialSet):
    normalizer: Normalizer

    def denormalize(self):
        frequency_sets_denormalized = [frequency_set.denormalize for frequency_set in self.frequency_sets]
        return MaterialSet(
            material_name=self.material_name,
            frequency_sets=frequency_sets_denormalized,
            frequencies=[freq_set.frequency for freq_set in frequency_sets_denormalized],
        )


class DataSet(eqx.Module):
    """Class to store measurement data for multiple materials.

    Args:
        material_sets (list[MaterialSet]): List of MaterialSet objects.
        material_names (jax.Array): Names of the materials as a 1D array.
    """

    material_sets: list[MaterialSet]
    material_names: list[str]
    _name_to_idx: dict[str, int]

    @classmethod
    def load_from_raw_data(
        cls,
        file_overview: pd.DataFrame,
        material_names: list[str],
        frequencies: list[float] | jax.Array,
    ) -> "DataSet":
        """Load a DataSet from raw data."""

        material_sets = []

        for material_name in tqdm.tqdm(material_names, desc="Loading MaterialSets"):
            material_set = MaterialSet.load_from_raw_data(file_overview, material_name, frequencies)
            material_sets.append(material_set)

        return cls(material_sets)

    @classmethod
    def load_from_file(cls, file_path: str):
        """Load a DataSet from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def save_to_file(self, file_path: str):
        """Save the DataSet to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def __init__(self, material_sets: list[MaterialSet]):
        """Initialize the DataSet with a list of MaterialSet objects."""
        self.material_sets = material_sets
        self.material_names = [material_set.material_name for material_set in material_sets]
        self._name_to_idx = {name: idx for idx, name in enumerate(self.material_names)}

    def __getitem__(self, idx: int) -> MaterialSet:
        """Return the material set at the given index."""
        return self.material_sets[idx]

    def __iter__(self):
        """Return an iterator over the material sets."""
        return iter(self.material_sets)

    def at_material(self, material_name: str) -> MaterialSet:
        """Return the material set at the given index."""
        return self.material_sets[self._name_to_idx[material_name]]

    def filter_temperatures(self, temperatures: list[float] | jax.Array) -> "DataSet":
        """Filter the dataset by temperatures."""
        return DataSet([material_set.filter_temperatures(temperatures) for material_set in self.material_sets])

    def filter_frequencies(self, frequencies: list[float]) -> "DataSet":
        """Filter the dataset by frequencies."""
        return DataSet([material_set.filter_frequencies(frequencies) for material_set in self.material_sets])

    def filter_materials(self, material_names: list[str]) -> "DataSet":
        """Filter the dataset by material names."""
        filtered_material_sets = []

        for material_set in self.material_sets:
            if material_set.material_name in material_names:
                filtered_material_sets.append(material_set)

        return DataSet(filtered_material_sets)


def load_data_into_pandas_df(
    material: str = None, number: int = None, training: bool = True, n_rows: int = None
) -> dict:
    """Load data selectively from raw CSV files if cache does not exist yet. Caches loaded data for next time."""
    # TODO implement training vs testing data

    if material is None:
        # load all materials
        raise NotImplementedError()
    else:
        mat_folder = DATA_ROOT / "raw" / material
        assert mat_folder.is_dir(), f"Folder does not exist: {mat_folder}"
        if number is None:
            # load all sequences
            data_ret_d = {}
            csv_file_paths_l = list(mat_folder.glob(f"{material}*.csv"))
            for csv_file in tqdm.tqdm(sorted(csv_file_paths_l), desc=f"Loading data for {material}"):
                expected_cache_file = CACHE_ROOT / csv_file.with_suffix(".parquet").name
                if expected_cache_file.exists():
                    df = pd.read_parquet(expected_cache_file)
                else:
                    df = pd.read_csv(csv_file, header=None)
                    df.to_parquet(expected_cache_file, index=False)  # store cache
                data_ret_d[csv_file.stem] = df

        else:
            data_ret_d = {}
            for suffix in list("BHT"):
                filepath = mat_folder / f"{material}_{number}_{suffix}.csv"
                cached_filepath = CACHE_ROOT / filepath.with_suffix(".parquet").name
                if cached_filepath.exists():
                    df = pd.read_parquet(cached_filepath)
                else:
                    assert filepath.exists(), f"File does not exist: {filepath}"
                    df = pd.read_csv(filepath, header=None)
                    df.to_parquet(cached_filepath, index=False)  # store cache
                data_ret_d[filepath.stem] = df
    return data_ret_d


def book_keeping(logs_d: Dict):
    exp_id = str(uuid4())[:8]
    mat = logs_d.get("material", "unknown_material")
    logs_root = EXPERIMENT_LOGS_ROOT / f"{mat}_{exp_id}"
    logs_root.mkdir(parents=True, exist_ok=True)
    # store predictions and ground truth
    for l_key, l_v in logs_d.items():
        if l_key.startswith("predictions_MS"):
            seq_i = l_key.split("_")[-1]

            pd.DataFrame(l_v).to_parquet(
                logs_root / f"seed_{logs_d['seed']}_seq_{seq_i}_preds.parquet",
                index=False,
            )

            pd.DataFrame(logs_d[f"ground_truth_MS_{seq_i}"]).to_parquet(
                logs_root / f"seed_{logs_d['seed']}_seq_{seq_i}_gt.parquet",
                index=False,
            )
    # store trends
    pd.DataFrame(
        np.column_stack([logs_d["loss_trends_train"], logs_d["loss_trends_val"]]), columns=["train", "val"]
    ).to_parquet(logs_root / f"seed_{logs_d['seed']}_loss_trends.parquet", index=False)

    # store model state_dict (pytorch)
    if "model_state_dict" in logs_d:
        torch_save_path = logs_root / f"{mat}_{exp_id}.pt"
        torch.save(logs_d["model_state_dict"], torch_save_path)


def get_train_val_test_pandas_dicts(
    data_dict: dict[str, pd.DataFrame] = None,
    material_name: str = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 12,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    if data_dict is None and material_name is None:
        raise ValueError("Either data_dict or material_name must be provided.")

    if data_dict is None:
        data_dict = load_data_into_pandas_df(material=material_name)

    mat_set = MaterialSet.from_pandas_dict(data_dict)

    train_set, val_set, test_set = mat_set.split_into_train_val_test(
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    data_train = train_set.to_pandas_dict()
    data_val = val_set.to_pandas_dict()
    data_test = test_set.to_pandas_dict()

    return data_train, data_val, data_test


def load_hdf5_pretest_data(
    mat: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns B, T, H_init, H_true, msks_scenarios_N_tup where
    H_init has NaNs for unknown samples, and msks_scenarios_N_tup is a tuple of boolean masks for each
    scenario of unknown samples count.
    B, T, H_init, H_true are all of shape (num_time_series, num_time_steps)."""
    pretest_root = PRETEST_DATA_ROOT / f"{mat}"
    with h5py.File(pretest_root / f"{mat}_Testing_Padded.h5", "r") as f:
        B = f["B_seq"][:]
        H_init = f["H_seq"][:]
        T = f["T"][:]
    with h5py.File(pretest_root / f"{mat}_Testing_True.h5", "r") as f:
        H_true = f["H_seq"][:]
    unknowns_N = np.isnan(H_init).sum(axis=1)
    unknown_samples_variants, counts = np.unique(unknowns_N, return_counts=True)
    assert len(unknown_samples_variants) == 3, "Expecting 3 variants of unknown samples"
    msk_scenario_0 = unknowns_N == unknown_samples_variants[0]
    msk_scenario_1 = unknowns_N == unknown_samples_variants[1]
    msk_scenario_2 = unknowns_N == unknown_samples_variants[2]
    print(f"Scenario counts: {counts}")
    return B, T, H_init, H_true, (msk_scenario_0, msk_scenario_1, msk_scenario_2)
