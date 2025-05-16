import tqdm
import pandas as pd
import pickle

import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path

from mc2.utils.data_inspection import load_and_process_single_from_full_file_overview


DATA_ROOT = Path(__file__).parent.parent / "data"
CACHE_ROOT = DATA_ROOT / "cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


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
            file_overview,
            material_name=material_name,
            data_type=["B", "T", "H"],
            frequency=[frequency],
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
    def load_from_raw_data(
        cls, file_overview: pd.DataFrame, material_name: str, frequencies: list[float] | jax.Array
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
        cls, file_overview: pd.DataFrame, material_names: list[str], frequencies: list[float] | jax.Array
    ) -> "DataSet":
        """Load a DataSet from raw data."""
        material_sets = []

        for material_name in tqdm.tqdm(material_names):
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


def load_data_into_pandas_df(material: str = None, number: int = None, training: bool = True, n_rows: int = None):
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
            for csv_file in tqdm.tqdm(sorted(csv_file_paths_l)):
                expected_cache_file = CACHE_ROOT / csv_file.with_suffix(".parquet")
                if (expected_cache_file.exists()):
                    df = pd.read_parquet(expected_cache_file)
                else:
                    df = pd.read_csv(csv_file, header=None)
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
                    df.to_parquet(cached_filepath)  # store cache
                data_ret_d[filepath.stem] = df
    return data_ret_d
