import glob
import pathlib
from typing import List

import numpy as np
import pandas as pd


## Start inspect files ############################################################################


def get_available_material_names(raw_data_path: pathlib.Path) -> List[str]:
    """Get all available material names from the given data path."""
    all_file_paths = glob.glob(str(raw_data_path / pathlib.Path("*") / pathlib.Path("*.csv")))
    all_materials = []

    for material_path in all_file_paths:
        material_name = material_path.split("/")[-1].split("_")[0]
        if material_name not in all_materials:
            all_materials.append(material_name)

    return all_materials


def get_file_overview(raw_data_path: pathlib.Path, material_names: List[str], frequencies: List[float]) -> pd.DataFrame:
    """Prepare an overview over all files available at the given data path for the specified materials."""
    all_file_paths = []

    for material_name in material_names:
        material_file_paths = glob.glob(str(raw_data_path / pathlib.Path(material_name) / pathlib.Path("*.csv")))
        all_file_paths += material_file_paths

    file_overview = {
        "path": [],
        "material_name": [],
        "data_type": [],
        "frequency": [],
        "set_idx": [],
    }

    for material_path in all_file_paths:
        material_name = material_path.split("/")[-1].split("_")[0]
        set_idx = int(material_path.split("_")[1])
        data_type = material_path.split("_")[-1].split(".")[0]

        file_overview["path"].append(material_path)
        file_overview["material_name"].append(material_name)
        file_overview["data_type"].append(data_type)
        file_overview["frequency"].append(frequencies[set_idx - 1])
        file_overview["set_idx"].append(set_idx)

    return pd.DataFrame.from_dict(file_overview)


def filter_file_overview(
    file_overview: pd.DataFrame,
    material_name: str | List[str] | None = None,
    data_type: str | List[str] | None = None,
    set_idx: int | List[int] | None = None,
    frequency: float | List[int] | None = None,
) -> pd.DataFrame:
    """Filter the file overview by material name and/or data type.

    List inputs are treated as OR conditions.
    """

    for key, filter_value in {
        "material_name": material_name,
        "data_type": data_type,
        "set_idx": set_idx,
        "frequency": frequency,
    }.items():
        if filter_value is not None:
            if not isinstance(filter_value, list):
                file_overview = file_overview[file_overview[key] == filter_value]
            else:
                if filter_value is not None:
                    file_overview = file_overview[file_overview[key].isin(filter_value)]
        else:
            continue

    return file_overview


## End inspect files ##############################################################################


## Start loading data #############################################################################


def load_single_file(file_path):
    return np.genfromtxt(file_path, delimiter=",")


def load_data_raw_from_file_overview(file_overview: pd.DataFrame) -> List[np.ndarray]:
    """Load all files in the file overview and return a list with the data."""
    all_data = []
    for _, row in file_overview.iterrows():
        all_data.append(load_single_file(row["path"]))
    return all_data


def load_and_process_single_from_single_set_overview(single_set_overview: pd.DataFrame) -> dict:
    raw_data_dict = single_set_overview.to_dict()

    processed_data_dict = {}
    processed_data_dict["raw_paths"] = raw_data_dict["path"]

    for key in ["material_name", "frequency", "set_idx"]:
        proposed_value = list(raw_data_dict[key].values())[0]
        assert all(
            [proposed_value == value for value in raw_data_dict[key].values()]
        ), f"Key '{key}' has multiple different values in the specified overview. This function is designed to work with a single set of data only."
        processed_data_dict[key] = proposed_value

    for data_type in ["H", "B", "T"]:
        values = load_data_raw_from_file_overview(filter_file_overview(single_set_overview, data_type=data_type))

        assert len(values) == 1
        processed_data_dict[data_type] = values[0]

    return processed_data_dict


def load_and_process_single_from_full_file_overview(
    file_overview: pd.DataFrame,
    material_name: str | List[str] | None = None,
    data_type: str | List[str] | None = None,
    set_idx: int | List[int] | None = None,
    frequency: float | List[int] | None = None,
) -> dict:
    return load_and_process_single_from_single_set_overview(
        filter_file_overview(file_overview, material_name, data_type, set_idx, frequency)
    )


## End loading data ###############################################################################
