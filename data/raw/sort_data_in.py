"""Manipulates the downloaded files so that the data_management can work with it properly.

Assumes the following folder setup:

    ```text
    └── data/raw
        ├── Material A/
        ├── Material B/
        ├── Material C/
        ├── Material D/
        ├── Material E/
        ├── (optional further folders, e.g., 3C90, N49, ...)
        ├── ...
        ├── ...
        └── sort_raw_data.py
    ```

"""

import pathlib
import pandas


RAW_DATA_PATH = pathlib.Path(__file__).parent
for folder_path in RAW_DATA_PATH.glob("*"):
    if folder_path.is_file():
        continue
    elif "Material " in str(folder_path.name):
        target_name = str(folder_path.name).split("Material ")[-1]
        print(f"Renaming folder '{folder_path.name}' to '{target_name}'.")
        folder_path.rename(folder_path.parent / target_name)

B_mat_folder = (RAW_DATA_PATH / "B")
if B_mat_folder.is_dir():
    csv_file_paths = B_mat_folder.glob("*.csv")
    if len(list(csv_file_paths)) == 21:
        pass
    else:
        print("Restructuring files for Material 'B'.")
        for start, target in zip([5, 4, 3, 2], [7, 5, 4, 3]):
            for data_type in ["H", "B", "T"]:
                (B_mat_folder / f"B_{start}_{data_type}.csv").rename(B_mat_folder / f"B_{target}_{data_type}.csv")
        
        for freq_idx in [2, 6]:
            for data_type in ["H", "B", "T"]:
                file_name = (B_mat_folder / f"B_{freq_idx}_{data_type}.csv")
                with open(file_name, 'w'):
                    pass

print("Succesfully sorted raw data in. Data sets are ready to use.")