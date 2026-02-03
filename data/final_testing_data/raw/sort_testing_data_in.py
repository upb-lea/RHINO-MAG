"""Manipulates the downloaded files so that the data_management can work with it properly.

Assumes the following folder setup:

    ```text
    └── data/final_testing_data/raw
        ├── Material A - 3C92/
        ├── Material B - 3C95/
        ├── Material C - FEC007/
        ├── Material D - FEC014/
        ├── Material E - T37/
        └── sort_raw_data.py
    ```

"""

import pathlib


RAW_DATA_PATH = pathlib.Path(__file__).parent
for folder_path in RAW_DATA_PATH.glob("*"):
    if folder_path.is_file():
        continue
    elif "Material " in str(folder_path.name):

        material_name = str(folder_path.name).split("-")[-1][1:]

        target_material_name = str(folder_path.name).split("Material ")[-1][0]

        for file_name in folder_path.glob("*.csv"):
            target_name = target_material_name + str(file_name.name).split(material_name)[-1]
            print(f"Renaming folder '{folder_path.name}' to '{target_name}'.")
            file_name.rename(folder_path / target_name)

        print(f"Renaming folder '{folder_path.name}' to '{target_material_name}'.")
        folder_path.rename(folder_path.parent / target_material_name)


print("Succesfully sorted raw testing data in. Data sets are ready to use.")
