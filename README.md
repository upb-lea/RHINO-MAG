# magnet-challenge-2

This is the contribution of Team "Siegen and Paderborn" to the MagNet Challenge 2 (MC2).

Official site for the second magnet challenge https://github.com/minjiechen/magnetchallenge-2

## Installation:
- use `python3.11` (specifically python3.11.11, should not make a difference though)
- `git clone git@github.com:upb-lea/magnet-challenge-2.git` the repo to your PC or workstation
- create a fresh virtual enviornment (e.g., `python -m venv mc2-venv`)
- activate it (e.g., `source path/to/venv/mc2-venv/bin/activate` (linux) or `.\path\to\venv\mc2-venv\bin\activate.sh` (windows))
- navigate to the downloaded repo
- install it with `pip install -e .` (this is to have installed as an editable site package)
- now you should be able to import `mc2` from within your venv
 
## Repository structure:

The repository is structured as follows:

- `data/` holds the material data, stored models, experiment logs, etc.
    - `data/raw/` should contain the unprocessed material folders (e.g., `raw/A/A_1_B.csv`). Upon first load, a cached version of the data will be stored in `data/cache/`
    - `data/single_file_models` contains the models as a single `.eqx` file
- `dev/` holds a variety of jupyter notebooks, these will generally not be maintained, i.e., they might work, but could also be outdated
- `examples/` holds example notebooks that will always be maintained
    - `examples/model_inspection.ipynb` shows how to load models and data, and how to evaluate and visualize the performance of models
    - `examples/model_training.ipynb` shows how to train models
    - `examples/final_test_data_evaluation.ipynb` shows how to apply the models to the test data provided by the MC2 hosts
- `mc2/` holds the source code and training scripts:
    - `mc2/features` implemenation of features
    - `mc2/model_interface` interface for the models to interact with the material data
    - `mc2/models/` these model implementations could generally be used for different tasks, and the correct model interface is necessary so that they may properly interact with the material data set
    - `mc2/runners/` trainings scripts
    - `mc2/training` some training specific utilies
    - `mc2/utils/` some general utilities regarding model evaluation, plotting, processing of test data, etc.
    - `mc2/data_management.py` general management of data sets (e.g. loading from disk, splitting into traing, eval, test)
    - `mc2/losses.py` implementation of the training loss functions
    - `mc2/metrics.py` implementation of evaluation metrics
