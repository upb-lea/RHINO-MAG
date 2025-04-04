# magnet-challenge-2
Official site for the second magnet challenge https://github.com/minjiechen/magnetchallenge-2

## Timeline (copied from official site):
- 05-01-2025 1-Page Letter of Intent Due with Signature 
- 06-01-2025 2-Page Concept Proposal Due
- 07-01-2025 Notification of Acceptance
- 08-01-2025 Expert Feedback on the Concept Proposal
- 11-01-2025 Preliminary Submission Due
- 11-01-2025 Testing Data for 5 New Materials Available
- 12-24-2025 Final Submission Due
- 03-01-2026 Winners Selected

## Installation:
- use `python3.11` (specifically python3.11.11, should not make a difference though)
- `git clone git@github.com:upb-lea/magnet-challenge-2.git` the repo to your PC or workstation
- create a fresh virtual enviornment (e.g., `python -m venv mc2-venv`)
- activate it (e.g., `source path/to/venv/mc2-venv/bin/activate` (linux) or `.\path\to\venv\mc2-venv\bin\activate.sh` (windows))
- navigate to the downloaded repo
- install it with `pip install -e .` (this is to have installed as an editable site package)
- now you should be able to import `mc2` from within your venv

## Getting Started:
The raw dataset is rougly 15 GB (compressed) and 30 GB (uncompressed).
It will not be uploaded, but it might make sense to put it into the 'LEA' network.
You can download the dataset from the official mc2 site and decompress it.
To use it properly:

- download and unzip dataset
- place all the data into `data/raw/`
- ensure that the `.csv` file lie direcly in the material folder, i.e., `raw/{material_name}/{file_name}.csv`
- go to `dev/data_inspection/inspect_raw_data.ipynb`
- run the notebook (you may have to change the path slightly)
- now you should have your `ten_mat_data.pickle` file in the `data/processed/` folder
- look into `dev/data_inspection/exploratory_data_analysis.ipynb` for loading and inspecting of the data
- look into `dev/model_tests/RNN_tests.ipynb` for a simple usage example with a basic RNN


> ⚠️: Note that EVERYTHING is essentially WIP and a lot more implementation and finetuning will be necessary and bugs might exist (even though I hope they don't.)
> So please look at the code critically.


  
