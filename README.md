# pulse-project
This repository contains the codes for all the model training and experiments performed in the paper Path Signature-Based Life Prognostics of Li-ion Battery Using Pulse Test Data.

## Set up
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/pulse-project.git
    ```
1. Navigate to the root folder, create a python virtual environment by running
    ```
    python -m venv .venv
    ``` 
    Note that Python 3.10 was used in this research.

1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Prepare all modules and required directories by running the following:
    ```
    make setup
    make create-required-dir
    ```
1. Download the refined data from https://acdc.alcf.anl.gov/mdf/detail/camp_2023_v3.5/ and put them in `noah_raw_data`.

## Usage
After setting up the project environment as instructed above, start running the provided entrypoints (`run_train.py`, `run_experiment.py`, `run_val.py`, and `run_plot.py` for the model training, experiment, leave-one-group-out cross-validation, and plotting pipelines respectively) with their respective arguments as CLI. For instance to train the end of life (eol) model using the proposed train-test cell splits given that the data has not been loaded to a Python dictionary, run:
```
python run_train.py --not-loaded --model-type eol
```
To see all the arguments or options available to any entry point, e.g., for training pipeline entrypoint run:
```
python run_train.py --help
```
