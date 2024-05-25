# pulse-project
This repository contains the codes for all the model training and experiments performed in the paper Path Signature-Based Life Prognostics of Li-ion Battery Using Pulse Test Data.

## Folder analysis
1. `config` contains the model configuration file
1. `steps` contains the model and experiment steps
1. `pipelines` contains the model and experiment pipelines
1. `utils` contains the custom modules for training model and data exploration
1. `train_test_cells` contains the csv files of the names of the cells in the train and test splits used in this study (there is an option to use this for reproducibilty)

## Set up for running locally
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/pulse-project.git
    ```
1. Navigate to the root folder, i.e., `pulse-project` and create a python virtual environment (note that Python 3.10 was used in this project) by running
    ```
    python3.10 -m venv .venv
    ``` 
1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Upgrade `pip` by running 
   ```
   pip install --upgrade pip
   ``` 
1. Prepare all modules by running
    ```
    pip install -e .
    ```
1. Install all the required Python libraries by running 
    ```
    pip install -r requirements.txt
    ```
1. Create a folder named `noah_raw_data` in the root folder. Download the refined data from https://acdc.alcf.anl.gov/mdf/detail/camp_2023_v3.5/
   and put them in this folder.

1. Create folders named `data`, `plots` and `models` in the root folder to store the generated experimental data, figures and models respectively.

1. Start running entry points (`run_train.py`, `run_experiment.py`, `run_val.py`, and `run_plot.py` for the model training, experiment, leave-one-group-out cross-validation, and plotting pipelines respectively) with their respective arguments as CLI. For instance to train the end of life (eol) model using the proposed train-test cell splits given that the data has not been loaded, run:
    ```
    python run_train.py --not-loaded --model-type eol
    ```
    To see all the arguments or options available to an entry point, e.g., for training pipeline entry point run:
    ```
    python run_train.py --help
    ```
1. When you are done experimenting, deactivate the virtual environment by running
    ```
    deactivate
    ``` 

_This project is available for all under the [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode) licence._
