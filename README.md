# pulse-project
This repository contains the codes for all the experiments performed in the paper Prediction of RUL of Li-ion cells from a single HPPC test.

## Folder analysis
1. `config` contains the model configuration file
1. `experiments` contains the Jupyter notebooks of all experiments in sequential order. They must be run in that order especially those of 01 and 02
1. `utils` contains the custom modules for training model and data exploration

## Set up for running locally
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/pulse-project.git
    ```
1. Navigate to the root folder, i.e., `pulse-project` and create a python virtual environment by running
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

1. Start running jupyter notebooks in the `experiments` folder sequentially.

1. When you are done experimenting, deactivate the virtual environment by running
    ```
    deactivate
    ``` 

_Licence: [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode)_
