from batdata.data import BatteryDataset
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, List
from utils import generic_helper
import importlib

importlib.reload(generic_helper)


def get_cell_cathode_group_nominal_capacity(path_to_cell: str) -> Tuple[str, float]:
    """
    Get the cathode group to which a cell belong
    as well as the nominal capacity.

    Args:
    ----
          path_to_cell: absolute path to the cell's data

    Returns:
    -------
            corresponding cathode group- and nominal capacity
    """
    batdata = BatteryDataset.from_batdata_hdf(path_or_buf=path_to_cell)

    metadata = dict(batdata.metadata)
    metadata = dict(metadata["battery"])

    return dict(metadata["cathode"])["name"], metadata["nominal_capacity"]


def load_h5_columns_needed(
    path_to_cell: str, return_all: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function load the .h5 data into pandas dataframe
    using the battery-data-toolkit library (https://pypi.org/project/battery-data-toolkit/)

    Args:
    ----
          path_to_cell: path to the cell in question
          return_all: whether to return all columns or selected columns

    Returns:
    -------
           a tuple of dataframes: pulse data and summary data
    """

    batdata = BatteryDataset.from_batdata_hdf(path_or_buf=path_to_cell)

    # get the raw_data
    raw_data = batdata.raw_data
    summary_data = batdata.cycle_stats

    # drop nan from each data
    raw_data.dropna(axis=0, inplace=True)
    summary_data.dropna(axis=0, inplace=True)

    if return_all:
        return raw_data, summary_data

    # extract columns needed from raw data
    cols_needed_raw = [
        "cycle_number",
        "test_time",
        "state",
        "current",
        "voltage",
        "method",
    ]
    raw_data = raw_data[cols_needed_raw]

    # extract only pulses, charging and discharging profiles
    raw_data = raw_data[
        (raw_data["method"] == "pulse")
        & (raw_data["state"].isin(["charging", "discharging"]))
    ]

    # extract columns needed from summary data
    cols_needed_summary = ["cycle_number", "discharge_capacity"]
    summary_data = summary_data[cols_needed_summary].copy()

    # discharge capacity is logged as negative capacity; find abs of it
    summary_data.loc[:, "discharge_capacity"] = summary_data["discharge_capacity"].abs()

    return raw_data, summary_data


def remove_rest_profile_from_pulse(
    pulse_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Here we remove rest time from the pulse data.
    """

    current = pulse_data["current"].values
    voltage = pulse_data["voltage"].values

    time_diff = np.diff(pulse_data["test_time"].values)
    odd_indices = np.where(time_diff >= time_diff.min() + 1)[0]

    current = [current[i] for i in range(len(current)) if i not in odd_indices + 1]
    voltage = [voltage[i] for i in range(len(voltage)) if i not in odd_indices + 1]

    time_diff = [time_diff[i] for i in range(len(time_diff)) if i not in odd_indices]
    time = np.cumsum(time_diff)
    time = np.insert(time, 0, 0.0)

    return time, np.array(current), np.array(voltage)


def get_structured_data(path_to_files: List[str]) -> dict:
    """
    Here, this function takes a list of paths to the refined
    data from (https://acdc.alcf.anl.gov/mdf/detail/camp_2023_v3.5/),
    and creates a structured data containing pulse measurements of
    voltage, current, and time; summary data of cycles and the
    corresponding capacities.

    Args:
    ----
         path_to_files: a list of paths to dowloaded refined data

    Return:
    ------
          a dictionary of structured data
    """

    structured_data = {}

    odd_cells = [  # cells with irregular capacity curves (obtained by inspection, can be plotted to see)
        "batch_B7A_cell_1",
        "batch_B7A_cell_4",
        "batch_B7A_cell_2",
        "batch_B7A_cell_8",
        "batch_B7A_cell_6",
        "batch_B7A_cell_5",
        "batch_B7A_cell_7",
        "batch_B7A_cell_3",
    ]
    for path in path_to_files:
        cell_name = (path.split("/")[-1]).split(".")[0]  # extract the actual cell name from path

        raw_data, summary_data = load_h5_columns_needed(
            path_to_cell=path, return_all=False
        )

        pulses = {}

        for cyc, group in raw_data.groupby("cycle_number"):
            time, current, voltage = remove_rest_profile_from_pulse(pulse_data=group)
            pulses[cyc] = {"time": time, "current": current, "voltage": voltage}

        cycle = summary_data["cycle_number"].values
        capacity = summary_data["discharge_capacity"].values

        if cell_name in odd_cells:
            max_cap_index = np.argmax(capacity)
            capacity = capacity[max_cap_index:]
            cycle = np.arange(len(capacity)) + 1

        cathode_group, _ = get_cell_cathode_group_nominal_capacity(path_to_cell=path)

        filtered_capacity = sp.signal.medfilt(volume=capacity, kernel_size=5)

        filtered_capacity = sp.signal.savgol_filter(
            x=filtered_capacity, window_length=50, polyorder=1
        )

        isotonic_reg = IsotonicRegression(increasing=False)
        filtered_capacity = isotonic_reg.fit_transform(cycle, filtered_capacity)

        nominal_capacity = np.median(filtered_capacity[:10])
        end_of_life_bool = capacity >= 0.8 * nominal_capacity
        end_of_life = len(capacity[end_of_life_bool])

        summary = {
            "cycle": cycle,
            "capacity": capacity,
            "filtered_capacity": filtered_capacity,
            "cathode_group": cathode_group,
            "nominal_capacity": nominal_capacity,
            "end_of_life": end_of_life,
        }

        structured_data[cell_name] = {"pulses": pulses, "summary": summary}

    return structured_data


def get_unique_cathode_groups(structured_data: dict) -> np.ndarray:
    cathode_groups = [
        structured_data[k]["summary"]["cathode_group"] for k in structured_data
    ]
    return np.unique(cathode_groups)
