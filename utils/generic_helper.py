import numpy as np
import random
import os
import pickle
import iisignature
import esig
import yaml
from typing import Tuple, Any


def data_splitting(
    data: dict,
    test_ratio: float,
    cathode_group: str = None,
) -> Tuple[dict, dict]:
    """
    Function to split data into train and test sets.
    It ensures that each split has at least a cell
    from each cathode chemistries.

    Args:
    ----
        data:          data to be split
        cathode_group: group of cells based on cathode material
        test_ratio:    fraction of test set

    Returns:
    -------
            train, test splits
    """

    random.seed(42)

    # get train ratio
    train_ratio = 1.0 - test_ratio

    if cathode_group is not None:
        # get array of all cells corresponding to the given cathode group
        data = {
            k: data[k]
            for k in data
            if data[k]["summary"]["cathode_group"] == cathode_group
        }

    cells = np.array(list(data.keys()))
    num_cells = cells.shape[0]

    # shuffle the cells
    random.shuffle(cells)

    if cathode_group == " Li1.35Ni0.33Mn0.67O2.35":  # this cathode set has only 2 cells
        num_train_cells = 1

    else:
        num_train_cells = int(train_ratio * num_cells)

    # get the training cells
    train_cells = cells[:num_train_cells]

    # get test cells
    test_cells = cells[num_train_cells:]

    return ({k: data[k] for k in train_cells}, {k: data[k] for k in test_cells})


def bring_splits_together(
    data: dict, cathode_groups: list, test_ratio: float
) -> Tuple[dict, dict]:
    train, test = {}, {}

    for group in cathode_groups:
        tr, te = data_splitting(data=data, cathode_group=group, test_ratio=test_ratio)

        train |= tr
        test |= te

    return shuffle_dictionary(train), shuffle_dictionary(test)


def shuffle_dictionary(dictionary: dict) -> dict:
    # for reproducibility
    random.seed(42)
    keys = list(dictionary.keys())
    random.shuffle(keys)

    return {key: dictionary[key] for key in keys}


def read_data(path: str, fname: str) -> Any:
    with open(os.path.join(path, fname), "rb") as fp:
        data = pickle.load(fp)

    return data


def dump_data(data: Any, path: str, fname: str) -> None:
    with open(os.path.join(path, fname), "wb") as fp:
        pickle.dump(data, fp)


def get_sig_convention(dimension: int, depth: int) -> np.ndarray:
    raw_convention = esig.sigkeys(dimension=dimension, depth=depth)
    raw_convention = raw_convention.split(" ")[2:]

    sig_convention = []

    for conv in raw_convention:
        conv = conv.replace("(", "{")
        conv = conv.replace(")", "}")
        sig_convention.append(f"$S^{conv}$")

    return np.array(sig_convention)


def get_path_signatures(
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    signature_depth: int = 2,
    threshold: float = None,
) -> np.ndarray:
    """
    Function that calculates the signature of the
    path P = {time, current, voltage} up to a given depth
    or level.

    Args:
    ----
         time (numpy.ndarray):           array of time elapsed
         current (numpy.ndarray):        array of pulse current
         voltage (numpy.ndarray):        array of pulse voltage
         signature_depth (positive int): level of signature
         threshold (float):              time threshold in sec

    Returns:
    -------
            numpy array of path signatures.
    """

    if threshold is None:
        path = np.stack((time, current, voltage), axis=-1)

    else:
        if isinstance(threshold, int):
            threshold_bool = time <= threshold
        else:
            threshold_bool = (time >= threshold[0]) & (time <= threshold[1])

        IS_EMPTY = 0
        if time[threshold_bool].shape[0] == IS_EMPTY:
            return np.array([])

        time = time[threshold_bool] - time[threshold_bool].min()
        current = current[threshold_bool]
        voltage = voltage[threshold_bool]
        path = np.stack((time, current, voltage), axis=-1)

    return iisignature.sig(path, signature_depth)


def jaccard_similarity(list1: list, list2: list) -> float:
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def load_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data
