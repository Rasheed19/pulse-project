import importlib
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Callable
from utils import generic_helper

importlib.reload(generic_helper)


def get_data_for_eol_prediction(
    structured_data: dict,
    signature_depth: int,
    threshold: float,
    *,
    with_cathode_groups: bool = False
) -> np.ndarray:
    """
    This function creates features (for end of life
    prediction) from a structured data
    containing pulse time, current, and voltage.

    args:
    ----
        structured_data: a dictionary containing pulse and summary data
        signature_depth: depth of path signature
        threshold:       time threshold for the current and voltage profiles
        randomness:      whether to use a random pulse testing for feature extraction
        with_cathode_groups: either to return an array of cathode groups alongside with
                             the main target
    returns:
    -------
            numpy nd array of extracted features.
    """

    extracted_features = []
    extracted_eol = []
    extracted_cathode_groups = []

    for value in structured_data.values():
        pulse_list = list(value["pulses"].keys())
        pulse = pulse_list[0]

        extracted_features.append(
            generic_helper.get_path_signatures(
                time=value["pulses"][pulse]["time"],
                current=value["pulses"][pulse]["current"],
                voltage=value["pulses"][pulse]["voltage"],
                signature_depth=signature_depth,
                threshold=threshold,
            ).tolist()
        )
        extracted_eol.append(value["summary"]["end_of_life"])

        if with_cathode_groups:
            extracted_cathode_groups.append(value["summary"]["cathode_group"])

    if with_cathode_groups:
        return (
            np.array(extracted_features),
            np.array(extracted_eol),
            np.array(extracted_cathode_groups),
        )

    return np.array(extracted_features), np.array(extracted_eol)


def get_data_for_rul_prediction(
    structured_data: dict,
    signature_depth: int,
    threshold: float,
    *,
    with_cathode_groups: bool = False
) -> tuple:
    X = []
    y = []
    y_cathode_groups = []

    for value in structured_data.values():
        for pulse in value["pulses"]:
            if pulse < value["summary"]["end_of_life"]:
                X.append(
                    generic_helper.get_path_signatures(
                        time=value["pulses"][pulse]["time"],
                        current=value["pulses"][pulse]["current"],
                        voltage=value["pulses"][pulse]["voltage"],
                        signature_depth=signature_depth,
                        threshold=threshold,
                    ).tolist()
                )
                y.append(value["summary"]["end_of_life"] - pulse)
                if with_cathode_groups:
                    y_cathode_groups.append(value["summary"]["cathode_group"])

    X = np.array(X)
    y = np.array(y)
    y_cathode_groups = np.array(y_cathode_groups)

    # for reproducibility
    random.seed(42)

    # shuffle the entries
    shuffled_indices = np.arange(X.shape[0])
    random.shuffle(shuffled_indices)

    if with_cathode_groups:
        return (
            X[shuffled_indices],
            y[shuffled_indices],
            y_cathode_groups[shuffled_indices],
        )

    return X[shuffled_indices], y[shuffled_indices]


def get_data_for_classification(
    structured_data: dict, signature_depth: int, threshold: float
) -> tuple:
    X = []
    y = []

    for value in structured_data.values():
        for pulse in value["pulses"]:
            temp_features = generic_helper.get_path_signatures(
                time=value["pulses"][pulse]["time"],
                current=value["pulses"][pulse]["current"],
                voltage=value["pulses"][pulse]["voltage"],
                signature_depth=signature_depth,
                threshold=threshold,
            )

            if len(temp_features) > 0:
                if pulse > value["summary"]["end_of_life"]:
                    y.append(0)  # negative class: passed eol
                else:
                    y.append(1)  # positive calss: has not passed eol

                # X.append(
                #     generic_helper.get_path_signatures(
                #         time=value["pulses"][pulse]['time'],
                #         current=value["pulses"][pulse]['current'],
                #         voltage=value["pulses"][pulse]['voltage'],
                #         signature_depth=signature_depth,
                #         threshold=threshold
                # ).tolist())

                X.append(temp_features.tolist())

    X = np.array(X)
    y = np.array(y)

    # for reproducibility
    random.seed(42)

    # shuffle the entries
    shuffled_indices = np.arange(X.shape[0])
    random.shuffle(shuffled_indices)

    return X[shuffled_indices], y[shuffled_indices]


def processing_pipeline():
    return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])


class FeaturesTargetExtractor:
    """
    This class aims to transform raw data to features
    and targets that can be used for prediction.
    """

    def __init__(
        self,
        signature_depth: int,
        threshold: float,
        trans_func: Callable[[dict, int], tuple],
    ):
        self.signature_depth = signature_depth
        self.threshold = threshold
        self.trans_func = trans_func
        self.imputer_scaler_pipeline = processing_pipeline()

    def fit_transform(self, structured_data: dict):
        X, y = self.trans_func(
            structured_data=structured_data,
            signature_depth=self.signature_depth,
            threshold=self.threshold,
        )
        X = self.imputer_scaler_pipeline.fit_transform(X)

        return X, y

    def transform(self, structured_data: dict):
        X, y = self.trans_func(
            structured_data=structured_data,
            signature_depth=self.signature_depth,
            threshold=self.threshold,
        )

        return self.imputer_scaler_pipeline.transform(X), y
