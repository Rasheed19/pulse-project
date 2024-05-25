import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .generic_helper import get_path_signatures


def get_data_for_eol_prediction(
    structured_data: dict,
    signature_depth: int,
    threshold: float | tuple[float, float],
    with_cathode_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:

    extracted_features = []
    extracted_eol = []
    extracted_cathode_groups = []

    for value in structured_data.values():
        pulse_list = list(value["pulses"].keys())
        pulse = pulse_list[0]

        extracted_features.append(
            get_path_signatures(
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
    threshold: float | tuple[float, float],
    with_cathode_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = []
    y = []
    y_cathode_groups = []

    for value in structured_data.values():
        for pulse in value["pulses"]:
            if pulse < value["summary"]["end_of_life"]:
                X.append(
                    get_path_signatures(
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
    structured_data: dict,
    signature_depth: int,
    threshold: float | tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for value in structured_data.values():
        for pulse in value["pulses"]:
            temp_features = get_path_signatures(
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

                X.append(temp_features.tolist())

    X = np.array(X)
    y = np.array(y)

    # for reproducibility
    random.seed(42)

    # shuffle the entries
    shuffled_indices = np.arange(X.shape[0])
    random.shuffle(shuffled_indices)

    return X[shuffled_indices], y[shuffled_indices]


def inputer_scaler_pipeline() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
