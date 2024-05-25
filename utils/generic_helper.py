import numpy as np
import pandas as pd
import random
import os
import pickle
import iisignature
import esig
import yaml
from typing import Any
from scipy.interpolate import interp1d
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from scipy import stats
from tqdm import tqdm
import shap
import logging

from .definitions import ROOT_DIR


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Times"], "size": MEDIUM_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.bottom": True,
        "ytick.left": True,
    }

    return rc_params


def data_splitting(
    data: dict,
    test_ratio: float,
    cathode_group: str = None,
) -> tuple[dict, dict]:
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
) -> tuple[dict, dict]:
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
    threshold: float | tuple[float, float] = None,
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


def metric_calculator_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
    }


def metric_calculator_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    mode: str = "score",
) -> dict[dict, float]:
    metrics = {
        "precision": precision_score(y_true=y_true, y_pred=y_pred),
        "recall": recall_score(y_true=y_true, y_pred=y_pred),
        "f1_score": f1_score(y_true=y_true, y_pred=y_pred),
        "roc_auc_score": roc_auc_score(y_true=y_true, y_score=y_pred_proba),
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
    }

    if mode == "score":
        return metrics

    elif mode == "error":
        return {key: 1 - value for key, value in metrics.items()}

    else:
        raise ValueError("mode can only be 'score' or 'error'")


def naive_confidence_interval(
    data: np.ndarray, confidence_level: float, percentile_interval: bool = False
) -> tuple[float, float]:
    """
    Calculate confidence interval via normality assumption
    or percentiles.

    Args:
    ----
         data:             data from which CI will be calculated
         confidence level: level of confidence
         percentile_interval: whether calculate percentile interval or not

    Returns:
    -------
            confidence interval
    """
    if percentile_interval:
        alpha_tail = (1 - confidence_level) / 2
        data_sorted = np.sort(data)
        n_bootstraps = len(data_sorted)

        return (
            data_sorted[int(alpha_tail * n_bootstraps)],
            data_sorted[int((1 - alpha_tail) * n_bootstraps)],
        )

    sample_mean = np.mean(data)
    # Use ddof=1 for sample standard deviation
    sample_std = np.std(data, ddof=1)

    tail_prob = (1 - confidence_level) / 2
    upper_z = stats.norm.ppf(1 - tail_prob)

    return sample_mean, upper_z * sample_std


def bootstrap_metric(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    n_bootstraps: int,
    *,
    predicted_proba: np.ndarray = None,
    model_type: str = "eol",
) -> np.ndarray:

    boot_dim = 2 if model_type in ["eol", "rul"] else 5

    bootstraps = np.zeros(shape=(n_bootstraps, boot_dim))
    sample_size = len(actual_values)

    for i in tqdm(range(n_bootstraps)):
        if model_type == "classification":
            sample_idx_ = np.random.randint(
                low=0, high=sample_size, size=sample_size
            )  # end value is excluded

            metrics = metric_calculator_classification(
                y_true=actual_values[sample_idx_],
                y_pred=predicted_values[sample_idx_],
                y_pred_proba=predicted_proba[sample_idx_],
                mode="score",
            )

            for j, value in enumerate(metrics.values()):
                bootstraps[i, j] = value

        elif model_type in ["eol", "rul"]:
            sample_idx_ = np.random.randint(
                low=0, high=sample_size, size=sample_size
            )  # end value is excluded

            metrics = metric_calculator_regression(
                y_true=actual_values[sample_idx_], y_pred=predicted_values[sample_idx_]
            )

            for j, value in enumerate(metrics.values()):
                bootstraps[i, j] = value

    return bootstraps


def empirical_cdf(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cumulative distribution function (CDF)
    from an array of data points.

    Args:
    ----
         data: the input data.

    Returns:
    -------
            x (numpy array): The unique data points sorted in ascending order.
            cdf (numpy array): The corresponding CDF values for each data point.
    """
    # Sort the data in ascending order
    sorted_data = np.sort(data)

    # Calculate the unique data points and their counts
    unique_data, counts = np.unique(sorted_data, return_counts=True)

    # Calculate the CDF values
    cdf = np.cumsum(counts) / len(data)

    return unique_data, cdf


def inverse_empirical_cdf(data: np.ndarray, probability: float) -> float:
    """
    Calculate the inverse empirical cumulative distribution
    function (CDF) from a list of data points.

    Args:
    ----
        data: the input data.

    Returns:
    -------
            inverse cdf corresponding to the probability
    """

    # Ensure the probability is within [0, 1]
    if 0 > probability > 1:
        raise ValueError("probabilty must be in the interval [0, 1]")

    # Calculate the empirical CDF
    x, cdf = empirical_cdf(data)

    # Create an interpolating function for the CDF
    cdf_interpolated = interp1d(
        cdf, x, kind="linear", fill_value=(x[0], x[-1]), bounds_error=False
    )

    # Use the inverse CDF function to find the corresponding value
    return cdf_interpolated(probability)


def pivotal_confidence_interval(
    estimate_on_all_samples: float, bootstraps: np.ndarray, alpha: float
) -> tuple[float, float]:

    bootstraps_shifted = bootstraps - estimate_on_all_samples
    return (
        estimate_on_all_samples
        - inverse_empirical_cdf(bootstraps_shifted, 1 - (alpha / 2)),
        estimate_on_all_samples - inverse_empirical_cdf(bootstraps_shifted, alpha / 2),
    )


def rounder(value: float, decimal_place: int = 2) -> float:
    return np.round(value, decimal_place)


def training_result_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RegressorMixin,
    alpha: float,
) -> tuple[pd.DataFrame, dict]:

    split_data = dict(train=(X_train, y_train), test=(X_test, y_test))

    prediction_data = {}
    metric_ci_data = pd.DataFrame(
        index=split_data.keys(),
        columns=[m + f" ({int(100*(1-alpha))}% CI)" for m in ["MAE", "RMSE"]],
    )
    for key, value in split_data.items():
        X, y = value
        prediction = model.predict(X)

        metric = metric_calculator_regression(y_true=y, y_pred=prediction)
        prediction_data[key] = {"actual": y, "prediction": prediction, "score": metric}

        # fix model, bootstrap predictions on data
        bootstraps = bootstrap_metric(
            actual_values=y, predicted_values=prediction, n_bootstraps=10_000
        )
        ci_list = [
            pivotal_confidence_interval(
                estimate_on_all_samples=metric[m],
                bootstraps=bootstraps[:, i],
                alpha=alpha,
            )
            for i, m in enumerate(metric.keys())
        ]
        metric_ci_data.loc[key, metric_ci_data.columns] = [
            f"{rounder(m)} {(rounder(ci[0]), rounder(ci[1]))}"
            for m, ci in zip(metric.values(), ci_list)
        ]

    return metric_ci_data, prediction_data


def training_result_clf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: ClassifierMixin,
    alpha: float,
) -> tuple[pd.DataFrame, dict]:

    split_data = dict(train=(X_train, y_train), test=(X_test, y_test))

    prediction_data = {}
    metric_ci_data = pd.DataFrame(
        index=split_data.keys(),
        columns=[
            m + f" ({int(100*(1-alpha))}% CI)"
            for m in ["precision", "recall", "f1_score", "roc_auc_score", "accuracy"]
        ],
    )
    for key, value in split_data.items():
        X, y = value
        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)

        metric = metric_calculator_classification(
            y_true=y,
            y_pred=prediction,
            y_pred_proba=prediction_proba[:, 1],
            mode="score",
        )

        prediction_data[key] = {
            "actual": y,
            "prediction": prediction,
            "prediction_prob": prediction_proba[:, 1],
            "score": metric,
        }

        # fix model, bootstrap predictions on data
        bootstraps = bootstrap_metric(
            actual_values=y,
            predicted_values=prediction,
            n_bootstraps=10_000,
            predicted_proba=prediction_proba[:, 1],
            model_type="classification",
        )
        ci_list = [
            pivotal_confidence_interval(
                estimate_on_all_samples=metric[m],
                bootstraps=bootstraps[:, i],
                alpha=alpha,
            )
            for i, m in enumerate(metric.keys())
        ]
        metric_ci_data.loc[key, metric_ci_data.columns] = [
            f"{rounder(m*100)} {(rounder(ci[0]*100), rounder(ci[1]*100))}"
            for m, ci in zip(metric.values(), ci_list)
        ]

    return metric_ci_data, prediction_data


def log_model_pipeline(
    pipeline: Pipeline,
    model: RegressorMixin | ClassifierMixin,
    model_type: str,
) -> None:
    # dump the pipeline
    dump_data(
        data=pipeline, path=f"{ROOT_DIR}/models", fname=f"{model_type}_pipeline.pkl"
    )

    # dump trained model
    dump_data(data=model, path=f"{ROOT_DIR}/models", fname=f"{model_type}_model.pkl")

    return None


def permutation_feature_importance(
    fitted_model: RegressorMixin | ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    scoring_function: str,
) -> np.ndarray:
    result = permutation_importance(
        estimator=fitted_model,
        X=X,
        y=y,
        scoring=scoring_function,
        n_repeats=10,
        random_state=42,
    )

    return result.importances_mean


def shap_feature_importance(fitted_model: object, X: np.ndarray) -> np.ndarray:
    explainer = shap.Explainer(fitted_model, X)
    shap_values = explainer(X)
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    return mean_shap_values


def cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: RegressorMixin | ClassifierMixin,
    scoring: dict[str, str],
) -> pd.DataFrame:
    cv_results = cross_validate(
        model, X_train, y_train, scoring=list(scoring.values()), cv=5
    )

    results_tabulated = pd.DataFrame(
        columns=["Mean", "Standard deviation"], index=scoring.keys()
    )
    results_tabulated["Mean"] = [
        abs(cv_results[f"test_{sc}"]).mean() for sc in scoring.values()
    ]
    results_tabulated["Standard deviation"] = [
        cv_results[f"test_{sc}"].std() for sc in scoring.values()
    ]

    return results_tabulated


class CustomFormatter(logging.Formatter):

    green = "\x1b[0;32m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def config_logger() -> None:

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )
