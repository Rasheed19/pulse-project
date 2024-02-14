import importlib
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Union, List
from scipy.interpolate import interp1d
import shap
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
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
from utils.definitions import ROOT_DIR
from utils import extraction, generic_helper
from scipy import stats
from tqdm import tqdm


importlib.reload(extraction)
importlib.reload(generic_helper)


def metric_calculator_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
    }


def metric_calculator_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    mode: str = "score",
) -> dict:
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
) -> Tuple[float, float]:
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
    mode: str = "regression",
) -> np.ndarray:

    boot_dim = 2 if mode == "regression" else 5

    bootstraps = np.zeros(shape=(n_bootstraps, boot_dim))
    sample_size = len(actual_values)

    for i in tqdm(range(n_bootstraps)):
        if mode == "classification":
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

        elif mode == "regression":
            sample_idx_ = np.random.randint(
                low=0, high=sample_size, size=sample_size
            )  # end value is excluded

            metrics = metric_calculator_regression(
                y_true=actual_values[sample_idx_], y_pred=predicted_values[sample_idx_]
            )

            for j, value in enumerate(metrics.values()):
                bootstraps[i, j] = value

    return bootstraps


def empirical_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[float, float]:

    bootstraps_shifted = bootstraps - estimate_on_all_samples
    return (
        estimate_on_all_samples
        - inverse_empirical_cdf(bootstraps_shifted, 1 - (alpha / 2)),
        estimate_on_all_samples - inverse_empirical_cdf(bootstraps_shifted, alpha / 2),
    )


def rounder(value: float, decimal_place: int = 2) -> float:
    return np.round(value, decimal_place)


def display_training_result(
    pipeline: extraction.FeaturesTargetExtractor,
    model: RegressorMixin,
    split_data: dict,
    alpha: float,
) -> Tuple[pd.DataFrame, dict]:

    prediction_data = {}
    metric_ci_data = pd.DataFrame(
        index=split_data.keys(),
        columns=[m + f" ({int(100*(1-alpha))}% CI)" for m in ["MAE", "RMSE"]],
    )
    for key, value in split_data.items():
        X, y = pipeline.transform(value)
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


def get_fitted_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict,
    *,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    problem_type: str = "regression",
) -> object:
    if problem_type == "regression":
        model = XGBRegressor(**model_params)
    elif problem_type == "classification":
        model = XGBClassifier(**model_params)
    else:
        raise ValueError(
            "Incorrect problem type: can either be 'regression' or 'classification'"
        )

    if X_val is None or y_val is None:
        return model.fit(X_train, y_train)

    fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}
    model.fit(X_train, y_train, **fit_params)

    return model


def display_training_result_clf(
    pipeline: extraction.FeaturesTargetExtractor,
    model: ClassifierMixin,
    split_data: dict,
    alpha: float,
) -> Tuple[pd.DataFrame, dict]:

    prediction_data = {}

    metric_ci_data = pd.DataFrame(
        index=split_data.keys(),
        columns=[
            m + f" ({int(100*(1-alpha))}% CI)"
            for m in ["precision", "recall", "f1_score", "roc_auc_score", "accuracy"]
        ],
    )
    for key, value in split_data.items():
        X, y = pipeline.transform(value)
        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)

        metric = metric_calculator_classification(
            y_true=y,
            y_pred=prediction,
            y_pred_proba=prediction_proba[:, 1],
            mode="score",
        )

        prediction_data[key] = {"actual": y, "prediction": prediction, "score": metric}

        # fix model, bootstrap predictions on data
        bootstraps = bootstrap_metric(
            actual_values=y,
            predicted_values=prediction,
            n_bootstraps=10_000,
            predicted_proba=prediction_proba[:, 1],
            mode="classification",
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


def permutation_feature_importance(
    fitted_model: Union[RegressorMixin, ClassifierMixin],
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


def feature_importance_analysis(
    model_feature_dict: dict, *, mode: str = "xgboost", validation_set: dict = None
) -> dict:
    """
    Calculate feature importance for all the
    fitted models.

    Args:
    ----
        model_feature_dict: dict of fitted models and features,
        mode:               "xgboost" to use xgboost feature
                            importance or "permutation" to use
                            permutation importance or "shap"
                            to use shap
        validation_set:     validation set to use with
                            permutation importance, if chosen
    Returns:
    -------
        dict of feature importance for each fitted model
    """

    # Create a lambda function to scale importance values to the interval [0, 1]
    def scaler(x):
        return (x - x.min()) / (x.max() - x.min())

    analysis_result = {}

    for key, value in model_feature_dict.items():
        if mode == "permutation" and validation_set is not None:
            X, y = value["pipeline"].transform(validation_set)

            if key == "classification":
                feature_importance = permutation_feature_importance(
                    fitted_model=value["fitted_model"], X=X, y=y, scoring_function="f1"
                )
            else:
                feature_importance = feature_importance = (
                    permutation_feature_importance(
                        fitted_model=value["fitted_model"],
                        X=X,
                        y=y,
                        scoring_function="neg_mean_squared_error",
                    )
                )

            feature_importance = scaler(feature_importance)

        elif mode == "xgboost":
            feature_importance = (
                scaler(value["fitted_model"].feature_importances_)
                if key == "classification"
                else scaler(value["fitted_model"].regressor_.feature_importances_)
            )

        elif mode == "shap" and validation_set is not None:
            X, y = value["pipeline"].transform(validation_set)
            feature_importance = scaler(
                shap_feature_importance(
                    fitted_model=(
                        value["fitted_model"]
                        if key == "classification"
                        else value["fitted_model"].regressor_
                    ),
                    X=X,
                )
            )

        sorted_indices_ = np.argsort(feature_importance)

        analysis_result[key] = {
            "features": value["features"][sorted_indices_],
            "importance": feature_importance[sorted_indices_],
        }

    return analysis_result


def train_model(
    train_data: dict,
    signature_depth: int,
    threshold: float,
    param_grid: dict,
    problem_type: str,
    trans_func: Callable[[dict, int], tuple],
    scorer: str,
    cv: int = 5,
) -> Tuple[
    extraction.FeaturesTargetExtractor,
    dict,
    Union[RegressorMixin, ClassifierMixin],
    float,
    int,
]:
    pipeline = extraction.FeaturesTargetExtractor(
        signature_depth=signature_depth, threshold=threshold, trans_func=trans_func
    )
    X_train, y_train = pipeline.fit_transform(structured_data=train_data)

    if problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=XGBRegressor(),
            transformer=QuantileTransformer(
                n_quantiles=int(X_train.shape[0] * (cv - 1) / cv),
                output_distribution="normal",
            ),
        )
    elif problem_type == "classification":
        estimator = XGBClassifier()
    else:
        raise ValueError(
            "Incorrect problem type: can either be 'regression' or 'classification'"
        )

    grid_search = GridSearchCV(
        estimator=estimator, param_grid=param_grid, scoring=scorer, refit=True, cv=cv
    )

    grid_search.fit(X=X_train, y=y_train)

    return (
        pipeline,
        grid_search.best_params_,
        grid_search.best_estimator_,
        grid_search.best_score_,
        grid_search.cv_results_["std_test_score"][grid_search.best_index_],
    )


def effect_time_threshold(
    train_data: dict,
    signature_depth: int,
    param_grid: dict,
    problem_type: str,
    trans_func: Callable[[dict, int], tuple],
    scorer: str,
    *,
    list_of_threshold: List[int] = None,
) -> Tuple[list, float, float]:
    threshold_score_cv = []
    threshold_std_cv = []

    if list_of_threshold is None:
        list_of_threshold = [i for i in range(20, 201, 20)]

    for threshold in list_of_threshold:
        _, _, _, best_score, best_std = train_model(
            train_data=train_data,
            threshold=threshold,
            signature_depth=signature_depth,
            param_grid=param_grid,
            problem_type=problem_type,
            trans_func=trans_func,
            scorer=scorer,
        )
        threshold_score_cv.append(abs(best_score))
        threshold_std_cv.append(best_std)

        print(
            f"threshold: {threshold} sec, cv score: {abs(best_score)},  cv std: {best_std}"
        )

    return list_of_threshold, threshold_score_cv, threshold_std_cv


def leave_one_group_out_validation(
    data: dict,
    trans_func: Callable[[dict, int], tuple],
    param_grid: dict,
    scorer: str,
    problem_type: str,
) -> dict:
    cathode_groups = ["NMC532", "HE5050", "5Vspinel", "NMC622", "NMC111", "NMC811"]

    group_metric = {}
    if problem_type == "regression":
        metric = mean_absolute_error
    elif problem_type == "classification":
        metric = f1_score

    for grp in cathode_groups:
        data_grp = {
            k: data[k] for k in data if data[k]["summary"]["cathode_group"] == grp
        }

        cathode_groups_temp = cathode_groups.copy()
        cathode_groups_temp.remove(grp)

        data_rest = {
            k: data[k]
            for k in data
            if data[k]["summary"]["cathode_group"] in cathode_groups_temp
        }

        # take threshold of 120 sec
        best_pipeline, best_params, best_model, best_score, best_std = train_model(
            train_data=data_rest,
            signature_depth=3,
            threshold=120,
            param_grid=param_grid,
            problem_type=problem_type,
            trans_func=trans_func,
            scorer=scorer,
        )

        X_val, y_val = best_pipeline.transform(data_grp)
        val_score = metric(y_val, best_model.predict(X_val))
        group_metric[grp] = val_score

        print(
            f"Model validated on {grp}; cv score: {abs(best_score)}, cv std: {best_std}, val score: {val_score}"
        )

    return group_metric


def log_model_pipeline(
    pipeline: extraction.FeaturesTargetExtractor,
    model: Union[RegressorMixin, ClassifierMixin],
    model_name: str,
) -> None:
    # dump the pipeline
    generic_helper.dump_data(
        data=pipeline, path=f"{ROOT_DIR}/models", fname=f"{model_name}_pipeline.pkl"
    )

    # dump trained model
    generic_helper.dump_data(
        data=model, path=f"{ROOT_DIR}/models", fname=f"{model_name}_model.pkl"
    )

    return None


def time_threshold_effect_feature_importance(
    train_data: dict,
    signature_depth: int,
    param_grid: dict,
    problem_type: str,
    trans_func: Callable[[dict, int], tuple],
    scorer: str,
    max_features: int = 10,
) -> dict:
    result = {}
    feature_names = generic_helper.get_sig_convention(
        dimension=3, depth=signature_depth
    )

    list_of_threshold = [i for i in range(20, 201, 20)]

    for threshold in list_of_threshold:
        _, _, best_model, _, _ = train_model(
            train_data=train_data,
            threshold=threshold,
            signature_depth=signature_depth,
            param_grid=param_grid,
            problem_type=problem_type,
            trans_func=trans_func,
            scorer=scorer,
        )

        temp_model_feature_dict = {
            problem_type: {"fitted_model": best_model, "features": feature_names}
        }

        temp_analysis_res = feature_importance_analysis(
            model_feature_dict=temp_model_feature_dict, mode="xgboost"
        )

        result[threshold] = temp_analysis_res[problem_type]["features"][-max_features:][
            ::-1
        ]

        print(f"Running threshold: {threshold} sec")

    return result


def cross_validation(
    X_train: np.ndarray, y_train: np.ndarray, model: object, scoring: dict
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
