import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import RegressorMixin, ClassifierMixin
from xgboost import XGBClassifier, XGBRegressor


@dataclass(frozen=True)
class ModelTrainerOutput:
    best_params: dict
    best_estimator: RegressorMixin | ClassifierMixin
    best_score: float
    best_std: float


def get_fitted_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict,
    *,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    model_type: str = "regression",
) -> RegressorMixin | ClassifierMixin:
    if model_type in ["eol", "rul"]:
        model = XGBRegressor(**model_params)
    elif model_type == "classification":
        model = XGBClassifier(**model_params)
    else:
        raise ValueError(
            f"""Model type must be one of 'eol', 'rul' or 'classification'
            but {model_type} is given."""
        )

    if X_val is None or y_val is None:
        return model.fit(X_train, y_train)

    fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}
    model.fit(X_train, y_train, **fit_params)

    return model


def model_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    model_type: str,
    scorer: str,
    cv: int = 5,
) -> ModelTrainerOutput:

    if model_type in ["eol", "rul"]:
        estimator = TransformedTargetRegressor(
            regressor=XGBRegressor(),
            transformer=QuantileTransformer(
                n_quantiles=int(((cv - 1) / cv) * X_train.shape[0]),
                output_distribution="normal",
            ),
        )
    elif model_type == "classification":
        estimator = XGBClassifier()
    else:
        raise ValueError(
            f"""Model type must be one of 'eol', 'rul' or 'classification'
            but {model_type} is given."""
        )

    grid_search = GridSearchCV(
        estimator=estimator, param_grid=param_grid, scoring=scorer, refit=True, cv=cv
    )

    grid_search.fit(X=X_train, y=y_train)

    return ModelTrainerOutput(
        best_params=grid_search.best_params_,
        best_estimator=grid_search.best_estimator_,
        best_score=grid_search.best_score_,
        best_std=grid_search.cv_results_["std_test_score"][grid_search.best_index_],
    )
