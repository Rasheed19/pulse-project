import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from utils.generic_helper import training_result_clf, training_result_reg


def model_evaluator(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: ClassifierMixin | RegressorMixin,
    alpha: float,
) -> tuple[pd.DataFrame, dict]:

    if model_type in ["eol", "rul"]:
        return training_result_reg(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=model,
            alpha=alpha,
        )
    elif model_type == "classification":
        return training_result_clf(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=model,
            alpha=alpha,
        )

    else:
        raise ValueError(
            f"""Model type must be one of 'eol', 'rul' or 'classification'
            but {model_type} is given."""
        )
