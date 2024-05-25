from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import Pipeline

from utils.extraction import (
    get_data_for_eol_prediction,
    get_data_for_rul_prediction,
    get_data_for_classification,
    inputer_scaler_pipeline,
)


@dataclass(frozen=True)
class TrainingTestData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocess_pipeline: Pipeline


def data_preprocessor(
    train_data: dict[str, dict],
    test_data: dict[str, dict],
    model_type: str,
    signature_depth: float,
    threshold: float,
) -> TrainingTestData:

    model_types = dict(
        eol=get_data_for_eol_prediction,
        rul=get_data_for_rul_prediction,
        classification=get_data_for_classification,
    )
    if model_type not in model_type:
        raise ValueError(
            f"""Model type must be one of 'eol', 'rul' or 'classification'
            but {model_type} is given."""
        )

    X_train, y_train = model_types[model_type](
        structured_data=train_data,
        signature_depth=signature_depth,
        threshold=threshold,
    )
    X_test, y_test = model_types[model_type](
        structured_data=test_data,
        signature_depth=signature_depth,
        threshold=threshold,
    )

    preprocess_pipeline = inputer_scaler_pipeline()
    X_train = preprocess_pipeline.fit_transform(X_train)
    X_test = preprocess_pipeline.transform(X_test)

    return TrainingTestData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocess_pipeline=preprocess_pipeline,
    )
