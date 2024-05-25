import numpy as np

from .preprocessor import data_preprocessor
from utils.generic_helper import permutation_feature_importance


def feature_importance_analysis(
    model_feature_dict: dict,
    mode: str = "xgboost",
    validation_set: tuple[dict, dict] = None,
) -> dict[str, np.ndarray]:

    # Create a lambda function to scale importance values to the interval [0, 1]
    def scaler(x):
        return (x - x.min()) / (x.max() - x.min())

    analysis_result = {}

    for key, value in model_feature_dict.items():

        if mode == "xgboost":
            feature_importance = (
                scaler(value["fitted_model"].feature_importances_)
                if key == "classification"
                else scaler(value["fitted_model"].regressor_.feature_importances_)
            )

        elif mode == "permutation" and validation_set is not None:
            train_data, test_data = validation_set
            preprocessor = data_preprocessor(
                train_data=train_data,
                test_data=test_data,
                model_type=key,
                signature_depth=value["signature_depth"],
                threshold=120,
            )
            if key == "classification":
                feature_importance = permutation_feature_importance(
                    fitted_model=value["fitted_model"],
                    X=preprocessor.X_train,
                    y=preprocessor.y_train,
                    scoring_function="f1",
                )
            else:
                feature_importance = permutation_feature_importance(
                    fitted_model=value["fitted_model"],
                    X=preprocessor.X_train,
                    y=preprocessor.y_train,
                    scoring_function="neg_mean_squared_error",
                )

            feature_importance = scaler(feature_importance)

        sorted_indices_ = np.argsort(feature_importance)
        analysis_result[key] = {
            "features": value["features"][sorted_indices_],
            "importance": feature_importance[sorted_indices_],
        }

    return analysis_result
