import logging

from steps import (
    data_loader,
    data_splitter,
    data_preprocessor,
    model_trainer,
    model_evaluator,
)
from utils.definitions import ROOT_DIR
from utils.generic_helper import (
    dump_data,
    log_model_pipeline,
    cross_validation,
    config_logger,
)


def training_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    test_size: float,
    model_type: str,
    signature_depth: float,
    threshold: float,
    param_grid: dict,
    scorer: str,
) -> None:

    config_logger()
    logger = logging.getLogger(__name__)

    # load data
    logger.info("Loading data...")
    loaded_data = data_loader(not_loaded=not_loaded)

    # get splits
    train_data, test_data = data_splitter(
        loaded_data=loaded_data,
        no_proposed_split=no_proposed_split,
        test_size=test_size,
    )

    # preprocess
    logger.info("Preprocessing data...")
    preprocessor = data_preprocessor(
        train_data=train_data,
        test_data=test_data,
        model_type=model_type,
        signature_depth=signature_depth,
        threshold=threshold,
    )

    # train model
    logger.info("Identifying best hyperparameters and training model...")
    model_trainer_output = model_trainer(
        X_train=preprocessor.X_train,
        y_train=preprocessor.y_train,
        param_grid=param_grid,
        model_type=model_type,
        scorer=scorer,
    )
    print(f"best params: {model_trainer_output.best_params}")

    # evaluate model
    logger.info("Evaluating model performance...")
    metric_ci_data, prediction_data = model_evaluator(
        model_type=model_type,
        X_train=preprocessor.X_train,
        y_train=preprocessor.y_train,
        X_test=preprocessor.X_test,
        y_test=preprocessor.y_test,
        model=model_trainer_output.best_estimator,
        alpha=0.05,
    )
    print(f"model metrics:\n{metric_ci_data}")

    # cross-validate the model
    logger.info("Cross-validating model...")
    scoring = dict(
        eol={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"},
        rul={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"},
        classification={
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1",
            "roc_auc_score": "roc_auc",
            "accuracy": "accuracy",
        },
    )
    cross_val_res = cross_validation(
        X_train=preprocessor.X_train,
        y_train=preprocessor.y_train,
        model=model_trainer_output.best_estimator,
        scoring=scoring[model_type],
    )
    print(f"cross-validation:\n{cross_val_res}")

    # save prediction data for parity plot
    logger.info("Logging model artifacts...")
    dump_data(
        data=prediction_data,
        path=f"{ROOT_DIR}/data",
        fname=f"{model_type}_prediction_data.pkl",
    )

    # log pipeline and model
    log_model_pipeline(
        pipeline=preprocessor.preprocess_pipeline,
        model=model_trainer_output.best_estimator,
        model_type=model_type,
    )

    logger.info("Model training pipeline finished successfully.")

    return None
