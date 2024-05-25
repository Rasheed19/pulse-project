import logging

from steps import (
    data_loader,
    data_splitter,
    data_preprocessor,
    model_trainer,
    feature_importance_analysis,
)
from utils.definitions import ROOT_DIR
from utils.generic_helper import dump_data, get_sig_convention, config_logger


def threshold_effect_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    test_size: float,
    model_type: str,
    threshold_type: str,
    signature_depth: float,
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

    logger.info("Running experiment...")
    if threshold_type == "interval":
        threshold_list = [(0, 40), (40, 80), (80, 120)]
        for threshold in threshold_list:
            # preprocess
            preprocessor = data_preprocessor(
                train_data=train_data,
                test_data=test_data,
                model_type=model_type,
                signature_depth=signature_depth,
                threshold=threshold,
            )

            # train model
            model_trainer_output = model_trainer(
                X_train=preprocessor.X_train,
                y_train=preprocessor.y_train,
                param_grid=param_grid,
                model_type=model_type,
                scorer=scorer,
            )
            print(
                f"threshold: {threshold}, cv score: {abs(abs(model_trainer_output.best_score)):.2f}, cv std: {model_trainer_output.best_std:.2f}"
            )

        return None

    elif threshold_type == "point":

        threshold_list = list(range(20, 201, 20))
        feature_names = get_sig_convention(dimension=3, depth=signature_depth)

        feature_analysis_result = dict()
        threshold_cv_score = []
        threshold_cv_std = []

        for threshold in threshold_list:
            # preprocess
            preprocessor = data_preprocessor(
                train_data=train_data,
                test_data=test_data,
                model_type=model_type,
                signature_depth=signature_depth,
                threshold=threshold,
            )

            # train model
            model_trainer_output = model_trainer(
                X_train=preprocessor.X_train,
                y_train=preprocessor.y_train,
                param_grid=param_grid,
                model_type=model_type,
                scorer=scorer,
            )
            print(
                f"threshold: {threshold} sec, cv score: {abs(abs(model_trainer_output.best_score)):.2f}, cv std: {model_trainer_output.best_std:.2f}"
            )

            threshold_cv_score.append(abs(model_trainer_output.best_score))
            threshold_cv_std.append(model_trainer_output.best_std)

            temp_model_feature_dict = {
                model_type: {
                    "fitted_model": model_trainer_output.best_estimator,
                    "features": feature_names,
                }
            }
            temp_analysis_res = feature_importance_analysis(
                model_feature_dict=temp_model_feature_dict, mode="xgboost"
            )

            feature_analysis_result[threshold] = temp_analysis_res[model_type][
                "features"
            ][-10:][::-1]

        # save threshold results for plotting
        logger.info("Logging experiment artifacts...")
        dump_data(
            data=(threshold_list, threshold_cv_score, threshold_cv_std),
            path=f"{ROOT_DIR}/data",
            fname=f"{model_type}_threshold_data.pkl",
        )
        dump_data(
            data=feature_analysis_result,
            path=f"{ROOT_DIR}/data",
            fname=f"{model_type}_threshold_feature_importance.pkl",
        )

        logger.info("Experiment pipeline finished successfully.")

        return None

    else:
        raise ValueError(
            f"""Invalid threshold type. Valid options are 'point'
            and 'interval' but {threshold_type} is given.
            """
        )
