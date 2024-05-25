import logging

from sklearn.metrics import (
    mean_absolute_error,
    f1_score,
)
from steps import (
    data_loader,
    data_preprocessor,
    model_trainer,
)
from utils.generic_helper import config_logger


def leave_one_group_out_pipeline(
    not_loaded: bool,
    model_type: str,
    signature_depth: float,
    threshold: float,
    param_grid: dict,
    scorer: str,
) -> None:

    config_logger()
    logger = logging.getLogger(__name__)

    cathode_groups = ["NMC532", "HE5050", "5Vspinel", "NMC622", "NMC111", "NMC811"]

    if model_type in ["eol", "rul"]:
        metric = mean_absolute_error
    elif model_type == "classification":
        metric = f1_score

    # load data
    logger.info("Loading data...")
    data = data_loader(not_loaded=not_loaded)

    # filter out cells that live more than 950 cycles
    data = {k: data[k] for k in data if data[k]["summary"]["end_of_life"] <= 950}

    logger.info("Running leave-one-group-out cross-validation...")
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

        # preprocess
        preprocessor = data_preprocessor(
            train_data=data_rest,
            test_data=data_grp,
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

        test_score = metric(
            preprocessor.y_test,
            model_trainer_output.best_estimator.predict(preprocessor.X_test),
        )
        print(f"Model validated on {grp}, val score: {test_score:.2f}")

    logger.info("Leave-one-group-out cross-validation pipeline finished successfully.")

    return None
