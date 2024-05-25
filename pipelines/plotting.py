import numpy as np
import logging

from steps import data_splitter, feature_importance_analysis
from utils import plotter
from utils.definitions import ROOT_DIR
from utils.generic_helper import read_data, get_sig_convention, config_logger
from utils.structure_noah import get_unique_cathode_groups
from utils.extraction import get_data_for_eol_prediction, get_data_for_rul_prediction


def plot_pipeline() -> None:
    config_logger()
    logger = logging.getLogger(__name__)

    PATH_TO_DATA = f"{ROOT_DIR}/data"
    PATH_TO_MODEL = f"{ROOT_DIR}/models"

    logger.info("Loading data...")
    REFINED_DATA = read_data(
        path=PATH_TO_DATA, fname="noah_structured_data_refined.pkl"
    )
    SPLIT_DATA = data_splitter(
        loaded_data=REFINED_DATA,
        no_proposed_split=False,
        test_size=None,
    )

    logger.info("Generating plots...")
    # check some pulse voltage for some cells
    sample_cells = list(REFINED_DATA.keys())[:4]
    plotter.plot_pulse_voltage_current(
        structured_data_with_pulse=REFINED_DATA, sample_cell=sample_cells[1]
    )

    # plot first pulse cycle distribution
    plotter.distribution_of_firstpulse_cycle_life(
        structured_data_with_pulse=REFINED_DATA, pulse=True
    )

    # plot cells' end of life distribution
    plotter.distribution_of_firstpulse_cycle_life(
        structured_data_with_pulse=REFINED_DATA
    )

    # plot sample filtered capacity
    cathode_groups = get_unique_cathode_groups(structured_data=REFINED_DATA)
    cells_to_plot_for_filtered_capacity = []

    for cathode_group in cathode_groups:
        data_grp = {
            k: REFINED_DATA[k]
            for k in REFINED_DATA
            if REFINED_DATA[k]["summary"]["cathode_group"] == cathode_group
        }
        cells_to_plot_for_filtered_capacity.append(
            np.random.choice(list(data_grp.keys()))
        )  # just choose a cell in the group at random

    plotter.plot_filtered_capacity(
        sample_cells=cells_to_plot_for_filtered_capacity, structured_data=REFINED_DATA
    )

    # get the end of life for training cells
    _, eol_data = get_data_for_eol_prediction(
        structured_data=SPLIT_DATA[0],
        signature_depth=2,  # for completeness, we need only eol
        threshold=120,
    )
    # get the rul for the training cells
    _, rul_data = get_data_for_rul_prediction(
        structured_data=SPLIT_DATA[0],
        signature_depth=2,  # for completeness, we need only rul
        threshold=120,
    )
    # compare eol and its quantile transformation
    plotter.plot_target_transform_comparison(
        data=eol_data,
        bins=25,
        x_label="End of life",
        save_name="eol_transfromation_comparison",
    )
    # for the remaining useful life
    plotter.plot_target_transform_comparison(
        data=rul_data,
        bins=25,
        x_label="Remaining useful life",
        save_name="rul_transfromation_comparison",
        fig_labels=["c", "d"],
    )

    # parity plot for eol and rul
    prediction_data_list = [
        read_data(path=PATH_TO_DATA, fname=fname)
        for fname in ["eol_prediction_data.pkl", "rul_prediction_data.pkl"]
    ]
    plotter.parity_plot(prediction_data_list=prediction_data_list, tag="eol_rul_parity")

    # plot roc curve and confusion matrix
    classification_prediction_data = read_data(
        path=PATH_TO_DATA, fname="classification_prediction_data.pkl"
    )["test"]
    plotter.plot_cunfusion_matrix_roc_curve(
        y_true=classification_prediction_data["actual"],
        y_pred=classification_prediction_data["prediction"],
        y_score=classification_prediction_data["prediction_prob"],
        classes=["Passed", "Not passed"],
    )

    # plot feature importance
    model_names = ["eol", "rul", "classification"]
    depths = [3, 6, 6]
    model_feature_dict = {
        name: {
            "fitted_model": read_data(path=PATH_TO_MODEL, fname=f"{name}_model.pkl"),
            "features": get_sig_convention(dimension=3, depth=depth),
            "signature_depth": depth,
        }
        for name, depth in zip(model_names, depths)
    }

    for m, l in zip(["xgboost", "permutation"], [None, ["d", "e", "f"]]):

        analysis_result = feature_importance_analysis(
            model_feature_dict=model_feature_dict,
            mode=m,
            validation_set=(None if m == "xgboost" else SPLIT_DATA),
        )
        plotter.plot_feature_importance(
            analysis_result=analysis_result,
            threshold=10,
            tag=f"{m}_feature_importance",
            fig_labels=l,
        )

    # plot number of cells per group and first pulse cycle distribution
    plotter.plot_num_cells_first_pulse_dist(structured_data_with_pulse=REFINED_DATA)

    # plot the effect of time threshold on models
    threshold_data_dict = {
        model_name: read_data(
            path=PATH_TO_DATA, fname=f"{model_name}_threshold_data.pkl"
        )
        for model_name in ["rul", "classification"]
    }

    plotter.plot_time_threshold_effect(threshold_data_dict=threshold_data_dict)

    # plot the effect of time threshold on feature importance
    plotter.plot_combined_feature_similarity(
        data_list=[
            read_data(path=PATH_TO_DATA, fname=f"{m}_threshold_feature_importance.pkl")
            for m in ["eol", "rul", "classification"]
        ],
        fig_labels=["g", "h", "i"],
    )

    # plot analysis chart for graphical abstract
    plotter.plot_analysis_graphical_abstract(
        prediction_data_list=prediction_data_list, analysis_result=analysis_result
    )

    # get stripplot of first pulse cycles and end of life
    for fl, yl, l in zip(
        [True, False], ["First pulse cycle", "End of life"], ["a", "b"]
    ):

        plotter.strip_plot_firstpulse_cycle_life(
            structured_data_with_pulse=REFINED_DATA,
            pulse_cycle=fl,
            ylabel=yl,
            fig_label=l,
        )

    # plot a full HPPC test for a sample cell for a given cycle
    plotter.plot_relplot_pulse_profile(
        path_to_sample_cell=f"{ROOT_DIR}/noah_raw_data/batch_B9A_cell_1.h5"
    )

    # plot cropped pulse profiles for a sample cell for a given cycle
    plotter.plot_full_pulse_profile(
        path_to_sample_cell=f"{ROOT_DIR}/noah_raw_data/batch_B9A_cell_1.h5",
        pulse_cycle=14,
        style="uncropped",
    )

    # for target description in the graphical abstract
    plotter.plot_target_graphical_abstract(
        structured_data=REFINED_DATA, sample_cell=list(REFINED_DATA.keys())[-3]
    )

    logger.info(
        "Plotting pipeline finished successfully. See the 'plots' folder for the results."
    )

    return None
