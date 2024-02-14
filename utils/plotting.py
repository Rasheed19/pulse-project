import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Any, Union, List
from utils.definitions import ROOT_DIR
from utils import experiment, generic_helper, structure_noah
import importlib
import itertools
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import quantile_transform

importlib.reload(experiment)
importlib.reload(generic_helper)
importlib.reload(structure_noah)


# configure plotting style
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

font = {"family": "serif", "serif": ["Times"], "size": MEDIUM_SIZE}

rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

rc("font", **font)
rc("text", usetex=True)

# define text width from overleaf
TEXT_WIDTH = 360.0


def set_size(
    width: float = TEXT_WIDTH, fraction: float = 1.0, subplots: tuple = (1, 1)
) -> Tuple[float, float]:
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def axis_to_fig(axis: Any) -> Callable[[tuple], Any]:
    """
    Converts axis to fig object.

    Args:
    ----
         axis (object): axis object

    Returns:
    -------
            transformed axis oject.
    """

    fig = axis.figure

    def transform(coord: Union[tuple, list]):
        return fig.transFigure.inverted().transform(axis.transAxes.transform(coord))

    return transform


def add_sub_axes(axis: Any, rect: Union[tuple, list]) -> Any:
    """
    Adds sub-axis to existing axis object.

    Args:
    ----
         axis (object):        axis object
         rect (list or tuple): list or tuple specifying axis dimension

    Returns:
    -------
           fig object with added axis.
    """
    fig = axis.figure
    left, bottom, width, height = rect
    trans = axis_to_fig(axis)
    figleft, figbottom = trans((left, bottom))
    figwidth, figheight = trans([width, height]) - trans([0, 0])

    return fig.add_axes([figleft, figbottom, figwidth, figheight])


def parity_plot(prediction_data_list: List[dict], tag: str) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    fig_labels = ["a", "b"]
    marker_style = dict(
        facecolor="white",  # Interior color
        # edgecolor='black'    # Border color
    )

    for i, prediction_data in enumerate(prediction_data_list):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        # ax.scatter(
        #     prediction_data["train"]["actual"],
        #     prediction_data["train"]["prediction"],
        #     color="royalblue",
        #     s=20,
        #     #alpha=0.5,
        #     marker='o',
        #     label="Train",
        #     **marker_style
        # )

        ax.scatter(
            prediction_data["test"]["actual"],
            prediction_data["test"]["prediction"],
            color="red",
            s=20,
            # alpha=0.5,
            marker="o",
            label="Test",
            **marker_style,
        )

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against each other
        ax.plot(lims, lims, "k", zorder=100)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)

        if i == 0:
            ax.set_ylabel("Predicted values")

        ax.set_xlabel("Measured values")

        # embed histogram of residuals
        subaxis = add_sub_axes(ax, [0.29, 0.85, 0.3, 0.2])
        residuals = []

        for value in prediction_data.values():
            residuals.extend((value["actual"] - value["prediction"]).tolist())

        subaxis.hist(residuals, bins=50, color="black", alpha=0.75, ec="black")
        subaxis.set_xlim(-max(residuals), max(residuals))
        subaxis.set_xlabel("Residuals")
        subaxis.set_ylabel("Count")
        subaxis.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc='upper center',
    #           ncol=2, bbox_to_anchor=(-0.2, -0.4))

    plt.savefig(fname=f"{ROOT_DIR}/plots/pulse_project_{tag}.pdf", bbox_inches="tight")

    return None


def plot_pulse_voltage_current(
    structured_data_with_pulse: dict, sample_cell: str
) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    line_styles = ["solid", "dashed", "dotted"]
    line_colors = ["black", "brown", "blue"]
    fig_labels = ["a", "b"]

    for i, (qty, label) in enumerate(
        zip(["current", "voltage"], ["Current (A)", "Voltage (V)"])
    ):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        for j, pulse in enumerate(
            list(structured_data_with_pulse[sample_cell]["pulses"])[:3]
        ):
            time = structured_data_with_pulse[sample_cell]["pulses"][pulse]["time"]

            quantity = structured_data_with_pulse[sample_cell]["pulses"][pulse][qty]

            ax.plot(
                time,
                quantity,
                linestyle=line_styles[j],
                color=line_colors[j],
                label=f"Cycle: {pulse}",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)

        # ax.set_title(sample_cell)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(-0.2, -0.3))

    plt.savefig(f"{ROOT_DIR}/plots/pulse_project_vc_plot.pdf", bbox_inches="tight")

    plt.show()
    return None


def distribution_of_firstpulse_cycle_life(
    structured_data_with_pulse: dict, pulse: bool = False
) -> None:
    cathode_groups = [
        "5Vspinel",
        "HE5050",
        "NMC111",
        "NMC532",
        "NMC622",
        "NMC811",
        "FCG",
        " Li1.2Ni0.3Mn0.6O2",
        " Li1.35Ni0.33Mn0.67O2.35",
    ]
    _, ax = plt.subplots(figsize=set_size())
    markers = ["o", "s", "D", "^", "v", ">", "<", "p", "8"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "purple", "orange"]

    for i, grp in enumerate(cathode_groups, start=1):
        grp_data = {
            k: structured_data_with_pulse[k]
            for k in structured_data_with_pulse
            if structured_data_with_pulse[k]["summary"]["cathode_group"] == grp
        }
        if pulse:
            x_values = [list(grp_data[cell]["pulses"])[0] for cell in grp_data]
        else:
            x_values = [grp_data[cell]["summary"]["end_of_life"] for cell in grp_data]

        ax.scatter(
            x_values,
            [i] * len(x_values),
            marker=markers[i - 1],
            color=colors[i - 1],
            alpha=0.7,
            facecolor="white",
            label=f"{len(x_values)} cells",
        )

    # rename the last two cathodes to have a nice look
    cathode_groups[-2] = r"Li$_{1.2}$Ni$_{0.3}$Mn$_{0.6}$O$_2$"
    cathode_groups[-1] = r"Li$_{1.35}$Ni$_{0.33}$Mn$_{0.67}$O$_{2.35}$"

    ax.set_yticks(ticks=list(range(1, 10)), labels=cathode_groups)
    ax.spines[
        [
            "top",
            "right",
        ]
    ].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    # if not pulse:
    ax.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.2))
    ax.set_xlabel("First pulse cycles" if pulse else "End of life")

    save_tag = "first_pulse" if pulse else "end_of_life"
    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_count_{save_tag}.pdf", bbox_inches="tight"
    )

    return None


def plot_num_cells_first_pulse_dist(structured_data_with_pulse: dict) -> None:
    cathode_groups = [
        "NMC532",
        "HE5050",
        "5Vspinel",
        "NMC622",
        "NMC111",
        "NMC811",
        "FCG",
        " Li1.2Ni0.3Mn0.6O2",
        " Li1.35Ni0.33Mn0.67O2.35",
    ]
    cathode_groups_as_labels = cathode_groups.copy()
    cathode_groups_as_labels[-2] = r"Li$_{1.2}$Ni$_{0.3}$Mn$_{0.6}$O$_2$"
    cathode_groups_as_labels[-1] = r"Li$_{1.35}$Ni$_{0.33}$Mn$_{0.67}$O$_{2.35}$"

    number_of_cells = []
    first_pulse_cycle = []

    for grp in cathode_groups:
        grp_data = {
            k: structured_data_with_pulse[k]
            for k in structured_data_with_pulse
            if structured_data_with_pulse[k]["summary"]["cathode_group"] == grp
        }
        x_temp = [list(grp_data[cell]["pulses"])[0] for cell in grp_data]
        number_of_cells.append(len(x_temp))
        first_pulse_cycle.extend(x_temp)

    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    fig_labels = ["a", "b"]

    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.spines[["top", "right"]].set_visible(False)

        if i == 0:
            ax.bar(
                cathode_groups_as_labels,
                number_of_cells,
                color="red",
                ec="black",
                alpha=0.75,
            )
            ax.tick_params(axis="x", rotation=90)

            ax.set_xlabel("Cathode groups")
            ax.set_ylabel("Number of cells")

            for j, p in enumerate(ax.patches):
                ax.annotate(
                    number_of_cells[j],
                    (p.get_x() + p.get_width() / 5.0, p.get_height()),
                    ha="left",
                    va="center",
                    xytext=(0, 10),
                    textcoords="offset points",
                )

        else:
            ax.hist(first_pulse_cycle, bins=10, alpha=0.75, color="red", ec="black")

            ax.set_xlabel("First pulse cycle number")
            ax.set_ylabel("Count")

    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_cells_per_group_1st_pulse_dist.pdf",
        bbox_inches="tight",
    )
    plt.show()

    return None


def plot_sample_discharge_capacity(structured_data_with_pulse: dict, sample_cells: str):
    fig = plt.figure(figsize=set_size(subplots=(2, 2)))

    for i, cell in enumerate(sample_cells):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(
            structured_data_with_pulse[cell]["summary"]["cycle"],
            structured_data_with_pulse[cell]["summary"]["capacity"],
        )

        # if i in [2, 3]:
        ax.set_xlabel("Cycle")

        if i % 2 == 0:
            ax.set_ylabel("Discharge capacity")
        # ; Cell name: {cell}")
        ax.set_title(
            f"Cathode group: {structured_data_with_pulse[cell]['summary']['cathode_group']}"
        )
        ax.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)

    fig.tight_layout()
    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_sample_capacity.pdf", bbox_inches="tight"
    )

    return None


def plot_filtered_capacity(sample_cells: List[str], structured_data: dict) -> None:
    fig = plt.figure(figsize=set_size(subplots=(3, 3)))
    fig_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    for i, cell in enumerate(sample_cells):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.text(
            x=-0.15,
            y=1.4,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.plot(
            structured_data[cell]["summary"]["cycle"],
            structured_data[cell]["summary"]["capacity"],
            label="Measured capacity",
        )
        ax.plot(
            structured_data[cell]["summary"]["cycle"],
            structured_data[cell]["summary"]["filtered_capacity"],
            label="Filtered capacity",
            color="red",
        )

        ax.axvline(
            x=structured_data[cell]["summary"]["end_of_life"],
            label="End of life",
            linestyle="--",
            color="black",
        )

        if i in [6, 7, 8]:
            ax.set_xlabel("Cycle")

        if i in [0, 3, 6]:
            ax.set_ylabel("Capacity (Ah)")

        cathode_group = structured_data[cell]["summary"]["cathode_group"]
        if cathode_group == " Li1.2Ni0.3Mn0.6O2":
            cathode_group = r"Li$_{1.2}$Ni$_{0.3}$Mn$_{0.6}$O$_2$"
        elif cathode_group == " Li1.35Ni0.33Mn0.67O2.35":
            cathode_group = r"Li$_{1.35}$Ni$_{0.33}$Mn$_{0.67}$O$_{2.35}$"

        ax.set_title(f"{cathode_group}")
        ax.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)

    fig.tight_layout(pad=0.1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(-0.8, -1.0))

    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_sample_filtered_capacity.pdf",
        bbox_inches="tight",
    )
    plt.show()

    return None


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    classes: List[str],
    cmap: Any = plt.cm.Blues,
) -> None:
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    _, ax = plt.subplots(figsize=set_size(fraction=0.5, subplots=(2, 1)))
    ax.imshow(cm_percent, interpolation="nearest", cmap=cmap)
    tick_marks = np.arange(len(classes))

    ax.set_xticks(
        ticks=tick_marks,
        labels=classes,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(
        ticks=tick_marks,
        labels=classes,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    fmt = ".2f"
    thresh = cm_percent.max() / 2.0
    for i, j in itertools.product(
        range(cm_percent.shape[0]), range(cm_percent.shape[1])
    ):
        ax.text(
            j,
            i,
            f"{format(cm_percent[i, j] * 100., fmt)}\%",
            horizontalalignment="center",
            color="white" if cm_percent[i, j] > thresh else "black",
        )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_confusion_matrix.pdf", bbox_inches="tight"
    )
    plt.show()

    return None


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    _, ax = plt.subplots(figsize=set_size(fraction=0.5, subplots=(2, 1)))
    ax.plot(fpr, tpr, label="XGBoost classifier")
    ax.plot([0, 1], [0, 1], "k--", label="Worst classifier")
    ax.set_xlabel("False negative rate")
    ax.set_ylabel("True positive rate")
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend()

    plt.savefig(f"{ROOT_DIR}/plots/pulse_project_roc_curve.pdf", bbox_inches="tight")
    plt.show()

    return None


def plot_target_transform_comparison(
    data: np.ndarray, bins: int, x_label: str, save_name: str
) -> None:
    data_transformed = quantile_transform(
        X=data.reshape(-1, 1),
        n_quantiles=data.shape[0],
        output_distribution="normal",
    )

    label_data_dict = dict(
        zip([x_label, f"Transformed {x_label}"], [data, data_transformed])
    )
    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    fig_labels = ["a", "b"]

    for i, key in enumerate(label_data_dict):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.hist(label_data_dict[key], bins=bins, alpha=0.75, color="red", ec="black")

        ax.set_xlabel(key)

        if i == 0:
            ax.set_ylabel("Count")

        ax.spines[
            [
                "top",
                "right",
            ]
        ].set_visible(False)

    plt.savefig(f"{ROOT_DIR}/plots/pulse_project_{save_name}.pdf", bbox_inches="tight")
    plt.show()

    return None


def plot_cunfusion_matrix_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classes: List[str],
    cmap: Any = plt.cm.Reds,
) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    fig_labels = ["a", "b"]

    for i, fig_label in enumerate(fig_labels):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf \Large {}".format(fig_label),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        # for the confusion matrix
        if i == 0:
            cm = confusion_matrix(y_true, y_pred)
            cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            ax.imshow(cm_percent, interpolation="nearest", cmap=cmap)
            tick_marks = np.arange(len(classes))

            ax.set_xticks(
                ticks=tick_marks,
                labels=classes,
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set_yticks(
                ticks=tick_marks,
                labels=classes,
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

            fmt = ".2f"
            thresh = cm_percent.max() / 2.0
            for i, j in itertools.product(
                range(cm_percent.shape[0]), range(cm_percent.shape[1])
            ):
                ax.text(
                    j,
                    i,
                    f"{format(cm_percent[i, j] * 100., fmt)}\%",
                    horizontalalignment="center",
                    color="white" if cm_percent[i, j] > thresh else "black",
                )

            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")

        # for the roc curve
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, "r-", label="XGBoost")
            ax.plot([0, 1], [0, 1], "k--", label="Chance level")
            ax.set_xlabel("False negative rate")
            ax.set_ylabel("True positive rate")
            ax.spines[["top", "right"]].set_visible(False)

            ax.legend()

    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_roc_confusion_matrix.pdf", bbox_inches="tight"
    )
    plt.show()

    return None


def plot_feature_importance(analysis_result: dict, threshold: int, tag: str) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 3)))
    fig_labels = ["a", "b", "c"]

    for i, value in enumerate(analysis_result.values()):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        ax.bar(
            value["features"][-threshold:][::-1],
            value["importance"][-threshold:][::-1],
            color="red",
            ec="black",
            alpha=0.75,
        )
        ax.tick_params(axis="x", rotation=90)
        ax.spines[["top", "right"]].set_visible(False)

        if i != 0:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Feature importance")

    plt.savefig(fname=f"{ROOT_DIR}/plots/pulse_project_{tag}.pdf", bbox_inches="tight")
    plt.show()

    return None


def bifurcation_discriminant(
    mu_range: np.ndarray,
    critical_point: Callable[[np.ndarray], np.ndarray],
    grad_f: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    critical_values = critical_point(mu_range)
    df_values = grad_f(critical_values, mu_range)

    # Create a mask for positive and negative values
    positive_mask = df_values > 0
    negative_mask = df_values < 0

    return {
        "unstable": (mu_range[positive_mask], critical_values[positive_mask]),
        "stable": (mu_range[negative_mask], critical_values[negative_mask]),
    }


def bifurcation_diagram(bifurcation_list: List[dict]) -> None:
    _, ax = plt.subplots(figsize=set_size())

    for i, bl in enumerate(bifurcation_list):
        ax.plot(
            bl["unstable"][0],
            bl["unstable"][1],
            "--",
            c="black",
            linewidth=2.0,
            label="Unstable" if i == 0 else None,
        )
        ax.plot(
            bl["stable"][0],
            bl["stable"][1],
            "-",
            c="black",
            linewidth=2.0,
            label="Stable" if i == 0 else None,
        )

    ax.legend()
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$x$")
    ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/ds_bifurcation_diagram_qn1.pdf", bbox_inches="tight"
    )

    return None


def plot_time_threshold_effect(threshold_data_dict: dict) -> None:
    fig = plt.figure(figsize=set_size(fraction=1, subplots=(1, 2)))
    fig_labels = ["a", "b"]

    for i, (key, value) in enumerate(threshold_data_dict.items()):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.15,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
        ax.plot(
            value[0],
            value[1] if key in ["rul", "eol"] else np.array(value[1]) * 100.0,
            color="red",
            marker="o",
            label="Mean",
        )

        if key in ["rul", "eol"]:
            ax.set_ylabel("MAE (cycles)")

        elif key == "classification":
            ax.set_ylabel(r"$F_1$-score (\%)")

        ax.set_xlabel("Time threshold (sec)")
        ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/pulse_project_time_threshold_effect.pdf",
        bbox_inches="tight",
    )

    return None


def plot_feature_similarity(data: dict, tag: str, fig_label: str) -> None:
    similarity_scores = []

    for t1 in data.keys():
        temp_sim_score = []
        for t2 in data.keys():
            temp_sim_score.append(
                generic_helper.jaccard_similarity(data[t1].tolist(), data[t2].tolist())
            )

        similarity_scores.append(temp_sim_score)

    similarity_scores = np.array(similarity_scores)

    _, ax = plt.subplots(figsize=set_size(fraction=0.5, subplots=(1, 1)))
    axis_labels = [int(i) for i in data.keys()]
    # ax.set_xticklabels(axis_labels)
    # ax.set_yticklabels(axis_labels)

    # similarity matix is symmetric, only show the lower triangular part
    mask = np.tril(np.ones_like(similarity_scores, dtype=bool))

    sns.heatmap(
        similarity_scores,
        vmin=0,
        vmax=1,
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        linewidth=0.5,
        # linecolor='black',
        ax=ax,
        cbar_kws={"label": "Jaccard similarity"},
        annot=False,
        mask=~mask,
        # fmt=".2f",
        cmap=plt.cm.Reds,
    )

    ax.text(
        x=-0.1,
        y=1.1,
        s=r"\bf \Large {}".format(fig_label),
        transform=ax.transAxes,
        fontweight="bold",
        va="top",
    )

    ax.set_xlabel("Time threshold (sec)")
    ax.set_ylabel("Time threshold (sec)")

    plt.yticks(rotation=0)
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/pulse_project_{tag}_similarity_scores.pdf",
        bbox_inches="tight",
    )


def plot_combined_feature_similarity(data_list: List[dict]) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 3)))
    fig_labels = ["a", "b", "c"]

    for i, data in enumerate(data_list):
        similarity_scores = []

        for t1 in data.keys():
            temp_sim_score = []
            for t2 in data.keys():
                temp_sim_score.append(
                    generic_helper.jaccard_similarity(
                        data[t1].tolist(), data[t2].tolist()
                    )
                )

            similarity_scores.append(temp_sim_score)

        similarity_scores = np.array(similarity_scores)

        ax = fig.add_subplot(1, 3, i + 1)
        ax.text(
            x=-0.15,
            y=1.2,
            s=r"\bf \Large {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        axis_labels = [int(el) for el in data.keys()]

        # similarity matix is symmetric, only show the lower triangular part
        mask = np.tril(np.ones_like(similarity_scores, dtype=bool))

        sns.heatmap(
            similarity_scores,
            vmin=0,
            vmax=1,
            xticklabels=axis_labels,
            yticklabels=axis_labels,
            linewidth=0.5,
            ax=ax,
            cbar=True if i == 2 else False,
            cbar_kws={"label": "Jaccard similarity"} if i == 2 else None,
            annot=False,
            mask=~mask,
            cmap=plt.cm.Reds,
        )

        if i != 0:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Time threshold (sec)")

        ax.set_xlabel("Time threshold (sec)")

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/pulse_project_combined_similarity_scores.pdf",
        bbox_inches="tight",
    )


def plot_analysis_graphical_abstract(
    prediction_data_list: list, analysis_result: dict
) -> None:
    _, axes = plt.subplots(
        2, 1, figsize=set_size(subplots=(2, 1), fraction=0.5, width=400)
    )
    marker_style = dict(
        facecolor="white",  # Interior color
        # edgecolor='black'    # Border color
    )

    for i, ax in enumerate(axes):
        if i == 0:
            ax.scatter(
                prediction_data_list[0]["test"]["actual"],
                prediction_data_list[0]["test"]["prediction"],
                color="red",
                s=20,
                marker="o",
                label="Test",
                **marker_style,
            )

            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]

            # now plot both limits against each other
            ax.plot(lims, lims, "k", zorder=100)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.spines[
                [
                    "top",
                    "right",
                ]
            ].set_visible(False)

            ax.set_ylabel("Predicted values")
            ax.set_xlabel("Measured values")

        elif i == 1:
            value = list(analysis_result.values())
            ax.bar(
                value[0]["features"][-10:][::-1],
                value[0]["importance"][-10:][::-1],
                color="red",
                ec="black",
            )
            ax.tick_params(axis="x", rotation=90)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel("Feature importance")

    plt.tight_layout()
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/parity_polt_feature_importance.svg",
        bbox_inches="tight",
    )


def strip_plot_firstpulse_cycle_life(
    structured_data_with_pulse: dict, *, pulse_cycle: bool, ylabel: str
) -> None:
    cathode_groups = [
        " Li1.35Ni0.33Mn0.67O2.35",
        " Li1.2Ni0.3Mn0.6O2",
        "FCG",
        "NMC811",
        "NMC622",
        "NMC532",
        "NMC111",
        "HE5050",
        "5Vspinel",
    ]
    _, ax = plt.subplots(figsize=set_size())

    groups = []
    cycles = []

    for grp in cathode_groups:
        grp_data = {
            k: structured_data_with_pulse[k]
            for k in structured_data_with_pulse
            if structured_data_with_pulse[k]["summary"]["cathode_group"] == grp
        }
        if pulse_cycle:
            x_values = [list(grp_data[cell]["pulses"])[0] for cell in grp_data]
        else:
            x_values = [grp_data[cell]["summary"]["end_of_life"] for cell in grp_data]

        cycles.extend(x_values)
        groups.extend([grp] * len(x_values))

    cycle_group_df = pd.DataFrame()
    cycle_group_df[ylabel] = cycles
    cycle_group_df["Cathode group"] = groups

    # rename the last two cathodes to have a nice look
    mod_cathode_groups = cathode_groups.copy()
    mod_cathode_groups[0] = r"Li$_{1.35}$Ni$_{0.33}$Mn$_{0.67}$O$_{2.35}$"
    mod_cathode_groups[1] = r"Li$_{1.2}$Ni$_{0.3}$Mn$_{0.6}$O$_2$"

    sns.stripplot(
        data=cycle_group_df,
        x=ylabel,
        y="Cathode group",
        alpha=0.5,
        color="blue",
        marker="o",
        ax=ax,
        linewidth=1,
    )

    ax.set_yticks(ticks=cathode_groups, labels=mod_cathode_groups)
    ax.spines[
        [
            "top",
            "right",
        ]
    ].set_visible(False)
    # ax.tick_params(axis="x", rotation=90)

    save_tag = "first_pulse" if pulse_cycle else "end_of_life"
    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_stripplot_{save_tag}.pdf", bbox_inches="tight"
    )

    return None


def plot_full_pulse_profile(
    path_to_sample_cell: str,
    style: str = "cropped",
) -> None:
    pulse, _ = structure_noah.load_h5_columns_needed(path_to_cell=path_to_sample_cell)
    cycles = pulse["cycle_number"].unique()
    pulse = pulse[
        pulse["cycle_number"] == cycles[0]
    ]  # use the first pulse cycle as a case

    if style == "cropped":
        t, y1, y2 = structure_noah.remove_rest_profile_from_pulse(pulse_data=pulse)

    elif style == "uncropped":
        t, y1, y2 = (
            pulse["test_time"].values,
            pulse["current"].values,
            pulse["voltage"].values,
        )
        t = t - t.min()

    else:
        raise ValueError(
            f"style option must be either 'cropped' or 'uncropped', {style} is provided"
        )

    _, ax1 = plt.subplots(figsize=set_size())
    ax1.plot(t, y1, "--", label="Current", color="red")
    ax1.set_ylabel("Current (A)")
    ax1.set_xlabel("Time (s)")

    ax2 = ax1.twinx()
    ax2.plot(t, y2, label="Voltage", color="blue")
    ax2.set_ylabel("Voltage (V)")

    for ax, loc in zip([ax1, ax2], [0.4, 0.75]):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(loc, -0.2)
        )

        ax.spines["top"].set_visible(False)

    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_v_i_twin_plot_{style}.svg",
        bbox_inches="tight",
    )

    plt.show()


def plot_relplot_pulse_profile(path_to_sample_cell: str) -> None:
    pulse, _ = structure_noah.load_h5_columns_needed(path_to_cell=path_to_sample_cell)
    cycles = pulse["cycle_number"].unique()
    pulse = pulse[
        pulse["cycle_number"] == cycles[0]
    ]  # use the first pulse cycle as a case

    time_diff = np.diff(pulse["test_time"].values)
    odd_indices = np.where(time_diff >= time_diff.min() + 1)[0]

    t = pulse["test_time"].values - pulse["test_time"].min()

    data = {
        "time": t,
        "current": pulse["current"].values,
        "voltage": pulse["voltage"].values,
    }

    df = pd.DataFrame(data)
    thresholds = [0] + [t[i] for i in odd_indices if i % 2 != 0] + [float("inf")]
    labels = [f"HPPC round {i}" for i in range(1, len(thresholds))]

    df["time_group"] = pd.cut(
        df["time"],
        bins=thresholds,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    for p, l in {"current": ["a", "(A)"], "voltage": ["b", "(V)"]}.items():
        fig = plt.figure(figsize=set_size(subplots=(3, 3)))

        for i, r in enumerate(df["time_group"].unique()):
            tmp_df = df[df["time_group"] == r]
            ax = fig.add_subplot(3, 3, i + 1)
            if i == 0:
                ax.text(
                    x=-0.2,
                    y=1.3,
                    s=r"\bf \Large {}".format(l[0]),
                    transform=ax.transAxes,
                    fontweight="bold",
                    va="top",
                )

            ax.scatter(
                tmp_df["time"],
                tmp_df[p],
                s=1,
                color="red" if p == "current" else "blue",
            )

            if i not in [0, 3, 6]:
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.yaxis.set_visible(False)
            else:
                ax.spines[
                    [
                        "top",
                        "right",
                    ]
                ].set_visible(False)
                ax.set_ylim([data[p].min(), data[p].max() + 0.5])

        fig.text(0.5, 0.0, "Time (s)", ha="center", va="center", fontsize=SMALL_SIZE)
        fig.text(
            0.0,
            0.5,
            f"{p.capitalize()} {l[1]}",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=SMALL_SIZE,
        )

        fig.tight_layout()
        plt.savefig(
            f"{ROOT_DIR}/plots/pulse_project_{p}_profile.pdf",
            bbox_inches="tight",
        )
        plt.show()
        plt.close()

    return None


def plot_target_graphical_abstract(structured_data: dict, sample_cell: str) -> None:

    _, ax = plt.subplots(
        figsize=set_size(fraction=1.0),
    )

    ax.plot(
        structured_data[sample_cell]["summary"]["cycle"],
        structured_data[sample_cell]["summary"]["filtered_capacity"],
        color="red",
        linewidth=2.0,
    )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")

    ax.spines[
        [
            "top",
            "right",
        ]
    ].set_visible(False)

    eol = structured_data[sample_cell]["summary"]["end_of_life"]
    cap_at_eol = 0.8 * structured_data[sample_cell]["summary"]["nominal_capacity"]

    ax.axhline(
        y=cap_at_eol,
        linestyle="--",
        color="black",
    )

    plt.savefig(
        f"{ROOT_DIR}/plots/pulse_project_target_graphical_abstract.svg",
        bbox_inches="tight",
    )
