import pandas as pd

from utils.definitions import ROOT_DIR
from utils.generic_helper import bring_splits_together
from utils.structure_noah import get_unique_cathode_groups


def data_splitter(
    loaded_data: dict[str, dict],
    no_proposed_split: bool = False,
    test_size: float | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:

    if no_proposed_split:

        cathode_groups = get_unique_cathode_groups(loaded_data)
        train_data, test_data = bring_splits_together(
            data=loaded_data, cathode_groups=cathode_groups, test_ratio=test_size
        )

        return train_data, test_data

    train_cells = pd.read_csv(f"{ROOT_DIR}/train_test_cells/train_cells.csv")
    test_cells = pd.read_csv(f"{ROOT_DIR}/train_test_cells/test_cells.csv")

    train_data = {cell: loaded_data[cell] for cell in train_cells["train_cells"].values}
    test_data = {cell: loaded_data[cell] for cell in test_cells["test_cells"].values}

    return train_data, test_data
