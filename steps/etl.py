import os

from utils.generic_helper import dump_data, read_data
from utils.structure_noah import get_structured_data
from utils.definitions import ROOT_DIR


def data_loader(not_loaded: bool) -> dict[str, dict]:

    if not_loaded:
        absolute_path = f"{ROOT_DIR}/noah_raw_data"
        extracted_contents = os.listdir(absolute_path)

        # create a list of absolute paths to cells' data
        absolute_path_to_files = [
            f"{absolute_path}/{file}" for file in extracted_contents
        ]

        # get structured data
        structured_data = get_structured_data(path_to_files=absolute_path_to_files)

        # there are some cells without pulse testing
        # check them out and remove them
        cells_without_pulse = [
            cell
            for cell in structured_data
            if len(structured_data[cell]["pulses"].keys()) == 0
        ]

        # remove cells with irregular voltage
        irregular_cells = ["batch_B1A_cell_4", "batch_B35H_cell_1", "batch_B26K_cell_2"]

        # get structured data for only cells with pulse testing and not in irregular cells list
        data = {
            k: structured_data[k]
            for k in structured_data.keys()
            if k not in cells_without_pulse + irregular_cells
        }

        # remove cells whose end of life is less than
        # the cycle at which first pulse test takes place
        data = {
            k: data[k]
            for k in data
            if list(data[k]["pulses"])[0] < data[k]["summary"]["end_of_life"]
        }

        # dump the refined data
        dump_data(
            data=data, path=f"{ROOT_DIR}/data", fname="noah_structured_data_refined.pkl"
        )

        return data

    data = read_data(path=ROOT_DIR, fname="data/noah_structured_data_refined.pkl")

    return data
