# Created by X at 30.07.24

import numpy as np
import torch

import config
from src.utils import file_utils,data_utils
def grid_kp(data):
    pass


def get_data(data_path, top_data):

    data = file_utils.load_json(data_path / "data.json")
    dataset = []

    for file_type in ["true", "false"]:
        if file_type == "true":
            label = torch.tensor(config.obj_true)
        else:
            label = torch.tensor(config.obj_false)
        files = file_utils.get_all_files(data_path / file_type, "png", True)
        indices = np.random.choice(len(files), size=top_data, replace=False)

        for f_i in range(len(files)):
            if f_i not in indices:
                continue
            task_id, example_id, group_type, group_id, data_type = files[f_i].split("_")
            data_type = data_type.split(".")[0]
            matrix = data[task_id][data_type][int(example_id)][group_type][int(group_id)]
            matrix = data_utils.patch2tensor(matrix)
            rows, cols = matrix.shape
            if rows > 4 and cols > 4:
                dataset.append((matrix, label))


if __name__ == "__main__":
    # label_name = "line"
    top_data = 580
    exp_name = "kp_cha_01"
    data_path = config.kp_dataset / exp_name / f'train'
    data = get_data(data_path, top_data)
    main(label_name, top_data)