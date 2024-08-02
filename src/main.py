# Created by jing at 17.06.24
from tqdm import tqdm
import torch
import numpy as np
import os

import config
import grouping
from percept import perception
from utils import visual_utils, file_utils, args_utils, data_utils
from reasoning import reasoning


def percept_objs(args, task):
    task_objs = []
    for e_i in range(len(task)):
        example = task[e_i]
        # acquire the probability of grouping type: color/shape/area/...
        objs = perception.percept_objs(args, example)
        task_objs.append(objs)
    return task_objs


def reasoning_obj_relations(objs):
    relations = None
    return relations


def prepare_kp_sy_data(args):
    data_path = config.kp_dataset / args.exp_name
    dataset = []

    files = file_utils.get_all_files(data_path / "true", "png", True)
    indices = np.random.choice(len(files), size=args.top_data, replace=False)


    label = torch.tensor(config.obj_true)

    for f_i in range(len(files)):
        if f_i not in indices:
            continue
        file_name, file_extension = files[f_i].split(".")
        data = file_utils.load_json(data_path / "true" / f"{file_name}.json")
        if len(data) > 16:
            patch = data_utils.oco2patch(data).unsqueeze(0)
            dataset.append((patch, label))

    return dataset


def main():
    args = args_utils.get_args()
    # data file
    dataset = prepare_kp_sy_data(args)
    os.makedirs(config.output / f"kp_sy_{args.exp_name}", exist_ok=True)

    for task_i in tqdm(range(len(dataset)), desc="Reasoning Training Dataset"):
        # percept objs in a task
        objs = percept_objs(args, dataset[task_i])
        relations = reasoning_obj_relations(objs)
        print(f"task {task_i}")

    print("program finished")


if __name__ == "__main__":
    main()
