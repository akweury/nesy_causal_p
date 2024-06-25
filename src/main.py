# Created by jing at 17.06.24
from tqdm import tqdm

import config
import grouping, visual
from alpha import alpha
from utils import visual_utils, file_utils, args_utils

# arguments
args = args_utils.get_args()

# data file
raw_data = file_utils.get_raw_data()
train_cha = raw_data["train_cha"]
train_sol = raw_data["train_sol"]


def main():
    args = args_utils.get_args()
    # data file
    raw_data = file_utils.get_raw_data()

    color_groups_cha = grouping.group_by_color(raw_data["train_cha"])
    visual.export_groups_as_images(raw_data["train_cha"], color_groups_cha, "train_cha")

    for task_i in tqdm(range(len(raw_data["train_cha"])), desc="Reasoning Task"):
        task = raw_data["train_cha"][task_i]
        # acquire the probability of grouping type: color/shape/area/...
        task_features = grouping.percept_task_features(args, task)
        task_relations = alpha.alpha(args, task_features)

    print("program finished")


if __name__ == "__main__":
    main()
