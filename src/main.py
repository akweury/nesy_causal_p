# Created by jing at 17.06.24
from tqdm import tqdm

import config
import grouping, visual
from alpha import alpha
from utils import visual_utils, file_utils, args_utils



def main():
    args = args_utils.get_args()
    # data file
    raw_data = file_utils.get_raw_data()
    # g_train = grouping.group_by_color(raw_data["train"])
    # g_eval = grouping.group_by_color(raw_data["eval"])
    # visual.export_groups_as_images(raw_data["train"], g_train, "train")
    for task_i in tqdm(range(len(raw_data["train"]["cha"])), desc="Reasoning Training Dataset"):
        task = raw_data["train"]["cha"][task_i]["train"]
        # acquire the probability of grouping type: color/shape/area/...
        task_features = grouping.percept_task_features(args, task)
        task_relations = alpha.alpha(args, task_features)
        # hlps = llm.generate_hlps(task_relations)

    print("program finished")


if __name__ == "__main__":
    main()
