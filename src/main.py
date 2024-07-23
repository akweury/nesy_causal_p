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

    for task_i in tqdm(range(93, len(raw_data["train"]["cha"])), desc="Reasoning Training Dataset"):
        task = raw_data["train"]["cha"][task_i]["train"]
        task_features = []
        task_relations = []
        for e_i in range(len(task)):
            example = task[e_i]
            # acquire the probability of grouping type: color/shape/area/...
            example_features = grouping.percept_task_features(args, example)
            # example_features = grouping.percept_task_features(args, example)
            example_relations = alpha.alpha(args, example_features, config.alpha_mode['inter_io_group'])
            task_features.append(example_features)
            task_relations.append(example_relations)

        print("task i")
        # hlps = llm.generate_hlps(task_relations)

    print("program finished")


if __name__ == "__main__":
    main()
