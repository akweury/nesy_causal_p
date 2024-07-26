# Created by jing at 17.06.24
from tqdm import tqdm

import grouping
from percept import perception
from utils import visual_utils, file_utils, args_utils
from reasoning import reasoning


def percept_objs(args, task):
    task_objs = []
    for e_i in range(len(task)):
        example = task[e_i]
        # acquire the probability of grouping type: color/shape/area/...
        example_features = grouping.percept_task_features(args, example)
        objs = perception.percept_objs(args, example_features)
        task_objs.append(objs)
    return task_objs
def reasoning_obj_relations(objs):

    relations = None
    return relations

def main():
    args = args_utils.get_args()
    # data file
    raw_data = file_utils.get_raw_data()

    for task_i in tqdm(range(93, len(raw_data["train"]["cha"])), desc="Reasoning Training Dataset"):
        task = raw_data["train"]["cha"][task_i]["train"]

        # percept objs in a task
        objs = percept_objs(args, task)
        relations = reasoning_obj_relations(objs)
        print(f"task {task_i}")

    print("program finished")


if __name__ == "__main__":
    main()
