# Created by X at 22.10.24
import os
from pathlib import Path
import config
from src import train_nsfr
from src.eval_nsfr import check_clause
from src.utils import args_utils
from src import dataset
from src.percept import perception
from src import llama_call
from kandinsky_generator import generate_training_patterns


def init_io_folders(args, data_folder):
    args.train_folder = data_folder / "train" / "task_true_pattern"
    os.makedirs(args.train_folder, exist_ok=True)
    args.test_true_folder = data_folder / "test" / "task_true_pattern"
    os.makedirs(args.test_true_folder, exist_ok=True)
    args.test_random_folder = data_folder / "test" / "task_random_pattern"
    os.makedirs(args.test_random_folder, exist_ok=True)
    args.test_cf_folder = data_folder / "test" / "task_cf_pattern"
    os.makedirs(args.test_cf_folder, exist_ok=True)

    exp_name = args.exp_setting["task_name"]
    args.out_train_folder = config.output / exp_name / "train" / "task_true_pattern"
    os.makedirs(args.out_train_folder, exist_ok=True)
    args.out_positive_folder = config.output / exp_name / "test" / "task_true_pattern"
    os.makedirs(args.out_positive_folder, exist_ok=True)
    args.out_random_folder = config.output / exp_name / "test" / "task_random_pattern"
    os.makedirs(args.out_random_folder, exist_ok=True)
    args.out_cf_folder = config.output / exp_name / "test" / "task_cf_pattern"
    os.makedirs(args.out_cf_folder, exist_ok=True)


def main():
    # load exp arguments
    args = args_utils.get_args()
    args.step_counter = 0
    args.total_step = 8

    # os.makedirs(args.output_file_prefix, exist_ok=True)
    # generate dataset
    generate_training_patterns.genGestaltTraining()
    # Import Generated Data
    data_loader = dataset.load_dataset(args)

    # Identify feature maps
    perception.collect_fms(args)
    # perception.test_fms(args, data_loader)

    for task_id, (train_data, test_data, principle) in enumerate(data_loader):
        if task_id != 0:
            continue
        args.output_file_prefix = config.models / f"t{task_id}_"
        imgs_train = train_data["img"]
        imgs_test = test_data["img"]
        # perception.test_od_accuracy(args, train_data)
        # grouping objects
        groups = perception.cluster_by_principle(args, imgs_train, "train", principle[0])
        # Learn Clauses from Training Data
        lang_obj, lang_group, rules = train_nsfr.train_clauses(args, groups)
        for rule in rules:
            print("Rule:", rule)
        # Test Patterns, statistic the accuracy
        check_results = check_clause(args, lang_obj, lang_group, rules, imgs_test, principle[0])
        # convert to natural language
        natural_rules = llama_call.convert_to_final_clauses(args, rules, check_results, principle[0], task_id)

    return


if __name__ == '__main__':
    main()
