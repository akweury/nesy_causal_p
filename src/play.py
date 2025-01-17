# Created by jing at 22.10.24
import os

import config
import train_nsfr
from eval_nsfr import check_clause
from kandinsky_generator import generate_training_patterns, generate_task_patterns
from utils import args_utils
from src import dataset, bk
from src.neural import models
from src.percept import perception
from src.utils.chart_utils import van


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
    args.output_file_prefix = config.models / "tmp"
    os.makedirs(args.output_file_prefix, exist_ok=True)
    # generate dataset
    generate_training_patterns.genGestaltTraining()

    # Identify feature maps
    perception.collect_fms(args)

    # Import Generated Data
    data_loader = dataset.load_dataset(args)

    for task_id, (train_data, test_data) in enumerate(data_loader):
        args.output_file_prefix = config.models / f"task_{task_id}_"
        imgs_train = train_data["img"]
        imgs_test = test_data["img"]
        # grouping objects
        groups = perception.cluster_by_principle(args, imgs_train)
        # Learn Clauses from Training Data
        lang, rules = train_nsfr.train_clauses(args, groups)

        # Test Patterns, statistic the accuracy
        acc = check_clause(args, lang, rules, imgs_test)
        acc_baseline = None
        acc_rand = None
        # final logger
    args.logger.info(f"\n"
                     f"================ Test Images Accuracy ======================"
                     f"[ pos|cf|rand "
                     f"{acc:.2f} | {acc_baseline:.2f} | {acc_rand:.2f} ]\n"
                     f"================ End of the Program ======================")

    return


if __name__ == '__main__':
    main()
