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


def init_io_folders(args, data_folder):
    args.train_folder = data_folder / "train" / "task_true_pattern"
    os.makedirs(args.train_folder, exist_ok=True)
    args.test_true_folder = data_folder / "test" / "task_true_pattern"
    os.makedirs(args.test_true_folder, exist_ok=True)
    args.test_random_folder = data_folder / "test" / "task_random_pattern"
    os.makedirs(args.test_random_folder, exist_ok=True)
    args.test_cf_folder = data_folder / "test" / "task_cf_pattern"
    os.makedirs(args.test_cf_folder, exist_ok=True)

    args.out_train_folder = config.output / args.exp_name / "train" / "task_true_pattern"
    os.makedirs(args.out_train_folder, exist_ok=True)
    args.out_positive_folder = config.output / args.exp_name / "test" / "task_true_pattern"
    os.makedirs(args.out_positive_folder, exist_ok=True)
    args.out_random_folder = config.output / args.exp_name / "test" / "task_random_pattern"
    os.makedirs(args.out_random_folder, exist_ok=True)
    args.out_cf_folder = config.output / args.exp_name / "test" / "task_cf_pattern"
    os.makedirs(args.out_cf_folder, exist_ok=True)


def main():
    # load exp arguments
    args = args_utils.get_args()
    exp_setting = bk.exp_triangle_group
    data_folder = config.kp_challenge_dataset / args.exp_name
    init_io_folders(args, data_folder)
    args.step_counter = 0
    args.total_step = 8

    # Generate Training Data -- Single Group Pattern
    generate_training_patterns.genShapeOnShape(args, exp_setting["bk_groups"], 100)

    # Generate Task Data -- Multiple Group Pattern
    generate_task_patterns.genShapeOnShapeTask(args, exp_setting, 10)

    # Identify feature maps
    perception.collect_fms(args, exp_setting["bk_groups"])

    # Train autoencoder
    models.train_autoencoder(args, exp_setting["bk_groups"])

    # Import Generated Data
    train_dl, test_pos_dl, test_rand_dl, test_cf_dl = dataset.load_dataset(args)

    # Learn Clauses from Training Data
    lang = train_nsfr.train_clauses(args, train_dl)

    # Test Positive Patterns, statistic the accuracy,
    acc_pos = check_clause(args, lang, test_pos_dl, "POSITIVE",
                           args.out_positive_folder)

    # Test counterfactual patterns
    acc_cf = check_clause(args, lang, test_cf_dl, "NEGATIVE",
                          args.out_cf_folder)

    # Step 7: Test random patterns
    acc_rand = check_clause(args, lang, test_rand_dl, "NEGATIVE",
                            args.out_random_folder)

    # final logger
    args.logger.info(f"\n"
                     f"================ Test Images Accuracy ======================"
                     f"[ pos|cf|rand "
                     f"{acc_pos:.2f} | {acc_cf:.2f} | {acc_rand:.2f} ]\n"
                     f"================ End of the Program ======================")

    return


if __name__ == '__main__':
    main()
