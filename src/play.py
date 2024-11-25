# Created by jing at 22.10.24
import os
# import logging
import colorlog

import config
from train_nsfr import train_clauses
from eval_nsfr import check_clause
import llama_call
from utils import file_utils, args_utils
from kandinsky_generator import generate_training_patterns, generate_task_patterns
from src.alpha.fol import bk

# Create a color handler
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)

# Add the color handler to the logger
logger = colorlog.getLogger("colorLogger")
logger.addHandler(handler)
# Prevent logs from propagating to the root logger
logger.propagate = False
logger.setLevel(colorlog.DEBUG)


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
    args = args_utils.get_args(logger)
    exp_setting = bk.exp_count_group
    data_folder = config.kp_dataset / args.exp_name
    init_io_folders(args, data_folder)
    step_counter = 0
    total_step = 8

    # Generate Training Data -- Single Group Pattern
    step_counter += 1
    logger.info(f"Step {step_counter}/{total_step}: "
                f"Generating {exp_setting['bk_groups']} training patterns")
    generate_training_patterns.genShapeOnShape(exp_setting["bk_groups"], 500)

    # Generate Task Data -- Multiple Group Pattern
    step_counter += 1
    logger.info(f"Step {step_counter}/{total_step}: "
                f"Generating {exp_setting['task_name']} task patterns")
    generate_task_patterns.genShapeOnShapeTask(exp_setting, 10)

    # Import Generated Data
    step_counter += 1
    logger.info(f"Step {step_counter}/{total_step}: "
                f"Importing training and testing data.")

    train_imges = file_utils.get_all_files(args.train_folder, "png", False)[:500]
    positive_images = file_utils.get_all_files(args.test_true_folder, "png", False)[:500]
    random_imges = file_utils.get_all_files(args.test_random_folder, "png", False)[:500]
    counterfactual_imges = file_utils.get_all_files(args.test_cf_folder, "png", False)[:500]

    # Learn Clauses from Training Data
    step_counter += 1
    lang = train_clauses(args, train_imges, args.out_train_folder)
    logger.info(f"Step {step_counter}/{total_step}: "
                f"Reasoned {len(lang.llm_clauses)} LLM Rules, "
                f"{len(lang.clauses)} Machine Clauses")

    # Test Positive Patterns, statistic the accuracy, return the satisfied and dissatisfied rules for each test data.
    step_counter += 1
    positive_acc = check_clause(args, lang, positive_images, "POSITIVE", args.out_positive_folder)
    logger.info(f"\n"
                f"Step {step_counter}/{total_step}: Test Positive Images\n"
                f"Confidence for each image: {positive_acc}\n"
                f"Average Accuracy: {positive_acc.mean(dim=0):.2f}\n")


    # Step 6: Test counterfactual patterns
    step_counter += 1

    cf_acc = check_clause(args, lang, counterfactual_imges, "NEGATIVE", args.out_cf_folder)
    logger.info(f"\n"
                f"Step {step_counter}/{total_step}: "
                f"Test Counterfactual Image Accuracy: {cf_acc.mean():.2f}\n"
                f"Confidence for each image: "
                f"{cf_acc}")

    # Step 7: Test random patterns
    step_counter += 1

    random_acc = check_clause(args, lang, random_imges, "NEGATIVE", args.out_random_folder)
    logger.info(f"\n"
                f"Step {step_counter}/{total_step}: "
                f"Random Image accuracy: {random_acc.mean():.2f}\n"
                f"Confidence for each image: {random_acc}")

    # final logger
    logger.info(f"\n"
                f"======================= Test Images Accuracy ============================"
                f"[ pos|cf|rand {positive_acc.mean():.2f} | {cf_acc.mean():.2f} | {random_acc.mean():.2f} ]\n"
                f"========================== End of the Program ====================================")

    return

if __name__ == '__main__':
    main()
