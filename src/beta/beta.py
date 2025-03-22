# Created by MacBookPro at 11.03.25
import os
import pickle
from args_utils import get_args
from dataset.loader import load_dataset_tasks
from dataset.image_utils import load_image, load_annotation
from features.detector import detect_features, beta_detector
from features.encoder import encode_features, beta_encoder
from reasoning.rules import reasoning_rules
from tokens.converter import convert_rules_to_tokens
from save_utils import save_results

import beta_config


def process_task(args, task):
    """
    Process a single task: load samples, detect and encode features,
    apply reasoning rules, convert them to natural language tokens, and save the results.
    """
    features = beta_detector(task)
    encoded_features = beta_encoder(features)
    rules = reasoning_rules(args, encoded_features)
    tokens = convert_rules_to_tokens(rules)
    return tokens


def process_all_tasks(args, dataset):
    """
    Process all tasks one by one and store the results in a nested dictionary.
    """
    output_dir, principles, splits = beta_config.OUTPUT_DIR, beta_config.GESTALT_PRINCIPLES, beta_config.SPLITS
    final_tokens = {}
    for principle in principles:
        final_tokens[principle] = {}
        for task_name, task in dataset[principle].items():
            print(f"Processing {principle} - {task_name} ...")
            tokens = process_task(args, task["train"])
            if task_name not in final_tokens[principle]:
                final_tokens[principle][task_name] = {}
            final_tokens[principle][task_name] = tokens
            # Save tokens for this task
            task_save_path = os.path.join(output_dir, f"{principle}_{task_name}_tokens.pkl")
            save_results(tokens, task_save_path)
            print(f"Saved tokens for {principle} - {task_name} to {task_save_path}")
        # Save overall tokens.
        save_results(final_tokens)
    return final_tokens


def main():
    args = get_args()
    # Load dataset tasks using configuration parameters from beta_config.py.
    dataset = load_dataset_tasks()
    # Process each task individually.
    process_all_tasks(args, dataset)


if __name__ == "__main__":
    main()
