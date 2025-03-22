# Created by X at 12.03.25


import os
from src.beta import beta_config


def load_task(dataset_dir, principle, task):
    """
    Loads tasks from the given directory.

    Expects each task directory to contain 'positive' and 'negative' folders,
    with each folder containing image (.png) and annotation (.json) files.

    Args:
        task_path (str): The directory containing task subdirectories.

    Returns:
        List[Dict]: A list of tasks, where each task is represented as a dictionary:
            - "task": the name of the task (directory name),
            - "polarity": either "positive" or "negative",
            - "samples": a list of dictionaries with keys "image_path" and "annotation_path".
    """
    tasks = {}
    task_dirs = {
        "train": dataset_dir / principle / "train" / task,
        "test": dataset_dir / principle / "test" / task
    }
    # Iterate over each task subdirectory in the given task_path.
    for task_type, task_dir in task_dirs.items():
        tasks[task_type] = {}
        # Process both positive and negative subdirectories.
        for polarity in ["positive", "negative"]:
            tasks[task_type][polarity] = {}
            polarity_dir = os.path.join(task_dir, polarity)
            if not os.path.exists(polarity_dir):
                continue  # Skip if the folder doesn't exist.
            # List files in the polarity directory.
            files = os.listdir(polarity_dir)
            # Identify image and annotation files.
            image_files = sorted([f for f in files if f.lower().endswith(".png")])
            annotation_files = sorted([f for f in files if f.lower().endswith(".json")])
            # Pair up images and annotations.
            tasks[task_type][polarity]["images"] =[]
            tasks[task_type][polarity]["annotations"] = []

            for img, ann in zip(image_files, annotation_files):
                tasks[task_type][polarity]["images"].append(os.path.join(polarity_dir, img))
                tasks[task_type][polarity]["annotations"].append(os.path.join(polarity_dir, ann))
    return tasks


def load_dataset_tasks():
    """
    Loads tasks for each gestalt principle and split using dictionary comprehensions.
    """
    dataset_dir, gestalt_principles, splits = beta_config.DATASET_DIR, beta_config.GESTALT_PRINCIPLES, beta_config.SPLITS
    dataset_tasks = {}
    for principle in gestalt_principles:
        dataset_tasks[principle] = {}
        principle_tasks = os.listdir(dataset_dir / principle / "train")
        for task in principle_tasks:
            dataset_tasks[principle][task] = {}
            dataset_tasks[principle][task] = load_task(dataset_dir, principle, task)

    return dataset_tasks
