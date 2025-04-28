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

import psutil
import pynvml
import time
import threading
from mbg.object import eval_patch_classifier
from mbg.group import eval_groups
from mbg.group import proximity_grouping
from mbg.scorer import scorer_config
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
    data_loader = dataset.load_dataset(args, mode="train")
    # check the individual object detection
    # evaluate_object_detection.evaluate_symbolic_detection(data_loader, predictor)


    # Identify feature maps
    # perception.collect_fms(args)
    obj_model = eval_patch_classifier.load_model(args.device)
    for task_id, data in enumerate(data_loader):
        args.output_file_prefix = config.models / f"t{task_id}_"

        # detect objects
        objs = eval_patch_classifier.evaluate_image(obj_model, data)
        # detect groups; multiple groups in one image
        groups = eval_groups.eval_groups(objs, data["symbolic_data"]["proximity"])

        # Learn Clauses from Training Data
        # Start CPU memory monitoring
        stop_monitor = monitor_cpu_memory()

        lang_obj, lang_group, rules = train_nsfr.train_clauses(args, groups)

        # Stop monitoring and print peak memory
        peak_cpu = stop_monitor()
        print(f"Peak CPU memory during train_clauses: {peak_cpu:.2f} MB")

        for rule in rules:
            print("Rule:", rule["rule"])
        # Test Patterns, statistic the accuracy
        check_results = check_clause(args, lang_obj, lang_group, rules, imgs_test, principle[0])
        # convert to natural language
        natural_rules = llama_call.convert_to_final_clauses(args, rules, check_results, principle[0], task_id)

    return


def monitor_cpu_memory(interval=0.1):
    process = psutil.Process()
    peak_cpu = 0
    running = True

    def monitor():
        nonlocal peak_cpu
        while running:
            mem = process.memory_info().rss / 1024 ** 2  # Convert to MB
            peak_cpu = max(peak_cpu, mem)
            time.sleep(interval)

    thread = threading.Thread(target=monitor)
    thread.start()

    def stop():
        nonlocal running
        running = False
        thread.join()
        return peak_cpu

    return stop


def monitor_memory(interval=0.1):
    process = psutil.Process()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    peak_cpu, peak_gpu = 0, 0
    running = True

    def monitor():
        nonlocal peak_cpu, peak_gpu
        while running:
            cpu_mem = process.memory_info().rss / 1024 ** 2  # MB
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 2  # MB
            peak_cpu = max(peak_cpu, cpu_mem)
            peak_gpu = max(peak_gpu, gpu_mem)
            time.sleep(interval)

    thread = threading.Thread(target=monitor)
    thread.start()
    return thread, lambda: setattr(thread, 'running', False), lambda: (peak_cpu, peak_gpu), lambda: setattr(globals(),
                                                                                                            'running',
                                                                                                            False)


if __name__ == '__main__':
    main()
