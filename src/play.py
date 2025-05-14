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
from collections import Counter, defaultdict, namedtuple
from mbg.grounding.predicates import HEAD_PREDICATES
from collections import Counter

import psutil
import pynvml
import time
import threading
from mbg.object import eval_patch_classifier
from mbg.group import eval_groups
from mbg.grounding import grounding
from mbg.language import clause_generation


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

    # a little helper to key on head+body pairs
    RuleKey = namedtuple("RuleKey", ["head", "body"])

    # 1) 收集每张图的频次 & group 数
    pos_per_task = defaultdict(list)  # task_id -> List[Counter[Clause,int]] (正例)
    neg_per_task = defaultdict(list)  # task_id -> List[Counter[Clause,int]] (负例)
    pos_group_counts = defaultdict(list)  # task_id -> List[int] 每张正例图的 group 数
    neg_group_counts = defaultdict(list)  # task_id -> List[int] 每张负例图的 group 数

    for data in data_loader:
        # --- 1. 基础信息读取 ---
        task_id = data["task"][0].split("_")[0]
        img_label = int(data["img_label"])  # 1 or 0

        # --- 2. 物体 & 分组检测 ---
        objs = eval_patch_classifier.evaluate_image(obj_model, data)
        groups = eval_groups.eval_groups(objs, data["symbolic_data"]["proximity"])
        num_groups = len(groups)

        # --- 3. Grounding & Clause Generation ---
        hard, soft = grounding.ground_facts(objs, groups)
        cg = clause_generation.ClauseGenerator()
        clauses = cg.generate(hard, soft)
        freq = Counter(clauses)

        # --- 4. 存入对应容器 ---
        if img_label == 1:
            pos_per_task[task_id].append(freq)
            pos_group_counts[task_id].append(num_groups)
        else:
            neg_per_task[task_id].append(freq)
            neg_group_counts[task_id].append(num_groups)

    # for data in data_loader:
    #     task_id = data["task"][0].split("_")[0]
    #     label = int(data["img_label"])  # 1 or 0
    #
    #     # detect objects
    #     objs = eval_patch_classifier.evaluate_image(obj_model, data)
    #     # detect groups; multiple groups in one image
    #     groups = eval_groups.eval_groups(objs, data["symbolic_data"]["proximity"])
    #     num_groups = len(groups)
    #
    #     # --- ground & generate ---
    #     hard, soft = grounding.ground_facts(objs, groups)
    #
    #     gen = clause_generation.ClauseGenerator()
    #     c_list = gen.generate(hard, soft)
    #     # Now count frequencies:
    #     freq = Counter(c_list)
    #
    #     if label == 1:
    #         pos_per_task[task_id].append(freq)
    #     else:
    #         neg_per_task[task_id].append(freq)


        # # now extract just the group‐target clauses & count them
        # grp_head = HEAD_PREDICATES["group"]   # e.g. "group_target"
        # group_rks = [ rk for rk in candidates if rk.head[0] == grp_head ]
        # ctr       = Counter(group_rks)
        #
        # # record (Counter, total_groups) for each image
        # if label == 1:
        #     pos_group_data[task_id].append((ctr, num_groups))
        # else:
        #     neg_group_data[task_id].append((ctr, num_groups))

    img_rules = clause_generation.filter_image_level_rules(pos_per_task,neg_per_task)
    exist_rules = clause_generation.filter_group_existential_rules(pos_per_task, neg_per_task)
    univ_rules = clause_generation.filter_group_universal_rules(pos_per_task,neg_per_task, pos_group_counts, neg_group_counts)

    pos_image_counts, pos_group_counts, neg_image_union, neg_group_counts = clause_generation.split_clauses_by_head(pos_per_task,neg_per_task)

    # compute number of positives per task:
    num_pos = {t: len(pos_image_counts[t]) for t in pos_image_counts}
    image_rules = clause_generation.finalize_image_level_rules(
        pos_image_counts=pos_image_counts,
        neg_image_union=neg_image_union,
        num_pos_images=num_pos
    )
    group_universal_rules = clause_generation.filter_group_universal(pos_group_data, neg_group_data)
    group_existential_rules = clause_generation.filter_group_existential(pos_group_counts, neg_group_counts)
    final_rules_per_task = clause_generation.finalize_rules_per_task(pos_per_task, neg_per_task)

    print(final_rules_per_task)
    #
    # # Learn Clauses from Training Data
    # # Start CPU memory monitoring
    # stop_monitor = monitor_cpu_memory()
    #
    # lang_obj, lang_group, rules = train_nsfr.train_clauses(args, groups)
    #
    # # Stop monitoring and print peak memory
    # peak_cpu = stop_monitor()
    # print(f"Peak CPU memory during train_clauses: {peak_cpu:.2f} MB")
    #
    # for rule in rules:
    #     print("Rule:", rule["rule"])
    # # Test Patterns, statistic the accuracy
    # check_results = check_clause(args, lang_obj, lang_group, rules, imgs_test, principle[0])
    # # convert to natural language
    # natural_rules = llama_call.convert_to_final_clauses(args, rules, check_results, principle[0], task_id)

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
