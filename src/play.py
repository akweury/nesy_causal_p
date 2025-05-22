# Created by X at 22.10.24
import os
import config
import psutil
import pynvml
import time
import threading

from src.utils import args_utils
from src import dataset
from kandinsky_generator import generate_training_patterns
from mbg.object import eval_patch_classifier


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



    # Import Generated Data
    train_loader = dataset.load_dataset(args, data=config.grb_prox/"train", mode='train')
    val_loader = dataset.load_dataset(args, data=config.grb_prox/"train", mode="val")
    test_loader = dataset.load_dataset(args, data=config.grb_prox/"test", mode='test')

    obj_model = eval_patch_classifier.load_model(args.device)

    from mbg.training import training
    # 1) hyperparam search
    best_prox, best_sim, best_topk = training.grid_search(args, train_loader, val_loader, obj_model)

    # 2) merge train+val into a single loader
    train_val_loader = dataset.load_train_val_dataset(args, train_loader, val_loader)

    # 3) re-learn your final rules on the combined set
    hyp_params = {"prox": best_prox, "sim": best_sim, "top_k": best_topk, "conf_th":0.5}
    final_rules = training.train_rules(train_val_loader, obj_model, hyp_params)

    # 4) evaluate on the held-out test set

    from mbg.evaluation import evaluation
    test_metrics = evaluation.eval_rules(test_loader, obj_model, final_rules, hyp_params)
    print("Test results:", test_metrics)
    # # ───────────────────────────────────────────────────────────────
    # # STEP C: FINAL RULE INDUCTION on TRAIN+VAL then TEST EVAL
    # # ───────────────────────────────────────────────────────────────
    # pos_per_task_train, neg_per_task_train, pos_group_counts_train, neg_group_counts_train = raw_rules_train
    #
    # # (Optionally merge train+val examples and re‐induce with your tuned settings)
    # all_pos = defaultdict(list, {**pos_per_task_train})
    # all_neg = defaultdict(list, {**neg_per_task_train})
    # all_pos_groups = defaultdict(list, {**pos_group_counts_train})
    # all_neg_groups = defaultdict(list, {**neg_group_counts_train})
    #
    # for data in val_loader:
    #     task_id = data["task"][0].split("_")[0]
    #     img_label = int(data["img_label"])
    #     objs = eval_patch_classifier.evaluate_image(obj_model, data)
    #     groups = eval_groups.eval_groups(objs, data["symbolic_data"]["proximity"])
    #     hard, soft = grounding.ground_facts(objs, groups)
    #     freq = Counter(clause_generation.ClauseGenerator(
    #         prox_thresh=best_prox,
    #         sim_thresh=best_sim
    #     ).generate(hard, soft))
    #
    #     gcount = len(groups)
    #     if img_label == 1:
    #         all_pos[task_id].append(freq)
    #         all_pos_groups[task_id].append(gcount)
    #     else:
    #         all_neg[task_id].append(freq)
    #         all_neg_groups[task_id].append(gcount)
    #
    # # re‐filter with best hyper‐params
    # img_rules_final = clause_generation.filter_image_level_rules(all_pos, all_neg)
    # exist_rules_final = clause_generation.filter_group_existential_rules(all_pos, all_neg)
    # univ_rules_final = clause_generation.filter_group_universal_rules(
    #     all_pos, all_neg,
    #     all_pos_groups,
    #     all_neg_groups
    # )
    # rules_final = clause_generation.assemble_final_rules(
    #     img_rules_final,
    #     exist_rules_final,
    #     univ_rules_final
    # )
    #
    # # persist final rules
    # clause_generation.export_rules_to_json(rules_final, config.output)
    # print("Final rules:", rules_train)
    #
    # # Evaluate on TEST
    # test_metrics = {}
    # for task_id, rules in rules_final.items():
    #     test_metrics[task_id] = compute_metrics(rules, test_loader)
    # print(json.dumps(test_metrics, indent=2))

    #
    # pos_image_counts, pos_group_counts, neg_image_union, neg_group_counts = clause_generation.split_clauses_by_head(pos_per_task,neg_per_task)
    #
    # # compute number of positives per task:
    # num_pos = {t: len(pos_image_counts[t]) for t in pos_image_counts}
    # image_rules = clause_generation.finalize_image_level_rules(
    #     pos_image_counts=pos_image_counts,
    #     neg_image_union=neg_image_union,
    #     num_pos_images=num_pos
    # )
    # group_universal_rules = clause_generation.filter_group_universal(pos_group_data, neg_group_data)
    # group_existential_rules = clause_generation.filter_group_existential(pos_group_counts, neg_group_counts)
    # final_rules_per_task = clause_generation.finalize_rules_per_task(pos_per_task, neg_per_task)

    # print(final_rules_per_task)
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
