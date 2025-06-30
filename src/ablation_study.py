# Created by MacBook Pro at 12.06.25
# ablation_main.py
import os
import json
import time

import torch
import wandb
from collections import defaultdict
from src.utils import args_utils
from src import dataset
from mbg.object import eval_patch_classifier
from mbg.training import training
from mbg.evaluation import evaluation
import config
from mbg.scorer import scorer_config

ABLATED_CONFIGS = {
    "hard_ogd": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": True, "use_deepproblog": True},

    "hard_obj": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False, "use_calibrator": False, "use_deepproblog": False},
    "hard_obj_calib": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False, "use_calibrator": True, "use_deepproblog": False},
    "hard_og": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": False, "use_deepproblog": False},
    "hard_ogc": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": True, "use_deepproblog": False},

}


def run_ablation(train_data, val_data, test_data, obj_model, group_model, train_principle, args, mode_name,
                 ablation_flags):
    task_name = train_data["task"]
    train_val_data = {
        "task": task_name,
        "positive": train_data["positive"] + val_data["positive"],
        "negative": train_data["negative"] + val_data["negative"]
    }
    hyp_params = {"prox": 0.9, "sim": 0.5,
                  "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [
        1] * len(train_val_data["positive"]) + [0] * len(train_val_data["negative"])

    t1 = time.time()
    # train rule + calibrator
    hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, group_model,
                                                                         hyp_params,
                                                                         train_principle, args.device, ablation_flags)
    t2 = time.time()
    base_rules = training.train_rules(hard, soft, obj_list, group_list, group_nums, train_img_labels, hyp_params,
                                      ablation_flags)
    t3 = time.time()
    final_rules = training.extend_rules(
        base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)
    t4 = time.time()
    
    
    calibrator = training.train_calibrator(final_rules, obj_list, group_list, hard, soft, train_img_labels, hyp_params,
                                           ablation_flags, args.device)
    t5 = time.time()
    eval_metrics = evaluation.eval_rules(test_data, obj_model, group_model, final_rules, hyp_params, train_principle,
                                         args.device, calibrator, ablation_flags)
    t6 = time.time()

    # d1 = t2 - t1  # grounding facts
    # d2 = t3 - t2  # base rules
    # d3 = t4 - t3  # extend rules
    # d4 = t5 - t4  # train calibrator
    # d5 = t6 - t5  # eval rules
    # print(f"Grounding facts in {d1} seconds")
    # print(f"Base Rules in {d2} seconds")
    # print(f"Extension in {d3} seconds")
    # print(f"Calibrator in {d4} seconds")
    # print(f"Evaluation Metrics in {d5} seconds")
    return eval_metrics


def main_ablation():
    args = args_utils.get_args()
    train_principle = args.principle
    principle_path = getattr(config, f"grb_{train_principle}")
    combined_loader = dataset.load_combined_dataset(principle_path)
    obj_model = eval_patch_classifier.load_model(args.device)
    group_model = scorer_config.load_scorer_model(train_principle, args.device)

    # wandb.init(project=f"grb_ablation_{train_principle}",
            #    config=args.__dict__, name=args.exp_name)
    # setting -> metric -> list
    results_summary = defaultdict(lambda: defaultdict(list))
    all_f1 = {conf: [] for conf in ABLATED_CONFIGS}
    all_auc = {conf: [] for conf in ABLATED_CONFIGS}
    all_acc = {conf: [] for conf in ABLATED_CONFIGS}
    for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
        task_name = train_data["task"]
        print(f"\nTask {task_idx + 1}/{len(combined_loader)}: {task_name}")

        log_dicts = {}
        for mode_name, ablation_flags in ABLATED_CONFIGS.items():

            t1 = time.time()
            test_metrics = run_ablation(train_data, val_data, test_data, obj_model, group_model,
                                        train_principle, args, mode_name, ablation_flags)
            t2 = time.time()
            print(f"  Running ablation: {mode_name} in {t2 - t1} seconds")
            for k in ["acc", "f1", "auc"]:
                results_summary[mode_name][k].append(test_metrics.get(k, 0))
                # print(f"task: {task_idx + 1}/{len(combined_loader)}: {k} {test_metrics.get(k, 0)}")
            test_acc = test_metrics.get("acc", 0)
            test_auc = test_metrics.get("auc", 0)
            test_f1 = test_metrics.get("f1", 0)

            all_f1[mode_name].append(test_f1)
            all_auc[mode_name].append(test_auc)
            all_acc[mode_name].append(test_acc)

            # log_dicts.update({f"{mode_name}_{k}": test_metrics.get(k, 0) for k in test_metrics})
            log_dicts.update({f"{mode_name}_acc_avg": torch.tensor(all_acc[mode_name]).mean(),
                              # f"{mode_name}_auc_avg": torch.tensor(all_auc[mode_name]).mean(),
                              # f"{mode_name}_f1_avg": torch.tensor(all_f1[mode_name]).mean()
                              })

        wandb.log(log_dicts)

    # save and summarize
    final_summary = {
        mode: {f"avg_{k}": float(torch.tensor(v).mean())
               for k, v in metric_dict.items()}
        for mode, metric_dict in results_summary.items()
    }
    with open(f"ablation_summary_{args.exp_name}.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print("\n=== Final Summary ===")
    for mode, metrics in final_summary.items():
        print(f"{mode}: {metrics}")

    wandb.finish()


if __name__ == "__main__":
    main_ablation()
