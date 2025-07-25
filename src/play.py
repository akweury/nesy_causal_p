# Created by X at 22.10.24
import os
import config
import psutil
import pynvml
import time
import threading
import wandb
from collections import defaultdict
import json

import torch
from src.utils import args_utils
from src import dataset

from mbg.object import eval_patch_classifier
from mbg.training import training
from mbg.evaluation import evaluation


def main():
    # load exp arguments
    args = args_utils.get_args()

    train_principle = args.principle

    if train_principle == "similarity":
        principle_path = config.grb_similarity
    elif train_principle == "proximity":
        principle_path = config.grb_proximity
    elif train_principle == "closure":
        principle_path = config.grb_closure
    elif train_principle == "continuity":
        principle_path = config.grb_continuity
    elif train_principle == "symmetry":
        principle_path = config.grb_symmetry
    else:
        raise ValueError

    combined_loader = dataset.load_combined_dataset(principle_path)
    obj_model = eval_patch_classifier.load_model(args.device)

    # initialize wandb
    # wandb.init(project=f"grb_{train_principle}", config=args.__dict__, name=args.exp_name)

    # store metrics per property value
    property_stats = defaultdict(lambda: defaultdict(list))  # {prop_name: {True: [], False: []}}
    all_f1 = []
    all_auc = []
    all_acc = []
    for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
        if task_idx < 127:
            continue
        task_name = train_data["task"]
        properties = {
            "non_overlap": train_data["non_overlap"],
            "qualifier_all": train_data["qualifier_all"],
            "qualifier_exist": train_data["qualifier_exist"],
            "prop_shape": train_data["prop_shape"],
            "prop_color": train_data["prop_color"],
            "prop_size": train_data["prop_size"],
            "prop_count": train_data["prop_count"],
        }

        # merge train + val
        train_val_data = {
            "task": task_name,
            "positive": train_data["positive"] + val_data["positive"],
            "negative": train_data["negative"] + val_data["negative"]
        }

        hyp_params = {"prox": 0.9, "sim": 0.5, "top_k": 5, "conf_th": 0.5, "patch_dim": 7}

        # train
        hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, hyp_params,
                                                                             train_principle, args.device)
        train_img_labels = [1] * (len(group_nums) // 2) + [0] * (len(group_nums) // 2)
        base_rules = training.train_rules(hard, soft, group_nums, train_img_labels, hyp_params)
        final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)
        calibrator = training.train_calibrator(final_rules, obj_list, group_list, hard, soft, train_img_labels,
                                               hyp_params)

        # test
        test_metrics = evaluation.eval_rules(test_data, obj_model, final_rules, hyp_params, train_principle,
                                             args.device, calibrator)

        test_acc = test_metrics.get("acc", 0)
        test_auc = test_metrics.get("auc", 0)
        test_f1 = test_metrics.get("f1", 0)

        all_f1.append(test_f1)
        all_auc.append(test_auc)
        all_acc.append(test_acc)

        # log raw results
        wandb.log({
            "test_accuracy": test_acc,
            "test_auc": test_auc,
            "test_f1": test_f1,
            "avg_acc": torch.tensor(all_acc).mean(),
            "avg_auc": torch.tensor(all_auc).mean(),
            "avg_f1": torch.tensor(all_f1).mean(),
        })
        print(f"{task_idx + 1}/{combined_loader.__len__()}[{task_name}] Test results:", test_metrics)

        # accumulate statistics
        for prop_name, prop_value in properties.items():
            property_stats[prop_name][prop_value].append(test_metrics)
    # analyze and log aggregated statistics
    analysis_summary = {}
    for prop_name, value_dict in property_stats.items():
        for value in [True, False]:
            metrics_list = value_dict.get(value, [])
            if metrics_list:
                avg_acc = sum(m.get("acc", 0) for m in metrics_list) / len(metrics_list)
                avg_f1 = sum(m.get("f1", 0) for m in metrics_list) / len(metrics_list)
                avg_auc = sum(m.get("auc", 0) for m in metrics_list) / len(metrics_list)

                key_prefix = f"{prop_name}_{value}"
                analysis_summary[key_prefix] = {
                    "avg_acc": avg_acc,
                    "avg_f1": avg_f1,
                    "avg_auc": avg_auc,
                    "count": len(metrics_list)
                }
    # save analysis to file
    with open(config.output / f"analysis_summary_{args.principle}.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)

    wandb.finish()



if __name__ == '__main__':
    main()
