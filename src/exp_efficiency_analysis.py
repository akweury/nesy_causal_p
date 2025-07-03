# Created by MacBook Pro at 03.07.25


import json
import time
import numpy as np
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
from mbg import exp_utils
from src import bk

ABLATED_CONFIGS = {
    "hard_obj_calib": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False, "use_calibrator": True,
                       "use_deepproblog": False},
    "hard_ogc": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": True,
                 "use_deepproblog": False},

}


def run_analysis(train_data, val_data, obj_model, group_model, train_principle, args):
    task_name = train_data["task"]
    train_val_data = {
        "task": task_name,
        "positive": train_data["positive"] + val_data["positive"],
        "negative": train_data["negative"] + val_data["negative"]
    }
    hyp_params = {"prox": 0.9, "sim": 0.5, "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [1] * len(train_val_data["positive"]) + [0] * len(train_val_data["negative"])

    obj_times = torch.zeros(len(train_img_labels))
    group_times = torch.zeros(len(train_img_labels))

    ablation_flags = ABLATED_CONFIGS["hard_obj_calib"]

    hard, soft, group_nums, obj_list, group_list = training.ground_facts(
        train_val_data, obj_model, group_model, hyp_params, train_principle, args.device, ablation_flags, obj_times)

    base_rules = training.train_rules(hard, soft, obj_list, group_list, group_nums, train_img_labels,
                                      hyp_params, ablation_flags, obj_times)
    final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)

    ablation_flags = ABLATED_CONFIGS["hard_ogc"]
    hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, group_model,
                                                                                      hyp_params, train_principle, args.device, ablation_flags, group_times)
    base_rules = training.train_rules(hard, soft, obj_list, group_list, group_nums, train_img_labels, hyp_params,
                                      ablation_flags, group_times)
    final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)

    # Dict: n_obj -> list of (obj_facts, group_facts, n_groups)
    obj_num_stats = {}
    time_stats = {}
    for i, img_hard in enumerate(hard):
        obj_fact_count, group_fact_count, n_obj, n_groups = exp_utils.count_symbolic_facts(img_hard)
        if n_obj not in obj_num_stats:
            obj_num_stats[n_obj] = {"obj_facts": [], "group_facts": [], "groups": []}
            time_stats[n_obj] = {"obj_time": [], "group_time": []}
        obj_num_stats[n_obj]["obj_facts"].append(obj_fact_count)
        obj_num_stats[n_obj]["group_facts"].append(group_fact_count)
        obj_num_stats[n_obj]["groups"].append(n_groups)
        time_stats[n_obj]["obj_time"].append(obj_times[i])
        time_stats[n_obj]["group_time"].append(group_times[i])

    # Compute averages for each object number
    obj_num_avg = {}
    for n_obj, stats in obj_num_stats.items():
        count = len(stats["obj_facts"])
        avg_obj_facts = sum(stats["obj_facts"]) / count if count > 0 else 0.0
        avg_group_facts = sum(stats["group_facts"]) / count if count > 0 else 0.0
        avg_groups = sum(stats["groups"]) / count if count > 0 else 0.0
        std_obj_facts = np.std(stats["obj_facts"]) if count > 0 else 0.0
        std_group_facts = np.std(stats["group_facts"]) if count > 0 else 0.0
        obj_num_avg[n_obj] = (avg_obj_facts, avg_group_facts, avg_groups, std_obj_facts, std_group_facts)
    return obj_num_avg, time_stats


def main_eff_analysis():
    args = args_utils.get_args()
    # Collect all principle names from config that start with 'grb_'
    principle_names = bk.gestalt_principles
    all_results = {}
    all_time_stats = {}

    for train_principle in principle_names:
        principle_path = getattr(config, f"grb_{train_principle}")
        combined_loader = dataset.load_combined_dataset(principle_path, task_num=args.top_data)
        obj_model = eval_patch_classifier.load_model(args.device)
        group_model = scorer_config.load_scorer_model(train_principle, args.device)

        # Dict: n_obj -> list of (obj_facts, group_facts, n_groups) for all tasks
        analysis_dict = {}
        time_dict = {}
        for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
            print(f"\nPrinciple: {train_principle} | Task {task_idx + 1}/{len(combined_loader)}: {train_data['task']}")
            obj_num_avg, time_stats = run_analysis(train_data, val_data, obj_model, group_model, train_principle, args)
            for n_obj, tup in obj_num_avg.items():
                if n_obj not in analysis_dict:
                    analysis_dict[n_obj] = {"obj_facts": [], "group_facts": [], "groups": [], "std_obj_facts": [], "std_group_facts": []}
                analysis_dict[n_obj]["obj_facts"].append(tup[0])
                analysis_dict[n_obj]["group_facts"].append(tup[1])
                analysis_dict[n_obj]["groups"].append(tup[2])
                analysis_dict[n_obj]["std_obj_facts"].append(tup[3])
                analysis_dict[n_obj]["std_group_facts"].append(tup[4])
            for n_obj, tstats in time_stats.items():
                if n_obj not in time_dict:
                    time_dict[n_obj] = {"obj_time": [], "group_time": []}
                time_dict[n_obj]["obj_time"].extend(tstats["obj_time"])
                time_dict[n_obj]["group_time"].extend(tstats["group_time"])

        # Average over all tasks for each object number
        avg_analysis = {}
        for n_obj, stats in analysis_dict.items():
            count = len(stats["obj_facts"])
            avg_obj_facts = sum(stats["obj_facts"]) / count if count > 0 else 0.0
            avg_group_facts = sum(stats["group_facts"]) / count if count > 0 else 0.0
            avg_groups = sum(stats["groups"]) / count if count > 0 else 0.0
            std_obj_facts = np.std(stats["obj_facts"]) if count > 0 else 0.0
            std_group_facts = np.std(stats["group_facts"]) if count > 0 else 0.0
            avg_analysis[n_obj] = {
                "avg_obj_facts": avg_obj_facts,
                "avg_group_facts": avg_group_facts,
                "avg_groups": avg_groups,
                "std_obj_facts": std_obj_facts,
                "std_group_facts": std_group_facts
            }
        all_results[train_principle] = avg_analysis
        all_time_stats[train_principle] = time_dict

    # Merge all_results across principles
    merged = {}
    for principle_result in all_results.values():
        for n_obj, stats in principle_result.items():
            if n_obj not in merged:
                merged[n_obj] = {"obj_facts": [], "group_facts": [], "groups": [], "std_obj_facts": [], "std_group_facts": []}
            merged[n_obj]["obj_facts"].append(stats["avg_obj_facts"])
            merged[n_obj]["group_facts"].append(stats["avg_group_facts"])
            merged[n_obj]["groups"].append(stats["avg_groups"])
            merged[n_obj]["std_obj_facts"].append(stats["std_obj_facts"])
            merged[n_obj]["std_group_facts"].append(stats["std_group_facts"])

    merged_time = {}
    for principle_time in all_time_stats.values():
        for n_obj, tstats in principle_time.items():
            if n_obj not in merged_time:
                merged_time[n_obj] = {"obj_time": [], "group_time": []}
            merged_time[n_obj]["obj_time"].extend(tstats["obj_time"])
            merged_time[n_obj]["group_time"].extend(tstats["group_time"])

    merged_avg = {}
    for n_obj, stats in merged.items():
        count = len(stats["obj_facts"])
        merged_avg[n_obj] = {
            "avg_obj_facts": sum(stats["obj_facts"]) / count if count > 0 else 0.0,
            "avg_group_facts": sum(stats["group_facts"]) / count if count > 0 else 0.0,
            "avg_groups": sum(stats["groups"]) / count if count > 0 else 0.0,
            "std_obj_facts": np.std(stats["obj_facts"]) if count > 0 else 0.0,
            "std_group_facts": np.std(stats["group_facts"]) if count > 0 else 0.0
        }

    merged_time_avg = {}
    for n_obj, tstats in merged_time.items():
        count_obj = len(tstats["obj_time"])
        count_group = len(tstats["group_time"])
        merged_time_avg[n_obj] = {
            "avg_obj_time": np.mean(tstats["obj_time"]) if count_obj > 0 else 0.0,
            "avg_group_time": np.mean(tstats["group_time"]) if count_group > 0 else 0.0,
            "std_obj_time": np.std(tstats["obj_time"]) if count_obj > 0 else 0.0,
            "std_group_time": np.std(tstats["group_time"]) if count_group > 0 else 0.0,
            "n_obj_time": count_obj,
            "n_group_time": count_group
        }

    # Draw and save the combined line chart
    chart_path = config.output / "fact_number_line_chart_all_principles.png"
    exp_utils.draw_fact_number_line_chart(merged_avg, chart_path)
    time_chart_path = config.output / "time_cost_line_chart_all_principles.png"
    exp_utils.draw_time_cost_line_chart(merged_time_avg, time_chart_path)

    # Save to JSON
    saved_json_file = config.output / f"efficiency_analysis.json"
    with open(saved_json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n=== Efficiency Analysis Summary ===")
    for principle, analysis in all_results.items():
        print(f"{principle}: {analysis}")


if __name__ == "__main__":
    main_eff_analysis()
