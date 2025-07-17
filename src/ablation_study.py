# Created by MacBook Pro at 12.06.25
# ablation_main.py
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
    "hard_ogc": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": True},
    "hard_obj": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False, "use_calibrator": False},
    "hard_og": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": False},
    "hard_obj_calib": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False, "use_calibrator": True},
}


def run_ablation(train_data, val_data, test_data, obj_model, group_model, train_principle, args, mode_name,
                 ablation_flags):
    task_name = train_data["task"]
    train_val_data = {
        "task": task_name,
        "positive": train_data["positive"] + val_data["positive"],
        "negative": train_data["negative"] + val_data["negative"]
    }
    hyp_params = {"prox": 0.9, "sim": 0.5, "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [1] * len(train_val_data["positive"]) + [0] * len(train_val_data["negative"])
    # train rule + calibrator
    obj_times = torch.zeros(len(train_img_labels))
    group_times = torch.zeros(len(train_img_labels))

    hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, group_model, hyp_params, train_principle, args.device, ablation_flags, obj_times)
    base_rules = training.train_rules(hard, soft, obj_list, group_list, group_nums, train_img_labels, hyp_params, ablation_flags, obj_times)
    final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)

    calibrator = training.train_calibrator(final_rules, obj_list, group_list, hard, soft, train_img_labels, hyp_params, ablation_flags, args.device)
    eval_metrics = evaluation.eval_rules(test_data, obj_model, group_model, final_rules, hyp_params, train_principle, args.device, calibrator)

    return eval_metrics


def main_ablation():
    args = args_utils.get_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    train_principle = args.principle
    principle_path = getattr(config, f"grb_{train_principle}")
    combined_loader = dataset.load_combined_dataset(principle_path)
    obj_model = eval_patch_classifier.load_model(args.device)
    group_model = scorer_config.load_scorer_model(train_principle, args.device)

    wandb.init(project=f"grb_ablation_{train_principle}",
               config=args.__dict__, name=args.exp_name)

    results_summary = defaultdict(lambda: defaultdict(list))
    error_summary = defaultdict(lambda: defaultdict(list))  # mode -> error_type -> list of counts
    topk_summary = defaultdict(lambda: defaultdict(list))  # mode -> topk_metric -> list
    per_task_results = defaultdict(list)  # mode -> list of dicts
    analysis_summary = defaultdict(lambda: defaultdict(list))

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

            error_stats = test_metrics.get("error_stats", None)
            if error_stats:
                for err_type, count in error_stats.items():
                    error_summary[mode_name][err_type].append(count)

            # Top-k clause analysis
            for k in ["topk_clause_recall", "topk_clause_precision"]:
                if k in test_metrics:
                    topk_summary[mode_name][k].append(test_metrics[k])

            if "analysis" in test_metrics:
                for k, values in test_metrics["analysis"].items():
                    analysis_summary[mode_name][k].extend(values)

            log_dicts.update({f"{mode_name}_{k}": test_metrics.get(k, 0) for k in test_metrics})
            log_dicts.update({f"{mode_name}_acc_avg": torch.tensor(all_acc[mode_name]).mean(),
                              f"{mode_name}_auc_avg": torch.tensor(all_auc[mode_name]).mean(),
                              f"{mode_name}_f1_avg": torch.tensor(all_f1[mode_name]).mean()
                              })
            # Store per-task results
            per_task_results[mode_name].append({
                "task_idx": task_idx,
                "task_name": task_name,
                **{k: test_metrics.get(k, 0) for k in test_metrics}
            })
        wandb.log(log_dicts)

    # save and summarize
    final_summary = {
        mode: {f"avg_{k}": float(torch.tensor(v).mean())
               for k, v in metric_dict.items()}
        for mode, metric_dict in results_summary.items()
    }
    # Include top-k clause stats
    for mode, topk_metrics in topk_summary.items():
        for k, values in topk_metrics.items():
            final_summary[mode][f"avg_{k}"] = float(torch.tensor(values).mean())
    for mode, analysis_dict in analysis_summary.items():
        for k, values in analysis_dict.items():
            final_summary[mode][f"avg_{k}"] = float(torch.tensor(values).float().mean())

    # Save both per-task and average results
    output_json = {
        "per_task_results": per_task_results,
        "summary": final_summary
    }
    with open(config.output / f"ablation_summary_{args.principle}_{timestamp}.json", "w") as f:
        json.dump(output_json, f, indent=2)
    print("\n=== Final Summary ===")
    for mode, metrics in final_summary.items():
        print(f"{mode}: {metrics}")

    final_error_stats = {}
    for mode, err_dict in error_summary.items():
        total_errors = torch.tensor(err_dict.get("total_errors", [1.0])).float()  # avoid div by zero
        mode_stats = {
            err_type: float(torch.tensor(counts).sum() / total_errors.sum())
            for err_type, counts in err_dict.items()
            if err_type != "total_errors"
        }
        final_error_stats[mode] = mode_stats
    # Save or print
    with open(config.output / f"error_summary_{args.principle}_{timestamp}.json", "w") as f:
        json.dump(final_error_stats, f, indent=2)
    print("\n=== Error Summary ===")
    for mode, stats in final_error_stats.items():
        print(f"{mode}: {stats}")
    wandb.finish()


if __name__ == "__main__":
    main_ablation()
